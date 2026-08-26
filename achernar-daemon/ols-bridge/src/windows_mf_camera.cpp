// Native Windows UVC camera backend for Achernar.
//
// Media Foundation talks to the standard Windows USB Video Class driver.  In
// particular, this does not require (or permit us to suggest) replacing the
// SV105's normal driver with WinUSB/Zadig.  The class implements the same OLS
// CameraDriver/Camera contract as the libuvc backend used on other platforms,
// so the Rust and web API layers continue to expose this as driver_id "uvc".

#ifndef _WIN32
#error "windows_mf_camera.cpp must only be compiled for Windows"
#endif

#define NOMINMAX
#include <windows.h>
#include <dshow.h>
#include <mfapi.h>
#include <mferror.h>
#include <mfidl.h>
#include <mfreadwrite.h>
#include <wrl/client.h>

#include "camera.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace ols {
namespace windows_mf {

using Microsoft::WRL::ComPtr;

class MfError : public CamError {
 public:
  explicit MfError(const std::string &message) : CamError(message) {}
};

static std::string utf8(const wchar_t *value) {
  if (!value || !*value) return std::string();
  const int length = WideCharToMultiByte(CP_UTF8, 0, value, -1, nullptr, 0,
                                         nullptr, nullptr);
  if (length <= 1) return std::string();
  std::string result(static_cast<size_t>(length), '\0');
  WideCharToMultiByte(CP_UTF8, 0, value, -1, &result[0], length, nullptr,
                      nullptr);
  result.pop_back();  // discard the terminator written by WideCharToMultiByte
  return result;
}

static std::string hresult_message(HRESULT hr) {
  wchar_t *buffer = nullptr;
  const DWORD count = FormatMessageW(
      FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
          FORMAT_MESSAGE_IGNORE_INSERTS,
      nullptr, static_cast<DWORD>(hr), 0,
      reinterpret_cast<wchar_t *>(&buffer), 0, nullptr);

  std::ostringstream out;
  out << "0x" << std::hex << std::uppercase
      << static_cast<unsigned long>(hr);
  if (count && buffer) {
    std::string message = utf8(buffer);
    while (!message.empty() &&
           (message.back() == '\r' || message.back() == '\n' ||
            message.back() == ' ')) {
      message.pop_back();
    }
    if (!message.empty()) out << " (" << message << ")";
  }
  if (buffer) LocalFree(buffer);
  return out.str();
}

static void check_hr(HRESULT hr, const char *operation) {
  if (FAILED(hr)) {
    throw MfError(std::string(operation) + " failed: " +
                  hresult_message(hr));
  }
}

// Rust can invoke a camera method from any Tokio blocking worker.  Entering an
// MTA for every public boundary keeps COM valid even when subsequent calls land
// on a different worker.  S_FALSE still requires a matching CoUninitialize;
// RPC_E_CHANGED_MODE means the embedding thread already chose an apartment and
// is usable as-is.
class ComScope {
 public:
  ComScope() {
    const HRESULT hr = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    if (hr == S_OK || hr == S_FALSE) {
      uninitialize_ = true;
    } else if (hr != RPC_E_CHANGED_MODE) {
      check_hr(hr, "CoInitializeEx");
    }
  }

  ~ComScope() {
    if (uninitialize_) CoUninitialize();
  }

 private:
  bool uninitialize_ = false;
};

// MFStartup is process-wide. Keep it alive for the process lifetime rather
// than cycling it as discovery creates short-lived driver instances. The
// keeper thread also keeps one COM MTA registration alive. Rust's blocking
// workers enter and leave that same process MTA for individual calls; without
// a keeper, the last worker's CoUninitialize could tear the apartment down
// while camera/source-reader COM pointers are still retained by OLS.
class MfRuntime {
 public:
  static MfRuntime &instance() {
    static MfRuntime runtime;
    return runtime;
  }

  void ensure_started() const {
    if (FAILED(result_)) check_hr(result_, "MFStartup");
  }

 private:
  MfRuntime() {
    keeper_ = std::thread([this] {
      HRESULT result = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
      const bool com_initialized = result == S_OK || result == S_FALSE;
      if (com_initialized) result = MFStartup(MF_VERSION, MFSTARTUP_FULL);

      {
        std::lock_guard<std::mutex> guard(mutex_);
        result_ = result;
        ready_ = true;
      }
      condition_.notify_all();

      if (SUCCEEDED(result)) {
        std::unique_lock<std::mutex> guard(mutex_);
        condition_.wait(guard, [this] { return stopping_; });
        guard.unlock();
        MFShutdown();
      }
      if (com_initialized) CoUninitialize();
    });

    std::unique_lock<std::mutex> guard(mutex_);
    condition_.wait(guard, [this] { return ready_; });
  }

  ~MfRuntime() {
    {
      std::lock_guard<std::mutex> guard(mutex_);
      stopping_ = true;
    }
    condition_.notify_all();
    if (keeper_.joinable()) keeper_.join();
  }

  mutable std::mutex mutex_;
  std::condition_variable condition_;
  bool ready_ = false;
  bool stopping_ = false;
  HRESULT result_ = E_UNEXPECTED;
  std::thread keeper_;
};

static double unix_now() {
  using namespace std::chrono;
  return duration<double>(system_clock::now().time_since_epoch()).count();
}

struct StreamState {
  bool begin(frame_callback_type &callback, CamStreamFormat &format) {
    std::lock_guard<std::mutex> guard(mutex);
    if (!active || !frame_callback) return false;
    ++in_flight;
    callback = frame_callback;
    format = selected_format;
    return true;
  }

  // A ReadSample request is part of the stream lifecycle just as much as the
  // callback it eventually produces. Without this guard, stop could observe
  // zero callbacks, complete Flush, and then race with a callback thread that
  // had already decided to enqueue the next read. That late request would make
  // an immediate restart fail with MF_E_NOTACCEPTING.
  bool begin_request() {
    std::lock_guard<std::mutex> guard(mutex);
    if (!active) return false;
    ++in_flight;
    return true;
  }

  void end() {
    std::lock_guard<std::mutex> guard(mutex);
    if (in_flight > 0) --in_flight;
    if (in_flight == 0) idle.notify_all();
  }

  void activate(frame_callback_type callback, CamStreamFormat format) {
    std::lock_guard<std::mutex> guard(mutex);
    frame_callback = std::move(callback);
    selected_format = format;
    frame_counter.store(0);
    active = true;
  }

  void fail() {
    std::lock_guard<std::mutex> guard(mutex);
    active = false;
    frame_callback = nullptr;
  }

  void deactivate_and_wait() {
    std::unique_lock<std::mutex> guard(mutex);
    active = false;
    frame_callback = nullptr;
    idle.wait(guard, [this] { return in_flight == 0; });
  }

  int next_frame_counter() { return frame_counter.fetch_add(1); }

  mutable std::mutex mutex;
  std::condition_variable idle;
  bool active = false;
  size_t in_flight = 0;
  frame_callback_type frame_callback;
  CamStreamFormat selected_format{};
  std::atomic<int> frame_counter{0};
};

class SourceReaderCallback final : public IMFSourceReaderCallback {
 public:
  explicit SourceReaderCallback(std::shared_ptr<StreamState> state)
      : state_(std::move(state)) {}

  STDMETHODIMP QueryInterface(REFIID iid, void **object) override {
    if (!object) return E_POINTER;
    if (iid == __uuidof(IUnknown) ||
        iid == __uuidof(IMFSourceReaderCallback)) {
      *object = static_cast<IMFSourceReaderCallback *>(this);
      AddRef();
      return S_OK;
    }
    *object = nullptr;
    return E_NOINTERFACE;
  }

  STDMETHODIMP_(ULONG) AddRef() override { return ++references_; }

  STDMETHODIMP_(ULONG) Release() override {
    const ULONG references = --references_;
    if (!references) delete this;
    return references;
  }

  STDMETHODIMP OnEvent(DWORD, IMFMediaEvent *) override { return S_OK; }

  STDMETHODIMP OnFlush(DWORD) override {
    std::lock_guard<std::mutex> guard(flush_mutex_);
    ++completed_flushes_;
    flush_complete_.notify_all();
    return S_OK;
  }

  STDMETHODIMP OnReadSample(HRESULT status, DWORD, DWORD flags,
                            LONGLONG, IMFSample *sample) override {
    try {
      if (FAILED(status)) {
        emit_error(std::string("Media Foundation frame read failed: ") +
                   hresult_message(status));
        state_->fail();
        return S_OK;
      }
      if (flags & MF_SOURCE_READERF_ENDOFSTREAM) {
        emit_error("Media Foundation camera stream ended");
        state_->fail();
        return S_OK;
      }
      if (sample) emit_sample(sample);
      request_next_if_active();
    } catch (const std::exception &error) {
      emit_error(error.what());
      state_->fail();
    } catch (...) {
      emit_error("Unknown exception in Media Foundation frame callback");
      state_->fail();
    }
    return S_OK;
  }

  void set_reader(IMFSourceReader *reader) {
    std::lock_guard<std::mutex> guard(reader_mutex_);
    reader_ = reader;
  }

  HRESULT request_first() {
    if (!state_->begin_request()) return MF_E_SHUTDOWN;
    InvocationGuard request(state_);
    return request_next();
  }

  uint64_t flush_token() const {
    std::lock_guard<std::mutex> guard(flush_mutex_);
    return completed_flushes_;
  }

  bool wait_for_flush_after(uint64_t token,
                            std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> guard(flush_mutex_);
    return flush_complete_.wait_for(
        guard, timeout, [&] { return completed_flushes_ > token; });
  }

 private:
  ~SourceReaderCallback() = default;

  struct InvocationGuard {
    explicit InvocationGuard(std::shared_ptr<StreamState> value)
        : state(std::move(value)) {}
    ~InvocationGuard() { state->end(); }
    std::shared_ptr<StreamState> state;
  };

  void emit_sample(IMFSample *sample) {
    frame_callback_type callback;
    CamStreamFormat format;
    if (!state_->begin(callback, format)) return;
    InvocationGuard invocation(state_);

    ComPtr<IMFMediaBuffer> buffer;
    check_hr(sample->ConvertToContiguousBuffer(&buffer),
             "IMFSample::ConvertToContiguousBuffer");

    BYTE *bytes = nullptr;
    DWORD maximum_length = 0;
    DWORD current_length = 0;
    check_hr(buffer->Lock(&bytes, &maximum_length, &current_length),
             "IMFMediaBuffer::Lock");

    struct UnlockGuard {
      IMFMediaBuffer *buffer;
      ~UnlockGuard() { buffer->Unlock(); }
    } unlock{buffer.Get()};

    if (!bytes || current_length == 0) {
      throw MfError("Media Foundation delivered an empty camera frame");
    }

    CamFrame frame{};
    frame.format = format.format;
    frame.bayer = bayer_na;
    frame.frame_counter = state_->next_frame_counter();
    frame.unix_timestamp = unix_now();
    frame.width = format.width;
    frame.height = format.height;

    std::vector<unsigned char> mono;
    if (format.format == stream_mono8) {
      const size_t pixels = static_cast<size_t>(format.width) *
                            static_cast<size_t>(format.height);
      const size_t yuy2_bytes = pixels * 2;
      if (current_length < yuy2_bytes) {
        throw MfError("Partial YUY2 frame: received " +
                      std::to_string(current_length) + ", expected at least " +
                      std::to_string(yuy2_bytes));
      }
      mono.resize(pixels);
      for (size_t index = 0; index < pixels; ++index) {
        mono[index] = bytes[index * 2];
      }
      frame.data = mono.data();
      frame.data_size = mono.size();
    } else if (format.format == stream_yuv2) {
      const size_t expected = static_cast<size_t>(format.width) *
                              static_cast<size_t>(format.height) * 2;
      if (current_length < expected) {
        throw MfError("Partial YUY2 frame: received " +
                      std::to_string(current_length) + ", expected at least " +
                      std::to_string(expected));
      }
      frame.data = bytes;
      frame.data_size = expected;
    } else {
      frame.data = bytes;
      frame.data_size = current_length;
    }

    try {
      callback(frame);
    } catch (...) {
      // A consumer exception must not escape a COM callback or terminate the
      // Media Foundation work-queue thread.
    }
  }

  void emit_error(const std::string &message) {
    frame_callback_type callback;
    CamStreamFormat format;
    if (!state_->begin(callback, format)) return;
    InvocationGuard invocation(state_);
    CamFrame frame{};
    frame.format = stream_error;
    frame.frame_counter = state_->next_frame_counter();
    frame.unix_timestamp = unix_now();
    frame.width = format.width;
    frame.height = format.height;
    frame.data = message.c_str();
    frame.data_size = message.size();
    try {
      callback(frame);
    } catch (...) {
      // See emit_sample(): C++ exceptions may not cross the COM boundary.
    }
  }

  HRESULT request_next() {
    ComPtr<IMFSourceReader> reader;
    {
      std::lock_guard<std::mutex> guard(reader_mutex_);
      if (!reader_) return MF_E_SHUTDOWN;
      reader = reader_;
    }
    return reader->ReadSample(MF_SOURCE_READER_FIRST_VIDEO_STREAM, 0, nullptr,
                              nullptr, nullptr, nullptr);
  }

  void request_next_if_active() {
    if (!state_->begin_request()) return;
    InvocationGuard request(state_);
    const HRESULT hr = request_next();
    if (FAILED(hr)) {
      emit_error(std::string("Could not request the next camera frame: ") +
                 hresult_message(hr));
      state_->fail();
    }
  }

  std::atomic<ULONG> references_{1};
  std::shared_ptr<StreamState> state_;
  std::mutex reader_mutex_;
  IMFSourceReader *reader_ = nullptr;  // guarded, non-owning
  mutable std::mutex flush_mutex_;
  std::condition_variable flush_complete_;
  uint64_t completed_flushes_ = 0;
};

struct NativeFormat {
  CamStreamFormat exposed{};
  ComPtr<IMFMediaType> media_type;
};

struct ControlRange {
  long minimum = 0;
  long maximum = 0;
  long step = 1;
  long default_value = 0;
  long capabilities = 0;
};

class MediaFoundationCamera final : public Camera {
 public:
  MediaFoundationCamera(std::wstring symbolic_link, std::string name)
      : symbolic_link_(std::move(symbolic_link)), name_(std::move(name)),
        state_(std::make_shared<StreamState>()) {
    ComScope com;
    MfRuntime::instance().ensure_started();
    try {
      create_source();
      source_.As(&camera_control_);
      source_.As(&video_control_);
      callback_.Attach(new SourceReaderCallback(state_));

      ComPtr<IMFAttributes> attributes;
      check_hr(MFCreateAttributes(&attributes, 2), "MFCreateAttributes");
      check_hr(attributes->SetUnknown(MF_SOURCE_READER_ASYNC_CALLBACK,
                                      callback_.Get()),
               "Set Source Reader callback");
      check_hr(attributes->SetUINT32(MF_READWRITE_DISABLE_CONVERTERS, TRUE),
               "Disable Source Reader converters");
      check_hr(MFCreateSourceReaderFromMediaSource(
                   source_.Get(), attributes.Get(), &reader_),
               "MFCreateSourceReaderFromMediaSource");
      callback_->set_reader(reader_.Get());
      // Some UVC/WDM stacks expose legacy camera controls directly on the
      // media source; others only surface them as a Source Reader service.
      // Probe both. Failure is intentionally non-fatal: controls are optional
      // UVC capabilities, while live capture can still work perfectly.
      if (!camera_control_) {
        reader_->GetServiceForStream(
            MF_SOURCE_READER_MEDIASOURCE, GUID_NULL,
            __uuidof(IAMCameraControl),
            reinterpret_cast<void **>(camera_control_.ReleaseAndGetAddressOf()));
      }
      if (!video_control_) {
        reader_->GetServiceForStream(
            MF_SOURCE_READER_MEDIASOURCE, GUID_NULL,
            __uuidof(IAMVideoProcAmp),
            reinterpret_cast<void **>(video_control_.ReleaseAndGetAddressOf()));
      }
      read_formats();
    } catch (...) {
      callback_.Reset();
      reader_.Reset();
      if (source_) source_->Shutdown();
      source_.Reset();
      throw;
    }
  }

  ~MediaFoundationCamera() override {
    ComScope com;
    CamErrorCode ignored;
    stop_stream(ignored);
    if (callback_) callback_->set_reader(nullptr);
    reader_.Reset();
    callback_.Reset();
    camera_control_.Reset();
    video_control_.Reset();
    if (source_) source_->Shutdown();
    source_.Reset();
  }

  std::string name(CamErrorCode &) override { return name_; }

  std::vector<CamStreamFormat> formats(CamErrorCode &error) override {
    ComScope com;
    try {
      std::vector<CamStreamFormat> result;
      result.reserve(formats_.size());
      for (const NativeFormat &format : formats_) result.push_back(format.exposed);
      return result;
    } catch (const std::exception &exception) {
      error = CamErrorCode(exception);
      return {};
    }
  }

  void start_stream(CamStreamFormat requested, frame_callback_type callback,
                    CamErrorCode &error) override {
    ComScope com;
    try {
      CamErrorCode stop_error;
      stop_stream(stop_error);
      if (stop_error) throw MfError(stop_error.message());

      NativeFormat *selected = find_format(requested);
      if (!selected) {
        throw MfError("The requested Media Foundation camera format is not available");
      }

      check_hr(reader_->SetStreamSelection(MF_SOURCE_READER_ALL_STREAMS, FALSE),
               "Disable Media Foundation source streams");
      check_hr(reader_->SetStreamSelection(
                   MF_SOURCE_READER_FIRST_VIDEO_STREAM, TRUE),
               "Enable camera video stream");
      check_hr(reader_->SetCurrentMediaType(
                   MF_SOURCE_READER_FIRST_VIDEO_STREAM, nullptr,
                   selected->media_type.Get()),
               "Select camera media type");

      state_->activate(std::move(callback), selected->exposed);
      const HRESULT hr = callback_->request_first();
      if (FAILED(hr)) {
        state_->deactivate_and_wait();
        check_hr(hr, "Request first camera frame");
      }
      stream_active_ = true;
    } catch (const std::exception &exception) {
      state_->deactivate_and_wait();
      stream_active_ = false;
      error = CamErrorCode(exception);
    }
  }

  void stop_stream(CamErrorCode &error) override {
    ComScope com;
    try {
      state_->deactivate_and_wait();
      if (reader_ && stream_active_) {
        const uint64_t flush_token = callback_->flush_token();
        check_hr(reader_->Flush(MF_SOURCE_READER_FIRST_VIDEO_STREAM),
                 "Flush camera stream");
        // An asynchronous Source Reader reports completion through OnFlush.
        // Waiting here makes stop_stream's OLS guarantee real: once it
        // returns, no old sample request can bleed into a restarted stream or
        // prevent SetCurrentMediaType with MF_E_NOTACCEPTING.
        if (!callback_->wait_for_flush_after(flush_token,
                                             std::chrono::seconds(5))) {
          throw MfError("Timed out while stopping Media Foundation camera stream");
        }
      }
      stream_active_ = false;
    } catch (const std::exception &exception) {
      stream_active_ = false;
      error = CamErrorCode(exception);
    }
  }

  std::vector<CamOptionId> supported_options(CamErrorCode &) override {
    ComScope com;
    std::vector<CamOptionId> options;
    ControlRange range;

    if (camera_range(CameraControl_Exposure, range)) {
      if (range.capabilities & CameraControl_Flags_Manual)
        options.push_back(opt_exp);
      // Toggling auto off must be possible when the daemon prepares a manual
      // science exposure, so expose the switch only when both modes exist.
      if ((range.capabilities & CameraControl_Flags_Auto) &&
          (range.capabilities & CameraControl_Flags_Manual))
        options.push_back(opt_auto_exp);
    }
    if (video_range(VideoProcAmp_WhiteBalance, range)) {
      if (range.capabilities & VideoProcAmp_Flags_Manual)
        options.push_back(opt_wb);
      if ((range.capabilities & VideoProcAmp_Flags_Auto) &&
          (range.capabilities & VideoProcAmp_Flags_Manual))
        options.push_back(opt_auto_wb);
    }
    append_video_option(options, opt_gain, VideoProcAmp_Gain);
    append_video_option(options, opt_gamma, VideoProcAmp_Gamma);
    append_video_option(options, opt_brightness, VideoProcAmp_Brightness);
    append_video_option(options, opt_contrast, VideoProcAmp_Contrast);
    return options;
  }

  CamParam get_parameter(CamOptionId id, bool,
                         CamErrorCode &error) override {
    ComScope com;
    CamParam result;
    result.option = id;
    try {
      switch (id) {
        case opt_auto_exp:
          result = get_camera_auto_parameter(id, CameraControl_Exposure);
          break;
        case opt_exp:
          result = get_exposure_parameter();
          break;
        case opt_auto_wb:
          result = get_video_auto_parameter(id, VideoProcAmp_WhiteBalance);
          break;
        case opt_wb:
          result = get_video_parameter(id, type_kelvin,
                                       VideoProcAmp_WhiteBalance, 1.0);
          break;
        case opt_gain:
          result = get_video_parameter(id, type_number, VideoProcAmp_Gain, 1.0);
          break;
        case opt_gamma:
          result = get_video_parameter(id, type_number, VideoProcAmp_Gamma, 0.01);
          break;
        case opt_brightness:
          result = get_video_parameter(id, type_number,
                                       VideoProcAmp_Brightness, 1.0);
          break;
        case opt_contrast:
          result = get_video_parameter(id, type_number,
                                       VideoProcAmp_Contrast, 1.0);
          break;
        default:
          throw MfError("Unsupported Windows UVC camera control");
      }
    } catch (const std::exception &exception) {
      error = CamErrorCode(exception);
    }
    return result;
  }

  void set_parameter(CamOptionId id, double value,
                     CamErrorCode &error) override {
    ComScope com;
    try {
      switch (id) {
        case opt_auto_exp:
          set_camera_auto(CameraControl_Exposure, value != 0.0);
          break;
        case opt_exp:
          set_exposure_milliseconds(value);
          break;
        case opt_auto_wb:
          set_video_auto(VideoProcAmp_WhiteBalance, value != 0.0);
          break;
        case opt_wb:
          set_video_value(VideoProcAmp_WhiteBalance, value, 1.0);
          break;
        case opt_gain:
          set_video_value(VideoProcAmp_Gain, value, 1.0);
          break;
        case opt_gamma:
          set_video_value(VideoProcAmp_Gamma, value, 0.01);
          break;
        case opt_brightness:
          set_video_value(VideoProcAmp_Brightness, value, 1.0);
          break;
        case opt_contrast:
          set_video_value(VideoProcAmp_Contrast, value, 1.0);
          break;
        default:
          throw MfError("Unsupported Windows UVC camera control");
      }
    } catch (const std::exception &exception) {
      error = CamErrorCode(exception);
    }
  }

 private:
  void create_source() {
    ComPtr<IMFAttributes> attributes;
    check_hr(MFCreateAttributes(&attributes, 2), "MFCreateAttributes");
    check_hr(attributes->SetGUID(
                 MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE,
                 MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID),
             "Set video capture source type");
    check_hr(attributes->SetString(
                 MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_SYMBOLIC_LINK,
                 symbolic_link_.c_str()),
             "Set camera symbolic link");
    check_hr(MFCreateDeviceSource(attributes.Get(), &source_),
             "Open Windows camera");
  }

  void read_formats() {
    for (DWORD index = 0;; ++index) {
      ComPtr<IMFMediaType> media_type;
      const HRESULT hr = reader_->GetNativeMediaType(
          MF_SOURCE_READER_FIRST_VIDEO_STREAM, index, &media_type);
      if (hr == MF_E_NO_MORE_TYPES) break;
      check_hr(hr, "Read native camera media type");

      GUID major_type{};
      GUID subtype{};
      if (FAILED(media_type->GetGUID(MF_MT_MAJOR_TYPE, &major_type)) ||
          major_type != MFMediaType_Video ||
          FAILED(media_type->GetGUID(MF_MT_SUBTYPE, &subtype))) {
        continue;
      }

      CamStreamType stream_type;
      if (subtype == MFVideoFormat_MJPG) {
        stream_type = stream_mjpeg;
      } else if (subtype == MFVideoFormat_YUY2) {
        stream_type = stream_yuv2;
      } else {
        continue;
      }

      UINT32 width = 0;
      UINT32 height = 0;
      if (FAILED(MFGetAttributeSize(media_type.Get(), MF_MT_FRAME_SIZE,
                                    &width, &height)) ||
          width == 0 || height == 0) {
        continue;
      }
      UINT32 numerator = 0;
      UINT32 denominator = 0;
      float frame_rate = -1.0f;
      if (SUCCEEDED(MFGetAttributeRatio(media_type.Get(), MF_MT_FRAME_RATE,
                                        &numerator, &denominator)) &&
          denominator != 0) {
        frame_rate = static_cast<float>(numerator) /
                     static_cast<float>(denominator);
      }

      CamStreamFormat format{};
      format.format = stream_type;
      format.width = static_cast<int>(width);
      format.height = static_cast<int>(height);
      format.framerate = frame_rate;
      append_format(format, media_type);

      if (stream_type == stream_yuv2) {
        format.format = stream_mono8;
        append_format(format, media_type);
      }
    }

    if (formats_.empty()) {
      throw MfError("Camera exposes no native MJPEG or YUY2 video formats");
    }
  }

  void append_format(const CamStreamFormat &format,
                     const ComPtr<IMFMediaType> &media_type) {
    const auto duplicate = std::find_if(
        formats_.begin(), formats_.end(), [&](const NativeFormat &existing) {
          return existing.exposed.format == format.format &&
                 existing.exposed.width == format.width &&
                 existing.exposed.height == format.height &&
                 std::fabs(existing.exposed.framerate - format.framerate) <
                     0.001f;
        });
    if (duplicate == formats_.end()) {
      formats_.push_back(NativeFormat{format, media_type});
    }
  }

  NativeFormat *find_format(const CamStreamFormat &requested) {
    auto found = std::find_if(
        formats_.begin(), formats_.end(), [&](const NativeFormat &candidate) {
          const bool frame_rate_matches =
              requested.framerate < 0.0f ||
              std::fabs(candidate.exposed.framerate - requested.framerate) <
                  0.001f;
          return candidate.exposed.format == requested.format &&
                 candidate.exposed.width == requested.width &&
                 candidate.exposed.height == requested.height &&
                 frame_rate_matches;
        });
    return found == formats_.end() ? nullptr : &*found;
  }

  bool camera_range(long property, ControlRange &range) const {
    return camera_control_ &&
           SUCCEEDED(camera_control_->GetRange(
               property, &range.minimum, &range.maximum, &range.step,
               &range.default_value, &range.capabilities));
  }

  bool video_range(long property, ControlRange &range) const {
    return video_control_ &&
           SUCCEEDED(video_control_->GetRange(
               property, &range.minimum, &range.maximum, &range.step,
               &range.default_value, &range.capabilities));
  }

  void append_video_option(std::vector<CamOptionId> &options, CamOptionId id,
                           long property) const {
    ControlRange range;
    if (video_range(property, range) &&
        (range.capabilities & VideoProcAmp_Flags_Manual)) {
      options.push_back(id);
    }
  }

  static double exposure_ms(long log_seconds) {
    return std::pow(2.0, static_cast<double>(log_seconds)) * 1000.0;
  }

  CamParam get_exposure_parameter() const {
    ControlRange range;
    if (!camera_range(CameraControl_Exposure, range) ||
        !(range.capabilities & CameraControl_Flags_Manual)) {
      throw MfError("Manual exposure is not supported by this camera");
    }
    long current = 0;
    long flags = 0;
    check_hr(camera_control_->Get(CameraControl_Exposure, &current, &flags),
             "Read camera exposure");

    CamParam result;
    result.option = opt_exp;
    result.type = type_msec;
    result.min_val = exposure_ms(range.minimum);
    result.max_val = exposure_ms(range.maximum);
    // Windows' legacy UVC exposure ladder is logarithmic, so there is no
    // truthful constant step in milliseconds. Zero tells adaptive clients to
    // use the returned/read-back value rather than assuming a linear ladder.
    result.step_size = 0.0;
    result.def_val = exposure_ms(range.default_value);
    result.cur_val = exposure_ms(current);
    return result;
  }

  CamParam get_camera_auto_parameter(CamOptionId id, long property) const {
    ControlRange range;
    if (!camera_range(property, range) ||
        !(range.capabilities & CameraControl_Flags_Auto) ||
        !(range.capabilities & CameraControl_Flags_Manual)) {
      throw MfError("Automatic exposure is not supported by this camera");
    }
    long current = 0;
    long flags = 0;
    check_hr(camera_control_->Get(property, &current, &flags),
             "Read automatic camera control");
    CamParam result;
    result.option = id;
    result.type = type_bool;
    result.min_val = 0;
    result.max_val = 1;
    result.step_size = 1;
    result.cur_val = (flags & CameraControl_Flags_Auto) ? 1 : 0;
    result.def_val = result.cur_val;
    return result;
  }

  CamParam get_video_auto_parameter(CamOptionId id, long property) const {
    ControlRange range;
    if (!video_range(property, range) ||
        !(range.capabilities & VideoProcAmp_Flags_Auto) ||
        !(range.capabilities & VideoProcAmp_Flags_Manual)) {
      throw MfError("Automatic video control is not supported by this camera");
    }
    long current = 0;
    long flags = 0;
    check_hr(video_control_->Get(property, &current, &flags),
             "Read automatic video control");
    CamParam result;
    result.option = id;
    result.type = type_bool;
    result.min_val = 0;
    result.max_val = 1;
    result.step_size = 1;
    result.cur_val = (flags & VideoProcAmp_Flags_Auto) ? 1 : 0;
    result.def_val = result.cur_val;
    return result;
  }

  CamParam get_video_parameter(CamOptionId id, CamOptionType type,
                               long property, double scale) const {
    ControlRange range;
    if (!video_range(property, range) ||
        !(range.capabilities & VideoProcAmp_Flags_Manual)) {
      throw MfError("Requested video control is not supported by this camera");
    }
    long current = 0;
    long flags = 0;
    check_hr(video_control_->Get(property, &current, &flags),
             "Read video control");
    CamParam result;
    result.option = id;
    result.type = type;
    result.min_val = range.minimum * scale;
    result.max_val = range.maximum * scale;
    result.step_size = std::max(1L, range.step) * scale;
    result.def_val = range.default_value * scale;
    result.cur_val = current * scale;
    return result;
  }

  static long quantize(double requested, const ControlRange &range) {
    const long step = std::max(1L, range.step);
    const double steps =
        std::round((requested - range.minimum) / static_cast<double>(step));
    const long value = range.minimum + static_cast<long>(steps) * step;
    return std::max(range.minimum, std::min(range.maximum, value));
  }

  void set_exposure_milliseconds(double milliseconds) {
    if (!(milliseconds > 0.0) || !std::isfinite(milliseconds))
      throw MfError("Exposure must be a positive finite number of milliseconds");
    ControlRange range;
    if (!camera_range(CameraControl_Exposure, range) ||
        !(range.capabilities & CameraControl_Flags_Manual)) {
      throw MfError("Manual exposure is not supported by this camera");
    }
    const double log_seconds = std::log2(milliseconds / 1000.0);
    const long selected = quantize(log_seconds, range);
    check_hr(camera_control_->Set(CameraControl_Exposure, selected,
                                  CameraControl_Flags_Manual),
             "Set camera exposure");
  }

  void set_camera_auto(long property, bool automatic) {
    ControlRange range;
    if (!camera_range(property, range) ||
        !(range.capabilities & CameraControl_Flags_Auto) ||
        !(range.capabilities & CameraControl_Flags_Manual)) {
      throw MfError("Automatic exposure switching is not supported by this camera");
    }
    long current = 0;
    long flags = 0;
    check_hr(camera_control_->Get(property, &current, &flags),
             "Read camera control before changing automatic mode");
    check_hr(camera_control_->Set(
                 property, current,
                 automatic ? CameraControl_Flags_Auto
                           : CameraControl_Flags_Manual),
             "Set automatic camera control");
  }

  void set_video_auto(long property, bool automatic) {
    ControlRange range;
    if (!video_range(property, range) ||
        !(range.capabilities & VideoProcAmp_Flags_Auto) ||
        !(range.capabilities & VideoProcAmp_Flags_Manual)) {
      throw MfError("Automatic video switching is not supported by this camera");
    }
    long current = 0;
    long flags = 0;
    check_hr(video_control_->Get(property, &current, &flags),
             "Read video control before changing automatic mode");
    check_hr(video_control_->Set(
                 property, current,
                 automatic ? VideoProcAmp_Flags_Auto
                           : VideoProcAmp_Flags_Manual),
             "Set automatic video control");
  }

  void set_video_value(long property, double value, double scale) {
    ControlRange range;
    if (!video_range(property, range) ||
        !(range.capabilities & VideoProcAmp_Flags_Manual)) {
      throw MfError("Manual video control is not supported by this camera");
    }
    const long selected = quantize(value / scale, range);
    check_hr(video_control_->Set(property, selected,
                                 VideoProcAmp_Flags_Manual),
             "Set video control");
  }

  std::wstring symbolic_link_;
  std::string name_;
  ComPtr<IMFMediaSource> source_;
  ComPtr<IMFSourceReader> reader_;
  ComPtr<SourceReaderCallback> callback_;
  ComPtr<IAMCameraControl> camera_control_;
  ComPtr<IAMVideoProcAmp> video_control_;
  std::shared_ptr<StreamState> state_;
  std::vector<NativeFormat> formats_;
  bool stream_active_ = false;
};

struct DeviceInfo {
  std::wstring symbolic_link;
  std::string friendly_name;
};

class MediaFoundationCameraDriver final : public CameraDriver {
 public:
  MediaFoundationCameraDriver() {
    ComScope com;
    MfRuntime::instance().ensure_started();
    enumerate();
  }

  std::vector<std::string> list_cameras(CamErrorCode &error) override {
    ComScope com;
    try {
      std::vector<std::string> result;
      result.reserve(devices_.size());
      for (size_t index = 0; index < devices_.size(); ++index) {
        result.push_back(devices_[index].friendly_name + "/" +
                         std::to_string(index));
      }
      return result;
    } catch (const std::exception &exception) {
      error = CamErrorCode(exception);
      return {};
    }
  }

  std::unique_ptr<Camera> open_camera(int id,
                                      CamErrorCode &error) override {
    ComScope com;
    try {
      if (id < 0 || static_cast<size_t>(id) >= devices_.size()) {
        throw MfError("No Windows camera with index #" + std::to_string(id));
      }
      const DeviceInfo &device = devices_[static_cast<size_t>(id)];
      return std::unique_ptr<Camera>(new MediaFoundationCamera(
          device.symbolic_link,
          device.friendly_name + "/" + std::to_string(id)));
    } catch (const std::exception &exception) {
      error = CamErrorCode(exception);
      return {};
    }
  }

 private:
  void enumerate() {
    ComPtr<IMFAttributes> attributes;
    check_hr(MFCreateAttributes(&attributes, 1), "MFCreateAttributes");
    check_hr(attributes->SetGUID(
                 MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE,
                 MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID),
             "Set video capture enumeration type");

    IMFActivate **activations = nullptr;
    UINT32 count = 0;
    check_hr(MFEnumDeviceSources(attributes.Get(), &activations, &count),
             "Enumerate Windows cameras");

    struct ActivationCleanup {
      IMFActivate **items;
      UINT32 count;
      ~ActivationCleanup() {
        if (!items) return;
        for (UINT32 index = 0; index < count; ++index) {
          if (items[index]) items[index]->Release();
        }
        CoTaskMemFree(items);
      }
    } cleanup{activations, count};

    for (UINT32 index = 0; index < count; ++index) {
      wchar_t *friendly = nullptr;
      wchar_t *symbolic_link = nullptr;
      UINT32 friendly_length = 0;
      UINT32 symbolic_length = 0;
      const HRESULT friendly_hr = activations[index]->GetAllocatedString(
          MF_DEVSOURCE_ATTRIBUTE_FRIENDLY_NAME, &friendly, &friendly_length);
      const HRESULT symbolic_hr = activations[index]->GetAllocatedString(
          MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_SYMBOLIC_LINK,
          &symbolic_link, &symbolic_length);
      if (SUCCEEDED(friendly_hr) && SUCCEEDED(symbolic_hr) && symbolic_link) {
        devices_.push_back(DeviceInfo{
            symbolic_link,
            friendly && *friendly ? utf8(friendly) : "Windows camera"});
      }
      if (friendly) CoTaskMemFree(friendly);
      if (symbolic_link) CoTaskMemFree(symbolic_link);
    }

    // Device enumeration order is not a persistent identity.  Sorting on the
    // OS symbolic link gives a deterministic index for an unchanged rig, while
    // the selected entry retains that exact link for open_camera().
    std::sort(devices_.begin(), devices_.end(),
              [](const DeviceInfo &left, const DeviceInfo &right) {
                return left.symbolic_link < right.symbolic_link;
              });
  }

  std::vector<DeviceInfo> devices_;
};

}  // namespace windows_mf
}  // namespace ols

extern "C" ols::CameraDriver *ols_get_uvc_driver(int fd,
                                                  ols::CamErrorCode *error) {
  try {
    if (fd >= 0) {
      throw ols::windows_mf::MfError(
          "Opening a Windows camera by Android file descriptor is unsupported");
    }
    return new ols::windows_mf::MediaFoundationCameraDriver();
  } catch (const std::exception &exception) {
    if (error) *error = ols::CamErrorCode(exception);
    return nullptr;
  }
}
