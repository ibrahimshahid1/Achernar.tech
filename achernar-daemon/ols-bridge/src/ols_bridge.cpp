/**
 * ols_bridge.cpp - Implementation of the C API bridge to OLS C++ drivers
 *
 * Strategy: We directly compile OLS driver sources into this bridge,
 * creating factory functions that instantiate OLS C++ objects behind
 * opaque C handles. cppcms/booster dependencies are replaced with
 * header-only shims (nlohmann/json, minimal socket wrappers).
 *
 * Supports: Sim camera, Alpaca camera, Alpaca mount, Alpaca focuser.
 * Vendor SDK cameras (ASI, SVB, Toup) are conditionally compiled.
 */

#include "ols_bridge.h"
#include "camera.h"
#include "mount.h"
#include "focuser.h"
#include "alpaca_mount.h"
#include "alpaca_focuser.h"
#include "os_util.h"
#include "tiffmat.h"
#include "util.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/imgcodecs.hpp>

// ─── Thread-local error message ───
static thread_local std::string g_last_error;

static OlsError set_error(OlsError code, const std::string &msg) {
  g_last_error = msg;
  return code;
}

static OlsError set_error(OlsError code, const std::exception &ex) {
  g_last_error = ex.what();
  return code;
}

// ═══════════════════════════════════════════════════════════════════
//  Minimal Sim Camera (inlined, no cppcms dependency)
// ═══════════════════════════════════════════════════════════════════

namespace ols_bridge {
using namespace ols;

static CamBayerType parse_bayer_from_info_json(const std::string &dir) {
  std::ifstream data(dir + "/info.json");
  if (!data)
    return bayer_na;

  std::string content((std::istreambuf_iterator<char>(data)),
                      std::istreambuf_iterator<char>());

  auto pos = content.find("\"bayer\"");
  if (pos == std::string::npos)
    return bayer_na;

  pos = content.find(':', pos);
  if (pos == std::string::npos)
    return bayer_na;

  auto q1 = content.find('"', pos + 1);
  if (q1 == std::string::npos)
    return bayer_na;

  auto q2 = content.find('"', q1 + 1);
  if (q2 == std::string::npos)
    return bayer_na;

  std::string bayer_str = content.substr(q1 + 1, q2 - q1 - 1);
  try {
    return bayer_type_from_str(bayer_str);
  } catch (...) {
    return bayer_na;
  }
}

class BridgeSIMCamera : public Camera {
public:
  BridgeSIMCamera(const std::string &dir) : dir_(dir) {
    stream_active_ = 0;
    load_frames();
  }

  std::string file_name(int frame_id, const std::string &ext) {
    char name[256];
    snprintf(name, sizeof(name), "/frame_%08d.%s", frame_id, ext.c_str());
    return dir_ + name;
  }

  void load_frames() {
    std::ifstream data(dir_ + "/log.txt");
    if (!data)
      throw CamError("Failed to read simulation log file.\n"
                     "Check that simulation data has been downloaded and "
                     "that sim.path points to it");

    bayer_ = bayer_na;
    std::string str;
    while (std::getline(data, str)) {
      size_t pos = str.find(',');
      std::string op = str.substr(0, pos);
      if (op == "PAUSE")
        continue;
      int frame_id = atoi(op.c_str());
      double timestamp = atof(str.substr(pos + 1).c_str());
      (void)timestamp;
      if (frames_.empty()) {
        is_jpeg_ = ols::exists(file_name(frame_id, "jpeg"));
      }
      std::string path = file_name(frame_id, is_jpeg_ ? "jpeg" : "tiff");
      if (frames_.empty()) {
        cv::Mat img;
        if (is_jpeg_)
          img = cv::imread(path);
        else
          img = load_tiff(path);
        width_ = img.cols;
        height_ = img.rows;
        if (is_jpeg_)
          stream_ = stream_mjpeg;
        else if (img.elemSize1() == 1 && img.channels() == 2)
          stream_ = stream_yuv2;
        else if (img.elemSize1() == 1 && img.channels() == 3)
          stream_ = stream_rgb24;
        else if (img.elemSize1() == 2 && img.channels() == 3)
          stream_ = stream_rgb48;
        else if (img.elemSize1() == 1 && img.channels() == 1) {
          bayer_ = parse_bayer_from_info_json(dir_);
          stream_ = bayer_ == bayer_na ? stream_mono8 : stream_raw8;
        } else if (img.elemSize1() == 2 && img.channels() == 1) {
          bayer_ = parse_bayer_from_info_json(dir_);
          stream_ = bayer_ == bayer_na ? stream_mono16 : stream_raw16;
        } else
          throw CamError("Unsupported image format for " + path);
      }
      frames_.push_back(std::make_pair(timestamp, path));
    }
    if (frames_.empty())
      throw CamError("No frames found in simulation directory " + dir_);
  }

  virtual ~BridgeSIMCamera() {
    try {
      CamErrorCode e;
      stop_stream(e);
    } catch (...) {
    }
  }

  virtual std::string name(CamErrorCode &) override { return "Simulation"; }

  virtual std::vector<CamStreamFormat> formats(CamErrorCode &) override {
    std::vector<CamStreamFormat> res;
    CamStreamFormat fmt;
    fmt.format = stream_;
    fmt.width = width_;
    fmt.height = height_;
    fmt.framerate = -1;
    res.push_back(fmt);
    return res;
  }

  void handle_frame(int index) {
    CamFrame frm;
    frm.unix_timestamp = frames_.at(index).first;
    frm.width = width_;
    frm.height = height_;
    std::vector<char> buf;
    cv::Mat img;
    if (is_jpeg_) {
      std::ifstream f(frames_.at(index).second, std::ifstream::binary);
      f.seekg(0, std::ifstream::end);
      size_t size = f.tellg();
      buf.resize(size);
      f.seekg(0);
      f.read(buf.data(), size);
      frm.data = buf.data();
      frm.data_size = size;
    } else {
      img = load_tiff(frames_.at(index).second);
      frm.data = img.data;
      frm.bayer = bayer_;
      frm.data_size = img.rows * img.cols * img.elemSize();
    }
    frm.frame_counter = frame_counter_++;
    frm.format = stream_;
    {
      std::unique_lock<std::mutex> guard(lock_);
      if (callback_) {
        try {
          callback_(frm);
        } catch (...) {
        }
      }
    }
  }

  double timestamp() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec + tv.tv_usec * 1e-6;
  }

  virtual void start_stream(CamStreamFormat format,
                            frame_callback_type callback,
                            CamErrorCode &e) override {
    try {
      if (stream_active_ != 0) {
        stop_stream(e);
        if (e) return;
      }
      if (format.format != stream_ || format.width != width_ ||
          format.height != height_)
        throw CamError("Invalid format");
      {
        std::unique_lock<std::mutex> guard(lock_);
        callback_ = callback;
      }
      stream_active_ = 1;
      thread_ = std::thread([=]() {
        int size = frames_.size();
        double next_time = timestamp() + exposure_ * 1e-3;
        while (stream_active_) {
          double remains;
          while ((remains = (next_time - timestamp())) >= 1e-3) {
            double ms = std::min(remains * 1000, 500.0);
            std::this_thread::sleep_for(std::chrono::milliseconds(int(ms)));
            if (stream_active_ != 1) return;
          }
          next_time += exposure_ * 1e-3;
          handle_frame(current_index_);
          if (current_dir_ == 1) {
            if (current_index_ >= size - 1) {
              current_index_ = std::max(0, current_index_ - 1);
              current_dir_ = -1;
            } else {
              current_index_++;
            }
          } else {
            if (current_index_ <= 0) {
              current_index_ = std::min(size - 1, current_index_ + 1);
              current_dir_ = +1;
            } else {
              current_index_--;
            }
          }
        }
      });
    } catch (std::exception const &err) {
      e = CamErrorCode(err);
    }
  }

  virtual void stop_stream(CamErrorCode &e) override {
    try {
      if (stream_active_ == 0) return;
      {
        std::unique_lock<std::mutex> guard(lock_);
        callback_ = nullptr;
      }
      stream_active_ = 0;
      thread_.join();
    } catch (std::exception const &err) {
      e = CamErrorCode(err);
    }
  }

  virtual std::vector<CamOptionId> supported_options(CamErrorCode &) override {
    return {opt_exp, opt_gamma};
  }

  virtual CamParam get_parameter(CamOptionId id, bool,
                                 CamErrorCode &e) override {
    CamParam r;
    r.option = id;
    switch (id) {
    case opt_exp:
      r.type = type_msec; r.min_val = 10; r.max_val = 10000;
      r.def_val = 1000; r.cur_val = exposure_; r.step_size = 1;
      break;
    case opt_gamma:
      r.type = type_number; r.min_val = 0.75; r.max_val = 3.0;
      r.def_val = 1.0; r.cur_val = gamma_; r.step_size = 0.01;
      break;
    default:
      e = CamErrorCode("Unimplemented: " + cam_option_id_to_name(id));
    }
    return r;
  }

  virtual void set_parameter(CamOptionId id, double value,
                             CamErrorCode &e) override {
    try {
      switch (id) {
      case opt_exp:
        if (value < 10 || value > 10000) throw CamError("Invalid range");
        exposure_ = value;
        break;
      case opt_gamma:
        if (value < 0.75 || value > 3.0) throw CamError("Invalid range");
        gamma_ = value;
        break;
      default:
        throw CamError("Unimplemented: " + cam_option_id_to_name(id));
      }
    } catch (std::exception const &err) {
      e = CamErrorCode(err);
    }
  }

private:
  std::string dir_;
  int exposure_ = 1000;
  double gamma_ = 1.0;
  CamStreamType stream_ = stream_mjpeg;
  CamBayerType bayer_ = bayer_na;
  bool is_jpeg_ = true;
  int width_ = 0, height_ = 0;
  std::atomic<int> stream_active_;
  int frame_counter_ = 0;
  int current_index_ = 0;
  int current_dir_ = +1;
  std::mutex lock_;
  frame_callback_type callback_;
  std::thread thread_;
  std::vector<std::pair<double, std::string>> frames_;
};

class BridgeSIMCameraDriver : public CameraDriver {
public:
  BridgeSIMCameraDriver(const std::string &data_dir) : data_dir_(data_dir) {}

  virtual std::vector<std::string> list_cameras(CamErrorCode &) override {
    return {"Simulation"};
  }

  virtual std::unique_ptr<Camera> open_camera(int id,
                                              CamErrorCode &e) override {
    try {
      if (id != 0) throw CamError("No such camera " + std::to_string(id));
      return std::unique_ptr<Camera>(new BridgeSIMCamera(data_dir_));
    } catch (std::exception const &err) {
      e = CamErrorCode(err);
      return nullptr;
    }
  }

private:
  std::string data_dir_;
};

// ─── Callback trampoline ───
struct CallbackTrampoline {
  OlsFrameCallback callback;
  void *user_data;
};

} // namespace ols_bridge

// ═══════════════════════════════════════════════════════════════════
//  C API Implementation
// ═══════════════════════════════════════════════════════════════════

extern "C" {

const char *ols_last_error(void) { return g_last_error.c_str(); }

const char *ols_bridge_version(void) { return "0.2.0"; }

// ═══════════════════════════════════════════════════════════════════
//  Camera Driver API
// ═══════════════════════════════════════════════════════════════════

struct OlsCameraDriver_t {
  std::unique_ptr<ols::CameraDriver> driver;
  std::vector<std::string> camera_names;
};

OlsError ols_sim_driver_create(const char *sim_data_dir,
                               OlsCameraDriver **out_driver) {
  try {
    auto *d = new OlsCameraDriver_t();
    d->driver.reset(new ols_bridge::BridgeSIMCameraDriver(sim_data_dir));
    *out_driver = d;
    return OLS_OK;
  } catch (std::exception const &ex) {
    return set_error(OLS_ERR_GENERIC, ex);
  }
}

OlsError ols_alpaca_camera_driver_create(const char *url,
                                          OlsCameraDriver **out_driver) {
  try {
    // TODO: Direct Alpaca camera instantiation
    // The OLS alpaca_camera.cpp uses AlpacaClient internally,
    // which requires httplib and our nlohmann/json shim.
    // For now, this returns an error. Enable by compiling
    // alpaca_camera.cpp into the bridge and implementing direct
    // factory calls here.
    (void)url;
    (void)out_driver;
    return set_error(OLS_ERR_GENERIC,
                     "Alpaca camera driver not yet compiled into bridge. "
                     "Use ASCOM Alpaca with an installed SDK camera driver instead.");
  } catch (std::exception const &ex) {
    return set_error(OLS_ERR_GENERIC, ex);
  }
}

OlsError ols_driver_list_cameras(OlsCameraDriver *driver, int32_t *out_count) {
  try {
    ols::CamErrorCode e;
    driver->camera_names = driver->driver->list_cameras(e);
    if (e)
      return set_error(OLS_ERR_IO, e.message());
    *out_count = (int32_t)driver->camera_names.size();
    return OLS_OK;
  } catch (std::exception const &ex) {
    return set_error(OLS_ERR_GENERIC, ex);
  }
}

// ─── Vendor SDK Camera Driver Factories ───

// Forward declarations for the externally defined drivers from the OLS codebase
extern "C" {
  ols::CameraDriver *ols_get_asi_driver(int cam_id, ols::CamErrorCode *e);
  ols::CameraDriver *ols_get_svb_driver(int cam_id, ols::CamErrorCode *e);
  ols::CameraDriver *ols_get_toup_driver(int cam_id, ols::CamErrorCode *e);
}

OlsError ols_asi_camera_driver_create(OlsCameraDriver **out_driver) {
#ifdef WITH_ASI
  try {
    ols::CamErrorCode e;
    ols::CameraDriver *d = ols_get_asi_driver(0, &e);
    if (e) return set_error(OLS_ERR_IO, e.message());
    if (!d) return set_error(OLS_ERR_IO, "Failed to instantiate ASI driver");
    
    auto *wrapper = new OlsCameraDriver_t();
    wrapper->driver.reset(d);
    *out_driver = wrapper;
    return OLS_OK;
  } catch (std::exception const &ex) {
    return set_error(OLS_ERR_GENERIC, ex);
  }
#else
  return set_error(OLS_ERR_NOT_FOUND, "ASI SDK support not compiled into ols_bridge");
#endif
}

OlsError ols_svb_camera_driver_create(OlsCameraDriver **out_driver) {
#ifdef WITH_SVB
  try {
    ols::CamErrorCode e;
    ols::CameraDriver *d = ols_get_svb_driver(0, &e);
    if (e) return set_error(OLS_ERR_IO, e.message());
    if (!d) return set_error(OLS_ERR_IO, "Failed to instantiate SVBony driver");
    
    auto *wrapper = new OlsCameraDriver_t();
    wrapper->driver.reset(d);
    *out_driver = wrapper;
    return OLS_OK;
  } catch (std::exception const &ex) {
    return set_error(OLS_ERR_GENERIC, ex);
  }
#else
  return set_error(OLS_ERR_NOT_FOUND, "SVBony SDK support not compiled into ols_bridge");
#endif
}

OlsError ols_toup_camera_driver_create(OlsCameraDriver **out_driver) {
#ifdef WITH_TOUP
  try {
    ols::CamErrorCode e;
    ols::CameraDriver *d = ols_get_toup_driver(0, &e);
    if (e) return set_error(OLS_ERR_IO, e.message());
    if (!d) return set_error(OLS_ERR_IO, "Failed to instantiate ToupTek driver");
    
    auto *wrapper = new OlsCameraDriver_t();
    wrapper->driver.reset(d);
    *out_driver = wrapper;
    return OLS_OK;
  } catch (std::exception const &ex) {
    return set_error(OLS_ERR_GENERIC, ex);
  }
#else
  return set_error(OLS_ERR_NOT_FOUND, "ToupTek SDK support not compiled into ols_bridge");
#endif
}

OlsError ols_driver_camera_name(OlsCameraDriver *driver, int32_t index,
                                char *name_buf, int32_t name_buf_len) {
  if (index < 0 || (size_t)index >= driver->camera_names.size())
    return set_error(OLS_ERR_NOT_FOUND, "Camera index out of range");
  const auto &name = driver->camera_names[index];
  strncpy(name_buf, name.c_str(), name_buf_len - 1);
  name_buf[name_buf_len - 1] = '\0';
  return OLS_OK;
}

struct OlsCamera_t {
  std::unique_ptr<ols::Camera> camera;
  std::vector<ols::CamStreamFormat> formats;
  std::vector<ols::CamOptionId> options;
  std::unique_ptr<ols_bridge::CallbackTrampoline> trampoline;
};

OlsError ols_driver_open_camera(OlsCameraDriver *driver, int32_t index,
                                OlsCamera **out_camera) {
  try {
    ols::CamErrorCode e;
    auto cam_ptr = driver->driver->open_camera(index, e);
    if (e) return set_error(OLS_ERR_IO, e.message());
    if (!cam_ptr) return set_error(OLS_ERR_IO, "Failed to open camera");

    auto *c = new OlsCamera_t();
    c->camera = std::move(cam_ptr);
    c->formats = c->camera->formats(e);
    if (e) { delete c; return set_error(OLS_ERR_IO, e.message()); }
    c->options = c->camera->supported_options(e);
    if (e) { delete c; return set_error(OLS_ERR_IO, e.message()); }

    *out_camera = c;
    return OLS_OK;
  } catch (std::exception const &ex) {
    return set_error(OLS_ERR_GENERIC, ex);
  }
}

void ols_driver_destroy(OlsCameraDriver *driver) { delete driver; }

OlsError ols_camera_name(OlsCamera *cam, char *name_buf, int32_t name_buf_len) {
  try {
    ols::CamErrorCode e;
    auto name = cam->camera->name(e);
    if (e) return set_error(OLS_ERR_IO, e.message());
    strncpy(name_buf, name.c_str(), name_buf_len - 1);
    name_buf[name_buf_len - 1] = '\0';
    return OLS_OK;
  } catch (std::exception const &ex) {
    return set_error(OLS_ERR_GENERIC, ex);
  }
}

OlsError ols_camera_format_count(OlsCamera *cam, int32_t *out_count) {
  *out_count = (int32_t)cam->formats.size();
  return OLS_OK;
}

OlsError ols_camera_format_get(OlsCamera *cam, int32_t index,
                               OlsStreamFormat *out_fmt) {
  if (index < 0 || (size_t)index >= cam->formats.size())
    return set_error(OLS_ERR_NOT_FOUND, "Format index out of range");
  auto &f = cam->formats[index];
  out_fmt->format = (OlsStreamType)f.format;
  out_fmt->width = f.width;
  out_fmt->height = f.height;
  out_fmt->bin = f.bin;
  out_fmt->framerate = f.framerate;
  return OLS_OK;
}

OlsError ols_camera_start_stream(OlsCamera *cam, const OlsStreamFormat *fmt,
                                 OlsFrameCallback callback, void *user_data) {
  try {
    ols::CamStreamFormat ols_fmt;
    ols_fmt.format = (ols::CamStreamType)fmt->format;
    ols_fmt.width = fmt->width;
    ols_fmt.height = fmt->height;
    ols_fmt.bin = fmt->bin;
    ols_fmt.framerate = fmt->framerate;

    cam->trampoline.reset(
        new ols_bridge::CallbackTrampoline{callback, user_data});
    auto *tramp = cam->trampoline.get();

    ols::CamErrorCode e;
    cam->camera->start_stream(
        ols_fmt,
        [tramp](const ols::CamFrame &frame) {
          OlsFrame f;
          f.format = (OlsStreamType)frame.format;
          f.bayer = (OlsBayerType)frame.bayer;
          f.frame_counter = frame.frame_counter;
          f.unix_timestamp = frame.unix_timestamp;
          f.width = frame.width;
          f.height = frame.height;
          f.data = (const uint8_t *)frame.data;
          f.data_size = frame.data_size;
          tramp->callback(&f, tramp->user_data);
        },
        e);
    if (e) return set_error(OLS_ERR_IO, e.message());
    return OLS_OK;
  } catch (std::exception const &ex) {
    return set_error(OLS_ERR_GENERIC, ex);
  }
}

OlsError ols_camera_stop_stream(OlsCamera *cam) {
  try {
    ols::CamErrorCode e;
    cam->camera->stop_stream(e);
    cam->trampoline.reset();
    if (e) return set_error(OLS_ERR_IO, e.message());
    return OLS_OK;
  } catch (std::exception const &ex) {
    return set_error(OLS_ERR_GENERIC, ex);
  }
}

OlsError ols_camera_option_count(OlsCamera *cam, int32_t *out_count) {
  *out_count = (int32_t)cam->options.size();
  return OLS_OK;
}

OlsError ols_camera_option_id(OlsCamera *cam, int32_t index,
                              OlsCamOptionId *out_id) {
  if (index < 0 || (size_t)index >= cam->options.size())
    return set_error(OLS_ERR_NOT_FOUND, "Option index out of range");
  *out_id = (OlsCamOptionId)cam->options[index];
  return OLS_OK;
}

OlsError ols_camera_get_param(OlsCamera *cam, OlsCamOptionId id,
                              OlsCamParam *out_param) {
  try {
    ols::CamErrorCode e;
    auto p = cam->camera->get_parameter((ols::CamOptionId)id, false, e);
    if (e) return set_error(OLS_ERR_IO, e.message());
    out_param->option = (OlsCamOptionId)p.option;
    out_param->type = (OlsCamOptionType)p.type;
    out_param->read_only = p.read_only ? 1 : 0;
    out_param->step_size = p.step_size;
    out_param->min_val = p.min_val;
    out_param->max_val = p.max_val;
    out_param->def_val = p.def_val;
    out_param->cur_val = p.cur_val;
    return OLS_OK;
  } catch (std::exception const &ex) {
    return set_error(OLS_ERR_GENERIC, ex);
  }
}

OlsError ols_camera_set_param(OlsCamera *cam, OlsCamOptionId id, double value) {
  try {
    ols::CamErrorCode e;
    cam->camera->set_parameter((ols::CamOptionId)id, value, e);
    if (e) return set_error(OLS_ERR_IO, e.message());
    return OLS_OK;
  } catch (std::exception const &ex) {
    return set_error(OLS_ERR_GENERIC, ex);
  }
}

void ols_camera_destroy(OlsCamera *cam) { delete cam; }

// ═══════════════════════════════════════════════════════════════════
//  Mount API Implementation
// ═══════════════════════════════════════════════════════════════════

struct OlsMountInterface_t {
  std::unique_ptr<ols::MountInterface> iface;
};

OlsError ols_mount_interface_create(OlsMountInterface **out_mi) {
  try {
    auto *mi = new OlsMountInterface_t();
    mi->iface.reset(new ols::MountInterface());
    *out_mi = mi;
    return OLS_OK;
  } catch (std::exception const &ex) {
    return set_error(OLS_ERR_GENERIC, ex);
  }
}

OlsError ols_mount_driver_list(OlsMountInterface *mi,
                                OlsMountDriverInfo *out_drivers, int32_t max_drivers,
                                int32_t *out_count) {
  try {
    auto g = mi->iface->guard();
    auto drivers = mi->iface->drivers_list();
    *out_count = std::min((int32_t)drivers.size(), max_drivers);
    for (int32_t i = 0; i < *out_count; i++) {
      strncpy(out_drivers[i].name, drivers[i].name.c_str(), 127);
      out_drivers[i].name[127] = '\0';
      strncpy(out_drivers[i].option, drivers[i].option.c_str(), 127);
      out_drivers[i].option[127] = '\0';
      out_drivers[i].has_option = drivers[i].has_option ? 1 : 0;
    }
    return OLS_OK;
  } catch (std::exception const &ex) {
    return set_error(OLS_ERR_GENERIC, ex);
  }
}

OlsError ols_mount_load_driver(OlsMountInterface *mi,
                                const char *driver_name, const char *option) {
  try {
    mi->iface->load_driver(driver_name, option ? option : "auto");
    return OLS_OK;
  } catch (std::exception const &ex) {
    return set_error(OLS_ERR_GENERIC, ex);
  }
}

OlsError ols_mount_load_mount(OlsMountInterface *mi) {
  try {
    mi->iface->load_mount();
    return OLS_OK;
  } catch (std::exception const &ex) {
    return set_error(OLS_ERR_GENERIC, ex);
  }
}

int32_t ols_mount_driver_loaded(OlsMountInterface *mi) {
  return mi->iface->driver_is_loaded() ? 1 : 0;
}

int32_t ols_mount_is_loaded(OlsMountInterface *mi) {
  return mi->iface->mount_is_loaded() ? 1 : 0;
}

OlsError ols_mount_connect(OlsMountInterface *mi, int32_t load_alignment) {
  try {
    auto *mount = mi->iface->client();
    ols::MountErrorCode e;
    mount->connect(load_alignment != 0, e);
    if (e) return set_error(OLS_ERR_IO, e.message());
    return OLS_OK;
  } catch (std::exception const &ex) {
    return set_error(OLS_ERR_GENERIC, ex);
  }
}

int32_t ols_mount_connected(OlsMountInterface *mi) {
  try {
    if (!mi->iface->mount_is_loaded()) return 0;
    return mi->iface->client()->connected() ? 1 : 0;
  } catch (...) {
    return 0;
  }
}

OlsError ols_mount_get_current(OlsMountInterface *mi, OlsEqCoord *out_coord) {
  try {
    auto *mount = mi->iface->client();
    ols::MountErrorCode e;
    auto coord = mount->get_current(e);
    if (e) return set_error(OLS_ERR_IO, e.message());
    out_coord->ra = coord.RA;
    out_coord->dec = coord.DEC;
    return OLS_OK;
  } catch (std::exception const &ex) {
    return set_error(OLS_ERR_GENERIC, ex);
  }
}

OlsError ols_mount_goto(OlsMountInterface *mi, double ra, double dec) {
  try {
    auto *mount = mi->iface->client();
    ols::MountErrorCode e;
    ols::EqCoord coord;
    coord.RA = ra;
    coord.DEC = dec;
    mount->go_to(coord, e);
    if (e) return set_error(OLS_ERR_IO, e.message());
    return OLS_OK;
  } catch (std::exception const &ex) {
    return set_error(OLS_ERR_GENERIC, ex);
  }
}

OlsError ols_mount_sync(OlsMountInterface *mi, double ra, double dec) {
  try {
    auto *mount = mi->iface->client();
    ols::MountErrorCode e;
    ols::EqCoord coord;
    coord.RA = ra;
    coord.DEC = dec;
    mount->sync(coord, e);
    if (e) return set_error(OLS_ERR_IO, e.message());
    return OLS_OK;
  } catch (std::exception const &ex) {
    return set_error(OLS_ERR_GENERIC, ex);
  }
}

OlsError ols_mount_slew(OlsMountInterface *mi, OlsMountSlew direction, int32_t speed) {
  try {
    auto *mount = mi->iface->client();
    ols::MountErrorCode e;
    mount->slew((ols::MountSlew)direction, speed, e);
    if (e) return set_error(OLS_ERR_IO, e.message());
    return OLS_OK;
  } catch (std::exception const &ex) {
    return set_error(OLS_ERR_GENERIC, ex);
  }
}

OlsError ols_mount_get_tracking(OlsMountInterface *mi, OlsMountTracking *out_mode) {
  try {
    auto *mount = mi->iface->client();
    ols::MountErrorCode e;
    auto mode = mount->get_tracking(e);
    if (e) return set_error(OLS_ERR_IO, e.message());
    *out_mode = (OlsMountTracking)mode;
    return OLS_OK;
  } catch (std::exception const &ex) {
    return set_error(OLS_ERR_GENERIC, ex);
  }
}

OlsError ols_mount_get_tracking_state(OlsMountInterface *mi, int32_t *out_state) {
  try {
    auto *mount = mi->iface->client();
    ols::MountErrorCode e;
    *out_state = mount->get_tracking_state(e);
    if (e) return set_error(OLS_ERR_IO, e.message());
    return OLS_OK;
  } catch (std::exception const &ex) {
    return set_error(OLS_ERR_GENERIC, ex);
  }
}

OlsError ols_mount_set_tracking_state(OlsMountInterface *mi, int32_t track) {
  try {
    auto *mount = mi->iface->client();
    ols::MountErrorCode e;
    mount->set_tracking_state(track != 0, e);
    if (e) return set_error(OLS_ERR_IO, e.message());
    return OLS_OK;
  } catch (std::exception const &ex) {
    return set_error(OLS_ERR_GENERIC, ex);
  }
}

OlsError ols_mount_set_tracking_mode(OlsMountInterface *mi, OlsMountTracking mode) {
  try {
    auto *mount = mi->iface->client();
    ols::MountErrorCode e;
    mount->set_tracking((ols::MountTrac)mode, e);
    if (e) return set_error(OLS_ERR_IO, e.message());
    return OLS_OK;
  } catch (std::exception const &ex) {
    return set_error(OLS_ERR_GENERIC, ex);
  }
}

OlsError ols_mount_set_lat_lon(OlsMountInterface *mi, double lat, double lon) {
  try {
    auto *mount = mi->iface->client();
    ols::MountErrorCode e;
    mount->set_lat_lon(lat, lon, e);
    if (e) return set_error(OLS_ERR_IO, e.message());
    return OLS_OK;
  } catch (std::exception const &ex) {
    return set_error(OLS_ERR_GENERIC, ex);
  }
}

OlsError ols_mount_get_lat_lon(OlsMountInterface *mi, double *out_lat, double *out_lon) {
  try {
    auto *mount = mi->iface->client();
    ols::MountErrorCode e;
    auto ll = mount->get_lat_lon(e);
    if (e) return set_error(OLS_ERR_IO, e.message());
    *out_lat = ll.first;
    *out_lon = ll.second;
    return OLS_OK;
  } catch (std::exception const &ex) {
    return set_error(OLS_ERR_GENERIC, ex);
  }
}

OlsError ols_mount_get_slew_rate(OlsMountInterface *mi, int32_t *out_current, int32_t *out_count) {
  try {
    auto *mount = mi->iface->client();
    ols::MountErrorCode e;
    auto rate = mount->get_slew_rate(e);
    if (e) return set_error(OLS_ERR_IO, e.message());
    *out_current = rate.first;
    *out_count = rate.second;
    return OLS_OK;
  } catch (std::exception const &ex) {
    return set_error(OLS_ERR_GENERIC, ex);
  }
}

OlsError ols_mount_pulse_guide_supported(OlsMountInterface *mi,
                                          int32_t *out_supported,
                                          float *out_ns_rate, float *out_we_rate) {
  try {
    auto *mount = mi->iface->client();
    ols::MountErrorCode e;
    float ns = 0, we = 0;
    bool supported = mount->pulse_guiding_supported(ns, we, e);
    if (e) return set_error(OLS_ERR_IO, e.message());
    *out_supported = supported ? 1 : 0;
    *out_ns_rate = ns;
    *out_we_rate = we;
    return OLS_OK;
  } catch (std::exception const &ex) {
    return set_error(OLS_ERR_GENERIC, ex);
  }
}

OlsError ols_mount_pulse_guide(OlsMountInterface *mi, float ns_ms, float we_ms) {
  try {
    auto *mount = mi->iface->client();
    ols::MountErrorCode e;
    mount->pulse_guide(ns_ms, we_ms, e);
    if (e) return set_error(OLS_ERR_IO, e.message());
    return OLS_OK;
  } catch (std::exception const &ex) {
    return set_error(OLS_ERR_GENERIC, ex);
  }
}

OlsError ols_mount_get_meridian(OlsMountInterface *mi, OlsMeridianFlip *out_flip) {
  try {
    auto *mount = mi->iface->client();
    ols::MountErrorCode e;
    auto flip = mount->get_meridian(e);
    if (e) return set_error(OLS_ERR_IO, e.message());
    *out_flip = (OlsMeridianFlip)flip;
    return OLS_OK;
  } catch (std::exception const &ex) {
    return set_error(OLS_ERR_GENERIC, ex);
  }
}

OlsError ols_mount_set_meridian(OlsMountInterface *mi, OlsMeridianFlip flip) {
  try {
    auto *mount = mi->iface->client();
    ols::MountErrorCode e;
    mount->set_meridian((ols::MountMeridianFlip)flip, e);
    if (e) return set_error(OLS_ERR_IO, e.message());
    return OLS_OK;
  } catch (std::exception const &ex) {
    return set_error(OLS_ERR_GENERIC, ex);
  }
}

OlsError ols_mount_get_alt_limits(OlsMountInterface *mi, int32_t *out_low, int32_t *out_high) {
  try {
    auto *mount = mi->iface->client();
    ols::MountErrorCode e;
    auto limits = mount->get_alt_limits(e);
    if (e) return set_error(OLS_ERR_IO, e.message());
    *out_low = limits.first;
    *out_high = limits.second;
    return OLS_OK;
  } catch (std::exception const &ex) {
    return set_error(OLS_ERR_GENERIC, ex);
  }
}

OlsError ols_mount_set_alt_limits(OlsMountInterface *mi, int32_t low, int32_t high) {
  try {
    auto *mount = mi->iface->client();
    ols::MountErrorCode e;
    mount->set_alt_limits(low, high, e);
    if (e) return set_error(OLS_ERR_IO, e.message());
    return OLS_OK;
  } catch (std::exception const &ex) {
    return set_error(OLS_ERR_GENERIC, ex);
  }
}

OlsError ols_mount_shutdown(OlsMountInterface *mi) {
  try {
    mi->iface->shutdown_mount();
    return OLS_OK;
  } catch (std::exception const &ex) {
    return set_error(OLS_ERR_GENERIC, ex);
  }
}

void ols_mount_interface_destroy(OlsMountInterface *mi) { delete mi; }

// ═══════════════════════════════════════════════════════════════════
//  Focuser API Implementation
// ═══════════════════════════════════════════════════════════════════

struct OlsFocuserInterface_t {
  // We use the OLS singleton directly but wrap access
  bool driver_loaded = false;
};

OlsError ols_focuser_interface_create(OlsFocuserInterface **out_fi) {
  try {
    auto *fi = new OlsFocuserInterface_t();
    *out_fi = fi;
    return OLS_OK;
  } catch (std::exception const &ex) {
    return set_error(OLS_ERR_GENERIC, ex);
  }
}

OlsError ols_focuser_driver_list(OlsFocuserInterface *fi,
                                  OlsFocuserDriverInfo *out_drivers, int32_t max_drivers,
                                  int32_t *out_count) {
  (void)fi;
  try {
    auto drivers = ols::FocuserInterface::instance().drivers_list();
    *out_count = std::min((int32_t)drivers.size(), max_drivers);
    for (int32_t i = 0; i < *out_count; i++) {
      strncpy(out_drivers[i].name, drivers[i].name.c_str(), 127);
      out_drivers[i].name[127] = '\0';
      strncpy(out_drivers[i].option, drivers[i].opt.c_str(), 127);
      out_drivers[i].option[127] = '\0';
    }
    return OLS_OK;
  } catch (std::exception const &ex) {
    return set_error(OLS_ERR_GENERIC, ex);
  }
}

OlsError ols_focuser_load_driver(OlsFocuserInterface *fi,
                                  const char *driver_name, const char *option) {
  try {
    ols::FocuserInfo info;
    info.name = driver_name;
    info.opt = option ? option : "auto";
    ols::FocuserErrorCode e;
    ols::FocuserInterface::instance().load_driver(info, e);
    if (e) return set_error(OLS_ERR_IO, e.message());
    fi->driver_loaded = true;
    return OLS_OK;
  } catch (std::exception const &ex) {
    return set_error(OLS_ERR_GENERIC, ex);
  }
}

int32_t ols_focuser_is_loaded(OlsFocuserInterface *fi) {
  (void)fi;
  return ols::FocuserInterface::focuser() != nullptr ? 1 : 0;
}

OlsError ols_focuser_position(OlsFocuserInterface *fi, int32_t *out_position) {
  (void)fi;
  try {
    auto *f = ols::FocuserInterface::focuser();
    if (!f) return set_error(OLS_ERR_NOT_CONNECTED, "Focuser not loaded");
    ols::FocuserErrorCode e;
    *out_position = f->position(e);
    if (e) return set_error(OLS_ERR_IO, e.message());
    return OLS_OK;
  } catch (std::exception const &ex) {
    return set_error(OLS_ERR_GENERIC, ex);
  }
}

OlsError ols_focuser_move_to(OlsFocuserInterface *fi, int32_t position) {
  (void)fi;
  try {
    auto *f = ols::FocuserInterface::focuser();
    if (!f) return set_error(OLS_ERR_NOT_CONNECTED, "Focuser not loaded");
    ols::FocuserErrorCode e;
    f->move_to(position, e);
    if (e) return set_error(OLS_ERR_IO, e.message());
    return OLS_OK;
  } catch (std::exception const &ex) {
    return set_error(OLS_ERR_GENERIC, ex);
  }
}

OlsError ols_focuser_move_by(OlsFocuserInterface *fi, int32_t delta) {
  (void)fi;
  try {
    auto *f = ols::FocuserInterface::focuser();
    if (!f) return set_error(OLS_ERR_NOT_CONNECTED, "Focuser not loaded");
    ols::FocuserErrorCode e;
    f->move_by(delta, e);
    if (e) return set_error(OLS_ERR_IO, e.message());
    return OLS_OK;
  } catch (std::exception const &ex) {
    return set_error(OLS_ERR_GENERIC, ex);
  }
}

OlsError ols_focuser_abort(OlsFocuserInterface *fi) {
  (void)fi;
  try {
    auto *f = ols::FocuserInterface::focuser();
    if (!f) return set_error(OLS_ERR_NOT_CONNECTED, "Focuser not loaded");
    ols::FocuserErrorCode e;
    f->abort(e);
    if (e) return set_error(OLS_ERR_IO, e.message());
    return OLS_OK;
  } catch (std::exception const &ex) {
    return set_error(OLS_ERR_GENERIC, ex);
  }
}

OlsError ols_focuser_is_moving(OlsFocuserInterface *fi, int32_t *out_moving) {
  (void)fi;
  try {
    auto *f = ols::FocuserInterface::focuser();
    if (!f) return set_error(OLS_ERR_NOT_CONNECTED, "Focuser not loaded");
    ols::FocuserErrorCode e;
    *out_moving = f->is_moving(e) ? 1 : 0;
    if (e) return set_error(OLS_ERR_IO, e.message());
    return OLS_OK;
  } catch (std::exception const &ex) {
    return set_error(OLS_ERR_GENERIC, ex);
  }
}

OlsError ols_focuser_is_absolute(OlsFocuserInterface *fi, int32_t *out_absolute) {
  (void)fi;
  try {
    auto *f = ols::FocuserInterface::focuser();
    if (!f) return set_error(OLS_ERR_NOT_CONNECTED, "Focuser not loaded");
    ols::FocuserErrorCode e;
    *out_absolute = f->is_absolute(e) ? 1 : 0;
    if (e) return set_error(OLS_ERR_IO, e.message());
    return OLS_OK;
  } catch (std::exception const &ex) {
    return set_error(OLS_ERR_GENERIC, ex);
  }
}

OlsError ols_focuser_max_range(OlsFocuserInterface *fi, int32_t *out_range) {
  (void)fi;
  try {
    auto *f = ols::FocuserInterface::focuser();
    if (!f) return set_error(OLS_ERR_NOT_CONNECTED, "Focuser not loaded");
    ols::FocuserErrorCode e;
    *out_range = f->max_range(e);
    if (e) return set_error(OLS_ERR_IO, e.message());
    return OLS_OK;
  } catch (std::exception const &ex) {
    return set_error(OLS_ERR_GENERIC, ex);
  }
}

OlsError ols_focuser_unload(OlsFocuserInterface *fi) {
  try {
    ols::FocuserInterface::instance().unload_driver();
    fi->driver_loaded = false;
    return OLS_OK;
  } catch (std::exception const &ex) {
    return set_error(OLS_ERR_GENERIC, ex);
  }
}

void ols_focuser_interface_destroy(OlsFocuserInterface *fi) {
  if (fi && fi->driver_loaded) {
    ols::FocuserInterface::instance().unload_driver();
  }
  delete fi;
}

} // extern "C"
