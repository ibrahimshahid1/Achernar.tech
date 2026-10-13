// Production control writes and a real production preview thread, with all
// camera-facing PTP calls mocked. No USB device is opened or shutter triggered.
#include <gphoto2/gphoto2.h>
extern "C" int fixture_widget_get_value(CameraWidget *, void *);
#define gp_widget_get_value fixture_widget_get_value
#include "../src/gphoto2_camera.cpp"
#undef gp_widget_get_value

#include <condition_variable>
#include <future>
#include <stdexcept>

namespace fixture {
struct Setting {
    std::vector<std::string> choices;
    std::string value;
    bool read_only = false;
};
std::map<std::string, Setting> settings = {
    {"iso", {{"100", "250", "800"}, "250"}},
    {"shutterspeed", {{"1/13", "1/125"}, "1/13"}},
    {"capturetarget", {{"Internal RAM", "Memory card"}, "Memory card"}},
};
std::mutex mutex;
std::condition_variable changed;
std::vector<unsigned char> jpeg;
std::atomic<int> active_ptp{0};
std::atomic<bool> overlap{false};
std::atomic<bool> fail_value_read{false};
std::atomic<int> config_result{GP_OK};
std::atomic<int> preview_result{GP_OK};
std::atomic<int> preview_calls{0};
int trigger_calls = 0;
bool fail_write = false;
bool block_preview = false;
bool preview_blocked = false;
bool release_preview = false;
int event_index = 0;
int committed_writes = 0;
std::thread::id last_writer;

struct PtpCall {
    PtpCall() { if(active_ptp.fetch_add(1) != 0) overlap = true; }
    ~PtpCall() { --active_ptp; }
};
void require(bool value, const char *message) {
    if(!value) throw std::runtime_error(message);
}
int populate_file(CameraFile *file) {
    char *bytes = static_cast<char *>(std::malloc(jpeg.size()));
    std::memcpy(bytes,jpeg.data(),jpeg.size());
    return gp_file_set_data_and_size(file,bytes,jpeg.size());
}
} // namespace fixture

extern "C" int fixture_widget_get_value(CameraWidget *widget, void *value) {
    if(fixture::fail_value_read.exchange(false)) return GP_ERROR_IO;
    return gp_widget_get_value(widget,value);
}

extern "C" int gp_camera_get_single_config(::Camera *, const char *key,
                                            CameraWidget **widget, GPContext *) {
    fixture::PtpCall call;
    std::lock_guard<std::mutex> guard(fixture::mutex);
    if(fixture::config_result != GP_OK) {
        *widget = nullptr;
        return fixture::config_result;
    }
    auto found = fixture::settings.find(key);
    *widget = nullptr;
    if(found == fixture::settings.end()) return GP_ERROR_NOT_SUPPORTED;
    const auto &setting = found->second;
    int status = gp_widget_new(GP_WIDGET_MENU,key,widget);
    if(status != GP_OK) return status;
    for(const auto &choice : setting.choices) {
        status = gp_widget_add_choice(*widget,choice.c_str());
        if(status != GP_OK) return status;
    }
    status = gp_widget_set_value(*widget,setting.value.c_str());
    if(status != GP_OK) return status;
    return gp_widget_set_readonly(*widget,setting.read_only);
}

extern "C" int gp_camera_set_single_config(::Camera *, const char *key,
                                            CameraWidget *widget, GPContext *) {
    fixture::PtpCall call;
    std::lock_guard<std::mutex> guard(fixture::mutex);
    if(fixture::fail_write) return GP_ERROR_IO;
    auto found = fixture::settings.find(key);
    if(found == fixture::settings.end()) return GP_ERROR_NOT_SUPPORTED;
    char const *value = nullptr;
    int status = gp_widget_get_value(widget,&value);
    if(status != GP_OK) return status;
    found->second.value = value;
    ++fixture::committed_writes;
    fixture::last_writer = std::this_thread::get_id();
    return GP_OK;
}

extern "C" int gp_camera_capture_preview(::Camera *, CameraFile *file, GPContext *) {
    fixture::PtpCall call;
    std::unique_lock<std::mutex> guard(fixture::mutex);
    ++fixture::preview_calls;
    if(fixture::preview_result != GP_OK) return fixture::preview_result;
    if(fixture::block_preview) {
        fixture::preview_blocked = true;
        fixture::changed.notify_all();
        fixture::changed.wait(guard,[] { return fixture::release_preview; });
        fixture::block_preview = false;
    }
    return fixture::populate_file(file);
}

extern "C" int gp_camera_trigger_capture(::Camera *, GPContext *) {
    fixture::PtpCall call;
    fixture::event_index = 0;
    ++fixture::trigger_calls;
    return GP_OK;
}

extern "C" int gp_camera_wait_for_event(::Camera *, int, CameraEventType *type,
                                         void **data, GPContext *) {
    fixture::PtpCall call;
    *data = nullptr;
    if(fixture::event_index++ == 0) {
        auto path = static_cast<CameraFilePath *>(std::calloc(1,sizeof(CameraFilePath)));
        std::strcpy(path->folder,"/mock");
        std::strcpy(path->name,"preview.JPG");
        *data = path;
        *type = GP_EVENT_FILE_ADDED;
    } else {
        *type = GP_EVENT_CAPTURE_COMPLETE;
    }
    return GP_OK;
}

extern "C" int gp_camera_file_get(::Camera *, const char *, const char *,
                                    CameraFileType, CameraFile *file, GPContext *) {
    fixture::PtpCall call;
    return fixture::populate_file(file);
}

int main() {
    using namespace fixture;
    try {
        require(cv::imencode(".jpg",cv::Mat(4,6,CV_8UC3,cv::Scalar(20,80,180)),jpeg),
                "Could not create fixture preview JPEG");
        std::pair<int,int> dimensions;
        require(ols::preview_dimensions(jpeg,dimensions) && dimensions == std::make_pair(6,4),
                "JPEG preview header dimensions were not preserved");
        require(!ols::preview_dimensions({0xff,0xd8,0xff,0xe0,0,20},dimensions),
                "Truncated preview header was accepted");
        require(!ols::preview_dimensions({0xff,0xd8,0xff,0xc0,0,8,8,0,0,0,6,1},dimensions),
                "Zero-height preview was accepted");
        ols::GP2Camera camera(nullptr,nullptr);
        ols::CamErrorCode error;
        require(camera.check_connection(error) && !error && trigger_calls == 0,
                "Presence check did not use a passive device transaction");
        config_result = GP_ERROR_IO_READ;
        require(camera.check_connection(error) && bool(error),
                "Unplugged camera presence error was suppressed");
        error = ols::CamErrorCode();
        config_result = GP_ERROR_CAMERA_BUSY;
        require(camera.check_connection(error) && !error,
                "A device that answered Busy was treated as disconnected");
        config_result = GP_OK;
        auto iso = camera.get_parameter(ols::opt_iso,false,error);
        require(!error && iso.cur_val == 1 && iso.names[1] == "250" && iso.step_size == 1,
                "ISO choices lost their native index semantics");

        settings.at("iso").read_only = true;
        error = ols::CamErrorCode();
        iso = camera.get_parameter(ols::opt_iso,false,error);
        require(!error && iso.read_only,"Camera read-only flag was lost");
        camera.set_parameter(ols::opt_iso,2,error);
        require(bool(error) && committed_writes == 0,"Read-only ISO write was accepted");
        settings.at("iso").read_only = false;

        fail_value_read = true;
        error = ols::CamErrorCode();
        camera.get_parameter(ols::opt_iso,false,error);
        require(bool(error),"Native current-value read failure was ignored");
        settings.at("iso").value = "unadvertised";
        error = ols::CamErrorCode();
        camera.get_parameter(ols::opt_iso,false,error);
        require(bool(error),"Unadvertised ISO was returned as an out-of-range index");
        settings.at("iso").value = "250";

        auto advertised = settings.at("iso").choices;
        settings.at("iso").choices.clear();
        error = ols::CamErrorCode();
        camera.get_parameter(ols::opt_iso,false,error);
        require(bool(error),"Empty choice list was reported as usable");
        settings.at("iso").choices = advertised;

        error = ols::CamErrorCode();
        const auto options = camera.supported_options(error);
        require(std::find(options.begin(),options.end(),ols::opt_capture_delay) == options.end(),
                "Inert inter-shot delay was advertised for supported acquisition");
        require(camera.get_parameter(ols::opt_keep_images,false,error).read_only,
                "Keep Images policy was not marked read-only");
        error = ols::CamErrorCode();
        const auto formats = camera.formats(error);
        require(!error && formats.size() == 1 && formats[0].framerate > 0,
                "Only nonactuating DSLR live view may be enumerated");
        require(trigger_calls == 0,
                "Live-view format discovery actuated the shutter");
        camera.formats(error);
        require(trigger_calls == 0,
                "Repeated live-view format discovery actuated the shutter");

        block_preview = true;
        camera.start_stream(formats.front(),[](const ols::CamFrame &) {},error);
        require(!error,"Synthetic production stream did not start");
        {
            std::unique_lock<std::mutex> guard(mutex);
            require(changed.wait_for(guard,std::chrono::seconds(2),[] { return preview_blocked; }),
                    "Synthetic preview never entered PTP call");
        }
        auto setter = std::async(std::launch::async,[&camera] {
            ols::CamErrorCode result;
            auto caller = std::this_thread::get_id();
            camera.set_parameter(ols::opt_iso,2,result);
            return std::make_pair(result,caller);
        });
        const bool waited_for_preview = setter.wait_for(std::chrono::milliseconds(50))
            == std::future_status::timeout;
        {
            std::lock_guard<std::mutex> guard(mutex);
            release_preview = true;
        }
        changed.notify_all();
        const auto result = setter.get();
        require(waited_for_preview,"Active-stream write returned before in-flight preview completed");
        require(!result.first && last_writer == result.second,
                "Active-stream write was queued rather than completed by its caller");
        error = ols::CamErrorCode();
        iso = camera.get_parameter(ols::opt_iso,false,error);
        require(!error && iso.cur_val == 2 && iso.names[2] == "800",
                "Immediate readback still saw the previous ISO");

        {
            std::lock_guard<std::mutex> guard(mutex);
            fail_write = true;
        }
        error = ols::CamErrorCode();
        camera.set_parameter(ols::opt_iso,0,error);
        require(bool(error),"Active-stream SDK write failure did not reach its caller");
        {
            std::lock_guard<std::mutex> guard(mutex);
            fail_write = false;
        }
        error = ols::CamErrorCode();
        iso = camera.get_parameter(ols::opt_iso,false,error);
        require(!error && iso.cur_val == 2,"Failed ISO write changed readback");
        camera.set_parameter(ols::opt_shutter,1,error);
        const auto shutter = camera.get_parameter(ols::opt_shutter,false,error);
        require(!error && shutter.cur_val == 1 && shutter.names[1] == "1/125",
                "Streaming shutter write did not complete before readback");
        camera.stop_stream(error);
        require(trigger_calls == 0,
                "Starting or stopping preview actuated the shutter");
        require(!overlap,"Config PTP traffic overlapped preview PTP traffic");
        preview_result = GP_ERROR_IO_READ;
        const int before_failure = preview_calls;
        std::promise<void> failed;
        auto fault = failed.get_future();
        camera.start_stream(formats.front(),[&failed](const ols::CamFrame &frame) {
            if(frame.format == ols::stream_error) failed.set_value();
        },error);
        require(fault.wait_for(std::chrono::seconds(2)) == std::future_status::ready,
                "Transport failure did not publish a stream fault");
        camera.stop_stream(error);
        require(preview_calls == before_failure + 1,
                "Preview kept retrying after a terminal transport failure");
        preview_result = GP_ERROR_NOT_SUPPORTED;
        error = ols::CamErrorCode();
        ols::GP2Camera no_preview_camera(nullptr,nullptr);
        require(no_preview_camera.formats(error).empty() && !error,
                "Unsupported live view did not return an empty nonactuating format list");
        require(trigger_calls == 0,
                "Missing live view triggered a still exposure as a fallback");
        std::cout << "Native controls: streaming write serialization/completion, immediate ISO/shutter readback, "
                     "SDK failure, read-only flags, invalid choice reads, retention policy and nonactuating format discovery passed.\n";
        return 0;
    } catch(const std::exception &error) {
        std::cerr << "Native controls regression failed: " << error.what() << '\n';
        return 1;
    }
}
