#include "camera.h"
#include <gphoto2/gphoto2.h>
#include <gphoto2/gphoto2-port.h>
#include <mutex>
#include <sstream>
#include <iostream>
#include <atomic>
#include <map>
#include <cctype>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>
#include <chrono>
#include <thread>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#ifdef WITH_LIBRAW
#include "libraw_wrapper.h"
#endif

namespace ols {

    static FILE *error_stream = NULL;
    extern "C" {
        static void ctx_error_func(GPContext *, char const *msg, void *)
        {
            FILE *f = !error_stream ? stderr : error_stream;
            fprintf(f,"GP2 Error:%s\n",msg);
            fflush(f);
        }
        static void ctx_status_func(GPContext *, char const *msg, void *)
        {
            FILE *f = !error_stream ? stderr : error_stream;
            fprintf(f,"GP2 Status:%s\n",msg);
            fflush(f);
        }
        static void errordumper(GPLogLevel /*level*/, const char *domain, const char *str,
                 void * /*data*/) 
        {
            FILE *f = !error_stream ? stderr : error_stream;
            fprintf(f,"GP2 LOG:%s:%s\n",domain,str);
            fflush(f);
        }
    }

    #define LOG(...) do { if( ::ols::error_stream) { fprintf( ::ols::error_stream,__VA_ARGS__); fflush(::ols::error_stream); }} while(0)

        
    static constexpr int MAX_FILES = 3;

    static bool check(int err,char const *msg,CamErrorCode &e)
    {
        if(err != GP_OK) {
            e = CamErrorCode(std::string(msg) + ":" + gp_result_as_string(err));
            return false;
        }
        return true;
    }

    class GPError : public CamError {
    public:
        GPError(std::string const &msg) : CamError(msg)
        {
        }
        GPError(std::string const &msg,int code) :
            CamError(std::string(msg) + ":" + gp_result_as_string(code))
        {

        }
    };

    static void check(int status,char const *msg)
    {
        if(status != GP_OK) {
            throw GPError(msg,status);
        }
    }

    /// ─── Shutter / ISO label arithmetic ───
    ///
    /// DSLR bodies expose exposure and gain as enumerations whose *labels*
    /// carry the physical meaning: choice 17 means nothing, "1/125" means
    /// eight milliseconds. Selecting an exposure therefore means parsing the
    /// ladder and picking a step.
    ///
    /// This mirrors the rule in the daemon's `hardware/dslr_controls.rs`,
    /// which does the same parsing for the *UI* (rendering "1/125" next to a
    /// slider). The two must agree; they are kept apart because capture must
    /// resolve the step on the camera thread, where the read-back happens,
    /// and cannot wait for a round trip to Rust. Any change to one belongs in
    /// the other, and dslr_controls.rs carries the unit tests.

    /// Parse a shutterspeed label into seconds. Handles "1/125", "30", "0.3",
    /// Panasonic's "30\"" and "2.5s". Returns <= 0 for non-durations
    /// ("bulb", "Auto"), which must never be treated as a time.
    static double parse_shutter_seconds(std::string const &label)
    {
        std::string s;
        for(char c : label) {
            if(!std::isspace(static_cast<unsigned char>(c)))
                s += c;
        }
        while(!s.empty() && (s.back()=='"' || s.back()=='s' || s.back()=='S' || s.back()=='\''))
            s.pop_back();
        if(s.empty())
            return -1;

        auto to_num = [](std::string const &t,double &out) -> bool {
            if(t.empty())
                return false;
            char *end = nullptr;
            double v = strtod(t.c_str(),&end);
            if(end != t.c_str() + t.size())
                return false;
            out = v;
            return true;
        };

        size_t slash = s.find('/');
        if(slash != std::string::npos) {
            double num=0,den=0;
            if(!to_num(s.substr(0,slash),num) || !to_num(s.substr(slash+1),den))
                return -1;
            if(den == 0)
                return -1;
            double v = num/den;
            return (std::isfinite(v) && v > 0) ? v : -1;
        }
        double v=0;
        if(!to_num(s,v))
            return -1;
        return (std::isfinite(v) && v > 0) ? v : -1;
    }

    /// Parse an ISO label. Accepts "800" and "ISO 800"; rejects "Auto" and the
    /// extended "Hi"/"Lo" pseudo-stops, which have no numeric gain.
    static double parse_iso(std::string const &label)
    {
        std::string s;
        size_t i = 0;
        if(label.size() >= 3) {
            std::string head = label.substr(0,3);
            if(head == "ISO" || head == "iso")
                i = 3;
        }
        for(;i<label.size();i++) {
            if(!std::isspace(static_cast<unsigned char>(label[i])))
                s += label[i];
        }
        if(s.empty())
            return -1;
        char *end = nullptr;
        double v = strtod(s.c_str(),&end);
        if(end != s.c_str() + s.size())
            return -1;
        return (std::isfinite(v) && v > 0) ? v : -1;
    }

    /// Index of the choice closest to `target`, compared in log space because
    /// both ladders are geometric — between 1/125 and 1/60, a request for
    /// 1/100 is near-equidistant in *stops*, and a linear comparison biases
    /// hard toward the longer exposure. Unparseable labels are skipped.
    /// Returns -1 when no label is numeric.
    static int nearest_choice(std::vector<std::string> const &labels,
                              double (*parse)(std::string const &),
                              double target,
                              double &out_value,
                              std::string &out_label)
    {
        if(!(std::isfinite(target) && target > 0))
            return -1;
        double log_target = std::log(target);
        int best = -1;
        double best_dist = 0;
        for(size_t i=0;i<labels.size();i++) {
            double v = parse(labels[i]);
            if(v <= 0)
                continue;
            double d = std::fabs(std::log(v) - log_target);
            if(best < 0 || d < best_dist) {
                best = int(i);
                best_dist = d;
                out_value = v;
                out_label = labels[i];
            }
        }
        return best;
    }

    // Read JPEG SOF dimensions without decoding every video frame to RGB.
    // This also handles a body changing its EVF geometry during the session.
    static bool preview_dimensions(std::vector<unsigned char> const &jpeg,
                                   std::pair<int,int> &dims)
    {
        if(jpeg.size() < 4 || jpeg[0] != 0xff || jpeg[1] != 0xd8)
            return false;
        size_t at = 2;
        while(at < jpeg.size()) {
            if(jpeg[at++] != 0xff)
                return false;
            while(at < jpeg.size() && jpeg[at] == 0xff) ++at;
            if(at >= jpeg.size()) return false;
            unsigned marker = jpeg[at++];
            if(marker == 0xd9 || marker == 0xda || marker == 0) return false;
            if(marker == 0x01 || (marker >= 0xd0 && marker <= 0xd8)) continue;
            if(jpeg.size() - at < 2) return false;
            size_t length = (size_t(jpeg[at]) << 8) | jpeg[at+1];
            if(length < 2 || length > jpeg.size() - at) return false;
            bool sof = marker >= 0xc0 && marker <= 0xcf &&
                       marker != 0xc4 && marker != 0xc8 && marker != 0xcc;
            if(sof) {
                if(length < 8) return false;
                int height = (int(jpeg[at+3]) << 8) | jpeg[at+4];
                int width = (int(jpeg[at+5]) << 8) | jpeg[at+6];
                if(width == 0 || height == 0) return false;
                dims = std::make_pair(width,height);
                return true;
            }
            at += length;
        }
        return false;
    }

    // Opt-in stage timing for real camera previews. Aggregate outside the SDK
    // mutex instead of logging per frame; normal acquisition adds no clocks.
    // SDK time includes libgphoto2's own event polls, retries and USB transfers.
    struct PreviewPullTiming {
        double lock_wait_ms = 0;
        double sdk_ms = 0;
        double payload_ms = 0;
    };

    static double preview_milliseconds(std::chrono::steady_clock::duration elapsed)
    {
        return std::chrono::duration<double,std::milli>(elapsed).count();
    }

    struct PreviewTimingMetric {
        double sum = 0;
        double maximum = 0;
        void add(double value) {
            sum += value;
            maximum = std::max(maximum,value);
        }
    };

    struct PreviewTimingWindow {
        std::chrono::steady_clock::time_point started = std::chrono::steady_clock::now();
        unsigned long long attempts = 0;
        unsigned long long frames = 0;
        unsigned long long bytes = 0;
        PreviewTimingMetric lock_wait, sdk, payload, callback, pacing;

        void add(PreviewPullTiming const &pull, bool delivered, size_t payload_bytes,
                 double callback_ms, double pacing_ms)
        {
            ++attempts;
            frames += delivered ? 1 : 0;
            bytes += delivered ? payload_bytes : 0;
            lock_wait.add(pull.lock_wait_ms);
            sdk.add(pull.sdk_ms);
            payload.add(pull.payload_ms);
            if(delivered) callback.add(callback_ms);
            pacing.add(pacing_ms);
        }

        // Mean pull/pacing times use attempts; callback mean uses delivered
        // frames. Pauses and SDK error backoff remain in wall-clock FPS.
        void write(FILE *sink, int frame_counter, int width, int height,
                   double requested_fps,
                   std::chrono::steady_clock::time_point now) const
        {
            if(attempts == 0) return;
            double elapsed_s = std::chrono::duration<double>(now-started).count();
            double mean_callback = frames ? callback.sum / frames : 0;
            fprintf(sink,
                "[gphoto2-preview-timing] {\"version\":1,\"attempts\":%llu,"
                "\"frames\":%llu,\"elapsed_s\":%.6f,\"fps\":%.6f,"
                "\"frame_counter\":%d,\"width\":%d,\"height\":%d,"
                "\"requested_fps\":%.3f,\"mean_jpeg_bytes\":%.3f,"
                "\"mean_lock_wait_ms\":%.6f,\"max_lock_wait_ms\":%.6f,"
                "\"mean_sdk_ms\":%.6f,\"max_sdk_ms\":%.6f,"
                "\"mean_payload_ms\":%.6f,\"max_payload_ms\":%.6f,"
                "\"mean_callback_ms\":%.6f,\"max_callback_ms\":%.6f,"
                "\"mean_pacing_ms\":%.6f,\"max_pacing_ms\":%.6f}\n",
                attempts,frames,elapsed_s,elapsed_s > 0 ? frames / elapsed_s : 0,
                frame_counter,width,height,requested_fps,frames ? double(bytes) / frames : 0,
                lock_wait.sum / attempts,lock_wait.maximum,sdk.sum / attempts,sdk.maximum,
                payload.sum / attempts,payload.maximum,mean_callback,callback.maximum,
                pacing.sum / attempts,pacing.maximum);
            fflush(sink);
        }
    };

    class GPWidget {
    public:
        GPWidget( ::Camera *c,GPContext *ct,std::string const &key) :
            cam_(c),ctx_(ct),key_(key),widget_(nullptr)
        {
            
        }

        void set(double value)
        {
            load();
            int res;
            int read_only = 0;
            res = gp_widget_get_readonly(widget_,&read_only);
            check(res,"get_widget_readonly");
            if(read_only)
                throw GPError("Camera control '" + key_ + "' is read-only");
            CameraWidgetType type;
            res = gp_widget_get_type(widget_,&type);
            check(res,"get_widget_type");
            switch(type) {
            case GP_WIDGET_TOGGLE:
                {
                    int val = value;
                    res = gp_widget_set_value(widget_,&val);
                    check(res,"set value");
                }
                break;
            case GP_WIDGET_RADIO:
            case GP_WIDGET_MENU:
                {
                    char const *val = nullptr;
                    res = gp_widget_get_choice(widget_,int(value),&val);
                    check(res,"widget_get_choice");
                    res = gp_widget_set_value(widget_,val);
                    check(res,"widget set value string");
                }
                break;
            default:
                throw GPError("Unsupported widget type");
            }
            res = gp_camera_set_single_config(cam_,key_.c_str(),widget_,ctx_);
            check(res,"set signle config");
        }

        CamParam get(CamOptionId id)
        {
            load();
            CamParam r;
            r.option = id;
            CameraWidgetType type;
            int res;
            res = gp_widget_get_type(widget_,&type);
            check(res,"get_widget_type");
            int read_only = 0;
            res = gp_widget_get_readonly(widget_,&read_only);
            check(res,"get_widget_readonly");
            r.read_only = read_only != 0;
            switch(type) {
            case GP_WIDGET_TOGGLE:
                {
                    int ival = -1;
                    res = gp_widget_get_value(widget_,&ival);
                    check(res,"widget_get_value");
                    r.type = type_bool;
                    r.step_size = 1;
                    r.min_val = 0;
                    r.max_val = 1;
                    r.def_val = r.cur_val = ival == 1; 
                }
                break;
            case GP_WIDGET_RADIO:
            case GP_WIDGET_MENU:
                {
                    int N = gp_widget_count_choices(widget_);
                    if(N <= 0)
                        throw GPError("Camera control '" + key_ + "' has no selectable choices");
                    char const *val = nullptr;
                    for(int i=0;i<N;i++) {
                        res = gp_widget_get_choice(widget_,i,&val);
                        check(res,"widget_get_choice");
                        r.names.push_back(val);
                    }
                    res = gp_widget_get_value(widget_,&val);
                    check(res,"widget_get_value");
                    if(!val)
                        throw GPError("Camera control '" + key_ + "' returned no current choice");
                    auto current = std::find(r.names.begin(),r.names.end(),std::string(val));
                    if(current == r.names.end())
                        throw GPError("Camera control '" + key_ + "' returned an unadvertised current choice: " + val);

                    r.type = type_selection;
                    r.step_size = 1;
                    r.min_val = 0;
                    r.max_val = N-1;
                    r.cur_val = current - r.names.begin();
                    r.def_val = r.cur_val;
                }
                break;
            default:
                throw GPError("Unsupported widget type " + key_);
            }
            return r;
        }

        bool try_load()
        {
            if(widget_)
                return true;
            int res = gp_camera_get_single_config(cam_,key_.c_str(),&widget_,ctx_);
            if(res < GP_OK)
                return false;
            return true;
        }

        void load()
        {
            if(widget_)
                return;
            int res = gp_camera_get_single_config(cam_,key_.c_str(),&widget_,ctx_);
            check(res,"get config");
        }
        ~GPWidget()
        {
            if(widget_)
                gp_widget_free(widget_);
        }
        
    private:
        ::Camera *cam_;
        GPContext *ctx_;
        std::string key_;
        CameraWidget *widget_;
    };

    class GPFile {
    public:
        GPFile(CameraFilePath const &path,::Camera *camera, GPContext *context) : 
            file_(nullptr),
            path_(path),
            cam_(camera),
            ctx_(context)
        {
        }
        ~GPFile()
        {
            free();
        }
        void free()
        {
            if(file_)
                gp_file_free(file_);
            file_ = nullptr;
            data_ = nullptr;
            size_ = 0;
        }
        std::string name()
        {
            return path_.folder + std::string("/") + path_.name;
        }
        bool loaded()
        {
            return data_ != nullptr;
        }
        // A failed transfer may leave useful bytes in CameraFile. Read them
        // even on a gphoto error so capture_still can preserve the partial file.
        std::string download_preserving_bytes()
        {
            free();
            int status = gp_file_new(&file_);
            if(status < GP_OK || !file_)
                return std::string("new file: ") + gp_result_as_string(status);
            int transfer_status = gp_camera_file_get(
                cam_,path_.folder,path_.name,GP_FILE_TYPE_NORMAL,file_,ctx_);
            int data_status = gp_file_get_data_and_size(file_,&data_,&size_);
            if(data_status < GP_OK) {
                data_ = nullptr;
                size_ = 0;
            }
            if(transfer_status < GP_OK)
                return std::string("get file: ") + gp_result_as_string(transfer_status);
            if(data_status < GP_OK)
                return std::string("get data: ") + gp_result_as_string(data_status);
            if(!data_ || size_ == 0)
                return "Camera returned an empty native file";
            return std::string();
        }
        void get()
        {
            std::string error = download_preserving_bytes();
            if(!error.empty())
                throw GPError(error);
        }

        std::pair<cv::Mat,CamBayerType> to_mat(int w=-1,int h=-1,int bin=1)
        {
            std::pair<cv::Mat,CamBayerType> r;
            if(!loaded()) {
                get();
            }
            if(is_jpeg()) {
                cv::Mat buffer(1,size_,CV_8UC1,const_cast<char*>(data_));
                r.first = cv::imdecode(buffer,cv::IMREAD_UNCHANGED);
                r.second = bayer_na;
                return r;
            }
            else {
#ifdef WITH_LIBRAW                
                return load_libraw(const_cast<char*>(data_),size_,h,w,bin);
#else
                throw GPError("OpenLiveStacker was build without libraw support");                
#endif                
            }
        }

        char const *data() {
            return data_;
        }
        size_t size() {
            return size_;
        }
        /// Lowercase extension without the dot, as the camera named the file:
        /// "cr2" (T3i/600D, T6s/760D), "cr3" (R50, M50), "rw2" (Lumix G95),
        /// "jpg", … Empty when the name carries no extension.
        std::string ext() const
        {
            char const *cext = strrchr(path_.name,'.');
            if(!cext)
                return std::string();
            std::string e(cext+1);
            for(size_t i=0;i<e.size();i++)
                e[i] = std::tolower(e[i]);
            return e;
        }
        bool is_jpeg()
        {
            std::string e = ext();
            if(e.empty())
                return false;
            return e == "jpeg" || e == "jpg" || e == "jpe" || e == "jif" || e=="jfif" || e=="jfi";
        }
    private:
        CameraFile *file_;
        CameraFilePath path_;
        ::Camera *cam_;
        GPContext *ctx_;
        char const *data_ = nullptr;
        unsigned long size_ = 0;
    };
    
    class GP2Camera : public Camera {
    public:
        /// for Android
        GP2Camera(::Camera *c,GPContext *ctx):
            cam_(c),ctx_(ctx)
        {
            stream_active_ = 0;
        }
        virtual ~GP2Camera()
        {
            CamErrorCode e;
            stop_stream(e);
            if(e) {
                fprintf(stderr,"Failed to close stream %s\n",e.message().c_str());
            }
        }
        /// Camera name.
        ///
        /// Reported as the body's actual model ("Canon EOS R50", "Panasonic
        /// DC-G95", …) rather than the literal string "Camera". This is not
        /// cosmetic: the daemon copies this into the FITS `INSTRUME` card and
        /// into per-device calibration/profile lookups, so a constant name
        /// makes every body on a multi-camera rig indistinguishable in the
        /// archive and silently merges their profiles.
        virtual std::string name(CamErrorCode &)
        {
            std::unique_lock<std::recursive_mutex> guard(clock_);
            CameraAbilities abilities;
            memset(&abilities,0,sizeof(abilities));
            if(gp_camera_get_abilities(cam_,&abilities) == GP_OK && abilities.model[0])
                return std::string(abilities.model);
            return "Camera";
        }

        virtual bool check_connection(CamErrorCode &e) override
        {
            std::unique_lock<std::recursive_mutex> guard(clock_);
            CameraWidget *widget = nullptr;
            // EOS config reads perform a keep-alive/event transaction even
            // when the property's value is cached. Never use name/formats:
            // they are cached, and format enumeration can trigger a still.
            int status = gp_camera_get_single_config(cam_,"iso",&widget,ctx_);
            if(widget)
                gp_widget_free(widget);
            if(status == GP_ERROR_NOT_SUPPORTED)
                return false;
            // A busy response is still evidence that the device answered.
            if(status == GP_ERROR_CAMERA_BUSY)
                return true;
            check(status,"Camera presence check",e);
            return true;
        }

        /// True when the body can deliver a live-view (mirror-up / EVF) stream.
        /// Probed once and cached; failure is not an error, only a capability.
        bool liveview_supported()
        {
            if(liveview_probe_done_)
                return liveview_available_;
            liveview_probe_done_ = true;
            try {
                std::pair<int,int> dims;
                std::vector<unsigned char> jpeg;
                liveview_available_ = grab_preview(jpeg,dims);
                if(liveview_available_) {
                    liveview_width_  = dims.first;
                    liveview_height_ = dims.second;
                    LOG("Liveview available %dx%d\n",liveview_width_,liveview_height_);
                }
            }
            catch(std::exception const &e) {
                LOG("Liveview probe failed: %s\n",e.what());
                liveview_available_ = false;
            }
            return liveview_available_;
        }

        /// Pull one live-view frame. Returns false when the body has no
        /// live-view at all (a plain "not supported" from libgphoto2), throws
        /// only on genuine I/O failure. `out` receives the JPEG bytes verbatim.
        bool grab_preview(std::vector<unsigned char> &out,std::pair<int,int> &dims,
                          PreviewPullTiming *timing = nullptr)
        {
            CameraFile *file = nullptr;
            int status = gp_file_new(&file);
            check(status,"new preview file");
            struct Guard {
                CameraFile *f;
                ~Guard() { if(f) gp_file_free(f); }
            } guard_file{file};

            auto lock_requested = timing ? std::chrono::steady_clock::now()
                                         : std::chrono::steady_clock::time_point();
            {
                std::unique_lock<std::recursive_mutex> guard(clock_);
                auto acquired = timing ? std::chrono::steady_clock::now()
                                       : std::chrono::steady_clock::time_point();
                status = gp_camera_capture_preview(cam_,file,ctx_);
                if(timing) {
                    auto completed = std::chrono::steady_clock::now();
                    timing->lock_wait_ms = preview_milliseconds(acquired-lock_requested);
                    timing->sdk_ms = preview_milliseconds(completed-acquired);
                }
            }
            if(status == GP_ERROR_NOT_SUPPORTED)
                return false;
            if(status == GP_ERROR_CAMERA_BUSY || status == GP_ERROR_TIMEOUT)
                return false;
            // A broken transport cannot be repaired by replaying the last
            // image or repeatedly fetching frames. The outer worker reports
            // this error to Rust and exits; Rust retires the connection.
            check(status,"capture preview");

            auto payload_started = timing ? std::chrono::steady_clock::now()
                                          : std::chrono::steady_clock::time_point();
            char const *data = nullptr;
            unsigned long size = 0;
            status = gp_file_get_data_and_size(file,&data,&size);
            check(status,"preview data");
            if(!data || size == 0)
                return false;

            out.assign(reinterpret_cast<unsigned char const*>(data),
                       reinterpret_cast<unsigned char const*>(data) + size);

            bool valid = preview_dimensions(out,dims);
            if(timing)
                timing->payload_ms = preview_milliseconds(
                    std::chrono::steady_clock::now()-payload_started);
            return valid;
        }

        /// Return list of suppored video formats

        void trigger()
        {
            std::unique_lock<std::recursive_mutex> guard(clock_);
            int status = gp_camera_trigger_capture(cam_,ctx_);
            check(status,"trigger");
        }

        int wait_event(CameraFilePath files[MAX_FILES],bool wait_multiple = false)
        {
            int fno = 0;
            while(fno <= 0 || wait_multiple) {
                if (stop_) {
                    return fno;
                }
                CameraEventType ev = GP_EVENT_UNKNOWN;
                void *ptr = NULL;
                int status;
                {
                    std::unique_lock<std::recursive_mutex> guard(clock_);
                    // keep timeout short such that other calls can go through
                    status = gp_camera_wait_for_event(cam_,500,&ev,&ptr,ctx_);
                }
                check(status,"Wait event");
                switch(ev) {
                case GP_EVENT_UNKNOWN:
                    LOG("EV:Unknown\n");
                    break;
                case GP_EVENT_TIMEOUT:
                    LOG("EV:Timeout\n");
                    if(fno > 0) {
                        return fno;
                    }
                    break;
                case GP_EVENT_FILE_ADDED:
                    if(fno < MAX_FILES) { 
                        CameraFilePath *fp = static_cast<CameraFilePath*>(ptr);
                        files[fno++] = *fp;
                        LOG("Got file %s/%s\n",fp->folder,fp->name);
                    }
                    break;
                case GP_EVENT_CAPTURE_COMPLETE:
                    LOG("EV:Capture Done\n");
                    if (fno > 0) {
                        return fno;
                    }
                    break;
                default:
                    LOG("EV:Other %d\n",int(ev));
                    break;
                }
                free(ptr);
            }
            return fno;
        }

        struct ExposureDrainResult {
            std::vector<CameraFilePath> files;
            bool event_complete = false;
            bool completion_grace_complete = false;
            std::string termination = "not_started";
            std::string error;
        };

        /// Collect the observed files for one exposure, retaining discoveries
        /// even if event collection fails. A completion event followed by the
        /// full grace is evidence for the observed set, not a predicted count.
        ExposureDrainResult drain_exposure(double timeout_s)
        {
            using clock = std::chrono::steady_clock;
            ExposureDrainResult result;
            auto deadline = clock::now() +
                std::chrono::milliseconds(static_cast<long long>(timeout_s * 1000.0));
            const auto grace = std::chrono::milliseconds(2000);
            clock::time_point grace_until;
            clock::time_point last_file_at;

            while(!stop_) {
                auto now = clock::now();
                if(result.event_complete && now >= grace_until) {
                    result.completion_grace_complete = true;
                    result.termination = "capture_complete_grace_elapsed";
                    return result;
                }
                if(now >= deadline) {
                    result.termination = "deadline";
                    result.error = "Exposure event drain reached its deadline";
                    return result;
                }
                if(!result.event_complete && !result.files.empty() &&
                   now - last_file_at >= grace) {
                    result.termination = "quiet_without_completion";
                    result.error = "Files arrived without a capture-complete event";
                    return result;
                }

                CameraEventType ev = GP_EVENT_UNKNOWN;
                void *ptr = nullptr;
                int status;
                {
                    std::unique_lock<std::recursive_mutex> guard(clock_);
                    status = gp_camera_wait_for_event(cam_,300,&ev,&ptr,ctx_);
                }
                if(status < GP_OK) {
                    free(ptr);
                    result.termination = "event_error";
                    result.error = std::string("wait_for_event: ") + gp_result_as_string(status);
                    return result;
                }
                switch(ev) {
                case GP_EVENT_FILE_ADDED:
                    {
                        CameraFilePath *fp = static_cast<CameraFilePath*>(ptr);
                        if(fp) {
                            auto existing = std::find_if(result.files.begin(),result.files.end(),
                                [fp](CameraFilePath const &file) {
                                    return strcmp(file.folder,fp->folder) == 0 &&
                                           strcmp(file.name,fp->name) == 0;
                                });
                            if(existing == result.files.end())
                                result.files.push_back(*fp);
                            last_file_at = clock::now();
                            LOG("Exposure produced %s/%s\n",fp->folder,fp->name);
                        }
                    }
                    break;
                case GP_EVENT_CAPTURE_COMPLETE:
                    LOG("EV:Capture Done (%zu file(s) so far)\n",result.files.size());
                    if(!result.event_complete)
                        grace_until = clock::now() + grace;
                    result.event_complete = true;
                    break;
                default:
                    break;
                }
                free(ptr);
            }
            result.termination = "stopped";
            result.error = "Exposure event drain was stopped before completion";
            return result;
        }

        /// Ask the running stream thread to stand down, and wait until it
        /// confirms it is idle. Interleaving preview traffic with a still
        /// capture on the same PTP session is what provokes "camera busy".
        /// Returns false if the stream did not go idle in time.
        bool pause_stream(double timeout_s = 5.0)
        {
            if(stream_active_ == 0)
                return true;
            pause_requested_ = true;
            auto deadline = std::chrono::steady_clock::now() +
                std::chrono::milliseconds(static_cast<long long>(timeout_s * 1000.0));
            while(!stream_idle_) {
                if(std::chrono::steady_clock::now() > deadline) {
                    LOG("Timed out waiting for stream to idle\n");
                    return false;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            return true;
        }

        void resume_stream()
        {
            pause_requested_ = false;
        }

        /// RAII pause guard so an exception mid-capture cannot leave the
        /// stream thread parked forever.
        struct StreamPause {
            GP2Camera *cam;
            explicit StreamPause(GP2Camera *c) : cam(c) {
                if(!cam->pause_stream()) {
                    cam->resume_stream();
                    throw GPError("Preview did not become idle; refusing a competing still event consumer");
                }
            }
            ~StreamPause() { cam->resume_stream(); }
        };

        virtual std::vector<CamStreamFormat> formats(CamErrorCode &e)
        {
            if(!formats_.empty())
                return formats_;
            try {
                // Format discovery must not take a photograph. Enumerate only
                // nonactuating live view; dedicated capture_still() discovers
                // native asset geometry from the explicitly requested exposure.
                if(liveview_supported()) {
                    CamStreamFormat fmt;
                    fmt.format = stream_mjpeg;
                    fmt.width  = liveview_width_;
                    fmt.height = liveview_height_;
                    fmt.bin = 1;
                    // Request ceiling, not a measured/vendor FPS promise. Camera
                    // preview cadence and PTP transport determine actual FPS.
                    fmt.framerate = 60.0f;
                    formats_.push_back(fmt);
                    liveview_format_index_ = 0;
                }

            }
            catch(std::exception const &err)  {
                e = err.what();
            }
            return formats_;
        }

        /// Locate `f` in the enumerated list so the capture thread knows which
        /// acquisition path the caller selected. Matching on geometry rather
        /// than trusting a flag in the struct keeps this correct across the
        /// C FFI boundary, where the format is rebuilt field by field.
        int find_format_index(CamStreamFormat const &f) const
        {
            for(size_t i=0;i<formats_.size();i++) {
                CamStreamFormat const &c = formats_[i];
                if(c.format == f.format && c.width == f.width &&
                   c.height == f.height && c.bin == f.bin)
                    return int(i);
            }
            return -1;
        }

        std::pair<cv::Mat,CamBayerType> to_rgb(std::pair<cv::Mat,CamBayerType> const &raw)
        {
            cv::Mat rgb,result;
            cv::Mat bayer = raw.first;
            switch(raw.second) {
            case bayer_rg:  cv::cvtColor(bayer,rgb,cv::COLOR_BayerBG2BGR); break; // COLOR_BayerRGGB2BGR = COLOR_BayerBG2BGR
            case bayer_gr:  cv::cvtColor(bayer,rgb,cv::COLOR_BayerGB2BGR); break; // COLOR_BayerGRBG2BGR = COLOR_BayerGB2BGR
            case bayer_bg:  cv::cvtColor(bayer,rgb,cv::COLOR_BayerRG2BGR); break; // COLOR_BayerBGGR2BGR = COLOR_BayerRG2BGR
            case bayer_gb:  cv::cvtColor(bayer,rgb,cv::COLOR_BayerGR2BGR); break; // COLOR_BayerGBRG2BGR = COLOR_BayerGR2BGR
            default:
                throw std::runtime_error("Invalid bayer patter");
            }
            if(format_.bin == 1)
                return std::make_pair(rgb,bayer_na);
            cv::resize(rgb,result,cv::Size(format_.width,format_.height),0.0,0.0,cv::INTER_AREA);
            return std::make_pair(result,bayer_na);
        }

        /// Deliver one frame to the consumer.
        ///
        /// `src`, when given, is the camera-authored file the pixels came from
        /// (CR2/CR3/RW2/JPEG). Its bytes ride along on the frame untouched so
        /// the archive can keep the original alongside the decode. Nothing is
        /// copied here — the pointer is valid for exactly as long as the
        /// callback runs, which is the same contract `data` already has.
        void handle_frame(std::pair<cv::Mat,CamBayerType> const &in,GPFile *src = nullptr)
        {
            struct timeval tv;
            gettimeofday(&tv,nullptr);
            CamFrame frm;
            frm.unix_timestamp = tv.tv_sec;
            frm.unix_timestamp += tv.tv_usec * 1e-6;
            frm.width = format_.width;
            frm.height = format_.height;
            frm.data = in.first.data;
            frm.format = format_.format;
            frm.data_size = in.first.total() * in.first.elemSize();
            frm.frame_counter = frame_counter_++;
            frm.bayer = in.second;

            std::string src_ext; // must outlive the callback below
            if(src && src->loaded()) {
                src_ext = src->ext();
                frm.source_data = src->data();
                frm.source_data_size = src->size();
                frm.source_ext = src_ext.empty() ? nullptr : src_ext.c_str();
            }
            {
                std::unique_lock<std::mutex> guard(lock_);
                if(callback_) {
                    callback_(frm);
                }
            }
        }

        /// Deliver one live-view frame. The JPEG is passed through as both the
        /// pixel payload and the original, since for live view they are the
        /// same bytes and nothing has been decoded.
        void handle_preview_frame(std::vector<unsigned char> const &jpeg)
        {
            struct timeval tv;
            gettimeofday(&tv,nullptr);
            CamFrame frm;
            frm.unix_timestamp = tv.tv_sec;
            frm.unix_timestamp += tv.tv_usec * 1e-6;
            frm.width = format_.width;
            frm.height = format_.height;
            frm.format = stream_mjpeg;
            frm.bayer = bayer_na;
            frm.data = jpeg.data();
            frm.data_size = jpeg.size();
            frm.frame_counter = frame_counter_++;
            // Live view is a throwaway preview stream, not a science frame.
            // Marking it with no original keeps the archive from filling up
            // with tens of thousands of framing JPEGs.
            frm.source_data = nullptr;
            frm.source_data_size = 0;
            frm.source_ext = nullptr;
            {
                std::unique_lock<std::mutex> guard(lock_);
                if(callback_) {
                    callback_(frm);
                }
            }
        }
        void handle_error(const char *msg)
        {
            CamFrame frm = CamFrame();
            frm.format = stream_error;
            frm.data = msg;
            frm.data_size = strlen(msg);
            std::unique_lock<std::mutex> guard(lock_);
            if (callback_)
            {
                callback_(frm);
            }
        }

        /// Start a video stream with provided callback 
        virtual void start_stream(CamStreamFormat format,frame_callback_type callback,CamErrorCode &e) 
        {
            // Refuse still formats before enumeration: enumeration can perform
            // a probe on bodies without cached dimensions (A18).
            if(!std::isfinite(format.framerate) || format.framerate <= 0) {
                e = "DSLR still streams cannot archive complete bundles; use dedicated still capture (A02)";
                return;
            }
            if(stream_active_ != 0) {
                stop_stream(e);
                if(e)
                    return;
            }
            format_ = format;

            // Decide once, here, whether this stream is live view or repeated
            // stills. Everything downstream branches on this flag.
            if(formats_.empty()) {
                CamErrorCode ignored;
                formats(ignored);
            }
            int idx = find_format_index(format);
            streaming_liveview_ = (idx >= 0 && idx == liveview_format_index_);
            if(!streaming_liveview_) {
                e = "DSLR still streams cannot archive complete bundles; use dedicated still capture (A02)";
                return;
            }
            LOG("start_stream: format index %d, mode=%s\n",
                idx,streaming_liveview_ ? "liveview" : "still");

            {
                std::unique_lock<std::mutex> guard(lock_);
                callback_ = callback;
            }
            stream_active_ = 1;
            stop_=false;
            thread_ = std::thread([=]() {
                capture_thread();
            });
        }
        void capture_thread()
        {
            try {
                if(streaming_liveview_)
                    liveview_thread();
                else
                    still_capture_thread();
            }
            catch(std::exception const &e) {
                handle_error(e.what());
            }
        }

        /// Live-view acquisition loop.
        ///
        /// Pulls frames with gp_camera_capture_preview(), which reads the
        /// sensor's video feed over PTP. No shutter moves, no file is written
        /// to the card, and nothing is deleted afterwards — so this loop can
        /// run for hours during framing and focusing without consuming any of
        /// the body's finite shutter life.
        void liveview_thread()
        {
            // Some bodies (Canon EOS in particular) need the viewfinder flag
            // raised before the preview feed starts. Best-effort: several
            // models start it implicitly and reject the explicit set.
            set_viewfinder(1);

            // Pace the loop to the advertised framerate. Without this the loop
            // saturates the USB bus and starves the config traffic that
            // set_parameter() needs to get through.
            double target_fps = format_.framerate > 0 ? format_.framerate : 60.0;
            auto frame_interval = std::chrono::microseconds(
                static_cast<long long>(1e6 / std::max(1.0,target_fps)));

            std::vector<unsigned char> jpeg;
            std::pair<int,int> dims;
            int consecutive_failures = 0;
            const char *timing_env = std::getenv("ACHERNAR_GPHOTO_PREVIEW_TIMING");
            const bool timing_enabled = timing_env && std::strcmp(timing_env,"1") == 0;
            PreviewTimingWindow timing_window;
            auto report_timing = [&](bool force) {
                if(!timing_enabled || timing_window.attempts == 0) return;
                auto now = std::chrono::steady_clock::now();
                if(!force && now-timing_window.started < std::chrono::seconds(5)) return;
                timing_window.write(stderr,frame_counter_,format_.width,format_.height,
                                    target_fps,now);
                timing_window = PreviewTimingWindow();
            };

            while(!stop_) {
                // Stand down while a still capture owns the camera.
                if(pause_requested_) {
                    stream_idle_ = true;
                    std::this_thread::sleep_for(std::chrono::milliseconds(20));
                    continue;
                }
                stream_idle_ = false;

                auto started = std::chrono::steady_clock::now();
                if(stop_)
                    break;

                PreviewPullTiming pull_timing;
                bool ok;
                try {
                    ok = grab_preview(jpeg,dims,timing_enabled ? &pull_timing : nullptr);
                }
                catch(...) {
                    if(timing_enabled) {
                        timing_window.add(pull_timing,false,0,0,0);
                        report_timing(true);
                    }
                    throw;
                }

                if(!ok) {
                    if(timing_enabled) timing_window.add(pull_timing,false,0,0,0);
                    report_timing(false);
                    // Tolerate transient hiccups — a body that is busy writing
                    // to its card will refuse previews for a moment — but do
                    // not spin forever on a camera that has gone away.
                    if(++consecutive_failures >= 10) {
                        handle_error("Live view stopped responding");
                        break;
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(200));
                    continue;
                }
                consecutive_failures = 0;
                format_.width = dims.first;
                format_.height = dims.second;
                auto callback_started = timing_enabled ? std::chrono::steady_clock::now()
                                                       : std::chrono::steady_clock::time_point();
                handle_preview_frame(jpeg);
                auto callback_completed = std::chrono::steady_clock::now();
                double callback_ms = timing_enabled
                    ? preview_milliseconds(callback_completed-callback_started) : 0;

                auto elapsed = callback_completed - started;
                double pacing_ms = 0;
                if(elapsed < frame_interval) {
                    std::this_thread::sleep_for(frame_interval - elapsed);
                    if(timing_enabled) pacing_ms = preview_milliseconds(
                        std::chrono::steady_clock::now()-callback_completed);
                }
                if(timing_enabled)
                    timing_window.add(pull_timing,true,jpeg.size(),callback_ms,pacing_ms);
                report_timing(false);
            }

            report_timing(true);
            set_viewfinder(0);
        }

        /// Full-resolution still acquisition loop — one shutter actuation per
        /// frame. Correct for science frames, ruinous for framing, which is
        /// why live view above is the default stream.
        ///
        /// Note this is the *preview-of-stills* path. A real science exposure
        /// goes through capture_still(), which owns its settings and files
        /// end to end. This loop exists so a body without live view can still
        /// show the operator something.
        void still_capture_thread()
        {
            trigger();
            while(!stop_) {
                if(pause_requested_) {
                    stream_idle_ = true;
                    std::this_thread::sleep_for(std::chrono::milliseconds(20));
                    continue;
                }
                stream_idle_ = false;

                // Drain the WHOLE exposure before doing anything else so a
                // RAW+JPEG companion cannot be mistaken for the next exposure.
                // Every file remains on the camera regardless of which one
                // this preview consumer selects.
                ExposureDrainResult drained = drain_exposure(120.0);
                std::vector<CameraFilePath> files = std::move(drained.files);
                if(!drained.error.empty()) {
                    handle_error(drained.error.c_str());
                    break;
                }
                if(stop_)
                    break;

                bool delay_enabled = capture_delay_ > 0;
                auto time_before_downloading_files = std::chrono::steady_clock::now();
                if (!delay_enabled && !pause_requested_) {
                    trigger();
                }

                for(size_t i=0;i<files.size();i++) {
                    std::unique_lock<std::recursive_mutex> guard(clock_);
                    GPFile f(files[i],cam_,ctx_);

                    // A preview callback is not a durable archive receipt.
                    // Decode the selected format while retaining every file,
                    // including ignored companions and decode failures.
                    bool selected =
                        (format_.format == stream_mjpeg) ? f.is_jpeg() : !f.is_jpeg();

                    if(selected) {
                        try {
                            if(format_.format == stream_mjpeg) {
                                f.get();
                                // pass data as is
                                cv::Mat buffer(1,f.size(),CV_8UC1,const_cast<char*>(f.data()));
                                handle_frame(std::make_pair(buffer,bayer_na),&f);
                            }
                            else if(format_.format == stream_raw16) {
                                handle_frame(f.to_mat(format_.width,format_.height,format_.bin),&f);
                            }
                            else if(format_.format == stream_rgb48) {
                                handle_frame(to_rgb(f.to_mat(format_.width * format_.bin, format_.height * format_.bin)),&f);
                            }
                        }
                        catch(std::exception const &err) {
                            // Report and keep the file on the card so the
                            // frame is recoverable by hand.
                            LOG("Failed to handle %s: %s\n",f.name().c_str(),err.what());
                            handle_error(err.what());
                        }
                    }

                }

                if (delay_enabled) {
                    auto time_elapsed = std::chrono::steady_clock::now() - time_before_downloading_files;
                    auto time_remaining = std::chrono::seconds(capture_delay_) - time_elapsed;
                    auto half_second = std::chrono::milliseconds(500);
                    while (time_remaining.count() > 0) {
                        auto time_to_sleep = time_remaining < half_second ? time_remaining : half_second;
                        std::this_thread::sleep_for(time_to_sleep);
                        if (stop_) {
                            return;
                        }
                        time_remaining -= time_to_sleep;
                    }
                    if(!pause_requested_)
                        trigger();
                }
            }
        }

        /// Best-effort viewfinder toggle. Absent on bodies that manage live
        /// view implicitly, so a failure here is logged, never propagated.
        void set_viewfinder(int on)
        {
            try {
                std::unique_lock<std::recursive_mutex> guard(clock_);
                GPWidget w(cam_,ctx_,"viewfinder");
                if(w.try_load())
                    w.set(on);
            }
            catch(std::exception const &e) {
                LOG("viewfinder %d not settable: %s\n",on,e.what());
            }
        }

        /// stop the stream - once function ends callback will not be called any more
        virtual void stop_stream(CamErrorCode &)
        {
            if(stream_active_ == 0)
                return;
            {
                std::unique_lock<std::mutex> guard(lock_);
                callback_ = nullptr;
                stop_ = true;
            }

            /// stop capture there
            stream_active_=0;
            thread_.join();
            stop_ = false;
        }

        // ═══════════════════════════════════════════════════════════════════
        //  Single-shot still capture
        // ═══════════════════════════════════════════════════════════════════

        virtual bool supports_still_capture() const override { return true; }

        static double now_unix()
        {
            struct timeval tv;
            gettimeofday(&tv,nullptr);
            return tv.tv_sec + tv.tv_usec * 1e-6;
        }

        /// Read a selection widget's choice labels and its current value.
        std::vector<std::string> choice_labels(std::string const &key,std::string &current)
        {
            std::vector<std::string> labels;
            current.clear();
            std::unique_lock<std::recursive_mutex> guard(clock_);
            GPWidget w(cam_,ctx_,key);
            if(!w.try_load())
                return labels;
            CamParam p = w.get(opt_iso); // option id is irrelevant to the label read
            labels = p.names;
            int idx = int(p.cur_val);
            if(idx >= 0 && size_t(idx) < labels.size())
                current = labels[size_t(idx)];
            return labels;
        }

        /// Apply one enumerated setting and read back what the camera ended up
        /// at. Returns the applied label; `out_value` is its parsed physical
        /// value, or <= 0 when the label is not numeric.
        ///
        /// The read-back is the whole point. Writing a config over PTP can be
        /// silently clamped, refused in certain exposure modes, or overridden
        /// by the body's own auto logic, so the requested value is not
        /// evidence of anything.
        std::string apply_enumerated(std::string const &key,
                                     double (*parse)(std::string const &),
                                     double target,
                                     double &out_value)
        {
            out_value = -1;
            std::string current;
            std::vector<std::string> labels = choice_labels(key,current);
            if(labels.empty())
                return std::string();

            if(target > 0) {
                double want_value = -1;
                std::string want_label;
                int idx = nearest_choice(labels,parse,target,want_value,want_label);
                if(idx >= 0 && want_label != current) {
                    std::unique_lock<std::recursive_mutex> guard(clock_);
                    GPWidget w(cam_,ctx_,key);
                    w.load();
                    w.set(double(idx));
                }
            }

            // Re-read from the body rather than trusting the write.
            std::string after;
            choice_labels(key,after);
            if(after.empty())
                after = current;
            out_value = parse(after);
            return after;
        }

        virtual StillCaptureResult capture_still(StillCaptureRequest const &request,
                                                 CamErrorCode &e) override
        {
            StillCaptureResult result;
            // One capture at a time per camera, regardless of how many callers
            // race for it.
            std::unique_lock<std::mutex> still_guard(still_lock_);
            try {
                // 1) Stand the preview down. Its destructor resumes it even if
                //    anything below throws.
                StreamPause pause(this);

                // 2) Apply and read back shutter and ISO synchronously, just
                //    like manual control writes. A request is not evidence of
                //    a setting until the camera has accepted it and read back.
                double applied_exposure = -1;
                double applied_iso = -1;
                result.shutter_label =
                    apply_enumerated("shutterspeed",parse_shutter_seconds,
                                     request.exposure_seconds,applied_exposure);
                result.iso_label =
                    apply_enumerated("iso",parse_iso,request.sensitivity,applied_iso);
                result.applied_exposure =
                    applied_exposure > 0 ? applied_exposure : request.exposure_seconds;
                result.applied_sensitivity = applied_iso;

                LOG("capture_still: requested %.4fs/ISO %.0f -> \"%s\"/\"%s\"\n",
                    request.exposure_seconds,request.sensitivity,
                    result.shutter_label.c_str(),result.iso_label.c_str());

                // 3) Timestamp immediately before the trigger. This is the
                //    value that belongs in FITS DATE-OBS; a clock read after
                //    the download would be off by the exposure plus however
                //    long a 30 MB CR3 takes to come over USB.
                result.started_at = now_unix();

                {
                    std::unique_lock<std::recursive_mutex> guard(clock_);
                    int status = gp_camera_trigger_capture(cam_,ctx_);
                    check(status,"trigger capture");
                    result.triggered = true;
                }

                // 4) Collect every file this exposure produced. Budget for the
                //    exposure itself plus download of multiple large files.
                double budget = result.applied_exposure * 2.0 + 60.0;
                ExposureDrainResult drained = drain_exposure(budget);
                result.event_complete = drained.event_complete;
                result.completion_grace_complete = drained.completion_grace_complete;
                result.drain_termination = drained.termination;
                result.completed_at = now_unix();
                if(!drained.error.empty())
                    result.errors.push_back(drained.error);

                // Attempt every discovered file independently. A missing or
                // broken companion must not discard already downloaded bytes.
                for(auto const &path : drained.files) {
                    std::string full = path.folder + std::string("/") + path.name;
                    result.discovered_paths.push_back(full);
                    CaptureDownloadEvidence evidence;
                    evidence.camera_path = full;
                    try {
                        std::unique_lock<std::recursive_mutex> guard(clock_);
                        GPFile file(path,cam_,ctx_);
                        evidence.error = file.download_preserving_bytes();
                        evidence.byte_length = file.data() ? file.size() : 0;
                        evidence.download_complete = evidence.error.empty() &&
                                                     evidence.byte_length > 0;
                        if(evidence.byte_length > 0) {
                            CapturedAsset asset;
                            asset.camera_path = full;
                            asset.extension = file.ext();
                            asset.download_complete = evidence.download_complete;
                            unsigned char const *begin =
                                reinterpret_cast<unsigned char const*>(file.data());
                            asset.bytes.assign(begin,begin + file.size());
                            result.assets.push_back(std::move(asset));
                        }
                    }
                    catch(std::exception const &error) {
                        evidence.download_complete = false;
                        evidence.error = error.what();
                    }
                    if(!evidence.error.empty())
                        result.errors.push_back(full + ": " + evidence.error);
                    result.downloads.push_back(std::move(evidence));
                }
                if(result.discovered_paths.empty())
                    result.errors.push_back("The exposure produced no discovered native file");
                result.bundle_complete = result.triggered && result.event_complete &&
                    result.completion_grace_complete && result.errors.empty() &&
                    !result.discovered_paths.empty() &&
                    result.assets.size() == result.discovered_paths.size() &&
                    std::all_of(result.assets.begin(),result.assets.end(),
                        [](CapturedAsset const &asset) {
                            return asset.download_complete && !asset.bytes.empty();
                        });
                if(!result.bundle_complete) {
                    if(result.errors.empty())
                        result.errors.push_back("Native exposure bundle completeness is unconfirmed");
                    e = result.errors.front();
                }

                // Retain every original on the camera. A local commit alone
                // cannot authorize deleting the retained second copy; deletion
                // stays disabled pending verified persistence and recovery (A12).
                return result;
            }
            catch(std::exception const &err) {
                e = err.what();
                result.errors.push_back(err.what());
                result.bundle_complete = false;
                if(result.triggered)
                    result.completed_at = now_unix();
                return result;
            }
        }

        virtual void delete_captured(std::vector<CapturedAsset> const &assets,
                                     CamErrorCode &e) override
        {
            (void)assets;
            // Also reject direct native callers; no local commit or boolean
            // setting substitutes for the future verified deletion receipt.
            e = "Camera deletion is disabled until verified capture persistence "
                "and recovery are qualified (A12)";
        }

        /// list of camera controls that the camera supports
        virtual std::vector<CamOptionId> supported_options(CamErrorCode &)
        {
            std::unique_lock<std::recursive_mutex> guard(clock_);
            std::vector<CamOptionId> opts;
            {
                GPWidget w(cam_,ctx_,"iso");
                if(w.try_load())
                    opts.push_back(opt_iso);
            }
            {
                GPWidget w(cam_,ctx_,"shutterspeed");
                if(w.try_load())
                    opts.push_back(opt_shutter);
            }
            {
                GPWidget w(cam_,ctx_,"viewfinder");
                if(w.try_load())
                    opts.push_back(opt_viewfinder);
            }
            {
                GPWidget w(cam_,ctx_,"capturetarget");
                if(w.try_load())
                    opts.push_back(opt_capturetarget);
            }
            opts.push_back(opt_keep_images);
            // Legacy capture_delay access remains available, but it only
            // paced the retired still-stream loop. Dedicated captures and
            // live view do not use it, so do not advertise an inert control.
            return opts;
        }
        std::string opt_to_name(CamOptionId id)
        {
            std::string name;
            switch(id) {
            case opt_iso:
                name = "iso";
                break;
            case opt_shutter:
                name = "shutterspeed";
                break;
            case opt_viewfinder:
                name = "viewfinder";
                break;
            case opt_capturetarget:
                name = "capturetarget";
                break;
            default:
                throw GPError("Unsupported camera option");
            }
            return name;
        }
        /// override to prevent loosing mutex
        virtual std::vector<CamParam> get_all_parameters(bool current_only,CamErrorCode &e)
        {
            std::unique_lock<std::recursive_mutex> guard(clock_);
            return Camera::get_all_parameters(current_only,e);
        }
        /// get camera control
        virtual CamParam get_parameter(CamOptionId id,bool /*current_only*/,CamErrorCode &e) 
        {
            try {
                switch(id) {
                case opt_keep_images:
                    {
                        CamParam r;
                        r.option = id;
                        r.type = type_bool;
                        r.read_only = true;
                        r.min_val = 0;
                        r.max_val = 1;
                        r.step_size = 1;
                        r.cur_val = keep_images_;
                        r.def_val = 1;
                        return r;
                    }
                    break;
                case opt_capture_delay:
                    {
                        CamParam r;
                        r.option = id;
                        r.type = type_number;
                        r.min_val = 0;
                        r.max_val = 30;
                        r.step_size = 1;
                        r.cur_val = capture_delay_;
                        r.def_val = 0;
                        return r;
                    }
                default:
                    {
                        std::string name = opt_to_name(id);
                        std::unique_lock<std::recursive_mutex> guard(clock_);
                        GPWidget w(cam_,ctx_,name);
                        return w.get(id);
                    }
                }
            }
            catch(std::exception const &er) {
                e = er.what();
            }
            return CamParam();
        }
        /// set camera control
        virtual void set_parameter(CamOptionId id,double value,CamErrorCode &e)
        {
            // Preserve the retention policy even for native callers that do
            // not inspect the read-only descriptor first.
            if(id == opt_keep_images && (!std::isfinite(value) || value != 1.0)) {
                e = "Keep images must remain enabled; camera deletion is disabled "
                    "until verified capture persistence and recovery are qualified (A12)";
                return;
            }
            // Every PTP preview/config/capture call uses this same mutex.
            // Wait for any in-flight preview, then complete the write here so
            // callers can read back the applied setting and receive SDK errors.
            // Queueing used to acknowledge the request before the camera saw
            // it, making immediate readback return the previous ISO/shutter.
            std::unique_lock<std::recursive_mutex> guard(clock_);
            set_parameter_internal(id,value,e);
        }
        void set_parameter_internal(CamOptionId id,double value,CamErrorCode &e)
        {
            try {
                switch(id) {
                case opt_keep_images:
                    {
                        // Enforce retention here too, including commands from
                        // any future internal caller that bypasses the queue.
                        if(!std::isfinite(value) || value != 1.0) {
                            e = "Keep images must remain enabled; camera deletion is disabled "
                                "until verified capture persistence and recovery are qualified (A12)";
                            return;
                        }
                        keep_images_ = 1;
                    }
                    break;
                case opt_capture_delay:
                    {
                        capture_delay_ = value;
                    }
                    break;
                default:
                    {
                        std::string name = opt_to_name(id);
                        GPWidget w(cam_,ctx_,name);
                        w.set(value);
                    }
                }
            }
            catch(std::exception const &er) {
                e = er.what();
            }
        }
    private:
        ::Camera *cam_;
        GPContext *ctx_;
        std::atomic<int> stream_active_;
        std::vector<CamStreamFormat> formats_;
        CamStreamFormat format_;
        int frame_counter_ = 0;
        // Preserve originals during probes and still streaming. Legacy false
        // values are not deletion consent; enabling deletion requires the
        // separately acknowledged, verified persistence policy in the daemon.
        int keep_images_ = 1;
        int capture_delay_ = 0;

        // ─── Live view state ───
        // Probed lazily on the first formats() call and cached: asking a body
        // for a preview it does not have is slow, so we ask exactly once.
        bool liveview_probe_done_ = false;
        bool liveview_available_ = false;
        int liveview_width_ = 0;
        int liveview_height_ = 0;
        /// Index of the live-view entry in formats_, or -1 when unavailable.
        int liveview_format_index_ = -1;
        /// Which acquisition path the active stream selected.
        bool streaming_liveview_ = false;

        // ─── Still-capture arbitration ───
        /// Raised by capture_still() to stand the stream thread down; the
        /// stream sets stream_idle_ once it has stopped touching the camera.
        std::atomic<bool> pause_requested_{false};
        std::atomic<bool> stream_idle_{true};
        /// Serialises capture_still() against itself.
        std::mutex still_lock_;

        // protected by mutex
        std::mutex lock_;
        std::recursive_mutex clock_;
        frame_callback_type callback_;
        std::thread thread_;
        /// Written by stop_stream() on the caller's thread, polled by the
        /// capture thread — must be atomic, not a plain bool.
        std::atomic<bool> stop_{false};

    };
 

    class GP2CameraDriver : public CameraDriver {
    public:
        virtual std::vector<std::string> list_cameras(CamErrorCode &e)
        {
            if(!camera_list_.empty()) {
                return camera_list_;
            }
            gp_camera_new(&camera_);
            int ret = gp_camera_init(camera_,ctx_);
            if(!check(ret,"init",e)) {
                gp_camera_free(camera_);
                camera_ = nullptr;
                return camera_list_;
            }
            // Report the detected model rather than the placeholder "Camera",
            // so device selection, logs and saved per-camera profiles refer to
            // an actual body ("Canon EOS R50", "Canon EOS 760D",
            // "Panasonic DC-G95", …).
            CameraAbilities abilities;
            memset(&abilities,0,sizeof(abilities));
            if(gp_camera_get_abilities(camera_,&abilities) == GP_OK && abilities.model[0]) {
                LOG("Detected camera: %s\n",abilities.model);
                camera_list_.push_back(abilities.model);
            }
            else {
                camera_list_.push_back("Camera");
            }
            return camera_list_;
        }
        virtual std::unique_ptr<Camera> open_camera(int id,CamErrorCode &e) 
        {
            if(size_t(id) >= camera_list_.size()) {
                e = "No such camera " + std::to_string(id);
                return std::unique_ptr<Camera>();
            }
            std::unique_ptr<Camera> cam(new GP2Camera(camera_,ctx_));
            return cam;
        }
        GP2CameraDriver() 
        {
            ctx_ = gp_context_new();
            gp_context_set_error_func(ctx_,ctx_error_func,nullptr);
            gp_context_set_status_func(ctx_,ctx_status_func,nullptr);
            LOG("context created\n");
        }
        ~GP2CameraDriver()
        {
            if(camera_)
                gp_camera_exit(camera_,ctx_);
        }
    private:
        GPContext *ctx_ = nullptr;
        ::Camera *camera_ = nullptr;
        std::vector<std::string> camera_list_;        
    };
}

extern "C" {

    void ols_set_gphoto2_driver_log(char const *log_path,int debug)
    {
        ols::error_stream = fopen(log_path,"w");
        gp_log_add_func(debug ? GP_LOG_DEBUG : GP_LOG_ERROR , ols::errordumper, NULL);
    }

#ifdef ANDROID_SUPPORT
    int ols_set_gphoto2_driver_config(char const *libdir)
    {
        setenv("CAMLIBS",libdir,1);
        setenv("IOLIBS", libdir,1);
        setenv("CAMLIBS_PREFIX","libgphoto2_camlib_",1);
        setenv("IOLIBS_PREFIX","libgphoto2_port_iolib_",1);
        LOG("Gphoto libdir %s\n",libdir);
        return 0;
    }
   
    ols::CameraDriver *ols_get_gphoto2_driver(int fd,ols::CamErrorCode *e)
    {
        if(fd >= 0) {
            LOG("Using file descriptor for USB device %d\n",fd);
            int status = gp_port_usb_set_sys_device(fd);
            if(!ols::check(status,"gp_port_usb_set_sys_device",*e))
                return nullptr;
        }
        return new ols::GP2CameraDriver();
    }
#else    
    ols::CameraDriver *ols_get_gphoto2_driver(int /*cam_id*/,ols::CamErrorCode * /*e*/)
    {
        return new ols::GP2CameraDriver();
    }
#endif    
}
