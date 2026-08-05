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
#include <queue>
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
                    char const *val = nullptr;
                    for(int i=0;i<N;i++) {
                        res = gp_widget_get_choice(widget_,i,&val);
                        check(res,"widget_get_choice");
                        r.names.push_back(val);
                    }
                    gp_widget_get_value(widget_,&val);
                    check(res,"widget_get_value");
                    
                    r.type = type_selection;
                    r.min_val = 0;
                    r.max_val = N-1;
                    r.cur_val = std::find(r.names.begin(),r.names.end(),std::string(val)) - r.names.begin();
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
        void del()
        {
            LOG("Deleting file %s/%s\n",path_.folder,path_.name);
            int status = gp_camera_file_delete(cam_,path_.folder,path_.name,ctx_);
            check(status,"delete");
        }
        bool loaded()
        {
            return data_ != nullptr;
        }
        void get()
        {
            int status = 0;
            free();
            status = gp_file_new(&file_);
            check(status,"new file");
            status = gp_camera_file_get(cam_,path_.folder,path_.name,GP_FILE_TYPE_NORMAL,file_,ctx_);
            check(status,"get file");
            status = gp_file_get_data_and_size(file_,&data_,&size_);
            check(status,"get data");
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
        bool grab_preview(std::vector<unsigned char> &out,std::pair<int,int> &dims)
        {
            CameraFile *file = nullptr;
            int status = gp_file_new(&file);
            check(status,"new preview file");
            struct Guard {
                CameraFile *f;
                ~Guard() { if(f) gp_file_free(f); }
            } guard_file{file};

            {
                std::unique_lock<std::recursive_mutex> guard(clock_);
                status = gp_camera_capture_preview(cam_,file,ctx_);
            }
            if(status == GP_ERROR_NOT_SUPPORTED)
                return false;
            if(status < GP_OK) {
                LOG("capture_preview failed: %s\n",gp_result_as_string(status));
                return false;
            }

            char const *data = nullptr;
            unsigned long size = 0;
            status = gp_file_get_data_and_size(file,&data,&size);
            check(status,"preview data");
            if(!data || size == 0)
                return false;

            out.assign(reinterpret_cast<unsigned char const*>(data),
                       reinterpret_cast<unsigned char const*>(data) + size);

            // Decode once, during the probe only, to learn the exact live-view
            // geometry. At stream time the payload is forwarded untouched as
            // MJPEG and never decoded here.
            cv::Mat buffer(1,int(out.size()),CV_8UC1,out.data());
            cv::Mat img = cv::imdecode(buffer,cv::IMREAD_COLOR);
            if(img.empty())
                return false;
            dims = std::make_pair(img.cols,img.rows);
            return true;
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

        /// Collect every file produced by ONE exposure.
        ///
        /// `wait_event()` above returns as soon as the first FILE_ADDED
        /// arrives. On a body set to RAW+JPEG that is only half the exposure:
        /// the companion is still in flight, and the caller — which triggers
        /// the next exposure immediately — sees it attributed to the *next*
        /// shot. The companion is never selected by the format filter, so it
        /// is never deleted either, and the card fills with orphans while the
        /// event queue runs a shot behind.
        ///
        /// This drains to GP_EVENT_CAPTURE_COMPLETE and then keeps listening
        /// through a short grace window, because several bodies emit the
        /// second FILE_ADDED *after* the completion event.
        std::vector<CameraFilePath> drain_exposure(double timeout_s,bool &saw_complete)
        {
            using clock = std::chrono::steady_clock;
            std::vector<CameraFilePath> files;
            saw_complete = false;

            auto deadline = clock::now() +
                std::chrono::milliseconds(static_cast<long long>(timeout_s * 1000.0));
            const auto grace = std::chrono::milliseconds(2000);
            bool grace_started = false;
            clock::time_point grace_until;
            int consecutive_timeouts = 0;

            while(!stop_) {
                auto now = clock::now();
                if(now > deadline)
                    break;
                if(grace_started && now > grace_until)
                    break;

                CameraEventType ev = GP_EVENT_UNKNOWN;
                void *ptr = nullptr;
                int status;
                {
                    std::unique_lock<std::recursive_mutex> guard(clock_);
                    status = gp_camera_wait_for_event(cam_,300,&ev,&ptr,ctx_);
                }
                if(status < GP_OK) {
                    free(ptr);
                    throw GPError("wait_for_event",status);
                }

                switch(ev) {
                case GP_EVENT_FILE_ADDED:
                    {
                        CameraFilePath *fp = static_cast<CameraFilePath*>(ptr);
                        if(fp) {
                            files.push_back(*fp);
                            LOG("Exposure produced %s/%s\n",fp->folder,fp->name);
                        }
                        consecutive_timeouts = 0;
                    }
                    break;
                case GP_EVENT_CAPTURE_COMPLETE:
                    LOG("EV:Capture Done (%zu file(s) so far)\n",files.size());
                    saw_complete = true;
                    if(!grace_started) {
                        grace_started = true;
                        grace_until = clock::now() + grace;
                    }
                    consecutive_timeouts = 0;
                    break;
                case GP_EVENT_TIMEOUT:
                    // Not every body emits CAPTURE_COMPLETE. Once files have
                    // arrived and the bus has gone quiet twice, the exposure
                    // is over.
                    consecutive_timeouts++;
                    if(!files.empty() && consecutive_timeouts >= 2) {
                        free(ptr);
                        return files;
                    }
                    break;
                default:
                    break;
                }
                free(ptr);
            }
            return files;
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
            explicit StreamPause(GP2Camera *c) : cam(c) { cam->pause_stream(); }
            ~StreamPause() { cam->resume_stream(); }
        };

        virtual std::vector<CamStreamFormat> formats(CamErrorCode &e)
        {
            if(!formats_.empty())
                return formats_;
            try {
                // ─── 1. Live view, first in the list ───
                //
                // Index 0 is what every caller that does not specify a format
                // ends up streaming, so it must be the mode that costs nothing.
                // Live view is a mirror-up/EVF video feed: no shutter
                // actuation, no card write, no download of a 30 MB file per
                // frame. Framing, focusing and polar alignment all belong here.
                //
                // The full-resolution still formats below remain available at
                // higher indices and are the correct choice for science frames
                // — the point is that you now opt *into* wearing out the
                // shutter rather than doing it by default.
                if(liveview_supported()) {
                    CamStreamFormat fmt;
                    fmt.format = stream_mjpeg;
                    fmt.width  = liveview_width_;
                    fmt.height = liveview_height_;
                    fmt.bin = 1;
                    fmt.framerate = 15.0f;
                    formats_.push_back(fmt);
                    liveview_format_index_ = 0;
                }

                // ─── 2. Still formats ───
                //
                // Determining the sensor geometry requires exactly one probe
                // exposure; libgphoto2 offers no reliable cross-vendor way to
                // read it from the config tree. This costs one actuation per
                // session, not one per frame.
                trigger();
                CameraFilePath files[MAX_FILES];
                int N = wait_event(files,true);
                if(N == 0) {
                    if(!formats_.empty()) {
                        // Live view works even though the still probe did not.
                        // Usable for framing; do not fail the whole open.
                        LOG("Still probe produced no file; live view only\n");
                        return formats_;
                    }
                    e="Failed to capture images";
                    return formats_;
                }
                // A body set to RAW+JPEG emits two files per actuation. Both
                // are enumerated: the JPEG contributes an MJPEG still format,
                // the RAW contributes the raw16/rgb48 ladder, and the capture
                // loop already picks whichever file matches the selected
                // format. Rejecting this combination (as this driver used to)
                // forces the operator to reconfigure the camera for no reason
                // — RAW+JPEG is the shipping default on the R50 and M50.
                for(int i=0;i<N;i++) {
                    std::unique_lock<std::recursive_mutex> guard(clock_);
                    GPFile file(files[i],cam_,ctx_);
                    file.get();
                    if(!keep_images_)
                        file.del();
                    if(file.is_jpeg()) {
                        CamStreamFormat fmt;
                        fmt.format = stream_mjpeg;
                        std::pair<cv::Mat,CamBayerType> res = file.to_mat();
                        fmt.width = res.first.cols;
                        fmt.height = res.first.rows;
                        formats_.push_back(fmt);
                    }
                    else {
                        #ifdef WITH_LIBRAW
                        std::pair<cv::Mat,CamBayerType> res = file.to_mat();
                        for(int is_rgb=0;is_rgb <=1;is_rgb++) {
                            for(int bin=1;bin<=3;bin++) {
                                for(int scale = 0;scale < 8;scale++) {
                                    int num,den;
                                    switch(scale) {
                                    case  0: num=1; den=1; break;
                                    case  1: num=2; den=3; break;
                                    default: num=1; den=scale;
                                    }
                                    int w = num * res.first.cols / den / (2*bin) * 2; // make sure round of 2
                                    int h = num * res.first.rows / den / (2*bin) * 2; // make sure round of 2

                                    if(std::min(w,h) < 480)
                                        break;
                                    
                                    CamStreamFormat fmt;
                                    fmt.format = is_rgb ? stream_rgb48 : stream_raw16;
                                    fmt.width = w;
                                    fmt.height = h;
                                    fmt.bin = bin;
                                    fmt.roi_num = num;
                                    fmt.roi_den = den;
                                    formats_.push_back(fmt);
                                }
                            }
                        }
                        #else
                        // Reached only when the bridge was built without
                        // LibRaw. The body is shooting RAW and we have no
                        // decoder, so no still format can be offered. Say so
                        // explicitly instead of returning an empty list that
                        // looks like a dead camera.
                        LOG("RAW file %s ignored: built without LibRaw\n",file.name().c_str());
                        if(formats_.empty())
                            e = "Camera produced a RAW file (" + file.ext() +
                                ") but this build has no LibRaw support. "
                                "Install LibRaw and rebuild, or set the camera to JPEG.";
                        continue;
                        #endif
                    }
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
        void handle_cmd_queue()
        {
            std::unique_lock<std::recursive_mutex> guard(clock_);
            while(!cmd_queue_.empty()){
                auto cmd = cmd_queue_.front();
                cmd_queue_.pop();
                CamErrorCode e;
                set_parameter_internal(cmd.first,cmd.second,e);
                if(e) {
                    printf("%s\n",e.message().c_str());
                    handle_error(e.message().c_str());
                }
            }
            
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
            double target_fps = format_.framerate > 0 ? format_.framerate : 15.0;
            auto frame_interval = std::chrono::microseconds(
                static_cast<long long>(1e6 / std::max(1.0,target_fps)));

            std::vector<unsigned char> jpeg;
            std::pair<int,int> dims;
            int consecutive_failures = 0;

            while(!stop_) {
                // Stand down while a still capture owns the camera.
                if(pause_requested_) {
                    stream_idle_ = true;
                    std::this_thread::sleep_for(std::chrono::milliseconds(20));
                    continue;
                }
                stream_idle_ = false;

                auto started = std::chrono::steady_clock::now();
                handle_cmd_queue();
                if(stop_)
                    break;

                bool ok = false;
                try {
                    ok = grab_preview(jpeg,dims);
                }
                catch(std::exception const &e) {
                    LOG("liveview grab failed: %s\n",e.what());
                    ok = false;
                }

                if(!ok) {
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
                handle_preview_frame(jpeg);

                auto elapsed = std::chrono::steady_clock::now() - started;
                if(elapsed < frame_interval)
                    std::this_thread::sleep_for(frame_interval - elapsed);
            }

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

                // Drain the WHOLE exposure before doing anything else. The old
                // code took the first FILE_ADDED and immediately re-triggered,
                // so on a RAW+JPEG body the companion arrived during the next
                // exposure, was never selected by the format filter, and was
                // therefore never deleted — the event queue ran a shot behind
                // and the card filled with orphans.
                bool saw_complete = false;
                std::vector<CameraFilePath> files;
                try {
                    files = drain_exposure(120.0,saw_complete);
                }
                catch(std::exception const &err) {
                    handle_error(err.what());
                    break;
                }
                if(stop_)
                    break;
                handle_cmd_queue();

                bool delay_enabled = capture_delay_ > 0;
                auto time_before_downloading_files = std::chrono::steady_clock::now();
                if (!delay_enabled && !pause_requested_) {
                    trigger();
                }

                for(size_t i=0;i<files.size();i++) {
                    std::unique_lock<std::recursive_mutex> guard(clock_);
                    GPFile f(files[i],cam_,ctx_);

                    // Three outcomes per file, and they need different
                    // handling. `delivered` = decoded and handed to the
                    // consumer. `companion` = a valid file from this exposure
                    // that simply is not the selected format (the JPEG when
                    // streaming RAW, or vice versa). `failed` = we tried and
                    // could not.
                    //
                    // The old code deleted everything unconditionally,
                    // including files whose decode threw — and since the
                    // camera's copy was the only copy, one LibRaw hiccup
                    // silently destroyed the exposure.
                    bool delivered = false;
                    bool failed = false;
                    bool selected =
                        (format_.format == stream_mjpeg) ? f.is_jpeg() : !f.is_jpeg();

                    if(selected) {
                        try {
                            if(format_.format == stream_mjpeg) {
                                f.get();
                                // pass data as is
                                cv::Mat buffer(1,f.size(),CV_8UC1,const_cast<char*>(f.data()));
                                handle_frame(std::make_pair(buffer,bayer_na),&f);
                                delivered = true;
                            }
                            else if(format_.format == stream_raw16) {
                                handle_frame(f.to_mat(format_.width,format_.height,format_.bin),&f);
                                delivered = true;
                            }
                            else if(format_.format == stream_rgb48) {
                                handle_frame(to_rgb(f.to_mat(format_.width * format_.bin, format_.height * format_.bin)),&f);
                                delivered = true;
                            }
                        }
                        catch(std::exception const &err) {
                            // Report and keep the file on the card so the
                            // frame is recoverable by hand.
                            LOG("Failed to handle %s: %s\n",f.name().c_str(),err.what());
                            handle_error(err.what());
                            failed = true;
                        }
                    }

                    if(keep_images_)
                        continue;

                    // Delete the frame we delivered, and delete companions we
                    // deliberately ignored — they belong to an exposure we
                    // have already consumed, and leaving them behind is what
                    // fills the card. Only a *failed* decode is preserved,
                    // because that is the case where the camera holds the
                    // sole copy of data we could not read.
                    if(delivered || (!selected && !failed)) {
                        try {
                            f.del();
                        }
                        catch(std::exception const &err) {
                            LOG("Failed to delete %s: %s\n",f.name().c_str(),err.what());
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

                // 2) Apply and read back shutter and ISO, synchronously. Note
                //    this bypasses the command queue used by set_parameter()
                //    during streaming: a queued write is acknowledged before
                //    it reaches the camera, which is precisely the failure we
                //    are avoiding here.
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
                }

                // 4) Collect every file this exposure produced. Budget for the
                //    exposure itself plus download of multiple large files.
                bool saw_complete = false;
                double budget = result.applied_exposure * 2.0 + 60.0;
                std::vector<CameraFilePath> paths = drain_exposure(budget,saw_complete);
                result.completed_at = now_unix();

                if(paths.empty()) {
                    e = saw_complete
                        ? "Exposure completed but the camera produced no file — "
                          "check that it is not set to a card-less or automatic mode"
                        : "Timed out waiting for the exposure to produce a file";
                    return StillCaptureResult();
                }

                // 5) Download everything: RAW, JPEG, or both. Nothing is
                //    filtered by stream format here — a still capture returns
                //    what the camera actually made.
                for(size_t i=0;i<paths.size();i++) {
                    std::unique_lock<std::recursive_mutex> guard(clock_);
                    GPFile f(paths[i],cam_,ctx_);
                    f.get();
                    if(!f.data() || f.size() == 0) {
                        LOG("Empty download for %s\n",f.name().c_str());
                        continue;
                    }
                    CapturedAsset asset;
                    asset.camera_path = f.name();
                    asset.extension = f.ext();
                    unsigned char const *begin =
                        reinterpret_cast<unsigned char const*>(f.data());
                    asset.bytes.assign(begin,begin + f.size());
                    result.assets.push_back(std::move(asset));
                }

                if(result.assets.empty()) {
                    e = "The camera reported files but none could be downloaded";
                    return StillCaptureResult();
                }

                // Deliberately NOT deleting from the camera here. The caller
                // deletes via delete_captured() once the bytes are durably
                // persisted; a crash between here and that commit would
                // otherwise destroy the only copy of the exposure.
                return result;
            }
            catch(std::exception const &err) {
                e = err.what();
                return StillCaptureResult();
            }
        }

        virtual void delete_captured(std::vector<CapturedAsset> const &assets,
                                     CamErrorCode &e) override
        {
            try {
                std::unique_lock<std::recursive_mutex> guard(clock_);
                for(size_t i=0;i<assets.size();i++) {
                    std::string const &full = assets[i].camera_path;
                    size_t slash = full.rfind('/');
                    if(slash == std::string::npos)
                        continue;
                    std::string folder = full.substr(0,slash);
                    std::string name = full.substr(slash+1);
                    if(folder.empty())
                        folder = "/";
                    LOG("Deleting %s/%s\n",folder.c_str(),name.c_str());
                    int status = gp_camera_file_delete(cam_,folder.c_str(),name.c_str(),ctx_);
                    if(status != GP_OK) {
                        // Report, but keep going: a card that refuses one
                        // delete should not strand the rest.
                        LOG("Delete failed for %s: %s\n",full.c_str(),gp_result_as_string(status));
                        e = std::string("Failed to delete ") + full + ": " +
                            gp_result_as_string(status);
                    }
                }
            }
            catch(std::exception const &err) {
                e = err.what();
            }
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
            opts.push_back(opt_capture_delay);
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
                        r.min_val = 0;
                        r.max_val = 1;
                        r.step_size = 1;
                        r.cur_val = keep_images_;
                        r.def_val = 0;
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
            if(stream_active_) {
                std::unique_lock<std::recursive_mutex> guard(clock_);
                cmd_queue_.push(std::make_pair(id,value));
            }
            else {
                std::unique_lock<std::recursive_mutex> guard(clock_);
                set_parameter_internal(id,value,e);
            }
        }
        void set_parameter_internal(CamOptionId id,double value,CamErrorCode &e)
        {
            try {
                switch(id) {
                case opt_keep_images:
                    {
                        keep_images_ = !!value;
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
        std::queue<std::pair<CamOptionId,double>> cmd_queue_;
        ::Camera *cam_;
        GPContext *ctx_;
        std::atomic<int> stream_active_;
        std::vector<CamStreamFormat> formats_;
        CamStreamFormat format_;
        int frame_counter_ = 0;
        int keep_images_ = 0;
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
