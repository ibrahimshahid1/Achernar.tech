#pragma once
#include <stdexcept>
#include <string>
#include <ostream>
#include <vector>
#include <memory>
#include <functional>

namespace ols {


    class CamError : public std::runtime_error {
    public:
        CamError(std::string const &msg);
    };

    class CamErrorCode {
    public:
        CamErrorCode() : status_(false), message_("ok") {}
        CamErrorCode(std::string const &msg) :status_(true), message_(msg) {}
        CamErrorCode(std::exception const &e) : status_(true), message_(e.what())
        {
        }
        CamErrorCode &operator=(std::string const &msg)
        {
            status_ = true;
            message_ = msg;
            return *this;
        }
        CamErrorCode(CamErrorCode const &) = default;
        CamErrorCode &operator=(CamErrorCode const &) = default;
        explicit operator bool() const
        {
            return status_;
        }
        std::string const &message() const
        {
            return message_;
        }
        void check()
        {
            if(status_)
                throw CamError(message_);
        }
    private:
        bool status_;
        std::string message_;
    };

    enum CamOptionId {
        opt_auto_exp,
        opt_auto_wb,
        opt_auto_focus,
        opt_exp,
        opt_wb,
        opt_wb_r,
        opt_wb_b,
        opt_focus,
        opt_gain,
        opt_gamma,
        opt_brightness,
        opt_contrast,
        opt_temperature,
        opt_cooler_target,
        opt_cooler_on,
        opt_fan_on,
        opt_cooler_power_perc,
        opt_average_bin,
        opt_black_level,
        opt_conv_gain_hcg,
        opt_conv_gain_hdr,
        opt_low_noise, // 	low noise mode (Higher signal noise ratio, lower frame rate)
        opt_high_fullwell, // high fullwell capacity
        /// DSLR
        opt_iso,
        opt_shutter,
        opt_viewfinder,
        opt_capturetarget,
        opt_keep_images,
        opt_capture_delay,
        /// Android
        opt_zoom,
        /// External non-device specific options start HERE!
        opt_live_stretch,
        opt_count
    };

    inline bool is_external_option(CamOptionId opt) { return opt >= opt_live_stretch; }

    //int str_to_enum(std::string const &name,char const **names,size_t names_size);
    //std::string enum_to_str(int v,char const **names,size_t names_size);

    std::string cam_option_id_to_string_id(CamOptionId id);
    std::string cam_option_id_to_name(CamOptionId id);
    CamOptionId cam_option_id_from_string_id(std::string const &name);

    enum CamOptionType {
        type_bool,
        type_number,
        type_msec,
        type_percent,
        type_kelvin,
        type_celsius,
        type_selection,
    };

    std::string cam_option_type_to_str(CamOptionType type);
    CamOptionType cam_option_type_from_str(std::string const &name);

    struct CamParam {
        CamOptionId option = opt_auto_exp;
        CamOptionType type = type_number;
        bool read_only = false;
        double step_size = 0;
        double min_val = 0;
        double max_val = 0;
        double def_val = 0;
        double cur_val = 0;
        std::vector<std::string> names;
    };

    enum CamStreamType {
        stream_yuv2,
        stream_mjpeg,
        stream_rgb24,
        stream_rgb48,
        stream_raw8,
        stream_raw16,
        stream_mono8,
        stream_mono16,
        stream_error,
    };

    enum CamBayerType {
        bayer_na,
        bayer_rg,
        bayer_bg,
        bayer_gr,
        bayer_gb
    };

    std::string bayer_type_to_str(CamBayerType bayer);
    CamBayerType bayer_type_from_str(std::string const &s);

    std::string stream_type_to_str(CamStreamType s);
    CamStreamType stream_type_from_str(std::string s);

    inline bool is_mono_stream(CamStreamType s)
    {
        return s==stream_mono8 || s==stream_mono16;
    }

    struct CamStreamFormat {
        CamStreamType format;
        int width,height;
        int bin = 1;
        int roi_num = 1; 
        int roi_den = 1; 
        // roi_num = 2, roi_den=3 -> 2/3
        float framerate = -1.0f; // -1 for unknown
    };

    std::ostream &operator<<(std::ostream &out,CamStreamFormat const &fmt);

    ///
    /// frame received by \a frame_callback_type 
    ///
    struct CamFrame {
        CamStreamType format; /// image format
        CamBayerType bayer = bayer_na;
        int frame_counter;    /// frame counter since camera started
        double unix_timestamp; /// time in sconds since Jan 1, 1970 when image was captured, inlcuding subsecond units
        int width;  /// image width
        int height; /// image height
        void const *data; /// data - pointer becomes invalid when callback ends, points to null terminamted char const * for \a stream_error format to indicate caputre error
        size_t data_size; /// size of data in bytes

        /// ─── Optional original-file passthrough ───
        ///
        /// When a driver obtains its pixels by decoding a container the camera
        /// itself produced (a DSLR writing CR2/CR3/RW2/NEF, or a JPEG), these
        /// point at the *undecoded, unmodified* bytes of that file.
        ///
        /// This exists because `data` above is already lossy with respect to
        /// the original: LibRaw applies black-level subtraction and a scale
        /// factor, crops to the active area, and discards the maker notes that
        /// carry per-shot metadata. A frame archive built only from `data` can
        /// never be re-processed with a different demosaic or calibration, so
        /// any model trained from it is permanently locked to today's decode
        /// path. Consumers that want a canonical archive copy should persist
        /// these bytes verbatim.
        ///
        /// Lifetime matches `data`: valid only for the duration of the
        /// callback. `nullptr` when the driver has no meaningful original
        /// (sensor drivers that DMA pixels straight out of hardware).
        void const *source_data = nullptr;
        /// Size of \a source_data in bytes; 0 when absent.
        size_t source_data_size = 0;
        /// Lowercase file extension without the dot ("cr3", "cr2", "rw2",
        /// "nef", "arw", "jpg"), or nullptr. Tells the consumer what container
        /// \a source_data actually is so it can be written with a truthful name.
        char const *source_ext = nullptr;
    };

    /// callback to pass to camera - called from separate thread when frame is ready
    typedef std::function<void(CamFrame const &)> frame_callback_type;

    /// ═══════════════════════════════════════════════════════════════════
    ///  Single-shot still capture
    /// ═══════════════════════════════════════════════════════════════════
    ///
    /// Deliberately separate from the streaming path. A stream is a
    /// free-running producer: frames arrive when they arrive, the consumer
    /// sees only the most recent one, and settings written mid-stream may or
    /// may not have taken effect by the time the next frame appears. That is
    /// fine for a preview and unusable for a science exposure, where the
    /// pixels and the metadata describing them must come from the same
    /// shutter actuation.
    ///
    /// capture_still() therefore owns its exposure end to end: it applies and
    /// reads back the settings, triggers exactly one exposure, collects every
    /// file that exposure produced, and returns them to the caller directly.
    /// Nothing passes through frame_callback_type.

    /// One file the camera itself wrote for a single exposure. A body set to
    /// RAW+JPEG produces two of these per actuation.
    struct CapturedAsset {
        /// Path on the camera's storage, e.g.
        /// "/store_00020001/DCIM/100CANON/IMG_0042.CR3". Retained for
        /// provenance; a local commit does not authorize deleting this copy.
        std::string camera_path;
        /// Lowercase extension without the dot: "cr3", "cr2", "rw2", "jpg".
        std::string extension;
        /// The downloaded bytes, verbatim. May be partial when a native
        /// transfer failed; incomplete bytes still need archival preservation.
        std::vector<unsigned char> bytes;
        bool download_complete = true;
    };

    struct CaptureDownloadEvidence {
        std::string camera_path;
        size_t byte_length = 0;
        bool download_complete = false;
        std::string error;
    };

    struct StillCaptureRequest {
        /// Requested exposure. Bodies with an enumerated shutter ladder select
        /// the nearest available step and report it back in
        /// StillCaptureResult::applied_exposure.
        double exposure_seconds = 1.0;
        /// Requested ISO / gain. <= 0 means "leave whatever the body is set
        /// to", so the operator's manual setting is never silently overridden.
        double sensitivity = 0.0;
    };

    struct StillCaptureResult {
        /// Unix timestamps bracketing the actual exposure, taken by the driver
        /// immediately before the trigger and after the last file arrives.
        /// `started_at` is what belongs in FITS DATE-OBS — not the wall clock
        /// read after the capture returns.
        double started_at = 0.0;
        double completed_at = 0.0;
        /// Settings as read back from the camera after being applied, never
        /// the requested values.
        double applied_exposure = 0.0;
        double applied_sensitivity = 0.0;
        /// The camera's own labels for those settings ("1/125", "800").
        std::string shutter_label;
        std::string iso_label;
        /// Every nonempty native buffer recovered, including partial transfers.
        std::vector<CapturedAsset> assets;
        bool triggered = false;
        bool event_complete = false;
        bool completion_grace_complete = false;
        bool bundle_complete = false;
        std::string drain_termination = "not_started";
        std::vector<std::string> discovered_paths;
        std::vector<CaptureDownloadEvidence> downloads;
        std::vector<std::string> errors;
    };


    ///
    /// Generic Camera interface - to be implemented for new camera driver
    ///
    /// Note we are using CamErrorCode instead of throwing C++ acception because 
    /// bloody Android NDK does not handle exceptions accross dynamically loaded
    /// shared object boundaries...
    ///
    /// So falling back to C--/C with classes methods :(
    ///
    /// it is not me... it is Android :-/
    ///
    class Camera {
    public:
        Camera() {}
        Camera(Camera const &) = delete;
        Camera(Camera &&) = delete;
        void operator = (Camera const &) = delete;
        void operator = (Camera &&) = delete;
        
        /// Camera name
        virtual std::string name(CamErrorCode &e) = 0;
        /// Return list of suppored video formats
        virtual std::vector<CamStreamFormat> formats(CamErrorCode &e) = 0;

        /// Start a video stream with provided callback 
        virtual void start_stream(CamStreamFormat format,frame_callback_type callback,CamErrorCode &e) = 0;

        /// stop the stream - once function ends callback will not be called any more
        virtual void stop_stream(CamErrorCode &e) = 0;

        /// Passive presence probe. True means the driver performed a real
        /// device transaction (with errors reported in e); false is unsupported.
        /// It must never capture, move hardware, or modify camera settings.
        virtual bool check_connection(CamErrorCode &e) { (void)e; return false; }

        /// True when this driver implements capture_still(). Callers should
        /// check rather than relying on the error, so a UI can offer a real
        /// still-capture button only where it means something.
        virtual bool supports_still_capture() const { return false; }

        /// Take exactly one exposure and return its files.
        ///
        /// Blocking, and safe to call while a stream is running: an
        /// implementation must suspend its own stream for the duration rather
        /// than interleaving preview traffic with the capture, which is what
        /// provokes "camera busy" on PTP bodies.
        ///
        /// Default: unsupported. Sensor drivers that DMA pixels out of
        /// hardware have no separate still path and keep using the stream.
        virtual StillCaptureResult capture_still(StillCaptureRequest const &request,
                                                 CamErrorCode &e)
        {
            (void)request;
            e = "This camera driver does not implement single-shot still capture";
            return StillCaptureResult();
        }

        /// Delete the given files from the camera's storage.
        ///
        /// Separate from capture_still() on purpose. Achernar currently
        /// blocks this operation pending verified persistence and recovery
        /// qualification (A12). A future deletion policy must also acknowledge
        /// removal of the retained camera copy; a local commit alone is not
        /// permission to delete.
        virtual void delete_captured(std::vector<CapturedAsset> const &assets,
                                     CamErrorCode &e)
        {
            (void)assets;
            e = "This camera driver does not implement captured-file deletion";
        }

        /// list of camera controls that the camera supports
        virtual std::vector<CamOptionId> supported_options(CamErrorCode &e) = 0;
        /// get camera control
        CamParam get_parameter(CamOptionId id,CamErrorCode &e) 
        {
            return get_parameter(id,false,e);
        }
        virtual CamParam get_parameter(CamOptionId id,bool current_only,CamErrorCode &e) = 0;
        /// set camera control
        virtual void set_parameter(CamOptionId id,double value,CamErrorCode &e) = 0;

        virtual std::vector<CamParam> get_all_parameters(bool current_only,CamErrorCode &e) 
        {
            std::vector<CamParam> r;
            std::vector<CamOptionId> ids = supported_options(e);
            if(e)
                return r;
            r.reserve(ids.size());
            for(auto opt: ids) {
                auto opt_val = get_parameter(opt,current_only,e);
                if(e)
                    return r;
                r.push_back(opt_val);
            }
            return r;
        }

        virtual ~Camera() {}
    };


    /// Base class for driver to implement
    class CameraDriver {
    public:
        static void load_driver(std::string const &name,std::string base_path="",char const *option = nullptr,std::string camera_log="",int cam_debug = 0);
        /// list supported drivers
        static std::vector<std::string> drivers();
        /// get driver - by its order in the \a drivers result
        static std::unique_ptr<CameraDriver> get(int id,int external_option);

        /// list all cameras connected and supported by this driver
        virtual std::vector<std::string> list_cameras(CamErrorCode &e) = 0;

        /// get camera object for camera according to the index in list_cameras
        virtual std::unique_ptr<Camera> open_camera(int id,CamErrorCode &e) = 0;

        virtual ~CameraDriver() {}
    };

} // namespace
