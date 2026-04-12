/**
 * ols_bridge.h - C API bridge to OpenLiveStacker C++ drivers
 *
 * This header exposes a flat C API that Rust can call via FFI.
 * All OLS C++ classes (Camera, Mount, Focuser) are wrapped behind
 * opaque handles with C-compatible error reporting.
 */
#ifndef OLS_BRIDGE_H
#define OLS_BRIDGE_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ═══════════════════════════════════════════════════════════════════
 *  Error Handling
 * ═══════════════════════════════════════════════════════════════════ */

typedef int OlsError;
#define OLS_OK           0
#define OLS_ERR_GENERIC  1
#define OLS_ERR_NOT_FOUND 2
#define OLS_ERR_INVALID   3
#define OLS_ERR_IO        4
#define OLS_ERR_NOT_CONNECTED 5

/**
 * Get the last error message (thread-local).
 * Valid until the next OLS call on the same thread.
 */
const char* ols_last_error(void);

/* ═══════════════════════════════════════════════════════════════════
 *  Stream / Pixel Format Types
 * ═══════════════════════════════════════════════════════════════════ */

typedef enum {
    OLS_STREAM_YUV2   = 0,
    OLS_STREAM_MJPEG  = 1,
    OLS_STREAM_RGB24  = 2,
    OLS_STREAM_RGB48  = 3,
    OLS_STREAM_RAW8   = 4,
    OLS_STREAM_RAW16  = 5,
    OLS_STREAM_MONO8  = 6,
    OLS_STREAM_MONO16 = 7,
    OLS_STREAM_ERROR  = 8,
} OlsStreamType;

typedef enum {
    OLS_BAYER_NA = 0,
    OLS_BAYER_RG = 1,
    OLS_BAYER_BG = 2,
    OLS_BAYER_GR = 3,
    OLS_BAYER_GB = 4,
} OlsBayerType;

/* ═══════════════════════════════════════════════════════════════════
 *  Camera Structures & Types
 * ═══════════════════════════════════════════════════════════════════ */

typedef struct {
    OlsStreamType format;
    int32_t width;
    int32_t height;
    int32_t bin;
    float framerate;
} OlsStreamFormat;

typedef struct {
    OlsStreamType format;
    OlsBayerType bayer;
    int32_t frame_counter;
    double unix_timestamp;
    int32_t width;
    int32_t height;
    const uint8_t* data;
    size_t data_size;
} OlsFrame;

typedef enum {
    OLS_OPT_AUTO_EXP = 0,
    OLS_OPT_AUTO_WB,
    OLS_OPT_AUTO_FOCUS,
    OLS_OPT_EXP,
    OLS_OPT_WB,
    OLS_OPT_WB_R,
    OLS_OPT_WB_B,
    OLS_OPT_FOCUS,
    OLS_OPT_GAIN,
    OLS_OPT_GAMMA,
    OLS_OPT_BRIGHTNESS,
    OLS_OPT_CONTRAST,
    OLS_OPT_TEMPERATURE,
    OLS_OPT_COOLER_TARGET,
    OLS_OPT_COOLER_ON,
    OLS_OPT_FAN_ON,
    OLS_OPT_COOLER_POWER_PERC,
} OlsCamOptionId;

typedef enum {
    OLS_TYPE_BOOL = 0,
    OLS_TYPE_NUMBER,
    OLS_TYPE_MSEC,
    OLS_TYPE_PERCENT,
    OLS_TYPE_KELVIN,
    OLS_TYPE_CELSIUS,
    OLS_TYPE_SELECTION,
} OlsCamOptionType;

typedef struct {
    OlsCamOptionId option;
    OlsCamOptionType type;
    int32_t read_only;
    double step_size;
    double min_val;
    double max_val;
    double def_val;
    double cur_val;
} OlsCamParam;

typedef void (*OlsFrameCallback)(const OlsFrame* frame, void* user_data);

/* ─── Camera Opaque Handles ─── */
typedef struct OlsCameraDriver_t OlsCameraDriver;
typedef struct OlsCamera_t OlsCamera;

/* ═══════════════════════════════════════════════════════════════════
 *  Camera Driver API
 * ═══════════════════════════════════════════════════════════════════ */

/** Create a sim camera driver. */
OlsError ols_sim_driver_create(const char* sim_data_dir, OlsCameraDriver** out_driver);

/** Create an Alpaca camera driver (connects to Alpaca server over network). */
OlsError ols_alpaca_camera_driver_create(const char* url, OlsCameraDriver** out_driver);

/** Get the number of cameras from this driver. */
OlsError ols_driver_list_cameras(OlsCameraDriver* driver, int32_t* out_count);

/** Get camera name by index. */
OlsError ols_driver_camera_name(OlsCameraDriver* driver, int32_t index,
                                  char* name_buf, int32_t name_buf_len);

/** Open a camera by index. */
OlsError ols_driver_open_camera(OlsCameraDriver* driver, int32_t index, OlsCamera** out_camera);

/** Destroy a camera driver. */
void ols_driver_destroy(OlsCameraDriver* driver);

/* ─── Camera Instance API ─── */

OlsError ols_camera_name(OlsCamera* cam, char* name_buf, int32_t name_buf_len);
OlsError ols_camera_format_count(OlsCamera* cam, int32_t* out_count);
OlsError ols_camera_format_get(OlsCamera* cam, int32_t index, OlsStreamFormat* out_fmt);
OlsError ols_camera_start_stream(OlsCamera* cam, const OlsStreamFormat* fmt,
                                   OlsFrameCallback callback, void* user_data);
OlsError ols_camera_stop_stream(OlsCamera* cam);
OlsError ols_camera_option_count(OlsCamera* cam, int32_t* out_count);
OlsError ols_camera_option_id(OlsCamera* cam, int32_t index, OlsCamOptionId* out_id);
OlsError ols_camera_get_param(OlsCamera* cam, OlsCamOptionId id, OlsCamParam* out_param);
OlsError ols_camera_set_param(OlsCamera* cam, OlsCamOptionId id, double value);
void ols_camera_destroy(OlsCamera* cam);

/* ═══════════════════════════════════════════════════════════════════
 *  Mount Types & Structures
 * ═══════════════════════════════════════════════════════════════════ */

typedef enum {
    OLS_MOUNT_PROTO_NONE   = 0,
    OLS_MOUNT_PROTO_SERIAL = 1,
    OLS_MOUNT_PROTO_INET   = 2,
} OlsMountProto;

typedef enum {
    OLS_MOUNT_TRAC_SIDEREAL = 0,
    OLS_MOUNT_TRAC_SOLAR    = 1,
    OLS_MOUNT_TRAC_LUNAR    = 2,
} OlsMountTracking;

typedef enum {
    OLS_MOUNT_SLEW_STOP    = 0,
    OLS_MOUNT_SLEW_N       = (1<<1),
    OLS_MOUNT_SLEW_S       = (3<<1),
    OLS_MOUNT_SLEW_W       = (1<<3),
    OLS_MOUNT_SLEW_E       = (3<<3),
} OlsMountSlew;

typedef enum {
    OLS_MERIDIAN_UNSUPPORTED = 0,
    OLS_MERIDIAN_FLIP        = 1,
    OLS_MERIDIAN_STOP        = 2,
} OlsMeridianFlip;

typedef struct {
    double ra;   /* hours */
    double dec;  /* degrees */
} OlsEqCoord;

typedef struct {
    double alt;  /* degrees */
    double az;   /* degrees */
} OlsAltAzCoord;

typedef struct {
    char name[128];
    char option[128];
    int32_t has_option;
} OlsMountDriverInfo;

/* ─── Mount Opaque Handle ─── */
typedef struct OlsMountInterface_t OlsMountInterface;

/* ═══════════════════════════════════════════════════════════════════
 *  Mount API
 * ═══════════════════════════════════════════════════════════════════ */

/** Create a mount interface. */
OlsError ols_mount_interface_create(OlsMountInterface** out_mi);

/** List available mount drivers. */
OlsError ols_mount_driver_list(OlsMountInterface* mi,
                                OlsMountDriverInfo* out_drivers, int32_t max_drivers,
                                int32_t* out_count);

/** Load a mount driver by name (e.g., "alpaca"). */
OlsError ols_mount_load_driver(OlsMountInterface* mi,
                                const char* driver_name, const char* option);

/** Instantiate the mount from the loaded driver. */
OlsError ols_mount_load_mount(OlsMountInterface* mi);

/** Check if mount driver is loaded. */
int32_t ols_mount_driver_loaded(OlsMountInterface* mi);

/** Check if mount is loaded and ready. */
int32_t ols_mount_is_loaded(OlsMountInterface* mi);

/** Connect the mount. */
OlsError ols_mount_connect(OlsMountInterface* mi, int32_t load_alignment);

/** Check if mount is connected. */
int32_t ols_mount_connected(OlsMountInterface* mi);

/** Get current equatorial coordinates. */
OlsError ols_mount_get_current(OlsMountInterface* mi, OlsEqCoord* out_coord);

/** GoTo equatorial coordinates. */
OlsError ols_mount_goto(OlsMountInterface* mi, double ra, double dec);

/** Sync to equatorial coordinates. */
OlsError ols_mount_sync(OlsMountInterface* mi, double ra, double dec);

/** Manual slew in a direction at a speed. */
OlsError ols_mount_slew(OlsMountInterface* mi, OlsMountSlew direction, int32_t speed);

/** Get tracking mode. */
OlsError ols_mount_get_tracking(OlsMountInterface* mi, OlsMountTracking* out_mode);

/** Get tracking state (1=tracking, 0=not, -1=unknown). */
OlsError ols_mount_get_tracking_state(OlsMountInterface* mi, int32_t* out_state);

/** Set tracking state on/off. */
OlsError ols_mount_set_tracking_state(OlsMountInterface* mi, int32_t track);

/** Set tracking mode. */
OlsError ols_mount_set_tracking_mode(OlsMountInterface* mi, OlsMountTracking mode);

/** Set site latitude and longitude. */
OlsError ols_mount_set_lat_lon(OlsMountInterface* mi, double lat, double lon);

/** Get site latitude and longitude. */
OlsError ols_mount_get_lat_lon(OlsMountInterface* mi, double* out_lat, double* out_lon);

/** Get slew rate info: current rate and total count. */
OlsError ols_mount_get_slew_rate(OlsMountInterface* mi, int32_t* out_current, int32_t* out_count);

/** Check if pulse guiding is supported, get guide rates. */
OlsError ols_mount_pulse_guide_supported(OlsMountInterface* mi,
                                          int32_t* out_supported,
                                          float* out_ns_rate, float* out_we_rate);

/** Send pulse guide command. */
OlsError ols_mount_pulse_guide(OlsMountInterface* mi, float ns_ms, float we_ms);

/** Get meridian flip setting. */
OlsError ols_mount_get_meridian(OlsMountInterface* mi, OlsMeridianFlip* out_flip);

/** Set meridian flip behavior. */
OlsError ols_mount_set_meridian(OlsMountInterface* mi, OlsMeridianFlip flip);

/** Get altitude limits. */
OlsError ols_mount_get_alt_limits(OlsMountInterface* mi, int32_t* out_low, int32_t* out_high);

/** Set altitude limits. */
OlsError ols_mount_set_alt_limits(OlsMountInterface* mi, int32_t low, int32_t high);

/** Shutdown mount (disconnects and unloads). */
OlsError ols_mount_shutdown(OlsMountInterface* mi);

/** Destroy the mount interface. */
void ols_mount_interface_destroy(OlsMountInterface* mi);

/* ═══════════════════════════════════════════════════════════════════
 *  Focuser Types & Structures
 * ═══════════════════════════════════════════════════════════════════ */

typedef struct {
    char name[128];
    char option[128];
} OlsFocuserDriverInfo;

/* ─── Focuser Opaque Handle ─── */
typedef struct OlsFocuserInterface_t OlsFocuserInterface;

/* ═══════════════════════════════════════════════════════════════════
 *  Focuser API
 * ═══════════════════════════════════════════════════════════════════ */

/** Create a focuser interface. */
OlsError ols_focuser_interface_create(OlsFocuserInterface** out_fi);

/** List available focuser drivers. */
OlsError ols_focuser_driver_list(OlsFocuserInterface* fi,
                                  OlsFocuserDriverInfo* out_drivers, int32_t max_drivers,
                                  int32_t* out_count);

/** Load a focuser driver by name and option. */
OlsError ols_focuser_load_driver(OlsFocuserInterface* fi,
                                  const char* driver_name, const char* option);

/** Check if focuser is loaded. */
int32_t ols_focuser_is_loaded(OlsFocuserInterface* fi);

/** Get current focuser position. */
OlsError ols_focuser_position(OlsFocuserInterface* fi, int32_t* out_position);

/** Move to absolute position. */
OlsError ols_focuser_move_to(OlsFocuserInterface* fi, int32_t position);

/** Move by relative steps. */
OlsError ols_focuser_move_by(OlsFocuserInterface* fi, int32_t delta);

/** Abort focuser motion. */
OlsError ols_focuser_abort(OlsFocuserInterface* fi);

/** Check if focuser is currently moving. */
OlsError ols_focuser_is_moving(OlsFocuserInterface* fi, int32_t* out_moving);

/** Check if focuser uses absolute positioning. */
OlsError ols_focuser_is_absolute(OlsFocuserInterface* fi, int32_t* out_absolute);

/** Get maximum range of focuser. */
OlsError ols_focuser_max_range(OlsFocuserInterface* fi, int32_t* out_range);

/** Unload focuser driver. */
OlsError ols_focuser_unload(OlsFocuserInterface* fi);

/** Destroy the focuser interface. */
void ols_focuser_interface_destroy(OlsFocuserInterface* fi);

/* ═══════════════════════════════════════════════════════════════════
 *  Version Info
 * ═══════════════════════════════════════════════════════════════════ */

const char* ols_bridge_version(void);

#ifdef __cplusplus
}
#endif

#endif /* OLS_BRIDGE_H */
