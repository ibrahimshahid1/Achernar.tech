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

/* ─── Error Handling ─── */

/** Error code: 0 = success, nonzero = error */
typedef int OlsError;
#define OLS_OK         0
#define OLS_ERR_GENERIC  1
#define OLS_ERR_NOT_FOUND 2
#define OLS_ERR_INVALID   3
#define OLS_ERR_IO        4

/**
 * Get the last error message (thread-local).
 * Returns a pointer valid until the next OLS call on the same thread.
 */
const char* ols_last_error(void);

/* ─── Stream / Pixel Format Types ─── */

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

/* ─── Camera Structures ─── */

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

/** Camera option IDs (mirrors OLS CamOptionId) */
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

/** Frame callback function pointer */
typedef void (*OlsFrameCallback)(const OlsFrame* frame, void* user_data);

/* ─── Opaque Handles ─── */
typedef struct OlsCameraDriver_t OlsCameraDriver;
typedef struct OlsCamera_t OlsCamera;

/* ─── Camera Driver API ─── */

/**
 * Create a sim camera driver.
 * @param sim_data_dir Path to simulation data directory (contains log.txt + frames)
 * @param out_driver   Receives the driver handle on success
 * @return OLS_OK on success
 */
OlsError ols_sim_driver_create(const char* sim_data_dir, OlsCameraDriver** out_driver);

/**
 * Get the number of cameras available from this driver.
 */
OlsError ols_driver_list_cameras(OlsCameraDriver* driver, int32_t* out_count);

/**
 * Get camera name by index.
 * @param name_buf    Buffer to write name into
 * @param name_buf_len Length of buffer
 */
OlsError ols_driver_camera_name(OlsCameraDriver* driver, int32_t index,
                                  char* name_buf, int32_t name_buf_len);

/**
 * Open a camera by index.
 */
OlsError ols_driver_open_camera(OlsCameraDriver* driver, int32_t index, OlsCamera** out_camera);

/**
 * Destroy a camera driver.
 */
void ols_driver_destroy(OlsCameraDriver* driver);

/* ─── Camera API ─── */

/**
 * Get camera name.
 */
OlsError ols_camera_name(OlsCamera* cam, char* name_buf, int32_t name_buf_len);

/**
 * Get number of supported stream formats.
 */
OlsError ols_camera_format_count(OlsCamera* cam, int32_t* out_count);

/**
 * Get a specific stream format by index.
 */
OlsError ols_camera_format_get(OlsCamera* cam, int32_t index, OlsStreamFormat* out_fmt);

/**
 * Start streaming with the given format.
 * The callback will be invoked from a background thread with each frame.
 */
OlsError ols_camera_start_stream(OlsCamera* cam, const OlsStreamFormat* fmt,
                                   OlsFrameCallback callback, void* user_data);

/**
 * Stop streaming. After this returns, the callback will not be called again.
 */
OlsError ols_camera_stop_stream(OlsCamera* cam);

/**
 * Get the number of supported camera options.
 */
OlsError ols_camera_option_count(OlsCamera* cam, int32_t* out_count);

/**
 * Get a supported option ID by index.
 */
OlsError ols_camera_option_id(OlsCamera* cam, int32_t index, OlsCamOptionId* out_id);

/**
 * Get parameter details for a given option.
 */
OlsError ols_camera_get_param(OlsCamera* cam, OlsCamOptionId id, OlsCamParam* out_param);

/**
 * Set a camera parameter value.
 */
OlsError ols_camera_set_param(OlsCamera* cam, OlsCamOptionId id, double value);

/**
 * Destroy a camera object.
 */
void ols_camera_destroy(OlsCamera* cam);

/* ─── Version Info ─── */
const char* ols_bridge_version(void);

#ifdef __cplusplus
}
#endif

#endif /* OLS_BRIDGE_H */
