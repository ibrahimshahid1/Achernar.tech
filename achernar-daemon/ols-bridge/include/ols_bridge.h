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
#define OLS_ERR_NOT_SUPPORTED 6

/**
 * Get the last error message (thread-local).
 * Valid until the next OLS call on the same thread.
 */
const char* ols_last_error(void);

/* ═══════════════════════════════════════════════════════════════════
 *  Build Capabilities
 * ═══════════════════════════════════════════════════════════════════ */

#define OLS_CAP_CAMERA_ASI      (1u << 0)
#define OLS_CAP_CAMERA_SVB      (1u << 1)
#define OLS_CAP_CAMERA_TOUP     (1u << 2)
#define OLS_CAP_CAMERA_GPHOTO2  (1u << 3)
#define OLS_CAP_CAMERA_UVC      (1u << 4)
#define OLS_CAP_CAMERA_INDI     (1u << 5)
#define OLS_CAP_CAMERA_INDIGO   (1u << 6)
#define OLS_CAP_CAMERA_WDIR     (1u << 7)

/* Decoder capabilities. Bit 16 onwards so the camera-driver block above stays
 * free to grow. Without LibRaw a gphoto2 body still connects, streams and
 * captures — it just returns a container nothing downstream can read, which is
 * a far more confusing failure than a missing driver. Report it at startup. */
#define OLS_CAP_DECODE_LIBRAW   (1u << 16)

/* ═══════════════════════════════════════════════════════════════════
 *  ABI Version
 * ═══════════════════════════════════════════════════════════════════ */

/**
 * Binary compatibility version of the structs and calls in this header.
 *
 * Bump this on ANY layout change to a struct crossing the boundary, or any
 * change to an existing function's signature. The Rust side compiles against
 * a copy of these declarations, so a bridge built from one revision and a
 * daemon built from another will not fail to link — they will silently read
 * each other's structs at the wrong offsets and produce garbage frames.
 *
 * History:
 *   1 — initial
 *   2 — OlsStreamFormat gained roi_num/roi_den; OlsFrame gained
 *       source_data/source_data_size/source_ext; added ols_camera_param_choices
 *       and the single-shot still capture API.
 *   3 — added the LibRaw decode API (OlsRawImage / OlsRawImageInfo,
 *       ols_decode_raw and friends). No existing struct or signature changed,
 *       but a v2 bridge does not export the new symbols, so the daemon must
 *       still refuse to run against one.
 */
/*   4 — additive still-outcome snapshot and bundle evidence API. Existing
 *       capture info/asset layouts remain unchanged. */
/*   5 — passive camera health-check export; existing layouts unchanged. */
/*   6 — additive versioned RAW metadata JSON query; shared untouched unpack. */
#define OLS_BRIDGE_ABI_VERSION 6u

/** Runtime ABI version of the compiled bridge. Compare against
 *  OLS_BRIDGE_ABI_VERSION at startup and refuse to run on a mismatch. */
uint32_t ols_bridge_abi_version(void);

/** Get the OLS bridge version string. */
const char* ols_bridge_version(void);

/** Bitmask of native camera drivers compiled into this bridge build. */
uint32_t ols_driver_capability_flags(void);

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
    /* Region-of-interest fraction (roi_num/roi_den) applied by the driver.
     * These are part of OLS's CamStreamFormat and identify a format uniquely:
     * a DSLR enumerates several entries that share width/height/bin and differ
     * only here. Dropping them meant a format selected by index could not be
     * matched back to its driver-side entry. */
    int32_t roi_num;
    int32_t roi_den;
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
    /* Optional: the unmodified bytes of the file the camera itself produced
     * (CR2/CR3/RW2/NEF/JPEG), when the driver has them. NULL otherwise.
     * `data` above may be cropped or converted for the selected format; only
     * these original container bytes permit independent decoder reprocessing. Valid only for the duration of the callback;
     * copy before returning if you intend to keep them. */
    const uint8_t* source_data;
    size_t source_data_size;
    /* Lowercase extension without the dot ("cr3", "cr2", "rw2", "jpg"), or
     * NULL. Identifies the container so it can be written with a true name. */
    const char* source_ext;
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
    OLS_OPT_AVERAGE_BIN,
    OLS_OPT_BLACK_LEVEL,
    OLS_OPT_CONV_GAIN_HCG,
    OLS_OPT_CONV_GAIN_HDR,
    OLS_OPT_LOW_NOISE,
    OLS_OPT_HIGH_FULLWELL,
    OLS_OPT_ISO,
    OLS_OPT_SHUTTER,
    OLS_OPT_VIEWFINDER,
    OLS_OPT_CAPTURE_TARGET,
    OLS_OPT_KEEP_IMAGES,
    OLS_OPT_CAPTURE_DELAY,
    OLS_OPT_ZOOM,
    OLS_OPT_LIVE_STRETCH,
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

/** Create an ASI camera driver (requires ASI SDK). */
OlsError ols_asi_camera_driver_create(OlsCameraDriver** out_driver);

/** Create an SVBony camera driver (requires SVBony SDK). */
OlsError ols_svb_camera_driver_create(OlsCameraDriver** out_driver);

/** Create a ToupTek camera driver (requires ToupTek SDK). */
OlsError ols_toup_camera_driver_create(OlsCameraDriver** out_driver);

/** Create a watch-directory pseudo-camera that ingests frames from a folder.
 *  `watch_dir` is the directory to monitor (no external dependencies). */
OlsError ols_wdir_camera_driver_create(const char* watch_dir, OlsCameraDriver** out_driver);

/** Create a gphoto2 (DSLR) camera driver (requires libgphoto2). */
OlsError ols_gphoto2_camera_driver_create(OlsCameraDriver** out_driver);

/** Create a UVC (USB video) camera driver (requires libuvc). */
OlsError ols_uvc_camera_driver_create(OlsCameraDriver** out_driver);

/** Create an INDI camera driver (requires libindi; connects to an INDI server). */
OlsError ols_indi_camera_driver_create(OlsCameraDriver** out_driver);

/** Create an INDIGO camera driver (requires INDIGO libraries). */
OlsError ols_indigo_camera_driver_create(OlsCameraDriver** out_driver);

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

/** Passive device presence transaction. NOT_SUPPORTED is distinct from an
 *  actual failed probe; never captures, moves hardware, or changes settings. */
OlsError ols_camera_check_health(OlsCamera* cam);
OlsError ols_camera_format_count(OlsCamera* cam, int32_t* out_count);
OlsError ols_camera_format_get(OlsCamera* cam, int32_t index, OlsStreamFormat* out_fmt);
OlsError ols_camera_start_stream(OlsCamera* cam, const OlsStreamFormat* fmt,
                                   OlsFrameCallback callback, void* user_data);
OlsError ols_camera_stop_stream(OlsCamera* cam);
OlsError ols_camera_option_count(OlsCamera* cam, int32_t* out_count);
OlsError ols_camera_option_id(OlsCamera* cam, int32_t index, OlsCamOptionId* out_id);
OlsError ols_camera_get_param(OlsCamera* cam, OlsCamOptionId id, OlsCamParam* out_param);
OlsError ols_camera_set_param(OlsCamera* cam, OlsCamOptionId id, double value);

/**
 * Fetch the human-readable choice labels of a OLS_TYPE_SELECTION option.
 *
 * DSLR and mirrorless bodies do not expose exposure and gain as continuous
 * numbers. They expose `shutterspeed` and `iso` as enumerations whose *labels*
 * carry the only physical meaning: index 17 means nothing, "1/125" means an
 * eight-millisecond exposure. OlsCamParam reports min/max/cur as bare indices,
 * so without this call the daemon can set an exposure but cannot know what
 * exposure it set — and any metadata it records is a guess.
 *
 * Labels are written into `buf` as consecutive NUL-terminated strings
 * ("100\0" "200\0" "400\0"), in choice-index order, so the Nth string is the
 * value to pass to ols_camera_set_param for choice N.
 *
 * `*out_count` always receives the number of choices, and `*out_required` the
 * total buffer size needed, even when `buf` is NULL or too small — call once
 * with buf=NULL to size the allocation, then again to fill it. Returns
 * OLS_ERR_INVALID (without writing) when `buf` is non-NULL but too small.
 *
 * A non-selection option is not an error: it reports zero choices.
 */
OlsError ols_camera_param_choices(OlsCamera* cam, OlsCamOptionId id,
                                  char* buf, size_t buf_len,
                                  int32_t* out_count, size_t* out_required);

/* ═══════════════════════════════════════════════════════════════════
 *  Single-Shot Still Capture
 * ═══════════════════════════════════════════════════════════════════
 *
 * Separate from the streaming API on purpose. A stream is free-running: the
 * consumer sees whatever frame happened to arrive most recently, and a setting
 * written mid-stream may not have reached the sensor before the next frame
 * appears. Fine for a preview, unusable for a science exposure, where the
 * pixels and the metadata describing them must come from one shutter
 * actuation.
 *
 * ols_camera_capture_still() applies and reads back the settings, triggers
 * exactly one exposure, and returns every file that exposure produced. None of
 * it passes through the frame callback.
 *
 * Ownership: the returned OlsStillCapture owns its byte buffers. Copy what you
 * need, then call ols_capture_destroy(). Releasing the handle retains camera
 * files. Camera deletion is disabled pending verified persistence and recovery
 * qualification (A12); a local commit alone does not authorize deletion. */

typedef struct {
    /** Requested exposure. A body with an enumerated shutter ladder selects
     *  the nearest step and reports it back in OlsStillCaptureInfo. */
    double exposure_seconds;
    /** Requested ISO / gain. <= 0 leaves the body's current setting alone. */
    double sensitivity;
} OlsStillRequest;

typedef struct {
    /** Unix timestamps bracketing the exposure, taken by the driver around the
     *  trigger. `started_at` is what belongs in FITS DATE-OBS — a clock read
     *  after the call returns is late by the exposure plus the download. */
    double started_at;
    double completed_at;
    /** Settings read back from the camera after being applied. */
    double applied_exposure;
    double applied_sensitivity;
    /** The camera's own labels, e.g. "1/125" and "800". Owned by the capture;
     *  valid until ols_capture_destroy(). Never NULL — empty string instead. */
    const char* shutter_label;
    const char* iso_label;
    int32_t asset_count;
} OlsStillCaptureInfo;

typedef struct {
    /** Original path on the camera's storage, retained for provenance.
     *  Owned by the capture. */
    const char* camera_path;
    /** Lowercase extension without the dot: "cr3", "cr2", "rw2", "jpg". */
    const char* extension;
    const uint8_t* bytes;
    size_t byte_length;
} OlsCapturedAsset;

typedef struct OlsStillCapture_t OlsStillCapture;

/** Evidence about the observed native file set, not a qualified durability
 *  receipt. Initialize struct_size before calling ols_capture_bundle_info.
 *  expected companion count remains unknown unless the camera reports it. */
typedef struct {
    uint32_t evidence_version;
    uint32_t struct_size;
    int32_t event_complete;
    int32_t discovered_count;
    int32_t downloaded_count;
    int32_t missing_count;
    int32_t error_count;
    int32_t bundle_complete;
    const char* evidence_json;
    const char* error;
} OlsCaptureBundleInfo;

/** Whether this camera implements single-shot still capture. */
OlsError ols_camera_supports_still(OlsCamera* cam, int32_t* out_supported);

/** Take one exposure. Blocking; suspends any running stream for the duration.
 *  On success `*out_capture` must be released with ols_capture_destroy(). */
OlsError ols_camera_capture_still(OlsCamera* cam, const OlsStillRequest* request,
                                  OlsStillCapture** out_capture);

/** Take one exposure and retain an outcome snapshot after incomplete event
 *  collection/download. OLS_OK means an outcome is available, not that the
 *  native bundle is complete. Inspect bundle_info and archive every returned
 *  buffer before reporting an incomplete outcome. A pre-trigger failure with
 *  no outcome returns an error and no handle. */
OlsError ols_camera_capture_still_snapshot(OlsCamera* cam, const OlsStillRequest* request,
                                          OlsStillCapture** out_capture);

OlsError ols_capture_bundle_info(const OlsStillCapture* capture,
                                OlsCaptureBundleInfo* out_info);

/** Timestamps, applied settings and asset count for a completed capture. */
OlsError ols_capture_info(const OlsStillCapture* capture, OlsStillCaptureInfo* out_info);

/** Number of files the exposure produced (2 on a RAW+JPEG body). */
int32_t ols_capture_asset_count(const OlsStillCapture* capture);

/** Borrow one asset. Pointers are owned by the capture and are valid until
 *  ols_capture_destroy(); copy before releasing. */
OlsError ols_capture_asset_get(const OlsStillCapture* capture, int32_t index,
                               OlsCapturedAsset* out_asset);

/** Reserved deletion entry point. Always returns OLS_ERR_INVALID with an A12
 *  qualification error, including for null handles; it never deletes files. */
OlsError ols_camera_delete_captured(OlsCamera* cam, const OlsStillCapture* capture);

/** Release a capture and its byte buffers. */
void ols_capture_destroy(OlsStillCapture* capture);

void ols_camera_destroy(OlsCamera* cam);

/* ═══════════════════════════════════════════════════════════════════
 *  RAW Decode (LibRaw)
 * ═══════════════════════════════════════════════════════════════════
 *
 * Turns the bytes of a camera-authored RAW container (CR2/CR3/RW2/NEF/ARW)
 * into a plain 16-bit Bayer plane. Not a camera method: the input is a byte
 * buffer, so a file downloaded hours ago on another machine decodes exactly
 * like one that just came off the shutter.
 *
 * ── What comes out: raw sensor ADU ──
 *
 * The pixels are the sensor's own numbers. No black-level subtraction, no
 * scaling, no white balance, no left-shift to "fill" 16 bits. Everything
 * needed to calibrate them later is reported in OlsRawImageInfo instead.
 *
 * The legacy load_libraw wrapper and this entry point share untouched_raw.h.
 * Both support bin=1 only. The guarantee starts at LibRaw unpack output;
 * camera/container/LibRaw transformations may already have occurred.
 * Full decoder metadata and black tables are available through the versioned
 * JSON query below. Metadata is not automatically applied as calibration.
 * A declared white code is not a measured nonlinearity limit.
 *
 * ── Ownership ──
 *
 * `data` is BORROWED for the duration of ols_decode_raw() only. The caller
 * keeps owning the container bytes and may free them the moment it returns.
 *
 * The returned OlsRawImage owns one tightly-packed pixel buffer and the
 * strings in its info. Release it with ols_rawimage_destroy(). LibRaw's own
 * (larger, margin-inclusive) buffers are freed before the call returns.
 */

typedef struct OlsRawImage_t OlsRawImage;

typedef struct {
    /** Post-crop dimensions of the returned plane. */
    int32_t       width;
    int32_t       height;
    /** Bayer phase at pixel (0,0) OF THE RETURNED PLANE, not of the sensor. */
    OlsBayerType  bayer;
    /** Storage depth of each sample: always 16. Container capacity, declared
     *  unpack white code and measured detector linearity are distinct. */
    int32_t       bits_per_sample;

    /* ── Provenance: what the decode did NOT do, so it can be done later ── */

    /** Global black pedestal in ADU. NOT subtracted from the pixels. */
    int32_t       black_level;
    /** Per-CFA-channel black offsets, added to black_level. Zero on bodies
     *  that use a single pedestal. Recorded because a black level that omits
     *  them is wrong by a few ADU per channel — small, but it is exactly the
     *  systematic that shows up as a colour cast in a stacked background. */
    int32_t       cblack[4];
    /** Dimensions of LibRaw's 2-D black-level table, when the body has one
     *  ({0,0} otherwise). Spatial table entries are separate from cblack[0..3],
     *  even for a 2x2 table. Use ols_rawimage_metadata_json for the full table. */
    int32_t       cblack_pattern[2];
    /** Declared unpack-domain white code, not a measured nonlinearity limit. */
    int32_t       white_level;
    /** As-shot white balance multipliers. Recorded, never applied. */
    float         cam_mul[4];
    /** Borrowed from the image; valid until ols_rawimage_destroy(). Never
     *  NULL — empty string instead. */
    const char   *make;
    const char   *model;
} OlsRawImageInfo;

/**
 * Decode `len` bytes of a RAW container.
 *
 * Pass width=-1, height=-1 for the full active area. A positive width/height
 * selects a centred sub-region; the offset is forced even so the Bayer phase
 * of the returned plane still matches `bayer`.
 *
 * `bin` must be 1. The parameter exists so binning can be added without
 * another ABI break, but binning here would be a lossy pre-calibration step of
 * exactly the kind this API refuses to take: a 2x2 sum saturates at 4x
 * white_level, so the returned plane would need saturation semantics that
 * OlsRawImageInfo cannot currently express. Bin after calibration.
 *
 * Returns OLS_ERR_NOT_FOUND when the bridge was built without LibRaw, and
 * OLS_ERR_INVALID for a container LibRaw parses but that has no 16-bit Bayer
 * plane (Foveon, sRAW, and float DNG all land here).
 */
OlsError ols_decode_raw(const uint8_t* data, size_t len,
                        int32_t width, int32_t height, int32_t bin,
                        OlsRawImage** out_image);

/** Dimensions, Bayer phase and calibration provenance. */
OlsError ols_rawimage_info(const OlsRawImage* image, OlsRawImageInfo* out_info);

/**
 * Borrow the pixel plane: `width * height * 2` bytes, row-major, tightly
 * packed, native-endian uint16. Valid until ols_rawimage_destroy().
 * Returns NULL on a null handle.
 */
const uint8_t* ols_rawimage_pixels(const OlsRawImage* image, size_t* out_len);

/** Versioned JSON metadata, borrowed until image destruction. Includes
 * decoder/version/options, raw/visible/crop geometry, orientation (unapplied),
 * complete spatial black table, declared white and per-channel linear limits.
 * NULL for a null handle. Existing OlsRawImageInfo layout is unchanged. */
const char* ols_rawimage_metadata_json(const OlsRawImage* image);

/** Release an image and its pixel buffer. */
void ols_rawimage_destroy(OlsRawImage* image);

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

#ifdef __cplusplus
}
#endif

#endif /* OLS_BRIDGE_H */
