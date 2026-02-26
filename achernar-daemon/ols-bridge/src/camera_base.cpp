/**
 * camera_base.cpp - Minimal OLS camera base utilities without cppcms/booster
 *
 * Provides the string conversion functions and CamError class from camera.cpp
 * but strips out the dynamic driver loading (booster::shared_object) which
 * is not needed when we directly link drivers into the bridge.
 */

#include "camera.h"
#include <algorithm>
#include <iomanip>

namespace ols {

CamError::CamError(std::string const &msg) : std::runtime_error(msg) {}

// We don't use the dynamic driver loading, so these are stubs
static std::vector<std::string> driver_names;

std::vector<std::string> CameraDriver::drivers() { return driver_names; }

std::ostream &operator<<(std::ostream &out, CamStreamFormat const &fmt) {
  std::string fmt_name = stream_type_to_str(fmt.format);
  out << fmt_name << ":" << fmt.width << "x" << fmt.height;
  if (fmt.bin > 1) {
    out << ":bin" << fmt.bin;
  }
  if (fmt.roi_num > 1 || fmt.roi_den > 1) {
    if (fmt.bin == 1)
      out << ":";
    else
      out << ",";
    out << "roi=" << fmt.roi_num << "/" << fmt.roi_den;
  }
  if (fmt.framerate > 0) {
    out << "@" << std::fixed << std::setprecision(1) << fmt.framerate;
  }
  return out;
}

static char const *option_string_ids[] = {"auto_exp",
                                          "auto_wb",
                                          "auto_focus",
                                          "exp",
                                          "wb",
                                          "wb_r",
                                          "wb_b",
                                          "focus",
                                          "gain",
                                          "gamma",
                                          "brightness",
                                          "contrast",
                                          "temperature",
                                          "cooler_target",
                                          "cooler_on",
                                          "fan_on",
                                          "cooler_power",
                                          "average_bin",
                                          "black_level",
                                          "conv_gain_hcg",
                                          "conv_gain_hdr",
                                          "low_noise",
                                          "high_fullwell",
                                          "iso",
                                          "shutter",
                                          "viewfinder",
                                          "capturetarget",
                                          "keep_images",
                                          "capture_delay",
                                          "zoom",
                                          "live_stretch"};

static char const *option_names[] = {"Auto Exp.",
                                     "Auto WB",
                                     "Auto Focus",
                                     "Exp.",
                                     "WB",
                                     "WB Red",
                                     "WB Blue",
                                     "Focus",
                                     "Gain",
                                     "Gamma",
                                     "Bright.",
                                     "Contr.",
                                     "Temp.",
                                     "Cooler Tgt.",
                                     "Cooler",
                                     "Fan",
                                     "Cooler Pwr.",
                                     "Avg. Bin",
                                     "Black Lvl.",
                                     "ConvGain L-0/H-1",
                                     "ConvGain Lcg-0/Hcg-1/Hdr-2",
                                     "Low Noise",
                                     "High Fullwell",
                                     "ISO",
                                     "Shutter",
                                     "View Finder",
                                     "Capture Target",
                                     "Keep Images",
                                     "Inter-Shot Delay",
                                     "Zoom",
                                     "Auto Str."};

std::string bayer_type_to_str(CamBayerType bayer) {
  switch (bayer) {
  case bayer_na:
    return "NA";
  case bayer_rg:
    return "RGGB";
  case bayer_gr:
    return "GRBG";
  case bayer_bg:
    return "BGGR";
  case bayer_gb:
    return "GBRG";
  };
  throw CamError("Invalid format");
}

CamBayerType bayer_type_from_str(std::string const &bayer) {
  if (bayer == "NA")
    return bayer_na;
  if (bayer == "RGGB")
    return bayer_rg;
  if (bayer == "GRBG")
    return bayer_gr;
  if (bayer == "BGGR")
    return bayer_bg;
  if (bayer == "GBRG")
    return bayer_gb;
  throw CamError("Invalid bayer format " + bayer);
}

std::string cam_option_id_to_string_id(CamOptionId id) {
  size_t nid = id;
  if (nid >= sizeof(option_string_ids) / sizeof(option_string_ids[0]))
    throw CamError("Invalid option id");
  return option_string_ids[nid];
}

std::string stream_type_to_str(CamStreamType s) {
  switch (s) {
  case stream_yuv2:
    return "yuv2";
  case stream_rgb24:
    return "rgb24";
  case stream_rgb48:
    return "rgb48";
  case stream_raw8:
    return "raw8";
  case stream_raw16:
    return "raw16";
  case stream_mono8:
    return "mono8";
  case stream_mono16:
    return "mono16";
  case stream_mjpeg:
    return "mjpeg";
  default:
    throw CamError("Invalid Stream type");
  }
}

CamStreamType stream_type_from_str(std::string s) {
  if (s == "mjpeg")
    return stream_mjpeg;
  if (s == "yuv2")
    return stream_yuv2;
  if (s == "rgb24")
    return stream_rgb24;
  if (s == "rgb48")
    return stream_rgb48;
  if (s == "raw8")
    return stream_raw8;
  if (s == "raw16")
    return stream_raw16;
  if (s == "mono8")
    return stream_mono8;
  if (s == "mono16")
    return stream_mono16;
  throw CamError("Invalid stream type " + s);
}

static char const *cam_option_type_names[] = {
    "bool", "number", "msec", "percent", "kelvin", "celsius", "select"};

std::string cam_option_type_to_str(CamOptionType type) {
  unsigned id = type;
  if (id >= sizeof(cam_option_type_names) / sizeof(cam_option_type_names[0]))
    throw CamError("Invalid type");
  return cam_option_type_names[id];
}

CamOptionType cam_option_type_from_str(std::string const &name) {
  for (unsigned i = 0;
       i < sizeof(cam_option_type_names) / sizeof(cam_option_type_names[0]);
       i++) {
    if (cam_option_type_names[i] == name) {
      return static_cast<CamOptionType>(i);
    }
  }
  throw CamError("Invalid type:" + name);
}

std::string cam_option_id_to_name(CamOptionId id) {
  size_t nid = id;
  if (nid >= sizeof(option_names) / sizeof(option_names[0]))
    throw CamError("Invalid option id");
  return option_names[nid];
}

CamOptionId cam_option_id_from_string_id(std::string const &name) {
  for (size_t i = 0;
       i < sizeof(option_string_ids) / sizeof(option_string_ids[0]); i++)
    if (name == option_string_ids[i])
      return static_cast<CamOptionId>(i);
  throw CamError("Invalid option id");
}

} // namespace ols
