// Exercises production gphoto capture and the production C ABI with synthetic
// PTP events/downloads. No camera is opened. The full grace uses real time so
// a companion arriving after two 300 ms timeouts catches the original bug.
#include "../src/ols_bridge.cpp"
#include "../../../OpenLiveStacker_Drivers/OpenLiveStacker/src/gphoto2_camera.cpp"

#include <map>
#include <stdexcept>

namespace fixture {
struct Event {
  CameraEventType type;
  int after_ms;
  std::string name;
};
struct Download {
  std::string bytes;
  bool fail = false;
};
std::vector<Event> events;
std::map<std::string, Download> downloads;
std::chrono::steady_clock::time_point started;
size_t next_event = 0;
bool fail_wait = false;
int triggers = 0;

void reset(std::vector<Event> input, std::map<std::string, Download> files,
           bool event_error = false) {
  events = std::move(input);
  downloads = std::move(files);
  next_event = 0;
  fail_wait = event_error;
}
void require(bool value, const char *message) {
  if (!value) throw std::runtime_error(message);
}
} // namespace fixture

extern "C" int gp_camera_get_single_config(::Camera *, const char *,
                                            CameraWidget **widget, GPContext *) {
  *widget = nullptr;
  return GP_ERROR_NOT_SUPPORTED;
}

extern "C" int gp_camera_trigger_capture(::Camera *, GPContext *) {
  ++fixture::triggers;
  fixture::started = std::chrono::steady_clock::now();
  return GP_OK;
}

extern "C" int gp_camera_wait_for_event(::Camera *, int timeout,
                                         CameraEventType *type, void **data,
                                         GPContext *) {
  *data = nullptr;
  using namespace std::chrono;
  if (fixture::next_event < fixture::events.size()) {
    const auto &event = fixture::events[fixture::next_event];
    const auto elapsed = duration_cast<milliseconds>(steady_clock::now() - fixture::started).count();
    if (elapsed < event.after_ms) {
      std::this_thread::sleep_for(milliseconds(std::min<long long>(timeout, event.after_ms - elapsed)));
      *type = GP_EVENT_TIMEOUT;
      return GP_OK;
    }
    ++fixture::next_event;
    *type = event.type;
    if (event.type == GP_EVENT_FILE_ADDED) {
      auto path = static_cast<CameraFilePath *>(std::calloc(1, sizeof(CameraFilePath)));
      std::strcpy(path->folder, "/card");
      std::strncpy(path->name, event.name.c_str(), sizeof(path->name) - 1);
      *data = path;
    }
    return GP_OK;
  }
  if (fixture::fail_wait) return GP_ERROR_IO;
  std::this_thread::sleep_for(milliseconds(timeout));
  *type = GP_EVENT_TIMEOUT;
  return GP_OK;
}

extern "C" int gp_camera_file_get(::Camera *, const char *, const char *name,
                                    CameraFileType, CameraFile *file, GPContext *) {
  auto found = fixture::downloads.find(name);
  if (found == fixture::downloads.end()) return GP_ERROR_FILE_NOT_FOUND;
  const auto &download = found->second;
  if (!download.bytes.empty()) {
    char *bytes = static_cast<char *>(std::malloc(download.bytes.size()));
    std::memcpy(bytes, download.bytes.data(), download.bytes.size());
    // CameraFile takes ownership, just as it does for a real PTP transfer.
    fixture::require(gp_file_set_data_and_size(file, bytes, download.bytes.size()) == GP_OK,
                     "Could not populate synthetic native bytes");
  }
  return download.fail ? GP_ERROR_IO : GP_OK;
}

static void inspect_snapshot(size_t expected_buffers, int expected_missing,
                             bool complete, bool strict = false) {
  OlsCamera camera;
  camera.camera = std::make_unique<ols::GP2Camera>(nullptr, nullptr);
  OlsStillRequest request{0.01, -1};
  OlsStillCapture *capture = nullptr;
  const auto status = strict
      ? ols_camera_capture_still(&camera, &request, &capture)
      : ols_camera_capture_still_snapshot(&camera, &request, &capture);
  if (strict && !complete) {
    fixture::require(status == OLS_ERR_IO && !capture, "Legacy strict API accepted an incomplete outcome");
    return;
  }
  fixture::require(status == OLS_OK && capture, "Snapshot lost a triggered outcome");
  std::unique_ptr<OlsStillCapture, decltype(&ols_capture_destroy)> owner(capture, ols_capture_destroy);
  OlsCaptureBundleInfo info{};
  info.struct_size = sizeof(info);
  fixture::require(ols_capture_bundle_info(capture, &info) == OLS_OK, "Bundle info failed");
  fixture::require(info.evidence_version == 1, "Unexpected evidence contract");
  fixture::require(info.bundle_complete == int(complete), "Incorrect native completeness");
  fixture::require(info.downloaded_count == int(expected_buffers), "Recovered native buffer was lost");
  fixture::require(info.missing_count == expected_missing, "Missing companion evidence was lost");
  const auto evidence = nlohmann::json::parse(info.evidence_json);
  fixture::require(evidence["expected_asset_count"].is_null(), "Expected companion count was invented");
  fixture::require(evidence["bundle_complete"] == complete, "JSON/ABI completeness disagree");
  for (int index = 0; index < info.downloaded_count; ++index) {
    OlsCapturedAsset asset{};
    fixture::require(ols_capture_asset_get(capture, index, &asset) == OLS_OK, "Could not read recovered asset");
    std::string path(asset.camera_path);
    const auto name = path.substr(path.rfind('/') + 1);
    fixture::require(std::string(reinterpret_cast<const char *>(asset.bytes), asset.byte_length)
                         == fixture::downloads.at(name).bytes,
                     "Native bytes changed or partial transfer was discarded");
  }
  info.struct_size = 0;
  fixture::require(ols_capture_bundle_info(capture, &info) == OLS_ERR_INVALID,
                   "Invalid evidence struct size was accepted");
}

int main() {
  using namespace fixture;
  try {
    reset({{GP_EVENT_FILE_ADDED, 0, "image.CR3"}, {GP_EVENT_CAPTURE_COMPLETE, 0, ""}},
          {{"image.CR3", {"RAW", false}}});
    inspect_snapshot(1, 0, true);
    reset({{GP_EVENT_FILE_ADDED, 0, "image.JPG"}, {GP_EVENT_CAPTURE_COMPLETE, 0, ""}},
          {{"image.JPG", {"JPEG", false}}});
    inspect_snapshot(1, 0, true);
    reset({{GP_EVENT_FILE_ADDED, 0, "image.CR3"}, {GP_EVENT_CAPTURE_COMPLETE, 0, ""},
           {GP_EVENT_FILE_ADDED, 900, "image.JPG"}},
          {{"image.CR3", {"RAW", false}}, {"image.JPG", {"JPEG", false}}});
    inspect_snapshot(2, 0, true);
    require(next_event == 3, "Late RAW+JPEG companion was not drained");
    reset({{GP_EVENT_FILE_ADDED, 0, "image.CR3"}, {GP_EVENT_FILE_ADDED, 0, "image.JPG"},
           {GP_EVENT_CAPTURE_COMPLETE, 0, ""}},
          {{"image.CR3", {"RAW", false}}, {"image.JPG", {"JPE", true}}});
    inspect_snapshot(2, 1, false);
    reset({{GP_EVENT_FILE_ADDED, 0, "image.CR3"}, {GP_EVENT_FILE_ADDED, 0, "image.JPG"},
           {GP_EVENT_CAPTURE_COMPLETE, 0, ""}}, {{"image.CR3", {"RAW", false}}});
    inspect_snapshot(1, 1, false);
    reset({{GP_EVENT_FILE_ADDED, 0, "image.CR3"}}, {{"image.CR3", {"RAW", false}}}, true);
    inspect_snapshot(1, 0, false);
    reset({{GP_EVENT_CAPTURE_COMPLETE, 0, ""}}, {});
    inspect_snapshot(0, 0, false);
    reset({{GP_EVENT_FILE_ADDED, 0, "image.CR3"}}, {{"image.CR3", {"RAW", false}}}, true);
    inspect_snapshot(1, 0, false, true);
    require(triggers == 8, "A snapshot triggered more than one exposure");
    std::cout << "Native snapshot: RAW, JPEG, late companion, partial/missing download, event error, empty outcome and legacy strict rejection passed.\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "Native snapshot regression failed: " << error.what() << '\n';
    return 1;
  }
}
