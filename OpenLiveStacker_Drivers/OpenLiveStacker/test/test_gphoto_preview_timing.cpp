// Reuse PTP mocks, never opening a USB camera. Exercise the production preview
// loop with known blocking inside the SDK and the synchronous Rust callback.
#define main controls_fixture_main
#include "test_gphoto_controls.cpp"
#undef main

int main()
{
    using namespace fixture;
    try {
        require(cv::imencode(".jpg",cv::Mat(4,6,CV_8UC3,cv::Scalar(20,80,180)),jpeg),
                "Could not create preview timing fixture JPEG");
        require(setenv("ACHERNAR_GPHOTO_PREVIEW_TIMING","1",1) == 0,
                "Could not enable preview timing fixture");
        ols::GP2Camera camera(nullptr,nullptr);
        ols::CamErrorCode error;
        const auto formats = camera.formats(error);
        require(!error && !formats.empty(),"Could not enumerate mocked preview");
        require(trigger_calls == 0,"Format discovery triggered a still exposure");
        const int initial_trigger_calls = trigger_calls;
        const int initial_committed_writes = committed_writes;
        block_preview = true;
        preview_blocked = false;
        release_preview = false;
        std::atomic<int> frames{0};
        std::promise<void> callback_completed;
        auto received = callback_completed.get_future();
        camera.start_stream(formats.front(),[&](const ols::CamFrame &frame) {
            require(frame.format == ols::stream_mjpeg,"Unexpected preview format");
            if(frames.fetch_add(1) == 0) callback_completed.set_value();
            std::this_thread::sleep_for(std::chrono::milliseconds(30));
        },error);
        require(!error,"Mock preview timing stream did not start");
        {
            std::unique_lock<std::mutex> guard(mutex);
            require(changed.wait_for(guard,std::chrono::seconds(2),[] { return preview_blocked; }),
                    "Mock preview never entered SDK call");
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(90));
        {
            std::lock_guard<std::mutex> guard(mutex);
            release_preview = true;
        }
        changed.notify_all();
        require(received.wait_for(std::chrono::seconds(2)) == std::future_status::ready,
                "Mock callback never completed");
        camera.stop_stream(error);
        require(!error && frames > 0,"Mock stream failed to deliver real fixture frames");
        require(trigger_calls == initial_trigger_calls,"Timing diagnostics triggered a still exposure");
        require(committed_writes == initial_committed_writes,"Timing diagnostics changed exposure settings");
        std::cout << "Native preview timing: production SDK/callback stage separation passed.\n";
        return 0;
    }
    catch(std::exception const &error) {
        std::cerr << "Preview timing fixture failed: " << error.what() << '\n';
        return 1;
    }
}
