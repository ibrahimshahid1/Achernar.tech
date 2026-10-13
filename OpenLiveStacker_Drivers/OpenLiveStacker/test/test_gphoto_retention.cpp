// Run from the Achernar repository root:
//     python3 tools/run_gphoto_retention_test.py
//
// Compile the production driver into this test to exercise its concrete
// Keep Images controls. These operations do not dereference the native camera
// or context; null handles ensure this test neither opens nor triggers hardware.
// This verifies the policy guard, not card retention on a physical camera.
#include "../src/gphoto2_camera.cpp"
#include "ols_bridge.h"

#include <limits>

namespace {

void require(bool condition, char const *message)
{
    if(!condition)
        throw std::runtime_error(message);
}

void require_retained(ols::GP2Camera &camera)
{
    ols::CamErrorCode error;
    auto parameter = camera.get_parameter(ols::opt_keep_images,false,error);
    require(!error,"Reading Keep Images failed");
    require(parameter.cur_val == 1.0,"Keep Images was disabled");
    require(parameter.def_val == 1.0,"Keep Images default is not retention");
}

} // namespace

int main()
{
    try {
        ols::GP2Camera camera(nullptr,nullptr);
        require_retained(camera);

        double const invalid[] = {
            0.0, -1.0, 0.5, 2.0,
            std::numeric_limits<double>::infinity(),
            -std::numeric_limits<double>::infinity(),
            std::numeric_limits<double>::quiet_NaN(),
        };
        for(double value : invalid) {
            for(bool internal : {false,true}) {
                ols::CamErrorCode error;
                if(internal)
                    camera.set_parameter_internal(ols::opt_keep_images,value,error);
                else
                    camera.set_parameter(ols::opt_keep_images,value,error);
                require(bool(error),"Invalid Keep Images value was accepted");
                require(error.message().find("A12") != std::string::npos,
                        "Blocked deletion needs an actionable qualification error");
                require_retained(camera);
            }
        }

        for(bool internal : {false,true}) {
            ols::CamErrorCode error;
            if(internal)
                camera.set_parameter_internal(ols::opt_keep_images,1.0,error);
            else
                camera.set_parameter(ols::opt_keep_images,1.0,error);
            require(!error,"Enabling Keep Images failed");
            require_retained(camera);
        }

        // A still stream must be refused before format enumeration can probe
        // the body. This fixture has no physical camera and no format cache.
        for(double framerate : {0.0,-1.0,std::numeric_limits<double>::quiet_NaN()}) {
            ols::CamStreamFormat format{};
            format.framerate = framerate;
            ols::CamErrorCode error;
            camera.start_stream(format,{},error);
            require(bool(error),"Unarchivable DSLR still stream was accepted");
            require(error.message().find("A02") != std::string::npos,
                    "Still streaming needs an actionable archive error");
        }

        // A direct native caller must also be blocked, even with a plausible
        // camera path and downloaded bytes, or with an empty asset list.
        std::vector<ols::CapturedAsset> assets;
        for(bool populated : {false,true}) {
            if(populated) {
                ols::CapturedAsset asset;
                asset.camera_path = "/store/IMG_0042.CR3";
                asset.extension = "cr3";
                asset.bytes = {1,2,3};
                assets.push_back(asset);
            }
            ols::CamErrorCode error;
            camera.delete_captured(assets,error);
            require(bool(error),"Direct native deletion was allowed");
            require(error.message().find("A12") != std::string::npos,
                    "Native deletion needs an actionable qualification error");
        }

        // These opaque placeholders are deliberately not camera objects. The
        // ABI must reject before inspecting either handle, including nulls.
        int camera_placeholder = 0;
        int capture_placeholder = 0;
        auto camera_handle = reinterpret_cast<OlsCamera *>(&camera_placeholder);
        auto capture_handle = reinterpret_cast<OlsStillCapture *>(&capture_placeholder);
        for(auto cam : {static_cast<OlsCamera *>(nullptr),camera_handle}) {
            for(auto capture : {static_cast<OlsStillCapture *>(nullptr),capture_handle}) {
                require(ols_camera_delete_captured(cam,capture) == OLS_ERR_INVALID,
                        "The C ABI deletion policy gate was bypassed");
                require(std::string(ols_last_error()).find("A12") != std::string::npos,
                        "The C ABI needs an actionable qualification error");
            }
        }

        std::cout << "Native default retention and 14 invalid setter cases passed; "
                     "both setters accept 1.0; native and C ABI deletion are blocked.\n";
        return 0;
    }
    catch(std::exception const &error) {
        std::cerr << "GPhoto retention regression failed: " << error.what() << '\n';
        return 1;
    }
}
