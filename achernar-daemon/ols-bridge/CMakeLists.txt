cmake_minimum_required(VERSION 3.21)
project(ols_bridge LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Keep the first Windows bring-up deliberately small and deterministic.  The
# vendor SDK bundles currently checked out in this repository contain macOS
# and Linux binaries, not MSVC import libraries.  They must never be selected
# merely because their headers happen to be present on a Windows checkout.
option(ACHERNAR_ENABLE_VENDOR_DRIVERS
    "Build camera drivers backed by the bundled vendor SDKs" ON)
if(WIN32 AND ACHERNAR_ENABLE_VENDOR_DRIVERS)
    message(STATUS
        "Bundled vendor camera SDKs are not Windows-compatible; ASI/SVB/Toup "
        "drivers will remain disabled until Windows SDK packages are supplied")
endif()

# ─── Paths ───
set(PROJECT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")
set(OLS_ROOT "${PROJECT_ROOT}/OpenLiveStacker_Drivers/OpenLiveStacker")
set(OLS_INC "${OLS_ROOT}/include")
set(OLS_SRC "${OLS_ROOT}/src")
set(SDKS_ROOT "${PROJECT_ROOT}/sdks")
set(BRIDGE_INC "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(BRIDGE_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(SHIMS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shims")
set(EXTERNAL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external")

# ─── Detect macOS architecture ───
if(APPLE)
    execute_process(COMMAND uname -m OUTPUT_VARIABLE HOST_ARCH OUTPUT_STRIP_TRAILING_WHITESPACE)
    message(STATUS "Host architecture: ${HOST_ARCH}")
endif()

# ─── Find required packages ───
#
# Windows uses vcpkg's CMake package configs.  Requiring pkg-config there is
# both unnecessary and hostile to a standard MSVC developer environment.  On
# Unix we retain the existing pkg-config behavior, including Homebrew's keg
# handling below.
if(WIN32)
    find_package(OpenCV 4 CONFIG REQUIRED COMPONENTS core imgproc imgcodecs)
    find_package(TIFF REQUIRED)

    set(ACHERNAR_OPENCV_INCLUDE_DIRS ${OpenCV_INCLUDE_DIRS})
    set(ACHERNAR_OPENCV_LIBS ${OpenCV_LIBS})
    set(ACHERNAR_TIFF_INCLUDE_DIRS ${TIFF_INCLUDE_DIRS})
    if(TARGET TIFF::TIFF)
        set(ACHERNAR_TIFF_LIBS TIFF::TIFF)
    elseif(TARGET TIFF::tiff)
        # Some libtiff config packages use the lower-case target name.
        set(ACHERNAR_TIFF_LIBS TIFF::tiff)
    else()
        set(ACHERNAR_TIFF_LIBS ${TIFF_LIBRARIES})
    endif()
else()
    find_package(PkgConfig REQUIRED)
endif()

# Homebrew's main `opencv` formula now tracks OpenCV 5, which ships opencv5.pc
# and NOT opencv4.pc. OpenCV 4 is available only via the `opencv@4` formula,
# which is keg-only (not symlinked into the brew prefix) precisely because the
# unversioned formula occupies the same namespace — so its pkgconfig directory
# is invisible to pkg-config unless it is added explicitly.
#
# Probe brew for the keg prefixes we depend on and prepend their pkgconfig dirs
# before the REQUIRED checks below. Without this, a machine with a perfectly
# good `brew install opencv@4` still fails configure with "Package 'opencv4'
# not found", and every developer has to discover PKG_CONFIG_PATH themselves.
# Non-macOS builds are untouched; a missing formula is skipped silently.
if(APPLE)
    find_program(BREW_EXECUTABLE brew)
    if(BREW_EXECUTABLE)
        foreach(KEG_FORMULA "opencv@4" "libtiff" "libraw" "libgphoto2")
            execute_process(
                COMMAND ${BREW_EXECUTABLE} --prefix ${KEG_FORMULA}
                OUTPUT_VARIABLE KEG_PREFIX
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET)
            if(KEG_PREFIX AND EXISTS "${KEG_PREFIX}/lib/pkgconfig")
                set(ENV{PKG_CONFIG_PATH} "${KEG_PREFIX}/lib/pkgconfig:$ENV{PKG_CONFIG_PATH}")
                list(APPEND CMAKE_PREFIX_PATH "${KEG_PREFIX}")
                message(STATUS "Using brew keg ${KEG_FORMULA} at ${KEG_PREFIX}")
            endif()
        endforeach()
    endif()
endif()

if(NOT WIN32)
    pkg_check_modules(OPENCV REQUIRED opencv4)
    pkg_check_modules(TIFF REQUIRED libtiff-4)
    set(ACHERNAR_OPENCV_INCLUDE_DIRS ${OPENCV_INCLUDE_DIRS})
    set(ACHERNAR_OPENCV_LIBS ${OPENCV_LIBRARIES})
    set(ACHERNAR_TIFF_INCLUDE_DIRS ${TIFF_INCLUDE_DIRS})
    set(ACHERNAR_TIFF_LIBS ${TIFF_LIBRARIES})
endif()

# ─── Core sources (always compiled) ───
set(BRIDGE_SOURCES
    # Our bridge layer
    ${BRIDGE_SRC}/ols_bridge.cpp
    ${BRIDGE_SRC}/camera_base.cpp

    # OLS Alpaca drivers (no vendor SDK needed)
    ${OLS_SRC}/alpaca_camera.cpp
    ${OLS_SRC}/mount.cpp
    ${OLS_SRC}/alpaca_mount.cpp
    ${OLS_SRC}/focuser.cpp
    ${OLS_SRC}/alpaca_focuser.cpp
    ${OLS_SRC}/alpaca_client.cpp

    # OLS utilities (needed by sim camera for file I/O)
    ${OLS_SRC}/util.cpp
    ${OLS_SRC}/tiffmat.cpp

)

set(EXTRA_INCLUDE_DIRS "")
set(EXTRA_LINK_DIRS "")
set(EXTRA_LIBS "")

# Watch-directory pseudo-camera uses Linux inotify via OLS DirWatch.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    list(APPEND BRIDGE_SOURCES ${OLS_SRC}/wdir_camera.cpp)
else()
    message(STATUS "ℹ️  Watch-directory camera disabled — requires Linux inotify")
endif()

# ═══════════════════════════════════════════════════════════════════
#  ASI (ZWO) Camera SDK
# ═══════════════════════════════════════════════════════════════════
set(ASI_SDK_DIR "${SDKS_ROOT}/ASI/ASI_Camera_SDK/ASI_linux_mac_SDK_V1.41")
set(ASI_INC "${ASI_SDK_DIR}/include")

if(ACHERNAR_ENABLE_VENDOR_DRIVERS AND NOT WIN32 AND
   EXISTS "${ASI_INC}/ASICamera2.h")
    message(STATUS "✅ ASI SDK found at ${ASI_SDK_DIR}")
    list(APPEND BRIDGE_SOURCES ${OLS_SRC}/asi_camera.cpp)
    list(APPEND EXTRA_INCLUDE_DIRS ${ASI_INC})
    add_definitions(-DWITH_ASI)

    # Select correct library for platform/arch
    if(APPLE)
        if(HOST_ARCH STREQUAL "arm64")
            set(ASI_LIB_DIR "${ASI_SDK_DIR}/lib/mac_arm64")
        else()
            set(ASI_LIB_DIR "${ASI_SDK_DIR}/lib/mac")
        endif()
    else()
        set(ASI_LIB_DIR "${ASI_SDK_DIR}/lib/x64")
    endif()
    list(APPEND EXTRA_LINK_DIRS ${ASI_LIB_DIR})
    list(APPEND EXTRA_LIBS ASICamera2)
else()
    message(STATUS "❌ ASI SDK not found — ASI camera driver disabled")
endif()

# ═══════════════════════════════════════════════════════════════════
#  SVBony Camera SDK
# ═══════════════════════════════════════════════════════════════════
set(SVB_SDK_DIR "${SDKS_ROOT}/Svbony/macos/SVBCameraSDK")
set(SVB_INC "${SVB_SDK_DIR}/include")

if(ACHERNAR_ENABLE_VENDOR_DRIVERS AND NOT WIN32 AND
   EXISTS "${SVB_INC}/SVBCameraSDK.h")
    message(STATUS "✅ SVBony SDK found at ${SVB_SDK_DIR}")
    list(APPEND BRIDGE_SOURCES ${OLS_SRC}/svb_camera.cpp)
    list(APPEND EXTRA_INCLUDE_DIRS ${SVB_INC})
    add_definitions(-DWITH_SVB)

    # Select correct library for arch
    if(APPLE AND HOST_ARCH STREQUAL "arm64")
        set(SVB_LIB_DIR "${SVB_SDK_DIR}/lib/arm64")
    elseif(APPLE)
        set(SVB_LIB_DIR "${SVB_SDK_DIR}/lib/x64")
    else()
        set(SVB_LIB_DIR "${SVB_SDK_DIR}/lib/x64")
    endif()
    list(APPEND EXTRA_LINK_DIRS ${SVB_LIB_DIR})
    list(APPEND EXTRA_LIBS SVBCameraSDK)
else()
    message(STATUS "❌ SVBony SDK not found — SVBony camera driver disabled")
endif()

# ═══════════════════════════════════════════════════════════════════
#  ToupTek Camera SDK
# ═══════════════════════════════════════════════════════════════════
set(TOUP_SDK_DIR "${SDKS_ROOT}/touptek")

# ToupTek uses "inc/" not "include/", and "mac/" not "lib/"
find_file(TOUPCAM_HEADER toupcam.h
    PATHS "${TOUP_SDK_DIR}/inc" "${TOUP_SDK_DIR}/include" "${TOUP_SDK_DIR}"
    NO_DEFAULT_PATH
)

if(ACHERNAR_ENABLE_VENDOR_DRIVERS AND NOT WIN32 AND TOUPCAM_HEADER)
    get_filename_component(TOUP_INC "${TOUPCAM_HEADER}" DIRECTORY)
    message(STATUS "✅ ToupTek SDK found: ${TOUPCAM_HEADER}")
    list(APPEND BRIDGE_SOURCES ${OLS_SRC}/toup_camera.cpp)
    list(APPEND EXTRA_INCLUDE_DIRS ${TOUP_INC})
    add_definitions(-DWITH_TOUP)

    # ToupTek uses mac/ for macOS, linux/ for Linux
    if(APPLE)
        set(TOUP_LIB_DIR "${TOUP_SDK_DIR}/mac")
    else()
        set(TOUP_LIB_DIR "${TOUP_SDK_DIR}/linux/x64")
    endif()
    if(EXISTS "${TOUP_LIB_DIR}")
        list(APPEND EXTRA_LINK_DIRS ${TOUP_LIB_DIR})
        list(APPEND EXTRA_LIBS toupcam)
    endif()
else()
    message(STATUS "❌ ToupTek SDK not found — ToupTek camera driver disabled")
endif()

# ═══════════════════════════════════════════════════════════════════
#  INDI (optional)
# ═══════════════════════════════════════════════════════════════════
set(INDI_HINT_PREFIXES
    "$ENV{INDI_ROOT}"
    "$ENV{INDI_DIR}"
    "/opt/homebrew"
    "/usr/local"
    "/usr"
    "/opt/local"
)
list(REMOVE_ITEM INDI_HINT_PREFIXES "")

if(NOT WIN32)
    pkg_check_modules(LIBINDI QUIET libindi)
    pkg_check_modules(INDICLIENT QUIET indiclient)
endif()

if(NOT WIN32 AND NOT LIBINDI_FOUND AND NOT INDICLIENT_FOUND)
    find_path(INDI_INCLUDE_DIR
        NAMES libindi/baseclient.h
        PATHS ${INDI_HINT_PREFIXES}
        PATH_SUFFIXES include
    )
    find_library(INDI_CLIENT_LIBRARY
        NAMES indiclient libindiclient
        PATHS ${INDI_HINT_PREFIXES}
        PATH_SUFFIXES lib
    )
endif()

if(NOT WIN32 AND
   (LIBINDI_FOUND OR INDICLIENT_FOUND OR
    (INDI_INCLUDE_DIR AND INDI_CLIENT_LIBRARY)))
    message(STATUS "✅ INDI found — INDI mount/camera drivers enabled")
    list(APPEND BRIDGE_SOURCES ${OLS_SRC}/indi_mount.cpp)
    list(APPEND BRIDGE_SOURCES ${OLS_SRC}/indi_camera.cpp)
    add_definitions(-DWITH_INDI)

    if(LIBINDI_FOUND)
        foreach(INDI_PKG_INCLUDE_DIR IN LISTS LIBINDI_INCLUDE_DIRS)
            list(APPEND EXTRA_INCLUDE_DIRS ${INDI_PKG_INCLUDE_DIR})
            get_filename_component(INDI_PKG_INCLUDE_NAME "${INDI_PKG_INCLUDE_DIR}" NAME)
            if(INDI_PKG_INCLUDE_NAME STREQUAL "libindi")
                get_filename_component(INDI_PKG_INCLUDE_PARENT "${INDI_PKG_INCLUDE_DIR}" DIRECTORY)
                list(APPEND EXTRA_INCLUDE_DIRS ${INDI_PKG_INCLUDE_PARENT})
            endif()
        endforeach()
        list(APPEND EXTRA_LINK_DIRS ${LIBINDI_LIBRARY_DIRS})
        list(APPEND EXTRA_LIBS ${LIBINDI_LIBRARIES})
    elseif(INDICLIENT_FOUND)
        foreach(INDI_PKG_INCLUDE_DIR IN LISTS INDICLIENT_INCLUDE_DIRS)
            list(APPEND EXTRA_INCLUDE_DIRS ${INDI_PKG_INCLUDE_DIR})
            get_filename_component(INDI_PKG_INCLUDE_NAME "${INDI_PKG_INCLUDE_DIR}" NAME)
            if(INDI_PKG_INCLUDE_NAME STREQUAL "libindi")
                get_filename_component(INDI_PKG_INCLUDE_PARENT "${INDI_PKG_INCLUDE_DIR}" DIRECTORY)
                list(APPEND EXTRA_INCLUDE_DIRS ${INDI_PKG_INCLUDE_PARENT})
            endif()
        endforeach()
        list(APPEND EXTRA_LINK_DIRS ${INDICLIENT_LIBRARY_DIRS})
        list(APPEND EXTRA_LIBS ${INDICLIENT_LIBRARIES})
    else()
        list(APPEND EXTRA_INCLUDE_DIRS ${INDI_INCLUDE_DIR})
        list(APPEND EXTRA_LIBS ${INDI_CLIENT_LIBRARY})
    endif()
else()
    message(STATUS "ℹ️  INDI not found — INDI drivers disabled")
endif()

# ═══════════════════════════════════════════════════════════════════
#  UVC (USB video cameras / webcams, optional — requires libuvc)
# ═══════════════════════════════════════════════════════════════════
if(WIN32)
    # Use Media Foundation through the standard Windows camera driver. This
    # deliberately avoids libusb/WinUSB and therefore does not require Zadig
    # or break the camera for SharpCap and other astronomy applications.
    set(WINDOWS_MF_CAMERA_SOURCE "${BRIDGE_SRC}/windows_mf_camera.cpp")
    if(EXISTS "${WINDOWS_MF_CAMERA_SOURCE}")
        message(STATUS "Windows Media Foundation camera driver enabled")
        list(APPEND BRIDGE_SOURCES ${WINDOWS_MF_CAMERA_SOURCE})
        list(APPEND EXTRA_LIBS
            mfplat mf mfreadwrite mfuuid strmiids ole32 oleaut32)
        add_definitions(-DWITH_UVC)
    else()
        message(STATUS
            "Windows Media Foundation source not present - USB video driver disabled")
    endif()
else()
    pkg_check_modules(LIBUVC QUIET libuvc)
endif()
if(NOT WIN32 AND LIBUVC_FOUND)
    message(STATUS "✅ libuvc found — UVC camera driver enabled")
    list(APPEND BRIDGE_SOURCES ${OLS_SRC}/uvc_camera.cpp)
    list(APPEND EXTRA_INCLUDE_DIRS ${LIBUVC_INCLUDE_DIRS})
    list(APPEND EXTRA_LINK_DIRS ${LIBUVC_LIBRARY_DIRS})
    list(APPEND EXTRA_LIBS ${LIBUVC_LIBRARIES})
    add_definitions(-DWITH_UVC)
elseif(NOT WIN32)
    message(STATUS "ℹ️  libuvc not found — UVC camera driver disabled")
endif()

# ═══════════════════════════════════════════════════════════════════
#  INDIGO (optional — requires INDIGO libraries)
# ═══════════════════════════════════════════════════════════════════
if(NOT WIN32)
    find_file(INDIGO_HEADER indigo/indigo_bus.h
        PATHS "/opt/homebrew/include" "/usr/local/include" "/usr/include"
    )
endif()
if(NOT WIN32 AND INDIGO_HEADER)
    get_filename_component(INDIGO_INC_DIR "${INDIGO_HEADER}" DIRECTORY)  # .../indigo/
    get_filename_component(INDIGO_INC_DIR "${INDIGO_INC_DIR}" DIRECTORY) # .../include/
    message(STATUS "✅ INDIGO found — INDIGO camera driver enabled (${INDIGO_INC_DIR})")
    list(APPEND BRIDGE_SOURCES ${OLS_SRC}/indigo_camera.cpp)
    list(APPEND EXTRA_INCLUDE_DIRS ${INDIGO_INC_DIR})
    list(APPEND EXTRA_LIBS indigo)
    add_definitions(-DWITH_INDIGO)
else()
    message(STATUS "ℹ️  INDIGO not found — INDIGO camera driver disabled")
endif()

# ═══════════════════════════════════════════════════════════════════
#  LibRaw (RAW decoding — REQUIRED for any RAW-shooting DSLR/mirrorless)
# ═══════════════════════════════════════════════════════════════════
#
# This is not merely "optional" the way a vendor SDK is. Canon bodies set to
# RAW (CR2 on the T3i/600D and T6s/760D, CR3 on the R50/M50) and Panasonic
# bodies set to RW2 emit *only* a raw file per shutter actuation. Without a
# decoder, `GPFile::to_mat()` throws "built without libraw support", so
# `formats()` enumerates nothing for those bodies and the camera appears to
# work right up until it produces zero usable frames. Compiling gphoto2 while
# leaving LibRaw out is therefore the single most misleading build state we
# can ship, and it is what the tree did before this block existed.
#
# Probe pkg-config first (LibRaw installs libraw.pc), then fall back to a
# header search so source builds and non-pkg-config installs still work.
if(NOT WIN32)
    pkg_check_modules(LIBRAW QUIET libraw)
endif()

if(NOT WIN32 AND NOT LIBRAW_FOUND)
    find_file(LIBRAW_HEADER libraw/libraw.h
        PATHS "/opt/homebrew/include" "/usr/local/include" "/usr/include" "/opt/local/include"
    )
    find_library(LIBRAW_LIBRARY
        NAMES raw raw_r libraw libraw_r
        PATHS "/opt/homebrew/lib" "/usr/local/lib" "/usr/lib" "/opt/local/lib"
    )
endif()

if(NOT WIN32 AND (LIBRAW_FOUND OR (LIBRAW_HEADER AND LIBRAW_LIBRARY)))
    if(LIBRAW_FOUND)
        message(STATUS "✅ LibRaw found via pkg-config ${LIBRAW_VERSION} — RAW decoding enabled")
        list(APPEND EXTRA_INCLUDE_DIRS ${LIBRAW_INCLUDE_DIRS})
        list(APPEND EXTRA_LINK_DIRS ${LIBRAW_LIBRARY_DIRS})
        list(APPEND EXTRA_LIBS ${LIBRAW_LIBRARIES})
    else()
        # LIBRAW_HEADER is .../include/libraw/libraw.h; `#include <libraw/libraw.h>`
        # needs the grandparent ("include/") on the search path.
        get_filename_component(LIBRAW_INC_DIR "${LIBRAW_HEADER}" DIRECTORY)  # .../libraw/
        get_filename_component(LIBRAW_INC_DIR "${LIBRAW_INC_DIR}" DIRECTORY) # .../include/
        message(STATUS "✅ LibRaw found at ${LIBRAW_LIBRARY} — RAW decoding enabled")
        list(APPEND EXTRA_INCLUDE_DIRS ${LIBRAW_INC_DIR})
        list(APPEND EXTRA_LIBS ${LIBRAW_LIBRARY})
    endif()
    add_definitions(-DWITH_LIBRAW)
    set(OLS_HAVE_LIBRAW TRUE)
else()
    set(OLS_HAVE_LIBRAW FALSE)
    message(STATUS "ℹ️  LibRaw not found — RAW (CR2/CR3/RW2/NEF/ARW) decoding disabled")
endif()

# ═══════════════════════════════════════════════════════════════════
#  GPhoto2 (DSLR / mirrorless support, optional)
# ═══════════════════════════════════════════════════════════════════
if(NOT WIN32)
    find_file(GPHOTO2_HEADER gphoto2/gphoto2.h
        PATHS "/opt/homebrew/include" "/usr/local/include" "/usr/include"
    )
endif()
if(NOT WIN32 AND GPHOTO2_HEADER)
    # GPHOTO2_HEADER is e.g. /opt/homebrew/.../include/gphoto2/gphoto2.h
    # We need the grandparent dir (the "include/" dir) on the include path
    get_filename_component(GPHOTO2_INC_DIR "${GPHOTO2_HEADER}" DIRECTORY)  # .../gphoto2/
    get_filename_component(GPHOTO2_INC_DIR "${GPHOTO2_INC_DIR}" DIRECTORY) # .../include/
    message(STATUS "✅ GPhoto2 found — DSLR camera driver enabled (${GPHOTO2_INC_DIR})")
    list(APPEND BRIDGE_SOURCES ${OLS_SRC}/gphoto2_camera.cpp)
    list(APPEND EXTRA_INCLUDE_DIRS ${GPHOTO2_INC_DIR})
    add_definitions(-DWITH_GPHOTO2)
    list(APPEND EXTRA_LIBS gphoto2 gphoto2_port)

    # Loud, actionable warning rather than a silent half-capable build: the
    # driver will connect to the body, report a name, and then fail to produce
    # a single frame if the camera is not set to JPEG.
    if(NOT OLS_HAVE_LIBRAW)
        message(WARNING
            "GPhoto2 is enabled but LibRaw is NOT. RAW-only capture will fail at "
            "runtime — only cameras set to JPEG will stream. Install LibRaw "
            "(macOS: `brew install libraw`, Debian/Ubuntu: `apt install libraw-dev`) "
            "and re-run cmake.")
    endif()
else()
    message(STATUS "ℹ️  GPhoto2 not found — DSLR camera driver disabled")
endif()

# ══════════════════════════════════════════════════════════════════
#  Build the native bridge
# ═══════════════════════════════════════════════════════════════════
# A DLL is intentional on Windows: CMake can then own the complete native link
# graph (OpenCV, TIFF and their vcpkg dependencies), while Cargo links only the
# generated import library.  Unix keeps the historical static bridge because
# build.rs already emits the corresponding pkg-config link flags there.
if(WIN32)
    add_library(ols_bridge SHARED ${BRIDGE_SOURCES})
    set_target_properties(ols_bridge PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
else()
    add_library(ols_bridge STATIC ${BRIDGE_SOURCES})
endif()

target_include_directories(ols_bridge PRIVATE
    ${BRIDGE_INC}
    ${OLS_INC}
    ${ACHERNAR_OPENCV_INCLUDE_DIRS}
    ${ACHERNAR_TIFF_INCLUDE_DIRS}
    # Shims MUST come before system paths so our cppcms/booster
    # replacements are found instead of any real installations
    ${SHIMS_DIR}
    ${EXTERNAL_DIR}
    # Vendor SDK includes
    ${EXTRA_INCLUDE_DIRS}
)

target_link_directories(ols_bridge PRIVATE ${EXTRA_LINK_DIRS})

target_link_libraries(ols_bridge PRIVATE
    ${ACHERNAR_OPENCV_LIBS}
    ${ACHERNAR_TIFF_LIBS}
    ${EXTRA_LIBS}
)

if(WIN32)
    # Alpaca discovery and HTTP transport use Winsock and adapter enumeration.
    target_link_libraries(ols_bridge PRIVATE ws2_32 iphlpapi)

    # Put vcpkg runtime DLLs alongside ols_bridge.dll. build.rs subsequently
    # stages this directory beside Cargo executables and test binaries.
    add_custom_command(TARGET ols_bridge POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_RUNTIME_DLLS:ols_bridge>
                $<TARGET_FILE_DIR:ols_bridge>
        COMMAND_EXPAND_LISTS
        VERBATIM)
endif()

target_compile_definitions(ols_bridge PRIVATE
    OLS_BRIDGE_NO_CPPCMS=1
)

# Suppress warnings in third-party code without passing GCC flags to cl.exe.
if(MSVC)
    target_compile_options(ols_bridge PRIVATE /W3 /EHsc /permissive- /utf-8)
    target_compile_definitions(ols_bridge PRIVATE
        NOMINMAX
        WIN32_LEAN_AND_MEAN
        _USE_MATH_DEFINES
        _CRT_SECURE_NO_WARNINGS)
else()
    target_compile_options(ols_bridge PRIVATE
        -Wno-deprecated-declarations
        -Wno-unused-parameter
        -Wno-sign-compare
        -Wno-unused-variable
        -Wno-unused-function
        -Wno-missing-field-initializers)
endif()

# ─── Status Summary ───
message(STATUS "")
message(STATUS "═══ OLS Bridge Build Summary ═══")
message(STATUS "  Core:     Camera (Sim, Alpaca, WatchDir), Mount (Alpaca), Focuser (Alpaca)")
get_directory_property(COMPILE_DEFS COMPILE_DEFINITIONS)
if("WITH_ASI" IN_LIST COMPILE_DEFS)
    message(STATUS "  Camera:   ✅ ASI (ZWO)")
endif()
if("WITH_SVB" IN_LIST COMPILE_DEFS)
    message(STATUS "  Camera:   ✅ SVBony")
endif()
if("WITH_TOUP" IN_LIST COMPILE_DEFS)
    message(STATUS "  Camera:   ✅ ToupTek")
endif()
if("WITH_UVC" IN_LIST COMPILE_DEFS)
    message(STATUS "  Camera:   ✅ UVC (USB video)")
endif()
if("WITH_GPHOTO2" IN_LIST COMPILE_DEFS)
    message(STATUS "  Camera:   ✅ GPhoto2 (DSLR / mirrorless)")
endif()
if("WITH_LIBRAW" IN_LIST COMPILE_DEFS)
    message(STATUS "  Decode:   ✅ LibRaw (CR2/CR3/RW2/NEF/ARW)")
else()
    message(STATUS "  Decode:   ❌ LibRaw missing — RAW capture unavailable")
endif()
if("WITH_INDI" IN_LIST COMPILE_DEFS)
    message(STATUS "  Protocol: ✅ INDI (camera + mount)")
endif()
if("WITH_INDIGO" IN_LIST COMPILE_DEFS)
    message(STATUS "  Protocol: ✅ INDIGO (camera)")
endif()
message(STATUS "════════════════════════════════")
