cmake_minimum_required(VERSION 3.16)
project(ols_bridge LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

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
find_package(PkgConfig REQUIRED)
pkg_check_modules(OPENCV REQUIRED opencv4)
pkg_check_modules(TIFF REQUIRED libtiff-4)

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

    # Watch-directory pseudo-camera (no vendor SDK / external libs needed)
    ${OLS_SRC}/wdir_camera.cpp
)

set(EXTRA_INCLUDE_DIRS "")
set(EXTRA_LINK_DIRS "")
set(EXTRA_LIBS "")

# ═══════════════════════════════════════════════════════════════════
#  ASI (ZWO) Camera SDK
# ═══════════════════════════════════════════════════════════════════
set(ASI_SDK_DIR "${SDKS_ROOT}/ASI/ASI_Camera_SDK/ASI_linux_mac_SDK_V1.41")
set(ASI_INC "${ASI_SDK_DIR}/include")

if(EXISTS "${ASI_INC}/ASICamera2.h")
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

if(EXISTS "${SVB_INC}/SVBCameraSDK.h")
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

if(TOUPCAM_HEADER)
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
#  INDI (Linux only, optional)
# ═══════════════════════════════════════════════════════════════════
if(EXISTS "/usr/include/libindi" OR EXISTS "/usr/local/include/libindi")
    message(STATUS "✅ INDI found — INDI mount/camera drivers enabled")
    list(APPEND BRIDGE_SOURCES ${OLS_SRC}/indi_mount.cpp)
    list(APPEND BRIDGE_SOURCES ${OLS_SRC}/indi_camera.cpp)
    add_definitions(-DWITH_INDI)
    if(EXISTS "/usr/include/libindi")
        list(APPEND EXTRA_INCLUDE_DIRS /usr/include/libindi)
    else()
        list(APPEND EXTRA_INCLUDE_DIRS /usr/local/include/libindi)
    endif()
    list(APPEND EXTRA_LIBS indiclient)
else()
    message(STATUS "ℹ️  INDI not found — INDI drivers disabled (Linux only)")
endif()

# ═══════════════════════════════════════════════════════════════════
#  UVC (USB video cameras / webcams, optional — requires libuvc)
# ═══════════════════════════════════════════════════════════════════
pkg_check_modules(LIBUVC libuvc)
if(LIBUVC_FOUND)
    message(STATUS "✅ libuvc found — UVC camera driver enabled")
    list(APPEND BRIDGE_SOURCES ${OLS_SRC}/uvc_camera.cpp)
    list(APPEND EXTRA_INCLUDE_DIRS ${LIBUVC_INCLUDE_DIRS})
    list(APPEND EXTRA_LINK_DIRS ${LIBUVC_LIBRARY_DIRS})
    list(APPEND EXTRA_LIBS ${LIBUVC_LIBRARIES})
    add_definitions(-DWITH_UVC)
else()
    message(STATUS "ℹ️  libuvc not found — UVC camera driver disabled")
endif()

# ═══════════════════════════════════════════════════════════════════
#  INDIGO (optional — requires INDIGO libraries)
# ═══════════════════════════════════════════════════════════════════
find_file(INDIGO_HEADER indigo/indigo_bus.h
    PATHS "/opt/homebrew/include" "/usr/local/include" "/usr/include"
)
if(INDIGO_HEADER)
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
#  GPhoto2 (DSLR support, optional)
# ═══════════════════════════════════════════════════════════════════
find_file(GPHOTO2_HEADER gphoto2/gphoto2.h
    PATHS "/opt/homebrew/include" "/usr/local/include" "/usr/include"
)
if(GPHOTO2_HEADER)
    # GPHOTO2_HEADER is e.g. /opt/homebrew/.../include/gphoto2/gphoto2.h
    # We need the grandparent dir (the "include/" dir) on the include path
    get_filename_component(GPHOTO2_INC_DIR "${GPHOTO2_HEADER}" DIRECTORY)  # .../gphoto2/
    get_filename_component(GPHOTO2_INC_DIR "${GPHOTO2_INC_DIR}" DIRECTORY) # .../include/
    message(STATUS "✅ GPhoto2 found — DSLR camera driver enabled (${GPHOTO2_INC_DIR})")
    list(APPEND BRIDGE_SOURCES ${OLS_SRC}/gphoto2_camera.cpp)
    list(APPEND EXTRA_INCLUDE_DIRS ${GPHOTO2_INC_DIR})
    add_definitions(-DWITH_GPHOTO2)
    list(APPEND EXTRA_LIBS gphoto2 gphoto2_port)
else()
    message(STATUS "ℹ️  GPhoto2 not found — DSLR camera driver disabled")
endif()

# ═══════════════════════════════════════════════════════════════════
#  Build the static library
# ═══════════════════════════════════════════════════════════════════
add_library(ols_bridge STATIC ${BRIDGE_SOURCES})

target_include_directories(ols_bridge PRIVATE
    ${BRIDGE_INC}
    ${OLS_INC}
    ${OPENCV_INCLUDE_DIRS}
    ${TIFF_INCLUDE_DIRS}
    # Shims MUST come before system paths so our cppcms/booster
    # replacements are found instead of any real installations
    ${SHIMS_DIR}
    ${EXTERNAL_DIR}
    # Vendor SDK includes
    ${EXTRA_INCLUDE_DIRS}
)

target_link_directories(ols_bridge PRIVATE ${EXTRA_LINK_DIRS})

target_link_libraries(ols_bridge
    ${OPENCV_LIBRARIES}
    ${TIFF_LIBRARIES}
    ${EXTRA_LIBS}
)

target_compile_definitions(ols_bridge PRIVATE
    OLS_BRIDGE_NO_CPPCMS=1
)

# Suppress warnings in third-party code
target_compile_options(ols_bridge PRIVATE
    -Wno-deprecated-declarations
    -Wno-unused-parameter
    -Wno-sign-compare
    -Wno-unused-variable
    -Wno-unused-function
    -Wno-missing-field-initializers
)

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
    message(STATUS "  Camera:   ✅ GPhoto2 (DSLR)")
endif()
if("WITH_INDI" IN_LIST COMPILE_DEFS)
    message(STATUS "  Protocol: ✅ INDI (camera + mount)")
endif()
if("WITH_INDIGO" IN_LIST COMPILE_DEFS)
    message(STATUS "  Protocol: ✅ INDIGO (camera)")
endif()
message(STATUS "════════════════════════════════")
