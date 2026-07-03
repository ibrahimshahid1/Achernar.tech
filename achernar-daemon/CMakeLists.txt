cmake_minimum_required(VERSION 3.16)
project(achernar_daemon_index LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(PkgConfig REQUIRED)
pkg_check_modules(OPENCV REQUIRED opencv4)
pkg_check_modules(TIFF REQUIRED libtiff-4)

set(PROJECT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/..")
set(OLS_ROOT "${PROJECT_ROOT}/OpenLiveStacker_Drivers/OpenLiveStacker")
set(OLS_INC "${OLS_ROOT}/include")

include_directories(ols-bridge/external)
include_directories(ols-bridge/include)
include_directories(ols-bridge/shims)
include_directories(ols-bridge/shims/booster)
include_directories(ols-bridge/shims/booster/aio)
include_directories(ols-bridge/shims/cppcms)
include_directories(${OLS_INC})
include_directories(${OPENCV_INCLUDE_DIRS})
include_directories(${TIFF_INCLUDE_DIRS})
include_directories(target)
include_directories(target/debug)
include_directories(target/debug/build)
include_directories(target/debug/build/libusb1-sys-33457c577c407448)
include_directories(target/debug/build/libusb1-sys-33457c577c407448/out)
include_directories(target/debug/build/libusb1-sys-33457c577c407448/out/include)

add_library(achernar_daemon_ols_bridge_index STATIC
        ols-bridge/external/json.hpp
        ols-bridge/include/ols_bridge.h
        ols-bridge/shims/booster/aio/basic_socket.h
        ols-bridge/shims/booster/aio/endpoint.h
        ols-bridge/shims/booster/log.h
        ols-bridge/shims/booster/regex.h
        ols-bridge/shims/cppcms/json.h
        ols-bridge/src/camera_base.cpp
        ols-bridge/src/ols_bridge.cpp
        target/debug/build/libusb1-sys-33457c577c407448/out/include/config.h
        target/debug/build/libusb1-sys-33457c577c407448/out/include/libusb.h)
