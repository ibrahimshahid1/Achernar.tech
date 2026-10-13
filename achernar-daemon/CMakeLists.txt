cmake_minimum_required(VERSION 3.16)
project(achernar_daemon_index LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# opencv@4 is keg-only on Homebrew; same probe as ols-bridge/CMakeLists.txt.
if(APPLE)
    find_program(BREW_EXECUTABLE brew)
    if(BREW_EXECUTABLE)
        foreach(KEG_FORMULA "opencv@4" "libtiff")
            execute_process(
                COMMAND ${BREW_EXECUTABLE} --prefix ${KEG_FORMULA}
                OUTPUT_VARIABLE KEG_PREFIX
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET)
            if(KEG_PREFIX AND EXISTS "${KEG_PREFIX}/lib/pkgconfig")
                set(ENV{PKG_CONFIG_PATH} "${KEG_PREFIX}/lib/pkgconfig:$ENV{PKG_CONFIG_PATH}")
            endif()
        endforeach()
    endif()
endif()

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

add_library(achernar_daemon_ols_bridge_index STATIC
        ols-bridge/external/json.hpp
        ols-bridge/include/ols_bridge.h
        ols-bridge/shims/booster/aio/basic_socket.h
        ols-bridge/shims/booster/aio/endpoint.h
        ols-bridge/shims/booster/log.h
        ols-bridge/shims/booster/regex.h
        ols-bridge/shims/cppcms/json.h
        ols-bridge/src/camera_base.cpp
        ols-bridge/src/ols_bridge.cpp)
