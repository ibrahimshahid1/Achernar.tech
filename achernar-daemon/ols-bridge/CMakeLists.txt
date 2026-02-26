cmake_minimum_required(VERSION 3.16)
project(ols_bridge CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ─── OLS source directory ───
set(OLS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../OpenLiveStacker_Drivers/OpenLiveStacker")
set(OLS_INCLUDE "${OLS_ROOT}/include")
set(OLS_SRC "${OLS_ROOT}/src")

# ─── Find OpenCV ───
find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc)

# ─── Find libtiff ───
find_library(LIBTIFF tiff)
if(NOT LIBTIFF)
    message(FATAL_ERROR "libtiff not found. Install with: brew install libtiff")
endif()
# Find tiff include directory
find_path(TIFF_INCLUDE_DIR tiffio.h
    HINTS /opt/homebrew/include /usr/local/include /usr/include
)

# ─── Build the bridge as a static library ───
add_library(ols_bridge STATIC
    # Bridge code
    src/ols_bridge.cpp
    # Our own minimal camera base (no booster/cppcms dependency!)
    src/camera_base.cpp
    # OLS utility sources
    ${OLS_SRC}/tiffmat.cpp
    ${OLS_SRC}/util.cpp
)

target_include_directories(ols_bridge PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OLS_INCLUDE}
)

# Add tiff include if found
if(TIFF_INCLUDE_DIR)
    target_include_directories(ols_bridge PRIVATE ${TIFF_INCLUDE_DIR})
endif()

target_link_libraries(ols_bridge PUBLIC
    ${OpenCV_LIBS}
    ${LIBTIFF}
)

# Suppress warnings from OLS code
target_compile_options(ols_bridge PRIVATE
    -Wno-unused-parameter
    -Wno-sign-compare
    -Wno-unused-variable
)

# We define ACHERNAR_BRIDGE to allow conditional compilation in bridge code
target_compile_definitions(ols_bridge PRIVATE ACHERNAR_BRIDGE)

# ─── Install target for the Rust build.rs to find ───
install(TARGETS ols_bridge
    ARCHIVE DESTINATION lib
)
install(FILES include/ols_bridge.h
    DESTINATION include
)
