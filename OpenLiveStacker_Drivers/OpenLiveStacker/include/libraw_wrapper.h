#pragma once
#include "camera.h"
#include "untouched_raw.h"
#include <libraw/libraw.h>
namespace ols {
    
    // Both legacy file/buffer callers and the dedicated bridge use the same
    // untouched unpack/crop implementation. Display stretching belongs later.
    inline std::pair<cv::Mat,CamBayerType> load_libraw(LibRaw &raw,int h=-1,int w=-1,int bin=1)
    {
        auto plane = untouched_raw::unpack(raw,w,h,bin);
        cv::Mat view(plane.region.height,plane.region.width,CV_16UC1,plane.pixels.data());
        const auto bayer = plane.pattern.empty() ? bayer_na : bayer_type_from_str(plane.pattern);
        return std::make_pair(view.clone(),bayer);
    }

    inline std::pair<cv::Mat,CamBayerType> load_libraw(std::string const &path,int h=-1,int w=-1,int bin=1)
    {
        LibRaw raw;
        int code=0;
        if((code = raw.open_file(path.c_str()))!=LIBRAW_SUCCESS)
            throw std::runtime_error("Failed to open raw file " + path + ": " + libraw_strerror(code));
        return load_libraw(raw,h,w,bin);
    }
    inline std::pair<cv::Mat,CamBayerType> load_libraw(char *data,size_t size,int h=-1,int w=-1,int bin=1)
    {
        LibRaw raw;
        int code=0;
        if((code = raw.open_buffer(data,size))!=LIBRAW_SUCCESS) {
            throw std::runtime_error("Failed to open raw data : " + std::string(libraw_strerror(code)));
        }
        return load_libraw(raw,h,w,bin);
    }
}
