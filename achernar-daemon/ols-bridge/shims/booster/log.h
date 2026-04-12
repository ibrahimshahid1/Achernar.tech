/**
 * booster/log.h shim
 * Replaces BOOSTER_INFO/BOOSTER_ERROR macros with stderr logging.
 * Used by OLS stacker, mount_ctl, etc.
 */
#pragma once
#include <iostream>
#include <sstream>

// Swallow log messages - they go to a NullStream that discards output.
// If you want actual logging, redirect to stderr.
namespace booster_shim {
    class NullStream {
    public:
        template<typename T>
        NullStream &operator<<(T const &) { return *this; }
        NullStream &operator<<(std::ostream&(*)(std::ostream&)) { return *this; }
    };
}

#ifndef BOOSTER_INFO
#define BOOSTER_INFO(tag) booster_shim::NullStream()
#endif
#ifndef BOOSTER_ERROR
#define BOOSTER_ERROR(tag) std::cerr << "[ERROR:" << tag << "] "
#endif
#ifndef BOOSTER_WARNING
#define BOOSTER_WARNING(tag) booster_shim::NullStream()
#endif
