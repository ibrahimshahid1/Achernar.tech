/**
 * booster/regex.h shim
 *
 * Maps the small subset of booster::regex used by OLS onto <regex> from the
 * C++ standard library, so sources that include <booster/regex.h> compile
 * against the bridge without the real Booster library.
 *
 * (The watch-directory camera includes this header but does not actually use
 * booster::regex; the aliases below keep other potential users working too.)
 */
#pragma once
#include <regex>

namespace booster {
    using regex = std::regex;
    using smatch = std::smatch;
    using cmatch = std::cmatch;
    using std::regex_search;
    using std::regex_match;
    using std::regex_replace;
    namespace regex_constants = std::regex_constants;
}
