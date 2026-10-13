#pragma once
#include <limits>
#include <stdexcept>
#include <string>

namespace ols {
// Explicit, round-trippable device selection. URL fragments are local options
// and never sent to the HTTP server. A plain URL retains legacy first-device
// selection; an explicit selector must never silently fall back to that device.
struct AlpacaSelection {
    AlpacaSelection(std::string const &url = "", int number = -1)
        : base_url(url), device_number(number) {}
    std::string base_url;
    int device_number;
};

inline AlpacaSelection parse_alpaca_selection(std::string const &option)
{
    auto hash = option.find('#');
    if(hash == std::string::npos)
        return {option, -1};
    std::string suffix = option.substr(hash);
    std::string prefix = "#device=";
    if(hash == 0 || option.substr(0, hash) == "auto" ||
       suffix.compare(0, prefix.size(), prefix) != 0 || suffix.size() == prefix.size())
        throw std::invalid_argument("Invalid Alpaca device selector; use an explicit server URL#device=N");
    unsigned int number = 0;
    for(char digit : suffix.substr(prefix.size())) {
        if(digit < '0' || digit > '9' ||
           number > (static_cast<unsigned int>(std::numeric_limits<int>::max()) - (digit - '0')) / 10)
            throw std::invalid_argument("Alpaca device number must be a nonnegative int32");
        number = number * 10 + (digit - '0');
    }
    return {option.substr(0, hash), static_cast<int>(number)};
}

inline bool matches_alpaca_device(AlpacaSelection const &selection, int number)
{
    return number >= 0 && (selection.device_number < 0 || selection.device_number == number);
}
}
