#include "alpaca_selection.h"
#include <cassert>
#include <iostream>
#include <vector>

int main()
{
    auto plain = ols::parse_alpaca_selection("http://127.0.0.1:11111");
    assert(plain.device_number == -1);
    assert(ols::matches_alpaca_device(plain, 0));
    assert(ols::matches_alpaca_device(plain, 7));
    auto chosen = ols::parse_alpaca_selection("http://127.0.0.1:11111#device=7");
    assert(chosen.base_url == plain.base_url);
    std::vector<int> available{0, 3, 7};
    std::vector<int> matches;
    for(int number : available)
        if(ols::matches_alpaca_device(chosen, number)) matches.push_back(number);
    assert(matches == std::vector<int>{7});
    auto missing = ols::parse_alpaca_selection("http://127.0.0.1:11111#device=2");
    for(int number : available) assert(!ols::matches_alpaca_device(missing, number));
    assert(ols::parse_alpaca_selection("http://localhost#device=0").device_number == 0);
    assert(ols::parse_alpaca_selection("http://localhost#device=2147483647").device_number == 2147483647);
    for(auto const &bad : {"http://localhost#device=", "http://localhost#device=-1",
         "http://localhost#device=2147483648", "http://localhost#device=7x",
         "http://localhost#device=3#device=4", "http://localhost#7",
         "auto#device=1", "#device=1"}) {
        bool rejected = false;
        try { ols::parse_alpaca_selection(bad); }
        catch(std::invalid_argument const &) { rejected = true; }
        assert(rejected);
    }
    std::cout << "Alpaca device selector tests passed\n";
}
