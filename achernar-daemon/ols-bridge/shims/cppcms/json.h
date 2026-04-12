/**
 * cppcms/json.h compatibility shim
 *
 * Provides the cppcms::json::value and cppcms::json::array API surface
 * that OLS Alpaca client/mount/camera code uses, backed by nlohmann/json.
 * This lets us compile OLS sources without the real cppcms library.
 */
#pragma once
#include "../../external/json.hpp"
#include <string>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace cppcms {
namespace json {

    // Forward declaration
    class value;

    // cppcms::json::array is just std::vector<value>
    typedef std::vector<value> array;

    inline nlohmann::json object() {
        return nlohmann::json::object();
    }

    class value {
    public:
        value() : j_(nlohmann::json()) {}
        value(nlohmann::json const &j) : j_(j) {}
        value(value const &) = default;
        value &operator=(value const &) = default;
        value(value &&) = default;
        value &operator=(value &&) = default;

        // Subscript access - returns a new value wrapping the child
        value operator[](std::string const &key) const {
            if (j_.contains(key))
                return value(j_[key]);
            return value();
        }

        value operator[](size_t index) const {
            if (j_.is_array() && index < j_.size())
                return value(j_[index]);
            return value();
        }

        // get<T>(key) - typed access by key
        template<typename T>
        T get(std::string const &key) const {
            if (!j_.contains(key))
                throw std::runtime_error("Key not found: " + key);
            return j_[key].get<T>();
        }

        // get(key, default) - access with default value (string overload)
        std::string get(std::string const &key, std::string const &def) const {
            if (j_.contains(key) && !j_[key].is_null())
                return j_[key].get<std::string>();
            return def;
        }

        // get(key, default) - access with default value (int overload)
        int get(std::string const &key, int def) const {
            if (j_.contains(key) && !j_[key].is_null())
                return j_[key].get<int>();
            return def;
        }

        // .array() - convert to cppcms::json::array
        cppcms::json::array array() const {
            cppcms::json::array result;
            if (j_.is_array()) {
                for (auto const &item : j_) {
                    result.push_back(value(item));
                }
            }
            return result;
        }

        // Load from string range (used by alpaca_client)
        bool load(char const *begin, char const *end, bool, int *lineno = nullptr) {
            try {
                std::string s(begin, end);
                j_ = nlohmann::json::parse(s);
                return true;
            } catch (...) {
                if (lineno) *lineno = 0;
                return false;
            }
        }

        // Load from stream
        bool load(std::istream &in, bool) {
            try {
                j_ = nlohmann::json::parse(in);
                return true;
            } catch (...) {
                return false;
            }
        }

        // Access underlying nlohmann::json
        nlohmann::json const &raw() const { return j_; }

        // Stream output
        friend std::ostream &operator<<(std::ostream &os, value const &v) {
            os << v.j_.dump();
            return os;
        }

    private:
        nlohmann::json j_;
    };

} // namespace json
} // namespace cppcms
