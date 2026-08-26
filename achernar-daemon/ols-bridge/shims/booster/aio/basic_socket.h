/**
 * booster/aio/basic_socket.h shim
 * Minimal replacement for booster::aio::basic_socket used in Alpaca discovery.
 */
#pragma once
#ifdef _WIN32
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <winsock2.h>
#  include <ws2tcpip.h>
#else
#  include <sys/socket.h>
#  include <unistd.h>
#endif

#include <mutex>
#include <stdexcept>
#include <string>

namespace booster {
namespace aio {

    class basic_socket {
    public:
        basic_socket() : fd_(invalid_socket()) {
#ifdef _WIN32
            // Winsock has process-wide startup semantics.  Keep initialization
            // alive for the daemon's lifetime rather than balancing it for
            // every short-lived Alpaca discovery socket.
            static std::once_flag initialized;
            static int startup_error = 0;
            std::call_once(initialized, [] {
                WSADATA data{};
                startup_error = ::WSAStartup(MAKEWORD(2, 2), &data);
            });
            if (startup_error != 0)
                throw std::runtime_error("Failed to initialize Winsock: " +
                                         std::to_string(startup_error));
#endif
        }

        ~basic_socket() {
            if (fd_ == invalid_socket())
                return;
#ifdef _WIN32
            ::closesocket(fd_);
#else
            ::close(fd_);
#endif
        }

        void open(int domain, int type) {
            fd_ = ::socket(domain, type, 0);
            if (fd_ == invalid_socket())
                throw std::runtime_error("Failed to create socket");
        }

        auto native() const { return fd_; }
    private:
#ifdef _WIN32
        using native_socket = SOCKET;
        static constexpr native_socket invalid_socket() { return INVALID_SOCKET; }
#else
        using native_socket = int;
        static constexpr native_socket invalid_socket() { return -1; }
#endif
        native_socket fd_;
    };

} // namespace aio
} // namespace booster
