/**
 * booster/aio/basic_socket.h shim
 * Minimal replacement for booster::aio::basic_socket used in Alpaca discovery.
 */
#pragma once
#include <sys/socket.h>
#include <unistd.h>
#include <stdexcept>

namespace booster {
namespace aio {

    class basic_socket {
    public:
        basic_socket() : fd_(-1) {}
        ~basic_socket() { if (fd_ >= 0) ::close(fd_); }

        void open(int domain, int type) {
            fd_ = ::socket(domain, type, 0);
            if (fd_ < 0) throw std::runtime_error("Failed to create socket");
        }

        int native() const { return fd_; }
    private:
        int fd_;
    };

} // namespace aio
} // namespace booster
