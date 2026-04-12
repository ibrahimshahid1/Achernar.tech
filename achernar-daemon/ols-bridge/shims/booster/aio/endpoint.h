/**
 * booster/aio/endpoint.h shim
 * Minimal replacement for booster::aio::endpoint used in Alpaca discovery.
 */
#pragma once
#include <string>
#include <cstring>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <netinet/in.h>

namespace booster {
namespace aio {

    enum { pf_inet = AF_INET, sock_datagram = SOCK_DGRAM };

    class endpoint {
    public:
        endpoint() { memset(&addr_, 0, sizeof(addr_)); }
        endpoint(std::string const &ip, int port) {
            memset(&addr_, 0, sizeof(addr_));
            addr_.sin_family = AF_INET;
            addr_.sin_port = htons(port);
            inet_pton(AF_INET, ip.c_str(), &addr_.sin_addr);
        }
        std::pair<struct sockaddr *, socklen_t> raw() {
            return {(struct sockaddr *)&addr_, sizeof(addr_)};
        }
        void raw(struct sockaddr *sa, socklen_t len) {
            memcpy(&addr_, sa, std::min(len, (socklen_t)sizeof(addr_)));
        }
        std::string ip() const {
            char buf[INET_ADDRSTRLEN + 1] = {};
            inet_ntop(AF_INET, &addr_.sin_addr, buf, INET_ADDRSTRLEN);
            return buf;
        }
    private:
        struct sockaddr_in addr_;
    };

} // namespace aio
} // namespace booster
