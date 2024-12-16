/*
 *
 * Copyright 2021-2024 Software Radio Systems Limited
 *
 * By using this file, you agree to the terms and conditions set
 * forth in the LICENSE file which can be found at the top level of
 * the distribution.
 *
 */

#pragma once

#include "srsran/adt/expected.h"
#include "srsran/srslog/logger.h"
#include "srsran/support/format/fmt_optional.h"
#include "srsran/support/io/unique_fd.h"
#include <chrono>
#include <cstdint>
#include <sys/socket.h>

namespace srsran {

struct sctp_socket_params {
  /// Name of the interface for logging purposes.
  std::string            if_name;
  int                    ai_family;
  int                    ai_socktype;
  bool                   reuse_addr        = false;
  bool                   non_blocking_mode = false;
  std::chrono::seconds   rx_timeout{0};
  std::optional<int32_t> rto_initial;
  std::optional<int32_t> rto_min;
  std::optional<int32_t> rto_max;
  std::optional<int32_t> init_max_attempts;
  std::optional<int32_t> max_init_timeo;
  std::optional<bool>    nodelay;
};

/// SCTP socket instance.
class sctp_socket
{
public:
  static expected<sctp_socket> create(const sctp_socket_params& params);

  sctp_socket();
  sctp_socket(sctp_socket&& other) noexcept = default;
  sctp_socket& operator=(sctp_socket&& other) noexcept;

  bool close();

  [[nodiscard]] bool is_open() const { return sock_fd.is_open(); }
  const unique_fd&   fd() const { return sock_fd; }
  void               release()
  {
    int fd  = sock_fd.release();
    sock_fd = unique_fd(fd, false);
  }

  [[nodiscard]] bool bind(struct sockaddr& ai_addr, const socklen_t& ai_addrlen, const std::string& bind_interface);
  [[nodiscard]] bool connect(struct sockaddr& ai_addr, const socklen_t& ai_addrlen);
  /// \brief Start listening on socket.
  [[nodiscard]] bool listen();
  [[nodiscard]] bool set_non_blocking();

  /// \brief Return the port on which the socket is listening.
  ///
  /// In case the gateway was configured to listen on port 0, i.e. the operating system shall pick a random free port,
  /// this function can be used to get the actual port number.
  std::optional<uint16_t> get_listen_port() const;

private:
  bool set_sockopts(const sctp_socket_params& params);

  std::string           if_name;
  bool                  non_blocking_mode = false;
  srslog::basic_logger& logger;

  unique_fd sock_fd;
};

} // namespace srsran

namespace fmt {
template <>
struct formatter<srsran::sctp_socket_params> {
  template <typename ParseContext>
  auto parse(ParseContext& ctx)
  {
    return ctx.begin();
  }

  template <typename FormatContext>
  auto format(const srsran::sctp_socket_params& cfg, FormatContext& ctx)
  {
    return format_to(ctx.out(),
                     "if_name={} ai_family={} ai_socktype={} reuse_addr={} non_blockin_mode={} rx_timeout={} "
                     "rto_initial={} rto_min={} rto_max={} init_max_attempts={} max_init_timeo={} no_delay={}",
                     cfg.if_name,
                     cfg.ai_family,
                     cfg.ai_socktype,
                     cfg.reuse_addr,
                     cfg.non_blocking_mode,
                     cfg.rx_timeout.count(),
                     cfg.rto_initial,
                     cfg.rto_min,
                     cfg.rto_max,
                     cfg.init_max_attempts,
                     cfg.max_init_timeo,
                     cfg.nodelay);
  }
};
} // namespace fmt
