/*
 *
 *
 * Copyright 2021-2024 Software Radio Systems Limited
 *
 * By using this file, you agree to the terms and conditions set
 * forth in the LICENSE file which can be found at the top level of
 * the distribution.
 *
 */

#pragma once

#include "srsran/f1u/cu_up/f1u_bearer_logger.h"
#include "srsran/f1u/cu_up/f1u_gateway.h"
#include "srsran/f1u/cu_up/f1u_session_manager.h"
#include "srsran/gtpu/gtpu_config.h"
#include "srsran/gtpu/gtpu_demux.h"
#include "srsran/gtpu/gtpu_gateway.h"
#include "srsran/gtpu/gtpu_tunnel_common_tx.h"
#include "srsran/gtpu/gtpu_tunnel_nru.h"
#include "srsran/gtpu/gtpu_tunnel_nru_rx.h"
#include "srsran/pcap/dlt_pcap.h"
#include "srsran/srslog/srslog.h"
#include <cstdint>
#include <mutex>
#include <unordered_map>

namespace srsran::srs_cu_up {

class gtpu_tx_udp_gw_adapter;
class gtpu_rx_f1u_adapter;
class network_gateway_data_gtpu_demux_adapter;

/// \brief Object used to represent a bearer at the CU F1-U gateway
/// On the co-located case this is done by connecting both entities directly.
///
/// It will keep a notifier to the DU NR-U RX and provide the methods to pass
/// an SDU to it.
class f1u_split_gateway_cu_bearer final : public f1u_cu_up_gateway_bearer
{
public:
  f1u_split_gateway_cu_bearer(uint32_t                              ue_index_,
                              drb_id_t                              drb_id,
                              const up_transport_layer_info&        ul_tnl_info_,
                              f1u_cu_up_gateway_bearer_rx_notifier& cu_rx_,
                              gtpu_tnl_pdu_session&                 udp_session,
                              task_executor&                        ul_exec_,
                              srs_cu_up::f1u_bearer_disconnector&   disconnector_);

  ~f1u_split_gateway_cu_bearer() override;

  void stop() override;

  expected<std::string> get_bind_address() const override;

  void on_new_pdu(nru_dl_message msg) override
  {
    if (tunnel_tx == nullptr) {
      logger.log_debug("DL GTPU tunnel not connected. Discarding SDU.");
      return;
    }
    tunnel_tx->handle_sdu(std::move(msg));
  }

  void attach_tunnel_rx(std::unique_ptr<gtpu_tunnel_common_rx_upper_layer_interface> tunnel_rx_)
  {
    tunnel_rx = std::move(tunnel_rx_);
  }

  void attach_tunnel_tx(std::unique_ptr<gtpu_tunnel_nru_tx_lower_layer_interface> tunnel_tx_)
  {
    tunnel_tx = std::move(tunnel_tx_);
  }

  std::unique_ptr<gtpu_tx_udp_gw_adapter> gtpu_to_network_adapter;
  std::unique_ptr<gtpu_rx_f1u_adapter>    gtpu_to_f1u_adapter;

  gtpu_tunnel_common_rx_upper_layer_interface* get_tunnel_rx_interface() { return tunnel_rx.get(); }

  /// Holds the RX executor associated with the F1-U bearer.
  task_executor& ul_exec;
  uint32_t       ue_index;

private:
  bool                                                         stopped = false;
  srs_cu_up::f1u_bearer_logger                                 logger;
  srs_cu_up::f1u_bearer_disconnector&                          disconnector;
  up_transport_layer_info                                      ul_tnl_info;
  gtpu_tnl_pdu_session&                                        udp_session;
  std::unique_ptr<gtpu_tunnel_common_rx_upper_layer_interface> tunnel_rx;
  std::unique_ptr<gtpu_tunnel_nru_tx_lower_layer_interface>    tunnel_tx;

public:
  /// Holds notifier that will point to NR-U bearer on the UL path
  f1u_cu_up_gateway_bearer_rx_notifier& cu_rx;

  /// Holds the DL UP TNL info associated with the F1-U bearer.
  std::optional<up_transport_layer_info> dl_tnl_info;
};

/// \brief Object used to connect the DU and CU-UP F1-U bearers
/// On the co-located case this is done by connecting both entities directly.
///
/// Note that CU and DU bearer creation and removal can be performed from different threads and are therefore
/// protected by a common mutex.
class f1u_split_connector final : public f1u_cu_up_udp_gateway
{
public:
  f1u_split_connector(const std::vector<std::unique_ptr<gtpu_gateway>>& udp_gws,
                      gtpu_demux&                                       demux_,
                      dlt_pcap&                                         gtpu_pcap_,
                      uint16_t                                          peer_port_ = GTPU_PORT,
                      std::string                                       ext_addr_  = "auto");
  ~f1u_split_connector() override;

  f1u_cu_up_udp_gateway* get_f1u_cu_up_gateway() { return this; }

  /// TODO this should get a ue_index and drb id to be able to find the right port/ip
  std::optional<uint16_t> get_bind_port() const override { return udp_sessions[0]->get_bind_port(); }

  std::unique_ptr<f1u_cu_up_gateway_bearer> create_cu_bearer(uint32_t                              ue_index,
                                                             drb_id_t                              drb_id,
                                                             const srs_cu_up::f1u_config&          config,
                                                             const gtpu_teid_t&                    ul_teid,
                                                             f1u_cu_up_gateway_bearer_rx_notifier& rx_notifier,
                                                             task_executor&                        ul_exec) override;

  void attach_dl_teid(const up_transport_layer_info& ul_up_tnl_info,
                      const up_transport_layer_info& dl_up_tnl_info) override;

  void disconnect_cu_bearer(const up_transport_layer_info& ul_up_tnl_info) override;

  expected<std::string> get_cu_bind_address() const override;

private:
  srslog::basic_logger& logger_cu;
  // Key is the UL UP TNL Info (CU-CP address and UL TEID reserved by CU-CP)
  std::unordered_map<gtpu_teid_t, f1u_split_gateway_cu_bearer*, gtpu_teid_hasher_t> cu_map;
  std::mutex map_mutex; // shared mutex for access to cu_map

  std::unique_ptr<f1u_session_manager>                     f1u_session_mngr;
  uint16_t                                                 peer_port;
  std::string                                              ext_addr;
  std::vector<std::unique_ptr<gtpu_tnl_pdu_session>>       udp_sessions;
  gtpu_demux&                                              demux;
  std::unique_ptr<network_gateway_data_gtpu_demux_adapter> gw_data_gtpu_demux_adapter;
  dlt_pcap&                                                gtpu_pcap;
};

} // namespace srsran::srs_cu_up
