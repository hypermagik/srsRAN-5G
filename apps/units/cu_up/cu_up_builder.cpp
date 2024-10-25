/*
 *
 * Copyright 2021-2024 Software Radio Systems Limited
 *
 * This file is part of srsRAN.
 *
 * srsRAN is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as
 * published by the Free Software Foundation, either version 3 of
 * the License, or (at your option) any later version.
 *
 * srsRAN is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * A copy of the GNU Affero General Public License can be found in
 * the LICENSE file in the top-level directory of this distribution
 * and at http://www.gnu.org/licenses/.
 *
 */

#include "cu_up_builder.h"
#include "apps/services/worker_manager.h"
#include "cu_up_unit_config.h"
#include "cu_up_unit_config_translators.h"
#include "cu_up_wrapper.h"
#include "srsran/cu_up/cu_up_factory.h"

using namespace srsran;

std::unique_ptr<srs_cu_up::cu_up_interface> srsran::build_cu_up(const cu_up_unit_config&       unit_cfg,
                                                                const cu_up_unit_dependencies& dependencies)
{
  srs_cu_up::cu_up_configuration config = generate_cu_up_config(unit_cfg);
  config.exec_mapper                    = dependencies.workers->cu_up_exec_mapper.get();
  config.e1ap.e1_conn_client            = dependencies.e1ap_conn_client;
  config.f1u_gateway                    = dependencies.f1u_gateway;
  config.gtpu_pcap                      = dependencies.gtpu_pcap;
  config.timers                         = dependencies.timers;
  config.qos                            = generate_cu_up_qos_config(unit_cfg);

  auto address = config.f1u_gateway->get_cu_bind_address();
  srsran_assert(address.has_value(), "Invalid F1-U bind address");

  config.net_cfg.f1u_bind_addr = address.value();
  // Create NG-U gateway.
  std::unique_ptr<srs_cu_up::ngu_gateway> ngu_gw;
  if (not unit_cfg.upf_cfg.no_core) {
    udp_network_gateway_config ngu_gw_config = {};
    ngu_gw_config.bind_address               = config.net_cfg.n3_bind_addr;
    ngu_gw_config.bind_port                  = config.net_cfg.n3_bind_port;
    ngu_gw_config.bind_interface             = config.net_cfg.n3_bind_interface;
    ngu_gw_config.rx_max_mmsg                = config.net_cfg.n3_rx_max_mmsg;
    ngu_gw                                   = srs_cu_up::create_udp_ngu_gateway(
        ngu_gw_config, *dependencies.io_brk, dependencies.workers->cu_up_exec_mapper->io_ul_executor());
  } else {
    ngu_gw = srs_cu_up::create_no_core_ngu_gateway();
  }
  config.ngu_gw = ngu_gw.get();

  return std::make_unique<cu_up_wrapper>(std::move(ngu_gw), create_cu_up(config));
}
