/*
 *
 * Copyright 2013-2022 Software Radio Systems Limited
 *
 * By using this file, you agree to the terms and conditions set
 * forth in the LICENSE file which can be found at the top level of
 * the distribution.
 *
 */

#pragma once

#include "lib/du_manager/converters/scheduler_configuration_helpers.h"
#include "srsgnb/du/du_cell_config_helpers.h"
#include "srsgnb/mac/mac_configuration_helpers.h"
#include "srsgnb/scheduler/config/logical_channel_config_factory.h"
#include "srsgnb/scheduler/config/sched_cell_config_helpers.h"
#include "srsgnb/scheduler/config/serving_cell_config.h"
#include "srsgnb/scheduler/config/serving_cell_config_factory.h"
#include "srsgnb/scheduler/mac_scheduler.h"

namespace srsgnb {
namespace test_helpers {

inline cell_config_builder_params
make_custom_intial_params(bs_channel_bandwidth_fr1 bw         = bs_channel_bandwidth_fr1::MHz10,
                          subcarrier_spacing       scs_common = subcarrier_spacing::kHz15)
{
  return cell_config_builder_params(bw, scs_common);
}

inline sched_cell_configuration_request_message
make_default_sched_cell_configuration_request(const cell_config_builder_params& params = {})
{
  sched_cell_configuration_request_message sched_req{};
  sched_req.cell_index     = to_du_cell_index(0);
  sched_req.pci            = params.pci;
  sched_req.scs_common     = params.scs_common;
  sched_req.dl_carrier     = config_helpers::make_default_carrier_configuration(params);
  sched_req.ul_carrier     = config_helpers::make_default_carrier_configuration(params);
  sched_req.dl_cfg_common  = config_helpers::make_default_dl_config_common(params);
  sched_req.ul_cfg_common  = config_helpers::make_default_ul_config_common(params);
  sched_req.ssb_config     = config_helpers::make_default_ssb_config(params);
  sched_req.dmrs_typeA_pos = dmrs_typeA_position::pos2;
  if (not band_helper::is_paired_spectrum(sched_req.dl_carrier.band)) {
    sched_req.tdd_ul_dl_cfg_common = config_helpers::make_default_tdd_ul_dl_config_common(params);
  }

  sched_req.nof_beams     = 1;
  sched_req.nof_layers    = 1;
  sched_req.nof_ant_ports = 1;

  // SIB1 parameters.
  sched_req.coreset0          = params.coreset0_index;
  sched_req.searchspace0      = 0U;
  sched_req.sib1_payload_size = 101; // Random size.

  sched_req.pucch_guardbands = config_helpers::build_pucch_guardbands_list(params);

  return sched_req;
}

// Helper function to create a sched_cell_configuration_request_message that allows a configuration with either 15kHz or
// 30kHz SCS. By default, it creates a bandwidth of 20MHz.
inline sched_cell_configuration_request_message
make_default_sched_cell_configuration_request_scs(subcarrier_spacing scs, bool tdd_mode = false)
{
  cell_config_builder_params               params = make_custom_intial_params(bs_channel_bandwidth_fr1::MHz20, scs);
  sched_cell_configuration_request_message msg{make_default_sched_cell_configuration_request(params)};
  msg.ssb_config.scs                               = scs;
  msg.scs_common                                   = scs;
  msg.ul_cfg_common.init_ul_bwp.generic_params.scs = scs;
  msg.dl_cfg_common.init_dl_bwp.generic_params.scs = scs;
  // Change Carrier parameters when SCS is 15kHz.
  if (scs == subcarrier_spacing::kHz15) {
    // Band n5 for FDD, band n41 for TDD.
    msg.dl_carrier.arfcn = tdd_mode ? 499200 : 530000;
    msg.ul_carrier.arfcn = tdd_mode ? 499200 : band_helper::get_ul_arfcn_from_dl_arfcn(msg.ul_carrier.arfcn);
    msg.dl_cfg_common.freq_info_dl.scs_carrier_list.front().carrier_bandwidth = 106;
    msg.dl_cfg_common.init_dl_bwp.generic_params.crbs =
        crb_interval{0, msg.dl_cfg_common.freq_info_dl.scs_carrier_list.front().carrier_bandwidth};
    msg.ul_cfg_common.freq_info_ul.scs_carrier_list.front().carrier_bandwidth = 106;
    msg.ul_cfg_common.init_ul_bwp.generic_params.crbs =
        crb_interval{0, msg.ul_cfg_common.freq_info_ul.scs_carrier_list.front().carrier_bandwidth};
  }
  // Change Carrier parameters when SCS is 30kHz.
  else if (scs == subcarrier_spacing::kHz30) {
    // Band n5 for FDD, band n77 or n78 for TDD.
    msg.dl_carrier.arfcn = tdd_mode ? 630000 : 176000;
    msg.ul_carrier.arfcn = tdd_mode ? 630000 : band_helper::get_ul_arfcn_from_dl_arfcn(msg.ul_carrier.arfcn);
    msg.dl_cfg_common.freq_info_dl.scs_carrier_list.emplace_back(
        scs_specific_carrier{0, subcarrier_spacing::kHz30, 52});
    msg.dl_cfg_common.init_dl_bwp.generic_params.crbs = {
        0, msg.dl_cfg_common.freq_info_dl.scs_carrier_list[1].carrier_bandwidth};
    msg.ul_cfg_common.freq_info_ul.scs_carrier_list.emplace_back(
        scs_specific_carrier{0, subcarrier_spacing::kHz30, 52});
    msg.ul_cfg_common.init_ul_bwp.generic_params.crbs = {
        0, msg.ul_cfg_common.freq_info_ul.scs_carrier_list[1].carrier_bandwidth};
  }
  msg.dl_carrier.carrier_bw_mhz = 20;
  msg.dl_carrier.nof_ant        = 1;
  msg.ul_carrier.carrier_bw_mhz = 20;
  msg.ul_carrier.nof_ant        = 1;

  if (tdd_mode) {
    msg.tdd_ul_dl_cfg_common.emplace(config_helpers::make_default_tdd_ul_dl_config_common());
  }

  return msg;
}

inline serving_cell_config create_test_initial_ue_serving_cell_config()
{
  serving_cell_config serv_cell;
  serv_cell.cell_index = to_du_cell_index(0);

  // > PDCCH-Config.
  serv_cell.init_dl_bwp.pdcch_cfg.emplace();
  pdcch_config& pdcch_cfg = serv_cell.init_dl_bwp.pdcch_cfg.value();
  // >> Add CORESET#1.
  pdcch_cfg.coresets.push_back(config_helpers::make_default_coreset_config());
  pdcch_cfg.coresets[0].id = to_coreset_id(1);
  pdcch_cfg.coresets[0].pdcch_dmrs_scrambling_id.emplace();
  pdcch_cfg.coresets[0].pdcch_dmrs_scrambling_id.value() = 0;
  // >> Add SearchSpace#2.
  pdcch_cfg.search_spaces.push_back(config_helpers::make_default_ue_search_space_config());

  // > PDSCH-Config.
  serv_cell.init_dl_bwp.pdsch_cfg.emplace();
  pdsch_config& pdsch_cfg = serv_cell.init_dl_bwp.pdsch_cfg.value();
  pdsch_cfg.data_scrambling_id_pdsch.emplace(0);
  pdsch_cfg.pdsch_mapping_type_a_dmrs.emplace();
  dmrs_downlink_config& dmrs_type_a = pdsch_cfg.pdsch_mapping_type_a_dmrs.value();
  dmrs_type_a.additional_positions.emplace(dmrs_additional_positions::pos1);
  pdsch_cfg.tci_states.push_back(tci_state{
      .state_id  = static_cast<tci_state_id_t>(0),
      .qcl_type1 = {.ref_sig  = {.type = qcl_info::reference_signal::reference_signal_type::ssb,
                                 .ssb  = static_cast<ssb_id_t>(0)},
                    .qcl_type = qcl_info::qcl_type::type_d},
  });
  pdsch_cfg.res_alloc = pdsch_config::resource_allocation::resource_allocation_type_1;
  pdsch_cfg.rbg_sz    = rbg_size::config1;
  pdsch_cfg.prb_bndlg.bundling.emplace<prb_bundling::static_bundling>(
      prb_bundling::static_bundling({.sz = prb_bundling::static_bundling::bundling_size::wideband}));

  // > UL Config.
  serv_cell.ul_config.emplace(config_helpers::make_default_ue_uplink_config(make_custom_intial_params()));

  // > pdsch-ServingCellConfig.
  serv_cell.pdsch_serv_cell_cfg.emplace(srsgnb::config_helpers::make_default_pdsch_serving_cell_config());

  return serv_cell;
}

inline cell_config_dedicated create_test_initial_ue_spcell_cell_config()
{
  cell_config_dedicated cfg;
  cfg.serv_cell_idx = to_serv_cell_index(0);
  cfg.serv_cell_cfg = create_test_initial_ue_serving_cell_config();
  return cfg;
}

inline sched_ue_creation_request_message create_default_sched_ue_creation_request()
{
  sched_ue_creation_request_message msg{};

  msg.ue_index = to_du_ue_index(0);
  msg.crnti    = to_rnti(0x4601);

  scheduling_request_to_addmod sr_0{.sr_id = scheduling_request_id::SR_ID_MIN, .max_tx = sr_max_tx::n64};
  msg.cfg.sched_request_config_list.push_back(sr_0);

  msg.cfg.cells.push_back(create_test_initial_ue_spcell_cell_config());

  msg.cfg.lc_config_list.resize(2);
  msg.cfg.lc_config_list[0] = config_helpers::create_default_logical_channel_config(lcid_t::LCID_SRB0);
  msg.cfg.lc_config_list[1] = config_helpers::create_default_logical_channel_config(lcid_t::LCID_SRB1);

  return msg;
}

inline rach_indication_message generate_rach_ind_msg(slot_point prach_slot_rx, rnti_t temp_crnti, unsigned rapid = 0)
{
  rach_indication_message msg{};
  msg.cell_index = to_du_cell_index(0);
  msg.slot_rx    = prach_slot_rx;
  msg.occasions.emplace_back();
  msg.occasions.back().frequency_index = 0;
  msg.occasions.back().start_symbol    = 0;
  msg.occasions.back().preambles.emplace_back();
  msg.occasions.back().preambles.back().preamble_id  = rapid;
  msg.occasions.back().preambles.back().tc_rnti      = temp_crnti;
  msg.occasions.back().preambles.back().time_advance = phy_time_unit::from_seconds(0);
  return msg;
}

} // namespace test_helpers
} // namespace srsgnb
