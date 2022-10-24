/*
 *
 * Copyright 2013-2022 Software Radio Systems Limited
 *
 * By using this file, you agree to the terms and conditions set
 * forth in the LICENSE file which can be found at the top level of
 * the distribution.
 *
 */

#include "pucch_test_utils.h"

using namespace srsgnb;

pucch_info srsgnb::build_pucch_info(const bwp_configuration* bwp_cfg,
                                    unsigned                 pci,
                                    pucch_format             format,
                                    prb_interval             prbs,
                                    prb_interval             second_hop_prbs,
                                    ofdm_symbol_range        symbols,
                                    uint8_t                  initial_cyclic_shift,
                                    sr_nof_bits              sr_bits,
                                    unsigned                 harq_ack_nof_bits,
                                    uint8_t                  time_domain_occ)
{
  pucch_info pucch_test{.crnti = to_rnti(0x4601), .bwp_cfg = bwp_cfg};

  if (format == pucch_format::FORMAT_0) {
    pucch_test.resources.prbs            = prbs;
    pucch_test.resources.second_hop_prbs = second_hop_prbs;
    pucch_test.resources.symbols         = symbols;

    pucch_test.format_0.initial_cyclic_shift = initial_cyclic_shift;
    pucch_test.format_0.sr_bits              = sr_bits;
    pucch_test.format_0.harq_ack_nof_bits    = harq_ack_nof_bits;

    pucch_test.format_0.group_hopping = pucch_group_hopping::NEITHER;
    pucch_test.format                 = pucch_format::FORMAT_0;
    pucch_test.format_0.n_id_hopping  = pci;
  } else if (format == pucch_format::FORMAT_1) {
    pucch_test.resources.prbs            = prbs;
    pucch_test.resources.second_hop_prbs = second_hop_prbs;
    pucch_test.resources.symbols         = symbols;

    pucch_test.format_1.initial_cyclic_shift = initial_cyclic_shift;
    pucch_test.format_1.sr_bits              = sr_bits;
    pucch_test.format_1.harq_ack_nof_bits    = harq_ack_nof_bits;
    pucch_test.format_1.time_domain_occ      = time_domain_occ;

    pucch_test.format_1.group_hopping   = pucch_group_hopping::NEITHER;
    pucch_test.format                   = pucch_format::FORMAT_1;
    pucch_test.format_1.n_id_hopping    = pci;
    pucch_test.format_1.slot_repetition = pucch_repetition_tx_slot::no_multi_slot;
  } else {
    return pucch_info{};
  }

  return pucch_test;
}

// Verify if the PUCCH scheduler output (or PUCCH PDU) is correct.
bool srsgnb::assess_ul_pucch_info(const pucch_info& expected, const pucch_info& test)
{
  bool is_equal = expected.crnti == test.crnti && *expected.bwp_cfg == *test.bwp_cfg && expected.format == test.format;
  is_equal      = is_equal && expected.resources.prbs == test.resources.prbs &&
             expected.resources.symbols == test.resources.symbols &&
             expected.resources.second_hop_prbs == test.resources.second_hop_prbs;

  switch (expected.format) {
    case pucch_format::FORMAT_0: {
      const pucch_format_0& expected_f = expected.format_0;
      const pucch_format_0& test_f     = test.format_0;
      is_equal                         = is_equal && expected_f.group_hopping == test_f.group_hopping &&
                 expected_f.n_id_hopping == test_f.n_id_hopping &&
                 expected_f.initial_cyclic_shift == test_f.initial_cyclic_shift &&
                 expected_f.sr_bits == test_f.sr_bits && expected_f.harq_ack_nof_bits == test_f.harq_ack_nof_bits;

      break;
    }
    case pucch_format::FORMAT_1: {
      const pucch_format_1& expected_f = expected.format_1;
      const pucch_format_1& test_f     = test.format_1;
      is_equal                         = is_equal && expected_f.group_hopping == test_f.group_hopping &&
                 expected_f.n_id_hopping == test_f.n_id_hopping &&
                 expected_f.initial_cyclic_shift == test_f.initial_cyclic_shift &&
                 expected_f.sr_bits == test_f.sr_bits && expected_f.harq_ack_nof_bits == test_f.harq_ack_nof_bits &&
                 expected_f.slot_repetition == test_f.slot_repetition &&
                 expected_f.time_domain_occ == test_f.time_domain_occ;
      break;
    }
    default: {
      return false;
    };
  }

  return is_equal;
}

/////////        TEST BENCH for PUCCH scheduler        /////////

// Test bench with all that is needed for the PUCCH.

test_bench::test_bench(unsigned pucch_res_common, unsigned n_cces, sr_periodicity period, unsigned offset) :
  cell_cfg{make_custom_sched_cell_configuration_request(pucch_res_common)},
  coreset_cfg{config_helpers::make_default_coreset_config()},
  dci_info{make_default_dci(n_cces, &coreset_cfg)},
  k0(cell_cfg.dl_cfg_common.init_dl_bwp.pdsch_common.pdsch_td_alloc_list[0].k0),
  pucch_alloc{cell_cfg},
  pucch_sched{cell_cfg, pucch_alloc, ues},
  sl_tx{to_numerology_value(cell_cfg.dl_cfg_common.init_dl_bwp.generic_params.scs), 0}
{
  sched_ue_creation_request_message ue_req =
      make_scheduler_ue_creation_request(test_helpers::make_default_ue_creation_request());

  srsgnb_assert(
      not ue_req.cells.empty() and ue_req.cells.back().serv_cell_cfg.has_value() and
          ue_req.cells.back().serv_cell_cfg.value().ul_config.has_value() and
          ue_req.cells.back().serv_cell_cfg.value().ul_config.value().init_ul_bwp.pucch_cfg.has_value() and
          ue_req.cells.back().serv_cell_cfg.value().ul_config.value().init_ul_bwp.pucch_cfg->sr_res_list.size() == 1,
      "Hello");

  ue_req.cells.back().serv_cell_cfg.value().ul_config.value().init_ul_bwp.pucch_cfg->sr_res_list[0].period = period;
  ue_req.cells.back().serv_cell_cfg.value().ul_config.value().init_ul_bwp.pucch_cfg->sr_res_list[0].offset = offset;

  ues.insert(ue_idx, std::make_unique<ue>(cell_cfg, ue_req));
  slot_indication(sl_tx);
}

const ue& test_bench::get_ue() const
{
  auto user = ues.find(ue_idx);
  srsgnb_assert(user != ues.end(), "User not found");
  return *user;
}

void test_bench::slot_indication(slot_point slot_tx)
{
  pucch_alloc.slot_indication(slot_tx);
  mac_logger.set_context(slot_tx.to_uint());
  test_logger.set_context(slot_tx.to_uint());
  res_grid.slot_indication(slot_tx);
}
