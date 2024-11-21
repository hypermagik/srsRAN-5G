/*
 *
 * Copyright 2021-2024 Software Radio Systems Limited
 *
 * By using this file, you agree to the terms and conditions set
 * forth in the LICENSE file which can be found at the top level of
 * the distribution.
 *
 */

#include "tests/test_doubles/scheduler/cell_config_builder_profiles.h"
#include "tests/test_doubles/scheduler/scheduler_config_helper.h"
#include "tests/unittests/scheduler/test_utils/scheduler_test_simulator.h"
#include "srsran/adt/ranges/transform.h"
#include <gtest/gtest.h>

using namespace srsran;

struct multi_slice_test_params {
  std::vector<slice_rrm_policy_config> slices;
};

class base_multi_slice_scheduler_tester : public scheduler_test_simulator
{
protected:
  base_multi_slice_scheduler_tester(const multi_slice_test_params& params_) :
    scheduler_test_simulator(4, subcarrier_spacing::kHz30), params(params_)
  {
    // Add Cell.
    auto sched_cell_cfg_req = sched_config_helper::make_default_sched_cell_configuration_request(builder_params);
    sched_cell_cfg_req.rrm_policy_members = params.slices;
    this->add_cell(sched_cell_cfg_req);
  }

  rnti_t add_ue(const std::vector<std::pair<lcid_t, s_nssai_t>>& lcid_to_cfg)
  {
    auto                get_lcid  = [](const auto& e) { return e.first; };
    auto                only_lcid = views::transform(lcid_to_cfg, get_lcid);
    std::vector<lcid_t> lcid_list(only_lcid.begin(), only_lcid.end());
    auto ue_cfg = sched_config_helper::create_default_sched_ue_creation_request(builder_params, lcid_list);
    for (unsigned int i = 0; i < lcid_list.size(); i++) {
      ue_cfg.cfg.drb_info_list[i].s_nssai = lcid_to_cfg[i].second;

      auto it                = std::find_if(ue_cfg.cfg.lc_config_list->begin(),
                             ue_cfg.cfg.lc_config_list->end(),
                             [lcid = lcid_to_cfg[i].first](const auto& l) { return l.lcid == lcid; });
      it->rrm_policy.s_nssai = lcid_to_cfg[i].second;
    }
    ue_cfg.crnti    = to_rnti(0x4601 + ue_count);
    ue_cfg.ue_index = to_du_ue_index(ue_count);
    scheduler_test_simulator::add_ue(ue_cfg);

    ue_count++;
    return ue_cfg.crnti;
  }

  multi_slice_test_params    params;
  cell_config_builder_params builder_params = cell_config_builder_profiles::tdd();

  unsigned ue_count = 0;
};

class single_slice_limited_max_rbs_scheduler_test : public base_multi_slice_scheduler_tester, public ::testing::Test
{
protected:
  constexpr static unsigned max_slice_rbs = 10;

  static s_nssai_t test_nssai() { return s_nssai_t{slice_service_type{1}, slice_differentiator::create(1).value()}; }

  single_slice_limited_max_rbs_scheduler_test() :
    base_multi_slice_scheduler_tester(multi_slice_test_params{
        {slice_rrm_policy_config{rrm_policy_member{plmn_identity::test_value(), test_nssai()}, 0, max_slice_rbs}}})
  {
  }

private:
  unsigned ue_count = 0;
};

TEST_F(single_slice_limited_max_rbs_scheduler_test, single_ue_limited_to_max_rbs)
{
  // Create UE and fill its buffer.
  rnti_t rnti = this->add_ue({std::make_pair(LCID_MIN_DRB, test_nssai())});
  this->push_dl_buffer_state(dl_buffer_state_indication_message{to_du_ue_index(0), LCID_MIN_DRB, 500});

  ASSERT_TRUE(this->run_slot_until(
      [this, rnti]() { return find_ue_pdsch(rnti, this->last_sched_res_list[0]->dl.ue_grants) != nullptr; }));
  const dl_msg_alloc* msg = find_ue_pdsch(rnti, this->last_sched_res_list[0]->dl.ue_grants);

  ASSERT_TRUE(msg->pdsch_cfg.rbs.type1().length() <= max_slice_rbs);
}

TEST_F(single_slice_limited_max_rbs_scheduler_test, multi_ue_limited_to_max_rbs)
{
  // Create UE and fill its buffer.
  unsigned            nof_ues = test_rgen::uniform_int<unsigned>(2, 10);
  unsigned            dl_bo   = test_rgen::uniform_int<unsigned>(1, 50);
  std::vector<rnti_t> rntis;
  for (unsigned i = 0; i < nof_ues; i++) {
    rntis.push_back(this->add_ue({std::make_pair(LCID_MIN_DRB, test_nssai())}));
    this->push_dl_buffer_state(dl_buffer_state_indication_message{to_du_ue_index(i), LCID_MIN_DRB, dl_bo});
  }

  ASSERT_TRUE(this->run_slot_until(
      [&]() { return find_ue_pdsch(rntis.front(), this->last_sched_res_list[0]->dl.ue_grants) != nullptr; }));
  unsigned nof_rbs = 0;
  for (const dl_msg_alloc& msg : this->last_sched_res_list[0]->dl.ue_grants) {
    nof_rbs += msg.pdsch_cfg.rbs.type1().length();
  }

  ASSERT_TRUE(nof_rbs <= max_slice_rbs);
}
