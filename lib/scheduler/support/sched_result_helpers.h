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

#include "srsran/scheduler/config/scheduler_expert_config.h"
#include "srsran/scheduler/scheduler_slot_handler.h"

namespace srsran {

/// Determine the space left for a new UCI PDU of a gived C-RNTI in the scheduler result.
inline bool
has_space_for_uci_pdu(const sched_result& result, rnti_t crnti, const scheduler_ue_expert_config& expert_cfg)
{
  // Check if we can fit the UCI in an existing PUSCH PDU for the same UE.
  bool has_pusch = std::any_of(result.ul.puschs.begin(), result.ul.puschs.end(), [crnti](const ul_sched_info& grant) {
    return grant.pusch_cfg.rnti == crnti;
  });
  if (has_pusch) {
    return true;
  }

  // Check if we can fit the UCI in an existing PUCCH PDU for the same UE.
  bool has_pucch = std::any_of(result.ul.pucchs.begin(), result.ul.pucchs.end(), [crnti](const pucch_info& grant) {
    return grant.crnti == crnti;
  });
  if (has_pucch) {
    return true;
  }

  // New PUCCH PDU is required.
  // Check if max PUCCH limit is not reached.
  const unsigned max_pucchs = std::min(expert_cfg.max_pucchs_per_slot, (unsigned)result.ul.pucchs.capacity());
  unsigned       space      = max_pucchs - result.ul.pucchs.size();
  if (space == 0) {
    return false;
  }

  // Check if max UL grant limit (PUCCH + PUSCH) is not reached.
  unsigned ul_grants = result.ul.pucchs.size() + result.ul.puschs.size();
  return expert_cfg.max_ul_grants_per_slot > ul_grants;
}

/// Determine the space left for a new PUSCH PDU of a gived C-RNTI in the scheduler result.
inline unsigned
get_space_left_for_pusch_pdus(const sched_result& result, rnti_t crnti, const scheduler_ue_expert_config& expert_cfg)
{
  // Check if max PUSCH limit is not reached.
  const unsigned max_puschs = std::min(expert_cfg.max_puschs_per_slot, (unsigned)result.ul.puschs.capacity());
  unsigned       space      = max_puschs - result.ul.puschs.size();
  if (space == 0) {
    return 0;
  }

  // Check if max UL grant limit (PUCCH + PUSCH) is not reached.
  // Note: Consider that if an PUSCH grant is scheduled for this UE, its PUCCH PDUs are removed.
  unsigned nof_pucchs    = std::count_if(result.ul.pucchs.begin(),
                                      result.ul.pucchs.end(),
                                      [crnti](const pucch_info& pucch) { return pucch.crnti == crnti; });
  unsigned nof_ul_grants = result.ul.pucchs.size() + result.ul.puschs.size() - nof_pucchs;
  unsigned space2 =
      expert_cfg.max_ul_grants_per_slot >= nof_ul_grants ? expert_cfg.max_ul_grants_per_slot - nof_ul_grants : 0;

  return std::min(space, space2);
}

} // namespace srsran