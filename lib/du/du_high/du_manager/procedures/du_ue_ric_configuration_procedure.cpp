/*
 *
 * Copyright 2021-2025 Software Radio Systems Limited
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

#include "du_ue_ric_configuration_procedure.h"
#include "srsran/ran/band_helper.h"
#include "srsran/scheduler/config/scheduler_expert_config.h"
#include "srsran/support/async/execute_on_blocking.h"

using namespace srsran;
using namespace srs_du;

du_ue_ric_configuration_procedure::du_ue_ric_configuration_procedure(const du_mac_sched_control_config& request_,
                                                                     du_ue_manager_repository&          ue_mng_,
                                                                     const du_manager_params&           du_params_) :
  request(request_), ue_mng(ue_mng_), du_params(du_params_)
{
}

void du_ue_ric_configuration_procedure::operator()(coro_context<async_task<du_mac_sched_control_config_response>>& ctx)
{
  CORO_BEGIN(ctx);

  // Change execution context to DU manager.
  CORO_AWAIT(execute_on_blocking(du_params.services.du_mng_exec, du_params.services.timers));

  // Run config task inside the UE task loop and await for its completion.
  CORO_AWAIT(dispatch_ue_config_task());

  // TODO: Potentially change back to caller execution context.

  CORO_RETURN(ue_config_completed.get());
}

manual_event<du_mac_sched_control_config_response>& du_ue_ric_configuration_procedure::dispatch_ue_config_task()
{
  // Find UE context based on F1AP UE ID.
  ue = ue_mng.find_f1ap_ue_id(static_cast<gnb_du_ue_f1ap_id_t>(request.ue_id));
  if (ue == nullptr) {
    du_mac_sched_control_config_response fail{false, false, false};
    ue_config_completed.set(fail);
    return ue_config_completed;
  }

  // Dispatch UE configuration to UE task loop inside the UE manager.
  ue_mng.schedule_async_task(ue->ue_index, launch_async([this](coro_context<async_task<void>>& ctx) {
                               CORO_BEGIN(ctx);

                               // Await for UE configuration completion.
                               CORO_AWAIT_VALUE(const mac_ue_reconfiguration_response result, handle_mac_config());

                               // Signal completion of UE configuration to external coroutine.
                               ue_config_completed.set(
                                   du_mac_sched_control_config_response{result.result, result.result, result.result});

                               CORO_RETURN();
                             }));

  return ue_config_completed;
}

async_task<mac_ue_reconfiguration_response> du_ue_ric_configuration_procedure::handle_mac_config()
{
  mac_ue_reconfiguration_request mac_request;

  mac_request.ue_index    = ue->ue_index;
  mac_request.crnti       = ue->rnti;
  mac_request.pcell_index = to_du_cell_index(0);

  // Configure UE resource allocation parameters.
  mac_request.sched_cfg.res_alloc_cfg.emplace();
  auto& res_alloc_cfg = mac_request.sched_cfg.res_alloc_cfg.value();
  // For now take first parameter set, in future we will have to support multiple parameter sets for different slices.
  control_config_params  req     = request.param_list[0];
  rrm_policy_ratio_group dummy   = {};
  res_alloc_cfg.rrm_policy_group = req.rrm_policy_group.has_value() ? req.rrm_policy_group.value() : dummy;
  // TODO remove when RRM group support is added to scheduler.
  int min_prb_limit = 0;
  int max_prb_limit = MAX_NOF_PRBS;
  if (req.rrm_policy_group.has_value()) {
    unsigned int nof_prbs = get_max_Nprb(du_params.ran.cells[0].dl_carrier.carrier_bw_mhz,
                                         du_params.ran.cells[0].scs_common,
                                         band_helper::get_freq_range(du_params.ran.cells[0].dl_carrier.band));

    min_prb_limit = req.rrm_policy_group.value().min_prb_policy_ratio.has_value()
                        ? req.rrm_policy_group.value().min_prb_policy_ratio.value()
                        : 0;
    max_prb_limit = req.rrm_policy_group.value().max_prb_policy_ratio.has_value()
                        ? req.rrm_policy_group.value().max_prb_policy_ratio.value()
                        : MAX_NOF_PRBS;

    min_prb_limit = std::max(0, std::min(min_prb_limit, 100));
    max_prb_limit = std::max(0, std::min(max_prb_limit, 100));

    if (max_prb_limit < min_prb_limit) {
      std::swap(max_prb_limit, min_prb_limit);
    }

    min_prb_limit = static_cast<int>((1.0 * min_prb_limit / 100) * nof_prbs);
    max_prb_limit = static_cast<int>((1.0 * max_prb_limit / 100) * nof_prbs);
  }
  res_alloc_cfg.pdsch_grant_size_limits = {min_prb_limit, max_prb_limit};
  res_alloc_cfg.pusch_grant_size_limits = {min_prb_limit, max_prb_limit};

  res_alloc_cfg.max_pdsch_harq_retxs = req.num_harq_retransmissions.has_value()
                                           ? req.num_harq_retransmissions.value()
                                           : du_params.mac.sched_cfg.ue.max_nof_dl_harq_retxs;
  res_alloc_cfg.max_pusch_harq_retxs = res_alloc_cfg.max_pdsch_harq_retxs;

  return du_params.mac.ue_cfg.handle_ue_reconfiguration_request(mac_request);
}
