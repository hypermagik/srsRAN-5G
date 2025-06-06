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

#include "srsran/scheduler/config/scheduler_ue_config_validator.h"
#include "cell_configuration.h"
#include "srsran/scheduler/config/serving_cell_config_validator.h"
#include "srsran/support/config/validator_helpers.h"

using namespace srsran;
using namespace config_validators;

error_type<std::string>
srsran::config_validators::validate_sched_ue_creation_request_message(const sched_ue_creation_request_message& msg,
                                                                      const cell_configuration&                cell_cfg)
{
  // Verify the list of ServingCellConfig contains spCellConfig.
  VERIFY(msg.cfg.cells.has_value() and not msg.cfg.cells->empty(), "Empty list of ServingCellConfig");

  for (const cell_config_dedicated& cell : *msg.cfg.cells) {
    HANDLE_ERROR(validate_pdcch_cfg(cell.serv_cell_cfg, cell_cfg.dl_cfg_common));

    HANDLE_ERROR(validate_pdsch_cfg(cell.serv_cell_cfg));

    if (cell.serv_cell_cfg.ul_config.has_value()) {
      if (cell.serv_cell_cfg.ul_config->init_ul_bwp.pucch_cfg.has_value() and
          cell.serv_cell_cfg.ul_config->init_ul_bwp.srs_cfg.has_value() and
          cell_cfg.ul_cfg_common.init_ul_bwp.pucch_cfg_common.has_value()) {
        const pucch_config_common& pucch_cfg_common = cell_cfg.ul_cfg_common.init_ul_bwp.pucch_cfg_common.value();
        HANDLE_ERROR(validate_pucch_cfg(cell.serv_cell_cfg, pucch_cfg_common, cell_cfg.dl_carrier.nof_ant));
        HANDLE_ERROR(validate_srs_cfg(cell.serv_cell_cfg));
      }

      HANDLE_ERROR(
          validate_pusch_cfg(cell.serv_cell_cfg.ul_config.value(), cell.serv_cell_cfg.csi_meas_cfg.has_value()));
    }

    HANDLE_ERROR(validate_csi_meas_cfg(cell.serv_cell_cfg, cell_cfg.tdd_cfg_common));

    // At the moment, we only support the situation where all UEs have the same NZP-CSI-RS list.
    if (cell.serv_cell_cfg.csi_meas_cfg.has_value() and
        not cell.serv_cell_cfg.csi_meas_cfg->nzp_csi_rs_res_list.empty()) {
      VERIFY(cell.serv_cell_cfg.csi_meas_cfg->nzp_csi_rs_res_list == cell_cfg.nzp_csi_rs_list,
             "The NZP-CSI-RS Resource lists for the UE and the cell do not match");
    }
  }

  // TODO: Validate other parameters.
  return {};
}
