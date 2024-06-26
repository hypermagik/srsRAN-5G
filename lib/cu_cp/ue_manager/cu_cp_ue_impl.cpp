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

#include "cu_cp_ue_impl.h"

using namespace srsran;
using namespace srs_cu_cp;

cu_cp_ue::cu_cp_ue(const ue_index_t               ue_index_,
                   const up_resource_manager_cfg& up_cfg,
                   const security_manager_config& sec_cfg,
                   ue_task_scheduler_impl         task_sched_,
                   const pci_t                    pci_,
                   const rnti_t                   c_rnti_) :
  ue_index(ue_index_),
  task_sched(std::move(task_sched_)),
  up_mng(up_cfg),
  sec_mng(sec_cfg),
  rrc_ue_cu_cp_ev_notifier(ue_index)
{
  if (pci_ != INVALID_PCI) {
    pci = pci_;
  }

  if (c_rnti_ != rnti_t::INVALID_RNTI) {
    ue_ctxt.crnti = c_rnti_;
  }

  ue_ctxt.du_idx = get_du_index_from_ue_index(ue_index);

  rrc_ue_cu_cp_ue_ev_notifier.connect_ue(*this);
  ngap_cu_cp_ue_ev_notifier.connect_ue(*this);
}

/// \brief Update a UE with PCI and/or C-RNTI.
void cu_cp_ue::update_du_ue(gnb_du_id_t du_id_, pci_t pci_, rnti_t c_rnti_)
{
  if (du_id_ != gnb_du_id_t::invalid) {
    ue_ctxt.du_id = du_id_;
  }

  if (pci_ != INVALID_PCI) {
    pci = pci_;
  }

  if (c_rnti_ != rnti_t::INVALID_RNTI) {
    ue_ctxt.crnti = c_rnti_;
  }
}

/// \brief Set/update the measurement context of the UE.
void cu_cp_ue::update_meas_context(cell_meas_manager_ue_context meas_ctxt)
{
  meas_context = std::move(meas_ctxt);
}

/// \brief Set the DU and PCell index of the UE.
/// \param[in] pcell_index PCell index of the UE.
void cu_cp_ue::set_pcell_index(du_cell_index_t pcell_index_)
{
  pcell_index = pcell_index_;
}

/// \brief Set the RRC UE control message notifier of the UE.
/// \param[in] rrc_ue_notifier_ RRC UE control message notifier of the UE.
void cu_cp_ue::set_rrc_ue_notifier(du_processor_rrc_ue_control_message_notifier& rrc_ue_notifier_)
{
  rrc_ue_notifier = &rrc_ue_notifier_;
}

/// \brief Set the RRC UE SRB notifier of the UE.
/// \param[in] rrc_ue_srb_notifier_ RRC UE SRB control notifier of the UE.
void cu_cp_ue::set_rrc_ue_srb_notifier(du_processor_rrc_ue_srb_control_notifier& rrc_ue_srb_notifier_)
{
  rrc_ue_srb_notifier = &rrc_ue_srb_notifier_;
}

/// \brief Get the RRC UE PDU notifier of the UE.
ngap_rrc_ue_pdu_notifier& cu_cp_ue::get_rrc_ue_pdu_notifier()
{
  return ngap_rrc_ue_ev_notifier;
}

/// \brief Get the RRC UE control notifier of the UE.
ngap_rrc_ue_control_notifier& cu_cp_ue::get_rrc_ue_control_notifier()
{
  return ngap_rrc_ue_ev_notifier;
}

/// \brief Get the RRC UE control message notifier of the UE.
du_processor_rrc_ue_control_message_notifier& cu_cp_ue::get_rrc_ue_notifier()
{
  srsran_assert(rrc_ue_notifier != nullptr, "ue={}: RRC UE notifier was not set", ue_index);
  return *rrc_ue_notifier;
}

/// \brief Get the RRC UE SRB control notifier of the UE.
du_processor_rrc_ue_srb_control_notifier& cu_cp_ue::get_rrc_ue_srb_notifier()
{
  srsran_assert(rrc_ue_srb_notifier != nullptr, "ue={}: RRC UE SRB notifier was not set", ue_index);
  return *rrc_ue_srb_notifier;
}
