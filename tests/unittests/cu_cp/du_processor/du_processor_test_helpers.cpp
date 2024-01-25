/*
 *
 * Copyright 2021-2024 Software Radio Systems Limited
 *
 * By using this file, you agree to the terms and conditions set
 * forth in the LICENSE file which can be found at the top level of
 * the distribution.
 *
 */

#include "du_processor_test_helpers.h"
#include "tests/unittests/cu_cp/test_helpers.h"
#include "tests/unittests/f1ap/common/f1ap_cu_test_messages.h"
#include "srsran/cu_cp/cell_meas_manager.h"
#include "srsran/support/async/async_test_utils.h"
#include <memory>

using namespace srsran;
using namespace srs_cu_cp;

du_processor_test::du_processor_test()
{
  test_logger.set_level(srslog::basic_levels::debug);
  cu_cp_logger.set_level(srslog::basic_levels::debug);
  srslog::init();

  cu_cp_notifier = std::make_unique<dummy_du_processor_cu_cp_notifier>(ngap_ue_removal_handler, &ue_mng);

  // create ue task scheduler
  ue_task_sched = std::make_unique<dummy_du_processor_ue_task_scheduler>(timers, ctrl_worker);

  // create and start DU processor
  du_processor_config_t du_cfg = {};
  du_cfg.du_index              = uint_to_du_index(0);
  du_cfg.conn_handler          = &du_conn_notifier;

  du_processor_obj = create_du_processor(std::move(du_cfg),
                                         *cu_cp_notifier,
                                         f1ap_du_mgmt_notifier,
                                         f1ap_pdu_notifier,
                                         e1ap_ctrl_notifier,
                                         ngap_ctrl_notifier,
                                         f1ap_cu_cp_notifier,
                                         rrc_ue_ngap_notifier,
                                         rrc_ue_ngap_notifier,
                                         rrc_ue_cu_cp_notifier,
                                         *ue_task_sched,
                                         ue_mng,
                                         cell_meas_mng,
                                         ctrl_worker);
}

du_processor_test::~du_processor_test()
{
  // flush logger after each test
  srslog::flush();
}

void du_processor_test::attach_ue()
{
  // Generate valid F1SetupRequest
  f1ap_message f1_setup_req = generate_f1_setup_request();
  // Pass message to DU processor
  du_processor_obj->get_du_processor_f1ap_interface().get_f1ap_message_handler().handle_message(f1_setup_req);

  // Generate ue_creation message
  ue_index_t                ue_index        = ue_index_t::min;
  cu_cp_ue_creation_message ue_creation_msg = generate_ue_creation_message(ue_index, rnti_t::MIN_CRNTI, 6576);
  // Pass message to DU processor
  du_processor_obj->handle_ue_creation_request(ue_creation_msg);
}

void du_processor_test::receive_rrc_reconfig_complete()
{
  // inject RRC Reconfiguration complete into UE object
}
