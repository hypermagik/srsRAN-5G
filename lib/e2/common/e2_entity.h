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

#include "e2_impl.h"
#include "procedures/e2_setup_procedure.h"
#include "procedures/e2_subscription_setup_procedure.h"
#include "srsran/asn1/e2ap/e2ap.h"
#include "srsran/du/du_high/du_manager/du_configurator.h"
#include "srsran/e2/e2_connection_client.h"
#include "srsran/e2/e2_cu.h"
#include "srsran/e2/e2_du.h"
#include "srsran/e2/e2ap_configuration.h"
#include "srsran/e2/e2sm/e2sm_factory.h"
#include "srsran/e2/e2sm/e2sm_manager.h"
#include "srsran/f1ap/du/f1ap_du.h"
#include "srsran/ran/nr_cgi.h"
#include "srsran/support/async/fifo_async_task_scheduler.h"
#include <map>
#include <memory>

namespace srsran {

/// Wrapper for the E2 interface that puts function calls into its own task executor
class e2_entity final : public e2_interface
{
public:
  e2_entity(e2ap_configuration& cfg_, std::unique_ptr<e2_interface> decorated_e2_iface_, task_executor& task_exec_);

  e2_entity(e2ap_configuration&                                              cfg_,
            e2_connection_client*                                            e2_client_,
            std::variant<e2_du_metrics_interface*, e2_cu_metrics_interface*> e2_metrics_,
            srs_du::f1ap_ue_id_translator*                                   f1ap_ue_id_translator_,
            srs_du::du_configurator*                                         du_configurator_,
            timer_factory                                                    timers_,
            task_executor&                                                   task_exec_);

  void start() override;
  void stop() override;

  /// E2 connection manager functions.
  async_task<e2_setup_response_message> handle_e2_setup_request(e2_setup_request_message& request) override;
  async_task<e2_setup_response_message> start_initial_e2_setup_routine() override;

  /// E2_event_ handler functions.
  void handle_connection_loss() override;

  /// E2 message handler functions.
  void handle_message(const e2_message& msg) override;

private:
  srslog::basic_logger& logger;
  e2ap_configuration&   cfg;

  // Handler for E2AP tasks.
  task_executor&            task_exec;
  fifo_async_task_scheduler main_ctrl_loop;

  std::unique_ptr<e2_message_notifier>       e2_pdu_notifier    = nullptr;
  std::unique_ptr<e2sm_manager>              e2sm_mngr          = nullptr;
  std::unique_ptr<e2_subscription_manager>   subscription_mngr  = nullptr;
  std::unique_ptr<e2_interface>              decorated_e2_iface = nullptr;
  std::vector<std::unique_ptr<e2sm_handler>> e2sm_handlers;
};

} // namespace srsran
