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

#pragma once

#include "../e1ap_cu_cp_impl.h"
#include "../ue_context/e1ap_cu_cp_ue_context.h"
#include "common/e1ap_asn1_utils.h"
#include "srsran/e1ap/common/e1ap_message.h"
#include "srsran/e1ap/cu_cp/e1ap_cu_cp.h"
#include "srsran/support/async/async_task.h"

namespace srsran {
namespace srs_cu_cp {

class bearer_context_modification_procedure
{
public:
  bearer_context_modification_procedure(const e1ap_configuration&        e1ap_cfg_,
                                        const e1ap_message&              request_,
                                        e1ap_bearer_transaction_manager& ev_mng_,
                                        e1ap_message_notifier&           e1ap_notif_,
                                        e1ap_ue_logger&                  logger_);

  void operator()(coro_context<async_task<e1ap_bearer_context_modification_response>>& ctx);

  static const char* name() { return "Bearer Context Modification Procedure"; }

private:
  /// Send Bearer Context Modification Request to CU-UP.
  void send_bearer_context_modification_request();

  /// Creates procedure result to send back to procedure caller.
  e1ap_bearer_context_modification_response create_bearer_context_modification_result();

  const e1ap_configuration         e1ap_cfg;
  const e1ap_message               request;
  e1ap_bearer_transaction_manager& ev_mng;
  e1ap_message_notifier&           e1ap_notifier;
  e1ap_ue_logger&                  logger;

  protocol_transaction_outcome_observer<asn1::e1ap::bearer_context_mod_resp_s, asn1::e1ap::bearer_context_mod_fail_s>
      transaction_sink;
};

} // namespace srs_cu_cp
} // namespace srsran
