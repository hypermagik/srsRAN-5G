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

#include "srsran/asn1/f1ap/f1ap.h"
#include "srsran/asn1/f1ap/f1ap_pdu_contents.h"
#include "srsran/f1ap/du/f1ap_du.h"
#include "srsran/support/async/async_task.h"

namespace srsran {

class f1ap_message_notifier;

namespace srs_du {

class f1ap_du_ue_manager;

/// Implementation of the F1AP RESET procedure as per TS 38.473, section 8.2.1.
class reset_procedure
{
public:
  reset_procedure(const asn1::f1ap::reset_s& msg,
                  f1ap_du_configurator&      du_mng,
                  f1ap_du_ue_manager&        ue_mng,
                  f1ap_message_notifier&     msg_notifier);

  void operator()(coro_context<async_task<void>>& ctx);

private:
  const char* name() const { return "Reset"; }

  async_task<void>           handle_reset();
  std::vector<du_ue_index_t> create_ues_to_reset() const;
  void                       send_ack() const;

  const asn1::f1ap::reset_s msg;
  f1ap_du_configurator&     du_mng;
  f1ap_du_ue_manager&       ue_mng;
  f1ap_message_notifier&    msg_notifier;
  srslog::basic_logger&     logger;

  std::vector<asn1::f1ap::ue_associated_lc_f1_conn_item_s> ues_reset;
};

} // namespace srs_du
} // namespace srsran
