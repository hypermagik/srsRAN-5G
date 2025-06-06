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

#include "apps/units/flexible_o_du/split_6/o_du_low/split6_cell_configurator_plugin.h"
#include "split6_cell_configurator_plugin.h"

namespace srsran {

/// Split 6 cell configurator plugin dummy implementation.
class split6_cell_configurator_plugin_dummy : public split6_cell_configurator_plugin
{
public:
  // See interface for documentation.
  fapi::config_message_notifier& get_config_message_notifier() override;

  // See interface for documentation.
  fapi::slot_error_message_notifier& get_slot_error_message_notifier() override;
};

} // namespace srsran
