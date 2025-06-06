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

#include "split6_slot_configurator_plugin.h"
#include "srsran/du/du_low/o_du_low.h"
#include "srsran/ru/ru.h"
#include "srsran/ru/ru_adapters.h"
#include <memory>

namespace srsran {

/// \brief Split 6 flexible O-DU low implementation.
///
/// This implementation glues the split6 slot configurator plugin, the O-DU low and the RU in a single object, managing
/// the lifetime of the objects related to an active cell. When a cell starts, this object must be created, and when it
/// is stopped, this object can safely be destroyed.
class split6_flexible_o_du_low_impl
{
  static constexpr unsigned NOF_CELLS_SUPPORTED = 1U;

public:
  split6_flexible_o_du_low_impl() :
    ru_ul_adapt(NOF_CELLS_SUPPORTED), ru_timing_adapt(NOF_CELLS_SUPPORTED), ru_error_adapt(NOF_CELLS_SUPPORTED)
  {
  }

  ~split6_flexible_o_du_low_impl();

  /// Sets the dependencies to the given one
  void set_dependencies(std::unique_ptr<split6_slot_configurator_plugin> slot,
                        std::unique_ptr<srs_du::o_du_low>                du,
                        std::unique_ptr<radio_unit>                      radio);

  /// Getters to the adaptors.
  upper_phy_ru_ul_adapter&         get_upper_ru_ul_adapter() { return ru_ul_adapt; }
  upper_phy_ru_timing_adapter&     get_upper_ru_timing_adapter() { return ru_timing_adapt; }
  upper_phy_ru_error_adapter&      get_upper_ru_error_adapter() { return ru_error_adapt; }
  upper_phy_ru_dl_rg_adapter&      get_upper_ru_dl_rg_adapter() { return ru_dl_rg_adapt; }
  upper_phy_ru_ul_request_adapter& get_upper_ru_ul_request_adapter() { return ru_ul_request_adapt; }

private:
  upper_phy_ru_ul_adapter                          ru_ul_adapt;
  upper_phy_ru_timing_adapter                      ru_timing_adapt;
  upper_phy_ru_error_adapter                       ru_error_adapt;
  std::unique_ptr<split6_slot_configurator_plugin> slot_plugin;
  std::unique_ptr<srs_du::o_du_low>                odu_low;
  std::unique_ptr<radio_unit>                      ru;
  upper_phy_ru_dl_rg_adapter                       ru_dl_rg_adapt;
  upper_phy_ru_ul_request_adapter                  ru_ul_request_adapt;
};

} // namespace srsran
