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

#include "srsran/phy/upper/signal_processors/srs/srs_estimator.h"
#include "srsran/support/memory_pool/concurrent_thread_local_object_pool.h"

namespace srsran {

/// Concurrent Sounding Reference Signal propagation channel estimator pool.
class srs_estimator_pool : public srs_estimator
{
public:
  /// Creates a pool from a list of estimators. Ownership is transferred to the pool.
  explicit srs_estimator_pool(std::vector<std::unique_ptr<srs_estimator>> estimators_) :
    estimators(std::move(estimators_))
  {
  }

  // See interface for documentation.
  srs_estimator_result estimate(const resource_grid_reader& grid, const srs_estimator_configuration& config) override
  {
    return estimators.get().estimate(grid, config);
  }

private:
  concurrent_thread_local_object_pool<srs_estimator> estimators;
};

} // namespace srsran
