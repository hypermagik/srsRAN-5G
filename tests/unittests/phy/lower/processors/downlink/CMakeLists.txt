#
# Copyright 2021-2025 Software Radio Systems Limited
#
# This file is part of srsRAN
#
# srsRAN is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of
# the License, or (at your option) any later version.
#
# srsRAN is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# A copy of the GNU Affero General Public License can be found in
# the LICENSE file in the top-level directory of this distribution
# and at http://www.gnu.org/licenses/.
#

add_subdirectory(pdxch)

add_executable(lower_phy_downlink_processor_test lower_phy_downlink_processor_test.cpp)
target_link_libraries(lower_phy_downlink_processor_test srsran_lower_phy_downlink_processor srslog gtest gtest_main)
add_test(lower_phy_downlink_processor_test lower_phy_downlink_processor_test)