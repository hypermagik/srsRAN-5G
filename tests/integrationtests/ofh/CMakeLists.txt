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

set_directory_properties(PROPERTIES LABELS "ofh")

add_executable(ofh_integration_test ofh_integration_test.cpp)
target_link_libraries(ofh_integration_test
        srslog
        srsran_ru_ofh
        srsran_phy_support
        srsran_channel_precoder
        srsran_support)
add_test(ofh_integration_test ofh_integration_test -d \'{0,1}\')
set_tests_properties(ofh_integration_test PROPERTIES LABELS "tsan;NO_MEMCHECK")
