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

set_directory_properties(PROPERTIES LABELS "gateways")

add_executable(sctp_network_gateway_test
        sctp_network_server_test.cpp
        sctp_network_client_test.cpp
        sctp_network_gateway_test.cpp
        sctp_network_link_test.cpp)
target_link_libraries(sctp_network_gateway_test srsran_gateway srsran_support srslog gtest gtest_main)
add_test(sctp_network_gateway_test sctp_network_gateway_test)
set_tests_properties(sctp_network_gateway_test PROPERTIES LABELS "tsan")

add_executable(udp_network_gateway_test udp_network_gateway_test.cpp)
target_link_libraries(udp_network_gateway_test srsran_gateway srsran_support srslog gtest gtest_main)
add_test(udp_network_gateway_test udp_network_gateway_test)
set_tests_properties(udp_network_gateway_test PROPERTIES LABELS "tsan")

add_executable(udp_network_gateway_pool_depletion_test udp_network_gateway_pool_depletion_test.cpp)
target_link_libraries(udp_network_gateway_pool_depletion_test srsran_gateway srsran_support srslog gtest gtest_main)
add_test(udp_network_gateway_pool_depletion_test udp_network_gateway_pool_depletion_test)
set_tests_properties(udp_network_gateway_pool_depletion_test PROPERTIES LABELS "tsan")
