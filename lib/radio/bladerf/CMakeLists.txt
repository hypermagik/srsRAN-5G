#
# Copyright 2021-2023 Software Radio Systems Limited
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

set(SOURCES_BLADERF
        radio_config_bladerf_validator.cpp
        radio_bladerf_device.cpp
        radio_bladerf_impl.cpp
        radio_bladerf_rx_stream.cpp
        radio_bladerf_tx_stream.cpp)
add_library(srsran_radio_bladerf STATIC ${SOURCES_BLADERF})

target_link_libraries(srsran_radio_bladerf ${BLADERF_LIBRARIES} srsvec)

install(TARGETS srsran_radio_bladerf EXPORT srsran_export)
