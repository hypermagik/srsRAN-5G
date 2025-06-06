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


set(SOURCES
  ciphering_engine_nea1.cpp
  ciphering_engine_nea2.cpp
  ciphering_engine_nea3.cpp
  integrity_engine_generic.cpp
  integrity_engine_nia2_cmac.cpp
  integrity_engine_nia2_non_cmac.cpp
  security.cpp
  security_engine_impl.cpp
  s3g.cpp
  zuc.cpp
)

add_library(srsran_security STATIC ${SOURCES})
target_include_directories(srsran_security PUBLIC ${SEC_INCLUDE_DIRS})
target_link_libraries(srsran_security ${SEC_LIBRARIES})
