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

set_directory_properties(PROPERTIES LABELS "phy")

add_executable(dft_processor_benchmark dft_processor_benchmark.cpp)
target_link_libraries(dft_processor_benchmark srslog srsran_generic_funcs)

add_test(dft_processor_generic_benchmark dft_processor_benchmark -F generic -R 10 -s)
if (ENABLE_FFTW AND FFTW3F_FOUND)
    add_test(dft_processor_fftw_benchmark dft_processor_benchmark -F fftw -R 10 -s)
endif (ENABLE_FFTW AND FFTW3F_FOUND)