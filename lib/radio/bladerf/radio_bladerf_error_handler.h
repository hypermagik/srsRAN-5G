/*
 *
 * Copyright 2021-2023 Software Radio Systems Limited
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

#include "fmt/format.h"
#include <boost/exception/diagnostic_information.hpp>
#include <string>

#pragma GCC diagnostic push
#ifdef __clang__
#pragma GCC diagnostic ignored "-Wall"
#else // __clang__
#pragma GCC diagnostic ignored "-Wsuggest-override"
#endif // __clang__
#pragma GCC diagnostic pop

namespace srsran {

class bladerf_error_handler
{
public:
  template <typename S, typename... Args>
  void on_error(const S& format_str, Args&&... args)
  {
    fmt::memory_buffer str_buf;
    fmt::format_to(std::back_inserter(str_buf), format_str, std::forward<Args>(args)...);
    error_message = to_string(str_buf);
  }

  bool               is_successful() const { return error_message.empty(); }
  const std::string& get_error_message() const { return error_message; }

private:
  std::string error_message;
};

} // namespace srsran
