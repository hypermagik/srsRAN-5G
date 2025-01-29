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

#include "radio_bladerf_error_handler.h"
#include "radio_bladerf_rx_stream.h"
#include "radio_bladerf_tx_stream.h"
#include "srsran/radio/radio_session.h"
#include "srsran/srslog/srslog.h"

struct bladerf;

namespace srsran {

class radio_bladerf_device : public bladerf_error_handler
{
public:
  radio_bladerf_device();
  ~radio_bladerf_device();

  bool open(const std::string& device_address);

  void set_log_level(srslog::basic_levels log_level);
  bool set_sync_source(const radio_configuration::clock_sources& config);

  std::unique_ptr<radio_bladerf_tx_stream>
  create_tx_stream(radio_notification_handler&                        notifier,
                   const radio_bladerf_tx_stream::stream_description& description);
  std::unique_ptr<radio_bladerf_rx_stream>
  create_rx_stream(radio_notification_handler&                        notifier,
                   const radio_bladerf_rx_stream::stream_description& description,
                   radio_bladerf_tx_stream&                           tx_stream);

  bool set_tx_rate(unsigned ch, double rate);
  bool set_rx_rate(unsigned ch, double rate);
  void set_fir2();
  bool set_tx_gain(unsigned ch, double gain);
  bool set_rx_gain(unsigned ch, double gain);
  bool set_tx_freq(unsigned ch, const radio_configuration::lo_frequency& config);
  bool set_rx_freq(unsigned ch, const radio_configuration::lo_frequency& config);

  baseband_gateway_timestamp get_time_now();

private:
  srslog::basic_logger& logger;
  bladerf*              device = nullptr;
};

} // namespace srsran
