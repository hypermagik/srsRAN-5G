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
#include "srsran/gateways/baseband/baseband_gateway_receiver.h"
#include "srsran/gateways/baseband/buffer/baseband_gateway_buffer_writer.h"
#include "srsran/radio/radio_configuration.h"
#include "srsran/radio/radio_notification_handler.h"
#include <mutex>

struct bladerf;

namespace srsran {

/// Implements a gateway receiver based on bladeRF receive stream.
class radio_bladerf_rx_stream : public baseband_gateway_receiver, public bladerf_error_handler
{
private:
  /// The number of active USB transfers that may be in-flight at any given time.
  static constexpr unsigned NUM_TRANSFERS = 64;
  /// Number of device buffers.
  static constexpr unsigned NUM_BUFFERS = 256;
  /// Receive timeout.
  static constexpr unsigned TIMEOUT_MS = 200;

  /// Defines the Rx stream internal states.
  enum class states { UNINITIALIZED, SUCCESSFUL_INIT, STREAMING, STOP };
  /// Indicates the current stream state.
  std::atomic<states> state = {states::UNINITIALIZED};

  /// Owns the bladeRF Tx stream.
  bladerf* const device;
  /// Indicates the stream identification for notifications.
  unsigned stream_id;
  /// Sampling rate in Hz.
  double srate_Hz;
  /// Indicates the number of channels.
  unsigned nof_channels;
  /// Sample size.
  size_t sample_size;
  /// Buffer size.
  unsigned buffer_size;
  /// IQ scale.
  float iq_scale;
  /// Conversion buffer.
  int8_t buffer[128 * 1024 * sizeof(int16_t)];
  /// Start time.
  baseband_gateway_timestamp start_time;
  /// Radio notification interface.
  radio_notification_handler& notifier;

public:
  /// Describes the necessary parameters to create an bladeRF reception stream.
  struct stream_description {
    /// Identifies the stream.
    unsigned id;
    /// Over-the-wire format.
    radio_configuration::over_the_wire_format otw_format;
    /// Sampling rate in hertz.
    double srate_Hz;
    /// Indicates the number of channels.
    unsigned nof_channels;
  };

  /// \brief Constructs a bladeRF receive stream.
  /// \param[in] device Provides the bladeRF device handle.
  /// \param[in] description Provides the stream configuration parameters.
  /// \param[in] notifier_ Provides the radio event notification handler.
  radio_bladerf_rx_stream(bladerf*                    device,
                          const stream_description&   description,
                          radio_notification_handler& notifier_);

  /// Starts the stream reception.
  bool start(baseband_gateway_timestamp init_time);

  /// Gets the optimal reception buffer size.
  unsigned get_buffer_size() const;

  // See interface for documentation.
  metadata receive(baseband_gateway_buffer_writer& data) override;

  /// Stops the reception stream.
  void stop();
};
} // namespace srsran
