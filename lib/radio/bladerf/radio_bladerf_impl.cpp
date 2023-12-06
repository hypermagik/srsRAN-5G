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

#include "radio_bladerf_impl.h"

using namespace srsran;

bool radio_session_bladerf_impl::set_tx_gain(unsigned port_idx, double gain_dB)
{
  if (port_idx >= tx_port_map.size()) {
    fmt::print(
        "Error: transmit port index ({}) exceeds the number of ports ({}).\n", port_idx, (int)tx_port_map.size());
    return false;
  }

  // Setup gain.
  if (!device.set_tx_gain(port_idx, gain_dB)) {
    fmt::print("Error: setting gain for transmitter {}. {}\n", port_idx, device.get_error_message());
  }

  return true;
}

bool radio_session_bladerf_impl::set_rx_gain(unsigned port_idx, double gain_dB)
{
  if (port_idx >= rx_port_map.size()) {
    fmt::print("Error: receive port index ({}) exceeds the number of ports ({}).\n", port_idx, (int)rx_port_map.size());
    return false;
  }

  // Setup gain.
  if (!device.set_rx_gain(port_idx, gain_dB)) {
    fmt::print("Error: setting gain for receiver {}. {}\n", port_idx, device.get_error_message());
    return false;
  }

  return true;
}

bool radio_session_bladerf_impl::set_tx_freq(unsigned port_idx, radio_configuration::lo_frequency frequency)
{
  if (port_idx >= tx_port_map.size()) {
    fmt::print(
        "Error: transmit port index ({}) exceeds the number of ports ({}).\n", port_idx, (int)tx_port_map.size());
    return false;
  }

  // Setup frequency.
  if (!device.set_tx_freq(port_idx, frequency)) {
    fmt::print("Error: setting frequency for transmitter {}. {}\n", port_idx, device.get_error_message());
    return false;
  }

  return true;
}

bool radio_session_bladerf_impl::set_rx_freq(unsigned port_idx, radio_configuration::lo_frequency frequency)
{
  if (port_idx >= rx_port_map.size()) {
    fmt::print("Error: receive port index ({}) exceeds the number of ports ({}).\n", port_idx, (int)tx_port_map.size());
    return false;
  }

  // Setup frequency.
  if (!device.set_rx_freq(port_idx, frequency)) {
    fmt::print("Error: setting frequency for receiver {}. {}.\n", port_idx, device.get_error_message());
    return false;
  }

  return true;
}

bool radio_session_bladerf_impl::set_tx_rate(unsigned port_idx, double sampling_rate_hz)
{
  if (port_idx >= tx_port_map.size()) {
    fmt::print(
        "Error: transmit port index ({}) exceeds the number of ports ({}).\n", port_idx, (int)tx_port_map.size());
    return false;
  }

  // Setup frequency.
  if (!device.set_tx_rate(port_idx, sampling_rate_hz)) {
    fmt::print("Error: setting sampling rate for transmitter {}. {}\n", port_idx, device.get_error_message());
    return false;
  }

  return true;
}

bool radio_session_bladerf_impl::set_rx_rate(unsigned port_idx, double sampling_rate_hz)
{
  if (port_idx >= rx_port_map.size()) {
    fmt::print("Error: receive port index ({}) exceeds the number of ports ({}).\n", port_idx, (int)tx_port_map.size());
    return false;
  }

  // Setup frequency.
  if (!device.set_rx_rate(port_idx, sampling_rate_hz)) {
    fmt::print("Error: setting sampling rate for receiver {}. {}.\n", port_idx, device.get_error_message());
    return false;
  }

  return true;
}

bool radio_session_bladerf_impl::start_streams(baseband_gateway_timestamp init_time)
{
  // Prevent multiple threads from starting streams simultaneously.
  std::unique_lock<std::mutex> lock(stream_start_mutex);

  if (!stream_start_required) {
    return true;
  }

  // Flag stream start is no longer required.
  stream_start_required = false;

  // Issue all streams to start.
  for (auto& bb_gateway : bb_gateways) {
    if (!bb_gateway->get_tx_stream().start()) {
      return false;
    }
    if (!bb_gateway->get_rx_stream().start(init_time)) {
      return false;
    }
  }

  return true;
}

radio_session_bladerf_impl::radio_session_bladerf_impl(const radio_configuration::radio& radio_config,
                                                       radio_notification_handler&       notifier_) :
  notifier(notifier_)
{
  // Open device.
  if (!device.open(radio_config.args)) {
    fmt::print("Failed to open device with address '{}': {}\n", radio_config.args, device.get_error_message());
    return;
  }

  // Set the logging level.
  device.set_log_level(radio_config.log_level);

  // Set sync source.
  if (!device.set_sync_source(radio_config.clock)) {
    fmt::print("Error: couldn't set sync source: {}\n", device.get_error_message());
    return;
  }

  // Lists of stream descriptions.
  std::vector<radio_bladerf_tx_stream::stream_description> tx_stream_description_list;
  std::vector<radio_bladerf_rx_stream::stream_description> rx_stream_description_list;

  // For each transmit stream, create stream and configure RF ports.
  for (unsigned stream_idx = 0; stream_idx != radio_config.tx_streams.size(); ++stream_idx) {
    // Select stream.
    const radio_configuration::stream& stream = radio_config.tx_streams[stream_idx];

    // Prepare stream description.
    radio_bladerf_tx_stream::stream_description stream_description = {};
    stream_description.id                                          = stream_idx;
    stream_description.otw_format                                  = radio_config.otw_format;
    stream_description.srate_Hz                                    = radio_config.sampling_rate_hz;
    stream_description.nof_channels                                = stream.channels.size();

    // Setup ports.
    for (unsigned channel_idx = 0; channel_idx != stream.channels.size(); ++channel_idx) {
      // Save the stream and channel indexes for the port.
      tx_port_map.emplace_back(port_to_stream_channel(stream_idx, channel_idx));

      // Extract port configuration.
      const radio_configuration::channel& channel = stream.channels[channel_idx];

      // Setup frequency.
      if (!set_tx_freq(channel_idx, channel.freq)) {
        return;
      }

      // Set Tx rate.
      if (!set_tx_rate(channel_idx, radio_config.sampling_rate_hz)) {
        return;
      }

      // Setup gain.
      if (!set_tx_gain(channel_idx, channel.gain_dB)) {
        return;
      }
    }

    // Add stream description to the list.
    tx_stream_description_list.emplace_back(stream_description);
  }

  // For each receive stream, create stream and configure RF ports.
  for (unsigned stream_idx = 0; stream_idx != radio_config.rx_streams.size(); ++stream_idx) {
    // Select stream.
    const radio_configuration::stream& stream = radio_config.rx_streams[stream_idx];

    // Prepare stream description.
    radio_bladerf_rx_stream::stream_description stream_description = {};
    stream_description.id                                          = stream_idx;
    stream_description.otw_format                                  = radio_config.otw_format;
    stream_description.srate_Hz                                    = radio_config.sampling_rate_hz;
    stream_description.nof_channels                                = stream.channels.size();

    // Setup ports.
    for (unsigned channel_idx = 0; channel_idx != stream.channels.size(); ++channel_idx) {
      // Save the stream and channel indexes for the port.
      rx_port_map.emplace_back(port_to_stream_channel(stream_idx, channel_idx));

      // Extract port configuration.
      const radio_configuration::channel& channel = stream.channels[channel_idx];

      // Setup frequency.
      if (!set_rx_freq(channel_idx, channel.freq)) {
        return;
      }

      // Set Rx rate.
      if (!set_rx_rate(channel_idx, radio_config.sampling_rate_hz)) {
        return;
      }

      // Setup gain.
      if (!set_rx_gain(channel_idx, channel.gain_dB)) {
        return;
      }
    }

    // Add stream description to the list.
    rx_stream_description_list.emplace_back(stream_description);
  }

  // Create baseband gateways.
  for (unsigned i_stream = 0; i_stream != radio_config.tx_streams.size(); ++i_stream) {
    bb_gateways.emplace_back(std::make_unique<radio_bladerf_baseband_gateway>(
        device, notifier, tx_stream_description_list[i_stream], rx_stream_description_list[i_stream]));

    // Early return if the gateway was not successfully created.
    if (!bb_gateways.back()->is_successful()) {
      return;
    }
  }

  // Transition to successfully initialized.
  state = states::SUCCESSFUL_INIT;
}

void radio_session_bladerf_impl::stop()
{
  // Transition state to stop.
  state = states::STOP;

  // Signal stop for each transmit stream.
  for (auto& gateway : bb_gateways) {
    gateway->get_tx_stream().stop();
  }

  // Signal stop for each receive stream.
  for (auto& gateway : bb_gateways) {
    gateway->get_rx_stream().stop();
  }
}

void radio_session_bladerf_impl::start(baseband_gateway_timestamp init_time)
{
  if (!start_streams(init_time)) {
    fmt::print("Failed to start streams.\n");
  }
}

baseband_gateway_timestamp radio_session_bladerf_impl::read_current_time()
{
  return device.get_time_now();
}

std::unique_ptr<radio_session> radio_factory_bladerf_impl::create(const radio_configuration::radio& config,
                                                                  task_executor&                    async_task_executor,
                                                                  radio_notification_handler&       notifier)
{
  std::unique_ptr<radio_session_bladerf_impl> session = std::make_unique<radio_session_bladerf_impl>(config, notifier);
  if (!session->is_successful()) {
    return nullptr;
  }

  return std::move(session);
}

radio_config_bladerf_config_validator radio_factory_bladerf_impl::config_validator;
