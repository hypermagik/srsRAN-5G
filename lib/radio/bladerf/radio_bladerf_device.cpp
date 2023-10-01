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

#include "radio_bladerf_device.h"

#include <libbladeRF.h>

using namespace srsran;

static double to_MHz(double value_Hz)
{
  return value_Hz * 1e-6;
}

radio_bladerf_device::radio_bladerf_device() : logger(srslog::fetch_basic_logger("RF")) {}

radio_bladerf_device::~radio_bladerf_device()
{
  if (device != nullptr) {
    fmt::print(BLADERF_LOG_PREFIX "Closing bladeRF...\n");

    bladerf_close(device);
    device = nullptr;
  }
}

bool radio_bladerf_device::open(const std::string& device_address)
{
  fmt::print(BLADERF_LOG_PREFIX "Opening bladeRF...\n");

  int status = bladerf_open(&device, device_address.c_str());
  if (status) {
    on_error("bladerf_open() failed - {}", bladerf_strerror(status));
    device = nullptr;
    return false;
  }

  bladerf_tuning_mode tuning_mode     = BLADERF_TUNING_MODE_HOST;
  const char*         env_tuning_mode = getenv("TUNING_MODE_FPGA");
  if (env_tuning_mode != nullptr) {
    tuning_mode = static_cast<bladerf_tuning_mode>(atoi(env_tuning_mode));
  }

  fmt::print(BLADERF_LOG_PREFIX "Setting {} tuning mode...\n",
             tuning_mode == BLADERF_TUNING_MODE_FPGA ? "FPGA" : "host");

  status = bladerf_set_tuning_mode(device, tuning_mode);
  if (status) {
    on_error("bladerf_set_tuning_mode() failed - {}", bladerf_strerror(status));
    return false;
  }

  fmt::print(BLADERF_LOG_PREFIX "Setting manual Rx gain mode...\n");

  status = bladerf_set_gain_mode(device, BLADERF_RX_X1, BLADERF_GAIN_MGC);
  if (status) {
    on_error("bladerf_set_gain_mode() failed - {}", bladerf_strerror(status));
    return false;
  }

  status = bladerf_set_gain_mode(device, BLADERF_RX_X2, BLADERF_GAIN_MGC);
  if (status) {
    on_error("bladerf_set_gain_mode() failed - {}", bladerf_strerror(status));
    return false;
  }

  bladerf_set_rfic_rx_fir(device, bladerf_rfic_rxfir::BLADERF_RFIC_RXFIR_BYPASS);
  bladerf_set_rfic_tx_fir(device, bladerf_rfic_txfir::BLADERF_RFIC_TXFIR_BYPASS);

  return true;
}

void radio_bladerf_device::set_log_level(srslog::basic_levels log_level)
{
  bladerf_log_level rf_log_level = bladerf_log_level::BLADERF_LOG_LEVEL_INFO;

  if (log_level == srslog::basic_levels::warning) {
    rf_log_level = bladerf_log_level::BLADERF_LOG_LEVEL_WARNING;
  } else if (log_level == srslog::basic_levels::info) {
    rf_log_level = bladerf_log_level::BLADERF_LOG_LEVEL_INFO;
  } else if (log_level == srslog::basic_levels::debug) {
    rf_log_level = bladerf_log_level::BLADERF_LOG_LEVEL_VERBOSE;
  } else {
    rf_log_level = bladerf_log_level::BLADERF_LOG_LEVEL_ERROR;
  }

  fmt::print(BLADERF_LOG_PREFIX "Setting log level to {}...\n", srslog::basic_level_to_string(log_level));

  bladerf_log_set_verbosity(rf_log_level);
}

bool radio_bladerf_device::set_sync_source(const radio_configuration::clock_sources& config)
{
  if (config.clock == radio_configuration::clock_sources::source::EXTERNAL) {
    fmt::print(BLADERF_LOG_PREFIX "Enabling reference clock input...\n");
    int status = bladerf_set_pll_enable(device, true);
    if (status != 0) {
      on_error("bladerf_set_pll_enable() failed - {}", bladerf_strerror(status));
      return false;
    }
    fmt::print(BLADERF_LOG_PREFIX "Waiting for lock...\n");
    bool locked = false;
    while (!locked) {
      status = bladerf_get_pll_lock_state(device, &locked);
      if (status) {
        on_error("bladerf_get_pll_lock_state() failed - {}", bladerf_strerror(status));
        return false;
      }
      sleep(1);
    }
  } else {
    bool enabled = false;
    int  status  = bladerf_get_pll_enable(device, &enabled);
    if (status) {
      on_error("bladerf_get_pll_enable() failed - {}", bladerf_strerror(status));
      return false;
    }
    if (enabled) {
      fmt::print(BLADERF_LOG_PREFIX "Disabling reference clock input...\n");
      status = bladerf_set_pll_enable(device, false);
      if (status) {
        on_error("bladerf_set_pll_enable() failed - {}", bladerf_strerror(status));
        return false;
      }
    }
  }

  return true;
}

std::unique_ptr<radio_bladerf_tx_stream>
radio_bladerf_device::create_tx_stream(radio_notification_handler&                        notifier,
                                       const radio_bladerf_tx_stream::stream_description& description)
{
  std::unique_ptr<radio_bladerf_tx_stream> stream =
      std::make_unique<radio_bladerf_tx_stream>(device, description, notifier);

  if (stream->is_successful()) {
    return stream;
  }

  fmt::print(BLADERF_LOG_PREFIX "Error: failed to create transmit stream {}. {}.\n",
             description.id,
             stream->get_error_message().c_str());
  return nullptr;
}

std::unique_ptr<radio_bladerf_rx_stream>
radio_bladerf_device::create_rx_stream(radio_notification_handler&                        notifier,
                                       const radio_bladerf_rx_stream::stream_description& description,
                                       radio_bladerf_tx_stream&                           tx_stream)
{
  std::unique_ptr<radio_bladerf_rx_stream> stream =
      std::make_unique<radio_bladerf_rx_stream>(device, description, notifier, tx_stream);

  if (stream->is_successful()) {
    return stream;
  }

  fmt::print(BLADERF_LOG_PREFIX "Error: failed to create receive stream {}. {}.\n",
             description.id,
             stream->get_error_message().c_str());
  return nullptr;
}

bool radio_bladerf_device::set_tx_rate(double& actual_rate, double rate)
{
  fmt::print(BLADERF_LOG_PREFIX "Setting Tx Rate to {:.2f} MHz...\n", to_MHz(rate));

  bladerf_sample_rate rf_actual_rate;

  int status = bladerf_set_sample_rate(device, BLADERF_TX_X1, static_cast<bladerf_sample_rate>(rate), &rf_actual_rate);
  if (status != 0) {
    on_error("bladerf_set_sample_rate() failed - {}", bladerf_strerror(status));
    return false;
  }

  actual_rate = rf_actual_rate;

  bladerf_bandwidth rf_actual_bw;

  status = bladerf_set_bandwidth(device, BLADERF_TX_X1, static_cast<bladerf_bandwidth>(rate), &rf_actual_bw);
  if (status != 0) {
    on_error("bladerf_set_bandwidth() failed - {}", bladerf_strerror(status));
    return false;
  }

  fmt::print(BLADERF_LOG_PREFIX "... Tx sampling rate set to {:.2f} MHz and filter BW set to {:.2f} MHz\n",
             to_MHz(rf_actual_rate),
             to_MHz(rf_actual_bw));
  return true;
}

bool radio_bladerf_device::set_rx_rate(double& actual_rate, double rate)
{
  fmt::print(BLADERF_LOG_PREFIX "Setting Rx Rate to {:.2f} MHz...\n", to_MHz(rate));

  bladerf_sample_rate rf_actual_rate;

  int status = bladerf_set_sample_rate(device, BLADERF_RX_X1, static_cast<bladerf_sample_rate>(rate), &rf_actual_rate);
  if (status != 0) {
    on_error("bladerf_set_sample_rate() failed - {}", bladerf_strerror(status));
    return false;
  }

  actual_rate = rf_actual_rate;
  bladerf_bandwidth rf_actual_bw;

  status = bladerf_set_bandwidth(device, BLADERF_RX_X1, static_cast<bladerf_bandwidth>(rate * 0.8), &rf_actual_bw);
  if (status != 0) {
    on_error("bladerf_set_bandwidth() failed - {}", bladerf_strerror(status));
    return false;
  }

  fmt::print(BLADERF_LOG_PREFIX "... Rx sampling rate set to {:.2f} MHz and filter BW set to {:.2f} MHz\n",
             to_MHz(rf_actual_rate),
             to_MHz(rf_actual_bw));
  return true;
}

bool radio_bladerf_device::set_tx_gain(unsigned ch, double gain)
{
  fmt::print(BLADERF_LOG_PREFIX "Setting channel {} Tx gain to {:.2f} dB...\n", ch, gain);

  int status = bladerf_set_gain(device, ch == 0 ? BLADERF_TX_X1 : BLADERF_TX_X2, static_cast<bladerf_gain>(gain));
  if (status != 0) {
    on_error("bladerf_set_gain() failed - {}", bladerf_strerror(status));
    return false;
  }

  return true;
}

bool radio_bladerf_device::set_rx_gain(unsigned ch, double gain)
{
  fmt::print(BLADERF_LOG_PREFIX "Setting channel {} Rx gain to {:.2f} dB...\n", ch, gain);

  int status = bladerf_set_gain(device, ch == 0 ? BLADERF_RX_X1 : BLADERF_RX_X2, static_cast<bladerf_gain>(gain));
  if (status != 0) {
    on_error("bladerf_set_gain() failed - {}", bladerf_strerror(status));
    return false;
  }

  return true;
}

bool radio_bladerf_device::set_tx_freq(uint32_t ch, const radio_configuration::lo_frequency& config)
{
  fmt::print(
      BLADERF_LOG_PREFIX "Setting channel {} Tx frequency to {} MHz...\n", ch, to_MHz(config.center_frequency_hz));

  int status = bladerf_set_frequency(device,
                                     ch == 0 ? BLADERF_TX_X1 : BLADERF_TX_X2,
                                     static_cast<bladerf_frequency>(round(config.center_frequency_hz)));
  if (status != 0) {
    on_error("bladerf_set_frequency() failed - {}", bladerf_strerror(status));
    return false;
  }

  return true;
}

bool radio_bladerf_device::set_rx_freq(uint32_t ch, const radio_configuration::lo_frequency& config)
{
  fmt::print(
      BLADERF_LOG_PREFIX "Setting channel {} Rx frequency to {} MHz...\n", ch, to_MHz(config.center_frequency_hz));

  int status = bladerf_set_frequency(device,
                                     ch == 0 ? BLADERF_RX_X1 : BLADERF_RX_X2,
                                     static_cast<bladerf_frequency>(round(config.center_frequency_hz)));
  if (status != 0) {
    on_error("bladerf_set_frequency() failed - {}", bladerf_strerror(status));
    return false;
  }

  return true;
}

baseband_gateway_timestamp radio_bladerf_device::get_time_now()
{
  bladerf_timestamp timestamp;

  int status = bladerf_get_timestamp(device, BLADERF_RX, &timestamp);
  if (status != 0) {
    fmt::print(BLADERF_LOG_PREFIX "Failed to get current Rx timestamp: {}", bladerf_strerror(status));
  }

  return timestamp;
}
