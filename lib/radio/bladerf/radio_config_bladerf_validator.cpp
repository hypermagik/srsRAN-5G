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

#include "radio_config_bladerf_validator.h"
#include "srsran/srslog/srslog.h"
#include "fmt/format.h"
#include <libbladeRF.h>
#include <regex>
#include <set>

using namespace srsran;

static bool validate_clock_sources(const radio_configuration::clock_sources& sources)
{
  static const std::set<radio_configuration::clock_sources::source> valid_clock_sources = {
      radio_configuration::clock_sources::source::DEFAULT,
      radio_configuration::clock_sources::source::INTERNAL,
      radio_configuration::clock_sources::source::EXTERNAL};

  if (valid_clock_sources.count(sources.clock) == 0) {
    fmt::print("Invalid clock source.\n");
    return false;
  }

  if (valid_clock_sources.count(sources.sync) == 0) {
    fmt::print("Invalid sync source.\n");
    return false;
  }

  return true;
}

static bool validate_lo_freq(const radio_configuration::lo_frequency& lo_freq, bool is_tx)
{
  if (!std::isnormal(lo_freq.center_frequency_hz)) {
    fmt::print("{} center frequency must be non-zero, NAN nor infinite.\n", is_tx ? "TX" : "RX");
    return false;
  }

  return true;
}

static bool validate_channel(const radio_configuration::channel& channel, bool is_tx)
{
  if (!validate_lo_freq(channel.freq, is_tx)) {
    return false;
  }

  if (std::isnan(channel.gain_dB) || std::isinf(channel.gain_dB)) {
    fmt::print("Channel gain must not be NAN nor infinite.\n");
    return false;
  }

  return true;
}

static bool validate_stream(const radio_configuration::stream& stream, bool is_tx)
{
  if (stream.channels.empty()) {
    fmt::print("Streams must contain at least one channel.\n");
    return false;
  }

  for (const radio_configuration::channel& channel : stream.channels) {
    if (!validate_channel(channel, is_tx)) {
      return false;
    }
  }

  return true;
}

static bool validate_sampling_rate(double sampling_rate)
{
  if (!std::isnormal(sampling_rate)) {
    fmt::print("The sampling rate must be non-zero, NAN nor infinite.\n");
    return false;
  }

  if (sampling_rate < 0.0) {
    fmt::print("The sampling rate must be greater than zero.\n");
    return false;
  }

  return true;
}

static bool validate_otw_format(radio_configuration::over_the_wire_format otw_format)
{
  if (otw_format == radio_configuration::over_the_wire_format::SC12) {
    struct bladerf_version version;
    bladerf_version(&version);

    if (version.major < 2 || version.minor < 6) {
      fmt::print(
          "SC12 OTW format requires libbladeRF version >= 2.6.0 with support for BLADERF_FORMAT_SC16_Q11_PACKED_META");
      return false;
    }
  }

  return true;
}

bool radio_config_bladerf_config_validator::is_configuration_valid(const radio_configuration::radio& config) const
{
  if (!validate_clock_sources(config.clock)) {
    return false;
  }

  if (config.tx_streams.size() != config.rx_streams.size()) {
    fmt::print("Transmit and receive number of streams must be equal.\n");
    return false;
  }

  if (config.tx_streams.empty()) {
    fmt::print("At least one transmit and one receive stream must be configured.\n");
    return false;
  }

  for (const radio_configuration::stream& tx_stream : config.tx_streams) {
    if (!validate_stream(tx_stream, true)) {
      return false;
    }
  }

  for (const radio_configuration::stream& rx_stream : config.rx_streams) {
    if (!validate_stream(rx_stream, false)) {
      return false;
    }
  }

  if (!validate_sampling_rate(config.sampling_rate_hz)) {
    return false;
  }

  if (!validate_otw_format(config.otw_format)) {
    return false;
  }

  return true;
}
