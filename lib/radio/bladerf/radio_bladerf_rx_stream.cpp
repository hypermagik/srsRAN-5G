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

#include "radio_bladerf_rx_stream.h"
#include "srsran/srsvec/conversion.h"

#include <libbladeRF.h>

using namespace srsran;

radio_bladerf_rx_stream::radio_bladerf_rx_stream(bladerf*                    device_,
                                                 const stream_description&   description,
                                                 radio_notification_handler& notifier_) :
  device(device_),
  stream_id(description.id),
  srate_Hz(description.srate_Hz),
  nof_channels(description.nof_channels),
  notifier(notifier_)
{
  srsran_assert(std::isnormal(srate_Hz) && (srate_Hz > 0.0), "Invalid sampling rate {}.", srate_Hz);
  srsran_assert(nof_channels == 1 || nof_channels == 2, "Invalid number of channels {}.", nof_channels);
  srsran_assert((description.otw_format == radio_configuration::over_the_wire_format::DEFAULT ||
                 description.otw_format == radio_configuration::over_the_wire_format::SC8 ||
                 description.otw_format == radio_configuration::over_the_wire_format::SC16),
                "Invalid over the wire format {}.",
                description.otw_format);

  if (description.otw_format == radio_configuration::over_the_wire_format::SC8) {
    sample_size = sizeof(int8_t);
    iq_scale    = 128.f;
  } else {
    sample_size = sizeof(int16_t);
    iq_scale    = 2048.f;
  }

  unsigned num_transfers = NUM_TRANSFERS;
  unsigned num_buffers   = NUM_BUFFERS;
  buffer_size            = 2048 + 1024 * (int)(srate_Hz / 1e7);

  const char* env_num_transfers = getenv("RX_NUM_TRANSFERS");
  if (env_num_transfers != nullptr) {
    num_transfers = atoi(env_num_transfers);
  }
  const char* env_num_buffers = getenv("RX_NUM_BUFFERS");
  if (env_num_buffers != nullptr) {
    num_buffers = atoi(env_num_buffers);
  }
  const char* env_buffer_size = getenv("RX_BUFFER_SIZE");
  if (env_buffer_size != nullptr) {
    buffer_size = atoi(env_buffer_size);
  }

  fmt::print("Creating Rx stream with {} channels, {}-bit samples at {} MHz and {} buffers of {} samples...\n",
             nof_channels,
             sample_size == sizeof(int8_t) ? "8" : "16",
             srate_Hz / 1e6,
             num_buffers,
             buffer_size);

  const bladerf_channel_layout layout =
      nof_channels == 1 ? bladerf_channel_layout::BLADERF_RX_X1 : bladerf_channel_layout::BLADERF_RX_X2;
  const bladerf_format format = sample_size == sizeof(int8_t) ? bladerf_format::BLADERF_FORMAT_SC8_Q7_META
                                                              : bladerf_format::BLADERF_FORMAT_SC16_Q11_META;

  // Configure the device's Rx modules for use with the sync interface.
  int status = bladerf_sync_config(device, layout, format, num_buffers, buffer_size, num_transfers, TIMEOUT_MS);
  if (status != 0) {
    on_error("bladerf_sync_config(BLADERF_RX_X{}) failed - {}", nof_channels, bladerf_strerror(status));
    return;
  }

  state = states::SUCCESSFUL_INIT;
}

bool radio_bladerf_rx_stream::start(baseband_gateway_timestamp init_time)
{
  if (state != states::SUCCESSFUL_INIT) {
    return true;
  }

  for (size_t channel = 0; channel < nof_channels; channel++) {
    fmt::print("Enabling Rx module for channel {}...\n", channel + 1);

    int status = bladerf_enable_module(device, BLADERF_CHANNEL_RX(channel), true);
    if (status != 0) {
      on_error("bladerf_enable_module(BLADERF_CHANNEL_RX({}), true) failed - {}", channel, bladerf_strerror(status));
      return false;
    }
  }

  start_time = init_time;

  // Transition to streaming state.
  state = states::STREAMING;

  return true;
}

baseband_gateway_receiver::metadata radio_bladerf_rx_stream::receive(baseband_gateway_buffer_writer& buffs)
{
  // Make sure the number of channels is equal.
  srsran_assert(buffs.get_nof_channels() == nof_channels, "Number of channels does not match.");

  const unsigned nsamples = buffs.get_nof_samples();

  if (nsamples * nof_channels * 2 * sample_size > sizeof(buffer)) {
    fmt::print(
        "Error: nsamples %d exceeds buffer size %zu\n", nsamples, sizeof(buffer) / nof_channels / 2 / sample_size);
    return {};
  }

  baseband_gateway_receiver::metadata ret = {};

  bladerf_metadata meta;
  memset(&meta, 0, sizeof(meta));

  if (start_time != 0) {
    meta.timestamp = start_time;
    start_time     = 0;
  } else {
    meta.flags = BLADERF_META_FLAG_RX_NOW;
  }

  unsigned int rxd_samples = 0;
  while (rxd_samples != nsamples) {
    // Ignore reception if it is not streaming.
    if (state != states::STREAMING) {
      return {};
    }

    const unsigned int rx_samples = nsamples - rxd_samples;

    int status = bladerf_sync_rx(
        device, buffer + rxd_samples * nof_channels * 2 * sample_size, rx_samples * nof_channels, &meta, TIMEOUT_MS);

    if (status == BLADERF_ERR_TIMEOUT) {
      fmt::print("Error: reception timed out\n");
      return {};
    }

    if (status != 0) {
      fmt::print("Error: failed to receive packet. {}.\n", bladerf_strerror(status));
      return {};
    }

    if (meta.status & BLADERF_META_STATUS_OVERRUN) {
      radio_notification_handler::event_description event = {};
      event.stream_id                                     = stream_id;
      event.channel_id                                    = radio_notification_handler::UNKNOWN_ID;
      if (rx_samples * nof_channels != meta.actual_count) {
        event.source = radio_notification_handler::event_source::RECEIVE;
        event.type   = radio_notification_handler::event_type::OVERFLOW;
      } else {
        event.source = radio_notification_handler::event_source::TRANSMIT;
        event.type   = radio_notification_handler::event_type::UNDERFLOW;
      }
      notifier.on_radio_rt_event(event);
    }

    if (rxd_samples == 0) {
      ret.ts = meta.timestamp;
    }

    rxd_samples += meta.actual_count / nof_channels;
  }

  const bladerf_channel_layout layout =
      nof_channels == 1 ? bladerf_channel_layout::BLADERF_RX_X1 : bladerf_channel_layout::BLADERF_RX_X2;
  const bladerf_format format =
      sample_size == sizeof(int8_t) ? bladerf_format::BLADERF_FORMAT_SC8_Q7 : bladerf_format::BLADERF_FORMAT_SC16_Q11;

  int status = bladerf_deinterleave_stream_buffer(layout, format, nsamples * nof_channels, buffer);
  if (status != 0) {
    fmt::print("Error: could not deinterleave stream buffer. {}.\n", bladerf_strerror(status));
    return {};
  }

  for (size_t channel = 0; channel < nof_channels; channel++) {
    srsran_assert(nsamples == buffs[channel].size(),
                  "nsamples not equal to buffer size for channel {} - {} != {}",
                  channel + 1,
                  nsamples,
                  buffs[channel].size());

    if (sample_size == sizeof(int8_t)) {
      const srsran::span<const int8_t> samples{buffer, nsamples * 2 * nof_channels};
      const srsran::span<const int8_t> x = samples.subspan(nsamples * 2 * channel, nsamples * 2);
      srsran::srsvec::convert(x, iq_scale, buffs[channel]);
    } else {
      const srsran::span<const int16_t> samples{reinterpret_cast<const int16_t*>(buffer), nsamples * 2 * nof_channels};
      const srsran::span<const int16_t> x = samples.subspan(nsamples * 2 * channel, nsamples * 2);
      srsran::srsvec::convert(x, iq_scale, buffs[channel]);
    }
  }

  return ret;
}

void radio_bladerf_rx_stream::stop()
{
  // Transition state to stop before locking to prevent real time priority thread owning the lock constantly.
  state = states::STOP;

  for (size_t channel = 0; channel < nof_channels; channel++) {
    fmt::print("Disabling Rx module for channel {}...\n", channel + 1);

    int status = bladerf_enable_module(device, BLADERF_CHANNEL_RX(channel), false);
    if (status != 0) {
      on_error("bladerf_enable_module(BLADERF_CHANNEL_RX{}, false) failed - {}", channel, bladerf_strerror(status));
    }
  }
}

unsigned radio_bladerf_rx_stream::get_buffer_size() const
{
  return (buffer_size - (buffer_size / 1024) * 8) / nof_channels;
}
