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
#include "srsran/support/unique_thread.h"

using namespace srsran;

radio_bladerf_rx_stream::radio_bladerf_rx_stream(bladerf*                    device_,
                                                 const stream_description&   description,
                                                 radio_notification_handler& notifier_,
                                                 radio_bladerf_tx_stream&    tx_stream_) :
  stream_id(description.id),
  srate_Hz(description.srate_Hz),
  nof_channels(description.nof_channels),
  notifier(notifier_),
  tx_stream(tx_stream_),
  device(device_)
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

  // Around 10 transfers per 1ms, for more resolution.
  samples_per_buffer = nof_channels * srate_Hz / 1e3 / 10.f;
  samples_per_buffer = (samples_per_buffer + 1023) & ~1023;

  const char* env_buffer_size = getenv("RX_BUFFER_SIZE");
  if (env_buffer_size != nullptr) {
    samples_per_buffer = atoi(env_buffer_size);
  }

  nof_transfers = 16;

  const char* env_nof_transfers = getenv("RX_TRANSFERS");
  if (env_nof_transfers != nullptr) {
    nof_transfers = atoi(env_nof_transfers);
  }

  // Not using any additional buffers.
  unsigned nof_buffers = nof_transfers + 1;

  const char* env_print_counters = getenv("STATS");
  if (env_print_counters != nullptr) {
    print_counters = atoi(env_print_counters);
  }

  fmt::print(BLADERF_LOG_PREFIX "Creating Rx stream with {} channels and {}-bit samples at {} MHz...\n",
             nof_channels,
             sample_size == sizeof(int8_t) ? "8" : "16",
             srate_Hz / 1e6);

  samples_per_message             = bytes_to_samples(message_size - meta_size);
  samples_per_buffer_without_meta = samples_per_buffer - (samples_per_buffer / 1024) * 8;
  bytes_per_buffer                = samples_to_bytes(samples_per_buffer);
  us_per_buffer                   = 1000000 * samples_per_buffer_without_meta / nof_channels / srate_Hz;

  fmt::print(BLADERF_LOG_PREFIX "...{} transfers, {} buffers, {}/{} samples/buffer, {} bytes/buffer, {}us/buffer...\n",
             nof_transfers,
             nof_buffers,
             samples_per_buffer,
             samples_per_buffer_without_meta,
             bytes_per_buffer,
             us_per_buffer);

  const bladerf_format format = sample_size == sizeof(int8_t) ? bladerf_format::BLADERF_FORMAT_SC8_Q7_META
                                                              : bladerf_format::BLADERF_FORMAT_SC16_Q11_META;

  // Configure the device's Rx modules for use with the async interface.
  int status = bladerf_init_stream(
      &stream, device, stream_cb, &buffers, nof_buffers, format, samples_per_buffer, nof_transfers, this);
  if (status != 0) {
    on_error("bladerf_init_stream() failed - {}\n", nof_channels, bladerf_strerror(status));
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
    fmt::print(BLADERF_LOG_PREFIX "Enabling Rx module for channel {}...\n", channel + 1);

    int status = bladerf_enable_module(device, BLADERF_CHANNEL_RX(channel), true);
    if (status != 0) {
      on_error("bladerf_enable_module(BLADERF_CHANNEL_RX({}), true) failed - {}", channel, bladerf_strerror(status));
      return false;
    }
  }

  timestamp = init_timestamp = init_time;
  counters.last_reset_time   = now();

  std::thread thread([this]() {
    static const char* thread_name = "bladeRF-Rx";
    ::pthread_setname_np(::pthread_self(), thread_name);

    ::sched_param param{::sched_get_priority_max(SCHED_FIFO) - 2};
    if (::pthread_setschedparam(::pthread_self(), SCHED_FIFO, &param) != 0) {
      fmt::print(
          BLADERF_LOG_PREFIX "Could not set priority for the {} thread to {}\n", thread_name, param.sched_priority);
    }

    size_t    cpu = 0;
    cpu_set_t cpu_set{0};

    cpu_set_t available_cpuset = cpu_architecture_info::get().get_available_cpuset();
    for (size_t i = CPU_SETSIZE - 1; i >= 0; --i) {
      if (CPU_ISSET(i, &available_cpuset)) {
        cpu = i;
        break;
      }
    }

    CPU_SET(cpu, &cpu_set);
    if (::pthread_setaffinity_np(::pthread_self(), sizeof(cpu_set), &cpu_set) != 0) {
      fmt::print(BLADERF_LOG_PREFIX "Could not set affinity for the {} thread to {}\n", thread_name, cpu);
    }

    const bladerf_channel_layout layout =
        nof_channels == 1 ? bladerf_channel_layout::BLADERF_RX_X1 : bladerf_channel_layout::BLADERF_RX_X2;

    bladerf_stream(stream, layout);
  });

  cb_thread = std::move(thread);

  // Transition to streaming state.
  state = states::STREAMING;

  return true;
}

void* radio_bladerf_rx_stream::stream_cb(struct bladerf*          dev,
                                         struct bladerf_stream*   stream,
                                         struct bladerf_metadata* meta,
                                         void*                    samples,
                                         size_t                   nof_samples,
                                         void*                    user_data)
{
  radio_bladerf_rx_stream* rx_stream = static_cast<radio_bladerf_rx_stream*>(user_data);
  srsran_assert(rx_stream != nullptr, "null stream");

  if (rx_stream->state == states::STOP) {
    fmt::print(BLADERF_LOG_PREFIX "Shutting down Rx stream...\n");
    return BLADERF_STREAM_SHUTDOWN;
  }

  rx_stream->counters.on_callback(now());

  if (samples != nullptr) {
    rx_stream->counters.transfers_acked++;
    rx_stream->condition.notify_one();
  }

  return BLADERF_STREAM_NO_DATA;
}

baseband_gateway_receiver::metadata radio_bladerf_rx_stream::receive(baseband_gateway_buffer_writer& buffs)
{
  baseband_gateway_receiver::metadata ret{0};

  // Ignore if not streaming.
  if (state != states::STREAMING) {
    return {timestamp};
  }

  auto t0 = now();
  counters.on_receive_start(t0);

  // Make sure the number of channels is equal.
  srsran_assert(buffs.get_nof_channels() == nof_channels, "Number of channels does not match.");

  const size_t nsamples = buffs.get_nof_samples();

  // Make sure the number of samples is equal.
  srsran_assert(nsamples == samples_per_buffer_without_meta / nof_channels, "Number of samples does not match.");

  bool     rx_overflow     = false;
  bool     tx_underflow    = false;
  size_t   samples_dropped = 0;
  size_t   samples_missing = 0;
  uint64_t convert_time    = 0;

  size_t output_offset = 0;
  while (output_offset < nsamples) {
    wait_for_buffer();

    // Exit if stopped streaming.
    if (state != states::STREAMING) {
      return {timestamp};
    }

    int8_t* buffer = reinterpret_cast<int8_t*>(buffers[buffer_index]);

    // Handle each message in the buffer.
    while (output_offset < nsamples && buffer_byte_offset < bytes_per_buffer) {
      if (buffer_byte_offset % message_size == 0) {
        const uint64_t meta_timestamp = get_meta_timestamp(buffer + buffer_byte_offset);
        const uint32_t meta_flags     = get_meta_flags(buffer + buffer_byte_offset);

        rx_overflow |= meta_timestamp != timestamp;
        tx_underflow |= !!(meta_flags & BLADERF_META_FLAG_RX_HW_UNDERFLOW);

        buffer_byte_offset += meta_size;

        if (meta_timestamp > timestamp) {
          // Message starts in the future.
          output_offset += std::min(meta_timestamp - timestamp, nsamples - output_offset);
          samples_missing += meta_timestamp - timestamp;
          timestamp = meta_timestamp;
          if (output_offset == nsamples) {
            // No more samples available, return early.
            break;
          }
          // Handle this message again, at the new output offset.
          continue;
        }

        if (meta_timestamp < timestamp) {
          // Message starts in the past.
          const uint64_t next_timestamp = meta_timestamp + samples_per_message / nof_channels;
          if (next_timestamp <= timestamp) {
            // All samples are in the past, drop entire message.
            buffer_byte_offset += message_size - meta_size;
            samples_dropped += samples_per_message;
            // Handle next message.
            continue;
          }
          // Skip the samples that are in the past.
          buffer_byte_offset += samples_to_bytes(timestamp - meta_timestamp) * nof_channels;
          samples_dropped += (timestamp - meta_timestamp) * nof_channels;
        }
      }

      srsran_assert(output_offset < nsamples, "output buffer overflow");
      srsran_assert(buffer_byte_offset < bytes_per_buffer, "input buffer overflow");

      const size_t message_offset          = buffer_byte_offset % message_size;
      const size_t samples_in_msg          = bytes_to_samples(message_size - message_offset);
      const size_t channel_samples_to_read = std::min(samples_in_msg / nof_channels, nsamples - output_offset);

      if (ret.ts == 0) {
        ret.ts = timestamp;
      }

      t0 = now();

      // Convert samples.
      if (sample_size == sizeof(int8_t)) {
        if (nof_channels == 1) {
          const srsran::span<int8_t> x{buffer + buffer_byte_offset, channel_samples_to_read * 2};
          srsran::srsvec::convert(x, iq_scale, buffs[0].subspan(output_offset, channel_samples_to_read));
        } else {
          static uint16_t ch0[message_size / 2 / sizeof(uint16_t)];
          static uint16_t ch1[message_size / 2 / sizeof(uint16_t)];

          auto* x = reinterpret_cast<uint16_t*>(buffer + buffer_byte_offset);

          for (size_t i = 0; i < channel_samples_to_read; i++) {
            ch0[i] = x[2 * i + 0];
            ch1[i] = x[2 * i + 1];
          }

          const srsran::span<srsran::cf_t> z0 = buffs[0].subspan(output_offset, channel_samples_to_read);
          const srsran::span<srsran::cf_t> z1 = buffs[1].subspan(output_offset, channel_samples_to_read);
          srsran::srsvec::convert({reinterpret_cast<const int8_t*>(ch0), channel_samples_to_read * 2}, iq_scale, z0);
          srsran::srsvec::convert({reinterpret_cast<const int8_t*>(ch1), channel_samples_to_read * 2}, iq_scale, z1);
        }
      } else {
        if (nof_channels == 1) {
          srsran::span<int16_t> x{reinterpret_cast<int16_t*>(buffer + buffer_byte_offset), channel_samples_to_read * 2};
          srsran::srsvec::convert(x, iq_scale, buffs[0].subspan(output_offset, channel_samples_to_read));
        } else {
          static uint32_t ch0[message_size / 2 / sizeof(uint32_t)];
          static uint32_t ch1[message_size / 2 / sizeof(uint32_t)];

          auto* x = reinterpret_cast<uint32_t*>(buffer + buffer_byte_offset);

          for (size_t i = 0; i < channel_samples_to_read; i++) {
            ch0[i] = x[2 * i + 0];
            ch1[i] = x[2 * i + 1];
          }

          const srsran::span<srsran::cf_t> z0 = buffs[0].subspan(output_offset, channel_samples_to_read);
          const srsran::span<srsran::cf_t> z1 = buffs[1].subspan(output_offset, channel_samples_to_read);
          srsran::srsvec::convert({reinterpret_cast<const int16_t*>(ch0), channel_samples_to_read * 2}, iq_scale, z0);
          srsran::srsvec::convert({reinterpret_cast<const int16_t*>(ch1), channel_samples_to_read * 2}, iq_scale, z1);
        }
      }

      const auto t1 = now();
      convert_time += t1 - t0;
      t0 = t1;

      // Advance to next message.
      timestamp += channel_samples_to_read;
      output_offset += channel_samples_to_read;
      buffer_byte_offset += samples_to_bytes(channel_samples_to_read) * nof_channels;
    }

    srsran_assert(output_offset <= nsamples, "buffer overflow");
    srsran_assert(buffer_byte_offset <= bytes_per_buffer, "buffer overflow");

    counters.on_convert_complete(convert_time);

    // Resubmit buffer and advance to the next one.
    if (buffer_byte_offset == bytes_per_buffer) {
      // Resubmit the buffer.
      const int status = bladerf_submit_stream_buffer_nb(stream, buffers[buffer_index]);
      if (status != 0) {
        fmt::print(BLADERF_LOG_PREFIX "bladerf_submit_stream_buffer_nb() error - {}\n", bladerf_strerror(status));
      }

      counters.on_submit_complete(now() - t0);

      buffer_byte_offset = 0;
      buffer_index       = (buffer_index + 1) % nof_transfers;
    }
  }

  if (rx_overflow) {
    radio_notification_handler::event_description event = {};

    event.stream_id  = stream_id;
    event.channel_id = radio_notification_handler::UNKNOWN_ID;
    event.source     = radio_notification_handler::event_source::RECEIVE;
    event.type       = radio_notification_handler::event_type::OVERFLOW;
    event.timestamp  = ret.ts + output_offset;

    notifier.on_radio_rt_event(event);
  }

  if (tx_underflow) {
    tx_stream.on_underflow(ret.ts);
  }

  counters.samples_dropped += samples_dropped;
  counters.samples_missing += samples_missing;

  t0 = now();

  counters.on_receive_end(t0);

  if (counters.should_print(t0)) {
    if (print_counters) {
      fmt::print(BLADERF_LOG_PREFIX "Rx interval: [{}] "
                                    "{:4}..{:4}us, "
                                    "cb: {:4}..{:4}us, "
                                    "rx: {:4}..{:4}us, "
                                    "conv: {:3}..{:3}us, "
                                    "submit: {:3}..{:3}us, "
                                    "q: {}..{}, "
                                    "drop: {} ({:.1f}us) "
                                    "miss: {} ({:.1f}us)\n",
                 ret.ts - counters.last_timestamp,
                 counters.receive_interval.min,
                 counters.receive_interval.max,
                 counters.callback_interval.min,
                 counters.callback_interval.max,
                 counters.receive_time.min,
                 counters.receive_time.max,
                 counters.conversion_time.min,
                 counters.conversion_time.max,
                 counters.submit_time.min,
                 counters.submit_time.max,
                 counters.queued_transfers.min,
                 counters.queued_transfers.max,
                 counters.samples_dropped,
                 1000000 * counters.samples_dropped / srate_Hz / nof_channels,
                 counters.samples_missing,
                 1000000 * counters.samples_missing / srate_Hz / nof_channels);
    }
    counters.last_timestamp = ret.ts;
    counters.reset(t0);
  }

  return ret;
}

void radio_bladerf_rx_stream::stop()
{
  state = states::STOP;

  // Unblock thread.
  condition.notify_one();

  // Wait for uplink to stop.
  std::this_thread::sleep_for(std::chrono::milliseconds(500));

  bladerf_submit_stream_buffer_nb(stream, BLADERF_STREAM_SHUTDOWN);

  if (cb_thread.joinable()) {
    cb_thread.join();
  }

  bladerf_deinit_stream(stream);

  for (size_t channel = 0; channel < nof_channels; channel++) {
    fmt::print(BLADERF_LOG_PREFIX "Disabling Rx module for channel {}...\n", channel + 1);

    int status = bladerf_enable_module(device, BLADERF_CHANNEL_RX(channel), false);
    if (status != 0) {
      on_error("bladerf_enable_module(BLADERF_CHANNEL_RX{}, false) failed - {}", channel, bladerf_strerror(status));
    }
  }
}

unsigned radio_bladerf_rx_stream::get_buffer_size() const
{
  return samples_per_buffer_without_meta / nof_channels;
}
