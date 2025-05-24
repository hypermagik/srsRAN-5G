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
#include "srsran/support/executors/unique_thread.h"
#include "fmt/base.h"
#include <emmintrin.h>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

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

  if (description.otw_format == radio_configuration::over_the_wire_format::SC8) {
    sample_size = 2;
    iq_scale    = 128.f;
    meta_size   = 2 * sizeof(uint64_t);
  } else if (description.otw_format == radio_configuration::over_the_wire_format::SC12) {
    sample_size = 3;
    iq_scale    = 2048.f;
    meta_size   = 4 * sizeof(uint64_t);
  } else {
    sample_size = 4;
    iq_scale    = 2048.f;
    meta_size   = 2 * sizeof(uint64_t);
  }

  // Around 10 transfers per 1ms, for more resolution.
  samples_per_buffer = nof_channels * srate_Hz / 1e3 / 10.f;
  if (sample_size == 2) {
    samples_per_buffer = (samples_per_buffer + 4095) & ~4095;
  } else if (sample_size == 3) {
    samples_per_buffer = (samples_per_buffer + 8191) & ~8191;
  } else {
    samples_per_buffer = (samples_per_buffer + 2047) & ~2047;
  }

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
             4 * sample_size,
             srate_Hz / 1e6);

  samples_per_message = bytes_to_samples(message_size - meta_size);
  samples_per_buffer_without_meta =
      samples_per_buffer - (sample_size == 3 ? 32 : 16 * samples_per_buffer * sample_size / message_size);
  bytes_per_buffer = samples_to_bytes(samples_per_buffer);
  us_per_buffer    = 1000000 * samples_per_buffer_without_meta / nof_channels / srate_Hz;

  fmt::print(BLADERF_LOG_PREFIX "...{} transfers, {} buffers, {}/{} samples/buffer, {} bytes/buffer, {}us/buffer...\n",
             nof_transfers,
             nof_buffers,
             samples_per_buffer,
             samples_per_buffer_without_meta,
             bytes_per_buffer,
             us_per_buffer);

  bladerf_format format = bladerf_format::BLADERF_FORMAT_SC16_Q11_META;
  if (sample_size == 2) {
    format = bladerf_format::BLADERF_FORMAT_SC8_Q7_META;
  } else if (sample_size == 3) {
#if defined(LIBBLADERF_API_VERSION) && (LIBBLADERF_API_VERSION >= 0x02060000)
    format = bladerf_format::BLADERF_FORMAT_SC16_Q11_PACKED_META;
#else
    #pragma message "SC12 OTW format requires libbladeRF version >= 2.6.0 with support for BLADERF_FORMAT_SC16_Q11_PACKED_META"
#endif
  }

  // Configure the device's Rx modules for use with the async interface.
  int status = bladerf_init_stream(
      &stream, device, stream_cb, &buffers, nof_buffers, format, samples_per_buffer, nof_transfers, this);
  if (status != 0) {
    on_error("bladerf_init_stream() failed - {}\n", nof_channels, bladerf_strerror(status));
    return;
  }

  const char* env_dump_rx = getenv("DUMP_RX");
  if (env_dump_rx != nullptr && atoi(env_dump_rx) != 0) {
    dump_fd = fopen("bladerf-rx.bin", "wb");
    fmt::print(BLADERF_LOG_PREFIX "Dumping Rx samples to bladerf-rx.bin...\n");
  }

  state = states::SUCCESSFUL_INIT;
}

bool radio_bladerf_rx_stream::start(baseband_gateway_timestamp init_time)
{
  if (state != states::SUCCESSFUL_INIT) {
    return true;
  }

  for (size_t channel = 0; channel < nof_channels; channel++) {
    fmt::print(BLADERF_LOG_PREFIX "Enabling Rx module for channel {}...\n", channel);

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
      if (sample_size == 2) {
        const srsran::span<int8_t> x{buffer + buffer_byte_offset, channel_samples_to_read * 2 * nof_channels};

        if (dump_fd != nullptr) {
          fwrite(x.data(), sizeof(int8_t), x.size(), dump_fd);
        }

        if (nof_channels == 1) {
          const auto& z = buffs[0].subspan(output_offset, channel_samples_to_read);
          srsran::srsvec::convert(x, iq_scale, z);
        } else {
          const auto& z0 = buffs[0].subspan(output_offset, channel_samples_to_read);
          const auto& z1 = buffs[1].subspan(output_offset, channel_samples_to_read);
          srsran::srsvec::convert(x, iq_scale, z0, z1);
        }
      } else {
        srsran::span<int16_t> x;

        if (sample_size == 3) {
#if defined(__AVX2__)
          static const uint8_t grouping[32] = {0,  1,  1,  2,  3,  4,  4,  5,  6,  7,  7,  8,  9,  10, 10, 11,
                                               16, 17, 17, 18, 19, 20, 20, 21, 22, 23, 23, 24, 25, 26, 26, 27};
          static const __m256i grouping256  = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(grouping));
#endif
          size_t n = channel_samples_to_read * nof_channels;
          expansion_buffer.resize(n * 2 * sizeof(int16_t));

          const auto* src = buffer + buffer_byte_offset;
          auto*       dst = reinterpret_cast<int16_t*>(expansion_buffer.data());

#if defined(__AVX2__)
          while (n >= 10) {
            const auto a = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src));
            const auto b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + 12));
            auto       v = _mm256_inserti128_si256(_mm256_castsi128_si256(a), b, 1);
            v            = _mm256_shuffle_epi8(v, grouping256);
            auto re      = _mm256_slli_epi16(v, 4);
            re           = _mm256_srai_epi16(re, 4);
            auto im      = _mm256_srai_epi16(v, 4);
            v            = _mm256_blend_epi16(re, im, 0b10101010);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst), v);
            src += 8 * 3;
            dst += 8 * 4 / sizeof(int16_t);
            n -= 8;
          }
#endif

          while (n > 0) {
            *dst++ = static_cast<int16_t>((*reinterpret_cast<const uint16_t*>(src) & 0x0fff) << 4) >> 4;
            *dst++ = *reinterpret_cast<const int16_t*>(src + 1) >> 4;
            src += 3;
            n--;
          }

          x = {reinterpret_cast<int16_t*>(expansion_buffer.data()), channel_samples_to_read * 2 * nof_channels};
        } else {
          x = {reinterpret_cast<int16_t*>(buffer + buffer_byte_offset), channel_samples_to_read * 2 * nof_channels};
        }

        if (dump_fd != nullptr) {
          fwrite(x.data(), sizeof(int16_t), x.size(), dump_fd);
        }

        if (nof_channels == 1) {
          const auto& z = buffs[0].subspan(output_offset, channel_samples_to_read);
          srsran::srsvec::convert(x, iq_scale, z);
        } else {
          const auto& z0 = buffs[0].subspan(output_offset, channel_samples_to_read);
          const auto& z1 = buffs[1].subspan(output_offset, channel_samples_to_read);
          srsran::srsvec::convert(x, iq_scale, z0, z1);
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

  if (rx_overflow && timestamp - init_timestamp > srate_Hz) {
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
    fmt::print(BLADERF_LOG_PREFIX "Disabling Rx module for channel {}...\n", channel);

    int status = bladerf_enable_module(device, BLADERF_CHANNEL_RX(channel), false);
    if (status != 0) {
      on_error("bladerf_enable_module(BLADERF_CHANNEL_RX{}, false) failed - {}", channel, bladerf_strerror(status));
    }
  }

  if (dump_fd != nullptr) {
    fclose(dump_fd);
    dump_fd = nullptr;
  }
}

unsigned radio_bladerf_rx_stream::get_buffer_size() const
{
  return samples_per_buffer_without_meta / nof_channels;
}
