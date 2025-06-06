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

#include "radio_bladerf_tx_stream.h"
#include "srsran/srsvec/conversion.h"
#include "srsran/support/executors/unique_thread.h"

#if defined(__AVX2__)
#include <immintrin.h>
#endif

using namespace srsran;

radio_bladerf_tx_stream::radio_bladerf_tx_stream(bladerf*                    device_,
                                                 const stream_description&   description,
                                                 radio_notification_handler& notifier_) :
  stream_id(description.id),
  srate_Hz(description.srate_Hz),
  nof_channels(description.nof_channels),
  notifier(notifier_),
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

  // Around 5 transfers per 1ms.
  samples_per_buffer = nof_channels * srate_Hz / 1e3 / 5.f;
  if (sample_size == 2) {
    samples_per_buffer = (samples_per_buffer + 4095) & ~4095;
  } else if (sample_size == 3) {
    samples_per_buffer = (samples_per_buffer + 8191) & ~8191;
  } else {
    samples_per_buffer = (samples_per_buffer + 2047) & ~2047;
  }

  const char* env_buffer_size = getenv("TX_BUFFER_SIZE");
  if (env_buffer_size != nullptr) {
    samples_per_buffer = atoi(env_buffer_size);
  }

  nof_transfers = 16;

  const char* env_nof_transfers = getenv("TX_TRANSFERS");
  if (env_nof_transfers != nullptr) {
    nof_transfers = atoi(env_nof_transfers);
  }

  nof_buffers = nof_transfers * 2;

  const char* env_print_counters = getenv("STATS");
  if (env_print_counters != nullptr) {
    print_counters = atoi(env_print_counters);
  }

  fmt::print(BLADERF_LOG_PREFIX "Creating Tx stream with {} channels and {}-bit samples at {} MHz...\n",
             nof_channels,
             4 * sample_size,
             srate_Hz / 1e6);

  samples_per_buffer_without_meta =
      samples_per_buffer - (sample_size == 3 ? 32 : 16 * samples_per_buffer * sample_size / message_size);
  bytes_per_buffer = samples_to_bytes(samples_per_buffer);
  us_per_buffer    = 1000000 * samples_per_buffer_without_meta / nof_channels / srate_Hz;

  const size_t flush_samples = nof_transfers * samples_per_buffer_without_meta + bytes_to_samples(device_buffer_bytes);
  flush_duration             = 1000000 * flush_samples / nof_channels / srate_Hz;

  fmt::print(BLADERF_LOG_PREFIX
             "...{} transfers, {} buffers, {}/{} samples/buffer, {} bytes/buffer, {}us/buffer, {}us/flush...\n",
             nof_transfers,
             nof_buffers,
             samples_per_buffer,
             samples_per_buffer_without_meta,
             bytes_per_buffer,
             us_per_buffer,
             flush_duration);

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

  // Configure the device's Tx modules for use with the async interface.
  int status = bladerf_init_stream(
      &stream, device, stream_cb, &buffers, nof_buffers, format, samples_per_buffer, nof_transfers, this);
  if (status != 0) {
    on_error("bladerf_init_stream() failed - {}", nof_channels, bladerf_strerror(status));
    return;
  }

  const char* env_dump_tx = getenv("DUMP_TX");
  if (env_dump_tx != nullptr && atoi(env_dump_tx) != 0) {
    dump_fd = fopen("bladerf-tx.bin", "wb");
    fmt::print(BLADERF_LOG_PREFIX "Dumping Tx samples to bladerf-tx.bin...\n");
  }

  // Disable libusb event handling on this stream and let the Rx thread do all the handling.
  status = bladerf_enable_feature(device, bladerf_feature::BLADERF_FEATURE_RX_ALL_EVENTS, true);
  if (status != 0) {
    on_error("bladerf_enable_feature(BLADERF_FEATURE_RX_ALL_EVENTS, true) failed - {}", bladerf_strerror(status));
  }

  for (size_t channel = 0; channel < nof_channels; channel++) {
    fmt::print(BLADERF_LOG_PREFIX "Enabling Tx module for channel {}...\n", channel);

    status = bladerf_enable_module(device, BLADERF_CHANNEL_TX(channel), true);
    if (status != 0) {
      on_error("bladerf_enable_module(BLADERF_CHANNEL_TX({}), true) failed - {}", channel, bladerf_strerror(status));
      bladerf_deinit_stream(stream);
      return;
    }
  }

  // Start the stream early to enable timestamping and get a proper init time.
  std::thread thread([this]() {
    static const char* thread_name = "bladeRF-Tx";
    ::pthread_setname_np(::pthread_self(), thread_name);

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
        nof_channels == 1 ? bladerf_channel_layout::BLADERF_TX_X1 : bladerf_channel_layout::BLADERF_TX_X2;

    bladerf_stream(stream, layout);
  });

  cb_thread = std::move(thread);

  // Wait for stream start to configure the device.
  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  state = states::SUCCESSFUL_INIT;
}

bool radio_bladerf_tx_stream::start()
{
  if (state != states::SUCCESSFUL_INIT) {
    return true;
  }

  counters.last_reset_time = now();

  // Transition to streaming state.
  state = states::STREAMING;

  return true;
}

void* radio_bladerf_tx_stream::stream_cb(struct bladerf*          dev,
                                         struct bladerf_stream*   stream,
                                         struct bladerf_metadata* meta,
                                         void*                    samples,
                                         size_t                   nof_samples,
                                         void*                    user_data)
{
  radio_bladerf_tx_stream* tx_stream = static_cast<radio_bladerf_tx_stream*>(user_data);
  srsran_assert(tx_stream != nullptr, "null stream");

  if (tx_stream->state == states::STOP) {
    fmt::print(BLADERF_LOG_PREFIX "Shutting down Tx stream...\n");
    return BLADERF_STREAM_SHUTDOWN;
  }

  tx_stream->counters.on_callback(now());

  if (samples != nullptr) {
    tx_stream->counters.transfers_acked++;
    if (tx_stream->counters.transfers_acked == tx_stream->counters.transfers_submitted) {
      if (tx_stream->counters.transfers_drain_start != 0) {
        tx_stream->counters.transfers_drain_time.update(now() - tx_stream->counters.transfers_drain_start);
        tx_stream->counters.transfers_drain_start = 0;
      }
    }
  }

  return BLADERF_STREAM_NO_DATA;
}

void radio_bladerf_tx_stream::transmit(const baseband_gateway_buffer_reader&        buffs,
                                       const baseband_gateway_transmitter_metadata& tx_md)
{
  // Ignore if not streaming.
  if (state != states::STREAMING) {
    return;
  }

  const auto t0 = now();

  // Make sure the number of channels is equal.
  srsran_assert(buffs.get_nof_channels() == nof_channels, "Number of channels does not match.");

  if (eob != 0) {
    if (eob > t0) {
      counters.on_transmit_skipped(t0);
      return;
    }
    // This is a bit racy...
    eob = 0;
  }

  if (tx_md.ts < timestamp) {
    fmt::print(BLADERF_LOG_PREFIX "Tx late by {} samples\n", timestamp - tx_md.ts);
    counters.on_transmit_skipped(t0);
    return;
  }

  counters.on_transmit_start(t0);

  const size_t nsamples = buffs.get_nof_samples();

  const size_t nof_required_buffers = nsamples / samples_per_buffer_without_meta + 1;
  srsran_assert(nof_required_buffers <= nof_buffers - nof_transfers, "buffer overflow");

  if (timestamp != tx_md.ts) {
    if (timestamp != 0) {
      counters.samples_dropped += (tx_md.ts - timestamp) * nof_channels;
    }
    timestamp = tx_md.ts;
  }

  size_t start_buffer_index = buffer_index;
  size_t buffers_filled     = 0;
  size_t input_offset       = 0;

  // Fill the buffers.
  while (input_offset < nsamples) {
    size_t  current_buffer_index = (start_buffer_index + buffers_filled) % nof_buffers;
    int8_t* buffer               = reinterpret_cast<int8_t*>(buffers[current_buffer_index]);

    if (buffer_byte_offset % message_size == 0) {
      set_meta_timestamp(buffer + buffer_byte_offset, timestamp);
      buffer_byte_offset += meta_size;
    }

    const size_t message_offset           = buffer_byte_offset % message_size;
    const size_t samples_in_msg           = bytes_to_samples(message_size - message_offset) / nof_channels;
    const size_t channel_samples_to_write = std::min(samples_in_msg, nsamples - input_offset);

    // Convert samples.
    if (sample_size == 2) {
      const srsran::span<int8_t> z = {buffer + buffer_byte_offset, channel_samples_to_write * 2 * nof_channels};

      if (nof_channels == 1) {
        const auto x = buffs[0].subspan(input_offset, channel_samples_to_write);
        srsran::srsvec::convert(x, iq_scale * 1.5f, z);
      } else {
        const auto x = buffs[0].subspan(input_offset, channel_samples_to_write);
        const auto y = buffs[1].subspan(input_offset, channel_samples_to_write);
        srsran::srsvec::convert(x, y, iq_scale * 1.5f, z);
      }

      if (dump_fd != nullptr) {
        fwrite(z.data(), sizeof(int8_t), z.size(), dump_fd);
      }
    } else {
      srsran::span<int16_t> z;

      if (sample_size == 3) {
        compaction_buffer.resize(channel_samples_to_write * 2 * nof_channels * sizeof(int16_t));
        z = {reinterpret_cast<int16_t*>(compaction_buffer.data()), channel_samples_to_write * 2 * nof_channels};
      } else {
        z = {reinterpret_cast<int16_t*>(buffer + buffer_byte_offset), channel_samples_to_write * 2 * nof_channels};
      }

      if (nof_channels == 1) {
        const auto x = buffs[0].subspan(input_offset, channel_samples_to_write);
        srsran::srsvec::convert(x, iq_scale, z);
      } else {
        const auto x = buffs[0].subspan(input_offset, channel_samples_to_write);
        const auto y = buffs[1].subspan(input_offset, channel_samples_to_write);
        srsran::srsvec::convert(x, y, iq_scale, z);
      }

      if (dump_fd != nullptr) {
        fwrite(z.data(), sizeof(int16_t), z.size(), dump_fd);
      }

      if (sample_size == 3) {
#if defined(__AVX2__)
        static const int8_t  grouping_re[32] = {0,  1,  2,  4,  5,  6,  8,  9,  10, 12, 13, 14, -1, -1, -1, -1,
                                                16, 17, 18, 20, 21, 22, 24, 25, 26, 28, 29, 30, -1, -1, -1, -1};
        static const __m256i grouping_re256  = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(grouping_re));
        static const uint8_t mask_re[32]     = {0xff, 0x0f, 0x00, 0x00, 0xff, 0x0f, 0x00, 0x00, 0xff, 0x0f, 0x00,
                                                0x00, 0xff, 0x0f, 0x00, 0x00, 0xff, 0x0f, 0x00, 0x00, 0xff, 0x0f,
                                                0x00, 0x00, 0xff, 0x0f, 0x00, 0x00, 0xff, 0x0f, 0x00, 0x00};
        static const __m256i mask_re256      = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(mask_re));
        static const int8_t  grouping_im[32] = {1,  2,  3,  5,  6,  7,  9,  10, 11, 13, 14, 15, -1, -1, -1, -1,
                                                17, 18, 19, 21, 22, 23, 25, 26, 27, 29, 30, 31, -1, -1, -1, -1};
        static const __m256i grouping_im256  = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(grouping_im));
        static const uint8_t mask_im[32]     = {0x00, 0x00, 0xff, 0xff, 0x00, 0x00, 0xff, 0xff, 0x00, 0x00, 0xff,
                                                0xff, 0x00, 0x00, 0xff, 0xff, 0x00, 0x00, 0xff, 0xff, 0x00, 0x00,
                                                0xff, 0xff, 0x00, 0x00, 0xff, 0xff, 0x00, 0x00, 0xff, 0xff};
        static const __m256i mask_im256      = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(mask_im));
#endif
        size_t n = channel_samples_to_write * nof_channels;

        const auto* src = compaction_buffer.data();
        auto*       dst = reinterpret_cast<int8_t*>(buffer + buffer_byte_offset);

#if defined(__AVX2__)
        while (n > 16) {
          auto v  = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src));
          auto re = _mm256_and_si256(v, mask_re256);
          re      = _mm256_shuffle_epi8(re, grouping_re256);
          auto im = _mm256_and_si256(v, mask_im256);
          im      = _mm256_slli_epi16(im, 4);
          im      = _mm256_shuffle_epi8(im, grouping_im256);
          v       = _mm256_or_si256(re, im);
          _mm_storeu_si128(reinterpret_cast<__m128i*>(dst), _mm256_extracti128_si256(v, 0));
          _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + 12), _mm256_extracti128_si256(v, 1));
          src += 8 * 4 / sizeof(int16_t);
          dst += 8 * 3;
          n -= 8;
          break;
        }
#endif

        while (n > 0) {
          int16_t re = *src++;
          int16_t im = *src++;
          dst[0]     = re & 0xFF;
          dst[1]     = ((re >> 8) & 0xF) | (im & 0x0F) << 4;
          dst[2]     = im >> 4;
          dst += 3;
          n--;
        }
      }
    }

    timestamp += channel_samples_to_write;
    input_offset += channel_samples_to_write;
    buffer_byte_offset += samples_to_bytes(channel_samples_to_write) * nof_channels;

    srsran_assert(buffer_byte_offset <= bytes_per_buffer, "buffer overflow");

    if (buffer_byte_offset == bytes_per_buffer) {
      // Move to next buffer.
      buffer_byte_offset = 0;
      buffers_filled += 1;
    }
  }

  const auto t1 = now();
  counters.on_convert_complete(t1 - t0);

  if (counters.transfers_submitted == counters.transfers_acked) {
    counters.transfers_drain_start = t1;
  }

  if (first_timestamp == 0) {
    first_timestamp = timestamp;
  }

  // Submit filled buffers.
  size_t transfers_submitted = 0;
  for (size_t i = 0; i < buffers_filled; i++) {
    const size_t index  = (start_buffer_index + i) % nof_buffers;
    const int    status = bladerf_submit_stream_buffer_nb(stream, buffers[index]);
    if (status == 0) {
      // Advance buffer index.
      buffer_index = (buffer_index + 1) % nof_buffers;
      transfers_submitted++;
      continue;
    }

    if (status == BLADERF_ERR_WOULD_BLOCK) {
      if (timestamp - first_timestamp > srate_Hz) {
        // We're sending faster than the device can receive.
        radio_notification_handler::event_description event_description = {};

        event_description.stream_id  = stream_id;
        event_description.channel_id = radio_notification_handler::UNKNOWN_ID;
        event_description.source     = radio_notification_handler::event_source::TRANSMIT;
        event_description.type       = radio_notification_handler::event_type::OVERFLOW;
        event_description.timestamp  = timestamp;

        notifier.on_radio_rt_event(event_description);
      }
    } else {
      fmt::print(BLADERF_LOG_PREFIX "bladerf_submit_stream_buffer_nb() error - {}\n", bladerf_strerror(status));
    }

    flush();

    // Drop remaining buffers.
    counters.samples_dropped += (buffers_filled - i) * samples_per_buffer_without_meta;
    break;
  }

  const auto t2 = now();

  counters.on_submit_complete(transfers_submitted, t2 - t1);
  counters.on_transmit_end(t2);

  if (counters.should_print(t2)) {
    if (print_counters) {
      fmt::print(BLADERF_LOG_PREFIX "Tx interval: [{}] "
                                    "{:4}..{:4}us, "
                                    "cb: {:4}..{:4}us, "
                                    "tx: {:4}..{:4}us, "
                                    "conv: {:3}..{:3}us, "
                                    "submit: {:3}..{:3}us, "
                                    "q: {}..{}, "
                                    "drop: {} ({:.1f}us), "
                                    "drain: {}..{}us\n",
                 timestamp - counters.last_timestamp,
                 counters.transmit_interval.min,
                 counters.transmit_interval.max,
                 counters.callback_interval.min,
                 counters.callback_interval.max,
                 counters.transmit_time.min,
                 counters.transmit_time.max,
                 counters.conversion_time.min,
                 counters.conversion_time.max,
                 counters.submit_time.min,
                 counters.submit_time.max,
                 counters.queued_transfers.min,
                 counters.queued_transfers.max,
                 counters.samples_dropped,
                 1000000 * counters.samples_dropped / srate_Hz / nof_channels,
                 counters.transfers_drain_time.min + 1,
                 counters.transfers_drain_time.max);
    }
    counters.last_timestamp = timestamp;
    counters.reset(t2);
  }
}

void radio_bladerf_tx_stream::stop()
{
  state = states::STOP;

  // Wait for downlink to stop.
  std::this_thread::sleep_for(std::chrono::milliseconds(500));

  bladerf_submit_stream_buffer_nb(stream, BLADERF_STREAM_SHUTDOWN);

  if (cb_thread.joinable()) {
    cb_thread.join();
  }

  bladerf_deinit_stream(stream);

  for (size_t channel = 0; channel < nof_channels; channel++) {
    fmt::print(BLADERF_LOG_PREFIX "Disabling Tx module for channel {}...\n", channel);

    int status = bladerf_enable_module(device, BLADERF_CHANNEL_TX(channel), false);
    if (status != 0) {
      on_error("bladerf_enable_module(BLADERF_CHANNEL_TX({}), false) failed - {}", channel, bladerf_strerror(status));
    }
  }

  if (dump_fd != nullptr) {
    fclose(dump_fd);
    dump_fd = nullptr;
  }
}

void radio_bladerf_tx_stream::on_underflow(uint64_t uf_timestamp)
{
  // Data was not sent fast enough.
  radio_notification_handler::event_description event_description = {};

  event_description.stream_id  = stream_id;
  event_description.channel_id = radio_notification_handler::UNKNOWN_ID;
  event_description.source     = radio_notification_handler::event_source::TRANSMIT;
  event_description.type       = radio_notification_handler::event_type::UNDERFLOW;
  event_description.timestamp  = uf_timestamp;

  notifier.on_radio_rt_event(event_description);

  flush();
}

void radio_bladerf_tx_stream::flush()
{
  if (eob == 0) {
    eob = now() + flush_duration;
  }
}

unsigned radio_bladerf_tx_stream::get_buffer_size() const
{
  return samples_per_buffer_without_meta / nof_channels;
}
