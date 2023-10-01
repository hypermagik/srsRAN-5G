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
#include "srsran/gateways/baseband/baseband_gateway_transmitter.h"
#include "srsran/gateways/baseband/buffer/baseband_gateway_buffer_reader.h"
#include "srsran/radio/radio_configuration.h"
#include "srsran/radio/radio_notification_handler.h"
#include <libbladeRF.h>
#include <thread>

#define BLADERF_LOG_PREFIX "\033[1m\033[32m[bladeRF]\033[0m "

namespace srsran {

static inline uint64_t now()
{
  const auto now = std::chrono::high_resolution_clock::now().time_since_epoch();
  return std::chrono::duration_cast<std::chrono::microseconds>(now).count();
}

struct MinMaxCounters {
  uint64_t min = UINT64_MAX;
  uint64_t max = 0;

  void reset()
  {
    min = UINT64_MAX;
    max = 0;
  }

  void update(uint64_t val)
  {
    if (val > max) {
      max = val;
    }
    if (val < min) {
      min = val;
    }
  }
};

/// Implements a gateway transmitter based on bladeRF transmit stream.
class radio_bladerf_tx_stream : public baseband_gateway_transmitter, public bladerf_error_handler
{
private:
  /// Defines the Rx stream internal states.
  enum class states { UNINITIALIZED, SUCCESSFUL_INIT, STREAMING, STOP };
  /// Indicates the current stream state.
  std::atomic<states> state = {states::UNINITIALIZED};

  /// Indicates the stream identification for notifications.
  unsigned stream_id;
  /// Sampling rate in Hz.
  double srate_Hz;
  /// Indicates the number of channels.
  unsigned nof_channels;
  /// Sample size.
  size_t sample_size;
  /// IQ scale.
  float iq_scale;
  /// Radio notification interface.
  radio_notification_handler& notifier;

  /// Owns the bladeRF Tx stream.
  bladerf* const device;
  /// Device stream.
  struct bladerf_stream* stream;
  /// Stream callback thread.
  std::thread cb_thread;
  /// Stream callback.
  static void* stream_cb(struct bladerf*          dev,
                         struct bladerf_stream*   stream,
                         struct bladerf_metadata* meta,
                         void*                    samples,
                         size_t                   nof_samples,
                         void*                    user_data);

  /// Buffer data.
  void** buffers;
  size_t nof_buffers;
  size_t nof_transfers;
  size_t samples_per_buffer;
  size_t samples_per_buffer_without_meta;
  size_t bytes_per_buffer;
  size_t us_per_buffer;
  size_t buffer_index       = 0;
  size_t buffer_byte_offset = 0;
  /// USB3 message size.
  static constexpr size_t message_size = 2048;
  /// Metadata size.
  static constexpr size_t meta_size = 2 * sizeof(uint64_t);
  /// Size of receive buffer on device.
  static constexpr size_t device_buffer_bytes = 64 * 1024;
  /// Current Tx timestamp.
  uint64_t timestamp = 0;
  /// End of buffers.
  uint64_t eob = 0;
  /// Time required to flush buffers.
  uint64_t flush_duration;

  /// @brief Convert samples to bytes, according to otw format.
  /// @param[in] samples Number of samples to convert.
  /// @return Number of bytes.
  size_t samples_to_bytes(size_t samples) const { return samples * 2 * sample_size; }

  /// @brief Convert bytes to samples, according to otw format.
  /// @param[in] bytes Number of bytes to convert.
  /// @return Number of samples.
  size_t bytes_to_samples(size_t bytes) const { return bytes / 2 / sample_size; }

  /// @brief Set timestamp in message metadata.
  /// @param message The message.
  /// @param timestamp The timestamp.
  static void set_meta_timestamp(int8_t* message, uint64_t timestamp)
  {
    *reinterpret_cast<uint64_t*>(message + sizeof(uint32_t)) = htole64(timestamp);
  }

  /// @brief Block transmission until all buffers are empty.
  void flush();

  /// Counters.
  bool print_counters = false;

  struct {
    uint64_t last_reset_time       = 0;
    uint64_t last_timestamp        = 0;
    uint64_t last_callback_start   = 0;
    uint64_t last_transmit_start   = 0;
    size_t   samples_dropped       = 0;
    size_t   transfers_drain_start = 0;
    size_t   transfers_submitted   = 0;
    size_t   transfers_acked       = 0;

    MinMaxCounters callback_interval;
    MinMaxCounters transmit_interval;
    MinMaxCounters transmit_time;
    MinMaxCounters conversion_time;
    MinMaxCounters submit_time;
    MinMaxCounters queued_transfers;
    MinMaxCounters transfers_drain_time;

    void on_callback(uint64_t now)
    {
      if (last_callback_start != 0) {
        callback_interval.update(now - last_callback_start);
      }
      last_callback_start = now;
    }

    void on_transmit_start(uint64_t now)
    {
      if (last_transmit_start != 0) {
        transmit_interval.update(now - last_transmit_start);
      }
      last_transmit_start = now;
    }

    void on_transmit_end(uint64_t now) { transmit_time.update(now - last_transmit_start); }

    void on_transmit_skipped(uint64_t now) { last_transmit_start = now; }

    void on_convert_complete(uint64_t delta) { conversion_time.update(delta); }

    void on_submit_complete(uint64_t n, uint64_t delta)
    {
      transfers_submitted += n;
      queued_transfers.update(transfers_submitted - transfers_acked);
      submit_time.update(delta);
    }

    bool should_print(uint64_t now) { return now - last_reset_time >= 1000000; }

    void reset(uint64_t now)
    {
      last_reset_time = now;
      samples_dropped = 0;
      callback_interval.reset();
      transmit_interval.reset();
      transmit_time.reset();
      conversion_time.reset();
      submit_time.reset();
      queued_transfers.reset();
      transfers_drain_time.reset();
    }
  } counters;

public:
  /// Describes the necessary parameters to create an bladeRF transmit stream.
  struct stream_description {
    /// Identifies the stream.
    unsigned id;
    /// Over-the-wire format.
    radio_configuration::over_the_wire_format otw_format;
    /// Sampling rate in Hz.
    double srate_Hz;
    /// Indicates the number of channels.
    unsigned nof_channels;
  };

  /// \brief Constructs a bladeRF transmit stream.
  /// \param[in] device Provides the bladeRF device handle.
  /// \param[in] description Provides the stream configuration parameters.
  /// \param[in] notifier_ Provides the radio event notification handler.
  radio_bladerf_tx_stream(bladerf*                    device,
                          const stream_description&   description,
                          radio_notification_handler& notifier_);

  /// Starts the stream transmission.
  bool start();

  /// Gets the optimal transmitter buffer size.
  unsigned get_buffer_size() const;

  // See interface for documentation.
  void transmit(const baseband_gateway_buffer_reader& data, const baseband_gateway_transmitter_metadata& metadata) override;

  /// Stop the transmission.
  void stop();

  /// Notification from Rx stream.
  void on_underflow(uint64_t timestamp);
};
} // namespace srsran
