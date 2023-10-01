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
#include "radio_bladerf_tx_stream.h"
#include "srsran/gateways/baseband/baseband_gateway_receiver.h"
#include "srsran/gateways/baseband/buffer/baseband_gateway_buffer_writer.h"
#include "srsran/radio/radio_configuration.h"
#include "srsran/radio/radio_notification_handler.h"
#include <condition_variable>
#include <libbladeRF.h>
#include <mutex>
#include <thread>

namespace srsran {

/// Implements a gateway receiver based on bladeRF receive stream.
class radio_bladerf_rx_stream : public baseband_gateway_receiver, public bladerf_error_handler
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
  /// Tx stream for underflow notifications.
  radio_bladerf_tx_stream& tx_stream;

  /// Owns the bladeRF Rx stream.
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
  size_t nof_transfers;
  size_t samples_per_buffer;
  size_t samples_per_buffer_without_meta;
  size_t bytes_per_buffer;
  size_t us_per_buffer;
  size_t buffer_index       = 0;
  size_t buffer_byte_offset = 0;
  /// Synchronization.
  std::mutex              mutex;
  std::condition_variable condition;
  /// USB3 message size.
  static constexpr size_t message_size = 2048;
  /// Number of samples per message, without metadata.
  size_t samples_per_message;
  /// Metadata size.
  static constexpr size_t meta_size = 2 * sizeof(uint64_t);
  /// Current Rx timestamp.
  uint64_t timestamp = 0;
  /// Starting timestamp.
  uint64_t init_timestamp = 0;

  /// Wait for a buffer to become available.
  void wait_for_buffer()
  {
    while (counters.transfers_acked <= counters.transfers_submitted && state == states::STREAMING) {
      std::unique_lock<std::mutex> lock(mutex);
      condition.wait(lock, [this] {
        return counters.transfers_acked > counters.transfers_submitted || state != states::STREAMING;
      });
    }
  }

  /// @brief Convert samples to bytes, according to otw format.
  /// @param[in] samples Number of samples to convert.
  /// @return Number of bytes.
  size_t samples_to_bytes(size_t samples) const { return samples * 2 * sample_size; }

  /// @brief Convert bytes to samples, according to otw format.
  /// @param[in] bytes Number of bytes to convert.
  /// @return Number of samples.
  size_t bytes_to_samples(size_t bytes) const { return bytes / 2 / sample_size; }

  /// @brief Extract timestamp from message metadata.
  /// @param message The message.
  /// @return The timestamp.
  static uint64_t get_meta_timestamp(const int8_t* message)
  {
    return le64toh(*reinterpret_cast<const uint64_t*>(message + sizeof(uint32_t)));
  }

  /// @brief Extract flags from message metadata.
  /// @param message The message.
  /// @return The flags.
  static uint32_t get_meta_flags(const int8_t* message)
  {
    return le32toh(*reinterpret_cast<const uint32_t*>(message + sizeof(uint32_t) + sizeof(uint64_t)));
  }

  /// Counters.
  bool print_counters = false;

  struct {
    uint64_t last_reset_time     = 0;
    uint64_t last_timestamp      = 0;
    uint64_t last_callback_start = 0;
    uint64_t last_receive_start  = 0;
    size_t   samples_dropped     = 0;
    size_t   samples_missing     = 0;
    size_t   transfers_submitted = 0;
    size_t   transfers_acked     = 0;

    MinMaxCounters callback_interval;
    MinMaxCounters receive_interval;
    MinMaxCounters receive_time;
    MinMaxCounters conversion_time;
    MinMaxCounters submit_time;
    MinMaxCounters queued_transfers;

    void on_callback(uint64_t now)
    {
      if (last_callback_start != 0) {
        callback_interval.update(now - last_callback_start);
      }
      last_callback_start = now;
    }

    void on_receive_start(uint64_t now)
    {
      if (last_receive_start != 0) {
        receive_interval.update(now - last_receive_start);
      }
      last_receive_start = now;
    }

    void on_receive_end(uint64_t now) { receive_time.update(now - last_receive_start); }

    void on_convert_complete(uint64_t delta) { conversion_time.update(delta); }

    void on_submit_complete(uint64_t delta)
    {
      queued_transfers.update(transfers_acked - ++transfers_submitted);
      submit_time.update(delta);
    }

    bool should_print(uint64_t now) { return now - last_reset_time >= 1000000; }

    void reset(uint64_t now)
    {
      last_reset_time = now;
      samples_dropped = 0;
      samples_missing = 0;
      callback_interval.reset();
      receive_interval.reset();
      receive_time.reset();
      conversion_time.reset();
      submit_time.reset();
      queued_transfers.reset();
    }
  } counters;

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
                          radio_notification_handler& notifier_,
                          radio_bladerf_tx_stream&    tx_stream_);

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
