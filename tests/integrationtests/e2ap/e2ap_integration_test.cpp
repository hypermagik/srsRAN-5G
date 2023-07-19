/*
 *
 * Copyright 2021-2023 Software Radio Systems Limited
 *
 * By using this file, you agree to the terms and conditions set
 * forth in the LICENSE file which can be found at the top level of
 * the distribution.
 *
 */

#include "lib/cu_cp/ue_manager_impl.h"
#include "lib/e2/common/e2ap_asn1_packer.h"
#include "tests/unittests/e2/common/e2_test_helpers.h"
#include "srsran/e2/e2_factory.h"
#include "srsran/e2/e2ap_configuration_helpers.h"
#include "srsran/gateways/sctp_network_gateway_factory.h"
#include "srsran/support/async/async_test_utils.h"
#include "srsran/support/executors/manual_task_worker.h"
#include "srsran/support/io/io_broker_factory.h"
#include "srsran/support/test_utils.h"
#include "srsran/support/timers.h"
#include <gtest/gtest.h>

using namespace srsran;
using namespace srs_cu_cp;

/// This test is an integration test between:
/// * E2AP (including ASN1 packer and E2 setup procedure)
/// * SCTP network gateway
/// * IO broker
class e2ap_network_adapter : public e2_message_notifier,
                             public e2_message_handler,
                             public sctp_network_gateway_control_notifier,
                             public network_gateway_data_notifier
{
public:
  e2ap_network_adapter(const sctp_network_gateway_config& nw_config_) :
    nw_config(nw_config_),
    epoll_broker(create_io_broker(io_broker_type::epoll)),
    gw(create_sctp_network_gateway({nw_config, *this, *this})),
    packer(*gw, *this, pcap)
  {
    gw->create_and_connect();
    epoll_broker->register_fd(gw->get_socket_fd(), [this](int fd) { gw->receive(); });
  }

  ~e2ap_network_adapter() {}

  void connect_e2ap(e2_interface* e2ap_) { e2ap = e2ap_; }

private:
  // E2AP calls interface to send (unpacked) E2AP PDUs
  void on_new_message(const e2_message& msg) override { packer.handle_message(msg); }

  // SCTP network gateway calls interface to inject received PDUs (ASN1 packed)
  void on_new_pdu(byte_buffer pdu) override { packer.handle_packed_pdu(pdu); }

  // The packer calls this interface to inject unpacked E2AP PDUs
  void handle_message(const e2_message& msg) override { e2ap->handle_message(msg); }

  /// \brief Simply log those events for now
  void on_connection_loss() override { test_logger.info("on_connection_loss"); }
  void on_connection_established() override { test_logger.info("on_connection_established"); }

  /// We require a network gateway and a packer
  const sctp_network_gateway_config&    nw_config;
  std::unique_ptr<io_broker>            epoll_broker;
  std::unique_ptr<sctp_network_gateway> gw;
  e2ap_asn1_packer                      packer;
  e2_interface*                         e2ap = nullptr;
  dummy_e2ap_pcap                       pcap;

  srslog::basic_logger& test_logger = srslog::fetch_basic_logger("TEST");
};

class e2ap_integration_test : public ::testing::Test
{
protected:
  void SetUp() override
  {
    srslog::fetch_basic_logger("TEST").set_level(srslog::basic_levels::debug);
    srslog::init();

    cfg = srsran::config_helpers::make_default_e2ap_config();

    sctp_network_gateway_config nw_config;
    nw_config.connect_address   = "127.0.0.1";
    nw_config.connect_port      = 36421;
    nw_config.bind_address      = "127.0.0.101";
    nw_config.bind_port         = 0;
    nw_config.non_blocking_mode = true;

    adapter              = std::make_unique<e2ap_network_adapter>(nw_config);
    e2_subscription_mngr = std::make_unique<dummy_e2_subscription_mngr>();
    du_metrics           = std::make_unique<dummy_e2_du_metrics>();
    factory              = timer_factory{timers, ctrl_worker};
    e2ap                 = create_e2(cfg, factory, *adapter, *e2_subscription_mngr);
    pcap                 = std::make_unique<dummy_e2ap_pcap>();
    adapter->connect_e2ap(e2ap.get());
  }

  e2ap_configuration                       cfg;
  timer_factory                            factory;
  timer_manager                            timers;
  std::unique_ptr<e2ap_network_adapter>    adapter;
  manual_task_worker                       ctrl_worker{128};
  std::unique_ptr<dummy_e2ap_pcap>         pcap;
  std::unique_ptr<e2_subscription_manager> e2_subscription_mngr;
  std::unique_ptr<e2_du_metrics_interface> du_metrics;
  std::unique_ptr<e2_interface>            e2ap;
  srslog::basic_logger&                    test_logger = srslog::fetch_basic_logger("TEST");
};

/// Test successful e2 setup procedure
TEST_F(e2ap_integration_test, when_e2_setup_response_received_then_ric_connected)
{
  // Action 1: Launch E2 setup procedure
  e2_message request_msg = generate_e2_setup_request_message();
  test_logger.info("Launching E2 setup procedure...");
  e2_setup_request_message request;
  request.request                                 = request_msg.pdu.init_msg().value.e2setup_request();
  async_task<e2_setup_response_message>         t = e2ap->handle_e2_setup_request(request);
  lazy_task_launcher<e2_setup_response_message> t_launcher(t);

  // Status: Procedure not yet ready.
  ASSERT_FALSE(t.ready());

  std::this_thread::sleep_for(std::chrono::seconds(3));
}
