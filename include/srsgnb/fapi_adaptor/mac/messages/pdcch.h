/*
 *
 * Copyright 2013-2022 Software Radio Systems Limited
 *
 * By using this file, you agree to the terms and conditions set
 * forth in the LICENSE file which can be found at the top level of
 * the distribution.
 *
 */

#ifndef SRSGNB_FAPI_ADAPTOR_MAC_MESSAGES_PDCCH_H
#define SRSGNB_FAPI_ADAPTOR_MAC_MESSAGES_PDCCH_H

#include "srsgnb/fapi/message_builders.h"
#include "srsgnb/mac/mac_cell_result.h"

namespace srsgnb {
namespace fapi_adaptor {

// Groups the DCI information.
struct dci_info {
  const pdcch_dl_information* parameters;
  const dci_payload*          payload;
};

// Groups the MAC PDCCH PDU.
struct mac_pdcch_pdu {
  const bwp_configuration*     bwp_cfg;
  const coreset_configuration* coreset_cfg;
  std::vector<dci_info>        dcis;
};

/// \brief Helper function that converts from a PDCCH MAC PDU to a PDCCH FAPI PDU.
///
/// \param[out] fapi_pdu PDCCH FAPI PDU that will store the converted data.
/// \param[in] bwp_cfg  Contains the BWP configuration information of the PDCCH PDU..
/// \param[in] coreset_cfg Contains the coreset information of the PDCCH PDU.
/// \param[in] dcis Span that contains the DL DCI information for the PDCCH PDU.
void convert_pdcch_mac_to_fapi(fapi::dl_pdcch_pdu& fapi_pdu, const mac_pdcch_pdu& mac_pdu);

/// \brief Helper function that converts from a PDCCH MAC PDU to a PDCCH FAPI PDU.
///
/// \param[out] builder PDCCH FAPI builder that helps to fill the PDU.
/// \param[in] bwp_cfg  Contains the BWP configuration information of the PDCCH PDU..
/// \param[in] coreset_cfg Contains the coreset information of the PDCCH PDU.
/// \param[in] dcis Span that contains the DL DCI information for the PDCCH PDU.
void convert_pdcch_mac_to_fapi(fapi::dl_pdcch_pdu_builder& builder, const mac_pdcch_pdu& mac_pdu);

} // namespace fapi_adaptor
} // namespace srsgnb

#endif // SRSGNB_FAPI_ADAPTOR_MAC_MESSAGES_PDCCH_H
