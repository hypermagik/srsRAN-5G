/*
 *
 * Copyright 2021-2025 Software Radio Systems Limited
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

#include "srsran/adt/span.h"
#include "srsran/fapi/messages/crc_indication.h"
#include "srsran/fapi/messages/dl_tti_request.h"
#include "srsran/fapi/messages/error_indication.h"
#include "srsran/fapi/messages/rach_indication.h"
#include "srsran/fapi/messages/rx_data_indication.h"
#include "srsran/fapi/messages/slot_indication.h"
#include "srsran/fapi/messages/srs_indication.h"
#include "srsran/fapi/messages/tx_data_request.h"
#include "srsran/fapi/messages/uci_indication.h"
#include "srsran/fapi/messages/ul_dci_request.h"
#include "srsran/fapi/messages/ul_tti_request.h"
#include "srsran/ran/dmrs.h"
#include "srsran/ran/pdcch/coreset.h"
#include "srsran/ran/pdcch/dci_packing.h"
#include "srsran/ran/ptrs/ptrs.h"
#include "srsran/ran/srs/srs_configuration.h"
#include "srsran/support/math/math_utils.h"
#include "srsran/support/shared_transport_block.h"
#include <algorithm>

namespace srsran {
namespace fapi {

namespace detail {

/// \brief Sets the value of a bit in the bitmap. When enable is true, it sets the bit, otherwise it clears the bit.
/// \param[in] bitmap Bitmap to modify.
/// \param[in] bit Bit to change.
/// \param[in] enable Value to set. If true, sets the bit(1), otherwise clears it(0).
/// \note Use this function with integer data types, otherwise it produces undefined behaviour.
template <typename Integer>
void set_bitmap_bit(Integer& bitmap, unsigned bit, bool enable)
{
  static_assert(std::is_integral<Integer>::value, "Integral required");
  srsran_assert(sizeof(bitmap) * 8 > bit, "Requested bit ({}), exceeds the bitmap size({})", bit, sizeof(bitmap) * 8);

  if (enable) {
    bitmap |= (1U << bit);
  } else {
    bitmap &= ~(1U << bit);
  }
}

/// \brief Checks the value of a bit in the bitmap and returns a true if the bit is set, otherwise false.
/// \param[in] bitmap Bitmap to check.
/// \param[in] bit Bit to check.
/// \return True when the bit equals 1, otherwise false.
/// \note Use this function with integer data types, otherwise it produces undefined behaviour.
template <typename Integer>
bool check_bitmap_bit(Integer bitmap, unsigned bit)
{
  static_assert(std::is_integral<Integer>::value, "Integral required");
  srsran_assert(sizeof(bitmap) * 8 > bit, "Requested bit ({}), exceeds the bitmap size({})", bit, sizeof(bitmap) * 8);

  return (bitmap & (1U << bit));
}

} // namespace detail

// :TODO: Review the builders documentation so it matches the UCI builder.

/// Helper class to fill the transmission precoding and beamforming parameters specified in SCF-222 v4.0
/// section 3.4.2.5.
class tx_precoding_and_beamforming_pdu_builder
{
  tx_precoding_and_beamforming_pdu& pdu;

public:
  explicit tx_precoding_and_beamforming_pdu_builder(tx_precoding_and_beamforming_pdu& pdu_) : pdu(pdu_)
  {
    // Mark the tx precoding and beamforming pdu as used when this builder is called.
    pdu.trp_scheme = 0U;
    // Initialize number of digital beamforming interfaces.
    pdu.dig_bf_interfaces = 0U;
  }

  /// Sets the basic parameters for the fields of the tranmission precoding and beamforming PDU.
  /// \note These parameters are specified in SCF-222 v4.0 section 3.4.2.5, in table Tx precoding and beamforming PDU.
  tx_precoding_and_beamforming_pdu_builder& set_basic_parameters(unsigned prg_size, unsigned dig_bf_interfaces)
  {
    pdu.prg_size          = prg_size;
    pdu.dig_bf_interfaces = dig_bf_interfaces;

    return *this;
  }

  /// Adds a PRG to the transmission precoding and beamforming PDU.
  /// \note These parameters are specified in SCF-222 v4.0 section 3.4.2.5, in table Tx precoding and beamforming PDU.
  tx_precoding_and_beamforming_pdu_builder& add_prg(unsigned pm_index, span<const uint16_t> beam_index)
  {
    tx_precoding_and_beamforming_pdu::prgs_info& prg = pdu.prgs.emplace_back();

    srsran_assert(pdu.dig_bf_interfaces == beam_index.size(),
                  "Error number of beam indexes={} does not match the expected={}",
                  beam_index.size(),
                  pdu.dig_bf_interfaces);

    prg.pm_index = pm_index;
    prg.beam_index.assign(beam_index.begin(), beam_index.end());

    return *this;
  }
};

/// Helper class to fill in the DL SSB PDU parameters specified in SCF-222 v4.0 section 3.4.2.4.
class dl_ssb_pdu_builder
{
public:
  explicit dl_ssb_pdu_builder(dl_ssb_pdu& pdu_) : pdu(pdu_), v3(pdu_.ssb_maintenance_v3) {}

  /// Sets the basic parameters for the fields of the SSB/PBCH PDU.
  /// \note These parameters are specified in SCF-222 v4.0 section 3.4.2.4, in table SSB/PBCH PDU.
  dl_ssb_pdu_builder& set_basic_parameters(pci_t                 phys_cell_id,
                                           beta_pss_profile_type beta_pss_profile_nr,
                                           uint8_t               ssb_block_index,
                                           uint8_t               ssb_subcarrier_offset,
                                           ssb_offset_to_pointA  ssb_offset_pointA)
  {
    pdu.phys_cell_id          = phys_cell_id;
    pdu.beta_pss_profile_nr   = beta_pss_profile_nr;
    pdu.ssb_block_index       = ssb_block_index;
    pdu.ssb_subcarrier_offset = ssb_subcarrier_offset;
    pdu.ssb_offset_pointA     = ssb_offset_pointA;

    return *this;
  }

  /// Sets the BCH payload configured by the MAC and returns a reference to the builder.
  /// \note Use this function when the MAC generates the full PBCH payload.
  /// \note These parameters are specified in SCF-222 v4.0 section 3.4.2.4, in table MAC generated MIB PDU.
  /// \note This function assumes that given bch_payload value is codified as: a0,a1,a2,...,a29,a30,a31, with the most
  /// significant bit being the leftmost (in this case a0 in position 31 of the uint32_t).
  dl_ssb_pdu_builder& set_bch_payload_mac_full(uint32_t bch_payload)
  {
    // Configure the BCH payload as fully generated by the MAC.
    pdu.bch_payload_flag        = bch_payload_type::mac_full;
    pdu.bch_payload.bch_payload = bch_payload;

    return *this;
  }

  /// Sets the BCH payload and returns a reference to the builder. PHY configures the timing PBCH bits.
  /// \note Use this function when the PHY generates the timing PBCH information.
  /// \note These parameters are specified in SCF-222 v4.0 section 3.4.2.4, in table MAC generated MIB PDU.
  /// \note This function assumes that given bch_payload value is codified as: 0,0,0,0,0,0,0,0,a0,a1,a2,...,a21,a22,a23,
  /// with the most significant bit being the leftmost (in this case a0 in position 24 of the uint32_t).
  dl_ssb_pdu_builder& set_bch_payload_phy_timing_info(uint32_t bch_payload)
  {
    static constexpr unsigned MAX_SIZE = (1U << 24U);

    srsran_assert(bch_payload < MAX_SIZE, "BCH payload value out of bounds");

    pdu.bch_payload_flag = bch_payload_type::phy_timing_info;
    // Clear unused bits in the high part of the payload.
    pdu.bch_payload.bch_payload = bch_payload & (MAX_SIZE - 1);

    return *this;
  }

  /// Sets the BCH payload configured by the PHY and returns a reference to the builder.
  /// \note Use this function when the PHY generates the full PBCH payload.
  /// \note These parameters are specified in SCF-222 v4.0 section 3.4.2.4, in table PHY generated MIB PDU.
  dl_ssb_pdu_builder& set_bch_payload_phy_full(dmrs_typeA_position dmrs_type_a_position,
                                               uint8_t             pdcch_config_sib1,
                                               bool                cell_barred,
                                               bool                intra_freq_reselection)
  {
    pdu.bch_payload_flag = bch_payload_type::phy_full;
    pdu.bch_payload.phy_mib_pdu.dmrs_typeA_position =
        (dmrs_type_a_position == dmrs_typeA_position::pos2) ? dmrs_typeA_pos::pos2 : dmrs_typeA_pos::pos3;
    pdu.bch_payload.phy_mib_pdu.pdcch_config_sib1     = pdcch_config_sib1;
    pdu.bch_payload.phy_mib_pdu.cell_barred           = cell_barred;
    pdu.bch_payload.phy_mib_pdu.intrafreq_reselection = intra_freq_reselection;

    return *this;
  }

  /// Sets the maintenance v3 basic parameters and returns a reference to the builder.
  /// \note These parameters are specified in SCF-222 v4.0 section 3.4.2.4, in table SSB/PBCH PDU maintenance FAPIv3.
  /// \note ssbPduIndex field is automatically filled when adding a new SSB PDU to the DL TTI request message.
  dl_ssb_pdu_builder&
  set_maintenance_v3_basic_parameters(ssb_pattern_case case_type, subcarrier_spacing scs, uint8_t L_max)
  {
    v3.case_type = case_type;
    v3.scs       = scs;
    v3.L_max     = L_max;

    return *this;
  }

  /// Returns a transmission precoding and beamforming PDU builder of this SSB PDU.
  tx_precoding_and_beamforming_pdu_builder get_tx_precoding_and_beamforming_pdu_builder()
  {
    tx_precoding_and_beamforming_pdu_builder builder(pdu.precoding_and_beamforming);

    return builder;
  }

private:
  dl_ssb_pdu&            pdu;
  dl_ssb_maintenance_v3& v3;
};

/// Helper class to fill in the DL DCI PDU parameters specified in SCF-222 v4.0 section 3.4.2.1, including the PDCCH PDU
/// maintenance FAPIv3 and PDCCH PDU FAPIv4 parameters.
class dl_dci_pdu_builder
{
public:
  dl_dci_pdu_builder(dl_dci_pdu&                                    pdu_,
                     dl_pdcch_pdu_maintenance_v3::maintenance_info& pdu_v3_,
                     dl_pdcch_pdu_parameters_v4::dci_params&        pdu_v4_) :
    pdu(pdu_), pdu_v3(pdu_v3_), pdu_v4(pdu_v4_)
  {
    pdu_v3.collocated_AL16_candidate = false;
  }

  /// Sets the basic parameters for the fields of the DL DCI PDU.
  /// \note These parameters are specified in SCF-222 v4.0 section 3.4.2.1, in table DL DCI PDU.
  dl_dci_pdu_builder& set_basic_parameters(rnti_t   rnti,
                                           uint16_t nid_pdcch_data,
                                           uint16_t nrnti_pdcch_data,
                                           uint8_t  cce_index,
                                           uint8_t  aggregation_level)
  {
    pdu.rnti              = rnti;
    pdu.nid_pdcch_data    = nid_pdcch_data;
    pdu.nrnti_pdcch_data  = nrnti_pdcch_data;
    pdu.cce_index         = cce_index;
    pdu.aggregation_level = aggregation_level;

    return *this;
  }

  /// Sets the transmission power info parameters for the fields of the DL DCI PDU.
  /// \note These parameters are specified in SCF-222 v4.0 section 3.4.2.1, in table DL DCI PDU.
  dl_dci_pdu_builder& set_tx_power_info_parameter(int power_control_offset_ss_dB)
  {
    pdu.power_control_offset_ss_profile_nr = power_control_offset_ss_dB;

    return *this;
  }

  /// Sets the payload of the DL DCI PDU.
  /// \note These parameters are specified in SCF-222 v4.0 section 3.4.2.1, in table DL DCI PDU.
  dl_dci_pdu_builder& set_payload(const dci_payload& payload)
  {
    pdu.payload = payload;

    return *this;
  }

  /// Sets the DCI parameters of the PDCCH parameters v4.
  /// \note These parameters are specified in SCF-222 v4.0 section 3.4.2.1, in table PDCCH PDU parameters FAPIv4.
  dl_dci_pdu_builder& set_parameters_v4_dci(uint16_t nid_pdcch_dmrs)
  {
    pdu_v4.nid_pdcch_dmrs = nid_pdcch_dmrs;

    return *this;
  }

  /// Sets the PDCCH context as vendor specific.
  dl_dci_pdu_builder set_context_vendor_specific(search_space_id         ss_id,
                                                 const char*             dci_format,
                                                 std::optional<unsigned> harq_feedback_timing)
  {
    pdu.context = pdcch_context(ss_id, dci_format, harq_feedback_timing);
    return *this;
  }

  /// Returns a transmission precoding and beamforming PDU builder of this DL DCI PDU.
  tx_precoding_and_beamforming_pdu_builder get_tx_precoding_and_beamforming_pdu_builder()
  {
    tx_precoding_and_beamforming_pdu_builder builder(pdu.precoding_and_beamforming);

    return builder;
  }

private:
  dl_dci_pdu&                                    pdu;
  dl_pdcch_pdu_maintenance_v3::maintenance_info& pdu_v3;
  dl_pdcch_pdu_parameters_v4::dci_params&        pdu_v4;
};

/// Helper class to fill in the DL PDCCH PDU parameters specified in SCF-222 v4.0 section 3.4.2.1.
class dl_pdcch_pdu_builder
{
public:
  explicit dl_pdcch_pdu_builder(dl_pdcch_pdu& pdu_) : pdu(pdu_) {}

  /// Sets the BWP parameters for the fields of the PDCCH PDU.
  /// \note These parameters are specified in SCF-222 v4.0 section 3.4.2.1, in table PDCCH PDU.
  dl_pdcch_pdu_builder&
  set_bwp_parameters(uint16_t coreset_bwp_size, uint16_t coreset_bwp_start, subcarrier_spacing scs, cyclic_prefix cp)
  {
    pdu.coreset_bwp_size  = coreset_bwp_size;
    pdu.coreset_bwp_start = coreset_bwp_start;
    pdu.scs               = scs;
    pdu.cp                = cp;

    return *this;
  }

  /// Sets the coreset parameters for the fields of the PDCCH PDU.
  /// \note These parameters are specified in SCF-222 v4.0 section 3.4.2.1, in table PDCCH PDU.
  dl_pdcch_pdu_builder& set_coreset_parameters(uint8_t                                          start_symbol_index,
                                               uint8_t                                          duration_symbols,
                                               const freq_resource_bitmap&                      freq_domain_resource,
                                               cce_to_reg_mapping_type                          cce_req_mapping_type,
                                               uint8_t                                          reg_bundle_size,
                                               uint8_t                                          interleaver_size,
                                               pdcch_coreset_type                               coreset_type,
                                               uint16_t                                         shift_index,
                                               coreset_configuration::precoder_granularity_type precoder_granularity)
  {
    pdu.start_symbol_index   = start_symbol_index;
    pdu.duration_symbols     = duration_symbols;
    pdu.freq_domain_resource = freq_domain_resource;
    pdu.cce_reg_mapping_type = cce_req_mapping_type;
    pdu.reg_bundle_size      = reg_bundle_size;
    pdu.interleaver_size     = interleaver_size;
    pdu.coreset_type         = coreset_type;
    pdu.shift_index          = shift_index;
    pdu.precoder_granularity = precoder_granularity;

    return *this;
  }

  /// Adds a DL DCI PDU to the PDCCH PDU.
  /// \note These parameters are specified in SCF-222 v4.0 section 3.4.2.1, in table PDCCH PDU.
  dl_dci_pdu_builder add_dl_dci()
  {
    // Save the size as the index value for the DL DCI.
    unsigned dci_id = pdu.dl_dci.size();

    // Set the DL DCI index.
    dl_pdcch_pdu_maintenance_v3::maintenance_info& info = pdu.maintenance_v3.info.emplace_back();
    info.dci_index                                      = dci_id;

    dl_dci_pdu_builder builder(pdu.dl_dci.emplace_back(), info, pdu.parameters_v4.params.emplace_back());

    return builder;
  }

private:
  dl_pdcch_pdu& pdu;
};

/// Builder that helps to fill the parameters of a DL PDSCH codeword.
class dl_pdsch_codeword_builder
{
public:
  dl_pdsch_codeword_builder(dl_pdsch_codeword& cw_, uint8_t& cbg_tx_information_) :
    cw(cw_), cbg_tx_information(cbg_tx_information_)
  {
  }

  /// Sets the codeword basic parameters.
  /// \note These parameters are specified in SCF-222 v4.0 section 3.4.2.2, in table PDSCH PDU.
  dl_pdsch_codeword_builder& set_basic_parameters(float        target_code_rate,
                                                  uint8_t      qam_mod,
                                                  uint8_t      mcs_index,
                                                  uint8_t      mcs_table,
                                                  uint8_t      rv_index,
                                                  units::bytes tb_size)
  {
    cw.target_code_rate = target_code_rate * 10.F;
    cw.qam_mod_order    = qam_mod;
    cw.mcs_index        = mcs_index;
    cw.mcs_table        = mcs_table;
    cw.rv_index         = rv_index;
    cw.tb_size          = tb_size;

    return *this;
  }

  /// Sets the maintenance v3 parameters of the codeword.
  /// \note These parameters are specified in SCF-222 v4.0 section 3.4.2.2, in table PDSCH maintenance parameters V3.
  dl_pdsch_codeword_builder& set_maintenance_v3_parameters(uint8_t cbg_tx_info)
  {
    cbg_tx_information = cbg_tx_info;

    return *this;
  }

private:
  dl_pdsch_codeword& cw;
  uint8_t&           cbg_tx_information;
};

/// DL PDSCH PDU builder that helps to fill the parameters specified in SCF-222 v4.0 section 3.4.2.2.
class dl_pdsch_pdu_builder
{
public:
  explicit dl_pdsch_pdu_builder(dl_pdsch_pdu& pdu_) : pdu(pdu_)
  {
    pdu.pdu_bitmap                           = 0U;
    pdu.is_last_cb_present                   = 0U;
    pdu.pdsch_maintenance_v3.tb_crc_required = 0U;
    pdu.rb_bitmap.fill(0);
    pdu.dl_tb_crc_cw.fill(0);
    pdu.pdsch_maintenance_v3.ssb_pdus_for_rate_matching.fill(0);
  }

  /// Returns the PDU index.
  unsigned get_pdu_id() const { return pdu.pdu_index; }

  /// Sets the basic parameters for the fields of the PDSCH PDU.
  /// \note These parameters are specified in SCF-222 v4.0 section 3.4.2.2, in table PDSCH PDU.
  dl_pdsch_pdu_builder& set_basic_parameters(rnti_t rnti)
  {
    pdu.rnti = rnti;

    return *this;
  }

  /// Sets the BWP parameters for the fields of the PDSCH PDU.
  /// \note These parameters are specified in SCF-222 v4.0 section 3.4.2.2, in table PDSCH PDU.
  dl_pdsch_pdu_builder&
  set_bwp_parameters(uint16_t bwp_size, uint16_t bwp_start, subcarrier_spacing scs, cyclic_prefix cp)
  {
    pdu.bwp_size  = bwp_size;
    pdu.bwp_start = bwp_start;
    pdu.scs       = scs;
    pdu.cp        = cp;

    return *this;
  }

  /// Adds a codeword to the PDSCH PDU and returns a codeword builder to fill the codeword parameters.
  /// \note These parameters are specified in SCF-222 v4.0 section 3.4.2.2, in table PDSCH PDU.
  dl_pdsch_codeword_builder add_codeword()
  {
    dl_pdsch_codeword_builder builder(pdu.cws.emplace_back(),
                                      pdu.pdsch_maintenance_v3.cbg_tx_information.emplace_back());

    return builder;
  }

  /// Sets the codeword information parameters for the fields of the PDSCH PDU.
  /// \note These parameters are specified in SCF-222 v4.0 section 3.4.2.2, in table PDSCH PDU.
  dl_pdsch_pdu_builder& set_codeword_information_parameters(uint16_t             n_id_pdsch,
                                                            uint8_t              num_layers,
                                                            uint8_t              trasnmission_scheme,
                                                            pdsch_ref_point_type ref_point)
  {
    pdu.nid_pdsch           = n_id_pdsch;
    pdu.num_layers          = num_layers;
    pdu.transmission_scheme = trasnmission_scheme;
    pdu.ref_point           = ref_point;

    return *this;
  }

  /// Sets the DMRS parameters for the fields of the PDSCH PDU.
  /// \note These parameters are specified in SCF-222 v4.0 section 3.4.2.2, in table PDSCH PDU.
  dl_pdsch_pdu_builder& set_dmrs_parameters(uint16_t           dl_dmrs_symb_pos,
                                            dmrs_config_type   dmrs_type,
                                            uint16_t           pdsch_dmrs_scrambling_id,
                                            uint16_t           pdsch_dmrs_scrambling_id_complement,
                                            low_papr_dmrs_type low_parp_dmrs,
                                            uint8_t            nscid,
                                            uint8_t            num_dmrs_cdm_groups_no_data,
                                            uint16_t           dmrs_ports)
  {
    pdu.dl_dmrs_symb_pos = dl_dmrs_symb_pos;
    pdu.dmrs_type        = (dmrs_type == dmrs_config_type::type1) ? dmrs_cfg_type::type_1 : dmrs_cfg_type::type_2;
    pdu.pdsch_dmrs_scrambling_id       = pdsch_dmrs_scrambling_id;
    pdu.pdsch_dmrs_scrambling_id_compl = pdsch_dmrs_scrambling_id_complement;
    pdu.low_papr_dmrs                  = low_parp_dmrs;
    pdu.nscid                          = nscid;
    pdu.num_dmrs_cdm_grps_no_data      = num_dmrs_cdm_groups_no_data;
    pdu.dmrs_ports                     = dmrs_ports;

    return *this;
  }

  /// Sets the PDSCH allocation in frequency type 0 parameters for the fields of the PDSCH PDU.
  /// \note These parameters are specified in SCF-222 v4.0 section 3.4.2.2, in table PDSCH PDU.
  dl_pdsch_pdu_builder& set_pdsch_allocation_in_frequency_type_0(span<const uint8_t>     rb_map,
                                                                 vrb_to_prb_mapping_type vrb_to_prb_mapping)
  {
    pdu.resource_alloc     = resource_allocation_type::type_0;
    pdu.vrb_to_prb_mapping = vrb_to_prb_mapping;

    srsran_assert(rb_map.size() <= dl_pdsch_pdu::MAX_SIZE_RB_BITMAP,
                  "[PDSCH Builder] - Incoming RB bitmap size {} exceeds FAPI bitmap field {}",
                  rb_map.size(),
                  int(dl_pdsch_pdu::MAX_SIZE_RB_BITMAP));

    std::copy(rb_map.begin(), rb_map.end(), pdu.rb_bitmap.begin());

    // Fill in these fields, although they belong to allocation type 1.
    pdu.rb_start = 0;
    pdu.rb_size  = 0;

    return *this;
  }

  /// Sets the PDSCH allocation in frequency type 1 parameters for the fields of the PDSCH PDU.
  /// \note These parameters are specified in SCF-222 v4.0 section 3.4.2.2, in table PDSCH PDU.
  dl_pdsch_pdu_builder& set_pdsch_allocation_in_frequency_type_1(uint16_t                rb_start,
                                                                 uint16_t                rb_size,
                                                                 vrb_to_prb_mapping_type vrb_to_prb_mapping)
  {
    pdu.resource_alloc     = resource_allocation_type::type_1;
    pdu.rb_start           = rb_start;
    pdu.rb_size            = rb_size;
    pdu.vrb_to_prb_mapping = vrb_to_prb_mapping;

    return *this;
  }

  /// Sets the PDSCH allocation in time parameters for the fields of the PDSCH PDU.
  /// \note These parameters are specified in SCF-222 v4.0 section 3.4.2.2, in table PDSCH PDU.
  dl_pdsch_pdu_builder& set_pdsch_allocation_in_time_parameters(uint8_t start_symbol_index, uint8_t nof_symbols)
  {
    pdu.start_symbol_index = start_symbol_index;
    pdu.nr_of_symbols      = nof_symbols;

    return *this;
  }

  dl_pdsch_pdu_builder& set_ptrs_params()
  {
    pdu.pdu_bitmap.set(dl_pdsch_pdu::PDU_BITMAP_PTRS_BIT);
    // :TODO: Implement me!

    return *this;
  }

  // :TODO: Beamforming.

  /// Sets the Tx Power info parameters for the fields of the PDSCH PDU.
  /// \note These parameters are specified in SCF-222 v4.0 section 3.4.2.2, in table PDSCH PDU.
  dl_pdsch_pdu_builder& set_tx_power_info_parameters(int                     power_control_offset,
                                                     power_control_offset_ss power_control_offset_ss)
  {
    pdu.power_control_offset_profile_nr    = power_control_offset;
    pdu.power_control_offset_ss_profile_nr = power_control_offset_ss;

    return *this;
  }

  /// Sets the CBG ReTx Ctrl parameters for the fields of the PDSCH PDU.
  /// \note These parameters are specified in SCF-222 v4.0 section 3.4.2.2, in table PDSCH PDU.
  dl_pdsch_pdu_builder& set_cbg_re_tx_ctrl_parameters(bool                 last_cb_present_first_tb,
                                                      bool                 last_cb_present_second_tb,
                                                      inline_tb_crc_type   tb_crc,
                                                      span<const uint32_t> dl_tb_crc_cw)
  {
    pdu.pdu_bitmap.set(dl_pdsch_pdu::PDU_BITMAP_CBG_RETX_CTRL_BIT);

    detail::set_bitmap_bit<uint8_t>(
        pdu.is_last_cb_present, dl_pdsch_pdu::LAST_CB_BITMAP_FIRST_TB_BIT, last_cb_present_first_tb);
    detail::set_bitmap_bit<uint8_t>(
        pdu.is_last_cb_present, dl_pdsch_pdu::LAST_CB_BITMAP_SECOND_TB_BIT, last_cb_present_second_tb);

    pdu.is_inline_tb_crc = tb_crc;

    srsran_assert(dl_tb_crc_cw.size() <= dl_pdsch_pdu::MAX_SIZE_DL_TB_CRC,
                  "[PDSCH Builder] - Incoming DL TB CRC size ({}) out of bounds ({})",
                  dl_tb_crc_cw.size(),
                  int(dl_pdsch_pdu::MAX_SIZE_DL_TB_CRC));
    std::copy(dl_tb_crc_cw.begin(), dl_tb_crc_cw.end(), pdu.dl_tb_crc_cw.begin());

    return *this;
  }

  /// Sets the maintenance v3 BWP information parameters for the fields of the PDSCH PDU.
  /// \note These parameters are specified in SCF-222 v4.0 section 3.4.2.2, in table PDSCH maintenance parameters v3.
  dl_pdsch_pdu_builder& set_maintenance_v3_bwp_parameters(pdsch_trans_type trans_type,
                                                          uint16_t         coreset_start_point,
                                                          uint16_t         initial_dl_bwp_size)
  {
    pdu.pdsch_maintenance_v3.trans_type          = trans_type;
    pdu.pdsch_maintenance_v3.coreset_start_point = coreset_start_point;
    pdu.pdsch_maintenance_v3.initial_dl_bwp_size = initial_dl_bwp_size;

    return *this;
  }

  /// Sets the maintenance v3 codeword information parameters for the fields of the PDSCH PDU.
  /// \note These parameters are specified in SCF-222 v4.0 section 3.4.2.2, in table PDSCH maintenance parameters v3.
  dl_pdsch_pdu_builder& set_maintenance_v3_codeword_parameters(ldpc_base_graph_type ldpc_base_graph,
                                                               units::bytes         tb_size_lbrm_bytes,
                                                               bool                 tb_crc_first_tb_required,
                                                               bool                 tb_crc_second_tb_required)
  {
    pdu.pdsch_maintenance_v3.ldpc_base_graph    = ldpc_base_graph;
    pdu.pdsch_maintenance_v3.tb_size_lbrm_bytes = tb_size_lbrm_bytes;

    // Fill the bitmap.
    detail::set_bitmap_bit<uint8_t>(pdu.pdsch_maintenance_v3.tb_crc_required,
                                    dl_pdsch_maintenance_parameters_v3::TB_BITMAP_FIRST_TB_BIT,
                                    tb_crc_first_tb_required);
    detail::set_bitmap_bit<uint8_t>(pdu.pdsch_maintenance_v3.tb_crc_required,
                                    dl_pdsch_maintenance_parameters_v3::TB_BITMAP_SECOND_TB_BIT,
                                    tb_crc_second_tb_required);

    return *this;
  }

  /// Sets the maintenance v3 CSI-RS rate matching references parameters for the fields of the PDSCH PDU.
  /// \note These parameters are specified in SCF-222 v4.0 section 3.4.2.2, in table PDSCH maintenance parameters v3.
  dl_pdsch_pdu_builder& set_maintenance_v3_csi_rm_references(span<const uint16_t> csi_rs_for_rm)
  {
    pdu.pdsch_maintenance_v3.csi_for_rm.assign(csi_rs_for_rm.begin(), csi_rs_for_rm.end());

    return *this;
  }

  /// Sets the maintenance v3 rate matching references parameters for the fields of the PDSCH PDU.
  /// \note These parameters are specified in SCF-222 v4.0 section 3.4.2.2, in table PDSCH maintenance parameters v3.
  dl_pdsch_pdu_builder&
  set_maintenance_v3_rm_references_parameters(span<const uint16_t> ssb_pdus_for_rm,
                                              uint16_t             ssb_config_for_rm,
                                              uint8_t              prb_sym_rm_pattern_bitmap_size,
                                              span<const uint8_t>  prb_sym_rm_pattern_bitmap_by_reference,
                                              uint16_t             pdcch_pdu_index,
                                              uint16_t             dci_index,
                                              uint8_t              lte_crs_rm_pattern_bitmap_size,
                                              span<const uint8_t>  lte_crs_rm_pattern)
  {
    srsran_assert(ssb_pdus_for_rm.size() <= dl_pdsch_maintenance_parameters_v3::MAX_SIZE_SSB_PDU_FOR_RM,
                  "[PDSCH Builder] - Incoming SSB PDUs for RM matching size ({}) doesn't fit the field ({})",
                  ssb_pdus_for_rm.size(),
                  int(dl_pdsch_maintenance_parameters_v3::MAX_SIZE_SSB_PDU_FOR_RM));
    std::copy(
        ssb_pdus_for_rm.begin(), ssb_pdus_for_rm.end(), pdu.pdsch_maintenance_v3.ssb_pdus_for_rate_matching.begin());

    pdu.pdsch_maintenance_v3.ssb_config_for_rate_matching         = ssb_config_for_rm;
    pdu.pdsch_maintenance_v3.prb_sym_rm_pattern_bitmap_size_byref = prb_sym_rm_pattern_bitmap_size;
    pdu.pdsch_maintenance_v3.prb_sym_rm_patt_bmp_byref.assign(prb_sym_rm_pattern_bitmap_by_reference.begin(),
                                                              prb_sym_rm_pattern_bitmap_by_reference.end());

    // These two parameters are set to 0 for this release FAPI v4.
    pdu.pdsch_maintenance_v3.num_prb_sym_rm_patts_by_value = 0U;
    pdu.pdsch_maintenance_v3.num_coreset_rm_patterns       = 0U;

    pdu.pdsch_maintenance_v3.pdcch_pdu_index = pdcch_pdu_index;
    pdu.pdsch_maintenance_v3.dci_index       = dci_index;

    pdu.pdsch_maintenance_v3.lte_crs_rm_pattern_bitmap_size = lte_crs_rm_pattern_bitmap_size;
    pdu.pdsch_maintenance_v3.lte_crs_rm_pattern.assign(lte_crs_rm_pattern.begin(), lte_crs_rm_pattern.end());

    return *this;
  }

  /// Sets the maintenance v3 CBG retx control parameters for the fields of the PDSCH PDU.
  /// \note These parameters are specified in SCF-222 v4.0 section 3.4.2.2, in table PDSCH maintenance parameters v3.
  dl_pdsch_pdu_builder& set_maintenance_v3_cbg_tx_crtl_parameters(uint8_t max_num_cbg_per_tb)
  {
    pdu.pdsch_maintenance_v3.max_num_cbg_per_tb = max_num_cbg_per_tb;

    return *this;
  }

  /// Sets the PDSCH maintenance v4 basic parameters for the fields of the PDSCH PDU.
  /// \note These parameters are specified in SCF-222 v4.0 section 3.4.2.2, in table PDSCH maintenance FAPIv4.
  dl_pdsch_pdu_builder& set_maintenance_v4_basic_parameters(uint8_t coreset_rm_pattern_bitmap_by_reference_bitmap_size,
                                                            span<const uint8_t> coreset_rm_pattern_bitmap_by_reference,
                                                            uint8_t             lte_crs_mbsfn_derivation_method,
                                                            span<const uint8_t> lte_crs_mbsfn_pattern)
  {
    pdu.pdsch_parameters_v4.lte_crs_mbsfn_derivation_method       = lte_crs_mbsfn_derivation_method;
    pdu.pdsch_parameters_v4.coreset_rm_pattern_bitmap_size_by_ref = coreset_rm_pattern_bitmap_by_reference_bitmap_size;

    pdu.pdsch_parameters_v4.coreset_rm_pattern_bmp_by_ref.assign(coreset_rm_pattern_bitmap_by_reference.begin(),
                                                                 coreset_rm_pattern_bitmap_by_reference.end());

    pdu.pdsch_parameters_v4.lte_crs_mbsfn_pattern.assign(lte_crs_mbsfn_pattern.begin(), lte_crs_mbsfn_pattern.end());

    return *this;
  }

  /// Sets the PDSCH context as vendor specific.
  dl_pdsch_pdu_builder& set_context_vendor_specific(harq_id_t harq_id, unsigned k1, unsigned nof_retxs)
  {
    pdu.context = pdsch_context(harq_id, k1, nof_retxs);
    return *this;
  }

  /// Returns a transmission precoding and beamforming PDU builder of this PDSCH PDU.
  tx_precoding_and_beamforming_pdu_builder get_tx_precoding_and_beamforming_pdu_builder()
  {
    tx_precoding_and_beamforming_pdu_builder builder(pdu.precoding_and_beamforming);

    return builder;
  }

  // :TODO: FAPIv4 MU-MIMO.

private:
  dl_pdsch_pdu& pdu;
};

/// CSI-RS PDU builder that helps to fill in the parameters specified in SCF-222 v4.0 section 3.4.2.3.
class dl_csi_rs_pdu_builder
{
  dl_csi_rs_pdu& pdu;

public:
  explicit dl_csi_rs_pdu_builder(dl_csi_rs_pdu& pdu_) : pdu(pdu_) {}

  /// Sets the CSI-RS PDU basic parameters and returns a reference to the builder.
  /// \note These parameters are specified in SCF-222 v4.0 section 3.4.2.3 in table CSI-RS PDU.
  dl_csi_rs_pdu_builder& set_basic_parameters(uint16_t                         start_rb,
                                              uint16_t                         nof_rbs,
                                              csi_rs_type                      type,
                                              uint8_t                          row,
                                              const bounded_bitset<12, false>& freq_domain,
                                              uint8_t                          symb_l0,
                                              uint8_t                          symb_l1,
                                              csi_rs_cdm_type                  cdm_type,
                                              csi_rs_freq_density_type         freq_density,
                                              uint16_t                         scrambling_id)
  {
    pdu.start_rb     = start_rb;
    pdu.num_rbs      = nof_rbs;
    pdu.type         = type;
    pdu.row          = row;
    pdu.freq_domain  = freq_domain;
    pdu.symb_L0      = symb_l0;
    pdu.symb_L1      = symb_l1;
    pdu.cdm_type     = cdm_type;
    pdu.freq_density = freq_density;
    pdu.scramb_id    = scrambling_id;

    return *this;
  }

  /// Sets the CSI-RS PDU BWP parameters and returns a reference to the builder.
  /// \note These parameters are specified in SCF-222 v4.0 section 3.4.2.3 in table CSI-RS PDU.
  dl_csi_rs_pdu_builder& set_bwp_parameters(subcarrier_spacing scs, cyclic_prefix cp)
  {
    pdu.scs = scs;
    pdu.cp  = cp;

    return *this;
  }

  /// Sets the vendor specific CSI-RS PDU BWP parameters and returns a reference to the builder.
  /// \note These parameters are vendor specific.
  dl_csi_rs_pdu_builder& set_vendor_specific_bwp_parameters(unsigned bwp_size, unsigned bwp_start)
  {
    pdu.bwp_size  = bwp_size;
    pdu.bwp_start = bwp_start;

    return *this;
  }

  /// Sets the CSI-RS PDU tx power info parameters and returns a reference to the builder.
  /// \note These parameters are specified in SCF-222 v4.0 section 3.4.2.3 in table CSI-RS PDU.
  dl_csi_rs_pdu_builder& set_tx_power_info_parameters(int                     power_control_offset,
                                                      power_control_offset_ss power_control_offset_ss)
  {
    pdu.power_control_offset_ss_profile_nr = power_control_offset_ss;
    pdu.power_control_offset_profile_nr    = power_control_offset;

    return *this;
  }

  /// Returns a transmission precoding and beamforming PDU builder of this CSI-RS PDU.
  tx_precoding_and_beamforming_pdu_builder get_tx_precoding_and_beamforming_pdu_builder()
  {
    tx_precoding_and_beamforming_pdu_builder builder(pdu.precoding_and_beamforming);

    return builder;
  }
};

/// PRS PDU builder that helps to fill in the parameters specified in SCF-222 v8.0 section 3.4.2.4a.
class dl_prs_pdu_builder
{
  dl_prs_pdu& pdu;

public:
  explicit dl_prs_pdu_builder(dl_prs_pdu& pdu_) : pdu(pdu_) {}

  /// Sets the PRS PDU basic parameters and returns a reference to the builder.
  dl_prs_pdu_builder& set_basic_parameters(subcarrier_spacing scs, cyclic_prefix cp)
  {
    pdu.scs = scs;
    pdu.cp  = cp;

    return *this;
  }

  /// Sets the PRS PDU N_ID parameter and returns a reference to the builder.
  dl_prs_pdu_builder& set_n_id(unsigned n_id)
  {
    pdu.nid_prs = n_id;

    return *this;
  }

  /// Sets the PRS PDU symbol parameters and returns a reference to the builder.
  dl_prs_pdu_builder& set_symbol_parameters(prs_num_symbols nof_symbols, unsigned first_symbol)
  {
    pdu.num_symbols  = nof_symbols;
    pdu.first_symbol = first_symbol;

    return *this;
  }

  /// Sets the PRS PDU RB parameters and returns a reference to the builder.
  dl_prs_pdu_builder& set_rb_parameters(unsigned nof_rb, unsigned start_rb)
  {
    pdu.num_rbs  = nof_rb;
    pdu.start_rb = start_rb;

    return *this;
  }

  /// Sets the PRS PDU power offset parameter and returns a reference to the builder.
  dl_prs_pdu_builder& set_power_offset(std::optional<float> power_offset)
  {
    pdu.prs_power_offset = power_offset;

    return *this;
  }

  /// Sets the PRS PDU transmission comb parameters and returns a reference to the builder.
  dl_prs_pdu_builder& set_comb_parameters(prs_comb_size comb_size, unsigned comb_offset)
  {
    pdu.comb_size   = comb_size;
    pdu.comb_offset = comb_offset;

    return *this;
  }
};

/// DL_TTI.request message builder that helps to fill in the parameters specified in SCF-222 v4.0 section 3.4.2.
class dl_tti_request_message_builder
{
public:
  /// Constructs a builder that will help to fill the given DL TTI request message.
  explicit dl_tti_request_message_builder(dl_tti_request_message& msg_) : msg(msg_)
  {
    msg.is_last_dl_message_in_slot = false;
  }

  /// Sets the DL_TTI.request basic parameters and returns a reference to the builder.
  /// \note nPDUs and nPDUsOfEachType properties are filled by the add_*_pdu() functions.
  /// \note These parameters are specified in SCF-222 v4.0 section 3.4.2 in table DL_TTI.request message body.
  dl_tti_request_message_builder& set_basic_parameters(uint16_t sfn, uint16_t slot, uint16_t n_groups)
  {
    msg.sfn        = sfn;
    msg.slot       = slot;
    msg.num_groups = n_groups;

    return *this;
  }

  /// Adds a PDCCH PDU to the message, fills its basic parameters using the given arguments and returns a PDCCH PDU
  /// builder.
  /// \param[in] nof_dci_in_pdu Number of DCIs in the PDCCH PDU.
  dl_pdcch_pdu_builder add_pdcch_pdu(unsigned nof_dci_in_pdu)
  {
    // Add a new pdu.
    dl_tti_request_pdu& pdu = msg.pdus.emplace_back();

    // Fill the PDCCH PDU index value. The index value will be the index of the pdu in the array of PDCCH PDUs.
    dl_pdcch_pdu_maintenance_v3& info          = pdu.pdcch_pdu.maintenance_v3;
    auto&                        num_pdcch_pdu = msg.num_pdus_of_each_type[static_cast<size_t>(dl_pdu_type::PDCCH)];
    info.pdcch_pdu_index                       = num_pdcch_pdu;

    // Increase the number of PDCCH pdus in the request.
    ++num_pdcch_pdu;
    msg.num_pdus_of_each_type[dl_tti_request_message::DL_DCI_INDEX] += nof_dci_in_pdu;

    pdu.pdu_type = dl_pdu_type::PDCCH;

    dl_pdcch_pdu_builder builder(pdu.pdcch_pdu);

    return builder;
  }

  /// Adds a PDSCH PDU to the message, fills its basic parameters using the given arguments and returns a PDSCH PDU
  /// builder.
  dl_pdsch_pdu_builder add_pdsch_pdu(rnti_t rnti)
  {
    dl_pdsch_pdu_builder builder = add_pdsch_pdu();
    builder.set_basic_parameters(rnti);

    return builder;
  }

  /// Adds a PDSCH PDU to the message, fills its basic parameters using the given arguments and returns a PDSCH PDU
  /// builder.
  dl_pdsch_pdu_builder add_pdsch_pdu()
  {
    // Add a new PDU.
    dl_tti_request_pdu& pdu = msg.pdus.emplace_back();

    // Fill the PDSCH PDU index value. The index value will be the index of the PDU in the array of PDSCH PDUs.
    auto& num_pdsch_pdu     = msg.num_pdus_of_each_type[static_cast<size_t>(dl_pdu_type::PDSCH)];
    pdu.pdsch_pdu.pdu_index = num_pdsch_pdu;

    // Increase the number of PDSCH PDU.
    ++num_pdsch_pdu;

    pdu.pdu_type = dl_pdu_type::PDSCH;

    dl_pdsch_pdu_builder builder(pdu.pdsch_pdu);

    return builder;
  }

  /// Adds a CSI-RS PDU to the message and returns a CSI-RS PDU builder.
  dl_csi_rs_pdu_builder add_csi_rs_pdu(uint16_t                         start_rb,
                                       uint16_t                         nof_rbs,
                                       csi_rs_type                      type,
                                       uint8_t                          row,
                                       const bounded_bitset<12, false>& freq_domain,
                                       uint8_t                          symb_l0,
                                       uint8_t                          symb_l1,
                                       csi_rs_cdm_type                  cdm_type,
                                       csi_rs_freq_density_type         freq_density,
                                       uint16_t                         scrambling_id)
  {
    // Add a new PDU.
    dl_tti_request_pdu& pdu = msg.pdus.emplace_back();

    // Fill the CSI PDU index value. The index value will be the index of the PDU in the array of CSI PDUs.
    auto& num_csi_pdu = msg.num_pdus_of_each_type[static_cast<size_t>(dl_pdu_type::CSI_RS)];
    pdu.csi_rs_pdu.csi_rs_maintenance_v3.csi_rs_pdu_index = num_csi_pdu;

    // Increase the number of CSI PDU.
    ++num_csi_pdu;

    pdu.pdu_type = dl_pdu_type::CSI_RS;

    dl_csi_rs_pdu_builder builder(pdu.csi_rs_pdu);

    builder.set_basic_parameters(
        start_rb, nof_rbs, type, row, freq_domain, symb_l0, symb_l1, cdm_type, freq_density, scrambling_id);

    return builder;
  }

  /// Adds a SSB PDU to the message, fills its basic parameters using the given arguments and returns a SSB PDU builder.
  dl_ssb_pdu_builder add_ssb_pdu(pci_t                 phys_cell_id,
                                 beta_pss_profile_type beta_pss_profile_nr,
                                 uint8_t               ssb_block_index,
                                 uint8_t               ssb_subcarrier_offset,
                                 ssb_offset_to_pointA  ssb_offset_pointA)
  {
    dl_ssb_pdu_builder builder = add_ssb_pdu();

    // Fill the PDU basic parameters.
    builder.set_basic_parameters(
        phys_cell_id, beta_pss_profile_nr, ssb_block_index, ssb_subcarrier_offset, ssb_offset_pointA);

    return builder;
  }

  /// Adds a SSB PDU to the message and returns a SSB PDU builder.
  dl_ssb_pdu_builder add_ssb_pdu()
  {
    // Add a new PDU.
    dl_tti_request_pdu& pdu = msg.pdus.emplace_back();

    // Fill the SSB PDU index value. The index value will be the index of the PDU in the array of SSB PDUs.
    dl_ssb_maintenance_v3& info        = pdu.ssb_pdu.ssb_maintenance_v3;
    auto&                  num_ssb_pdu = msg.num_pdus_of_each_type[static_cast<size_t>(dl_pdu_type::SSB)];
    info.ssb_pdu_index                 = num_ssb_pdu;

    // Increase the number of SSB PDUs in the request.
    ++num_ssb_pdu;

    pdu.pdu_type = dl_pdu_type::SSB;

    dl_ssb_pdu_builder builder(pdu.ssb_pdu);

    return builder;
  }

  /// Adds a PRS PDU to the message and returns a PRS PDU builder.
  dl_prs_pdu_builder add_prs_pdu()
  {
    // Add a new PDU.
    dl_tti_request_pdu& pdu = msg.pdus.emplace_back();

    // Fill the PRS PDU index value. The index value will be the index of the PDU in the array of PRS PDUs.
    dl_prs_pdu& info        = pdu.prs_pdu;
    auto&       num_prs_pdu = msg.num_pdus_of_each_type[static_cast<size_t>(dl_pdu_type::PRS)];
    info.pdu_index          = num_prs_pdu;

    // Increase the number of SSB PDUs in the request.
    ++num_prs_pdu;

    pdu.pdu_type = dl_pdu_type::PRS;

    dl_prs_pdu_builder builder(info);

    return builder;
  }

  /// Sets the flag of the last message in slot.
  dl_tti_request_message_builder& set_last_message_in_slot_flag()
  {
    msg.is_last_dl_message_in_slot = true;
    return *this;
  }

  //: TODO: PDU groups array
  //: TODO: top level rate match patterns

private:
  dl_tti_request_message& msg;
};

/// UL_DCI.request message builder that helps to fill in the parameters specified in SCF-222 v4.0 section 3.4.4.
class ul_dci_request_message_builder
{
  ul_dci_request_message& msg;

public:
  explicit ul_dci_request_message_builder(ul_dci_request_message& msg_) : msg(msg_)
  {
    msg.num_pdus_of_each_type.fill(0);
    msg.is_last_message_in_slot = false;
  }

  /// Sets the UL_DCI.request basic parameters and returns a reference to the builder.
  /// \note These parameters are specified in SCF-222 v4.0 section 3.4.4 in table UL_DCI.request message body.
  ul_dci_request_message_builder& set_basic_parameters(uint16_t sfn, uint16_t slot)
  {
    msg.sfn  = sfn;
    msg.slot = slot;

    return *this;
  }

  /// Adds a PDCCH PDU to the UL_DCI.request basic parameters and returns a reference to the PDCCH PDU builder.
  /// \note These parameters are specified in SCF-222 v4.0 section 3.4.4 in table UL_DCI.request message body.
  /// \param[in] nof_dci_in_pdu Number of DCIs in the PDCCH PDU.
  dl_pdcch_pdu_builder add_pdcch_pdu(unsigned nof_dci_in_pdu)
  {
    unsigned pdcch_index = msg.pdus.size();
    auto&    pdu         = msg.pdus.emplace_back();

    // Fill the pdcch pdu index value. The index value will be the index of the pdu in the array of PDCCH pdus.
    pdu.pdu.maintenance_v3.pdcch_pdu_index = pdcch_index;

    // Increase the number of PDCCH pdus in the request.
    ++msg.num_pdus_of_each_type[static_cast<size_t>(dl_pdu_type::PDCCH)];
    msg.num_pdus_of_each_type[ul_dci_request_message::DCI_INDEX] += nof_dci_in_pdu;

    pdu.pdu_type = ul_dci_pdu_type::PDCCH;

    dl_pdcch_pdu_builder builder(pdu.pdu);

    return builder;
  }

  /// Sets the flag of the last message in slot.
  ul_dci_request_message_builder& set_last_message_in_slot_flag()
  {
    msg.is_last_message_in_slot = true;
    return *this;
  }
};

/// Tx_Data.request message builder that helps to fill in the parameters specified in SCF-222 v4.0 section 3.4.6.
class tx_data_request_builder
{
public:
  explicit tx_data_request_builder(tx_data_request_message& msg_) : msg(msg_) {}

  /// Sets the Tx_Data.request basic parameters and returns a reference to the builder.
  /// \note These parameters are specified in SCF-222 v4.0 section 3.4.6 in table Tx_Data.request message body.
  tx_data_request_builder& set_basic_parameters(uint16_t sfn, uint16_t slot)
  {
    msg.sfn  = sfn;
    msg.slot = slot;

    // NOTE: Set to 0 temporarily.
    msg.control_length = 0U;

    return *this;
  }

  /// Adds a new PDU to the Tx_Data.request message and returns a reference to the builder.
  /// \note These parameters are specified in SCF-222 v4.0 section 3.4.6 in table Tx_Data.request message body.
  tx_data_request_builder& add_pdu(uint16_t pdu_index, uint8_t cw_index, const shared_transport_block& payload)
  {
    msg.pdus.emplace_back(pdu_index, cw_index, payload);
    return *this;
  }

private:
  tx_data_request_message& msg;
};

/// CRC.indication message builder that helps to fill in the parameters specified in SCF-222 v4.0 section 3.4.8.
class crc_indication_message_builder
{
  crc_indication_message& msg;

public:
  explicit crc_indication_message_builder(crc_indication_message& msg_) : msg(msg_) {}

  /// Sets the CRC.indication basic parameters and returns a reference to the builder.
  /// \note These parameters are specified in SCF-222 v4.0 section 3.4.8 in table CRC.indication message body.
  crc_indication_message_builder& set_basic_parameters(uint16_t sfn, uint16_t slot)
  {
    msg.sfn  = sfn;
    msg.slot = slot;

    return *this;
  }

  /// Adds a CRC.indication PDU to the message and returns a reference to the builder.
  /// \note These parameters are specified in SCF-222 v4.0 section 3.4.8 in table CRC.indication message body.
  crc_indication_message_builder& add_pdu(uint32_t                handle,
                                          rnti_t                  rnti,
                                          std::optional<uint8_t>  rapid,
                                          uint8_t                 harq_id,
                                          bool                    tb_crc_status_ok,
                                          uint16_t                num_cb,
                                          span<const uint8_t>     cb_crc_status,
                                          std::optional<float>    ul_sinr_dB,
                                          std::optional<unsigned> timing_advance_offset,
                                          std::optional<int>      timing_advance_offset_in_ns,
                                          std::optional<float>    rssi_dB,
                                          std::optional<float>    rsrp,
                                          bool                    rsrp_use_dBm = false)
  {
    auto& pdu = msg.pdus.emplace_back();

    pdu.handle           = handle;
    pdu.rnti             = rnti;
    pdu.rapid            = (rapid) ? rapid.value() : 255U;
    pdu.harq_id          = harq_id;
    pdu.tb_crc_status_ok = tb_crc_status_ok;
    pdu.num_cb           = num_cb;
    pdu.cb_crc_status.assign(cb_crc_status.begin(), cb_crc_status.end());
    pdu.timing_advance_offset =
        (timing_advance_offset) ? timing_advance_offset.value() : std::numeric_limits<uint16_t>::max();
    pdu.timing_advance_offset_ns =
        (timing_advance_offset_in_ns) ? timing_advance_offset_in_ns.value() : std::numeric_limits<int16_t>::min();

    unsigned rssi =
        (rssi_dB) ? static_cast<unsigned>((rssi_dB.value() + 128.F) * 10.F) : std::numeric_limits<uint16_t>::max();

    srsran_assert(rssi <= std::numeric_limits<uint16_t>::max(),
                  "RSSI ({}) exceeds the maximum ({}).",
                  rssi,
                  std::numeric_limits<uint16_t>::max());

    pdu.rssi = static_cast<uint16_t>(rssi);

    unsigned rsrp_value = (rsrp) ? static_cast<unsigned>((rsrp.value() + ((rsrp_use_dBm) ? 140.F : 128.F)) * 10.F)
                                 : std::numeric_limits<uint16_t>::max();

    srsran_assert(rsrp_value <= std::numeric_limits<uint16_t>::max(),
                  "RSRP ({}) exceeds the maximum ({}).",
                  rsrp_value,
                  std::numeric_limits<uint16_t>::max());

    pdu.rsrp = static_cast<uint16_t>(rsrp_value);

    int ul_sinr = (ul_sinr_dB) ? static_cast<int>(ul_sinr_dB.value() * 500.F) : std::numeric_limits<int16_t>::min();

    srsran_assert(ul_sinr <= std::numeric_limits<int16_t>::max(),
                  "UL SINR metric ({}) exceeds the maximum ({}).",
                  ul_sinr,
                  std::numeric_limits<int16_t>::max());

    srsran_assert(ul_sinr >= std::numeric_limits<int16_t>::min(),
                  "UL SINR metric ({}) is under the minimum ({}).",
                  ul_sinr,
                  std::numeric_limits<int16_t>::min());

    pdu.ul_sinr_metric = static_cast<int16_t>(ul_sinr);
    return *this;
  }
};

/// RACH.indication PDU builder that helps to fill in the parameters specified in SCF-222 v4.0 section 3.4.11.
class rach_indication_pdu_builder
{
  rach_indication_pdu& pdu;

public:
  explicit rach_indication_pdu_builder(rach_indication_pdu& pdu_) : pdu(pdu_) {}

  /// Sets the basic parameters of the RACH.indication PDU and returns a reference to the builder.
  /// \note These parameters are specified in SCF-222 v4.0 section 3.4.11 in table RACH.indication message body.
  rach_indication_pdu_builder& set_basic_params(uint16_t             handle,
                                                uint8_t              symbol_index,
                                                uint8_t              slot_index,
                                                uint8_t              ra_index,
                                                std::optional<float> avg_rssi_dB,
                                                std::optional<float> rsrp,
                                                std::optional<float> avg_snr_dB,
                                                bool                 rsrp_use_dBm = false)

  {
    pdu.handle       = handle;
    pdu.symbol_index = symbol_index;
    pdu.slot_index   = slot_index;
    pdu.ra_index     = ra_index;

    pdu.avg_rssi = (avg_rssi_dB) ? static_cast<uint32_t>((avg_rssi_dB.value() + 140.F) * 1000.F)
                                 : std::numeric_limits<uint32_t>::max();

    unsigned avg_snr =
        (avg_snr_dB) ? static_cast<unsigned>((avg_snr_dB.value() + 64.F) * 2) : std::numeric_limits<uint8_t>::max();

    srsran_assert(avg_snr <= std::numeric_limits<uint8_t>::max(),
                  "Average SNR ({}) exceeds the maximum ({}).",
                  avg_snr,
                  std::numeric_limits<uint8_t>::max());
    pdu.avg_snr = static_cast<uint8_t>(avg_snr);

    unsigned rsrp_value = (rsrp) ? static_cast<unsigned>((rsrp.value() + ((rsrp_use_dBm) ? 140.F : 128.F)) * 10.F)
                                 : std::numeric_limits<uint16_t>::max();

    srsran_assert(rsrp_value <= std::numeric_limits<uint16_t>::max(),
                  "RSRP ({}) exceeds the maximum ({}).",
                  rsrp_value,
                  std::numeric_limits<uint16_t>::max());

    pdu.rsrp = static_cast<uint16_t>(rsrp_value);

    return *this;
  }

  /// Adds a preamble to the RACH.indication PDU and returns a reference to the builder.
  /// \note These parameters are specified in SCF-222 v4.0 section 3.4.11 in table RACH.indication message body.
  /// \note Units for timing advace offset parameter are specified in SCF-222 v4.0 section 3.4.11 in table
  /// RACH.indication message body, and this function expect this units.
  rach_indication_pdu_builder& add_preamble(unsigned                preamble_index,
                                            std::optional<unsigned> timing_advance_offset,
                                            std::optional<uint32_t> timing_advance_offset_ns,
                                            std::optional<float>    preamble_power,
                                            std::optional<float>    preamble_snr)

  {
    auto& preamble = pdu.preambles.emplace_back();

    preamble.preamble_index = preamble_index;

    preamble.timing_advance_offset =
        (timing_advance_offset) ? timing_advance_offset.value() : std::numeric_limits<uint16_t>::max();

    preamble.timing_advance_offset_ns =
        (timing_advance_offset_ns) ? timing_advance_offset_ns.value() : std::numeric_limits<uint32_t>::max();

    preamble.preamble_pwr = (preamble_power) ? static_cast<uint32_t>((preamble_power.value() + 140.F) * 1000.F)
                                             : std::numeric_limits<uint32_t>::max();

    unsigned snr = (preamble_snr) ? static_cast<unsigned>((preamble_snr.value() + 64.F) * 2.F)
                                  : std::numeric_limits<uint8_t>::max();

    srsran_assert(snr <= std::numeric_limits<uint8_t>::max(),
                  "Preamble SNR ({}) exceeds the maximum ({}).",
                  snr,
                  std::numeric_limits<uint8_t>::max());

    preamble.preamble_snr = static_cast<uint8_t>(snr);

    return *this;
  }
};

/// \e RACH.indication message builder that helps to fill in the parameters specified in SCF-222 v4.0 Section 3.4.11.
class rach_indication_message_builder
{
  rach_indication_message& msg;

public:
  explicit rach_indication_message_builder(rach_indication_message& msg_) : msg(msg_) {}

  /// Sets the basic parameters of the RACH.indication message and returns a reference to the builder.
  /// \note These parameters are specified in SCF-222 v4.0 section 3.4.11 in table RACH.indication message body.
  rach_indication_message_builder& set_basic_parameters(uint16_t sfn, uint16_t slot)
  {
    msg.sfn  = sfn;
    msg.slot = slot;

    return *this;
  }

  /// Adds a PDU to the RACH.indication message and returns a reference to the builder.
  /// \note These parameters are specified in SCF-222 v4.0 section 3.4.11 in table RACH.indication message body.
  rach_indication_pdu_builder add_pdu(uint16_t             handle,
                                      uint8_t              symbol_index,
                                      uint8_t              slot_index,
                                      uint8_t              ra_index,
                                      std::optional<float> avg_rssi,
                                      std::optional<float> rsrp,
                                      std::optional<float> avg_snr,
                                      bool                 rsrp_use_dBm = false)
  {
    auto& pdu = msg.pdus.emplace_back();

    rach_indication_pdu_builder builder(pdu);

    builder.set_basic_params(handle, symbol_index, slot_index, ra_index, avg_rssi, rsrp, avg_snr, rsrp_use_dBm);

    return builder;
  }
};

/// UCI PUSCH PDU builder that helps fill in the parameters specified in SCF-222 v4.0 section 3.4.9.1.
class uci_pusch_pdu_builder
{
  uci_pusch_pdu& pdu;

public:
  explicit uci_pusch_pdu_builder(uci_pusch_pdu& pdu_) : pdu(pdu_) { pdu.pdu_bitmap = 0; }

  /// \brief Sets the UCI PUSCH PDU basic parameters and returns a reference to the builder.
  /// \note These parameters are specified in SCF-222 v4.0 Section 3.4.9.1 in Table UCI PUSCH PDU.
  uci_pusch_pdu_builder& set_basic_parameters(uint32_t handle, rnti_t rnti)
  {
    pdu.handle = handle;
    pdu.rnti   = to_value(rnti);

    return *this;
  }

  /// \brief Sets the UCI PUSCH PDU metrics parameters and returns a reference to the builder.
  /// \note These parameters are specified in SCF-222 v4.0 Section 3.4.9.1 in Table UCI PUSCH PDU.
  uci_pusch_pdu_builder& set_metrics_parameters(std::optional<float>    ul_sinr_metric,
                                                std::optional<unsigned> timing_advance_offset,
                                                std::optional<int>      timing_advance_offset_ns,
                                                std::optional<float>    rssi,
                                                std::optional<float>    rsrp,
                                                bool                    rsrp_use_dBm = false)
  {
    pdu.timing_advance_offset    = (timing_advance_offset) ? static_cast<uint16_t>(timing_advance_offset.value())
                                                           : std::numeric_limits<uint16_t>::max();
    pdu.timing_advance_offset_ns = (timing_advance_offset_ns) ? static_cast<int16_t>(timing_advance_offset_ns.value())
                                                              : std::numeric_limits<int16_t>::min();

    // SINR.
    int sinr =
        (ul_sinr_metric) ? static_cast<int>(ul_sinr_metric.value() * 500.F) : std::numeric_limits<int16_t>::min();

    srsran_assert(sinr <= std::numeric_limits<int16_t>::max(),
                  "UL SINR metric ({}) exceeds the maximum ({}).",
                  sinr,
                  std::numeric_limits<int16_t>::max());

    srsran_assert(sinr >= std::numeric_limits<int16_t>::min(),
                  "UL SINR metric ({}) is under the minimum ({}).",
                  sinr,
                  std::numeric_limits<int16_t>::min());

    pdu.ul_sinr_metric = static_cast<int16_t>(sinr);

    // RSSI.
    unsigned rssi_value =
        (rssi) ? static_cast<unsigned>((rssi.value() + 128.F) * 10.F) : std::numeric_limits<uint16_t>::max();

    srsran_assert(rssi_value <= std::numeric_limits<uint16_t>::max(),
                  "RSSI metric ({}) exceeds the maximum ({}).",
                  rssi_value,
                  std::numeric_limits<uint16_t>::max());

    pdu.rssi = static_cast<uint16_t>(rssi_value);

    // RSRP.
    unsigned rsrp_value = (rsrp) ? static_cast<unsigned>((rsrp.value() + ((rsrp_use_dBm) ? 140.F : 128.F)) * 10.F)
                                 : std::numeric_limits<uint16_t>::max();

    srsran_assert(rsrp_value <= std::numeric_limits<uint16_t>::max(),
                  "RSRP metric ({}) exceeds the maximum ({}).",
                  rsrp_value,
                  std::numeric_limits<uint16_t>::max());

    pdu.rsrp = static_cast<uint16_t>(rsrp_value);

    return *this;
  }

  /// \brief Sets the HARQ PDU parameters and returns a reference to the builder.
  /// \note These parameters are specified in SCF-222 v4.0 Section 3.4.9.4 in Table HARQ PDU for Format 2, Format 3 or
  /// Format 4 or for PUSCH.
  uci_pusch_pdu_builder& set_harq_parameters(uci_pusch_or_pucch_f2_3_4_detection_status detection,
                                             uint16_t                                   expected_bit_length,
                                             const bounded_bitset<uci_constants::MAX_NOF_HARQ_BITS>& payload)
  {
    pdu.pdu_bitmap.set(uci_pusch_pdu::HARQ_BIT);

    auto& harq               = pdu.harq;
    harq.detection_status    = detection;
    harq.expected_bit_length = expected_bit_length;
    harq.payload             = payload;

    return *this;
  }

  /// \brief Sets the CSI Part 1 PDU parameters and returns a reference to the builder.
  /// \note These parameters are specified in SCF-222 v4.0 Section 3.4.9.4 in Table CSI Part 1 PDU.
  uci_pusch_pdu_builder&
  set_csi_part1_parameters(uci_pusch_or_pucch_f2_3_4_detection_status                            detection,
                           uint16_t                                                              expected_bit_length,
                           const bounded_bitset<uci_constants::MAX_NOF_CSI_PART1_OR_PART2_BITS>& payload)
  {
    pdu.pdu_bitmap.set(uci_pusch_pdu::CSI_PART1_BIT);

    auto& csi               = pdu.csi_part1;
    csi.detection_status    = detection;
    csi.expected_bit_length = expected_bit_length;
    csi.payload             = payload;

    return *this;
  }

  /// \brief Sets the CSI Part 2 PDU parameters and returns a reference to the builder.
  /// \note These parameters are specified in SCF-222 v4.0 Section 3.4.9.4 in Table CSI Part 2 PDU.
  uci_pusch_pdu_builder&
  set_csi_part2_parameters(uci_pusch_or_pucch_f2_3_4_detection_status                            detection,
                           uint16_t                                                              expected_bit_length,
                           const bounded_bitset<uci_constants::MAX_NOF_CSI_PART1_OR_PART2_BITS>& payload)
  {
    pdu.pdu_bitmap.set(uci_pusch_pdu::CSI_PART2_BIT);

    auto& csi               = pdu.csi_part2;
    csi.detection_status    = detection;
    csi.expected_bit_length = expected_bit_length;
    csi.payload             = payload;

    return *this;
  }
};

/// UCI PUCCH PDU Format 0 or Format 1 builder that helps fill in the parameters specified in SCF-222 v4.0
/// Section 3.4.9.2.
class uci_pucch_pdu_format_0_1_builder
{
  uci_pucch_pdu_format_0_1& pdu;

public:
  explicit uci_pucch_pdu_format_0_1_builder(uci_pucch_pdu_format_0_1& pdu_) : pdu(pdu_) { pdu.pdu_bitmap = 0; }

  /// \brief Sets the UCI PUCCH PDU Format 0 and Format 1 basic parameters and returns a reference to the builder.
  /// \note These parameters are specified in SCF-222 v4.0 Section 3.4.9.2 in Table UCI PUCCH Format 0 or Format 1 PDU.
  uci_pucch_pdu_format_0_1_builder& set_basic_parameters(uint32_t handle, rnti_t rnti, pucch_format type)
  {
    pdu.handle = handle;
    pdu.rnti   = to_value(rnti);
    switch (type) {
      case pucch_format::FORMAT_0:
        pdu.pucch_format = uci_pucch_pdu_format_0_1::format_type::format_0;
        break;
      case pucch_format::FORMAT_1:
        pdu.pucch_format = uci_pucch_pdu_format_0_1::format_type::format_1;
        break;
      default:
        srsran_assert(0, "PUCCH format={} is not supported by this PDU", fmt::underlying(type));
        break;
    }

    return *this;
  }

  /// \brief Sets the UCI PUCCH PDU Format 0 and Format 1 metrics parameters and returns a reference to the builder.
  /// \note These parameters are specified in SCF-222 v4.0 Section 3.4.9.2 in Table UCI PUCCH Format 0 or Format 1 PDU.
  uci_pucch_pdu_format_0_1_builder& set_metrics_parameters(std::optional<float>    ul_sinr_metric,
                                                           std::optional<unsigned> timing_advance_offset,
                                                           std::optional<int>      timing_advance_offset_ns,
                                                           std::optional<float>    rssi,
                                                           std::optional<float>    rsrp,
                                                           bool                    rsrp_use_dBm = false)
  {
    pdu.timing_advance_offset    = (timing_advance_offset) ? static_cast<uint16_t>(timing_advance_offset.value())
                                                           : std::numeric_limits<uint16_t>::max();
    pdu.timing_advance_offset_ns = (timing_advance_offset_ns) ? static_cast<int16_t>(timing_advance_offset_ns.value())
                                                              : std::numeric_limits<int16_t>::min();

    // SINR.
    int sinr =
        (ul_sinr_metric) ? static_cast<int>(ul_sinr_metric.value() * 500.F) : std::numeric_limits<int16_t>::min();

    srsran_assert(sinr <= std::numeric_limits<int16_t>::max(),
                  "UL SINR metric ({}) exceeds the maximum ({}).",
                  sinr,
                  std::numeric_limits<int16_t>::max());

    srsran_assert(sinr >= std::numeric_limits<int16_t>::min(),
                  "UL SINR metric ({}) is under the minimum ({}).",
                  sinr,
                  std::numeric_limits<int16_t>::min());

    pdu.ul_sinr_metric = static_cast<int16_t>(sinr);

    // RSSI.
    unsigned rssi_value =
        (rssi) ? static_cast<unsigned>((rssi.value() + 128.F) * 10.F) : std::numeric_limits<uint16_t>::max();

    srsran_assert(rssi_value <= std::numeric_limits<uint16_t>::max(),
                  "RSSI metric ({}) exceeds the maximum ({}).",
                  rssi_value,
                  std::numeric_limits<uint16_t>::max());

    pdu.rssi = static_cast<uint16_t>(rssi_value);

    // RSRP.
    unsigned rsrp_value = (rsrp) ? static_cast<unsigned>((rsrp.value() + ((rsrp_use_dBm) ? 140.F : 128.F)) * 10.F)
                                 : std::numeric_limits<uint16_t>::max();

    srsran_assert(rsrp_value <= std::numeric_limits<uint16_t>::max(),
                  "RSRP metric ({}) exceeds the maximum ({}).",
                  rsrp_value,
                  std::numeric_limits<uint16_t>::max());

    pdu.rsrp = static_cast<uint16_t>(rsrp_value);

    return *this;
  }

  /// \brief Sets the SR PDU parameters and returns a reference to the builder.
  /// \note These parameters are specified in SCF-222 v4.0 Section 3.4.9.4 in Table SR PDU for Format 0 or Format 1 PDU.
  uci_pucch_pdu_format_0_1_builder& set_sr_parameters(bool detected, std::optional<unsigned> confidence_level)
  {
    pdu.pdu_bitmap.set(uci_pucch_pdu_format_0_1::SR_BIT);

    auto& sr_pdu = pdu.sr;

    sr_pdu.sr_indication = detected;

    sr_pdu.sr_confidence_level = (confidence_level) ? confidence_level.value() : std::numeric_limits<uint8_t>::max();

    return *this;
  }

  /// \brief Sets the HARQ PDU parameters and returns a reference to the builder.
  /// \note These parameters are specified in SCF-222 v4.0 Section 3.4.9.2 in Table HARQ PDU for Format 0 or Format 1
  /// PDU.
  uci_pucch_pdu_format_0_1_builder& set_harq_parameters(std::optional<unsigned>                    confidence_level,
                                                        span<const uci_pucch_f0_or_f1_harq_values> value)
  {
    pdu.pdu_bitmap.set(uci_pucch_pdu_format_0_1::HARQ_BIT);

    auto& harq                 = pdu.harq;
    harq.harq_confidence_level = (confidence_level) ? confidence_level.value() : std::numeric_limits<uint8_t>::max();
    harq.harq_values.assign(value.begin(), value.end());

    return *this;
  }
};

/// UCI PUSCH PDU Format 2, Format 3 or Format 4 builder that helps fill in the parameters specified in SCF-222 v4.0
/// Section 3.4.9.3.
class uci_pucch_pdu_format_2_3_4_builder
{
  uci_pucch_pdu_format_2_3_4& pdu;

public:
  explicit uci_pucch_pdu_format_2_3_4_builder(uci_pucch_pdu_format_2_3_4& pdu_) : pdu(pdu_) { pdu.pdu_bitmap = 0; }

  /// \brief Sets the UCI PUCCH Format 2, Format 3 and Format 4 PDU basic parameters and returns a reference to the
  /// builder. \note These parameters are specified in SCF-222 v4.0 Section 3.4.9.3 in Table UCI PUCCH Format 2, Format
  /// 3 or Format 4 PDU.
  uci_pucch_pdu_format_2_3_4_builder& set_basic_parameters(uint32_t handle, rnti_t rnti, pucch_format type)
  {
    pdu.handle = handle;
    pdu.rnti   = to_value(rnti);
    switch (type) {
      case pucch_format::FORMAT_2:
        pdu.pucch_format = uci_pucch_pdu_format_2_3_4::format_type::format_2;
        break;
      case pucch_format::FORMAT_3:
        pdu.pucch_format = uci_pucch_pdu_format_2_3_4::format_type::format_3;
        break;
      case pucch_format::FORMAT_4:
        pdu.pucch_format = uci_pucch_pdu_format_2_3_4::format_type::format_4;
        break;
      default:
        srsran_assert(0, "PUCCH format={} is not supported by this PDU", fmt::underlying(type));
        break;
    }

    return *this;
  }

  /// \brief Sets the UCI PUCCH Format 2, Format 3 and Format 4 PDU metric parameters and returns a reference to the
  /// builder.
  /// \note These parameters are specified in SCF-222 v4.0 Section 3.4.9.3 in Table UCI PUCCH Format 2, Format
  /// 3 or Format 4 PDU.
  uci_pucch_pdu_format_2_3_4_builder& set_metrics_parameters(std::optional<float>    ul_sinr_metric,
                                                             std::optional<unsigned> timing_advance_offset,
                                                             std::optional<int>      timing_advance_offset_ns,
                                                             std::optional<float>    rssi,
                                                             std::optional<float>    rsrp,
                                                             bool                    rsrp_use_dBm = false)
  {
    pdu.timing_advance_offset    = (timing_advance_offset) ? static_cast<uint16_t>(timing_advance_offset.value())
                                                           : std::numeric_limits<uint16_t>::max();
    pdu.timing_advance_offset_ns = (timing_advance_offset_ns) ? static_cast<int16_t>(timing_advance_offset_ns.value())
                                                              : std::numeric_limits<int16_t>::min();

    // SINR.
    int sinr =
        (ul_sinr_metric) ? static_cast<int>(ul_sinr_metric.value() * 500.F) : std::numeric_limits<int16_t>::min();

    srsran_assert(sinr <= std::numeric_limits<int16_t>::max(),
                  "UL SINR metric ({}) exceeds the maximum ({}).",
                  sinr,
                  std::numeric_limits<int16_t>::max());

    srsran_assert(sinr >= std::numeric_limits<int16_t>::min(),
                  "UL SINR metric ({}) is under the minimum ({}).",
                  sinr,
                  std::numeric_limits<int16_t>::min());

    pdu.ul_sinr_metric = static_cast<int16_t>(sinr);

    // RSSI.
    unsigned rssi_value =
        (rssi) ? static_cast<unsigned>((rssi.value() + 128.F) * 10.F) : std::numeric_limits<uint16_t>::max();

    srsran_assert(rssi_value <= std::numeric_limits<uint16_t>::max(),
                  "RSSI metric ({}) exceeds the maximum ({}).",
                  rssi_value,
                  std::numeric_limits<uint16_t>::max());

    pdu.rssi = static_cast<uint16_t>(rssi_value);

    // RSRP.
    unsigned rsrp_value = (rsrp) ? static_cast<unsigned>((rsrp.value() + ((rsrp_use_dBm) ? 140.F : 128.F)) * 10.F)
                                 : std::numeric_limits<uint16_t>::max();

    srsran_assert(rsrp_value <= std::numeric_limits<uint16_t>::max(),
                  "RSRP metric ({}) exceeds the maximum ({}).",
                  rsrp_value,
                  std::numeric_limits<uint16_t>::max());

    pdu.rsrp = static_cast<uint16_t>(rsrp_value);

    return *this;
  }

  /// \brief Sets the SR PDU parameters and returns a reference to the builder.
  /// \note These parameters are specified in SCF-222 v4.0 Section 3.4.9.3 in Table UCI PUCCH Format 2, Format 3 or
  /// Format 4 PDU.
  uci_pucch_pdu_format_2_3_4_builder&
  set_sr_parameters(uint16_t                                                             bit_length,
                    const bounded_bitset<sr_pdu_format_2_3_4::MAX_SR_PAYLOAD_SIZE_BITS>& sr_payload)
  {
    pdu.pdu_bitmap.set(uci_pucch_pdu_format_2_3_4::SR_BIT);

    auto& sr_pdu = pdu.sr;

    sr_pdu.sr_bitlen  = bit_length;
    sr_pdu.sr_payload = sr_payload;

    return *this;
  }

  /// \brief Sets the HARQ PDU parameters and returns a reference to the builder.
  /// \note These parameters are specified in SCF-222 v4.0 Section 3.4.9.3 in Table UCI PUCCH Format 2, Format 3 or
  /// Format 4 PDU.
  uci_pucch_pdu_format_2_3_4_builder&
  set_harq_parameters(uci_pusch_or_pucch_f2_3_4_detection_status              detection,
                      uint16_t                                                expected_bit_length,
                      const bounded_bitset<uci_constants::MAX_NOF_HARQ_BITS>& payload)
  {
    pdu.pdu_bitmap.set(uci_pucch_pdu_format_2_3_4::HARQ_BIT);

    auto& harq               = pdu.harq;
    harq.detection_status    = detection;
    harq.expected_bit_length = expected_bit_length;
    harq.payload             = payload;

    return *this;
  }

  /// \brief Sets the CSI Part 1 PDU parameters and returns a reference to the builder.
  /// \note These parameters are specified in SCF-222 v4.0 Section 3.4.9.4 in Table CSI Part 1 PDU.
  uci_pucch_pdu_format_2_3_4_builder&
  set_csi_part1_parameters(uci_pusch_or_pucch_f2_3_4_detection_status                            detection,
                           uint16_t                                                              expected_bit_length,
                           const bounded_bitset<uci_constants::MAX_NOF_CSI_PART1_OR_PART2_BITS>& payload)
  {
    pdu.pdu_bitmap.set(uci_pucch_pdu_format_2_3_4::CSI_PART1_BIT);

    auto& csi               = pdu.csi_part1;
    csi.detection_status    = detection;
    csi.expected_bit_length = expected_bit_length;
    csi.payload             = payload;

    return *this;
  }

  /// \brief Sets the CSI Part 2 PDU parameters and returns a reference to the builder.
  /// \note These parameters are specified in SCF-222 v4.0 Section 3.4.9.4 in Table CSI Part 2 PDU.
  uci_pucch_pdu_format_2_3_4_builder&
  set_csi_part2_parameters(uci_pusch_or_pucch_f2_3_4_detection_status                            detection,
                           uint16_t                                                              expected_bit_length,
                           const bounded_bitset<uci_constants::MAX_NOF_CSI_PART1_OR_PART2_BITS>& payload)
  {
    pdu.pdu_bitmap.set(uci_pucch_pdu_format_2_3_4::CSI_PART2_BIT);

    auto& csi               = pdu.csi_part2;
    csi.detection_status    = detection;
    csi.expected_bit_length = expected_bit_length;
    csi.payload             = payload;

    return *this;
  }
};

/// SRS indication PDU builder that helps fill in the parameters specified in SCF-222 v4.0 section 3.4.10.
class srs_indication_pdu_builder
{
  srs_indication_pdu& pdu;

public:
  explicit srs_indication_pdu_builder(srs_indication_pdu& pdu_) : pdu(pdu_) {}

  /// \brief Sets the SRS indication PDU basic parameters and returns a reference to the builder.
  /// \note These parameters are specified in SCF-222 v4.0 Section 3.4.10.
  srs_indication_pdu_builder& set_basic_parameters(uint32_t handle, rnti_t rnti)
  {
    pdu.handle = handle;
    pdu.rnti   = rnti;

    return *this;
  }

  /// \brief Sets the SRS indication PDU metrics parameters and returns a reference to the builder.
  /// \note These parameters are specified in SCF-222 v4.0 Section 3.4.10.
  srs_indication_pdu_builder& set_metrics_parameters(std::optional<unsigned> timing_advance_offset,
                                                     std::optional<int>      timing_advance_offset_ns)
  {
    pdu.timing_advance_offset    = (timing_advance_offset) ? static_cast<uint16_t>(timing_advance_offset.value())
                                                           : std::numeric_limits<uint16_t>::max();
    pdu.timing_advance_offset_ns = (timing_advance_offset_ns) ? static_cast<int16_t>(timing_advance_offset_ns.value())
                                                              : std::numeric_limits<int16_t>::min();

    return *this;
  }

  /// \brief Sets the SRS indication PDU normalized channel I/Q matrix and returns a reference to the builder.
  /// \note These parameters are specified in SCF-222 v4.0 Section 3.4.10 Table 3-132.
  srs_indication_pdu_builder& set_codebook_report_matrix(const srs_channel_matrix& matrix)
  {
    pdu.usage       = srs_usage::codebook;
    pdu.report_type = srs_report_type::normalized_channel_iq_matrix;
    pdu.matrix      = matrix;

    return *this;
  }

  /// \brief Sets the SRS indication PDU positioning report and returns a reference to the builder.
  /// \note These parameters are specified in SCF-222 v8.0 Section 3.4.10 Table 3-209.
  srs_indication_pdu_builder& set_positioning_report_parameters(std::optional<phy_time_unit> ul_relative_toa,
                                                                std::optional<uint32_t>      gnb_rx_tx_difference,
                                                                std::optional<uint16_t>      ul_aoa,
                                                                std::optional<float>         rsrp)
  {
    pdu.report_type                      = srs_report_type::positioning;
    pdu.positioning.ul_relative_toa      = ul_relative_toa;
    pdu.positioning.gnb_rx_tx_difference = gnb_rx_tx_difference;
    pdu.positioning.ul_aoa               = ul_aoa;
    pdu.positioning.rsrp                 = rsrp;
    // [Implementation defined] Set this one to local as it is not used.
    pdu.positioning.coordinate_system_aoa = srs_coordinate_system_ul_aoa::local;
    // [Implementation defined] Set the usage to a enum value.
    pdu.usage = srs_usage::codebook;

    return *this;
  }
};

/// SRS.indication message builder that helps to fill in the parameters specified in SCF-222 v4.0 Section 3.4.10.
class srs_indication_message_builder
{
  srs_indication_message& msg;

public:
  explicit srs_indication_message_builder(srs_indication_message& msg_) : msg(msg_) {}

  /// \brief Sets the \e SRS.indication basic parameters and returns a reference to the builder.
  /// \note These parameters are specified in SCF-222 v4.0 Section 3.4.10 in table SRS.indication message body.
  srs_indication_message_builder& set_basic_parameters(uint16_t sfn, uint16_t slot)
  {
    msg.sfn            = sfn;
    msg.slot           = slot;
    msg.control_length = 0;

    return *this;
  }

  /// Adds a SRS PDU to the \e SRS.indication message and returns a SRS PDU builder.
  srs_indication_pdu_builder add_srs_pdu(uint32_t handle, rnti_t rnti)
  {
    auto& pdu = msg.pdus.emplace_back();

    srs_indication_pdu_builder builder(pdu);
    builder.set_basic_parameters(handle, rnti);

    return builder;
  }
};

/// UCI.indication message builder that helps to fill in the parameters specified in SCF-222 v4.0 Section 3.4.9.
class uci_indication_message_builder
{
  uci_indication_message& msg;

public:
  explicit uci_indication_message_builder(uci_indication_message& msg_) : msg(msg_) {}

  /// \brief Sets the \e UCI.indication basic parameters and returns a reference to the builder.
  /// \note These parameters are specified in SCF-222 v4.0 Section 3.4.9 in table UCI.indication message body.
  uci_indication_message_builder& set_basic_parameters(uint16_t sfn, uint16_t slot)
  {
    msg.sfn  = sfn;
    msg.slot = slot;

    return *this;
  }

  /// Adds a PUSCH PDU to the \e UCI.indication message and returns a PUSCH PDU builder.
  uci_pusch_pdu_builder add_pusch_pdu(uint32_t handle, rnti_t rnti)
  {
    uci_pusch_pdu_builder builder = add_pusch_pdu();
    builder.set_basic_parameters(handle, rnti);

    return builder;
  }

  /// Adds a PUSCH PDU to the \e UCI.indication message and returns a PUSCH PDU builder.
  uci_pusch_pdu_builder add_pusch_pdu()
  {
    auto& pdu = msg.pdus.emplace_back();

    pdu.pdu_type = uci_pdu_type::PUSCH;

    uci_pusch_pdu_builder builder(pdu.pusch_pdu);

    return builder;
  }

  /// Adds a PUCCH Format 0 and Format 1 PDU to the \e UCI.indication message and returns a PUCCH Format 0 and Format 1
  /// PDU builder.
  uci_pucch_pdu_format_0_1_builder add_format_0_1_pucch_pdu(uint32_t handle, rnti_t rnti, pucch_format type)
  {
    uci_pucch_pdu_format_0_1_builder builder = add_format_0_1_pucch_pdu();
    builder.set_basic_parameters(handle, rnti, type);

    return builder;
  }

  /// Adds a PUCCH Format 0 and Format 1 PDU to the \e UCI.indication message and returns a PUCCH Format 0 and Format 1
  /// PDU builder.
  uci_pucch_pdu_format_0_1_builder add_format_0_1_pucch_pdu()
  {
    auto& pdu = msg.pdus.emplace_back();

    pdu.pdu_type = uci_pdu_type::PUCCH_format_0_1;

    uci_pucch_pdu_format_0_1_builder builder(pdu.pucch_pdu_f01);

    return builder;
  }

  /// Adds a PUCCH Format 2, Format 3 and Format 4  PDU to the \e UCI.indication message and returns a PUCCH Format 2,
  /// Format 3 and Format 4 PDU builder.
  uci_pucch_pdu_format_2_3_4_builder add_format_2_3_4_pucch_pdu(uint32_t handle, rnti_t rnti, pucch_format type)
  {
    uci_pucch_pdu_format_2_3_4_builder builder = add_format_2_3_4_pucch_pdu();
    builder.set_basic_parameters(handle, rnti, type);

    return builder;
  }

  /// Adds a PUCCH Format 2, Format 3 and Format 4  PDU to the \e UCI.indication message and returns a PUCCH Format 2,
  /// Format 3 and Format 4 PDU builder.
  uci_pucch_pdu_format_2_3_4_builder add_format_2_3_4_pucch_pdu()
  {
    auto& pdu = msg.pdus.emplace_back();

    pdu.pdu_type = uci_pdu_type::PUCCH_format_2_3_4;

    uci_pucch_pdu_format_2_3_4_builder builder(pdu.pucch_pdu_f234);

    return builder;
  }
};

/// Builds and returns a slot.indication message with the given parameters, as per SCF-222 v4.0 section 3.4.1 in table
/// Slot indication message body.
inline slot_indication_message
build_slot_indication_message(unsigned                                           sfn,
                              unsigned                                           slot,
                              std::chrono::time_point<std::chrono::system_clock> time_point)
{
  slot_indication_message msg;
  msg.message_type = message_type_id::slot_indication;

  msg.sfn        = sfn;
  msg.slot       = slot;
  msg.time_point = time_point;

  return msg;
}

/// \brief Builds and returns an ERROR.indication message with the given parameters, as per SCF-222 v4.0 section 3.3.6.1
/// in table ERROR.indication message body
/// \note This builder is used to build any error code id but OUT_OF_SYNC error.
inline error_indication_message
build_error_indication(uint16_t sfn, uint16_t slot, message_type_id msg_id, error_code_id error_id)
{
  error_indication_message msg;

  msg.message_type  = message_type_id::error_indication;
  msg.sfn           = sfn;
  msg.slot          = slot;
  msg.message_id    = msg_id;
  msg.error_code    = error_id;
  msg.expected_sfn  = std::numeric_limits<decltype(error_indication_message::expected_sfn)>::max();
  msg.expected_slot = std::numeric_limits<decltype(error_indication_message::expected_slot)>::max();

  return msg;
}

/// \brief Builds and returns an ERROR.indication message with the given parameters, as per SCF-222 v4.0 section 3.3.6.1
/// in table ERROR.indication message body
/// \note This builder is used to build only an OUT_OF_SYNC error code.
inline error_indication_message build_out_of_sync_error_indication(uint16_t        sfn,
                                                                   uint16_t        slot,
                                                                   message_type_id msg_id,
                                                                   uint16_t        expected_sfn,
                                                                   uint16_t        expected_slot)
{
  error_indication_message msg;

  msg.message_type  = message_type_id::error_indication;
  msg.sfn           = sfn;
  msg.slot          = slot;
  msg.message_id    = msg_id;
  msg.error_code    = error_code_id::out_of_sync;
  msg.expected_sfn  = expected_sfn;
  msg.expected_slot = expected_slot;

  return msg;
}

/// \brief Builds and returns an ERROR.indication message with the given parameters, as per SCF-222 v4.0 section 3.3.6.1
/// in table ERROR.indication message body
/// \note This builder is used to build only an MSG_INVALID_SFN error code.
inline error_indication_message build_invalid_sfn_error_indication(uint16_t        sfn,
                                                                   uint16_t        slot,
                                                                   message_type_id msg_id,
                                                                   uint16_t        expected_sfn,
                                                                   uint16_t        expected_slot)
{
  error_indication_message msg;

  msg.message_type  = message_type_id::error_indication;
  msg.sfn           = sfn;
  msg.slot          = slot;
  msg.message_id    = msg_id;
  msg.error_code    = error_code_id::msg_invalid_sfn;
  msg.expected_sfn  = expected_sfn;
  msg.expected_slot = expected_slot;

  return msg;
}

/// \brief Builds and returns an ERROR.indication message with the given parameters, as per SCF-222 v4.0 section 3.3.6.1
/// in table ERROR.indication message body
/// \note This builder is used to build only a MSG_SLOT_ERR error code.
inline error_indication_message build_msg_slot_error_indication(uint16_t sfn, uint16_t slot, message_type_id msg_id)
{
  error_indication_message msg;

  msg.message_type  = message_type_id::error_indication;
  msg.sfn           = sfn;
  msg.slot          = slot;
  msg.message_id    = msg_id;
  msg.error_code    = error_code_id::msg_slot_err;
  msg.expected_sfn  = std::numeric_limits<decltype(error_indication_message::expected_sfn)>::max();
  msg.expected_slot = std::numeric_limits<decltype(error_indication_message::expected_slot)>::max();

  return msg;
}

/// \brief Builds and returns an ERROR.indication message with the given parameters, as per SCF-222 v4.0 section 3.3.6.1
/// in table ERROR.indication message body
/// \note This builder is used to build only a MSG_TX_ERR error code.
inline error_indication_message build_msg_tx_error_indication(uint16_t sfn, uint16_t slot)
{
  error_indication_message msg;

  msg.message_type  = message_type_id::error_indication;
  msg.sfn           = sfn;
  msg.slot          = slot;
  msg.message_id    = message_type_id::tx_data_request;
  msg.error_code    = error_code_id::msg_tx_err;
  msg.expected_sfn  = std::numeric_limits<decltype(error_indication_message::expected_sfn)>::max();
  msg.expected_slot = std::numeric_limits<decltype(error_indication_message::expected_slot)>::max();

  return msg;
}

/// \brief Builds and returns an ERROR.indication message with the given parameters, as per SCF-222 v4.0 section 3.3.6.1
/// in table ERROR.indication message body
/// \note This builder is used to build only a MSG_UL_DCI_ERR error code.
inline error_indication_message build_msg_ul_dci_error_indication(uint16_t sfn, uint16_t slot)
{
  error_indication_message msg;

  msg.message_type  = message_type_id::error_indication;
  msg.sfn           = sfn;
  msg.slot          = slot;
  msg.message_id    = message_type_id::ul_dci_request;
  msg.error_code    = error_code_id::msg_ul_dci_err;
  msg.expected_sfn  = std::numeric_limits<decltype(error_indication_message::expected_sfn)>::max();
  msg.expected_slot = std::numeric_limits<decltype(error_indication_message::expected_slot)>::max();

  return msg;
}

/// Rx_Data.indication message builder that helps to fill in the parameters specified in SCF-222 v4.0 section 3.4.7.
class rx_data_indication_message_builder
{
  rx_data_indication_message& msg;

public:
  explicit rx_data_indication_message_builder(rx_data_indication_message& msg_) : msg(msg_) {}

  /// Sets the Rx_Data.indication basic parameters and returns a reference to the builder.
  /// \note These parameters are specified in SCF-222 v4.0 section 3.4.7 in table Rx_Data.indication message body.
  rx_data_indication_message_builder& set_basic_parameters(uint16_t sfn, uint16_t slot, uint16_t control_length)
  {
    msg.sfn            = sfn;
    msg.slot           = slot;
    msg.control_length = control_length;

    return *this;
  }

  /// Adds a PDU to the message and returns a reference to the builder.
  /// \note These parameters are specified in SCF-222 v4.0 section 3.4.7 in table Rx_Data.indication message body.
  rx_data_indication_message_builder&
  add_custom_pdu(uint32_t handle, rnti_t rnti, std::optional<unsigned> rapid, uint8_t harq_id, span<const uint8_t> data)
  {
    auto& pdu = msg.pdus.emplace_back();

    pdu.handle  = handle;
    pdu.rnti    = rnti;
    pdu.rapid   = (rapid) ? static_cast<uint8_t>(rapid.value()) : 255U;
    pdu.harq_id = harq_id;

    // Mark the PDU as custom. This part of the message is not compliant with FAPI.
    pdu.pdu_tag    = rx_data_indication_pdu::pdu_tag_type::custom;
    pdu.pdu_length = data.size();
    pdu.data       = data.data();

    return *this;
  }
};

/// PRACH PDU builder that helps to fill in the parameters specified in SCF-222 v4.0 section 3.4.3.1.
class ul_prach_pdu_builder
{
  ul_prach_pdu& pdu;

public:
  explicit ul_prach_pdu_builder(ul_prach_pdu& pdu_) : pdu(pdu_) {}

  /// Sets the PRACH PDU basic parameters and returns a reference to the builder.
  /// \note These parameters are specified in SCF-222 v4.0 section 3.4.3.1 in table PRACH PDU.
  ul_prach_pdu_builder& set_basic_parameters(pci_t             pci,
                                             uint8_t           num_occasions,
                                             prach_format_type format_type,
                                             uint8_t           index_fd_ra,
                                             uint8_t           prach_start_symbol,
                                             uint16_t          num_cs)
  {
    pdu.phys_cell_id                = pci;
    pdu.num_prach_ocas              = num_occasions;
    pdu.prach_format                = format_type;
    pdu.index_fd_ra                 = index_fd_ra;
    pdu.prach_start_symbol          = prach_start_symbol;
    pdu.num_cs                      = num_cs;
    pdu.is_msg_a_prach              = 0;
    pdu.has_msg_a_pusch_beamforming = false;

    return *this;
  }

  /// Sets the PRACH PDU maintenance v3 basic parameters and returns a reference to the builder.
  /// \note These parameters are specified in SCF-222 v4.0 section 3.4.3.1 in table PRACH maintenance FAPIv3.
  ul_prach_pdu_builder& set_maintenance_v3_basic_parameters(uint32_t                handle,
                                                            prach_config_scope_type prach_config_scope,
                                                            uint16_t                prach_res_config_index,
                                                            uint8_t                 num_fd_ra,
                                                            std::optional<uint8_t>  start_preamble_index,
                                                            uint8_t                 num_preambles_indices)
  {
    auto& v3                  = pdu.maintenance_v3;
    v3.handle                 = handle;
    v3.prach_config_scope     = prach_config_scope;
    v3.prach_res_config_index = prach_res_config_index;
    v3.num_fd_ra              = num_fd_ra;
    v3.start_preamble_index =
        (start_preamble_index) ? start_preamble_index.value() : std::numeric_limits<uint8_t>::max();
    v3.num_preamble_indices = num_preambles_indices;

    return *this;
  }

  //: TODO: beamforming
  //: TODO: uplink spatial assignment
};

/// PUCCH PDU builder that helps to fill in the parameters specified in SCF-222 v4.0 section 3.4.3.3.
class ul_pucch_pdu_builder
{
  ul_pucch_pdu& pdu;

public:
  explicit ul_pucch_pdu_builder(ul_pucch_pdu& pdu_) : pdu(pdu_) {}

  /// Sets the PUCCH PDU basic parameters and returns a reference to the builder.
  /// \note These parameters are specified in SCF-222 v4.0 section 3.4.3.3 in table PUCCH PDU.
  ul_pucch_pdu_builder& set_basic_parameters(rnti_t rnti, uint32_t handle)
  {
    pdu.rnti   = rnti;
    pdu.handle = handle;

    return *this;
  }

  /// Sets the PUCCH PDU common parameters and returns a reference to the builder.
  /// \note These parameters are specified in SCF-222 v4.0 section 3.4.3.3 in table PUCCH PDU.
  ul_pucch_pdu_builder&
  set_common_parameters(pucch_format format_type, pucch_repetition_tx_slot multi_slot_tx_type, bool pi2_bpsk)
  {
    pdu.format_type             = format_type;
    pdu.multi_slot_tx_indicator = multi_slot_tx_type;
    pdu.pi2_bpsk                = pi2_bpsk;

    return *this;
  }

  /// Sets the PUCCH PDU BWP parameters and returns a reference to the builder.
  /// \note These parameters are specified in SCF-222 v4.0 section 3.4.3.3 in table PUCCH PDU.
  ul_pucch_pdu_builder&
  set_bwp_parameters(uint16_t bwp_size, uint16_t bwp_start, subcarrier_spacing scs, cyclic_prefix cp)
  {
    pdu.bwp_size  = bwp_size;
    pdu.bwp_start = bwp_start;
    pdu.scs       = scs;
    pdu.cp        = cp;

    return *this;
  }

  /// Sets the PUCCH PDU allocation in frequency parameters and returns a reference to the builder.
  /// \note These parameters are specified in SCF-222 v4.0 section 3.4.3.3 in table PUCCH PDU.
  ul_pucch_pdu_builder& set_allocation_in_frequency_parameters(uint16_t prb_start, uint16_t prb_size)
  {
    pdu.prb_start = prb_start;
    pdu.prb_size  = prb_size;

    return *this;
  }

  /// Sets the PUCCH PDU allocation in time parameters and returns a reference to the builder.
  /// \note These parameters are specified in SCF-222 v4.0 section 3.4.3.3 in table PUCCH PDU.
  ul_pucch_pdu_builder& set_allocation_in_time_parameters(uint8_t start_symbol_index, uint8_t nof_symbols)
  {
    pdu.start_symbol_index = start_symbol_index;
    pdu.nr_of_symbols      = nof_symbols;

    return *this;
  }

  /// Sets the PUCCH PDU hopping information parameters and returns a reference to the builder.
  /// \note These parameters are specified in SCF-222 v4.0 section 3.4.3.3 in table PUCCH PDU.
  ul_pucch_pdu_builder& set_hopping_information_parameters(bool                intra_slot_frequency_hopping,
                                                           uint16_t            second_hop_prb,
                                                           pucch_group_hopping pucch_group_hopping,
                                                           uint16_t            nid_pucch_hopping,
                                                           uint16_t            initial_cyclic_shift)
  {
    pdu.intra_slot_frequency_hopping = intra_slot_frequency_hopping;
    pdu.second_hop_prb               = second_hop_prb;
    pdu.pucch_grp_hopping            = pucch_group_hopping;
    pdu.nid_pucch_hopping            = nid_pucch_hopping;
    pdu.initial_cyclic_shift         = initial_cyclic_shift;

    return *this;
  }

  /// Sets the PUCCH PDU hopping information parameters for PUCCH format 2 and returns a reference to the builder.
  /// \note These parameters are specified in SCF-222 v4.0 section 3.4.3.3 in table PUCCH PDU.
  ul_pucch_pdu_builder& set_hopping_information_format2_parameters(bool     intra_slot_frequency_hopping,
                                                                   uint16_t second_hop_prb)
  {
    pdu.intra_slot_frequency_hopping = intra_slot_frequency_hopping;
    pdu.second_hop_prb               = second_hop_prb;

    return *this;
  }

  /// Sets the PUCCH PDU scrambling parameters and returns a reference to the builder.
  /// \note These parameters are specified in SCF-222 v4.0 section 3.4.3.3 in table PUCCH PDU.
  ul_pucch_pdu_builder& set_scrambling_parameters(uint16_t nid_pucch_scrambling)
  {
    pdu.nid_pucch_scrambling = nid_pucch_scrambling;

    return *this;
  }

  /// Sets the PUCCH PDU scrambling parameters for DM-RS and returns a reference to the builder.
  /// \note These parameters are specified in SCF-222 v4.0 section 3.4.3.3 in table PUCCH PDU.
  ul_pucch_pdu_builder& set_dmrs_scrambling(uint16_t nid0_pucch_dmrs_scrambling)
  {
    pdu.nid0_pucch_dmrs_scrambling = nid0_pucch_dmrs_scrambling;

    return *this;
  }

  /// Sets the PUCCH PDU format 1 specific parameters and returns a reference to the builder.
  /// \note These parameters are specified in SCF-222 v4.0 section 3.4.3.3 in table PUCCH PDU.
  ul_pucch_pdu_builder& set_format1_parameters(uint8_t time_domain_occ_idx)
  {
    pdu.time_domain_occ_index = time_domain_occ_idx;

    return *this;
  }

  /// Sets the PUCCH PDU format 4 specific parameters and returns a reference to the builder.
  /// \note These parameters are specified in SCF-222 v4.0 section 3.4.3.3 in table PUCCH PDU.
  ul_pucch_pdu_builder& set_format4_parameters(uint8_t pre_dft_occ_idx, uint8_t pre_dft_occ_len)
  {
    pdu.pre_dft_occ_idx = pre_dft_occ_idx;
    pdu.pre_dft_occ_len = pre_dft_occ_len;

    return *this;
  }

  /// Sets the PUCCH PDU DMRS parameters and returns a reference to the builder.
  /// \note These parameters are specified in SCF-222 v4.0 section 3.4.3.3 in table PUCCH PDU.
  ul_pucch_pdu_builder&
  set_dmrs_parameters(bool add_dmrs_flag, uint16_t nid0_pucch_dmrs_scrambling, uint8_t m0_pucch_dmrs_cyclic_shift)
  {
    pdu.add_dmrs_flag              = add_dmrs_flag;
    pdu.nid0_pucch_dmrs_scrambling = nid0_pucch_dmrs_scrambling;
    pdu.m0_pucch_dmrs_cyclic_shift = m0_pucch_dmrs_cyclic_shift;

    return *this;
  }

  /// Sets the PUCCH PDU bit length for SR, HARQ and CSI Part 1 parameters and returns a reference to the builder.
  /// \note These parameters are specified in SCF-222 v4.0 section 3.4.3.3 in table PUCCH PDU.
  ul_pucch_pdu_builder&
  set_bit_length_parameters(uint8_t sr_bit_len, uint16_t bit_len_harq, uint16_t csi_part1_bit_length)
  {
    pdu.sr_bit_len           = sr_bit_len;
    pdu.bit_len_harq         = bit_len_harq;
    pdu.csi_part1_bit_length = csi_part1_bit_length;

    return *this;
  }

  /// Sets the PUCCH PDU maintenance v3 basic parameters and returns a reference to the builder.
  /// \note These parameters are specified in SCF-222 v4.0 section 3.4.3.3 in table PUCCH basic extension for FAPIv3.
  ul_pucch_pdu_builder& set_maintenance_v3_basic_parameters(std::optional<unsigned> max_code_rate,
                                                            std::optional<unsigned> ul_bwp_id)
  {
    pdu.pucch_maintenance_v3.max_code_rate =
        (max_code_rate) ? static_cast<unsigned>(max_code_rate.value()) : std::numeric_limits<uint8_t>::max();
    pdu.pucch_maintenance_v3.ul_bwp_id =
        (ul_bwp_id) ? static_cast<unsigned>(ul_bwp_id.value()) : std::numeric_limits<uint8_t>::max();

    return *this;
  }

  /// Adds a UCI part1 to part2 correspondence v3 to the PUCCH PDU and returns a reference to the builder.
  /// \note These parameters are specified in SCF-222 v4.0 section 3.4.3.2 in table UCI information for determining UCI
  /// Part1 to PArt2 correspondence, added in FAPIv3.
  ul_pucch_pdu_builder&
  add_uci_part1_part2_corresnpondence_v3(uint16_t                                             priority,
                                         span<const uint16_t>                                 param_offset,
                                         span<const uint8_t>                                  param_sizes,
                                         uint16_t                                             part2_size_map_index,
                                         uci_part1_to_part2_correspondence_v3::map_scope_type part2_size_map_scope)
  {
    srsran_assert(param_offset.size() == param_sizes.size(),
                  "Mismatching span sizes for param offset ({}) and param sizes ({})",
                  param_offset.size(),
                  param_sizes.size());

    auto& correspondence                = pdu.uci_correspondence.part2.emplace_back();
    correspondence.priority             = priority;
    correspondence.part2_size_map_index = part2_size_map_index;
    correspondence.part2_size_map_scope = part2_size_map_scope;

    correspondence.param_offsets.assign(param_offset.begin(), param_offset.end());
    correspondence.param_sizes.assign(param_sizes.begin(), param_sizes.end());

    return *this;
  }

  /// Sets the PUCCH PDU UCI part1 to part2 correspondence v3 basic parameters and returns a reference to the builder.
  /// \note These parameters are specified in SCF-222 v4.0 section 3.4.3.3 in table PUCCH basic extension for FAPIv3.
  ul_pucch_pdu_builder& set_uci_part1_part2_corresnpondence_v3_basic_parameters(
      span<const uci_part1_to_part2_correspondence_v3::part2_info> correspondence)
  {
    pdu.uci_correspondence.part2.assign(correspondence.begin(), correspondence.end());

    return *this;
  }
};

/// PUSCH PDU builder that helps to fill in the parameters specified in SCF-222 v4.0 section 3.4.3.2.
class ul_pusch_pdu_builder
{
  ul_pusch_pdu& pdu;

public:
  explicit ul_pusch_pdu_builder(ul_pusch_pdu& pdu_) : pdu(pdu_)
  {
    pdu.ul_dmrs_symb_pos = 0U;
    pdu.rb_bitmap.fill(0);

    ul_pusch_uci& uci        = pdu.pusch_uci;
    uci.harq_ack_bit_length  = 0U;
    uci.beta_offset_harq_ack = 0U;
    uci.csi_part1_bit_length = 0U;
    uci.beta_offset_csi1     = 0U;
    uci.flags_csi_part2      = 0U;
    uci.beta_offset_csi2     = 0U;
  }

  /// Sets the PUSCH PDU basic parameters and returns a reference to the builder.
  /// \note These parameters are specified in SCF-222 v4.0 section 3.4.3.2 in table PUSCH PDU.
  ul_pusch_pdu_builder& set_basic_parameters(rnti_t rnti, uint32_t handle)
  {
    pdu.rnti   = rnti;
    pdu.handle = handle;

    return *this;
  }

  /// Sets the PUSCH PDU BWP parameters and returns a reference to the builder.
  /// \note These parameters are specified in SCF-222 v4.0 section 3.4.3.2 in table PUSCH PDU.
  ul_pusch_pdu_builder&
  set_bwp_parameters(uint16_t bwp_size, uint16_t bwp_start, subcarrier_spacing scs, cyclic_prefix cp)
  {
    pdu.bwp_size  = bwp_size;
    pdu.bwp_start = bwp_start;
    pdu.scs       = scs;
    pdu.cp        = cp;

    return *this;
  }

  /// Sets the PUSCH PDU information parameters and returns a reference to the builder.
  /// \note These parameters are specified in SCF-222 v4.0 section 3.4.3.2 in table PUSCH PDU.
  ul_pusch_pdu_builder& set_information_parameters(float             target_code_rate,
                                                   modulation_scheme qam_mod_order,
                                                   uint8_t           mcs_index,
                                                   pusch_mcs_table   mcs_table,
                                                   bool              transform_precoding,
                                                   uint16_t          nid_pusch,
                                                   uint8_t           num_layers)
  {
    pdu.target_code_rate    = target_code_rate * 10.F;
    pdu.qam_mod_order       = qam_mod_order;
    pdu.mcs_index           = mcs_index;
    pdu.mcs_table           = mcs_table;
    pdu.transform_precoding = transform_precoding;
    pdu.nid_pusch           = nid_pusch;
    pdu.num_layers          = num_layers;

    return *this;
  }

  /// Sets the PUSCH PDU DMRS parameters and returns a reference to the builder.
  /// \note These parameters are specified in SCF-222 v4.0 section 3.4.3.2 in table PUSCH PDU.
  ul_pusch_pdu_builder& set_dmrs_parameters(uint16_t           ul_dmrs_symb_pos,
                                            dmrs_config_type   dmrs_type,
                                            uint16_t           pusch_dmrs_scrambling_id,
                                            uint16_t           pusch_dmrs_scrambling_id_complement,
                                            low_papr_dmrs_type low_papr_dmrs,
                                            uint16_t           pusch_dmrs_identity,
                                            uint8_t            nscid,
                                            uint8_t            num_dmrs_cdm_grps_no_data,
                                            uint16_t           dmrs_ports)
  {
    pdu.ul_dmrs_symb_pos = ul_dmrs_symb_pos;
    pdu.dmrs_type        = (dmrs_type == dmrs_config_type::type1) ? dmrs_cfg_type::type_1 : dmrs_cfg_type::type_2;
    pdu.pusch_dmrs_scrambling_id            = pusch_dmrs_scrambling_id;
    pdu.pusch_dmrs_scrambling_id_complement = pusch_dmrs_scrambling_id_complement;
    pdu.low_papr_dmrs                       = low_papr_dmrs;
    pdu.pusch_dmrs_identity                 = pusch_dmrs_identity;
    pdu.nscid                               = nscid;
    pdu.num_dmrs_cdm_grps_no_data           = num_dmrs_cdm_grps_no_data;
    pdu.dmrs_ports                          = dmrs_ports;

    return *this;
  }

  /// Sets the PUSCH PDU allocation in frequency domain type 0 parameters and returns a reference to the builder.
  /// \note These parameters are specified in SCF-222 v4.0 section 3.4.3.2 in table PUSCH PDU.
  ul_pusch_pdu_builder& set_allocation_in_frequency_type_0_parameters(span<const uint8_t> rb_bitmap,
                                                                      uint16_t            tx_direct_current_location,
                                                                      bool                uplink_frequency_shift_7p5hHz)
  {
    pdu.resource_alloc = resource_allocation_type::type_0;
    srsran_assert(pdu.rb_bitmap.size() == rb_bitmap.size(), "RB bitmap size doesn't match");
    std::copy(rb_bitmap.begin(), rb_bitmap.end(), pdu.rb_bitmap.begin());
    pdu.vrb_to_prb_mapping            = vrb_to_prb_mapping_type::non_interleaved;
    pdu.tx_direct_current_location    = tx_direct_current_location;
    pdu.uplink_frequency_shift_7p5kHz = uplink_frequency_shift_7p5hHz;

    // Set the parameters for type 1 to a value.
    pdu.rb_start                     = 0;
    pdu.rb_size                      = 0;
    pdu.intra_slot_frequency_hopping = false;

    return *this;
  }

  /// Sets the PUSCH PDU allocation in frequency domain type 1 parameters and returns a reference to the builder.
  /// \note These parameters are specified in SCF-222 v4.0 section 3.4.3.2 in table PUSCH PDU.
  ul_pusch_pdu_builder& set_allocation_in_frequency_type_1_parameters(uint16_t rb_start,
                                                                      uint16_t rb_size,
                                                                      bool     intra_slot_frequency_hopping,
                                                                      uint16_t tx_direct_current_location,
                                                                      bool     uplink_frequency_shift_7p5hHz)
  {
    pdu.resource_alloc                = resource_allocation_type::type_1;
    pdu.rb_start                      = rb_start;
    pdu.rb_size                       = rb_size;
    pdu.intra_slot_frequency_hopping  = intra_slot_frequency_hopping;
    pdu.vrb_to_prb_mapping            = vrb_to_prb_mapping_type::non_interleaved;
    pdu.tx_direct_current_location    = tx_direct_current_location;
    pdu.uplink_frequency_shift_7p5kHz = uplink_frequency_shift_7p5hHz;

    return *this;
  }

  /// Sets the PUSCH PDU allocation in time domain type 0 parameters and returns a reference to the builder.
  /// \note These parameters are specified in SCF-222 v4.0 section 3.4.3.2 in table PUSCH PDU.
  ul_pusch_pdu_builder& set_allocation_in_time_parameters(uint8_t start_symbol_index, uint8_t num_symbols)
  {
    pdu.start_symbol_index = start_symbol_index;
    pdu.nr_of_symbols      = num_symbols;

    return *this;
  }

  /// Sets the PUSCH PDU maintenance v3 BWP parameters and returns a reference to the builder.
  /// \note These parameters are specified in SCF-222 v4.0 section 3.4.3.2 in table PUSCH maintenance FAPIv3.
  ul_pusch_pdu_builder& set_maintenance_v3_bwp_parameters(uint8_t  pusch_trans_type,
                                                          uint16_t delta_bwp0_start_from_active_bwp,
                                                          uint16_t initial_ul_bwp_size)
  {
    auto& v3                            = pdu.pusch_maintenance_v3;
    v3.pusch_trans_type                 = pusch_trans_type;
    v3.delta_bwp0_start_from_active_bwp = delta_bwp0_start_from_active_bwp;
    v3.initial_ul_bwp_size              = initial_ul_bwp_size;

    return *this;
  }

  /// Sets the PUSCH PDU maintenance v3 DMRS parameters and returns a reference to the builder.
  /// \note These parameters are specified in SCF-222 v4.0 section 3.4.3.2 in table PUSCH maintenance FAPIv3.
  ul_pusch_pdu_builder& set_maintenance_v3_dmrs_parameters(uint8_t group_or_sequence_hopping)
  {
    pdu.pusch_maintenance_v3.group_or_sequence_hopping = group_or_sequence_hopping;

    return *this;
  }

  /// Sets the PUSCH PDU maintenance v3 frequency domain allocation parameters and returns a reference to the builder.
  /// \note These parameters are specified in SCF-222 v4.0 section 3.4.3.2 in table PUSCH maintenance FAPIv3.
  ul_pusch_pdu_builder& set_maintenance_v3_frequency_allocation_parameters(uint16_t             pusch_second_hop_prb,
                                                                           ldpc_base_graph_type ldpc_graph,
                                                                           units::bytes         tb_size_lbrm_bytes)
  {
    auto& v3                = pdu.pusch_maintenance_v3;
    v3.pusch_second_hop_prb = pusch_second_hop_prb;
    v3.ldpc_base_graph      = ldpc_graph;
    v3.tb_size_lbrm_bytes   = tb_size_lbrm_bytes;

    return *this;
  }

  /// Sets the PUSCH PDU parameters v4 basic parameters and returns a reference to the builder.
  /// \note These parameters are specified in SCF-222 v4.0 section 3.4.3.2 in table PUSCH parameters v4.
  ul_pusch_pdu_builder& set_parameters_v4_basic_parameters(bool cb_crc_status_request)
  {
    pdu.pusch_params_v4.cb_crc_status_request = cb_crc_status_request;

    return *this;
  }

  /// Adds a UCI part1 to part2 correspondence v3 to the PUSCH PDU and returns a reference to the builder.
  /// \note These parameters are specified in SCF-222 v4.0 section 3.4.3.2 in table UCI information for determining UCI
  /// Part1 to PArt2 correspondence, added in FAPIv3.
  ul_pusch_pdu_builder&
  add_uci_part1_part2_corresnpondence_v3(uint16_t                                             priority,
                                         span<const uint16_t>                                 param_offset,
                                         span<const uint8_t>                                  param_sizes,
                                         uint16_t                                             part2_size_map_index,
                                         uci_part1_to_part2_correspondence_v3::map_scope_type part2_size_map_scope)
  {
    srsran_assert(param_offset.size() == param_sizes.size(),
                  "Mismatching span sizes for param offset ({}) and param sizes ({})",
                  param_offset.size(),
                  param_sizes.size());

    auto& correspondence                = pdu.uci_correspondence.part2.emplace_back();
    correspondence.priority             = priority;
    correspondence.part2_size_map_index = part2_size_map_index;
    correspondence.part2_size_map_scope = part2_size_map_scope;

    correspondence.param_offsets.assign(param_offset.begin(), param_offset.end());
    correspondence.param_sizes.assign(param_sizes.begin(), param_sizes.end());

    return *this;
  }

  // :TODO: UL MIMO parameters in FAPIv4.

  /// Adds optional PUSCH data information to the PUSCH PDU and returns a reference to the builder.
  /// \note These parameters are specified in SCF-222 v4.0 section 3.4.3.2 in table optional PUSCH data information.
  // :TODO: analyze in the future this function. I'd suggest to change the last 2 arguments with a bounded_bitset or a
  // vector of bool.
  ul_pusch_pdu_builder& add_optional_pusch_data(uint8_t             rv_index,
                                                uint8_t             harq_process_id,
                                                bool                new_data,
                                                units::bytes        tb_size,
                                                uint16_t            num_cb,
                                                span<const uint8_t> cb_present_and_position)
  {
    pdu.pdu_bitmap.set(ul_pusch_pdu::PUSCH_DATA_BIT);

    auto& data           = pdu.pusch_data;
    data.rv_index        = rv_index;
    data.harq_process_id = harq_process_id;
    data.new_data        = new_data;
    data.tb_size         = tb_size;
    data.num_cb          = num_cb;
    data.cb_present_and_position.assign(cb_present_and_position.begin(), cb_present_and_position.end());

    return *this;
  }

  /// Adds optional PUSCH UCI alpha information to the PUSCH PDU and returns a reference to the builder.
  /// \note These parameters are specified in SCF-222 v4.0 section 3.4.3.2 in table optional PUSCH UCI information.
  ul_pusch_pdu_builder& add_optional_pusch_uci_alpha(alpha_scaling_opt alpha_scaling)
  {
    pdu.pdu_bitmap.set(ul_pusch_pdu::PUSCH_UCI_BIT);

    auto& uci         = pdu.pusch_uci;
    uci.alpha_scaling = alpha_scaling;

    return *this;
  }

  /// Adds optional PUSCH UCI HARQ information to the PUSCH PDU and returns a reference to the builder.
  /// \note These parameters are specified in SCF-222 v4.0 section 3.4.3.2 in table optional PUSCH UCI information.
  ul_pusch_pdu_builder& add_optional_pusch_uci_harq(uint16_t harq_ack_bit_len, uint8_t beta_offset_harq_ack)
  {
    pdu.pdu_bitmap.set(ul_pusch_pdu::PUSCH_UCI_BIT);

    auto& uci = pdu.pusch_uci;

    uci.harq_ack_bit_length  = harq_ack_bit_len;
    uci.beta_offset_harq_ack = beta_offset_harq_ack;

    return *this;
  }

  /// Adds optional PUSCH UCI CSI1 information to the PUSCH PDU and returns a reference to the builder.
  /// \note These parameters are specified in SCF-222 v4.0 section 3.4.3.2 in table optional PUSCH UCI information.
  ul_pusch_pdu_builder& add_optional_pusch_uci_csi1(uint16_t csi_part1_bit_len, uint8_t beta_offset_csi_1)
  {
    pdu.pdu_bitmap.set(ul_pusch_pdu::PUSCH_UCI_BIT);

    auto& uci = pdu.pusch_uci;

    uci.csi_part1_bit_length = csi_part1_bit_len;
    uci.beta_offset_csi1     = beta_offset_csi_1;

    return *this;
  }

  /// Adds optional PUSCH UCI CSI2 information to the PUSCH PDU and returns a reference to the builder.
  /// \note These parameters are specified in SCF-222 v4.0 section 3.4.3.2 in table optional PUSCH UCI information.
  ul_pusch_pdu_builder& add_optional_pusch_uci_csi2(uint8_t beta_offset_csi_2)
  {
    auto& uci = pdu.pusch_uci;

    uci.flags_csi_part2  = std::numeric_limits<decltype(fapi::ul_pusch_uci::flags_csi_part2)>::max();
    uci.beta_offset_csi2 = beta_offset_csi_2;

    return *this;
  }

  /// Adds optional PUSCH PTRS information to the PUSCH PDU and returns a reference to the builder.
  /// \note These parameters are specified in SCF-222 v4.0 section 3.4.3.2 in table optional PUSCH PTRS information.
  ul_pusch_pdu_builder& add_optional_pusch_ptrs(span<const ul_pusch_ptrs::ptrs_port_info> port_info,
                                                ptrs_time_density                         ptrs_time_density,
                                                ptrs_frequency_density                    ptrs_freq_density,
                                                ul_ptrs_power_type                        ul_ptrs_power)
  {
    pdu.pdu_bitmap.set(ul_pusch_pdu::PUSCH_PTRS_BIT);

    auto& ptrs = pdu.pusch_ptrs;

    ptrs.port_info.assign(port_info.begin(), port_info.end());
    ptrs.ul_ptrs_power     = ul_ptrs_power;
    ptrs.ptrs_time_density = static_cast<uint8_t>(ptrs_time_density) / 2U;
    ptrs.ptrs_freq_density = static_cast<uint8_t>(ptrs_freq_density) / 4U;

    return *this;
  }

  /// Adds optional PUSCH DFTS OFDM information to the PUSCH PDU and returns a reference to the builder.
  /// \note These parameters are specified in SCF-222 v4.0 section 3.4.3.2 in table optional PUSCH DFTS OFDM
  /// information.
  ul_pusch_pdu_builder& add_optional_dfts_ofdm(uint8_t  low_papr_group_number,
                                               uint16_t low_papr_sequence_number,
                                               uint8_t  ul_ptrs_sample_density,
                                               uint8_t  ul_ptrs_time_density_transform_precoding)
  {
    pdu.pdu_bitmap.set(ul_pusch_pdu::DFTS_OFDM_BIT);

    auto& ofdm = pdu.pusch_ofdm;

    ofdm.low_papr_group_number                    = low_papr_group_number;
    ofdm.low_papr_sequence_number                 = low_papr_sequence_number;
    ofdm.ul_ptrs_sample_density                   = ul_ptrs_sample_density;
    ofdm.ul_ptrs_time_density_transform_precoding = ul_ptrs_time_density_transform_precoding;

    return *this;
  }

  /// Sets the PUSCH context as vendor specific.
  ul_pusch_pdu_builder& set_context_vendor_specific(rnti_t rnti, harq_id_t harq_id)
  {
    pdu.context = pusch_context(rnti, harq_id);
    return *this;
  }
};

/// Uplink SRS PDU builder that helps to fill in the parameters specified in SCF-222 v4.0 section 3.4.3.3.
class ul_srs_pdu_builder
{
  ul_srs_pdu& pdu;

public:
  explicit ul_srs_pdu_builder(ul_srs_pdu& pdu_) : pdu(pdu_) { pdu.srs_params_v4.report_type = 0; }

  /// Sets the SRS PDU basic parameters and returns a reference to the builder.
  /// \note These parameters are specified in SCF-222 v4.0 section 3.4.3.3 in table SRS PDU.
  ul_srs_pdu_builder& set_basic_parameters(rnti_t rnti, uint32_t handle)
  {
    pdu.rnti   = rnti;
    pdu.handle = handle;

    return *this;
  }

  /// Sets the SRS PDU BWP parameters and returns a reference to the builder.
  /// \note These parameters are specified in SCF-222 v4.0 section 3.4.3.3 in table SRS PDU.
  ul_srs_pdu_builder&
  set_bwp_parameters(uint16_t bwp_size, uint16_t bwp_start, subcarrier_spacing scs, cyclic_prefix cp)
  {
    pdu.bwp_size  = bwp_size;
    pdu.bwp_start = bwp_start;
    pdu.scs       = scs;
    pdu.cp        = cp;

    return *this;
  }

  /// Sets the SRS PDU timing parameters and returns a reference to the builder.
  /// \note These parameters are specified in SCF-222 v4.0 section 3.4.3.3 in table SRS PDU.
  ul_srs_pdu_builder& set_timing_params(unsigned time_start_position, srs_periodicity t_srs, unsigned t_offset)
  {
    pdu.time_start_position = time_start_position;
    pdu.t_srs               = t_srs;
    pdu.t_offset            = t_offset;

    return *this;
  }

  /// Sets the SRS PDU comb parameters and returns a reference to the builder.
  /// \note These parameters are specified in SCF-222 v4.0 section 3.4.3.3 in table SRS PDU.
  ul_srs_pdu_builder& set_comb_params(tx_comb_size comb_size, unsigned comb_offset)
  {
    pdu.comb_size   = comb_size;
    pdu.comb_offset = comb_offset;

    return *this;
  }

  /// Sets the SRS report types and returns a reference to the builder.
  /// \note These parameters are specified in SCF-222 v8.0 section 3.4.3.4 in table SRS PDU.
  ul_srs_pdu_builder& set_report_params(bool enable_normalized_iq_matrix_report, bool enable_positioning_report)
  {
    if (enable_normalized_iq_matrix_report) {
      pdu.srs_params_v4.report_type.set(to_value(srs_report_type::normalized_channel_iq_matrix));
    }

    if (enable_positioning_report) {
      pdu.srs_params_v4.report_type.set(to_value(srs_report_type::positioning));
    }

    return *this;
  }

  /// Sets the SRS PDU frequency parameters and returns a reference to the builder.
  /// \note These parameters are specified in SCF-222 v4.0 section 3.4.3.3 in table SRS PDU.
  ul_srs_pdu_builder& set_frequency_params(unsigned                      frequency_position,
                                           unsigned                      frequency_shift,
                                           unsigned                      frequency_hopping,
                                           srs_group_or_sequence_hopping group_or_sequence_hopping)
  {
    pdu.frequency_position        = frequency_position;
    pdu.frequency_shift           = frequency_shift;
    pdu.frequency_hopping         = frequency_hopping;
    pdu.group_or_sequence_hopping = group_or_sequence_hopping;

    return *this;
  }

  /// Sets the SRS PDU parameters and returns a reference to the builder.
  /// \note These parameters are specified in SCF-222 v4.0 section 3.4.3.3 in table SRS PDU.
  ul_srs_pdu_builder& set_srs_params(unsigned          nof_antenna_ports,
                                     unsigned          nof_symbols,
                                     srs_nof_symbols   nof_repetitions,
                                     unsigned          config_index,
                                     unsigned          sequence_id,
                                     unsigned          bandwidth_index,
                                     unsigned          cyclic_shift,
                                     srs_resource_type resource_type)
  {
    pdu.num_ant_ports   = nof_antenna_ports;
    pdu.num_symbols     = nof_symbols;
    pdu.num_repetitions = nof_repetitions;
    pdu.config_index    = config_index;
    pdu.sequence_id     = sequence_id;
    pdu.bandwidth_index = bandwidth_index;
    pdu.cyclic_shift    = cyclic_shift;
    pdu.resource_type   = resource_type;

    return *this;
  }
};

/// UL_TTI.request message builder that helps to fill in the parameters specified in SCF-222 v4.0 section 3.4.3.
class ul_tti_request_message_builder
{
  ul_tti_request_message& msg;
  using pdu_type = ul_tti_request_message::pdu_type;

public:
  explicit ul_tti_request_message_builder(ul_tti_request_message& msg_) : msg(msg_)
  {
    msg.num_pdus_of_each_type.fill(0);
  }

  /// Sets the UL_TTI.request basic parameters and returns a reference to the builder.
  /// \note These parameters are specified in SCF-222 v4.0 section 3.4.3 in table UL_TTI,request message body.
  ul_tti_request_message_builder& set_basic_parameters(uint16_t sfn, uint16_t slot)
  {
    msg.sfn  = sfn;
    msg.slot = slot;

    // NOTE: number of PDU groups set to 0, because groups are not enabled in this FAPI message at the moment.
    msg.num_groups = 0;

    return *this;
  }

  /// Adds a PRACH PDU to the message and returns a builder that helps to fill the parameters.
  /// \note These parameters are specified in SCF-222 v4.0 section 3.4.3.1 in table PRACH PDU.
  ul_prach_pdu_builder add_prach_pdu(pci_t             pci,
                                     uint8_t           num_occasions,
                                     prach_format_type format_type,
                                     uint8_t           index_fd_ra,
                                     uint8_t           prach_start_symbol,
                                     uint16_t          num_cs)
  {
    ul_prach_pdu_builder builder = add_prach_pdu();
    builder.set_basic_parameters(pci, num_occasions, format_type, index_fd_ra, prach_start_symbol, num_cs);

    return builder;
  }

  /// Adds a PRACH PDU to the message and returns a builder that helps to fill the parameters.
  /// \note These parameters are specified in SCF-222 v4.0 section 3.4.3.1 in table PRACH PDU.
  ul_prach_pdu_builder add_prach_pdu()
  {
    auto& pdu    = msg.pdus.emplace_back();
    pdu.pdu_type = ul_pdu_type::PRACH;

    ++msg.num_pdus_of_each_type[static_cast<unsigned>(pdu_type::PRACH)];

    ul_prach_pdu_builder builder(pdu.prach_pdu);

    return builder;
  }

  /// Adds a PUCCH PDU to the message with the given format type and returns a builder that helps to fill the
  /// parameters.
  ul_pucch_pdu_builder add_pucch_pdu(pucch_format format_type)
  {
    auto& pdu    = msg.pdus.emplace_back();
    pdu.pdu_type = ul_pdu_type::PUCCH;

    if (format_type == pucch_format::FORMAT_0 || format_type == pucch_format::FORMAT_1) {
      ++msg.num_pdus_of_each_type[static_cast<unsigned>(pdu_type::PUCCH_format01)];
    } else {
      ++msg.num_pdus_of_each_type[static_cast<unsigned>(pdu_type::PUCCH_format234)];
    }

    ul_pucch_pdu_builder builder(pdu.pucch_pdu);
    pdu.pucch_pdu.format_type = format_type;

    return builder;
  }

  /// Adds a PUCCH PDU to the message and returns a builder that helps to fill the parameters.
  /// \note These parameters are specified in SCF-222 v4.0 section 3.4.3.3 in table PUCCH PDU.
  ul_pucch_pdu_builder add_pucch_pdu(rnti_t rnti, uint32_t handle, pucch_format format_type)
  {
    ul_pucch_pdu_builder builder = add_pucch_pdu(format_type);
    builder.set_basic_parameters(rnti, handle);

    return builder;
  }

  /// Adds a PUSCH PDU to the message and returns a builder that helps to fill the parameters.
  /// \note These parameters are specified in SCF-222 v4.0 section 3.4.3.2 in table PUSCH PDU.
  ul_pusch_pdu_builder add_pusch_pdu()
  {
    auto& pdu    = msg.pdus.emplace_back();
    pdu.pdu_type = ul_pdu_type::PUSCH;

    ++msg.num_pdus_of_each_type[static_cast<unsigned>(pdu_type::PUSCH)];

    ul_pusch_pdu_builder builder(pdu.pusch_pdu);

    return builder;
  }

  /// Adds a PUSCH PDU to the message and returns a builder that helps to fill the parameters.
  /// \note These parameters are specified in SCF-222 v4.0 section 3.4.3.2 in table PUSCH PDU.
  ul_pusch_pdu_builder add_pusch_pdu(rnti_t rnti, uint32_t handle)
  {
    ul_pusch_pdu_builder builder = add_pusch_pdu();
    builder.set_basic_parameters(rnti, handle);

    return builder;
  }

  /// Adds a SRS PDU to the message and returns a builder that helps to fill the parameters.
  /// \note These parameters are specified in SCF-222 v4.0 section 3.4.3.3 in table SRS PDU.
  ul_srs_pdu_builder add_srs_pdu()
  {
    auto& pdu    = msg.pdus.emplace_back();
    pdu.pdu_type = ul_pdu_type::SRS;

    ++msg.num_pdus_of_each_type[static_cast<unsigned>(pdu_type::SRS)];

    ul_srs_pdu_builder builder(pdu.srs_pdu);

    return builder;
  }
};

} // namespace fapi
} // namespace srsran
