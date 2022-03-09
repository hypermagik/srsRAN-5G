/// \file
/// \brief LDPC rate matching and recovering unit test.

#include "ldpc_rm_test_sets.h"
#include "srsgnb/phy/upper/channel_coding/ldpc.h"
#include "srsgnb/phy/upper/channel_coding/ldpc_rate_dematcher.h"
#include "srsgnb/phy/upper/channel_coding/ldpc_rate_matcher.h"
#include "srsgnb/support/srsran_assert.h"

using namespace srsgnb;
using namespace srsgnb::ldpc;

/// \brief Runs a single rate-matcher test for the specified configuration.
///
/// The test first takes a codeword, matches it to 5 different rates and compares the result with the provided
/// benchmark. In a second step, the correctness of the rate dematching procedure is verified in a similar manner.
/// \param[in] rv       Redundancy version, valid options {0, 1, 2, 3}.
/// \param[in] mod      Modulation scheme.
/// \param[in] lbrm     Activates the limited buffer rate matching mode when set to true.
/// \param[in] matched  Vector with expected outputs.
/// \return Zero if the test is successful (all generated bit sequences correspond to the expected ones), one otherwise.
int run_rv_mod_lbrm_test(unsigned                                           rv,
                         modulation_scheme                                  mod,
                         bool                                               lbrm,
                         const std::array<std::vector<uint8_t>, nof_rates>& matched);

static const std::unique_ptr<ldpc_rate_matcher>   matcher   = create_ldpc_rate_matcher();
static const std::unique_ptr<ldpc_rate_dematcher> dematcher = create_ldpc_rate_dematcher();

int main()
{
  // BPSK
  srsran_assert(run_rv_mod_lbrm_test(0, modulation_scheme::BPSK, false, matched_rv0_bpsk) == 0,
                "LDPC RM - BPSK RV0 no LBRM - failed.");
  srsran_assert(run_rv_mod_lbrm_test(1, modulation_scheme::BPSK, false, matched_rv1_bpsk) == 0,
                "LDPC RM - BPSK RV1 no LBRM - failed.");
  srsran_assert(run_rv_mod_lbrm_test(2, modulation_scheme::BPSK, false, matched_rv2_bpsk) == 0,
                "LDPC RM - BPSK RV2 no LBRM - failed.");
  srsran_assert(run_rv_mod_lbrm_test(3, modulation_scheme::BPSK, false, matched_rv3_bpsk) == 0,
                "LDPC RM - BPSK RV3 no LBRM - failed.");
  srsran_assert(run_rv_mod_lbrm_test(0, modulation_scheme::BPSK, true, matched_rv0_bpsk_lbrm) == 0,
                "LDPC RM - BPSK RV0 LBRM - failed.");
  srsran_assert(run_rv_mod_lbrm_test(1, modulation_scheme::BPSK, true, matched_rv1_bpsk_lbrm) == 0,
                "LDPC RM - BPSK RV1 LBRM - failed.");
  srsran_assert(run_rv_mod_lbrm_test(2, modulation_scheme::BPSK, true, matched_rv2_bpsk_lbrm) == 0,
                "LDPC RM - BPSK RV2 LBRM - failed.");
  srsran_assert(run_rv_mod_lbrm_test(3, modulation_scheme::BPSK, true, matched_rv3_bpsk_lbrm) == 0,
                "LDPC RM - BPSK RV3 LBRM - failed.");

  // QPSK
  srsran_assert(run_rv_mod_lbrm_test(0, modulation_scheme::QPSK, false, matched_rv0_qpsk) == 0,
                "LDPC RM - QPSK RV0 no LBRM - failed.");
  srsran_assert(run_rv_mod_lbrm_test(1, modulation_scheme::QPSK, false, matched_rv1_qpsk) == 0,
                "LDPC RM - QPSK RV1 no LBRM - failed.");
  srsran_assert(run_rv_mod_lbrm_test(2, modulation_scheme::QPSK, false, matched_rv2_qpsk) == 0,
                "LDPC RM - QPSK RV2 no LBRM - failed.");
  srsran_assert(run_rv_mod_lbrm_test(3, modulation_scheme::QPSK, false, matched_rv3_qpsk) == 0,
                "LDPC RM - QPSK RV3 no LBRM - failed.");
  srsran_assert(run_rv_mod_lbrm_test(0, modulation_scheme::QPSK, true, matched_rv0_qpsk_lbrm) == 0,
                "LDPC RM - QPSK RV0 LBRM - failed.");
  srsran_assert(run_rv_mod_lbrm_test(1, modulation_scheme::QPSK, true, matched_rv1_qpsk_lbrm) == 0,
                "LDPC RM - QPSK RV1 LBRM - failed.");
  srsran_assert(run_rv_mod_lbrm_test(2, modulation_scheme::QPSK, true, matched_rv2_qpsk_lbrm) == 0,
                "LDPC RM - QPSK RV2 LBRM - failed.");
  srsran_assert(run_rv_mod_lbrm_test(3, modulation_scheme::QPSK, true, matched_rv3_qpsk_lbrm) == 0,
                "LDPC RM - QPSK RV3 LBRM - failed.");

  // 16QAM
  srsran_assert(run_rv_mod_lbrm_test(0, modulation_scheme::QAM16, false, matched_rv0_16qam) == 0,
                "LDPC RM - 16QAM RV0 no LBRM - failed.");
  srsran_assert(run_rv_mod_lbrm_test(1, modulation_scheme::QAM16, false, matched_rv1_16qam) == 0,
                "LDPC RM - 16QAM RV1 no LBRM - failed.");
  srsran_assert(run_rv_mod_lbrm_test(2, modulation_scheme::QAM16, false, matched_rv2_16qam) == 0,
                "LDPC RM - 16QAM RV2 no LBRM - failed.");
  srsran_assert(run_rv_mod_lbrm_test(3, modulation_scheme::QAM16, false, matched_rv3_16qam) == 0,
                "LDPC RM - 16QAM RV3 no LBRM - failed.");
  srsran_assert(run_rv_mod_lbrm_test(0, modulation_scheme::QAM16, true, matched_rv0_16qam_lbrm) == 0,
                "LDPC RM - 16QAM RV0 LBRM - failed.");
  srsran_assert(run_rv_mod_lbrm_test(1, modulation_scheme::QAM16, true, matched_rv1_16qam_lbrm) == 0,
                "LDPC RM - 16QAM RV1 LBRM - failed.");
  srsran_assert(run_rv_mod_lbrm_test(2, modulation_scheme::QAM16, true, matched_rv2_16qam_lbrm) == 0,
                "LDPC RM - 16QAM RV2 LBRM - failed.");
  srsran_assert(run_rv_mod_lbrm_test(3, modulation_scheme::QAM16, true, matched_rv3_16qam_lbrm) == 0,
                "LDPC RM - 16QAM RV3 LBRM - failed.");

  // 64QAM
  srsran_assert(run_rv_mod_lbrm_test(0, modulation_scheme::QAM64, false, matched_rv0_64qam) == 0,
                "LDPC RM - 64QAM RV0 no LBRM - failed.");
  srsran_assert(run_rv_mod_lbrm_test(1, modulation_scheme::QAM64, false, matched_rv1_64qam) == 0,
                "LDPC RM - 64QAM RV1 no LBRM - failed.");
  srsran_assert(run_rv_mod_lbrm_test(2, modulation_scheme::QAM64, false, matched_rv2_64qam) == 0,
                "LDPC RM - 64QAM RV2 no LBRM - failed.");
  srsran_assert(run_rv_mod_lbrm_test(3, modulation_scheme::QAM64, false, matched_rv3_64qam) == 0,
                "LDPC RM - 64QAM RV3 no LBRM - failed.");
  srsran_assert(run_rv_mod_lbrm_test(0, modulation_scheme::QAM64, true, matched_rv0_64qam_lbrm) == 0,
                "LDPC RM - 64QAM RV0 LBRM - failed.");
  srsran_assert(run_rv_mod_lbrm_test(1, modulation_scheme::QAM64, true, matched_rv1_64qam_lbrm) == 0,
                "LDPC RM - 64QAM RV1 LBRM - failed.");
  srsran_assert(run_rv_mod_lbrm_test(2, modulation_scheme::QAM64, true, matched_rv2_64qam_lbrm) == 0,
                "LDPC RM - 64QAM RV2 LBRM - failed.");
  srsran_assert(run_rv_mod_lbrm_test(3, modulation_scheme::QAM64, true, matched_rv3_64qam_lbrm) == 0,
                "LDPC RM - 64QAM RV3 LBRM - failed.");

  // 256QAM
  srsran_assert(run_rv_mod_lbrm_test(0, modulation_scheme::QAM256, false, matched_rv0_256qam) == 0,
                "LDPC RM - 256QAM RV0 no LBRM - failed.");
  srsran_assert(run_rv_mod_lbrm_test(1, modulation_scheme::QAM256, false, matched_rv1_256qam) == 0,
                "LDPC RM - 256QAM RV1 no LBRM - failed.");
  srsran_assert(run_rv_mod_lbrm_test(2, modulation_scheme::QAM256, false, matched_rv2_256qam) == 0,
                "LDPC RM - 256QAM RV2 no LBRM - failed.");
  srsran_assert(run_rv_mod_lbrm_test(3, modulation_scheme::QAM256, false, matched_rv3_256qam) == 0,
                "LDPC RM - 256QAM RV3 no LBRM - failed.");
  srsran_assert(run_rv_mod_lbrm_test(0, modulation_scheme::QAM256, true, matched_rv0_256qam_lbrm) == 0,
                "LDPC RM - 256QAM RV0 LBRM - failed.");
  srsran_assert(run_rv_mod_lbrm_test(1, modulation_scheme::QAM256, true, matched_rv1_256qam_lbrm) == 0,
                "LDPC RM - 256QAM RV1 LBRM - failed.");
  srsran_assert(run_rv_mod_lbrm_test(2, modulation_scheme::QAM256, true, matched_rv2_256qam_lbrm) == 0,
                "LDPC RM - 256QAM RV2 LBRM - failed.");
  srsran_assert(run_rv_mod_lbrm_test(3, modulation_scheme::QAM256, true, matched_rv3_256qam_lbrm) == 0,
                "LDPC RM - 256QAM RV3 LBRM - failed.");
}

int run_rv_mod_lbrm_test(unsigned                                           rv,
                         modulation_scheme                                  mod,
                         bool                                               lbrm,
                         const std::array<std::vector<uint8_t>, nof_rates>& matched_bm)
{
  for (int i = 0; i != nof_rates; ++i) {
    unsigned mod_order = static_cast<unsigned>(mod);
    unsigned rm_length = static_cast<unsigned>(floor(message_length / rate_list[i] / mod_order)) * mod_order;
    std::vector<uint8_t>        matched(rm_length);
    unsigned                    n_ref = lbrm ? test_Nref : 0;
    ldpc_rate_matcher::config_t rm_cfg{rv, mod, n_ref};
    matcher->rate_match(matched, codeblock, rm_cfg);

    // Compare the rate matched codeblocks with the benchmark ones.
    assert(matched_bm[i].size() == rm_length);
    if (!std::equal(matched.cbegin(), matched.cend(), matched_bm[i].cbegin())) {
      return 1;
    }

    // Transform rate-matched bits into log-likelihood ratios.
    std::vector<int8_t> llrs(rm_length);
    auto                bit_to_llrs = [](const uint8_t& b) { return 1 - 2 * b; };
    std::transform(matched_bm[i].cbegin(), matched_bm[i].cend(), llrs.begin(), bit_to_llrs);

    std::vector<int8_t>           dematched(codeblock.size());
    ldpc_rate_dematcher::config_t cfg{rv, mod, n_ref};
    dematcher->rate_dematch(dematched, llrs, nof_filler_bits, cfg);

    // To check the dematcher output, we need to apply the rate matcher to it and compare with the output
    // obtained in the first part of the test. First, transform LLRs into hard bits.
    std::vector<uint8_t> hard(block_length);
    auto                 llrs_to_bit = [](const int8_t& b) {
      if (b == INT8_MAX) {
        return filler_bit;
      }
      return static_cast<uint8_t>((b >= 0) ? 0 : 1);
    };
    std::transform(dematched.cbegin(), dematched.cend(), hard.begin(), llrs_to_bit);

    // Now, apply the rate matcher and compare results.
    std::vector<uint8_t> matched2(rm_length);
    matcher->rate_match(matched2, hard, rm_cfg);
    if (!std::equal(matched.cbegin(), matched.cend(), matched2.cbegin())) {
      return 2;
    }
  }
  return 0;
}
