// Modified for JustFit: product identity and integration.
#pragma once

// Parameter layouts shared by host dispatch code and Metal kernels.
#include "metal/abi/ExecutionGeometry.h"
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

// The uniforms one lane draws a decode cycle with, in [0, 1): the first
// token after a prompt draws JUSTFIT_UNIFORM_INITIAL; the draft's sampled
// proposal at position p draws JUSTFIT_UNIFORM_PROPOSALS + p; acceptance tests
// draft token p against JUSTFIT_UNIFORM_ACCEPTANCE + p; and a sampled verify
// row draws its correction, or the bonus token after the whole draft, with
// JUSTFIT_UNIFORM_CORRECTION. Lane l's uniforms start at
// l * JUSTFIT_SAMPLING_UNIFORMS.
#define JUSTFIT_UNIFORM_INITIAL 0u
#define JUSTFIT_UNIFORM_PROPOSALS 1u
#define JUSTFIT_UNIFORM_ACCEPTANCE                                          \
  (JUSTFIT_UNIFORM_PROPOSALS + JUSTFIT_DRAFT_PROPOSAL_TOKENS)
#define JUSTFIT_UNIFORM_CORRECTION                                          \
  (JUSTFIT_UNIFORM_ACCEPTANCE + JUSTFIT_DRAFT_PROPOSAL_TOKENS)
#define JUSTFIT_SAMPLING_UNIFORMS 24u

// One target-policy dispatch over lanes of JUSTFIT_TARGET_VERIFY_ROWS logits
// rows, JUSTFIT_TARGET_VERIFY_ROWS + 1 constraint-mask rows and
// JUSTFIT_SAMPLING_UNIFORMS uniforms each. Selected row s is row s % rows of
// lane s / rows: its logits row is lane * JUSTFIT_TARGET_VERIFY_ROWS +
// logits_row + s % rows, its mask row lane * (JUSTFIT_TARGET_VERIFY_ROWS + 1) +
// mask_row + s % rows, and its draw takes uniform
// lane * JUSTFIT_SAMPLING_UNIFORMS + uniform; workspaces and output tokens are
// indexed by s. Rows below drafted_rows follow draft token s % rows (verify
// input row s % rows + 1).
struct TargetSamplingParams {
  uint32_t vocabulary;
  uint32_t mask_words;
  uint32_t rows;
  uint32_t logits_row;
  uint32_t mask_row;
  uint32_t uniform;
  uint32_t drafted_rows;
  uint32_t top_k[JUSTFIT_MAXIMUM_BATCH_WIDTH];
  float temperature[JUSTFIT_MAXIMUM_BATCH_WIDTH];
  float top_p[JUSTFIT_MAXIMUM_BATCH_WIDTH];
  float min_p[JUSTFIT_MAXIMUM_BATCH_WIDTH];
  // Lanes that sample; the others take the argmax.
  uint32_t sampling_mask;
  uint32_t constrained_mask;
  // Lanes that ignore end-of-sequence: they never select a stop token.
  uint32_t exclude_stop_mask;
  uint32_t stop_token_0;
  uint32_t stop_token_1;
};

static_assert(sizeof(TargetSamplingParams) == 112,
              "Target sampling parameters are 112 bytes on both sides");

// One shard's share of a sampled row's softmax denominator: the largest
// logit it admits, the sum of exp((logit - maximum) / temperature) over its
// admitted tokens, and how many it admits.
struct TargetShardMass {
  float maximum;
  float sum;
  uint32_t admitted;
};

static_assert(sizeof(TargetShardMass) == 12,
              "Target shard masses are 12 bytes on both sides");

// A sampled row's selection over the whole vocabulary. The search records
// the row's largest admitted logit and where its min-p/top-k/top-p
// distribution ends in the order of the logits (the key and id of its last
// token, metal/kernels/decode/sampling.metal). The draw writes a drafted
// row's target probability of its draft token; the token it draws goes to
// the output tokens: for a verify row with a draft token, the correction
// acceptance takes if it rejects that token (a draw from the residual
// distribution); otherwise a draw from the row's distribution.
struct TargetVocabularyRow {
  float maximum;
  uint32_t end_key;
  uint32_t end_last;
  float draft_probability;
};

static_assert(sizeof(TargetVocabularyRow) == 16,
              "Target vocabulary rows are 16 bytes on both sides");

// One range of the vocabulary in such a row's draw, which one simdgroup of
// the row's groups sums: the kept weight of its tokens other than the
// draft's candidates, and one past the last of those with weight.
struct TargetVocabularyRange {
  float rest;
  uint32_t after;
};

static_assert(sizeof(TargetVocabularyRange) == 8,
              "Target vocabulary ranges are 8 bytes on both sides");

// A penalized request's word for each vocabulary token, in its state lane's
// row of the penalty table (ops::Sampling::rebuildPenaltyWords): the
// prompt bit marks a prompt token, and the count is how often the target
// selected it.
#define JUSTFIT_PENALTY_PROMPT_BIT 0x80000000u
#define JUSTFIT_PENALTY_COUNT_MASK 0x7fffffffu

// The penalized lanes of one penalty dispatch. Each entry names the lane of
// its logits and the penalty table row it reads; rows penalizes that many
// rows of the lane's JUSTFIT_TARGET_VERIFY_ROWS, from row_offset.
// repetition_inverse is 1 / repetition, saturated to the largest float.
struct SamplingPenaltyParams {
  uint32_t vocabulary;
  uint32_t rows;
  uint32_t row_offset;
  uint32_t entries;
  uint32_t logits_lane[JUSTFIT_MAXIMUM_BATCH_WIDTH];
  uint32_t table_row[JUSTFIT_MAXIMUM_BATCH_WIDTH];
  float repetition[JUSTFIT_MAXIMUM_BATCH_WIDTH];
  float repetition_inverse[JUSTFIT_MAXIMUM_BATCH_WIDTH];
  float presence[JUSTFIT_MAXIMUM_BATCH_WIDTH];
  float frequency[JUSTFIT_MAXIMUM_BATCH_WIDTH];
};

static_assert(sizeof(SamplingPenaltyParams) == 112,
              "Sampling penalty parameters are 112 bytes on both sides");

// The batched selector and acceptance kernels' grids cover exactly the
// dispatch's lanes: per-lane arrays hold those lanes, and entries past them
// are zero and unread.
struct SelectorBatchParams {
  uint32_t anchor[JUSTFIT_MAXIMUM_BATCH_WIDTH];
  float temperature[JUSTFIT_MAXIMUM_BATCH_WIDTH];
  uint32_t lanes;
  uint32_t sampling_mask;
  uint32_t vocabulary;
};

static_assert(sizeof(SelectorBatchParams) == 44,
              "Draft selector parameters are 44 bytes on both sides");

struct AcceptBatchParams {
  uint32_t remaining[JUSTFIT_MAXIMUM_BATCH_WIDTH];
  uint32_t stop_token_0;
  uint32_t stop_token_1;
  uint32_t sampling_mask;
};

static_assert(sizeof(AcceptBatchParams) == 28,
              "Batched acceptance parameters are 28 bytes on both sides");
