// Modified for JustFit: draft-tree verification (experiment).
#pragma once

// Parameter layouts shared by host dispatch code and Metal kernels.
#include "metal/abi/ExecutionGeometry.h"
#include "metal/abi/GDN.h"
#include "metal/abi/PagedAttention.h"
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

// One lane's draft tree in its verify rows. Rows are in DFS preorder: row 0 is
// the anchor (depth 0) and every row's parent row is smaller than the row.
// A copy is one root-to-leaf path; path[c][d] is its row at depth d. Every
// row belongs to the first copy, in DFS order, whose path holds it, so its
// ancestors are path[copy[row]][0 .. depth[row]].
#define JUSTFIT_TREE_MAXIMUM_ROWS 16u
#define JUSTFIT_TREE_MAXIMUM_DEPTH 8u

struct JustFitTreeLane {
  uint32_t rows;
  uint32_t copies;
  uint32_t parent[JUSTFIT_TREE_MAXIMUM_ROWS];
  uint32_t depth[JUSTFIT_TREE_MAXIMUM_ROWS];
  uint32_t copy[JUSTFIT_TREE_MAXIMUM_ROWS];
  uint32_t token[JUSTFIT_TREE_MAXIMUM_ROWS];
  uint32_t path_length[JUSTFIT_TREE_MAXIMUM_ROWS];
  uint32_t path[JUSTFIT_TREE_MAXIMUM_ROWS][JUSTFIT_TREE_MAXIMUM_DEPTH];
};

// The tree of each lane of a verify batch of at most two lanes.
struct JustFitTreeBatch {
  JustFitTreeLane lane[2];
};

// Greedy acceptance of a tree batch: the accepted path's rows go to
// path_rows[lane * JUSTFIT_TREE_MAXIMUM_DEPTH + i].
struct JustFitTreeAcceptParams {
  JustFitTreeBatch tree;
  uint32_t remaining[2];
  uint32_t stop_token_0;
  uint32_t stop_token_1;
  // Sampled lanes (JUSTFIT_TREE_SAMPLED): rows 0 .. chain_rows - 1 of the
  // tree are the draft's sampled chain; chain_q[lane][r] is the draft's
  // probability of chain token r + 1 (row r + 1), the q of the acceptance test.
  uint32_t sampling_mask;
  uint32_t chain_rows[2];
  uint32_t reserved;
  float chain_q[2][JUSTFIT_TREE_MAXIMUM_DEPTH];
};

// In-place gather of the accepted rows of one tensor family to rows
// 0 .. retained - 1. Element (row, column) of lane b and layer y is at unit
// base + y * layer_stride + b * lane_stride + row * row_stride +
// (column / inner) * outer_stride + (column % inner) * inner_stride, in
// 16-bit units.
struct JustFitTreeGatherParams {
  uint32_t columns;
  uint32_t inner;
  uint32_t inner_stride;
  uint32_t outer_stride;
  uint32_t row_stride;
  uint32_t lane_stride;
  uint32_t layer_stride;
  uint32_t layers;
  uint32_t lanes;
};

// The GDN verify of a tree batch: the chain's parameters and the trees.
struct GDNTreeDecodeParams {
  GDNDecodeBatchParams base;
  JustFitTreeBatch tree;
};

// The tree verify attention of a batch: each lane's chain parameters and tree.
struct JustFitTreeAttentionParams {
  JustFitVerifyAttentionParams lane[2];
  JustFitTreeBatch tree;
};

static_assert(sizeof(JustFitTreeLane) == 8 + 5 * 64 + 512, "tree lane layout is plain on both sides");
static_assert(sizeof(JustFitTreeAcceptParams) == 2 * sizeof(JustFitTreeLane) + 32 + 64, "tree accept layout");
static_assert(sizeof(JustFitTreeGatherParams) == 36, "tree gather layout");
