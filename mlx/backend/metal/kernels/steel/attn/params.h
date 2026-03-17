// Copyright © 2024 Apple Inc.

#pragma once

///////////////////////////////////////////////////////////////////////////////
// Attn param classes
///////////////////////////////////////////////////////////////////////////////

namespace mlx {
namespace steel {

struct AttnParams {
  int B; ///< Batch Size
  int H; ///< Heads
  int D; ///< Head Dim

  int qL; ///< Query Sequence Length
  int kL; ///< Key Sequence Length

  int gqa_factor; ///< Group Query factor
  float scale; ///< Attention scale

  int NQ; ///< Number of query blocks
  int NK; ///< Number of key/value blocks

  int NQ_aligned; ///< Number of full query blocks
  int NK_aligned; ///< Number of full key/value blocks

  int qL_rem; ///< Remainder in last query block
  int kL_rem; ///< Remainder in last key/value block
  int qL_off; ///< Offset in query sequence start

  int64_t Q_strides[3]; ///< Query  strides (B, H, L, D = 1)
  int64_t K_strides[3]; ///< Key    strides (B, H, L, D = 1)
  int64_t V_strides[3]; ///< Value  strides (B, H, L, D = 1)
  int64_t O_strides[3]; ///< Output strides (B, H, L, D = 1)
};

struct AttnMaskParams {
  int64_t M_strides[3]; ///< Mask  strides (B, H, qL, kL = 1)
};

struct AttnVarlenParams {
  int H; ///< Number of heads
  int D; ///< Head dimension
  int gqa_factor; ///< GQA factor
  float scale; ///< Attention scale (pre-multiplied by M_LOG2E_F)
  int max_seqlen_q; ///< Max individual Q sequence length
  int max_seqlen_k; ///< Max individual K sequence length
  int num_sequences; ///< Number of packed sequences
  int total_q; ///< Sum of all Q lengths
  int total_k; ///< Sum of all K lengths
  int64_t Q_strides[2]; ///< [0]=head stride, [1]=token stride
  int64_t K_strides[2];
  int64_t V_strides[2];
  int64_t O_strides[2];
};

} // namespace steel
} // namespace mlx
