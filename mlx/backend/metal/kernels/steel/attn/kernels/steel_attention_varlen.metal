// Copyright © 2025 Apple Inc.

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"

#include "mlx/backend/metal/kernels/steel/attn/kernels/steel_attention_varlen.h"

#define instantiate_attn_varlen(tname, dtype, bq, bk, bd, wm, wn) \
  instantiate_kernel(                                              \
      "attention_varlen_" #tname "_bq" #bq "_bk" #bk "_bd" #bd    \
      "_wm" #wm "_wn" #wn,                                        \
  attention_varlen, dtype, bq, bk, bd, wm, wn, float)

#define instantiate_attn_varlen_shapes_helper(iname, itype)       \
    instantiate_attn_varlen(iname, itype, 32, 16, 128, 4, 1)     \
    instantiate_attn_varlen(iname, itype, 32, 32,  80, 4, 1)     \
    instantiate_attn_varlen(iname, itype, 32, 32,  64, 4, 1)

instantiate_attn_varlen_shapes_helper(float16, half);
instantiate_attn_varlen_shapes_helper(bfloat16, bfloat16_t);
instantiate_attn_varlen_shapes_helper(float32, float);
// clang-format on
