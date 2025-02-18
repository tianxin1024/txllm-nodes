#pragma once

#include <bmengine/core/core.h>
#include <bmengine/functions/transpose.h>

namespace nn {

using namespace bmengine;
using bmengine::functions::transpose_2_1;

void attn_softmax(const core::Context &ctx,
                  float scale,
                  const core::Tensor &attn_score,     // (batch, num_heads, len_q, len_buf)
                  const core::Tensor &mask,           // (batch, len_q, len_buf)
                  const core::Tensor &position_bias); // if relative (batch, num_head, len_q, len_buf) else if core::Tensor()

} // namespace nn
