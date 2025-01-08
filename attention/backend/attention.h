#pragma once
#include <bmengine/core/core.h>
#include "backend/model_config.h"

using namespace bmengine;

class Attention : public core::Layer {
    BM_LAYER_DEF(Attention);

    Attention(const core::Context &ctx,
              model::ModelConfig block_config,
              model::QuantConfig quant_config,
              bool parallel);

    core::Tensor forward(const core::Context &ctx,
                         const core::Tensor &inp,           // (len_q, dim_model)
                         const core::Tensor &mask,          // (len_q, len_buf)
                         const core::Tensor &position_bias, // if relative (num_head, len_q, len_buf) else if rotary (len_q)
                         const core::Tensor &seqlens_q,     // (batch?, 1,)  int32
                         const core::Tensor &seqlens_kv,    // (batch?, 1,)  int32
                         const core::Tensor *past_k,        // (num_head, len_buf, dim_head)
                         const core::Tensor *past_v,        // (num_head, len_buf, dim_head)
                         const core::Tensor *block_table,   // (batch_size, block_per_seq)
                         const core::Tensor *placement,     // (batch?, len_q) int32
                         core::Tensor *output);

}; // end of class Attention
