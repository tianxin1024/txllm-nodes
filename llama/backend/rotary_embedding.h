#pragma once

#include <bmengine/core/core.h>
#include "backend/model_config.h"

namespace nn {

using namespace bmengine;

class RotaryEmbedding : public core::Layer {
    BM_LAYER_DEF(RotaryEmbedding);

    RotaryEmbedding(const core::Context &ctx, model::ModelConfig block_config);

    std::tuple<core::Tensor, core::Tensor> forward(const core::Context &ctx,
                                                   const core::Tensor &pos, // (batch, seq_len)
                                                   const core::Tensor &q,   // (batch, seq_len, dim_model)
                                                   const core::Tensor &k);  // (batch, seq_len, dim_model)

}; // end of class RotaryEmbedding

} // namespace nn
