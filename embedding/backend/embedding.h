#pragma once
#include <bmengine/core/core.h>

using namespace bmengine;

class Embedding : public core::Layer {
    BM_LAYER_DEF(Embedding);

    Embedding(const core::Context &ctx, int dim_model, int vocab_size, bool scale_weights = false, core::DataType dtype = core::DataType::kHalf);

    core::Tensor forward(
        const core::Context &ctx,
        const core::Tensor &ids,    // (seq_len)
        const core::Tensor &ids_sub // (seq_len)
    );
};
