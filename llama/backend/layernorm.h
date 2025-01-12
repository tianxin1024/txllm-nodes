#pragma once

#include <bmengine/core/core.h>

namespace nn {

using namespace bmengine;

class LayerNorm : public core::Layer {
    BM_LAYER_DEF(LayerNorm);

    LayerNorm(const core::Context &ctx,
              int dim_model, bool quant = false, float eps = 1e-6, float scale = 1.0,
              core::DataType dtype = core::DataType::kHalf,
              int num_head = 1);

    void set_rms(bool b); // If false, use standard LayerNorm
};

} // namespace nn
