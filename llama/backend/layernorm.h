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

    core::Tensor forward(const core::Context &ctx, const core::Tensor &x);

    void set_rms(bool b); // If false, use standard LayerNorm

    void load_state_dict(const core::Context &ctx,
                         const std::map<std::string, const core::Tensor> &state_dict,
                         const std::string &prefix,
                         bool allow_missing) override;
};

} // namespace nn
