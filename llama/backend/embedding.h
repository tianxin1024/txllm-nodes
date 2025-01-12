#pragma once
#include <bmengine/core/core.h>

namespace nn {

using namespace bmengine;

class RawEmbedding : public core::Layer {
    BM_LAYER_DEF(RawEmbedding);

    RawEmbedding(const core::Context &ctx,
                 int dim_model,
                 int vocab_size,
                 bool scale_weights = false,
                 core::DataType dtype = core::DataType::kHalf,
                 bool parallel = false);

    void set_scale_weights(bool b);
    void set_scale_factor(float b);
    void set_logit_scale(float b);

}; // end of class RawEmbedding

} // namespace nn
