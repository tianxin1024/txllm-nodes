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

    core::Tensor forward(const core::Context &ctx,
                         const core::Tensor &ids); // (seq_len)

    core::Tensor projection(const core::Context &ctx,
                            const core::Tensor &input); // (seq_len, dim_model)

    void load_state_dict(const core::Context &ctx,
                         const std::map<std::string, const core::Tensor> &state_dict,
                         const std::string &prefix,
                         bool allow_missing) override;

}; // end of class RawEmbedding

} // namespace nn
