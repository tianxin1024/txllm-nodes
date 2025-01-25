#pragma once
#include <bmengine/core/core.h>
#include "backend/model.h"

namespace model {
class ModelContext;
}

namespace nn {
using namespace bmengine;

class EncoderLayer : public core::Layer {
    BM_LAYER_DEF(EncoderLayer);
    int layer_id{0};
    bool parallel;

    EncoderLayer(const core::Context &ctx,
                 model::ModelConfig config,
                 model::QuantConfig quant_config,
                 bool parallel);

    core::Tensor forward(const core::Context &ctx,
                         const core::Tensor &inp,           // (batch, len_q, dim_model)
                         const core::Tensor &mask,          // (batch, len_q, len_buf)
                         const core::Tensor &position_bias, // if relative (batch, num_head, len_q, len_buf) else if rotary (batch, len_q)
                         const core::Tensor &seqlens_q,     // (batch, 1)
                         const core::Tensor &seqlens_kv,    // (batch, 1)
                         const core::Tensor *past_k,        // (batch, num_head, len_buf, dim_head)
                         const core::Tensor *past_v,        // (batch, num_head, len_buf, dim_head)
                         const core::Tensor *block_table,   // (batch, blocks_per_seq)
                         const core::Tensor *placement);    // (batch, len_q) int32

    void load_state_dict(const core::Context &ctx,
                         const std::map<std::string, const core::Tensor> &state_dict,
                         const std::string &prefix,
                         bool allow_missing = false) override;

}; // end of class EncoderLayer

} // namespace nn
