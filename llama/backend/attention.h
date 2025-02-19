#pragma once

#include <bmengine/core/core.h>
#include "backend/model_context.h"

namespace nn {

using namespace kvcache;
using namespace bmengine;

class Attention : public core::Layer {
    BM_LAYER_DEF(Attention);

    Attention(const core::Context &ctx,
              model::ModelConfig block_config,
              model::QuantConfig quant_config,
              bool parallel);

    core::Tensor dyn_rag_forward(model::ModelContext &ctx,
                                 const core::Tensor &inp,
                                 const core::Tensor &position,
                                 core::Tensor *outptu = nullptr);

    core::Tensor forward(const core::Context &ctx,
                         const core::Tensor &inp,           // (len_q, dim_model)
                         const core::Tensor &mask,          // (len_q, len_buf)
                         const core::Tensor &position_bias, // if relative (num_head, len_q, len_buf) else if rotary (len_q)
                         const core::Tensor &seqlens_q,     // (batch?, 1)  int32
                         const core::Tensor &seqlens_kv,    // (batch?, 1,)  int32
                         const core::Tensor *past_k,        // (num_head, len_buf, dim_head)
                         const core::Tensor *past_v,        // (num_head, len_buf, dim_head)
                         const core::Tensor *block_table,   // (batch_size, blocks_per_seq)
                         const core::Tensor *placement,     // (batch?, len_q, ) int32
                         core::Tensor *output);

    void load_state_dict(const core::Context &ctx,
                         const std::map<std::string, const core::Tensor> &state_dict,
                         const std::string &prefix,
                         bool allow_missing = false) override;

}; // end of class Attention

} // namespace nn
