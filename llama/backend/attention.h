#pragma once

#include <bmengine/core/core.h>
#include "backend/model.h"
#include "backend/model_context.h"

namespace nn {

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

    void load_state_dict(const core::Context &ctx,
                         const std::map<std::string, const core::Tensor> &state_dict,
                         const std::string &prefix,
                         bool allow_missing = false) override;

}; // end of class Attention

} // namespace nn
