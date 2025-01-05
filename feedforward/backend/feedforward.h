#pragma once
#include <bmengine/core/core.h>
#include "backend/model_config.h"
#include "backend/linear.h"

using namespace bmengine;

class Linear;

class FeedForward : public core::Layer {
    BM_LAYER_DEF(FeedForward);

    FeedForward(const core::Context &ctx,
                model::ModelConfig block_config,
                model::QuantConfig quant_config,
                bool parallel);

    core::Tensor forward(const core::Context &ctx, const core::Tensor &inp);

    void load_state_dict(const core::Context &ctx,
                         const std::map<std::string, const core::Tensor> &state_dict,
                         const std::string &prefix,
                         bool allow_missing) override;
}; // end of FeedForward
