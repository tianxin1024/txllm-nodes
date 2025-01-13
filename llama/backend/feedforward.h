#pragma once

#include <bmengine/core/core.h>
#include "backend/model.h"

namespace nn {
using namespace bmengine;

class FeedForward : public core::Layer {
    BM_LAYER_DEF(FeedForward);

    FeedForward(const core::Context &ctx,
                model::ModelConfig block_config,
                model::QuantConfig quant_config,
                bool parallel);

}; // end of class FeedForward

} // namespace nn
