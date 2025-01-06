#pragma once
#include <bmengine/core/core.h>
#include "backend/model_config.h"

using namespace bmengine;

class Attention : public core::Layer {
    BM_LAYER_DEF(Attention);

    Attention(const core::Context &ctx,
              model::ModelConfig block_config,
              model::QuantConfig quant_config,
              bool parallel);

}; // end of class Attention
