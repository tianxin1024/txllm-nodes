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

}; // end of class EncoderLayer

} // namespace nn
