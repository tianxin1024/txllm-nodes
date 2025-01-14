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

    void load_state_dict(const core::Context &ctx,
                         const std::map<std::string, const core::Tensor> &state_dict,
                         const std::string &prefix,
                         bool allow_missing = false) override;

}; // end of class EncoderLayer

} // namespace nn
