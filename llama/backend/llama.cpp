#include "backend/llama.h"

namespace model {

LLaMA::LLaMA(core::Context &ctx, ModelConfig model_config, QuantConfig quant_config, bool parallel) :
    LLaMALike(model_config) {
}

} // namespace model
