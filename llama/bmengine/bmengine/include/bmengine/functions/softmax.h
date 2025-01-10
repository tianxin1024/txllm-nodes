#pragma once
#include "bmengine/core/core.h"

namespace bmengine {

namespace functions {

void softmax(
    const core::Context& ctx,
    const core::Tensor& logits,
    const core::Tensor& output,
    float temperature = 1.0f);

}

}