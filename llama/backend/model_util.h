#pragma once

#include <bmengine/core/core.h>

namespace model {

using namespace bmengine;

core::Tensor convert_fp32(const core::Context &ctx, const core::Tensor &logits);

} // namespace model
