#pragma once

#include <bmengine/core/core.h>
#include <memory>

namespace nn {

using namespace bmengine;

void element_add_scale_out(
    const core::Context &ctx, const core::Tensor &a, const core::Tensor &b, core::Tensor &c, float scale, bool scale_residual = true);

} // namespace nn
