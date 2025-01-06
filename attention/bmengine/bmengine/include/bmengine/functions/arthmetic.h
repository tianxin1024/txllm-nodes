#pragma once
#include <vector>
#include "bmengine/core/core.h"

namespace bmengine {

namespace functions {

core::Tensor sum(const core::Context& ctx, const core::Tensor& a);
core::Tensor reduce_abs_max(const core::Context& ctx, const core::Tensor& a, int dim=0);
core::Tensor div(
    const core::Context& ctx, const core::Tensor& a, const core::Tensor& b, float eps = 1e-6);
core::Tensor amax(const core::Context& ctx, const core::Tensor& a);
core::Tensor amin(const core::Context& ctx, const core::Tensor& a);
core::Tensor add(const core::Context& ctx, const core::Tensor& a, float b);

} // namespace functions
} // namespace bmengine