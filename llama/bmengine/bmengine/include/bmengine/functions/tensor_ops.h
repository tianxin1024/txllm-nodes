#pragma once
#include <vector>
#include "bmengine/core/core.h"

namespace bmengine {

namespace functions {
core::Tensor concat_tensor(
    const core::Context& ctx, const core::Tensor& A, const core::Tensor& B, int dim = -1);
core::Tensor concat_tensor(
    const core::Context& ctx, const std::vector<core::Tensor>& tensors, int dim = 0);

core::Tensor concat_broadcast_b(
    const core::Context& ctx, const core::Tensor& A, const core::Tensor& B);

core::Tensor stack_tensor(const core::Context& ctx, const std::vector<core::Tensor>& tensors);

} // namespace functions
} // namespace bmengine