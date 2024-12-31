#pragma once
#include "bmengine/core/core.h"

namespace bmengine {

namespace functions {

void bitonic_topk(
    const core::Context& ctx,
    const core::Tensor& x,
    const core::Tensor& out,
    const core::Tensor& pos);

class TopK : public core::Layer {
    BM_LAYER_DEF(TopK)

    TopK(const core::Context& ctx);

    /*  Returns score and index of the top-k elements. */
    std::pair<core::Tensor, core::Tensor> forward(
        const core::Context& ctx, const core::Tensor& inp, int top);
};

} // namespace functions

} // namespace bmengine
