#pragma once
#include "bmengine/core/core.h"

namespace bmengine {
namespace functions {

class Transpose : public core::Layer {
    BM_LAYER_DEF(Transpose)

    Transpose(const core::Context& ctx);

    core::Tensor forward(
        const core::Context& ctx,
        const core::Tensor& input,
        core::Tensor* output = nullptr
    );
};

// transpose dim1 with dim2
core::Tensor transpose_2_1(
    const core::Context& ctx,
    const core::Tensor& input, // (batch?, dim1, dim2, last_dim)
    core::Tensor* out_ptr = nullptr // (batch?, dim2, dim1, last_dim)
);

} // namespace functions
} // namespace bmengine
