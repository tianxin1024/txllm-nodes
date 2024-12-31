#include "backend/embedding.h"
#include <bmengine/core/core.h>
#include <bmengine/functions/gemm.h>

class Embedding::impl {
public:
    core::Tensor weight;
    unsigned int dim_model;
    unsigned int vocab_size;
    core::DataType dtype;

    float scale_factor;
    functions::Gemm local_gemm;
    functions::Gemm local_gemm_alpha;
    impl(
        const core::Context &ctx,
        unsigned int vocab_size,
        unsigned int dim_model,
        bool scale_weights,
        core::DataType dtype) :
        weight(ctx.parameter({vocab_size, dim_model}, dtype)),
        dim_model(dim_model),
        vocab_size(vocab_size),
        dtype(dtype),
        scale_factor(scale_weights ? 1.0 / sqrtf(dim_model) : 1.0),
        local_gemm(ctx, dtype, false, true),
        local_gemm_alpha(ctx, dtype, false, true, scale_factor) {
        if (ctx.high_precision() >= 1) {
            local_gemm.set_compute_type(CUBLAS_COMPUTE_32F);
            local_gemm_alpha.set_compute_type(CUBLAS_COMPUTE_32F);
        }
    }

}; // end of class Embedding::impl

Embedding::Embedding(const core::Context &ctx, int dim_model, int vocab_size, bool scale_weights, core::DataType dtype) :
    pimpl(new impl(ctx, vocab_size, dim_model, scale_weights, dtype)), core::Layer() {
    add_parameter("weight", pimpl->weight);
}

Embedding::~Embedding() = default;
