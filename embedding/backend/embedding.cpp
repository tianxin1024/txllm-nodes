#include "embedding.h"
#include <bmengine/core/core.h>

using namespace bmengine;

class Embedding::impl {
public:
    core::Tensor weight;
    unsigned int dim_model;
    unsigned int vocab_size;
    core::DataType dtype;
    float scale_factor;

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
        scale_factor(scale_weights ? 1.0 / sqrtf(dim_model) : 1.0) {
        std::cout << ">>>>> Embeding impl create" << std::endl;
    }

}; // end of class Embedding::impl

Embedding::Embedding(const core::Context &ctx, int dim_model, int vocab_size, bool scale_weights, core::DataType dtype) :
    pimpl(new impl(ctx, vocab_size, dim_model, scale_weights, dtype)), core::Layer() {
}
