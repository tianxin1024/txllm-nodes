#include "backend/layernorm.h"

namespace nn {

class LayerNorm::impl {
public:
    // class QuantImpl;
    class MultiHeadImpl;
    core::Tensor weight;
    core::Tensor bias;
    float eps;
    float scale;
    bool rms{true};
    impl(const core::Context &ctx, unsigned int dim_model, float eps, float scale, core::DataType dtype) :
        weight(ctx.parameter({dim_model}, dtype)), eps(eps), scale(scale) {
    }

    virtual ~impl() = default;
    impl(const impl &) = default;
    impl(impl &&) = default;

    void set_rms(bool b) {
        rms = b;
    }

}; // end of class LayerNorm::impl

class LayerNorm::impl::MultiHeadImpl : public LayerNorm::impl {
public:
    MultiHeadImpl(const core::Context &ctx,
                  unsigned int dim_model,
                  float eps,
                  float scale,
                  core::DataType dtype,
                  size_t num_head) :
        LayerNorm::impl(ctx, dim_model, eps, scale, dtype) {
        weight = ctx.parameter({num_head, size_t(dim_model) / num_head}, dtype);
    }

    virtual ~MultiHeadImpl() = default;

}; // end of class LayerNorm::impl::MultiHeadImpl

LayerNorm::LayerNorm(const core::Context &ctx,
                     int dim_model,
                     bool quant,
                     float eps,
                     float scale,
                     core::DataType dtype,
                     int num_head) :
    core::Layer() {
    if (num_head > 1) {
        pimpl.reset(new impl::MultiHeadImpl(ctx, dim_model, eps, scale, dtype, num_head));
    } else {
        // pimpl->reset(quant ?
        //                  new impl::QuantImpl(ctx, dim_model, eps, scale, dtype) :
        //                  new impl(ctx, dim_model, eps, scale, dtype));
    }
    add_parameter("weight", pimpl->weight);
}

LayerNorm::~LayerNorm() = default;

void LayerNorm::set_rms(bool b) {
    pimpl->set_rms(b);
}

void LayerNorm::load_state_dict(const core::Context &ctx,
                                const std::map<std::string, const core::Tensor> &state_dict,
                                const std::string &prefix,
                                bool allow_missing) {
    std::cout << "LayerNorm::load_state_dict" << std::endl;
}

} // namespace nn
