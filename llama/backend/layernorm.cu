#include "backend/layernorm.h"

namespace nn {

__host__ void layernorm(const core::Tensor &input,  // (batch, seq_len, dim_model)
                        const core::Tensor &weight, // (dim_model)
                        core::Tensor *output,       // (batch, seq_len, dim_model)
                        float eps,
                        float scale,
                        bool rms,
                        cudaStream_t stream,
                        const core::Tensor *input2 = nullptr,
                        core::Tensor *out_sum = nullptr) {
    // TODO tianx ...
}

class LayerNorm::impl {
public:
    class QuantImpl;
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

    virtual core::Tensor forward(const core::Context &ctx,
                                 const core::Tensor &input) { // (batch, seq_len, dim_model)

        BM_ASSERT(input.dtype() == weight.dtype(), "dtype mismatch");
        BM_ASSERT_EQ(input.size(-1), weight.size(0), "dim mismatch");
        BM_ASSERT(input.ndim() >= 2, "input.ndim() must be >= 2");
        BM_ASSERT(input.device() == weight.device(), "Input and weight must be on the same device");
        core::Tensor output = ctx.tensor(input.shape(), input.dtype());
        BM_ASSERT(output.device() == weight.device(), "Output and weight must be on the same device");
        layernorm(input, weight, &output, eps, scale, rms, ctx.current_stream()->ptr);
        return output;
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

class LayerNorm::impl::QuantImpl : public LayerNorm::impl {
public:
    QuantImpl(const core::Context &ctx, unsigned int dim_model, float eps, float scale, core::DataType dtype) :
        LayerNorm::impl(ctx, dim_model, eps, scale, dtype) {
    }

    virtual ~QuantImpl() = default;
};

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
        pimpl.reset(quant ?
                        new impl::QuantImpl(ctx, dim_model, eps, scale, dtype) :
                        new impl(ctx, dim_model, eps, scale, dtype));
    }
    add_parameter("weight", pimpl->weight);
}

LayerNorm::~LayerNorm() = default;

core::Tensor LayerNorm::forward(const core::Context &ctx, const core::Tensor &x) {
    size_t M = x.numel() / x.size(-1);
    core::EventScope event_scope(ctx, "LayerNorm[M=" + std::to_string(M) + "]", 1, 2 * x.nbytes());
    return pimpl->forward(ctx, x);
}

void LayerNorm::set_rms(bool b) {
    pimpl->set_rms(b);
}

void LayerNorm::load_state_dict(const core::Context &ctx,
                                const std::map<std::string, const core::Tensor> &state_dict,
                                const std::string &prefix,
                                bool allow_missing) {
    impl::MultiHeadImpl *p = dynamic_cast<impl::MultiHeadImpl *>(pimpl.get());
    if (p) {
        auto name = prefix + ".weight";
        ctx.load_parameter(&p->weight, name, state_dict, true, core::DistLayout::ROW);
    } else {
        core::Layer::load_state_dict(ctx, state_dict, prefix, allow_missing);
    }
}

} // namespace nn
