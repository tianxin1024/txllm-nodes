#include "backend/feedforward.h"
#include "backend/linear.h"
#include <bmengine/functions/all.h>

namespace nn {

class FeedForward::impl {
public:
    class NormalImpl;

    virtual ~impl() = default;

    enum class WeightType {
        kIn,
        kGated,
        kOut,
    };
}; // end of class FeedForward::impl

class FeedForward::impl::NormalImpl : public FeedForward::impl {
public:
    int dim_model;
    int dim_ff;
    std::string act_fn_type;
    bool scale_weights;
    bool weight_transposed;
    core::DataType dtype;
    model::QuantConfig quant;
    bool parallel;
    Linear w_in, w_gated, w_out;
    std::unique_ptr<Linear> w_fuse_in_gated;

    NormalImpl(const core::Context &ctx,
               model::ModelConfig cfg,
               model::QuantConfig quant,
               bool parallel = false) :
        dim_model(cfg.dim_model),
        dim_ff(cfg.dim_ff),
        act_fn_type(cfg.activate_fn),
        scale_weights(cfg.scale_weights),
        weight_transposed(cfg.weight_transposed),
        dtype(cfg.dtype),
        quant(quant),
        parallel(parallel),
        w_in(ctx, dim_model, dim_ff, "", quant, scale_weights, weight_transposed, parallel, core::DistLayout::COLUMNAR, dtype),
        w_gated(ctx, dim_model, dim_ff, "", quant, scale_weights, weight_transposed, parallel, core::DistLayout::COLUMNAR, dtype),
        w_out(ctx, dim_ff, dim_model, "", quant, scale_weights, weight_transposed, parallel, core::DistLayout::ROW, dtype) {
    }

    virtual ~NormalImpl() = default;

}; // end of class FeedForward::impl::NormalImpl

FeedForward::FeedForward(const core::Context &ctx,
                         model::ModelConfig cfg,
                         model::QuantConfig quant_config,
                         bool parallel) {
    impl::NormalImpl *p = new impl::NormalImpl(ctx, cfg, quant_config, parallel);
    pimpl.reset(p);
    add_submodule("w_in", p->w_in);
    add_submodule("w_gated", p->w_gated);
    add_submodule("w_out", p->w_out);
}

FeedForward::~FeedForward() = default;

} // namespace nn
