#include "backend/feedforward.h"
#include "backend/utils.h"
#include "backend/activation_kernel.h"

using namespace bmengine;
using bmengine::core::DistLayout;
using bmengine::core::Tensor;

class FeedForward::impl {
public:
    class NormalImpl;
    class MOEImpl;

    virtual ~impl() = default;
    virtual Tensor forward(const core::Context &ctx, const Tensor &input, bool quant_back) = 0;

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

    virtual Tensor forward(const core::Context &ctx, const Tensor &input, bool quant_back) {
        Tensor up;
        static int fuse_v2_thres = utils::get_int_env("FUSE_V2_THRES", 8);
        // if (w_in.support_fuse_gptq_gate_in(input)
        //     && input.ndim() == 2 && input.size(0) <= fuse_v2_thres
        //     && act_fn_type == "silu" && input.size(0) <= 2
        //     && input.nbytes() < ctx.get_max_shared_memory()) {
        //     // up = gptq_fuse_up(ctx, input);
        // }
        auto w_0 = w_in.forward(ctx, input);
        {
            auto w_1 = w_gated.forward(ctx, input);
            ctx.recordEvent("gate_activate_multiply", 3);
            // aactivate(w_0) * w_1
            nn::gate_mul_inplace(ctx, w_0, w_1, act_fn_type);
        }
        up = w_0;

        auto ret = w_out.forward(ctx, up, parallel || !quant.fuse_block() || quant_back);
        return ret;
    }
};

class FeedForward::impl::MOEImpl : public FeedForward::impl {
public:
    int dim_model, dim_ff;
};

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

core::Tensor FeedForward::forward(const core::Context &ctx, const core::Tensor &input) {
    auto moe_impl = dynamic_cast<impl::MOEImpl *>(pimpl.get());
    core::EventScope event_scope(ctx, moe_impl ? "MOE" : "FeedForward", 1);
    auto shape2d = {input.numel() / input.size(-1), input.size(-1)};
    const Tensor &input2d = input.ndim() == 2 ? input : input.view(shape2d);
    Tensor ret = pimpl->forward(ctx, input2d, true);
    if (input.ndim() == 2) {
        return ret;
    } else {
        auto shape_nd = input.shape();
        *shape_nd.rbegin() = ret.size(-1);
        return ret.view(shape_nd);
    }
}

void FeedForward::load_state_dict(const core::Context &ctx,
                                  const std::map<std::string, const core::Tensor> &state_dict,
                                  const std::string &prefix,
                                  bool allow_missing) {
    std::cout << ">>>>>>>>>>>> FeedForward: " << std::endl;
    core::Layer::load_state_dict(ctx, state_dict, prefix, allow_missing);
    // int fuse_w_in = utils::get_int_env("CPM_FUSE_FF_IN", 0);
    // auto normal_impl = dynamic_cast<impl::NormalImpl *>(pimpl.get());
}
