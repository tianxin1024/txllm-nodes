#include "backend/feedforward.h"
#include "backend/linear.h"
#include "backend/utils.h"
#include "backend/activation_kernel.h"
#include <bmengine/functions/all.h>

namespace nn {

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

    virtual Tensor forward(const core::Context &ctx,
                           const Tensor &input,
                           bool quant_back) {
        Tensor up;
        static int fuse_v2_thres = utils::get_int_env("FUSE_V2_THRES", 8);

        auto w_0 = w_in.forward(ctx, input);
        {
            auto w_1 = w_gated.forward(ctx, input);
            ctx.recordEvent("gate_activate_multiply", 3);
            // activate(w_0) * w_1
            nn::gate_mul_inplace(ctx, w_0, w_1, act_fn_type);
        }
        up = w_0;
        auto ret = w_out.forward(ctx, up, parallel || !quant.fuse_block() || quant_back);
        return ret;
    }

    void try_fuse_up_weights(const core::Context &ctx) {
        auto a = Linear::fuse(ctx, w_in, w_gated);
        w_fuse_in_gated = std::unique_ptr<Linear>(a);
    }

}; // end of class FeedForward::impl::NormalImpl

class FeedForward::impl::MOEImpl : public FeedForward::impl {
public:
    int dim_model, dim_ff;
    int num_experts;
    int num_experts_may_share;
    int num_local_experts;
    int top_k;
    int top_k_may_share;
    bool norm_topk_prob;
    float routed_scaling_factor;
    int topk_group;
    int n_group;
    core::DataType dtype;
    bool parallel;
    bool exp_parallel;
    bool dyn_shared{false};
    int n_shared_experts{0};
    const int world_size;
    const int local_rank;

    Linear router;

    MOEImpl(const core::Context &ctx,
            model::ModelConfig cfg,
            model::QuantConfig quant_config,
            bool parallel,
            bool dyn_shared = false) :
        dim_model(cfg.dim_model),
        dim_ff(cfg.dim_ff),
        num_experts(cfg.moe_num_experts),
        num_experts_may_share(cfg.moe_num_experts),
        num_local_experts(cfg.moe_num_experts),
        top_k(cfg.moe_top_k),
        top_k_may_share(cfg.moe_top_k),
        norm_topk_prob(cfg.norm_topk_prob),
        routed_scaling_factor(cfg.routed_scaling_factor),
        topk_group(cfg.moe_topk_group),
        n_group(cfg.moe_n_group),
        dtype(cfg.dtype),
        parallel(parallel),
        dyn_shared(dyn_shared),
        world_size(ctx.world_size()),
        local_rank(ctx.rank()),
        router(ctx, dim_model, num_experts, "", 0, false, false, false, core::DistLayout::ROW, dtype) {
        std::cout << ">>>> FeedForward::impl::MOEImpl: " << std::endl;
    }
}; // end of class FeedForward::impl::MOEImpl

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
    const core::Tensor &input2d = input.ndim() == 2 ? input : input.view(shape2d);
    core::Tensor ret = pimpl->forward(ctx, input2d, true);
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
    core::Layer::load_state_dict(ctx, state_dict, prefix, allow_missing);
    int fuse_w_in = utils::get_int_env("CPM_FUSE_FF_IN", 0);
    auto normal_impl = dynamic_cast<impl::NormalImpl *>(pimpl.get());
    normal_impl->try_fuse_up_weights(ctx);
}

} // namespace nn
