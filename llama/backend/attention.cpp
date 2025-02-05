#include "backend/attention.h"
#include "backend/utils.h"
#include "backend/linear.h"
#include "backend/layernorm.h"
#include <bmengine/core/core.h>
#include <bmengine/functions/all.h>

namespace nn {

using namespace bmengine;

class Attention::impl {
public:
    class NormalImpl;
    impl() = default;
    virtual ~impl() = default;
    impl(const impl &) = delete;
    impl(impl &&) = delete;

    virtual core::Tensor dynamic_batch_forward(model::ModelContext &ctx,
                                               const core::Tensor &hidden_q,
                                               const core::Tensor &position_or_bias,
                                               core::Tensor *output) {
        throw std::runtime_error("Unsupported");
    }

    virtual void on_load(const core::Context &ctx) {
    }
}; // end of class Attention::impl

class Attention::impl::NormalImpl : public Attention::impl {
public:
    unsigned int dim_model;
    unsigned int dim_head;
    unsigned int num_heads;
    unsigned int num_kv_heads;
    unsigned int num_head_groups;
    core::DataType dtype{core::DataType::kHalf};
    bool parallel;

    std::string pos_bias_type;
    float attn_scale;
    model::QuantConfig quant_kv;
    bool scale_weights;
    bool weight_transposed;
    float rope_theta;

    Linear project_q, project_k, project_v;
    Linear attn_out;

    // fuse project_q, project_k and project_v
    std::unique_ptr<Linear> linear_qkv;
    std::unique_ptr<LayerNorm> q_norm;
    std::unique_ptr<LayerNorm> k_norm;

    functions::Gemm gemm_attn;
    functions::Gemm gemm_transB;
    functions::Gemm gemm_score_v;
    functions::Transpose transpose;

    int max_shared_memory;

    static model::QuantConfig as_quant_kv(model::QuantConfig quant) {
        if (quant.quant_weight_kv == 0) {
            quant.quant_type = model::QuantType::NoQuant;
        }
        return quant;
    }

    NormalImpl(const core::Context &ctx, model::ModelConfig cfg, model::QuantConfig quant, bool parallel) :
        dim_model(cfg.dim_model),
        dim_head(cfg.dim_head),
        num_heads(cfg.num_heads),
        num_kv_heads(cfg.num_kv_heads),
        num_head_groups(num_heads / num_kv_heads),
        dtype(cfg.dtype),
        parallel(parallel),
        pos_bias_type(cfg.pos_bias_type),
        attn_scale(1. / sqrtf(dim_head)),
        quant_kv(as_quant_kv(quant)),
        scale_weights(cfg.scale_weights),
        weight_transposed(cfg.weight_transposed),
        rope_theta(cfg.rope_theta),
        project_q(ctx, dim_model, dim_head * num_heads, "", quant, scale_weights, weight_transposed, parallel, core::DistLayout::COLUMNAR, dtype),
        project_k(ctx, dim_model, dim_head * num_kv_heads, "", quant_kv, scale_weights, weight_transposed, parallel, num_kv_heads > 1 ? core::DistLayout::COLUMNAR : core::DistLayout::REPLICATED, dtype),
        project_v(ctx, dim_model, dim_head * num_kv_heads, "", quant_kv, scale_weights, weight_transposed, parallel, num_kv_heads > 1 ? core::DistLayout::COLUMNAR : core::DistLayout::REPLICATED, dtype),
        attn_out(ctx, dim_head * num_heads, dim_model, "", quant, scale_weights, weight_transposed, parallel, core::DistLayout::ROW, dtype),
        gemm_attn(ctx, dtype, true, true),
        gemm_transB(ctx, dtype, false, true),
        gemm_score_v(ctx, dtype, false, false),
        transpose(ctx) {
        if (cfg.model_type == "qwen2" || cfg.model_type == "qwen2_moe") {
            project_q.set_has_bias(true);
            project_k.set_has_bias(true);
            project_v.set_has_bias(true);
        }
        if (cfg.use_qk_norm) {
            q_norm = std::make_unique<LayerNorm>(ctx, dim_head * num_heads, false, cfg.eps, 1, dtype, num_heads);
            k_norm = std::make_unique<LayerNorm>(ctx, dim_head * num_kv_heads, false, cfg.eps, 1, dtype, num_kv_heads);
        }
        if (parallel) {
            if (ctx.high_precision() >= 2) {
                // use float to reduce sum
                attn_out.set_output_type(core::DataType::kFloat);
            }
            int ws = ctx.world_size();
            BM_ASSERT(num_heads % ws == 0, "num_heads must be dividable by world_size");
            BM_ASSERT(num_kv_heads % ws == 0, "num_kv_heads must be dividable by world_size");
            this->num_heads = num_heads / ctx.world_size();
            this->num_kv_heads = num_kv_heads / ctx.world_size();
        }
        max_shared_memory = ctx.get_max_shared_memory();
    }

}; // end of lcass Attention::impl::NormalImpl

Attention::Attention(const core::Context &ctx,
                     model::ModelConfig cfg,
                     model::QuantConfig quant_cfg,
                     bool parallel) :
    core::Layer() {
    impl::NormalImpl *normal_impl = nullptr;
    if (cfg.kv_lora_rank > 0) {
        // pimpl.reset(impl::create_mla_impl(ctx, cfg, quant_cfg));
        // pimpl->add_submodules(this);
    } else {
        normal_impl = new impl::NormalImpl(ctx, cfg, quant_cfg, parallel);
    }
}

Attention::~Attention() = default;

core::Tensor Attention::dyn_rag_forward(model::ModelContext &ctx,
                                        const core::Tensor &inp,      // (grouped_len_q, dim_model)
                                        const core::Tensor &position, // (grouped_len_q)
                                        core::Tensor *output) {
    return pimpl->dynamic_batch_forward(ctx, inp, position, output);
}

void Attention::load_state_dict(const core::Context &ctx,
                                const std::map<std::string, const core::Tensor> &state_dict,
                                const std::string &prefix,
                                bool allow_missing) {
    core::Layer::load_state_dict(ctx, state_dict, prefix, allow_missing);
    int fuse_qkv = utils::get_int_env("CPM_FUSE_QKV", 0);
    impl::NormalImpl *p = dynamic_cast<impl::NormalImpl *>(pimpl.get());
    if (fuse_qkv && p) {
        // auto a = Linear::fuse(ctx, p->project_q, p->project_k, p->project_v);
        // p->linear_qkv = std::unique_ptr<Linear>(a);
    }
    // pimpl->on_load(ctx);
}

} // namespace nn
