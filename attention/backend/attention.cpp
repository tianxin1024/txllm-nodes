#include <bmengine/core/core.h>
#include <bmengine/functions/all.h>
#include "backend/attention.h"
#include "backend/linear.h"
#include "backend/rotary_embedding.h"

using namespace bmengine;
using bmengine::core::DataType;
using bmengine::core::DistLayout;

class Attention::impl {
public:
    class NormalImpl;
    impl() = default;
    virtual ~impl() = default;
    impl(const impl &) = delete;
    impl(impl &&) = default;

}; // end of class Attention::impl

class Attention::impl::NormalImpl : public Attention::impl {
public:
    unsigned int dim_model;
    unsigned int dim_head;
    unsigned int num_heads;
    unsigned int num_kv_heads;
    unsigned int num_head_groups;
    core::DataType dtype{DataType::kHalf};
    bool parallel;

    std::string pos_bias_type;
    float attn_scale;
    model::QuantConfig quant_kv;
    bool scale_weights;
    bool weight_transposed;
    float rope_theta;

    Linear project_q, project_k, project_v;
    Linear attn_out;

    nn::RotaryEmbedding rotary_embedding;

    // fuse project_q, project_k and project_v
    std::unique_ptr<Linear> linear_qkv;
    // std::unique_ptr<LayerNorm> q_norm;
    // std::unique_ptr<LayerNorm> k_norm;

    functions::Gemm gemm_attn;
    functions::Gemm gemm_transB;
    functions::Gemm gemm_score_v;
    functions::Transpose transpose;

    // FlashDecoding flash_decoding;

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
        // quant_kv(as_quant_kv(quant)),
        scale_weights(cfg.scale_weights),
        weight_transposed(cfg.weight_transposed),
        rotary_embedding(ctx, cfg),
        rope_theta(cfg.rope_theta),
        project_q(ctx, dim_model, dim_head * num_heads, "", quant, scale_weights, weight_transposed, parallel, DistLayout::COLUMNAR, dtype),
        project_k(ctx, dim_model, dim_head * num_kv_heads, "", quant_kv, scale_weights, weight_transposed, parallel, num_kv_heads > 1 ? DistLayout::COLUMNAR : DistLayout::REPLICATED, dtype),
        project_v(ctx, dim_model, dim_head * num_kv_heads, "", quant_kv, scale_weights, weight_transposed, parallel, num_kv_heads > 1 ? DistLayout::COLUMNAR : DistLayout::REPLICATED, dtype),
        attn_out(ctx, dim_head * num_heads, dim_model, "", quant, scale_weights, weight_transposed, parallel, DistLayout::ROW, dtype),
        gemm_attn(ctx, dtype, true, true),
        gemm_transB(ctx, dtype, false, true),
        gemm_score_v(ctx, dtype, false, false),
        transpose(ctx) {
    }
};

Attention::Attention(const core::Context &ctx, model::ModelConfig cfg, model::QuantConfig quant_cfg, bool parallel) :
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
