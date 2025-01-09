#include <bmengine/core/core.h>
#include <bmengine/functions/all.h>
#include "backend/attention.h"
#include "backend/linear.h"
#include "backend/rotary_embedding.h"
#include "backend/model_context.h"
#include "backend/transformer_buffer.h"
#include "backend/attention_kernel.h"

using namespace bmengine;
using bmengine::core::DataType;
using bmengine::core::DistLayout;
using model::ModelContext;

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

    virtual core::Tensor forward(const core::Context &ctx,
                                 const core::Tensor &hidden_q,      // (batch?, len_q, dim_model)
                                 const core::Tensor &mask,          // (batch?, len_q, len_buf) int8
                                 const core::Tensor &position_bias, // if relative (batch, num_head, len_q, len_buf) else if rotary (len_q)
                                 const core::Tensor &seqlens_q,     // (batch?, 1) int32
                                 const core::Tensor &seqlens_kv,    // (batch?, 1)  int32
                                 core::Tensor *past_k,              // (batch, num_heads, len_buf, dim_head)
                                 core::Tensor *past_v,              // (batch, num_heads, len_buf, dim_head)
                                 const core::Tensor *block_table,   // (batch, blocks_per_seq)
                                 const core::Tensor *placement,     // (batch? , len_q)
                                 core::Tensor *output) {
        if (seqlens_kv.numel() == 0) {
            core::EventScope event_scope(ctx, "Attention", 1);
            return forward_BHSD(ctx, hidden_q, mask, position_bias, past_k, past_v, placement);
        }
    }

    int get_event_level(const core::Context &ctx) {
        if (ctx.current_layer() == 1000 && ctx.active_device() == 0 && ctx.rank() == 0) {
            return 0;
        }
        return 2;
    }

    core::Tensor forward_BHSD(const core::Context &ctx,
                              const core::Tensor &hidden_q,      // (batch?, len_q, dim_model)
                              const core::Tensor &mask,          // (batch?, len_q, len_buf) int8
                              const core::Tensor &position_bias, // if relative (batch, num_head, len_q, len_buf) else if rotary (len_q)
                              core::Tensor *past_k,              // (batch, num_heads, len_buf, dim_head)
                              core::Tensor *past_v,              // (batch, num_heads, len_buf, dim_head)
                              const core::Tensor *placement) {
        int event_level = get_event_level(ctx);

        size_t batch = (mask.ndim() == 2) ? 1 : mask.size(0);
        uint32_t len_q = mask.size(-2);
        uint32_t len_buf = mask.size(-1);

        // std::cout << "past_k info: " << past_k->info() << std::endl;

        const core::Tensor &key_buf =
            past_k == nullptr ? ctx.tensor({batch, num_kv_heads, len_buf, dim_head}, dtype) :
                                past_k->view({batch, num_kv_heads, len_buf, dim_head});
        const core::Tensor &val_buf =
            past_v == nullptr ? ctx.tensor({batch, num_kv_heads, len_buf, dim_head}, dtype) :
                                past_v->view({batch, num_kv_heads, len_buf, dim_head});

        int active_dev = ctx.active_device();
        BM_ASSERT(active_dev == key_buf.device(), "Invalid past_k device");
        BM_ASSERT(active_dev == val_buf.device(), "Invalid past_v device");
        if (placement != nullptr) {
            BM_ASSERT(active_dev == placement->device(), "Invalid placement device");
        }

        core::Tensor h_q = project_q(ctx, hidden_q); // (batch?, len_q, num_heads * dim_head)
        core::Tensor h_k = project_k(ctx, hidden_q); // (batch?, len_q, num_kv_heads * dim_head)
        core::Tensor h_v = project_v(ctx, hidden_q); // (batch?, len_q, num_kv_heads * dim_head)

        if (pos_bias_type == "rotary") {
            ctx.recordEvent("rotary", event_level);
            auto h_qk = rotary_embedding(ctx, position_bias, h_q, h_k);
            h_q = std::get<0>(h_qk);
            h_k = std::get<1>(h_qk);
        }

        cudaStream_t stream = ctx.current_stream()->ptr;
        ctx.recordEvent("copy_to_buffer,K&V", event_level);
        h_k = h_k.view({batch, len_q, num_kv_heads, dim_head});
        h_v = h_v.view({batch, len_q, num_kv_heads, dim_head});
        kvcache::copy_to_buffer(num_kv_heads, len_q, len_buf, dim_head, placement, h_k, key_buf, stream);
        kvcache::copy_to_buffer(num_kv_heads, len_q, len_buf, dim_head, placement, h_v, val_buf, stream);

        // (batch, len_q, num_heads, dim_head) => (batch, num_heads, len_q, dim_head)
        ctx.recordEvent("transposeQ", event_level);
        h_q = bmengine::functions::transpose_2_1(ctx, h_q.view({batch, len_q, num_heads, dim_head}));
        h_q = h_q.view({batch, num_kv_heads, num_head_groups * len_q, dim_head});

        // Q * K
        ctx.recordEvent("Q*K", event_level);
        core::Tensor attn_score = gemm_transB.forward(
            ctx,
            h_q,    // ColMajor: (batch, num_kv_heads, dim_head, num_head_groups * len_q)
            key_buf // ColMajor: (batch, num_kv_heads, len_buf, dim_head)T
        );          // (batch, num_kv_heads, num_head_groups * len_q, len_buf)

        // attn_softmax in-place update attn_score
        ctx.recordEvent("attn_softmax", event_level);
        const core::Tensor &pos_bias = pos_bias_type == "relative" ? position_bias : core::Tensor();
        core::Tensor attn_score_q = attn_score.view({batch, num_heads, len_q, len_buf});
        nn::attn_softmax(ctx, attn_scale, attn_score_q, mask, pos_bias);
    }

}; // end of class Attention::impl::NormalImp

Attention::Attention(const core::Context &ctx, model::ModelConfig cfg, model::QuantConfig quant_cfg, bool parallel) :
    core::Layer() {
    impl::NormalImpl *normal_impl = nullptr;
    std::cout << ">>>>>>>>>>>>>>>>> cfg.kv_lora_rank: " << cfg.kv_lora_rank << std::endl;
    if (cfg.kv_lora_rank > 0) {
        // pimpl.reset(impl::create_mla_impl(ctx, cfg, quant_cfg));
        // pimpl->add_submodules(this);
    } else {
        normal_impl = new impl::NormalImpl(ctx, cfg, quant_cfg, parallel);
    }

    if (normal_impl) {
        add_submodule("project_q", normal_impl->project_q);
        add_submodule("project_k", normal_impl->project_k);
        add_submodule("project_v", normal_impl->project_v);
        add_submodule("attn_out", normal_impl->attn_out);
        // gemm has no weight; add only for set prefix
        add_submodule("gemm_attn", normal_impl->gemm_attn);
        add_submodule("gemm_transB", normal_impl->gemm_transB);
        if (ctx.high_precision() >= 1) {
            normal_impl->gemm_attn.set_compute_type(CUBLAS_COMPUTE_32F);
            normal_impl->gemm_transB.set_compute_type(CUBLAS_COMPUTE_32F);
        }
        // if (normal_impl->q_norm) {
        //     add_submodule("q_norm", normal_impl->q_norm.get());
        //     add_submodule("k_norm", normal_impl->k_norm.get());
        // }
        pimpl.reset(normal_impl);
    }
}

Attention::~Attention() = default;

core::Tensor Attention::forward(const core::Context &ctx,
                                const core::Tensor &hidden_q,      // (len_q, dim_model)
                                const core::Tensor &mask,          // (len_q, len_buf)
                                const core::Tensor &position_bias, // if relative (num_head, len_q, len_buf) else if rotary (len_q)
                                const core::Tensor &seqlens_q,     // (batch?, 1,)  int32
                                const core::Tensor &seqlens_kv,    // (batch?, 1,)  int32
                                const core::Tensor *c_past_k,      // (num_head, len_buf, dim_head)
                                const core::Tensor *c_past_v,      // (num_head, len_buf, dim_head)
                                const core::Tensor *block_table,   // (batch_size, block_per_seq)
                                const core::Tensor *placement,     // (batch?, len_q) int32
                                core::Tensor *output) {
    ModelContext *m_ctx = dynamic_cast<ModelContext *>(const_cast<core::Context *>(&ctx));
    // if (m_ctx && m_ctx->dyn_batch()) {
    //     return pimpl->dynamic_batch_forward(*m_ctx, hidden_q, position_bias, output);
    // }

    core::Tensor *past_k = const_cast<core::Tensor *>(c_past_k);
    core::Tensor *past_v = const_cast<core::Tensor *>(c_past_v);

    impl::NormalImpl *p = dynamic_cast<impl::NormalImpl *>(pimpl.get());
    return p->forward(ctx, hidden_q, mask, position_bias, seqlens_q, seqlens_kv,
                      past_k, past_v, block_table, placement, output);
    std::cout << "-========== code line ====================" << std::endl;
}
