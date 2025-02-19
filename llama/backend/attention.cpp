#include "backend/attention.h"
#include "backend/utils.h"
#include "backend/linear.h"
#include "backend/layernorm.h"
#include <bmengine/core/core.h>
#include <bmengine/functions/all.h>
#include "backend/transformer_buffer.h"
#include "backend/attention_kernel.h"
#include "backend/attention_base.h"
#include "backend/rotary_embedding.h"
#include "backend/dyn_batch_context.h"
#include "backend/rag_buffer_context.h"
#include <bmengine/logger/std_log_op.hpp>
#include "private/allocator.h"

namespace nn {

using namespace bmengine;
using model::ModelContext;
using bmengine::core::Tensor;
using bmengine::functions::concat_tensor;
using bmengine::functions::BinaryElementwiseOp;
using model::RagBufferContext;
using bmengine::logger::str_cat;
typedef std::vector<size_t> ShapeT;

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

    RotaryEmbedding rotary_embedding;

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
        rotary_embedding(ctx, cfg),
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

    virtual ~NormalImpl() = default;

    core::Tensor dynamic_batch_forward(model::ModelContext &ctx,
                                       const core::Tensor &hidden_q,
                                       const core::Tensor &position_or_bias,
                                       core::Tensor *output);

    Tensor attn_encode_group(model::ModelContext &ctx,
                             Tensor h_q_enc,
                             Tensor h_k_enc,
                             Tensor h_v_enc,
                             Tensor attn_value_enc);

    int get_split(model::ModelContext &ctx, size_t mem) {
        auto allocator = ctx.get_allocator();
        size_t free_memory = allocator->get_memory_limit() - allocator->used_memory();
        // const size_t mem_limit = 128 * 1024 * 1024 * sizeof(half)
        const size_t mem_limit = free_memory / 2;
        int n_split = 1;
        while (mem > mem_limit && (n_split * 2) <= num_kv_heads
               && (num_kv_heads % (n_split * 2) == 0)) {
            n_split *= 2;
            mem /= 2;
        }
        return n_split;
    }

    int get_event_level(const core::Context &ctx) {
        if (ctx.current_layer() == 1000 && ctx.active_device() == 0 && ctx.rank() == 0) {
            return 0;
        }
        return 2;
    }

    virtual core::Tensor forward(const core::Context &ctx,
                                 const core::Tensor &hidden_q,      // (batch?, len_q, dim_model)
                                 const core::Tensor &mask,          // (batch?, len_q, len_buf)  int8
                                 const core::Tensor &position_bias, // if relative (batch, num_head, len_q, len_buf) else if rotary (len_q)
                                 const core::Tensor &seqlens_q,     // (batch?, 1,)  int32
                                 const core::Tensor &seqlens_kv,    // (batch?, 1,) int32
                                 core::Tensor *past_k,              // (batch, num_heads, len_buf, dim_head)
                                 core::Tensor *past_v,              // (batch, num_heads, len_buf, dim_head)
                                 const core::Tensor *block_table,   // (batch, blocks_per_seq)
                                 const core::Tensor *placement,     // (batch?, len_q, ) int32
                                 core::Tensor *output) {
        if (seqlens_kv.numel() == 0) {
            core::EventScope event_scope(ctx, "Attention", 1);
            return forward_BHSD(ctx, hidden_q, mask, position_bias, past_k, past_v, placement);
        } else {
            core::EventScope event_scope(ctx, "Attention(Flash)", 1);
            // return forward_BSHD(ctx, hidden_q, position_bias, seqlens_q, seqlens_kv, past_k, past_v, block_table);
        }
    }

    core::Tensor forward_BHSD(const core::Context &ctx,
                              const core::Tensor &hidden_q,      // (batch?, len_q, dim_model)
                              const core::Tensor &mask,          // (batch?, len_q, len_buf) int8
                              const core::Tensor &position_bias, // if relative (batch, num_head, len_q, len_buf) else if rotary (len_q)
                              core::Tensor *past_k,              // (batch, num_heads, leb_buf, dim_head)
                              core::Tensor *past_v,              // (batch, num_heads, len_buf, dim_head)
                              const core::Tensor *placement) {   // (batch?, len_q,) int32
        int event_level = get_event_level(ctx);

        size_t batch = (mask.ndim() == 2) ? 1 : mask.size(0);
        uint32_t len_q = mask.size(-2);
        uint32_t len_buf = mask.size(-1);

        const core::Tensor &key_buf =
            past_k == nullptr ?
                ctx.tensor({batch, num_kv_heads, len_buf, dim_head}, dtype) :
                past_k->view({batch, num_kv_heads, len_buf, dim_head});
        const core::Tensor &val_buf =
            past_v == nullptr ?
                ctx.tensor({batch, num_kv_heads, len_buf, dim_head}, dtype) :
                past_v->view({batch, num_kv_heads, len_buf, dim_head});

        int active_dev = ctx.active_device();
        BM_ASSERT(active_dev == key_buf.device(), "Invalid past_k device");
        BM_ASSERT(active_dev == val_buf.device(), "invalid past_v device");
        if (placement != nullptr) {
            BM_ASSERT(active_dev == placement->device(), "Invalid placement device");
        }

        Tensor h_q = project_q(ctx, hidden_q); // (batch?, len_q, num_heads * dim_head)
        Tensor h_k = project_k(ctx, hidden_q); // (batch?, len_q, num_kv_heads * dim_head)
        Tensor h_v = project_v(ctx, hidden_q); // (batch?, len_q, num_kv_heads * dim_head)

        cudaStream_t stream = ctx.current_stream()->ptr;
        ctx.recordEvent("copy_to_buffer,K&V", event_level);
        h_k = h_k.view({batch, len_q, num_kv_heads, dim_head});
        h_v = h_v.view({batch, len_q, num_kv_heads, dim_head});
        kvcache::copy_to_buffer(num_kv_heads, len_q, len_buf, dim_head, placement, h_k, key_buf, stream);
        kvcache::copy_to_buffer(num_kv_heads, len_q, len_buf, dim_head, placement, h_v, val_buf, stream);

        // (batch, len_q, num_heads, dim_head) => (batch, num_heads, len_q, dim_head)
        ctx.recordEvent("transposeQ", event_level);
        h_q = transpose_2_1(ctx, h_q.view({batch, len_q, num_heads, dim_head}));
        h_q = h_q.view({batch, num_kv_heads, num_head_groups * len_q, dim_head});

        // Q * K
        ctx.recordEvent("Q*K", event_level);
        Tensor attn_score = gemm_transB.forward(
            ctx,
            h_q,    // ColMajor: (batch, num_kv_heads, dim_head, num_head_group * len_q)
            key_buf // ColMajor: (batch, num_kv_heads, len_buf, dim_head)T
        );          // (batch, num_kv_heads, num_head_groups * len_q, len_buf)

        // attn_softmax in-place update attn_score
        ctx.recordEvent("attn_softmax", event_level);
        const Tensor &pos_bias = pos_bias_type == "relative" ? position_bias : core::Tensor();
        Tensor attn_score_q = attn_score.view({batch, num_heads, len_q, len_buf});
        attn_softmax(ctx, attn_scale, attn_score_q, mask, pos_bias);

        // Score * V
        ctx.recordEvent("Score*V", event_level);
        Tensor attn_res = gemm_score_v(
            ctx,
            attn_score, // ColMajor: (batch, num_kv_heads, len_buf, num_head_groups * len_q)
            val_buf     // ColMajor: (batch, num_kv_heads, dim_head, len_buf)
        );              // (batch, num_kv_heads, num_head_groups * len_q, dim_head)

        // transpose: (batch, num_heads, len_q, dim_head) => (batch, len_q, num_heads, dim_head)
        ctx.recordEvent("transposeAV", event_level);
        Tensor attn_value_t = transpose_2_1(ctx, attn_res.view({batch, num_heads, len_q, dim_head}));
        ctx.recordEvent("End>transposeAV", event_level);

        ShapeT attn_value_shape = (mask.ndim() == 2) ?
                                      ShapeT({len_q, num_heads * dim_head}) :
                                      ShapeT({batch, len_q, num_heads * dim_head});
        return attn_out(ctx, attn_value_t.view(attn_value_shape)); // return (batch?, len_q, dim_model)
    }

}; // end of lcass Attention::impl::NormalImpl

Tensor Attention::impl::NormalImpl::dynamic_batch_forward(model::ModelContext &ctx,
                                                          const core::Tensor &hidden_q, // (group_len_q, dim_model)
                                                          const core::Tensor &position_bias,
                                                          core::Tensor *output) {
    model::DynBatchContext *dyn_batch = ctx.dyn_batch().get();
    cudaStream_t stream = ctx.current_stream()->ptr;
    size_t n_rep = num_head_groups;

    BM_ASSERT(ctx.rag_buffer(), "");
    int event_level = get_event_level(ctx);
    core::EventScope ev(ctx, "Attention(DynBatch)", 1);

    Tensor g_h_q;
    Tensor g_h_k;
    Tensor g_h_v;
    static int fuse_pkv = utils::get_int_env("CPM_FUSE_QKV", 0);
    static int fuse_v2_thres = utils::get_int_env("FUSE_V2_THRES", 8);
    if (linear_qkv.get() && (fuse_pkv == 1 || fuse_pkv == 2 && hidden_q.size(0) <= fuse_v2_thres)) {
        // fuse Q, K, V
        auto a = linear_qkv->forward(ctx, hidden_q); // (group_len_q, (num_heads + 2 * num_kv_heads) * dim_head)
        BM_ASSERT_EQ(a.size(-1), (num_heads + 2 * num_kv_heads) * dim_head, "");
        // TODO ...
    } else {
        g_h_q = project_q(ctx, hidden_q); // (group_len_q, num_heads * dim_head)
        g_h_k = project_k(ctx, hidden_q); // (group_len_q, num_kv_heads * dim_head)
        g_h_v = project_v(ctx, hidden_q); // (group_len_q, num_kv_heads * dim_head)

        if (q_norm) {
            g_h_q = q_norm->forward(ctx, g_h_q);
            g_h_k = k_norm->forward(ctx, g_h_k);
        }
        if (pos_bias_type == "rotary") {
            auto h_qk = rotary_embedding(ctx, position_bias, g_h_q, g_h_k);
            g_h_q = std::get<0>(h_qk);
            g_h_k = std::get<1>(h_qk);
        }
    }

    bool has_encode = !dyn_batch->ev_batch.empty();
    size_t num_enc = dyn_batch->e_placement.numel();
    size_t num_s = dyn_batch->s_placement.numel();
    std::cout << "num_enc=" << num_enc << ", num_s=" << num_s << ", all=" << g_h_q.size(0) << std::endl;
    BM_ASSERT_EQ(num_enc + num_s, g_h_q.size(0), "dim mismatch");
    Tensor attn_val_g = ctx.tensor({g_h_q.size(0), num_heads * dim_head}, dtype);

    Tensor attn_value_enc;
    if (has_encode) {
        attn_value_enc = attn_encode_group(ctx,
                                           g_h_q.slice_dim0(0, num_enc),
                                           g_h_k.slice_dim0(0, num_enc),
                                           g_h_v.slice_dim0(0, num_enc),
                                           attn_val_g.slice_dim0(0, num_enc));
    }

    if (num_s == 0) {
        return attn_out.forward(ctx, attn_val_g); // (group_len_q, dim_model)
    }

    auto ret = attn_out.forward(ctx, attn_val_g); // (group_len_q, dim_model)
    return ret;
}

Tensor Attention::impl::NormalImpl::attn_encode_group(model::ModelContext &ctx,
                                                      Tensor h_q_enc,
                                                      Tensor h_k_enc,
                                                      Tensor h_v_enc,
                                                      Tensor attn_value_enc) { // num_enc, num_heads * dim_head
    model::DynBatchContext *dyn_batch = ctx.dyn_batch().get();
    model::RagBufferContext *rag_buffer = ctx.rag_buffer().get();

    Tensor *past_k = ctx.buf_k(ctx.current_layer()); // (batch, num_heads, len_buf, dim_head)
    Tensor *past_v = ctx.buf_v(ctx.current_layer()); // (batch, num_heads, len_buf, dim_head)

    cudaStream_t stream = ctx.current_stream()->ptr;
    size_t n_rep = num_head_groups;
    int event_level = get_event_level(ctx);
    size_t num_enc = dyn_batch->e_placement.numel();
    attn_value_enc = attn_value_enc.view({num_enc, num_heads * dim_head});

    const Tensor *e_batch = ctx.identity(&dyn_batch->e_batch, "e_batch");
    const Tensor *e_placement = ctx.identity(&dyn_batch->e_placement, "e_placement");

    h_q_enc = h_q_enc.view({num_enc, num_heads, dim_head});
    h_k_enc = h_k_enc.view({num_enc, num_kv_heads, dim_head});
    h_v_enc = h_v_enc.view({num_enc, num_kv_heads, dim_head});

    BM_ASSERT(rag_buffer, "No rag buffer");

    size_t offset = 0;
    size_t batch_enc = dyn_batch->ev_batch.size();
    for (size_t i = 0; i < batch_enc; ++i) {
        int b = dyn_batch->ev_batch[i];
        size_t input_len = dyn_batch->ev_input_len[i];
        size_t full_input_len = dyn_batch->full_input_len[i]; // = input_len + cache_len
        size_t len_buf_b = dyn_batch->ev_len_buf[i];
        size_t len_buf = !rag_buffer ? past_v->size(ctx.is_BSHD() ? -3 : -2) : 0;

        // split current batch from group
        Tensor h_q = h_q_enc.slice_dim0_len(offset, input_len).view({input_len, num_heads, dim_head});
        Tensor h_k = h_k_enc.slice_dim0_len(offset, input_len).view({input_len, num_kv_heads, dim_head});
        Tensor h_v = h_v_enc.slice_dim0_len(offset, input_len).view({input_len, num_kv_heads, dim_head});
        Tensor placement = e_placement->slice_dim0_len(offset, input_len);

        auto ev_name = str_cat("Encode[", ctx.is_BSHD() ? "flash=True," : "", "heads=", num_heads, "]");
        core::EventScope ev_encode1(ctx, ev_name, event_level);

        Tensor key_buf;
        Tensor val_buf;
        Tensor mask = dyn_batch->encode_mask(ctx, i);
        Tensor v_t = attn_value_enc.slice_dim0_len(offset, input_len)
                         .view({input_len, num_heads, dim_head});
        if (rag_buffer->buf_k(b).is_quantized()) {
            BM_ASSERT(ctx.is_BSHD(), "flash attention only");
            // TODO tianx ...
        } else {
            size_t old_len = full_input_len - input_len;
            // key_buf = rag_buffer->buf_k(b).copy(ctx, ctx.current_layer(), h_k, placement, old_len);
            // val_buf = rag_buffer->buf_v(b).copy(ctx, ctx.current_layer(), h_v, placement, old_len);
            key_buf = rag_buffer->buf_k(b, ctx.current_layer());
            val_buf = rag_buffer->buf_v(b, ctx.current_layer());
            copy_to_buffer(num_kv_heads, input_len, len_buf_b, dim_head, &placement, h_k, key_buf, stream, ctx.is_BSHD());
            copy_to_buffer(num_kv_heads, input_len, len_buf_b, dim_head, &placement, h_v, val_buf, stream, ctx.is_BSHD());

            int custom_attn = utils::get_int_env("CPM_CUSTOM_SELF_ATTN", 0);
            if (custom_attn && num_head_groups == 8 && !ctx.is_BSHD()) {
                // multi_query_self_attention(ctx, h_q, key_buf, val_buf, mask, attn_scale, v_t, 0);
                // offset += input_len;
                // continue;
            }
        }

        if (ctx.is_BSHD()) {
            // TODO tianx ...
        }

        ctx.recordEvent("tranposeQ", event_level);
        h_q = transpose_2_1(ctx, h_q).view({num_kv_heads, n_rep * input_len, dim_head});
        std::cout << ">>>>>> len_buf_b: " << len_buf_b << ", len_buf: " << len_buf << std::endl;
        if (len_buf_b < len_buf) {
            ctx.recordEvent("CopyKV", event_level);
            // TODO tianx ...
        }

        const Tensor &pos_bias = pos_bias_type == "relative" ? dyn_batch->e_position_bias(ctx, i) : Tensor();
        Tensor attn_res = ctx.tensor({num_kv_heads, num_head_groups * input_len, dim_head}, dtype);
        // split attn_score(space: O(n^2)) to reduce memory usage
        size_t attn_score_memory = num_heads * input_len * len_buf_b * core::get_elem_size(dtype);
        int n_split = rag_buffer ? get_split(ctx, attn_score_memory) : 1;
        if (n_split == 1) {
            // Q * K
            ctx.recordEvent("Q*K", event_level);
            Tensor attn_score = gemm_transB.forward(ctx,
                                                    h_q,    // ColMajor: (num_kv_heads, dim_head, n_rep * input_len)
                                                    key_buf // ColMajor: (num_kv_heads, len_buf, dim_head)T
            );                                              // (num_kv_heads, n_rep * input_len, len_buf_b)

            // attn_softmax in-place update attn_score
            ctx.recordEvent("attn_softmax", event_level);
            Tensor attn_score_q = attn_score.view({num_heads, input_len, len_buf_b});
            attn_softmax(ctx, attn_scale, attn_score_q, mask, pos_bias);

            // Score * V
            ctx.recordEvent("Score*V", event_level);
            gemm_score_v(ctx,
                         attn_score, // ColMajor: (num_kv_heads, len_buf, num_head_groups * len_q)
                         val_buf,    // ColMajor: (num_kv_heads, dim_head, len_buf)
                         &attn_res   // (num_kv_heads, num_head_groups * len_q, dim_head)
            );
        } else {
            // TODO tianx ...
        }

        // transpose: (num_heads, len_q, dim_head) => (len_q, num_heads * dim_head)
        ctx.recordEvent("transposeAV", event_level);
        transpose_2_1(ctx, attn_res.view({num_heads, input_len, dim_head}), &v_t);

        offset += input_len;
    }
    return attn_value_enc;
}

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
        if (normal_impl->q_norm) {
            add_submodule("q_norm", normal_impl->q_norm.get());
            add_submodule("k_norm", normal_impl->k_norm.get());
        }
        pimpl.reset(normal_impl);
    }
}

Attention::~Attention() = default;

core::Tensor Attention::forward(const core::Context &ctx,
                                const core::Tensor &hidden_q,      // (len_q, dim_model)
                                const core::Tensor &mask,          // (len_q, len_buf)
                                const core::Tensor &position_bias, // if relative (num_head, len_q, len_buf) else if rotary (len_q)
                                const core::Tensor &seqlens_q,     // (batch?, 1)  int32
                                const core::Tensor &seqlens_kv,    // (batch?, 1,)  int32
                                const core::Tensor *c_past_k,      // (num_head, len_buf, dim_head)
                                const core::Tensor *c_past_v,      // (num_head, len_buf, dim_head)
                                const core::Tensor *block_table,   // (batch_size, blocks_per_seq)
                                const core::Tensor *placement,     // (batch?, len_q, ) int32
                                core::Tensor *output) {
    // ModelContext *m_ctx = dynamic_cast<ModelContext *>(const_cast<core::Context *>(&ctx));
    ModelContext *m_ctx = dynamic_cast<ModelContext *>(const_cast<core::Context *>(&ctx));
    if (m_ctx && m_ctx->dyn_batch()) {
        // impl::NormalImpl *p = dynamic_cast<impl::NormalImpl *>(pimpl.get());
        // return p->dynamic_batch_forward(*m_ctx, hidden_q, position_bias, output);
        return pimpl->dynamic_batch_forward(*m_ctx, hidden_q, position_bias, output);
    }
    // core::EventScope event_score(ctx, "Attention", 1);
    Tensor *past_k = const_cast<Tensor *>(c_past_k);
    Tensor *past_v = const_cast<Tensor *>(c_past_v);
    impl::NormalImpl *p = dynamic_cast<impl::NormalImpl *>(pimpl.get());
    return p->forward(ctx, hidden_q, mask, position_bias,
                      seqlens_q, seqlens_kv,
                      past_k, past_v,
                      block_table, placement, output);
}

core::Tensor Attention::dyn_rag_forward(model::ModelContext &ctx,
                                        const core::Tensor &inp,      // (grouped_len_q, dim_model)
                                        const core::Tensor &position, // (grouped_len_q)
                                        core::Tensor *output) {
    impl::NormalImpl *p = dynamic_cast<impl::NormalImpl *>(pimpl.get());
    return p->dynamic_batch_forward(ctx, inp, position, output);
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
    pimpl->on_load(ctx);
}

} // namespace nn
