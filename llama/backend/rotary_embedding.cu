#include "backend/rotary_embedding.h"
#include "backend/model_context.h"
#include "backend/dyn_batch_context.h"
#include <bmengine/logger/std_log_op.hpp>

namespace nn {

using namespace bmengine;
using bmengine::core::Tensor;

// gridDim (seq_len, batch, dim_model / 1024), blockDim (1024, 1, 1)
template <typename T, bool IsDynamic = false>
static __global__ void KERNEL_rotary_embedding(
    int dim_model,
    int dim_head,
    size_t hidden_stride,
    size_t pos_stride,
    const int32_t *__restrict__ pos, // (batch, seq_len)
    const T *__restrict__ in,        // (batch, seq_len, dim_model)
    T *__restrict__ out,             // (batch, seq_len, dim_model)
    float rope_theta,
    int max_position_embeddings,
    float scaling_factor) {
    int batch_id = blockIdx.y;
    int target_pos = pos[batch_id * pos_stride + blockIdx.x];
    int ith = blockIdx.z * blockDim.x + threadIdx.x;
    int col = ith % dim_head;
    size_t offset = batch_id * hidden_stride + blockIdx.x * dim_model;

    if constexpr (IsDynamic) {
        int seq_len = pos[batch_id * pos_stride + gridDim.x - 1];
        if (seq_len > max_position_embeddings) {
            rope_theta *= powf((scaling_factor * seq_len / max_position_embeddings) - (scaling_factor - 1.f),
                               dim_head / (dim_head - 2));
        }
    }
    if (ith >= dim_model) return;
    int half_dim = dim_head / 2;
    if (col < half_dim) {
        float freq = target_pos * powf(rope_theta, -float(col * 2) / dim_head);
        float cos_freq = cos(freq);
        float sin_freq = sin(freq);
        out[offset + ith] = in[offset + ith] * T(cos_freq) - in[offset + ith + half_dim] * T(sin_freq);
    } else {
        float freq = target_pos * powf(rope_theta, -float((col - half_dim) * 2) / dim_head);
        float cos_freq = cos(freq);
        float sin_freq = sin(freq);
        out[offset + ith] = in[offset + ith] * T(cos_freq) + in[offset + ith - half_dim] * T(sin_freq);
    }
}

// gridDim (seq_len, num_heads), blockDim (dim_head)
template <typename T>
static __global__ void KERNEL_rope_with_cache(
    const float *__restrict__ g_cos, // (seq_len, dim_head)
    const float *__restrict__ g_sin, // (seq_len, dim_head)
    const T *__restrict__ in,        // (seq_len, num_hedas, dim_head)
    T *__restrict__ g_out,           // (seq_len, num_heads, dim_head)
    uint32_t src_stride1,
    uint32_t src_stride2,
    uint32_t dst_stride1,
    uint32_t dst_stride2,
    bool neox_style = true) {
    int dim_head = blockDim.x;
    int col = threadIdx.x;
    int half_dim = dim_head / 2;

    size_t src_offset = size_t(blockIdx.x) * src_stride1 + blockIdx.y * src_stride2 + col;
    size_t dst_offset = size_t(blockIdx.x) * dst_stride1 + blockIdx.y * dst_stride2 + col;

    float cos_freq = g_cos[blockIdx.x * dim_head + col];
    float sin_freq = g_sin[blockIdx.x * dim_head + col];

    // neox_style
    float t;
    if (col < half_dim) {
        t = float(in[src_offset]) * cos_freq - float(in[src_offset + half_dim]) * sin_freq;
    } else {
        t = float(in[src_offset]) * cos_freq + float(in[src_offset - half_dim]) * sin_freq;
    }
    g_out[dst_offset] = t;
}

class RotaryEmbedding::impl {
public:
    class NormalImpl;
    int dim_head;
    float rope_theta;
    std::string rope_scaling_type;
    float scaling_factor;
    int max_position_embeddings;
    bool neox_style = true;

    impl(const core::Context &ctx, model::ModelConfig cfg) :
        dim_head(cfg.dim_head),
        rope_theta(cfg.rope_theta),
        rope_scaling_type(cfg.rope_cfg.type),
        scaling_factor(cfg.rope_cfg.factor),
        max_position_embeddings(cfg.max_position_embeddings) {
        if (cfg.qk_rope_head_dim > 0) {
            dim_head = cfg.qk_rope_head_dim;
        }
    }
    virtual ~impl() {
    }

    virtual std::tuple<core::Tensor, core::Tensor> forward(
        const core::Context &ctx,
        const core::Tensor &pos, // (batch, seq_len)
        const core::Tensor &q,   // (batch, seq_len, dim_model)
        const core::Tensor &k    // (batch, seq_len, dim_model)
        ) = 0;

    virtual Tensor rotate(const core::Context &ctx,
                          const core::Tensor &pos, // (batch, seq_len)
                          const core::Tensor &q,   // (batch, seq_len, dim_model)
                          core::Tensor *output = nullptr) {
        throw std::runtime_error("Unsupported");
    }

}; // end of class RotaryEmbedding::impl

class RotaryEmbedding::impl::NormalImpl : public RotaryEmbedding::impl {
public:
    NormalImpl(const core::Context &ctx, model::ModelConfig cfg) :
        impl(ctx, cfg) {
    }

    void rotate_with_cache(const core::Context &ctx,
                           const core::Tensor &cos, // (batch?, seq_len, dim_head)
                           const core::Tensor &sin, // (batch?, seq_len, dim_head)
                           const core::Tensor &q,   // (batch?, seq_len, dim_model) or (batch?, seq_len, num_heads, dim_head)
                           core::Tensor &out_q) {
        BM_ASSERT_EQ(core::DataType::kFloat, cos.dtype(), "dtype mismatch");
        BM_ASSERT_EQ(sin.shape(), cos.shape(), "shape mismatch");
        BM_ASSERT(q.ndim() <= cos.ndim() + 1, "Dim mismatch");
        BM_ASSERT(q.size(-1) % dim_head == 0, "dim_model mismatch");
        BM_ASSERT_EQ(q.size(0), cos.size(0), "batch or seq_len mismatch");
        if (cos.ndim() == 3) {
            BM_ASSERT_EQ(q.size(1), cos.size(1), "seq_len mismatch");
        }
        if (q.ndim() == cos.ndim() + 1) {
            BM_ASSERT_EQ(q.size(-1), dim_head, "dim_head mismatch");
        }

        uint32_t src_stride1 = q.stride(-2);
        uint32_t src_stride2 = dim_head;
        uint32_t dst_stride1 = out_q.stride(-2);
        uint32_t dst_stride2 = dim_head;

        if (q.ndim() == cos.ndim() + 1) {
            src_stride1 = q.stride(-1);
            src_stride2 = q.stride(-2);
            dst_stride1 = out_q.stride(-3);
            dst_stride2 = out_q.stride(-2);
        }

        size_t seq_len = cos.numel() / cos.size(-1);
        size_t num_heads = (q.ndim() == cos.ndim() + 1) ? q.size(-2) : (q.size(-1) / dim_head);
        if (ctx.is_layer(1000)) {
            std::cout << "seq_len: " << seq_len << std::endl;
            std::cout << "num_heads: " << num_heads << std::endl;
        }
        {
            dim3 gridDim(seq_len, num_heads);
            auto stream = ctx.current_stream()->ptr;
            BM_DTYPE_DISPATCH_HALF(q.dtype(), {
                KERNEL_rope_with_cache<<<gridDim, dim_head, 0, stream>>>(
                    cos.data<float>(),
                    sin.data<float>(),
                    q.data<scalar_t>(),
                    out_q.data<scalar_t>(),
                    src_stride1, src_stride2, dst_stride1, dst_stride2);
            });
            BM_CUDART_ASSERT(cudaGetLastError());
        }
    }

    Tensor rotate(const core::Context &ctx,
                  const core::Tensor &pos, // (batch, seq_len)
                  const core::Tensor &q,   // (batch, seq_len, dim_model)
                  core::Tensor *output) override {
        auto m_ctx = model::ModelContext::cast(ctx);
        auto out_q = output ? *output : ctx.tensor(q.size(), q.dtype());
        if (m_ctx && m_ctx->dyn_batch() && m_ctx->dyn_batch()->rope_cache.cos.numel() > 0) {
            auto &cos = m_ctx->dyn_batch()->rope_cache.cos;
            auto &sin = m_ctx->dyn_batch()->rope_cache.sin;
            rotate_with_cache(ctx, cos, sin, q, out_q);
            return out_q;
        }
        BM_ASSERT_EQ(q.ndim(), pos.ndim() + 1, "Dim mismatch");
        BM_ASSERT(q.size(-1) % dim_head == 0, "dim_model mismatch");
        BM_ASSERT_EQ(q.size(0), pos.size(0), "shape mismatch");
        if (pos.ndim() > 1) {
            BM_ASSERT_EQ(q.size(1), pos.size(1), "shape mismatch");
        }
        if (output) {
            BM_ASSERT_EQ(output->size(), q.size(), "shape mismatch");
        }

        int batch = (q.ndim() == 2) ? 1 : q.size(0);
        int pos_stride = (pos.ndim() == 1) ? 0 : pos.stride(0);
        int seq_len = q.size(-2);
        {
            int hidden_stride = (q.ndim() == 2) ? 0 : q.stride(0);
            int dim_model = q.size(-1);
            int threads = std::min(1024, round_up(dim_model, 32));
            dim3 gridDim(seq_len, batch, round_up(dim_model, threads) / threads);
            dim3 blockDim(threads, 1, 1);
            auto stream = ctx.current_stream()->ptr;
            BM_DTYPE_DISPATCH_FLOAT(q.dtype(), {
                auto kernel = KERNEL_rotary_embedding<scalar_t, false>;
                if (rope_scaling_type == "dynamic") {
                    kernel = KERNEL_rotary_embedding<scalar_t, true>;
                }
                kernel<<<gridDim, blockDim, 0, stream>>>(
                    dim_model, dim_head, hidden_stride, pos_stride,
                    pos.data<int32_t>(), q.data<scalar_t>(), out_q.data<scalar_t>(),
                    rope_theta, max_position_embeddings, scaling_factor);
            });
            BM_CUDART_ASSERT(cudaGetLastError());
        }
        return out_q;
    }

    std::tuple<core::Tensor, core::Tensor> forward(const core::Context &ctx,
                                                   const core::Tensor &pos, // (batch, seq_len)
                                                   const core::Tensor &q,   // (batch, seq_len, dim_model)
                                                   const core::Tensor &k) { // (batch, seq_len, dim_model)
        Tensor out_q = rotate(ctx, pos, q, nullptr);
        Tensor out_k = rotate(ctx, pos, k, nullptr);
        return std::make_tuple(out_q, out_k);
    }

}; // end of class RotaryEmbedding::impl::NormalImpl

RotaryEmbedding::RotaryEmbedding(const core::Context &ctx, model::ModelConfig cfg) {
    if (cfg.rope_cfg.type == "yarn") {
        // pimpl = std::make_unique<impl::YarmImpl>(ctx, cfg);
    } else {
        pimpl = std::make_unique<impl::NormalImpl>(ctx, cfg);
        if (cfg.model_type == "cohere") {
            pimpl->neox_style = false;
        }
    }
}

RotaryEmbedding::~RotaryEmbedding() {
}

std::tuple<core::Tensor, core::Tensor> RotaryEmbedding::forward(const core::Context &ctx,
                                                                const core::Tensor &pos, // (batch?, seq_len)
                                                                const core::Tensor &q,   // (batch?, seq_len, dim_model)
                                                                const core::Tensor &k) { // (batch?, seq_len, dim_model)
    core::EventScope ev(ctx, "RotaryEmbedding2", 3);
    return pimpl->forward(ctx, pos, q, k);
}

} // namespace nn
