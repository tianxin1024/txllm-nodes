#include <assert.h>
#include "backend/embedding.h"
#include <bmengine/core/core.h>
#include <bmengine/functions/gemm.h>

// gridDim (seq_len, dim_model / 1024, 1), blockDim (1024, 1, 1)
template <typename T>
__global__ void BM_KERNEL(embedding)(
    int begin, int end, size_t dim_model, float scale,
    const int32_t *__restrict__ idx, // (batch, seq_len)
    const T *__restrict__ weight,    // (vocab_size, dim_model)
    T *__restrict__ out              // (batch, seq_len, dim_model)
) {
    int target_id = idx[blockIdx.x];
    int col = blockIdx.y * blockDim.x + threadIdx.x;
    size_t offset = blockIdx.x * dim_model;
    bool in_range = target_id >= begin && target_id < end;
    target_id -= begin;
    if (col < dim_model) {
        out[offset + col] = in_range ? T(float(weight[size_t(target_id) * dim_model + col]) * scale) : T(0.);
    }
}

// gridDim (seq_len, dim_model / 1024, batch),   blockDim (1024, 1, 1)
template <typename T>
static __global__ void BM_KERNEL(rotary_embedding)(
    int dim_model,
    int batch_stride,
    int emb_stride,
    const int32_t *__restrict__ pos, // (batch, seq_len)
    const T *__restrict__ emb,       // (batch, seq_len, dim_model)
    T *__restrict__ out              // (batch, seq_len, dim_model)
) {
    int batch_id = blockIdx.z;
    int target_pos = pos[batch_id * batch_stride + blockIdx.x] * 16;
    int col = blockIdx.y * blockDim.x + threadIdx.x;
    int offset = batch_id * emb_stride + blockIdx.x * dim_model;

    int half_dim_model = dim_model / 2;
    if (col < half_dim_model) {
        float freq = target_pos * powf(10000, -float(col * 2) / dim_model);
        float cos_freq = cos(freq);
        float sin_freq = sin(freq);
        T rorate_v = -emb[offset + col + half_dim_model];
        out[offset + col] = emb[offset + col] * T(cos_freq) + rorate_v * T(sin_freq);
    } else if (col < dim_model) {
        float freq = target_pos * powf(10000, -float((col - half_dim_model) * 2) / dim_model);
        float cos_freq = cos(freq);
        float sin_freq = sin(freq);
        T rotate_v = emb[offset + col - half_dim_model];
        out[offset + col] = emb[offset + col] * T(cos_freq) + rotate_v * T(sin_freq);
    }
}

static __host__ void embedding(
    const core::Tensor &idx,
    const core::Tensor &weight,
    const core::Tensor &out,
    int begin,
    int end,
    float scale,
    cudaStream_t stream) {
    int seq_len = idx.numel();
    int dim_model = weight.size(1);
    int threads = round_up_thread(dim_model);
    dim3 gridDim(seq_len, round_up(dim_model, threads) / threads);
    dim3 blockDim(threads);

    BM_DTYPE_DISPATCH_FLOAT(weight.dtype(), {
        BM_KERNEL(embedding)<scalar_t><<<gridDim, blockDim, 0, stream>>>(
            begin, end, dim_model, scale, idx.data<int32_t>(), weight.data<scalar_t>(), out.data<scalar_t>());
    });
    BM_CUDART_ASSERT(cudaGetLastError());
}

static __host__ core::Tensor rotary_embedding(
    const core::Context &ctx,
    const core::Tensor &pos, // (batch, seq_len)
    const core::Tensor &emb  // (batch, seq_len, dim_model)
) {
    int batch = pos.ndim() == 1 ? 1 : pos.size(0);
    int batch_stride = pos.ndim() == 1 ? 0 : pos.stride(0);
    int emb_stride = emb.ndim() == 2 ? 0 : emb.stride(0);
    int seq_len = emb.size(-2);
    int dim_model = emb.size(-1);
    int threads = std::min(1024, round_up(dim_model, 32));
    dim3 gridDim(seq_len, round_up(dim_model, threads) / threads, batch);
    dim3 blockDim(threads, 1, 1);
    auto stream = ctx.current_stream()->ptr;
    auto out = ctx.tensor(emb.size(), emb.dtype());
    BM_DTYPE_DISPATCH_FLOAT(emb.dtype(), {
        BM_KERNEL(rotary_embedding)<scalar_t><<<gridDim, blockDim, 0, stream>>>(
            dim_model,
            batch_stride,
            emb_stride,
            pos.data<int32_t>(),
            emb.data<scalar_t>(),
            out.data<scalar_t>());
    });
    BM_CUDART_ASSERT(cudaGetLastError());
    return out;
}

class Embedding::impl {
public:
    core::Tensor weight;
    unsigned int dim_model;
    unsigned int vocab_size;
    core::DataType dtype;

    float scale_factor;
    functions::Gemm local_gemm;
    functions::Gemm local_gemm_alpha;
    impl(
        const core::Context &ctx,
        unsigned int vocab_size,
        unsigned int dim_model,
        bool scale_weights,
        core::DataType dtype) :
        weight(ctx.parameter({vocab_size, dim_model}, dtype)),
        dim_model(dim_model),
        vocab_size(vocab_size),
        dtype(dtype),
        scale_factor(scale_weights ? 1.0 / sqrtf(dim_model) : 1.0),
        local_gemm(ctx, dtype, false, true),
        local_gemm_alpha(ctx, dtype, false, true, scale_factor) {
        if (ctx.high_precision() >= 1) {
            local_gemm.set_compute_type(CUBLAS_COMPUTE_32F);
            local_gemm_alpha.set_compute_type(CUBLAS_COMPUTE_32F);
        }
    }

    core::Tensor forward(
        const core::Context &ctx,
        const core::Tensor &ids,    // (batch, seq_len)
        const core::Tensor &ids_sub // (batch, seq_len)
    ) {
        BM_ASSERT(ids.dtype() == core::DataType::kInt32, "ids dtype mismatch");
        BM_ASSERT(ids.ndim() == 1 || ids.ndim() == 2, "ids must be 1d or 2d");

        auto shape = ids.size();
        shape.emplace_back(dim_model);
        core::Tensor ret = ctx.tensor(shape, dtype);

        embedding(ids, weight, ret, 0, weight.size(0), scale_factor, ctx.current_stream()->ptr);
        return rotary_embedding(ctx, ids_sub, ret);
    }

}; // end of class Embedding::impl

Embedding::Embedding(const core::Context &ctx, int dim_model, int vocab_size, bool scale_weights, core::DataType dtype) :
    pimpl(new impl(ctx, vocab_size, dim_model, scale_weights, dtype)), core::Layer() {
    add_parameter("weight", pimpl->weight);
}

Embedding::~Embedding() = default;

core::Tensor Embedding::forward(
    const core::Context &ctx,
    const core::Tensor &ids,    // (seq_len)
    const core::Tensor &ids_sub // (seq_len)

) {
    return pimpl->forward(ctx, ids, ids_sub);
}
