#include "backend/beam_util.h"
#include <bmengine/functions/reduce.cuh>
#include <bmengine/logger/std_log_op.hpp>
#include <cub/cub.cuh>

namespace beam_utility {

// gridDim (batch, 1, 1),  blockDim (1024, 1, 1)
template <typename T>
static __global__ void BM_KERNEL(log_softmax_bias_without_temperature)(
    size_t len_vocab,
    const float *__restrict__ bias, // (batch)
    const T *__restrict__ logits,   // (batch, len_vocab)
    T *__restrict__ out) {          // (batch, len_vocab)
    size_t offset = blockIdx.x * len_vocab;
    float local_max = -1e20;
    for (int i = threadIdx.x; i < len_vocab; i += blockDim.x) {
        local_max = fmaxf(local_max, logits[offset + i]);
    }
    local_max = functions::blockReduceMax<float>(local_max);
    float local_sum = 1e-20;
    for (int i = threadIdx.x; i < len_vocab; i += blockDim.x) {
        local_sum += expf((float)logits[offset + i] - local_max);
    }
    local_sum = functions::blockReduceSum<float>(local_sum);
    float b = bias[blockIdx.x];
    for (int i = threadIdx.x; i < len_vocab; i += blockDim.x) {
        out[offset + i] = (float)logits[offset + i] - local_max - logf(local_sum) + b;
    }
}

core::Tensor log_softmax_bias(const core::Context &ctx,
                              const core::Tensor &logits, // half (batch, dim_logits)
                              const core::Tensor &bias) { // float32 (batch)
    BM_ASSERT(logits.ndim() >= 2, "logits must be 2 or 3 dimensional");
    size_t dim_logits = logits.size(-1);
    int threads = round_up_thread(dim_logits);
    dim3 gridDim(logits.numel() / dim_logits);
    dim3 blockDim(threads);
    auto stream = ctx.current_stream()->ptr;
    auto out = ctx.tensor(logits.size(), logits.dtype());

    BM_DTYPE_DISPATCH_FLOAT(logits.dtype(), {
        BM_KERNEL(log_softmax_bias_without_temperature)<scalar_t><<<gridDim, blockDim, 0, stream>>>(
            dim_logits, bias.data<float>(), logits.data<scalar_t>(), out.data<scalar_t>());
    });
    BM_CUDART_ASSERT(cudaGetLastError());
    return out;
}

// gridDim (batch, 1, 1),   blockDim (1024, 1, 1)
template <typename T>
static __global__ void BM_KERNEL(log_softmax_bias)(
    size_t len_vocab,
    float temperature,
    const int32_t *__restrict__ bias, // (batch)
    const T *__restrict__ logits,     // (batch, len_vocab)
    T *__restrict__ out) {            // (batch, len_vocab)
    size_t offset = blockIdx.x * len_vocab;
    float local_max = -1e20;
    for (int i = threadIdx.x; i < len_vocab; i += blockDim.x) {
        local_max = fmaxf(local_max, logits[offset + i]);
    }
    local_max = functions::blockReduceMax<float>(local_max);
    float local_sum = 1e-20;
    for (int i = threadIdx.x; i < len_vocab; i += blockDim.x) {
        local_sum += expf(((float)logits[offset + i] - local_max) / temperature);
    }
    local_sum = functions::blockReduceSum<float>(local_sum);
    float b = bias[blockIdx.x];
    for (int i = threadIdx.x; i < len_vocab; i += blockDim.x) {
        out[offset + i] =
            ((float)logits[offset + i] - local_max) / temperature - logf(local_sum) + b;
    }
}

void log_softmax_bias(const core::Context &ctx,
                      const core::Tensor &logits,
                      const core::Tensor &bias,
                      float temperature,
                      core::Tensor *out) {
    BM_ASSERT(logits.ndim() >= 2, "logits must be 2 or 3 dimensional");
    BM_ASSERT_EQ(logits.shape(), out->shape(), "logits and out has different shape");

    size_t dim_logits = logits.size(-1);
    int threads = round_up_thread(dim_logits);
    dim3 gridDim(logits.numel() / dim_logits);
    dim3 blockDim(threads);
    auto stream = ctx.current_stream()->ptr;

    BM_DTYPE_DISPATCH_FLOAT(logits.dtype(), {
        BM_KERNEL(log_softmax_bias)<scalar_t><<<gridDim, blockDim, 0, stream>>>(
            dim_logits,
            temperature,
            bias.data<int32_t>(),
            logits.data<scalar_t>(),
            out->mutable_data<scalar_t>());
    });
    BM_CUDART_ASSERT(cudaGetLastError());
}

core::Tensor log_softmax_bias(const core::Context &ctx,
                              const core::Tensor &logits, // half (batch, dim_logits)
                              const core::Tensor &bias,   // float32 (batch)
                              float temperature) {
    if (temperature == 0.0) {
        return std::move(log_softmax_bias(ctx, logits, bias));
    }

    auto out = ctx.tensor(logits.size(), logits.dtype());
    log_softmax_bias(ctx, logits, bias, temperature, &out);
    return std::move(out);
}

// gridDim(N / 1024, 1, 1)   blockDim(1024, 1, 1)
template <typename T>
static __global__ void BM_KERNEL(gather_logits)(
    int N,
    const int32_t *__restrict__ indexes, // (N,)
    const T *__restrict__ logits_in,
    float *__restrict__ logits_out // (N,)
) {
    size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
    if (offset < N) {
        int32_t idx = indexes[offset];
        logits_out[offset] = (float)logits_in[idx];
    }
}

core::Tensor gather_logits(
    const core::Context &ctx, const core::Tensor &indexes, const core::Tensor &logits) {
    size_t N = indexes.numel();
    int threads = round_up(std::min(N, (size_t)1024), 32);
    dim3 gridDim(round_up(N, threads) / threads, 1, 1);
    dim3 blockDim(threads, 1, 1);
    auto stream = ctx.current_stream()->ptr;

    auto d_out = ctx.tensor(indexes.size(), core::DataType::kFloat);
    BM_DTYPE_DISPATCH_FLOAT(logits.dtype(), {
        BM_KERNEL(gather_logits)<scalar_t><<<gridDim, blockDim, 0, stream>>>(
            N, indexes.data<int32_t>(), logits.data<scalar_t>(), d_out.mutable_data<float>());
    });
    BM_CUDART_ASSERT(cudaGetLastError());
    return d_out;
}

// gridDim(N / 1024, 1, 1)   blockDim(1024, 1, 1)
template <typename T>
static __global__ void BM_KERNEL(apply_gumbel)(
    size_t N,
    const float *__restrict__ uniform_eps, // (N,)
    const T *__restrict__ logits_in,       // (N,)
    T *__restrict__ logits_out             // (N,)
) {
    size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
    if (offset < N) {
        logits_out[offset] = (float)logits_in[offset] - logf(-logf(uniform_eps[offset]));
    }
}

core::Tensor apply_gumbel_softmax(
    const core::Context &ctx, curandGenerator_t &gen, const core::Tensor &logits) {
    size_t N = logits.numel();
    int threads = round_up(std::min(N, (size_t)1024), 32);
    dim3 gridDim(round_up(N, threads) / threads, 1, 1);
    dim3 blockDim(threads, 1, 1);
    auto stream = ctx.current_stream()->ptr;

    auto d_eps = ctx.tensor({N}, core::DataType::kFloat);
    auto d_out = ctx.tensor(logits.size(), logits.dtype());
    CURAND_CHECK(curandGenerateUniform(gen, d_eps.data<float>(), N));
    BM_DTYPE_DISPATCH_FLOAT(logits.dtype(), {
        BM_KERNEL(apply_gumbel)<scalar_t><<<gridDim, blockDim, 0, stream>>>(
            N, d_eps.data<float>(), logits.data<scalar_t>(), d_out.data<scalar_t>());
    });
    BM_CUDART_ASSERT(cudaGetLastError());
    return d_out;
}

// gridDim (n / 1024, 1, 1),  blockDim(1024, 1, 1)
template <typename T>
static __global__ void BM_KERNEL(scatter_update)(
    int n,
    int stride,
    const float *__restrict__ values,
    const int32_t *__restrict__ indices,
    const int32_t *__restrict__ batch_ids,
    T *__restrict__ logits) { // (batch_size, stride)
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        logits[batch_ids[i] * stride + indices[i]] = T(values[i]);
    }
}

void scatter_update(const core::Context &ctx,
                    const std::vector<float> &values,
                    const std::vector<int32_t> &token_ids, // indices[1]
                    const std::vector<int32_t> &batch_ids, // indices[0]
                    core::Tensor &logits) {
    BM_ASSERT(token_ids.size() == batch_ids.size(), "tokens and batch_id must have the same size");
    BM_ASSERT(token_ids.size() > 0, "tokens and batch_id must have at least one element");
    auto device = logits.device();
    int n = batch_ids.size();
    int vocab_size = logits.size(-1);

    auto d_values = ctx.tensor_of(values);
    auto d_tokens = ctx.tensor_of(token_ids);
    auto d_batches = ctx.tensor_of(batch_ids);

    int threads = round_up_thread(n);
    dim3 gridDim(round_up(n, threads) / threads);
    dim3 blockDim(threads);
    auto stream = ctx.current_stream()->ptr;

    BM_DTYPE_DISPATCH_FLOAT(logits.dtype(), {
        BM_KERNEL(scatter_update)<scalar_t><<<gridDim, blockDim, 0, stream>>>(
            n, vocab_size,
            d_values.data<float>(), d_tokens.data<int32_t>(), d_batches.data<int32_t>(),
            logits.mutable_data<scalar_t>());
    });
    BM_CUDART_ASSERT(cudaGetLastError());
}

void apply_beam_repetition_penalty(model::ModelContext &ctx,
                                   const BeamBufferManager<int> &bm,
                                   const std::vector<int> &hypotheses_last_pos,
                                   float ngram_penalty,
                                   float repetition_penalty,
                                   core::Tensor *logits_all) {
    std::vector<float> value_penalty;
    std::vector<int32_t> tokens_penalty;
    std::vector<int32_t> batch_penalty;
    std::vector<int32_t> rev_token_ids;

    for (int i = 0; i < hypotheses_last_pos.size(); i++) {
        rev_token_ids.clear();
        // TODO tianx ...
        // bm.get_hypothesis_tokens(hypotheses_last_pos[i], &rev_token_ids, true);
        // auto ngram_map = calc_repetition_ngram(rev_token_ids, ngram_penalty);
        // for (const auto &kv : ngram_map) {
        //     tokens_penalty.push_back(kv.first);
        //     batch_penalty.push_back(i);
        //     value_penalty.push_back(kv.second * repetition_penalty);
        // }
    }
    // if (!tokens_penalty.empty()) {
    //     beam_repetition_penalty(ctx, value_penalty, tokens_penalty, batch_penalty, *logits_all);
    // }
}

__global__ void BM_KERNEL(arange)(
    int32_t length,
    int32_t *out // (n, len)
) {
    int32_t offset = blockIdx.y * blockDim.x + threadIdx.x;
    if (offset < length) {
        out[blockIdx.x * length + offset] = offset;
    }
}

// gridDim(1, 1, 1),   blockDim(1, 1, 1)
template <typename T>
__global__ void BM_KERNEL(random_sampler_gpu)(
    int num_classes,
    T *probs_cum,      // (num_classes)
    int32_t *indicies, // (num_classes)
    int32_t *select,
    float *ptr_p,
    float top_p,
    int top_k) {
    if (top_k > 0) {
        top_p = min((float)probs_cum[top_k - 1], top_p);
    }
    T v_p = ptr_p[0] * top_p * (float)probs_cum[num_classes - 1];
    int lf = -1;
    int rt = num_classes - 1;
    while (lf + 1 < rt) {
        int mid = (lf + rt) / 2;
        if (probs_cum[mid] < v_p) {
            lf = mid;
        } else {
            rt = mid;
        }
    }
    select[0] = indicies[rt];
}

/*
  samples i from \sum{0}^{i}P(i) <= p, with p ~ U(0, top_p).
  P is descendly sorted and cumulated, then do a binary search.
*/
void random_sampler_gpu(
    const core::Context &ctx,
    curandGenerator_t &gen,
    core::Tensor &probs,  // (..., n_classes)
    core::Tensor &select, // (...)
    float top_p,
    int top_k,
    int num_samples) {
    unsigned int n_classes = probs.size(probs.ndim() - 1);
    BM_ASSERT(top_p <= 1.0f && top_p >= 0.0f, "top_p must be in [0, 1]");
    BM_ASSERT(top_k >= 0 && top_k < probs.size(-1), "invalid top k");
    BM_ASSERT_EQ(select.size(0), probs.numel() / n_classes * num_samples, "invalid select size");
    unsigned int batch = probs.numel() / n_classes;
    unsigned int select_step = select.size(0) / batch;
    auto stream = ctx.current_stream()->ptr;

    core::Tensor indicies_in = ctx.tensor({n_classes}, core::DataType::kInt32);
    core::Tensor indicies_out = ctx.tensor({n_classes}, core::DataType::kInt32);
    core::Tensor values_out = ctx.tensor({n_classes}, probs.dtype());
    {
        int threads = min(round_up(n_classes, 32), 1024);
        dim3 gridDim(1, round_up(n_classes, threads) / threads, 1);
        dim3 blockDim(threads, 1, 1);

        BM_KERNEL(arange)<<<gridDim, blockDim, 0, stream>>>(n_classes, indicies_in.data<int32_t>());
        BM_CUDART_ASSERT(cudaGetLastError());
    }

    BM_DTYPE_DISPATCH_FLOAT(probs.dtype(), {
        size_t temp_buffer_size1, temp_buffer_size2;
        cub::DeviceRadixSort::SortPairsDescending(
            nullptr,
            temp_buffer_size1,
            values_out.data<scalar_t>(),
            values_out.data<scalar_t>(),
            indicies_in.data<int32_t>(),
            indicies_out.data<int32_t>(),
            n_classes);
        cub::DeviceScan::InclusiveSum(
            nullptr,
            temp_buffer_size2,
            values_out.data<scalar_t>(),
            values_out.data<scalar_t>(),
            n_classes);
        size_t temp_buffer_size = std::max(temp_buffer_size1, temp_buffer_size2);
        auto temp = ctx.tensor({temp_buffer_size}, core::DataType::kInt8);

        core::Tensor p_random =
            ctx.tensor({(size_t)(batch * select_step)}, core::DataType::kFloat);
        CURAND_CHECK(curandGenerateUniform(gen, p_random.data<float>(), p_random.size(0)));
        for (int i = 0; i < batch; i++) {
            scalar_t *offset_prob = (probs.data<scalar_t>() + i * n_classes);
            BM_CUDART_ASSERT(cub::DeviceRadixSort::SortPairsDescending(
                temp.data(),
                temp_buffer_size,
                offset_prob,
                values_out.data<scalar_t>(),
                indicies_in.data<int32_t>(),
                indicies_out.data<int32_t>(),
                n_classes,
                0,
                sizeof(scalar_t) * 8,
                stream));
            BM_CUDART_ASSERT(cub::DeviceScan::InclusiveSum(
                temp.data(),
                temp_buffer_size,
                values_out.data<scalar_t>(),
                offset_prob,
                n_classes,
                stream));

            dim3 gridDim(1, 1, 1);
            dim3 blockDim(1, 1, 1);
            for (int j = 0; j < select_step; j++) {
                BM_KERNEL(random_sampler_gpu)<<<gridDim, blockDim, 0, stream>>>(
                    n_classes,
                    offset_prob,
                    indicies_out.data<int32_t>(),
                    select.data<int32_t>() + (i * select_step + j),
                    p_random.data<float>() + (i * select_step + j),
                    top_p,
                    top_k);
                BM_CUDART_ASSERT(cudaGetLastError());
            }
        }
    });
}

} // namespace beam_utility
