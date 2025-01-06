#include "backend/utils.h"
#include <bmengine/functions/all.h>

namespace model {

using namespace bmengine;

template <typename T>
static __global__ void BM_KERNEL(convert_fp32)(
    size_t n, const T *__restrict__ a, float *__restrict__ b) {
    size_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < n) {
        b[pos] = float(a[pos]);
    }
}

static __global__ void BM_KERNEL(convert_fp16)(
    size_t n, const float *__restrict__ a, half *__restrict__ b) {
    size_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < n) {
        b[pos] = __float2half(a[pos]);
    }
}

core::Tensor concat_logits(
    const core::Context &ctx, const std::tuple<core::Tensor, core::Tensor> &logits_tuple) {
    return functions::concat_tensor(ctx, std::get<0>(logits_tuple), std::get<1>(logits_tuple));
}

core::Tensor convert_fp32(const core::Context &ctx, const core::Tensor &logits) {
    core::Tensor out = ctx.tensor(logits.size(), core::DataType::kFloat);
    size_t n = out.numel();

    int threads = std::min((size_t)1024, round_up(n, 32));
    dim3 gridDim(round_up(n, threads) / threads, 1, 1);
    dim3 blockDim(threads, 1, 1);
    auto stream = ctx.current_stream()->ptr;

    BM_DTYPE_DISPATCH_FLOAT(logits.dtype(), {
        BM_KERNEL(convert_fp32)<<<gridDim, blockDim, 0, stream>>>(
            n, logits.data<scalar_t>(), out.data<float>());
    });
    BM_CUDART_ASSERT(cudaGetLastError());
    return out;
}

} // namespace model
