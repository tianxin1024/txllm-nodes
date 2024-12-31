#include "bmengine/functions/softmax.h"
#include "bmengine/functions/utils.cuh"
#include "bmengine/functions/reduce.cuh"

namespace bmengine {
namespace functions {

// gridDim (batch, 1, 1),   blockDim (1024, 1, 1)
template<typename T>
static __global__ void BM_KERNEL(softmax)(
    int n,
    const T* logits, // (batch, n)
    T* output,       // (batch, n)
    float temperature) {
    int offset = blockIdx.x * n;
    float local_max = -1e20;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        local_max = fmaxf(local_max, logits[i + offset]);
    }
    local_max = functions::blockReduceMax<float>(local_max) / temperature;
    float local_sum = 1e-20;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        local_sum += expf((float) logits[i + offset] / temperature - local_max);
    }
    local_sum = functions::blockReduceSum<float>(local_sum);
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        output[i + offset] = expf((float) logits[i + offset] / temperature - local_max) / local_sum;
    }
}

// gridDim (batch, 1, 1),   blockDim (1024, 1, 1)
template<typename T>
static __global__ void BM_KERNEL(softmax_cached)(
    int n,
    const T* logits, // (batch, n)
    T* output,       // (batch, n)
    float temperature) {
    int offset = blockIdx.x * n;
    SharedMemory<T> shared;
    T* smem = shared.getPointer();

    float local_max = -1e20;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        T v = logits[i + offset] / T(temperature);
        local_max = fmaxf(local_max, v);
        smem[i] = v;
    }
    local_max = functions::blockReduceMax<float>(local_max);
    float local_sum = 1e-20;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = expf((float) smem[i] - local_max);
        local_sum += v;
        smem[i] = v;
    }
    local_sum = functions::blockReduceSum<float>(local_sum);
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        output[i + offset] = float(smem[i]) / local_sum;
    }
}

void softmax(
    const core::Context& ctx,
    const core::Tensor& logits,
    const core::Tensor& output,
    float temperature) {
    BM_ASSERT(logits.dtype() == output.dtype(), "dtype mismatch");
    BM_ASSERT(logits.device() == ctx.active_device(), "device mismatch");
    BM_ASSERT(output.device() == ctx.active_device(), "device mismatch");

    int batch = 1;
    int ndims = logits.ndim();
    for (int i = 0; i < ndims - 1; i++) {
        batch *= logits.size(i);
    }
    int n = logits.size(ndims - 1);

    dim3 gridDim(batch, 1, 1);
    dim3 blockDim(min(1024, round_up(n, 32)), 1, 1);
    auto stream = ctx.current_stream()->ptr;

    BM_DTYPE_DISPATCH_FLOAT(logits.dtype(), {
        size_t dynamic_size = sizeof(float) * n;
        if (dynamic_size < 48 * 1000) {
            BM_KERNEL(softmax_cached)<scalar_t><<<gridDim, blockDim, dynamic_size, stream>>>(
                n, logits.data<scalar_t>(), output.data<scalar_t>(), temperature);
        } else {
            BM_KERNEL(softmax)<scalar_t><<<gridDim, blockDim, 0, stream>>>(
                n, logits.data<scalar_t>(), output.data<scalar_t>(), temperature);
        }
    });
    BM_CUDART_ASSERT(cudaGetLastError());
}

}

}