#include "bmengine/core/core.h"
#include "bmengine/core/stream.h"
#include "bmengine/functions/reduce.cuh"
#include "bmengine/functions/tensor_ops.h"
#include "bmengine/functions/transpose.h"
#include "private/tensor_ops.h"
#include "private/allocator.h"

namespace bmengine {
namespace functions {

// gridDim (n, 1, 1),   blockDim (len, 1, 1)
template<typename T>
static __global__ void BM_KERNEL(reduce_abs_max)(
    int len,
    const T* inp, // (n, len)
    T* out        // (n)
) {
    float local_max = -1e4;
    int offset = blockIdx.x * len;
    for (int i = threadIdx.x; i < len; i += blockDim.x) {
        local_max = fmaxf(local_max, fabs(float(inp[offset + i])));
    }
    local_max = functions::blockReduceMax(local_max);
    if (threadIdx.x == 0)
        out[blockIdx.x] = local_max;
}

core::Tensor reduce_abs_max(const core::Context& ctx, const core::Tensor& a, int dim) {
    BM_ASSERT_EQ(a.ndim(), 2, "not 2-D tensor");
    auto v = dim == 0 ? Transpose(ctx).forward(ctx, a) : a;
    auto ret = ctx.tensor({ v.size(0) }, v.dtype());
    auto stream = ctx.current_stream()->ptr;
    int n = v.size(0);
    int len = v.size(1);

    dim3 gridDim(n);
    dim3 blockDim(round_up_thread(len));

    BM_DTYPE_DISPATCH_FLOAT(v.dtype(), {
        BM_KERNEL(reduce_abs_max)<scalar_t><<<gridDim, blockDim, 0, stream>>>(
            len, v.data<scalar_t>(), ret.mutable_data<scalar_t>());
    });
    BM_CUDART_ASSERT(cudaGetLastError());
    return ret;
}

// gridDim (n, 1, 1),   blockDim (head_len, 1, 1)
template<typename T>
static __global__ void BM_KERNEL(sum_kernel)(
    int head_len,
    const T* inp, // (n, len)
    T* out        // (n)
) {
    float local_sum = 0.0f;
    int offset = blockIdx.x * head_len;
    for (int i = threadIdx.x; i < head_len; i += blockDim.x) {
        local_sum += float(inp[offset + i]);
    }
    local_sum = functions::blockReduceSum(local_sum);
    for (int i = threadIdx.x; i < head_len; i += blockDim.x) {
        out[offset + i] = local_sum;
    }
}

core::Tensor sum(const core::Context& ctx, const core::Tensor& a) {
    auto ret = ctx.tensor(a.size(), a.dtype());
    auto stream = ctx.current_stream()->ptr;
    int n = 1;
    for (int i = 0; i < a.ndim() - 1; ++i) {
        n *= a.size(i);
    }
    int head_len = a.size(-1);

    dim3 gridDim(n);
    dim3 blockDim(round_up_thread(head_len));

    BM_DTYPE_DISPATCH_FLOAT(a.dtype(), {
        BM_KERNEL(sum_kernel)<scalar_t><<<gridDim, blockDim, 0, stream>>>(
            head_len, a.data<scalar_t>(), ret.mutable_data<scalar_t>());
    });
    BM_CUDART_ASSERT(cudaGetLastError());
    return ret;
}

// block <n / 1024>,    thread <1024>
template<typename T>
static __global__ void BM_KERNEL(div_kernel)(int n, const T* a, const T* b, T* c, const T eps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] / (b[idx] + eps);
    }
}

core::Tensor div(
    const core::Context& ctx, const core::Tensor& a, const core::Tensor& b, float eps) {
    auto ret = ctx.tensor(a.size(), a.dtype());
    auto stream = ctx.current_stream()->ptr;
    BM_ASSERT(vector_equal(a.size(), b.size()), "Tensor size mismatch");
    size_t total_size = a.numel();
    int threads = round_up(min((size_t) 1024, total_size), 32);
    int blocks = (total_size + threads - 1) / threads;
    dim3 gridDim(blocks, 1, 1);
    dim3 blockDim(threads, 1, 1);

    auto dtype = a.dtype();

    BM_DTYPE_DISPATCH_FLOAT(a.dtype(), {
        BM_KERNEL(div_kernel)<scalar_t><<<gridDim, blockDim, 0, stream>>>(
            total_size,
            a.data<scalar_t>(),
            b.data<scalar_t>(),
            ret.data<scalar_t>(),
            scalar_t(eps));
    });
    BM_CUDART_ASSERT(cudaGetLastError());
    return ret;
}

template<typename T>
__device__ inline T cuda_max(T val1, T val2) {
    return (val1 > val2) ? val1 : val2;
}

template<typename T>
__device__ inline T cuda_min(T val1, T val2) {
    return (val1 > val2) ? val2 : val1;
}

// gridDim (n, 1, 1),   blockDim (len, 1, 1)
template<typename T>
static __global__ void BM_KERNEL(amax_kernel)(
    int head_len,
    const T* inp, // (n, len)
    T* out        // (n)
) {
    float local_max = -1e4;
    int offset = blockIdx.x * head_len;
    for (int i = threadIdx.x; i < head_len; i += blockDim.x) {
        local_max = cuda_max(T(local_max), inp[offset + i]);
    }
    local_max = functions::blockReduceMax(local_max);

    if (threadIdx.x == 0)
        out[blockIdx.x] = local_max;
}

core::Tensor amax(const core::Context& ctx, const core::Tensor& a) {
    auto ret_size = a.size();
    ret_size[a.ndim() - 1] = 1;
    auto ret = ctx.tensor(ret_size, a.dtype());
    auto stream = ctx.current_stream()->ptr;
    int n = 1;
    for (int i = 0; i < a.ndim() - 1; ++i) {
        n *= a.size(i);
    }
    int head_len = a.size(-1);

    dim3 gridDim(n);
    dim3 blockDim(round_up_thread(head_len));

    BM_DTYPE_DISPATCH(a.dtype(), {
        BM_KERNEL(amax_kernel)<scalar_t><<<gridDim, blockDim, 0, stream>>>(
            head_len, a.data<scalar_t>(), ret.mutable_data<scalar_t>());
    });
    BM_CUDART_ASSERT(cudaGetLastError());
    return ret;
}

// gridDim (n, 1, 1),   blockDim (len, 1, 1)
template<typename T>
static __global__ void BM_KERNEL(amin_kernel)(
    int head_len,
    const T* inp, // (n, len)
    T* out        // (n)
) {
    float local_min = 1e4;
    int offset = blockIdx.x * head_len;
    for (int i = threadIdx.x; i < head_len; i += blockDim.x) {
        local_min = cuda_min(T(local_min), inp[offset + i]);
    }
    local_min = functions::blockReduceMin(local_min);

    if (threadIdx.x == 0)
        out[blockIdx.x] = local_min;
}

core::Tensor amin(const core::Context& ctx, const core::Tensor& a) {
    auto ret_size = a.size();
    ret_size[a.ndim() - 1] = 1;
    auto ret = ctx.tensor(ret_size, a.dtype());
    auto stream = ctx.current_stream()->ptr;
    int n = 1;
    for (int i = 0; i < a.ndim() - 1; ++i) {
        n *= a.size(i);
    }
    int head_len = a.size(-1);

    dim3 gridDim(n);
    dim3 blockDim(round_up_thread(head_len));

    BM_DTYPE_DISPATCH(a.dtype(), {
        BM_KERNEL(amin_kernel)<scalar_t><<<gridDim, blockDim, 0, stream>>>(
            head_len, a.data<scalar_t>(), ret.mutable_data<scalar_t>());
    });
    BM_CUDART_ASSERT(cudaGetLastError());
    return ret;
}

// gridDim (n, 1, 1),   blockDim (head_len, 1, 1)
template<typename T>
static __global__ void BM_KERNEL(add_scalar)(
    int head_len,
    const T* inp, // (n, len)
    float b,      // 1
    T* out        // (n)
) {
    int offset = blockIdx.x * head_len;
    for (int i = threadIdx.x; i < head_len; i += blockDim.x) {
        out[offset + i] = inp[offset + i] + T(b);
    }
}

core::Tensor add(const core::Context& ctx, const core::Tensor& a, float b) {
    auto ret = ctx.tensor(a.size(), a.dtype());
    auto stream = ctx.current_stream()->ptr;
    int n = 1;
    for (int i = 0; i < a.ndim() - 1; ++i) {
        n *= a.size(i);
    }
    int head_len = a.size(-1);

    dim3 gridDim(n);
    dim3 blockDim(round_up_thread(head_len));

    BM_DTYPE_DISPATCH(a.dtype(), {
        BM_KERNEL(add_scalar)<scalar_t><<<gridDim, blockDim, 0, stream>>>(
            head_len, a.data<scalar_t>(), b, ret.mutable_data<scalar_t>());
    });
    BM_CUDART_ASSERT(cudaGetLastError());
    return ret;
}

} // namespace functions
} // namespace bmengine
