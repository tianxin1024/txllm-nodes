#include "bmengine/functions/transpose.h"
#include "bmengine/logger/std_log_op.hpp"

namespace bmengine {

// gridDim (round(n / 32), round(m / 32), batch_size),  blockDim (32, 32, 1)
template<typename T>
static __global__ void BM_KERNEL(transpose)(
    int n,
    int m,
    const T* __restrict__ input, // (batch, n, m)
    T* __restrict__ output       // (batch, m, n)
) {
    __shared__ T shared[32][33];
    int offset = blockIdx.z * m * n;
    {
        int pos_x = blockIdx.x * 32 + threadIdx.y;
        int pos_y = blockIdx.y * 32 + threadIdx.x;
        if (pos_x < n && pos_y < m) {
            shared[threadIdx.x][threadIdx.y] = input[offset + pos_x * m + pos_y];
        }
    }
    __syncthreads();
    {
        int pos_x = blockIdx.y * 32 + threadIdx.y;
        int pos_y = blockIdx.x * 32 + threadIdx.x;
        if (pos_x < m && pos_y < n) {
            output[offset + pos_x * n + pos_y] = shared[threadIdx.y][threadIdx.x];
        }
    }
}

__host__ void transpose(
    const bmengine::core::Tensor& input,
    const bmengine::core::Tensor& output,
    cudaStream_t stream) {
    int n = input.size(-2);
    int m = input.size(-1);
    int batch_size = input.numel() / n / m;

    dim3 gridDim(round_up(n, 32) / 32, round_up(m, 32) / 32, batch_size);
    dim3 blockDim(32, 32, 1);

    BM_DTYPE_DISPATCH(input.dtype(), {
        BM_KERNEL(transpose)<scalar_t><<<gridDim, blockDim, 0, stream>>>(
            n, m, input.data<scalar_t>(), output.data<scalar_t>());
    });
    BM_CUDART_ASSERT(cudaGetLastError());
}

}

namespace bmengine {
namespace functions {

using bmengine::core::Tensor;

class Transpose::impl {
public:
    impl(const core::Context& ctx) { }
    ~impl() = default;
    impl(const impl&) = delete;
    impl(impl&&) = default;

    void forward(const core::Context& ctx, const core::Tensor& input, core::Tensor* output) {
        BM_ASSERT(input.ndim() >= 2, "input must be 2d or 3d");
        auto shape = input.shape();
        auto ndim = input.ndim();
        std::swap(shape[ndim-1], shape[ndim-2]);
        if (output->numel() > 0) {
            BM_ASSERT_EQ(output->shape(), shape, "shape mismatch");
            BM_ASSERT_EQ(output->dtype(), input.dtype(), "dtype shape mismatch");
        } else {
            *output = ctx.tensor(shape, input.dtype());
        }
        transpose(input, *output, ctx.current_stream()->ptr);
    }
};

Transpose::Transpose(const core::Context& ctx) : pimpl(new impl(ctx)), core::Layer() { }
Transpose::~Transpose() = default;

core::Tensor Transpose::forward(
    const core::Context& ctx, const core::Tensor& input, core::Tensor* output) {
    core::Tensor ret;
    if (!output)
        output = &ret;
    pimpl->forward(ctx, input, output);
    return *output;
}

// gridDim (batch, dim1, dim2),  blockDim (last_dim)
template<typename T>
static __global__ void BM_KERNEL(transpose_2_1)(
    size_t stride,
    size_t last_dim,
    const T* __restrict__ input, // (batch?, dim1, dim2, last_dim)
    T* __restrict__ output       // (batch?, dim2, dim1, last_dim)
) {
    size_t offset_src = ((blockIdx.x * gridDim.y + blockIdx.y) * gridDim.z + blockIdx.z) * last_dim;
    size_t offset_dst = ((blockIdx.x * gridDim.z + blockIdx.z) * gridDim.y + blockIdx.y) * last_dim;

    for (int i = threadIdx.x; i < last_dim; i += blockDim.x) {
        output[offset_dst + i] = input[offset_src + i];
    }
}

// transpose dim1 with dim2
core::Tensor transpose_2_1(
    const core::Context& ctx,
    const core::Tensor& input, // (batch?, dim1, dim2, last_dim)
    core::Tensor* out_ptr // (batch?, dim2, dim1, last_dim)
) {
    BM_ASSERT(input.ndim() >= 3, "wrong ndim");
    size_t batch = input.ndim() > 3 ? input.size(0) : 1;
    size_t stride = input.numel() / batch;
    size_t last_dim = input.size(-1);

    std::vector<size_t> shape = input.shape();
    int d = input.ndim() - 3;
    std::swap(shape[d], shape[d + 1]);
    if ((shape[d] == 1 || shape[d + 1] == 1) && !out_ptr)
        return input.view(shape);

    Tensor output;
    if (out_ptr) {
        BM_ASSERT_EQ(out_ptr->dtype(), input.dtype(), "type mismatch");
        BM_ASSERT_EQ(out_ptr->shape(), shape, "type mismatch");
    } else {
        output = ctx.tensor(shape, input.dtype());
        out_ptr = &output;
    }

    dim3 blockDim(round_up_thread(last_dim));
    dim3 gridDim(batch, shape[d + 1], shape[d]);
    auto stream = ctx.current_stream()->ptr;

    BM_DTYPE_DISPATCH(input.dtype(), {
        BM_KERNEL(transpose_2_1)<<<gridDim, blockDim, 0, stream>>>(
        stride, last_dim, input.data<scalar_t>(), out_ptr->mutable_data<scalar_t>());
    });
    BM_CUDART_ASSERT(cudaGetLastError());
    return *out_ptr;
}

} // namespace functions
} // namespace
