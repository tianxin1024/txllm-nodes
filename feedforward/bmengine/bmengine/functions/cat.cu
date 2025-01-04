#include "bmengine/functions/cat.h"
#include "bmengine/functions/utils.cuh"

namespace bmengine {
namespace functions {

// gridDim (n / 1024, 1, 1),    blockDim (1024, 1, 1)
template<typename T>
static __global__ void BM_KERNEL(concat)(
    size_t size_before,
    size_t stride_after,
    size_t x_dim,
    size_t y_dim,
    const T* __restrict__ x, // (size_before, x_dim, stride_after)
    const T* __restrict__ y, // (size_before, y_dim, stride_after)
    T* __restrict__ out      // (size_before, x_dim + y_dim, stride_after)
) {
    size_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    size_t pos_dim = (pos / stride_after) % (x_dim + y_dim);
    size_t pos_stride = pos % stride_after;
    size_t pos_before = pos / (stride_after * (x_dim + y_dim));
    if (pos_dim < x_dim) {
        out[pos] = x[(pos_before * x_dim + pos_dim) * stride_after + pos_stride];
    } else {
        out[pos] = y[(pos_before * y_dim + pos_dim - x_dim) * stride_after + pos_stride];
    }
}

void cat(
    const core::Context& ctx,
    const core::Tensor& x,
    const core::Tensor& y,
    const core::Tensor& out,
    int dim) {
    BM_ASSERT(x.ndim() == y.ndim(), "x and y must have the same number of dimensions");
    BM_ASSERT(x.ndim() == out.ndim(), "x and out must have the same number of dimensions");
    BM_ASSERT(dim >= 0 && dim < x.ndim(), "dim must be in the range [0, x.ndim())");
    BM_ASSERT(x.dtype() == y.dtype(), "x and y must have the same dtype");
    BM_ASSERT(x.dtype() == out.dtype(), "x and out must have the same dtype");
    BM_ASSERT(x.device() == y.device(), "x and y must be on the same device");
    BM_ASSERT(x.device() == out.device(), "x and out must be on the same device");

    int total_dim = x.ndim();
    size_t size_before = 1;
    size_t stride_after = 1;
    std::vector<size_t> output_dim;

    for (int i = 0; i < x.ndim(); i++) {
        if (i < dim) {
            size_t v = x.size(i);
            BM_ASSERT(
                v == y.size(i),
                "x and y must have the same size in dimension " + std::to_string(i));
            output_dim.push_back(v);
            size_before *= v;
        } else if (i == dim) {
            output_dim.push_back(x.size(i) + y.size(i));
        } else { // i > dim
            size_t v = x.size(i);
            BM_ASSERT(
                v == y.size(i),
                "x and y must have the same size in dimension " + std::to_string(i));
            output_dim.push_back(v);
            stride_after *= v;
        }
    }

    BM_ASSERT(
        vector_equal(output_dim, out.size()),
        "output dim must be equal to the concatenated dimensions");

    int num_threads = round_up(min(out.numel(), (size_t) 1024), 32);

    dim3 gridDim(round_up(out.numel(), num_threads) / num_threads, 1, 1);
    dim3 blockDim(num_threads, 1, 1);
    auto stream = ctx.current_stream()->ptr;

    BM_DTYPE_DISPATCH(out.dtype(), {
        BM_KERNEL(concat)<scalar_t><<<gridDim, blockDim, 0, stream>>>(
            size_before,
            stride_after,
            x.size(dim),
            y.size(dim),
            x.data<scalar_t>(),
            y.data<scalar_t>(),
            out.data<scalar_t>());
    });
    BM_CUDART_ASSERT(cudaGetLastError());
}

}
}