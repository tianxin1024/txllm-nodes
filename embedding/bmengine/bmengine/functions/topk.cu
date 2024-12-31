#include "bmengine/functions/topk.h"
#include "bmengine/functions/utils.cuh"

namespace bmengine {

namespace functions {

template<typename T, int N>
static __device__ inline void warpBitonicSort(T& v1, int& pos, bool asc) {
    int lane_id = threadIdx.x & (N - 1);
#pragma unroll
    for (int k = 2; k <= N; k *= 2) {
        bool desc = ((lane_id & k) == 0) ^ asc;
#pragma unroll
        for (int j = k / 2; j > 0; j /= 2) {
            T v2 = __shfl_xor_sync(0xFFFFFFFF, v1, j);
            int pos2 = __shfl_xor_sync(0xFFFFFFFF, pos, j);
            bool upper = (lane_id & j) != 0;

            if (desc ^ (v1 > v2 || (v1 == v2 && pos < pos2)) ^ upper) {
                v1 = v2;
                pos = pos2;
            }
        }
    }
}

template<typename T, int N>
static __device__ inline void warpBitonicMerge(T& v1, int& pos1, T& v2, int& pos2) {
    if (v1 < v2 || (v1 == v2 && pos1 > pos2)) {
        v1 = v2;
        pos1 = pos2;
    }
    int lane_id = threadIdx.x & (N - 1);
// resort
#pragma unroll
    for (int j = N / 2; j > 0; j /= 2) {
        v2 = __shfl_xor_sync(0xFFFFFFFF, v1, j);
        int pos2 = __shfl_xor_sync(0xFFFFFFFF, pos1, j);
        bool upper = (lane_id & j) != 0;
        if ((v1 < v2 || (v1 == v2 && pos1 > pos2)) ^ upper) {
            v1 = v2;
            pos1 = pos2;
        }
    }
}

template<typename T, int N>
static __device__ inline void blockBitonicReduce(T& v, int& pos) {
    __shared__ T shared_val[1024];
    __shared__ int shared_pos[1024];

    // block reduce
    shared_val[threadIdx.x] = v;
    shared_pos[threadIdx.x] = pos;

// inter warp reduce
#pragma unroll
    for (int i = 512; i >= 32; i >>= 1) {
        if (blockDim.x > i) {
            __syncthreads();
            if (threadIdx.x < i) {
                int idx_next = (i << 1) - threadIdx.x - 1;

                T nw_v = (idx_next < blockDim.x) ? shared_val[idx_next] : T(-Inf<T>());
                int nw_pos = (idx_next < blockDim.x) ? shared_pos[idx_next] : -1;
                warpBitonicMerge<T, N>(v, pos, nw_v, nw_pos); // merge and rebuild in desc order
                shared_val[threadIdx.x] = v;
                shared_pos[threadIdx.x] = pos;
            }
        }
    }

    // intra warp reduce
    if (threadIdx.x < 32) {
        warpBitonicSort<T, 32>(v, pos, false);
    }
}

template<typename T, int N>
static __global__ void BM_KERNEL(bitonic_topk)(
    int n,
    int top,
    T* inp,     // (batch, n)
    float* out, // (batch, top)
    int* idx    // (batch, top)
) {

    int offset_inp = blockIdx.x * n;
    int offset_out = blockIdx.x * top;

    T local_v = threadIdx.x < n ? inp[offset_inp + threadIdx.x] : -Inf<T>();
    int local_pos = threadIdx.x;
    warpBitonicSort<T, N>(local_v, local_pos, false); // local sort in desc order

    for (int i = blockDim.x; i < n; i += blockDim.x) {
        T nw_v = (i + threadIdx.x) < n ? inp[offset_inp + i + threadIdx.x] : -Inf<T>();
        int nw_pos = i + threadIdx.x;
        // step.1: local sort
        warpBitonicSort<T, N>(nw_v, nw_pos, true); // local sort in asc order

        // step.2&3: merge and rebuild
        warpBitonicMerge<T, N>(local_v, local_pos, nw_v, nw_pos); // merge and rebuild in desc order
    }

    blockBitonicReduce<T, N>(local_v, local_pos);
    if (threadIdx.x < top) {
        out[offset_out + threadIdx.x] = local_v;
        idx[offset_out + threadIdx.x] = local_pos;
    }
}

// intra-block topk
// gridDim(batch, n / 1024, 1), threadDim(1024, 1, 1)
template<typename T, int N, bool ordered>
static __global__ void BM_KERNEL(bitonic_topk_multiblock)(
    int n,
    const T* inp,       // (batch, n)
    const int* idx_inp, // (batch, n)
    T* out,             // (batch, n / 1024 * N)
    int* idx            // (batch, n / 1024 * N)
) {
    int offset_col = blockIdx.y * blockDim.x + threadIdx.x; // 0~n
    int offset_inp = blockIdx.x * n + offset_col;
    int offset_out = blockIdx.x * (gridDim.y * N) + blockIdx.y * N + threadIdx.x;

    T local_v = (offset_col < n) ? inp[offset_inp] : T(-Inf<T>());
    int local_pos = (idx_inp == nullptr || offset_col >= n) ? offset_col : idx_inp[offset_inp];
    if (!ordered)
        warpBitonicSort<T, N>(local_v, local_pos, false); // local sort in desc order

    blockBitonicReduce<T, N>(local_v, local_pos);

    if (threadIdx.x < N) {
        out[offset_out] = local_v;
        idx[offset_out] = local_pos;
    }
}

// copy kernel
// gridDim(batch, 1, 1),   blockDim(top, 1, 1)
template<typename T>
static __global__ void BM_KERNEL(bitonic_topk_multiblock_copy)(
    int n,
    int top,
    const T* inp,       // (batch, n)
    const int* idx_inp, // (batch, n)
    T* out,             // (batch, top)
    int* idx            // (batch, top)
) {
    int offset_inp = blockIdx.x * n + threadIdx.x;
    int offset_out = blockIdx.x * top + threadIdx.x;
    if (threadIdx.x < top) {
        out[offset_out] = inp[offset_inp];
        idx[offset_out] = idx_inp[offset_inp];
    }
}

#define TOPK_SIZE_DISPATCH(top, ...)                                                               \
    do {                                                                                           \
        const int& top_v = top;                                                                    \
        if (top_v > 32) {                                                                          \
            const int top_size = 64;                                                               \
            __VA_ARGS__                                                                            \
        } else if (top_v > 16) {                                                                   \
            const int top_size = 32;                                                               \
            __VA_ARGS__                                                                            \
        } else if (top_v > 8) {                                                                    \
            const int top_size = 16;                                                               \
            __VA_ARGS__                                                                            \
        } else if (top_v > 4) {                                                                    \
            const int top_size = 8;                                                                \
            __VA_ARGS__                                                                            \
        } else if (top_v > 2) {                                                                    \
            const int top_size = 4;                                                                \
            __VA_ARGS__                                                                            \
        } else if (top_v > 1) {                                                                    \
            const int top_size = 2;                                                                \
            __VA_ARGS__                                                                            \
        } else {                                                                                   \
            const int top_size = 1;                                                                \
            __VA_ARGS__                                                                            \
        }                                                                                          \
    } while (0)

void bitonic_topk(
    const core::Context& ctx,
    const core::Tensor& x,
    const core::Tensor& out,
    const core::Tensor& pos) {
    unsigned int dim = x.ndim();
    BM_ASSERT(dim == 1 || dim == 2, "x must be 1d or 2d");
    BM_ASSERT(out.ndim() == dim, "out must have the same dim as x");
    BM_ASSERT(vector_equal(out.size(), pos.size()), "out and pos must have the same size");

    unsigned int batch = (dim == 1) ? 1 : x.size(0);
    unsigned int n = (dim == 1) ? x.size(0) : x.size(1);
    unsigned int top = (dim == 1) ? out.size(0) : out.size(1);

    BM_ASSERT(top <= 32, "top must be <= 32"); // should not bigger than warpSize
    BM_ASSERT_LE(top, n, "top must be <= n");

    auto stream = ctx.current_stream()->ptr;
    auto dtype = x.dtype();

    BM_DTYPE_DISPATCH(dtype, {
        TOPK_SIZE_DISPATCH(top, {
            bool first = true;
            dim3 blockDim(1024, 1, 1);
            unsigned int tmp_n = n;

            core::Tensor buf_val =
                ctx.tensor({ batch, round_up(tmp_n, 1024) / 1024 * top_size }, dtype);
            core::Tensor buf_pos = ctx.tensor(
                { batch, round_up(tmp_n, 1024) / 1024 * top_size }, core::DataType::kInt32);
            do {
                // for each step: get top_size of 1024(max thread num)
                //   tmp_n => tmp_n * ( top_size / 1024)
                if (ctx.debug() >= 3) {
                    std::cout << "Topk step: tmp_n=" << tmp_n << ", n=" << n
                              << ", top_size=" << top_size << std::endl;
                }
                dim3 gridDim(batch, round_up(tmp_n, 1024) / 1024, 1);
                if (first) {
                    first = false;
                    BM_KERNEL(bitonic_topk_multiblock)<scalar_t, top_size, false>
                        <<<gridDim, blockDim, 0, stream>>>(
                            tmp_n,
                            x.data<scalar_t>(),
                            nullptr,
                            buf_val.data<scalar_t>(),
                            buf_pos.data<int>());
                } else {
                    core::Tensor nw_buf_val =
                        ctx.tensor({ batch, round_up(tmp_n, 1024) / 1024 * top_size }, dtype);
                    core::Tensor nw_buf_pos = ctx.tensor(
                        { batch, round_up(tmp_n, 1024) / 1024 * top_size }, core::DataType::kInt32);
                    BM_KERNEL(bitonic_topk_multiblock)<scalar_t, top_size, false>
                        <<<gridDim, blockDim, 0, stream>>>(
                            tmp_n,
                            buf_val.data<scalar_t>(),
                            buf_pos.data<int>(),
                            nw_buf_val.data<scalar_t>(),
                            nw_buf_pos.data<int>());
                    buf_val = nw_buf_val;
                    buf_pos = nw_buf_pos;
                }
                tmp_n = round_up(tmp_n, 1024) / 1024 * top_size;
            } while (tmp_n > top_size);

            BM_ASSERT(
                buf_val.size(0) == batch && buf_val.size(1) == top_size,
                "Unexpected buf_val size (" + std::to_string(buf_val.size(0)) + ", "
                    + std::to_string(buf_val.size(1)) + ")");
            BM_ASSERT(
                buf_pos.size(0) == batch && buf_pos.size(1) == top_size,
                "Unexpected buf_pos size (" + std::to_string(buf_pos.size(0)) + ", "
                    + std::to_string(buf_pos.size(1)) + ")");

            // copy to output tensor
            {
                dim3 gridDim(batch, 1, 1);
                blockDim = dim3(top_size, 1, 1);
                BM_KERNEL(bitonic_topk_multiblock_copy)<scalar_t><<<gridDim, blockDim, 0, stream>>>(
                    top_size,
                    top,
                    buf_val.data<scalar_t>(),
                    buf_pos.data<int>(),
                    out.data<scalar_t>(),
                    pos.data<int>());
            }
        });
    });
    BM_CUDART_ASSERT(cudaGetLastError());
}

class TopK::impl {
public:
    impl(const core::Context& ctx) { }
    ~impl() { }
    std::pair<core::Tensor, core::Tensor> forward(
        const core::Context& ctx, const core::Tensor& inp, unsigned int top) {
        BM_ASSERT(inp.ndim() == 2, "inp must be 2d");
        BM_ASSERT(top > 0, "top must be > 0");
        unsigned int batch = inp.size(0);
        unsigned int n = inp.size(1);

        auto ret = std::make_pair(
            ctx.tensor({ batch, top }, inp.dtype()),
            ctx.tensor({ batch, top }, core::DataType::kInt32));
        bitonic_topk(ctx, inp, ret.first, ret.second);
        return ret;
    }
};

TopK::TopK(const core::Context& ctx) : pimpl(new impl(ctx)) { }
TopK::~TopK() = default;

std::pair<core::Tensor, core::Tensor> TopK::forward(
    const core::Context& ctx, const core::Tensor& inp, int top) {
    return pimpl->forward(ctx, inp, top);
}

} // namespace functions

} // namespace bmengine
