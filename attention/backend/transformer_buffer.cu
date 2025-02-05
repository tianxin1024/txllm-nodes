#include "backend/transformer_buffer.h"
#include "bmengine/functions/init.h"
#include <assert.h>

namespace kvcache {

// gridDim (batch, len_kv, num_heads), blockDim (1024, 1, 1)
template <typename T>
static __global__ void BM_KERNEL(copy_to_buffer)(int len_buf, int num_heads, int dim_head,
                                                 size_t src_stride, size_t dst_stride, int place_stride,
                                                 const int32_t *__restrict__ placement, // (batch, len_kv)
                                                 const T *__restrict__ src,             // (batch, len_kv, num_heads, dim_head)
                                                 T *__restrict__ dst,                   // (batch, (num_heads, len_buf|len_buf, num_heads), dim_head)
                                                 bool BSHD) {
    int batch_id = blockIdx.x;
    int pos_buf = (placement == nullptr) ? blockIdx.y : placement[batch_id * place_stride + blockIdx.y];
    if (pos_buf < 0) return;

    assert(pos_buf < len_buf);
    size_t offset_src = batch_id * src_stride + (blockIdx.y * num_heads + blockIdx.z) * dim_head;
    size_t offset_dst;
    if (BSHD)
        offset_dst = batch_id * dst_stride + (blockIdx.z + num_heads * pos_buf) * dim_head;
    else
        offset_dst = batch_id * dst_stride + (blockIdx.z * len_buf + pos_buf) * dim_head;

    for (int i = threadIdx.x; i < dim_head; i += blockDim.x) {
        dst[offset_dst + i] = src[offset_src + i];
    }
}

void copy_to_buffer(int num_heads,
                    int len_kv,
                    int len_buf,
                    int dim_head,
                    const core::Tensor *placement, // (batch, len_q)
                    const core::Tensor &src,       // (batch, len_q, num_heads, dim_head)
                    const core::Tensor &dst,       // (batch, num_heads, len_buf, dim_head)
                    cudaStream_t stream,
                    bool BSHD) {
    int batch = (src.ndim() == 3) ? 1 : src.size(0);
    size_t src_stride = (src.ndim() == 3) ? 0 : src.stride(0);
    size_t dst_stride = (dst.ndim() == 3) ? 0 : dst.stride(0);
    int place_stride = (placement == nullptr || placement->ndim() == 1) ? 0 : placement->stride(0);
    dim3 gridDim(batch, len_kv, num_heads);
    dim3 blockDim(std::min(1024, round_up(dim_head, 32)), 1, 1);
    auto dtype = src.dtype();

    BM_ASSERT((src.ndim() == 3 && dst.ndim() == 3) || (src.ndim() == 4 && dst.ndim() == 4),
              "src and dst must be 3/4-dimensional");
    BM_ASSERT_EQ(dst.dtype(), dtype, "dst.dtype() != src.dtype()");
    BM_ASSERT_EQ(src.size(-1), dim_head, "dim mismatch");
    BM_ASSERT_EQ(src.size(-2), num_heads, "dim mismatch");
    BM_ASSERT_EQ(src.size(-3), len_kv, "dim mismatch");
    BM_ASSERT_EQ(dst.size(-1), dim_head, "dim mismatch");

    if (BSHD) {
        BM_ASSERT_EQ(dst.size(-2), num_heads, "dim mismatch");
        if (batch > 1) {
            BM_ASSERT_EQ(dst.size(-3), len_buf, "len_buf mismatch");
        } else {
            BM_ASSERT_LE(len_buf, dst.size(-3), "len_buf mismatch");
        }
    } else {
        BM_ASSERT_EQ(dst.size(-3), num_heads, "dim mismatch");
        BM_ASSERT_EQ(dst.size(-2), len_buf, "dim mismatch");
    }

    BM_DTYPE_DISPATCH(dtype, {
        BM_KERNEL(copy_to_buffer)<<<gridDim, blockDim, 0, stream>>>(
            len_buf, num_heads, dim_head, src_stride, dst_stride, place_stride,
            (placement == nullptr ? nullptr : placement->data<int32_t>()),
            src.data<scalar_t>(), dst.data<scalar_t>(), BSHD);
    });
    BM_CUDART_ASSERT(cudaGetLastError());
}

// gridDim (n / 1024, 1, 1),    blockDim (1024, 1, 1)
template <typename T>
__global__ void BM_KERNEL(resize_buffer)(
    size_t n, size_t old_stride, size_t new_stride, const T *old_buffer, T *new_buffer) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int in_pos = i % old_stride;
        if (in_pos < new_stride) {
            new_buffer[in_pos + (i / old_stride * new_stride)] = old_buffer[i];
        }
    }
}

__host__ void launch_resize_buffer(
    size_t old_stride,
    size_t new_stride,
    const core::Tensor &src,
    const core::Tensor &dst,
    cudaStream_t stream,
    size_t numel = 0) {
    BM_ASSERT(src.device() == dst.device(), "src and dst must be on the same device");
    size_t n = numel == 0 ? src.numel() : numel;
    int threads = round_up(std::min(n, (size_t)1024), 32);
    int blocks = round_up(n, threads) / threads;

    dim3 blockDim(threads, 1, 1);
    dim3 gridDim(blocks, 1, 1);

    auto dtype = src.dtype();
    BM_DTYPE_DISPATCH(src.dtype(), {
        BM_KERNEL(resize_buffer)<scalar_t><<<gridDim, blockDim, 0, stream>>>(
            n, old_stride, new_stride, src.data<scalar_t>(), dst.data<scalar_t>());
    });
    BM_CUDART_ASSERT(cudaGetLastError());
}

core::Tensor resize_buffer(
    const core::Context &ctx, const core::Tensor &buffer, int dim, size_t new_length) {
    auto shape = buffer.size();
    int normalized_dim = dim < 0 ? (shape.size() + dim) : dim;

    BM_ASSERT(
        (normalized_dim >= 0) && (normalized_dim < shape.size()),
        "Invalid dimension: dim must in [0, " + std::to_string(shape.size()) + "), but got "
            + std::to_string(dim));
    BM_ASSERT(ctx.active_device() == buffer.device(), "Invalid deivce");

    size_t stride_base = 1;
    for (int i = normalized_dim + 1; i < shape.size(); i++) {
        stride_base *= shape[i];
    }
    size_t old_stride = shape[normalized_dim] * stride_base;
    size_t new_stride = new_length * stride_base;

    auto new_shape = shape;
    new_shape[normalized_dim] = new_length;
    auto new_buffer = ctx.tensor(new_shape, buffer.dtype());
    BM_CUDART_ASSERT(
        cudaMemsetAsync(new_buffer.data(), 0, new_buffer.nbytes(), ctx.current_stream()->ptr));

    launch_resize_buffer(old_stride, new_stride, buffer, new_buffer, ctx.current_stream()->ptr);

    return new_buffer;
}

TransformerBuffer::TransformerBuffer(int batch_size,
                                     int num_layers,
                                     int num_heads,
                                     int dim_head,
                                     core::DataType dtype,
                                     bool parallel,
                                     bool BSHD) :
    KVCache(batch_size, num_layers, num_heads, dim_head, dtype, parallel, BSHD) {
    BM_ASSERT(is_dyn_batch() || batch_size > 0, "batch_size must be greater than 0");
    BM_ASSERT(num_layers > 0, "num_layers must be greater than 0");
    BM_ASSERT(num_heads > 0, "num_heads must be greater than 0");
    BM_ASSERT(dim_head > 0, "dim_head must be greater than 0");
    buffer.resize(num_layers);
}

TransformerBuffer::~TransformerBuffer() {
}

void TransformerBuffer::check_layer(int i) const {
    BM_ASSERT(i >= 0 && i < num_layers,
              "Invalid layer index: i must in [0, " + std::to_string(num_layers) + "), but got " + std::to_string(i));
}

const core::Tensor &TransformerBuffer::operator[](int i) const {
    check_layer(i);
    return buffer[i];
}

core::Tensor &TransformerBuffer::operator[](int i) {
    check_layer(i);
    return buffer[i];
}

void TransformerBuffer::resize(const core::Context &ctx, size_t new_length) {
    BM_ASSERT_EQ(layer_devices.size(), (size_t)(num_layers), "Invalid layer_devices");
    std::cout << "TransformerBuffer::resize: new_length=" << new_length << "\n";

    if (is_dyn_batch()) {
        resize_dyn_batch(ctx, new_length);
        return;
    }

    for (int i = 0; i < num_layers; ++i) {
        ctx.switch_to_device(layer_devices[i]);
        std::vector<size_t> shape;
        if (buffer[i].numel() == 0) {
            if (BSHD) {
                shape = {new_length, num_heads, dim_head};
            } else {
                shape = {num_heads, new_length, dim_head};
            }
            if (batch_size > 0) {
                shape.insert(shape.begin(), size_t(batch_size));
            }
            buffer[i] = ctx.tensor(shape, dtype);
            functions::zeros_(ctx, buffer[i]);
        } else {
            buffer[i] = resize_buffer(ctx, buffer[i], BSHD ? -3 : -2, new_length);
        }
    }
    ctx.switch_to_device(layer_devices[0]);
}

void TransformerBuffer::resize_dyn_batch(const core::Context &ctx, size_t new_length) {
    // for (int i = 0; i < num_layers;) {
    //     int dev = layer_devices[i];
    //     int j = i + 1;
    //     while (j < num_layers && layer_devices[j] == dev)
    //         j++;
    //     std::cout << "dev, i, j " << dev << ", " << i << ", " << j << std::endl;
    //     // resize MULTIPLE layer buffers on SAME device.
    //     ctx.switch_to_device(layer_devices[i]);
    //     resize_multi_layer(ctx, new_length, i, j, BSHD ? -3 : -2);
    //     if (scale_dtype_.get()) {
    //         BM_ASSERT_EQ(j, num_layers, "Only support TP mode");
    //         resize_scale(ctx, new_length, i, j, BSHD ? -2 : -1);
    //     }
    //     i = j;
    // }
    ctx.switch_to_device(layer_devices[0]);
}

} // namespace kvcache
