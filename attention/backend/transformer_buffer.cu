#include "backend/transformer_buffer.h"
#include "bmengine/functions/init.h"

namespace kvcache {

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
