#include "backend/attention_kernel.h"
#include <bmengine/functions/utils.cuh>
#include <bmengine/functions/reduce.cuh>
#include <stdio.h>

namespace nn {

using namespace bmengine;

// gridDim (num_heads, len_q, batch),  blockDim (len_buf~1024, 1, 1)
template <typename T>
static __global__ void BM_KERNEL(fused_scale_mask_softmax)(
    int len_buf, size_t len_q, T scale, size_t scale_stride, size_t mask_stride,
    const int8_t *__restrict__ mask, // (batch, len_q, len_buf)
    T *__restrict__ x                // (batch, num_heads, len_q, len_buf)
) {
    const float NEG_INFINITY = -functions::Inf<float>();
    int batch_id = blockIdx.z;
    x += batch_id * scale_stride + (blockIdx.x * len_q + blockIdx.y) * len_buf;
    mask += batch_id * mask_stride + blockIdx.y * len_buf;

    functions::SharedMemory<float> shared;
    float *smem = shared.getPointer(); // len_buf
    for (int i = threadIdx.x; i < len_buf; i += blockDim.x) {
        smem[i] = mask[i] > 0 ? float(x[i] * scale) : NEG_INFINITY;
    }
    float local_max = -1e20;
    for (int i = threadIdx.x; i < len_buf; i += blockDim.x) {
        local_max = fmaxf(local_max, smem[i]);
    }
    local_max = functions::blockReduceMax<float>(local_max);

    float local_sum = 1e-20;
    for (int i = threadIdx.x; i < len_buf; i += blockDim.x) {
        if (smem[i] == NEG_INFINITY) {
            smem[i] = 0;
        } else {
            float v = expf(float(smem[i]) - local_max);
            smem[i] = v;
            local_sum += v;
        }
    }
    local_sum = functions::blockReduceSum<float>(local_sum);
    for (int i = threadIdx.x; i < len_buf; i += blockDim.x) {
        x[i] = float(smem[i]) / local_sum;
    }
}

void attn_softmax(const core::Context &ctx,
                  float scale,
                  const core::Tensor &attn_score,      // (batch, num_heads, len_q, len_buf)
                  const core::Tensor &mask,            // (batch, len_q, len_buf)
                  const core::Tensor &position_bias) { // if relative (batch, num_head, len_q, len_buf) else if core::Tensor()
    auto dtype = attn_score.dtype();
    int batch = (attn_score.ndim() <= 3) ? 1 : attn_score.size(0);
    int score_stride = (attn_score.ndim() <= 3) ? 0 : attn_score.stride(0);
    int mask_stride = (mask.ndim() <= 2) ? 0 : mask.stride(0);
    int num_heads = attn_score.size(-3);
    int len_q = attn_score.size(-2);
    int len_buf = attn_score.size(-1);

    if (position_bias.numel() > 0) {
        BM_ASSERT_EQ(attn_score.numel(), position_bias.numel(), "shape mismatch");
    }

    dim3 gridDim(num_heads, len_q, batch);
    dim3 blockDim(std::min(1024, round_up(len_buf, 32)), 1, 1);
    auto stream = ctx.current_stream()->ptr;

    BM_DTYPE_DISPATCH_FLOAT(dtype, {
        size_t dynamic_size = len_buf * sizeof(float);
        if (dynamic_size < 48 * 1000 && position_bias.numel() > 0) {
            // TODO ...

        } else if (dynamic_size < ctx.get_max_shared_memory() && position_bias.numel() == 0) {
            std::cout << "dynamic_size: " << dynamic_size << std::endl;
            if (dynamic_size > 48 * 10000) {
                // TODO ...
            }
            BM_KERNEL(fused_scale_mask_softmax)<scalar_t><<<gridDim, blockDim, dynamic_size, stream>>>(
                len_buf, len_q, scale, score_stride, mask_stride, mask.data<int8_t>(), attn_score.data<scalar_t>());
        } else {
            // TODO ...
        }
    });
}
} // namespace nn
