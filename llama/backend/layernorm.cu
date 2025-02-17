#include "backend/layernorm.h"
#include <bmengine/functions/reduce.cuh>

namespace nn {

// gridDim (seq_len, 1, batch),   blockDim(1024, 1, 1)
template <typename T>
static __global__ void BM_KERNEL(layernorm_rms)(
    int dim_model, float eps,
    int input_stride,             // seq_len * dim_model
    const T *__restrict__ weight, // (dim_model)
    const T *__restrict__ input,  // (batch, seq_len, dim_model)
    T *__restrict__ output,       // (batch, seq_len, dim_model)
    float scale,
    T *input2 = nullptr,
    T *out_sum = nullptr) {
    extern __shared__ float smem[];

    int batch_id = blockIdx.z;
    int offset = batch_id * input_stride + blockIdx.x * dim_model;
    float local_sqr_sum = 0;
    bool need_add = out_sum != nullptr;
    for (int i = threadIdx.x; i < dim_model; i += blockDim.x) {
        float v = float(input[offset + i]);
        if (need_add) {
            v += float(input2[offset + i]);
            out_sum[offset + i] = T(v);
        }
        smem[i] = v;
        local_sqr_sum += v * v;
    }
    local_sqr_sum = functions::blockReduceSum<float>(local_sqr_sum) / (float)dim_model;
    local_sqr_sum = rsqrtf(local_sqr_sum + eps);
    for (int i = threadIdx.x; i < dim_model; i += blockDim.x) {
        output[offset + i] = T(smem[i] * local_sqr_sum * float(weight[i]) / scale);
    }
}

__host__ void layernorm(const core::Tensor &input,  // (batch, seq_len, dim_model)
                        const core::Tensor &weight, // (dim_model)
                        core::Tensor *output,       // (batch, seq_len, dim_model)
                        float eps,
                        float scale,
                        bool rms,
                        cudaStream_t stream,
                        const core::Tensor *input2 = nullptr,
                        core::Tensor *out_sum = nullptr) {
    int batch = (input.ndim() == 2) ? 1 : input.size(0);
    int input_stride = (input.ndim() == 2) ? 0 : input.stride(0);
    int seq_len = input.size(-2);
    int dim_model = input.size(-1);

    dim3 gridDim(seq_len, 1, batch);
    dim3 blockDim(std::min(round_up(dim_model, 32), 1024), 1, 1);

    BM_DTYPE_DISPATCH_FLOAT(input.dtype(), {
        auto kernel = BM_KERNEL(layernorm_rms)<scalar_t>;
        int smem_size = dim_model * sizeof(float);
        if (!rms) {
            // TODO tianxin to do
            // kernel = KERNEL_layer_norm_std<scalar_t>;
            // smem_size = dim_model * sizeof(scalar_t) + 4096;
        }
        if (smem_size > 48000) {
            BM_CUDART_ASSERT(cudaFuncSetAttribute(
                kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
        }
        scalar_t *input2_ptr = input2 == nullptr ? nullptr : input2->data<scalar_t>();
        scalar_t *out_sum_ptr = out_sum == nullptr ? nullptr : out_sum->data<scalar_t>();
        kernel<<<gridDim, blockDim, smem_size, stream>>>(
            dim_model, eps, input_stride,
            weight.data<scalar_t>(), input.data<scalar_t>(),
            output->mutable_data<scalar_t>(),
            scale, input2_ptr, out_sum_ptr);
    });
    BM_CUDART_ASSERT(cudaGetLastError());
}

class LayerNorm::impl {
public:
    class QuantImpl;
    class MultiHeadImpl;
    core::Tensor weight;
    core::Tensor bias;
    float eps;
    float scale;
    bool rms{true};
    impl(const core::Context &ctx, unsigned int dim_model, float eps, float scale, core::DataType dtype) :
        weight(ctx.parameter({dim_model}, dtype)), eps(eps), scale(scale) {
    }

    virtual ~impl() = default;
    impl(const impl &) = default;
    impl(impl &&) = default;

    void set_rms(bool b) {
        rms = b;
    }

    virtual core::Tensor forward(const core::Context &ctx,
                                 const core::Tensor &input) { // (batch, seq_len, dim_model)

        BM_ASSERT(input.dtype() == weight.dtype(), "dtype mismatch");
        BM_ASSERT_EQ(input.size(-1), weight.size(0), "dim mismatch");
        BM_ASSERT(input.ndim() >= 2, "input.ndim() must be >= 2");
        BM_ASSERT(input.device() == weight.device(), "Input and weight must be on the same device");
        core::Tensor output = ctx.tensor(input.shape(), input.dtype());
        BM_ASSERT(output.device() == weight.device(), "Output and weight must be on the same device");
        layernorm(input, weight, &output, eps, scale, rms, ctx.current_stream()->ptr);
        return output;
    }

}; // end of class LayerNorm::impl

class LayerNorm::impl::MultiHeadImpl : public LayerNorm::impl {
public:
    MultiHeadImpl(const core::Context &ctx,
                  unsigned int dim_model,
                  float eps,
                  float scale,
                  core::DataType dtype,
                  size_t num_head) :
        LayerNorm::impl(ctx, dim_model, eps, scale, dtype) {
        weight = ctx.parameter({num_head, size_t(dim_model) / num_head}, dtype);
    }

    virtual ~MultiHeadImpl() = default;

}; // end of class LayerNorm::impl::MultiHeadImpl

class LayerNorm::impl::QuantImpl : public LayerNorm::impl {
public:
    QuantImpl(const core::Context &ctx, unsigned int dim_model, float eps, float scale, core::DataType dtype) :
        LayerNorm::impl(ctx, dim_model, eps, scale, dtype) {
    }

    virtual ~QuantImpl() = default;
};

LayerNorm::LayerNorm(const core::Context &ctx,
                     int dim_model,
                     bool quant,
                     float eps,
                     float scale,
                     core::DataType dtype,
                     int num_head) :
    core::Layer() {
    if (num_head > 1) {
        pimpl.reset(new impl::MultiHeadImpl(ctx, dim_model, eps, scale, dtype, num_head));
    } else {
        pimpl.reset(quant ?
                        new impl::QuantImpl(ctx, dim_model, eps, scale, dtype) :
                        new impl(ctx, dim_model, eps, scale, dtype));
    }
    add_parameter("weight", pimpl->weight);
}

LayerNorm::~LayerNorm() = default;

core::Tensor LayerNorm::forward(const core::Context &ctx, const core::Tensor &x) {
    size_t M = x.numel() / x.size(-1);
    core::EventScope event_scope(ctx, "LayerNorm[M=" + std::to_string(M) + "]", 1, 2 * x.nbytes());
    return pimpl->forward(ctx, x);
}

void LayerNorm::set_rms(bool b) {
    pimpl->set_rms(b);
}

void LayerNorm::load_state_dict(const core::Context &ctx,
                                const std::map<std::string, const core::Tensor> &state_dict,
                                const std::string &prefix,
                                bool allow_missing) {
    impl::MultiHeadImpl *p = dynamic_cast<impl::MultiHeadImpl *>(pimpl.get());
    if (p) {
        auto name = prefix + ".weight";
        ctx.load_parameter(&p->weight, name, state_dict, true, core::DistLayout::ROW);
    } else {
        core::Layer::load_state_dict(ctx, state_dict, prefix, allow_missing);
    }
}

} // namespace nn
