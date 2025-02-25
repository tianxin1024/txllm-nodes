#include "backend/embedding.h"
#include "backend/utils.h"
#include <bmengine/functions/init.h>
#include <bmengine/functions/gemm.h>

namespace nn {

// gridDim (seq_len, dim_model / 1024, 1),   blockDim (1024, 1, 1)
template <typename T>
__global__ void BM_KERNEL(embedding)(int begin, int end,
                                     size_t dim_model, float scale,
                                     const int32_t *__restrict__ idx, // (batch, seq_len)
                                     const T *__restrict__ weight,    // (vocab_size, dim_model)
                                     T *__restrict__ out) {           // (batch, seq_len, dim_model)
    int target_id = idx[blockIdx.x];
    int col = blockIdx.y * blockDim.x + threadIdx.x;
    size_t offset = blockIdx.x * dim_model;
    bool in_range = target_id >= begin && target_id < end;
    target_id -= begin;
    if (col < dim_model) {
        out[offset + col] = in_range ? T(float(weight[size_t(target_id) * dim_model + col]) * scale) : T(0.);
    }
}

static __host__ void embedding(const core::Tensor &idx,
                               const core::Tensor &weight,
                               const core::Tensor &out,
                               int begin,
                               int end,
                               float scale,
                               cudaStream_t stream) {
    int seq_len = idx.numel();
    int dim_model = weight.size(1);
    int threads = round_up_thread(dim_model);
    dim3 gridDim(seq_len, round_up(dim_model, threads) / threads);
    dim3 blockDim(threads);

    BM_DTYPE_DISPATCH_FLOAT(weight.dtype(), {
        BM_KERNEL(embedding)<scalar_t><<<gridDim, blockDim, 0, stream>>>(
            begin, end, dim_model, scale,
            idx.data<int32_t>(), weight.data<scalar_t>(), out.data<scalar_t>());
    });
    BM_CUDART_ASSERT(cudaGetLastError());
}

class RawEmbedding::impl {
public:
    class NormalImpl;
    class ParallelImpl;
    class RowParallelImpl;
    float logit_scale{1.}; // For Cohere model
    virtual ~impl() = default;

    virtual core::Tensor &get_weight() = 0;

    virtual void set_scale_weights(bool b) = 0;
    virtual void set_scale_factor(float b) = 0;
    virtual void set_logit_scale(float s) {
        logit_scale = s;
    }

    virtual core::Tensor forward(const core::Context &ctx,
                                 const core::Tensor &ids // (seq_len)
                                 ) = 0;

    virtual core::Tensor projection(const core::Context &ctx,
                                    const core::Tensor &input // (seq_len, dim_model)
                                    ) = 0;

}; // end of class RawEmbedding

class RawEmbedding::impl::NormalImpl : public RawEmbedding::impl {
public:
    core::Tensor weight;
    unsigned int dim_model;
    core::DataType dtype;
    unsigned int begin;
    unsigned int end;
    float scale_factor;
    NormalImpl(const core::Context &ctx,
               unsigned int vocab_size,
               unsigned int dim_model,
               bool scale_weights,
               core::DataType dtype) :
        weight(ctx.parameter({vocab_size, dim_model}, dtype)),
        dim_model(dim_model),
        dtype(dtype),
        begin(0),
        end(vocab_size),
        scale_factor(scale_weights ? 1.0 / sqrtf(dim_model) : 1.0) {
    }

    core::Tensor &get_weight() {
        return weight;
    }
    void set_scale_weights(bool b) {
        scale_factor = (b ? 1.0 / sqrtf(dim_model) : 1.0);
    }
    void set_scale_factor(float b) {
        scale_factor = b;
    }

    core::Tensor forward(const core::Context &ctx,
                         const core::Tensor &ids) {
        BM_ASSERT(ids.dtype() == core::DataType::kInt32, "ids dtype mismatch");
        BM_ASSERT(ids.ndim() == 1 || ids.ndim() == 2, "ids must be 1d or 2d");

        auto out_shape = ids.shape();
        out_shape.push_back(dim_model);
        core::Tensor ret = ctx.tensor(out_shape, dtype);
        embedding(ids, weight, ret, begin, end, scale_factor, ctx.current_stream()->ptr);
        return ret;
    }

    core::Tensor projection(const core::Context &ctx,
                            const core::Tensor &input) {
        functions::Gemm local_gemm(ctx, dtype, false, true, scale_factor * logit_scale);
        if (ctx.high_precision() >= 1) {
            local_gemm.set_compute_type(CUBLAS_COMPUTE_32F);
        }
        std::cout << "seq_len: " << input.size(0) << " ,  dim_model: " << dim_model << std::endl;
        std::cout << "input: " << input.numel() << std::endl;
        std::cout << "weight: " << weight.numel() << std::endl;
        auto logits = local_gemm.forward(ctx,
                                         input, // (seq_len, dim_model)
                                         weight // (vocab_size, dim_model)T
        );                                      // (seq_len, vocab_size)
        // std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>>> current tianx >>>>>>>>>>>>>>>" << std::endl;
        return logits;
    }

}; // end of class RawEmbedding::impl::NormalImpl

class RawEmbedding::impl::RowParallelImpl : public RawEmbedding::impl::NormalImpl {
public:
    unsigned int vocab_size;
    RowParallelImpl(const core::Context &ctx,
                    unsigned int vocab_size,
                    unsigned int dim_model,
                    bool scale_weights,
                    core::DataType dtype) :
        NormalImpl(ctx, vocab_size, dim_model, scale_weights, dtype), vocab_size(vocab_size) {
    }

    void load_state_dict(const core::Context &ctx,
                         const std::map<std::string, const core::Tensor> &state_dict,
                         const std::string &prefix,
                         bool allow_missing) {
        unsigned int round_size = round_up(vocab_size, 128);
        unsigned int part_size = round_size / ctx.world_size();
        begin = ctx.rank() * part_size;
        end = begin + part_size;

        auto it = state_dict.find(prefix + ".weight");
        BM_ASSERT(it != state_dict.end(), "Weight not found: " + prefix + ".weight");
        auto part_src = it->second.slice_dim0(begin, std::min(end, vocab_size));
        weight = ctx.tensor({part_size, dim_model}, dtype);
        functions::zeros_(ctx, weight);
        auto weight_t = weight.slice_dim0(0, part_src.size(0));
        ctx.assign_or_copy(&weight_t, &part_src);
    }
};

RawEmbedding::RawEmbedding(const core::Context &ctx,
                           int dim_model,
                           int vocab_size,
                           bool scale_weights,
                           core::DataType dtype,
                           bool parallel) :
    core::Layer() {
    int row_parallel = utils::get_int_env("CPM_EMB_ROW_PAR", 1);
    if (parallel) {
        pimpl.reset(new impl::RowParallelImpl(ctx, vocab_size, dim_model, scale_weights, dtype));
    } else {
        pimpl.reset(new impl::NormalImpl(ctx, vocab_size, dim_model, scale_weights, dtype));
    }
    add_parameter("weight", pimpl->get_weight());
}

RawEmbedding::~RawEmbedding() = default;

void RawEmbedding::set_scale_weights(bool b) {
    pimpl->set_scale_factor(b);
}
void RawEmbedding::set_scale_factor(float b) {
    pimpl->set_scale_factor(b);
}

void RawEmbedding::set_logit_scale(float b) {
    pimpl->set_logit_scale(b);
}

core::Tensor RawEmbedding::forward(const core::Context &ctx,
                                   const core::Tensor &input) { // (seq_len)
    return pimpl->forward(ctx, input);
}

core::Tensor RawEmbedding::projection(const core::Context &ctx,
                                      const core::Tensor &input) { // (seq_len, dim_model)
    return pimpl->projection(ctx, input);
}

void RawEmbedding::load_state_dict(const core::Context &ctx,
                                   const std::map<std::string, const core::Tensor> &state_dict,
                                   const std::string &prefix,
                                   bool allow_missing) {
    auto row_ptr = dynamic_cast<impl::RowParallelImpl *>(pimpl.get());
    if (row_ptr) {
        row_ptr->load_state_dict(ctx, state_dict, prefix, allow_missing);
    } else {
        core::Layer::load_state_dict(ctx, state_dict, prefix, allow_missing);
    }
}

} // namespace nn
