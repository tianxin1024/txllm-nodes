#include "backend/embedding.h"
#include "backend/utils.h"
#include <bmengine/functions/init.h>

namespace nn {

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
