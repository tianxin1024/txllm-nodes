#include "backend/block.h"
#include "backend/layernorm.h"
#include "backend/attention.h"
#include "backend/feedforward.h"
#include "backend/model_context.h"
#include "backend/utils.h"
#include <bmengine/functions/element.h>
#include "private/allocator.h"
#include <bmengine/logger/std_log_op.hpp>

namespace nn {

using namespace bmengine;
using model::ModelContext;
using bmengine::core::Tensor;

class EncoderLayer::impl {
public:
    class CohereImpl;
    LayerNorm ln_attn, ln_ff;
    Attention attn;
    FeedForward ff;
    float scale;
    bool scale_residual;
    std::vector<bool> mask_modules;
    std::string prefix;
    int dev;
    bool parallel;
    core::DataType dtype;

    impl(const core::Context &ctx,
         model::ModelConfig cfg,
         model::QuantConfig quant_config,
         bool parallel) :
        ln_attn(ctx, cfg.dim_model, quant_config.fuse_ln_attn(), cfg.eps, 1.0, cfg.dtype),
        ln_ff(ctx, cfg.dim_model, quant_config.fuse_ln_ff(), cfg.eps, 1.0, cfg.dtype),
        attn(ctx, cfg, quant_config, parallel),
        ff(ctx, cfg, quant_config, parallel),
        scale(cfg.model_type == "cpm_dragonfly" ? sqrtf(float(cfg.num_layers)) / cfg.scale_depth : 1.0),
        scale_residual(cfg.model_type == "cpm_dragonfly" ? false : true),
        mask_modules(cfg.mask_modules[ctx.current_layer()]),
        parallel(parallel),
        dtype(cfg.dtype),
        dev(ctx.active_device_idx()) {
    }

    virtual ~impl() = default;

    virtual core::Tensor forward(const core::Context &ctx,
                                 const core::Tensor &inp,           // (batch, len_q, dim_model)
                                 const core::Tensor &mask,          // (batch, len_q, len_buf)
                                 const core::Tensor &position_bias, // if relative (batch, num_head, len_q, len_buf) else if rotary (batch, len_q)
                                 const core::Tensor &seqlens_q,     // (batch)
                                 const core::Tensor &seqlens_kv,    // (batch)
                                 const core::Tensor *past_k,        // (batch, num_head, len_buf, dim_head)
                                 const core::Tensor *past_v,        // (batch, num_head, len_buf, dim_head)
                                 const core::Tensor *block_table,   // (batch, num_head, len_buf, dim_head)
                                 const core::Tensor *placement) {   // (batch, len_q) int32

        ModelContext *m_ctx = dynamic_cast<ModelContext *>(const_cast<core::Context *>(&ctx));
        core::Tensor ret;

        if (!mask_modules[0]) {
            auto ln_out = ln_attn(ctx, inp);
            std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> block forward" << std::endl;
            if (m_ctx && m_ctx->is_calc_act_scales()) {
                m_ctx->update_act_scale(ln_attn.prefix + ".max_out", ln_out);
            }

            // ret = attn(ctx, ln_out, mask, position_bias,
            //            seqlens_q, seqlens_kv, past_k, past_v,
            //            block_table, placement, nullptr);
        }
    }

}; // end of class EncoderLayer::impl

class EncoderLayer::impl::CohereImpl : public EncoderLayer::impl {
public:
    CohereImpl(const core::Context &ctx,
               model::ModelConfig config,
               model::QuantConfig quant_config,
               bool parallel) :
        EncoderLayer::impl(ctx, config, quant_config, parallel) {
        ln_attn.set_rms(false);
    }

    core::Tensor dyn_forward(const core::Context &ctx,
                             const core::Tensor &input,
                             const core::Tensor &position) {
        ModelContext *m_ctx = ModelContext::cast(ctx);

        const Tensor &residual = input;
        Tensor hidden_states = ln_attn(ctx, input);
        std::cout << "input: " << input << endl;
        std::cout << "hidden_states: " << hidden_states << std::endl;
        Tensor attn_out = attn.dyn_rag_forward(*m_ctx, hidden_states, position);
        Tensor mlp_out = ff.forward(ctx, hidden_states);

        // Add everything together
        functions::BinaryElementwiseOp add_op(ctx, functions::BinaryElementwiseOp::Add);
        Tensor ret = add_op.forward(ctx, attn_out, mlp_out);
        if (parallel) {
            ret = ctx.reduce_sum(ret, dtype);
        }
        add_op.inplace(ctx, ret, residual);
        return ret;
    }

}; // end of class EncoderLayer::impl::CohereImpl

EncoderLayer::EncoderLayer(const core::Context &ctx,
                           model::ModelConfig config,
                           model::QuantConfig quant_config,
                           bool parallel) :
    core::Layer(),
    parallel(parallel) {
    bool is_cohere = config.model_type == "cohere";
    if (is_cohere) {
        pimpl.reset(new impl::CohereImpl(ctx, config, quant_config, parallel));
    } else {
        pimpl.reset(new impl(ctx, config, quant_config, parallel));
    }
    this->layer_id = ctx.current_layer();
    this->dev = ctx.active_device_idx();
    this->output_dev = dev;

    auto mask_modules = config.mask_modules[this->layer_id];
    if (!mask_modules[0]) {
        add_submodule("ln_attn", pimpl->ln_attn);
        add_submodule("attn", pimpl->attn);
    }
    if (!mask_modules[1]) {
        if (!is_cohere) {
            add_submodule("ln_ff", pimpl->ln_ff);
        }
        add_submodule("ff", pimpl->ff);
    }
}

EncoderLayer::~EncoderLayer() = default;

core::Tensor EncoderLayer::forward(const core::Context &ctx,
                                   const core::Tensor &inp,           // (batch, len_q, dim_model)
                                   const core::Tensor &mask,          // (batch, len_q, len_buf)
                                   const core::Tensor &position_bias, // if relative (batch, num_head, len_q, len_buf) else if rotary (batch, len_q)
                                   const core::Tensor &seqlens_q,     // (batch)
                                   const core::Tensor &seqlens_kv,    // (batch)
                                   const core::Tensor *past_k,        // (batch, num_head, len_buf, dim_head)
                                   const core::Tensor *past_v,        // (batch, num_head, len_buf, dim_head)
                                   const core::Tensor *block_table,   // (batch, blocks_per_seq)
                                   const core::Tensor *placement) {   // (batch, len_q) int32
    std::cout << ">>>>>>>>>>>>>>>>>>>>>> EncoderLayer::forward()" << std::endl;
    size_t M = inp.numel() / inp.size(-1);
    core::EventScope event_scope(ctx, logger::str_cat("EncoderLayer[M=", M, "]"), 1);
    {
        bool switched = ctx.switch_to_device(dev);
        if (switched && ctx.debug() >= 2) {
            std::cerr << "EncoderLayer[" << layer_id << "]::forward() switch to device " << dev << std::endl;
        }
    }
    // copy and cache to current layer's device, if necessary
    const core::Tensor *p_input = ctx.identity(&inp, "EncoderInput");
    const core::Tensor *p_mask = ctx.identity(&mask, "EncoderMask");
    const core::Tensor *p_pos_bias = ctx.identity(&position_bias, "EncoderPosBias");
    const core::Tensor *p_placement = ctx.identity(placement, "EncoderPosBias");
    const core::Tensor *p_seqlens_q = ctx.identity(&seqlens_q, "EncoderQSeqLens");
    const core::Tensor *p_seqlens_kv = ctx.identity(&seqlens_kv, "EncoderKVSeqLens");

    impl::CohereImpl *cohere = dynamic_cast<impl::CohereImpl *>(pimpl.get());
    if (cohere) {
        return cohere->dyn_forward(ctx, *p_input, *p_pos_bias);
    }
    core::Tensor tensor = pimpl->forward(ctx,
                                         *p_input,
                                         *p_mask,
                                         *p_pos_bias,
                                         *p_seqlens_q,
                                         *p_seqlens_kv,
                                         past_k,
                                         past_v,
                                         block_table,
                                         p_placement);

    if (output_dev != dev) {
        // at last layer, switch back to device 0, and copy output
        ctx.switch_to_device(output_dev);
        if (ctx.debug() >= 2) {
            std::cerr << "EncoderLayer[" << layer_id
                      << "]::forward(), last layer, switch back device to " << output_dev
                      << std::endl;
        }
        tensor = *ctx.identity(&tensor, "EncoderFinalOutput");
    }
    return std::move(tensor);
}

void EncoderLayer::load_state_dict(const core::Context &ctx,
                                   const std::map<std::string, const core::Tensor> &state_dict,
                                   const std::string &prefix,
                                   bool allow_missing) {
    using BinaryOp = bmengine::functions::BinaryElementwiseOp;
    BinaryOp mul_op(ctx, BinaryOp::Mul);
    BinaryOp div_op(ctx, BinaryOp::Div);

    // ModelContext *m_ctx = dynamic_cast<ModelContext *>(const_cast<core::Context *>(&ctx));

    pimpl->prefix = prefix;
    // if (m_ctx && m_ctx->smooth_quant_alpha() > 0) {
    // }
    core::Layer::load_state_dict(ctx, state_dict, prefix, allow_missing);
    int freeze_mem = utils::get_int_env("FREEZE_MEM_EACH_LAYER", 0);
    if (freeze_mem) {
        ctx.get_allocator()->freeze_model_memory();
    }
}

} // namespace nn
