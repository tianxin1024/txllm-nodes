#include "backend/block.h"
#include "backend/layernorm.h"

namespace nn {

class EncoderLayer::impl {
public:
    class CohereImpl;
    LayerNorm ln_attn, ln_ff;
    // Attention attn;
    // FeedForward ff;
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
        // attn(ctx, cfg, quant_config, parallel),
        // ff(ctx, cfg, quant_config, parallel),
        scale(cfg.model_type == "cpm_dragonfly" ? sqrtf(float(cfg.num_layers)) / cfg.scale_depth : 1.0),
        scale_residual(cfg.model_type == "cpm_dragonfly" ? false : true),
        mask_modules(cfg.mask_modules[ctx.current_layer()]),
        parallel(parallel),
        dtype(cfg.dtype),
        dev(ctx.active_device_idx()) {
    }

    virtual ~impl() = default;

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
        // add_submodule("attn", pimpl->attn);
    }
    if (!mask_modules[1]) {
        if (!is_cohere) {
            add_submodule("ln_ff", pimpl->ln_ff);
        }
        // add_submodule("ff", pimpl->ff);
    }
}

EncoderLayer::~EncoderLayer() = default;

} // namespace nn
