#include "backend/llama.h"
#include "backend/allocate_utils.h"

namespace model {

LLaMA::LLaMA(core::Context &ctx, ModelConfig model_config, QuantConfig quant_config, bool parallel) :
    LLaMALike(model_config),
    ln_after_enc(ctx, dim_model, false, eps,
                 model_config.model_type == "cpm_dragonfly" ? dim_model / model_config.dim_model_base : 1.0, dtype),
    token_embedding(ctx, dim_model, vocab_size, false, dtype, parallel),
    lm_head(ctx, dim_model, vocab_size, false, dtype, parallel),
    parallel(parallel),
    tie_lm_head(model_config.tie_lm_head) {
    std::cout << "LLaMA::LLaMA" << std::endl;
    std::vector<int> devices = partition_layer_devices(ctx, num_layers);

    for (int i = 0; i < num_layers; i++) {
        ctx.switch_to_device(devices[i]);

        ctx.set_current_layer(i);

        encoder.append(ctx, model_config, quant_config, parallel);
        encoder[i].output_dev = devices[i];
    }
    encoder[num_layers - 1].output_dev = 0;
    ctx.switch_to_device(0);

    if (model_config.model_type == "cpm_dragonfly") {
        token_embedding.set_scale_factor(model_config.scale_emb);
    }

    add_submodule("layers", encoder);
    add_submodule("output_layernorm", ln_after_enc);
    add_submodule("token_embedding", token_embedding);

    if (!tie_lm_head) {
        add_submodule("lm_head", lm_head);
    }

    if (model_config.model_type == "cohere") {
        tie_lm_head = true;
        token_embedding.set_logit_scale(model_config.logit_scale);
        ln_after_enc.set_rms(false);
    }
}

} // namespace model
