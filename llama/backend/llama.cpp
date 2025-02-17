#include "backend/llama.h"
#include "backend/allocate_utils.h"
#include "backend/model_context.h"
#include "backend/utils.h"

namespace model {

LLaMA::LLaMA(core::Context &ctx, ModelConfig model_config, QuantConfig quant_config, bool parallel) :
    LLaMALike(model_config),
    ln_after_enc(ctx, dim_model, false, eps,
                 model_config.model_type == "cpm_dragonfly" ? dim_model / model_config.dim_model_base : 1.0, dtype),
    token_embedding(ctx, dim_model, vocab_size, false, dtype, parallel),
    lm_head(ctx, dim_model, vocab_size, false, dtype, parallel),
    parallel(parallel),
    tie_lm_head(model_config.tie_lm_head) {
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

    add_submodule("output_layernorm", ln_after_enc);
    add_submodule("token_embedding", token_embedding);
    add_submodule("layers", encoder);

    if (!tie_lm_head) {
        add_submodule("lm_head", lm_head);
    }

    if (model_config.model_type == "cohere") {
        tie_lm_head = true;
        token_embedding.set_logit_scale(model_config.logit_scale);
        ln_after_enc.set_rms(false);
    }
}

core::Tensor LLaMA::encode(ModelContext &ctx,
                           const core::Tensor &ids,     // int32 (len_q)
                           const core::Tensor &pos_ids, // int32 (len_q)
                           const core::Tensor &seqlens_q,
                           const core::Tensor &seqlens_kv,
                           const core::Tensor &mask, // int8 (len_q, len_buf)
                           const core::Tensor &placement,
                           const core::Tensor &hidden_pass, // half (batch, len_q, dim_model)
                           bool ln_output) {
    std::cout << ">>>>>>>>>>>>>>>>>>> llama LLaMA::encode " << std::endl;
    std::cout << ">>>>>>>>>>>>>>>>>> input ids: " << ids.numel() << std::endl;
    ctx.set_current_layer(-1);
    Tensor hidden;
    if (hidden_pass.empty()) {
        std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> llama encode 0 >>>>>>>>>>>>>>>" << std::endl;
        hidden = token_embedding(ctx, ids);
        std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> llama encode 01 >>>>>>>>>>>>>>>" << std::endl;
    } else {
        hidden = functions::typecast(ctx, hidden_pass, dtype);
    }
    // if (rope_preparer && ctx.dyn_batch()) {
    //     auto &rope_cache = ctx.dyn_batch()->rope_cache;
    //     std::tie(rope_cache.cos, rope_cache.sin) = rope_preparer->forward(ctx, pos_ids);
    // }

    std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> llama encode 1 >>>>>>>>>>>>>>>" << std::endl;
    bool dual_stream = utils::get_int_env("DUAL_STREAM", 0) > 0 && ctx.world_size() > 1;
    std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> llama encode 1 >>>>>>>>>>>>>>>" << std::endl;
    int dual_stream_thres = utils::get_int_env("DUAL_STREAM_THRESHOLD", 1024);
    std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> llama encode 1 >>>>>>>>>>>>>>>" << std::endl;
    if (dual_stream && ctx.get_compute_capability() > 80 && ids.size(0) > dual_stream_thres) {
        std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> llama encode 1 >>>>>>>>>>>>>>>" << std::endl;
        // hidden = EncoderLayer::dual_stream_encode(ctx, encoder, hidden, pos_ids);
    } else {
        std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> llama encode 2 >>>>>>>>>>>>>>>" << std::endl;
        int debug_layer = utils::get_int_env("CPM_DEBUG_LAYER", -1);
        int debug_layer_level = utils::get_int_env("CPM_DEBUG_LAYER_LEVEL", 2);
        int event_level = utils::get_int_env("CPM_DEBUG_LAYER_EV_LEVEL", debug_layer_level);
        for (int i = 0; i < num_layers; i++) {
            ctx.set_current_layer(i);
            auto org_debug_level = ctx.debug();
            if (i == debug_layer && ctx.rank() == 0) {
                ctx.enable_debug(debug_layer_level);
                ctx.set_event_level(event_level);
            }
            hidden = encoder[i](ctx, hidden, mask, pos_ids, seqlens_q, seqlens_kv,
                                ctx.buf_k(i), ctx.buf_v(i), ctx.block_table(i),
                                &placement);
        }
    }
    ctx.set_current_layer(-1);
}

core::Tensor LLaMA::get_input_embeddings(ModelContext &ctx,
                                         const core::Tensor &ids) {
    ctx.set_current_layer(-1);
    return token_embedding.forward(ctx, ids);
}

} // namespace model
