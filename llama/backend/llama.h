#pragma once
#include "backend/model.h"
// #include "backend/model_context.h"
#include <bmengine/core/core.h>
#include <bmengine/functions/all.h>
#include "backend/layernorm.h"
#include "backend/embedding.h"
#include "backend/block.h"

namespace model {

using namespace bmengine;

class ModelContext;

class LLaMALike : public ModelBase {
public:
    explicit LLaMALike(ModelConfig model_config) :
        ModelBase(model_config) {
    }

    // virtual core::Tensor forward(ModelContext &ctx,
    //                              const core::Tensor &ids,        // int32 (batch, len_q)
    //                              const core::Tensor &pos_ids,    // int32 (batch, len_ext)
    //                              const core::Tensor &seqlens_q,  // int32 (batch)
    //                              const core::Tensor &seqlens_kv, // int32 (batch)
    //                              const core::Tensor &mask,       // int8 (batch, len_q, len_buf)
    //                              const core::Tensor &placement,
    //                              const core::Tensor &hidden_pass, // half (batch, len_q, dim_model)
    //                              core::Tensor *hidden_ptr = nullptr) = 0;

    virtual core::Tensor encode(ModelContext &ctx,
                                const core::Tensor &ids,
                                const core::Tensor &pos_ids,
                                const core::Tensor &seqlens_q,
                                const core::Tensor &seqlens_kv,
                                const core::Tensor &mask,
                                const core::Tensor &placement,
                                const core::Tensor &hidden_pass,
                                bool ln_output = true) = 0;

    virtual core::Tensor get_input_embeddings(ModelContext &ctx, const core::Tensor &ids) = 0;

    virtual core::Tensor get_logits(ModelContext &ctx, const core::Tensor &hidden, bool ln_input) = 0;

    // virtual functions::ModuleList<nn::EncoderLayer> &get_encoder() = 0;

    // set to all encoder layers
    void set_mask_modules(const std::vector<std::vector<bool>> &mask_modules);

    // set to all encoder layers
    void set_residual_scale(float residual_scale);

}; // end of class LLaMALike

class LLaMA : public LLaMALike {
private:
    bool parallel;
    bool tie_lm_head;

    functions::ModuleList<nn::EncoderLayer> encoder;
    nn::LayerNorm ln_after_enc;
    nn::RawEmbedding lm_head;
    nn::RawEmbedding token_embedding;
    // std::unique_ptr<nn::RopePreparer> rope_preparer;

    BM_LAYER_DEF_PUBLIC(LLaMA);

    LLaMA(core::Context &ctx, ModelConfig model_config, QuantConfig quant_config, bool parallel = false);

    // core::Tensor forward(ModelContext &ctx,
    //                      const core::Tensor &ids,        // int32 (batch, len_q)
    //                      const core::Tensor &pos_ids,    // int32 (batch, len_ext)
    //                      const core::Tensor &seqlens_q,  // int32 (batch)
    //                      const core::Tensor &seqlens_kv, // int32 (batch)
    //                      const core::Tensor &mask,       // int8 (batch, len_q, len_buf)
    //                      const core::Tensor &placement,
    //                      const core::Tensor &hidden_pass, // half (batch, len_q, dim_model)
    //                      core::Tensor *hidden_ptr = nullptr) override;

    core::Tensor encode(ModelContext &ctx,
                        const core::Tensor &ids,        // int32 (batch, len_q)
                        const core::Tensor &pos_ids,    // int32 (batch, len_ext)
                        const core::Tensor &seqlens_q,  // int32 (batch)
                        const core::Tensor &seqlens_kv, // int32 (batch)
                        const core::Tensor &mask,       // int8 (batch, len_q, len_buf)
                        const core::Tensor &placement,
                        const core::Tensor &hidden_pass, // half (batch, len_q, lem_model)
                        bool ln_output = true) override;

    core::Tensor get_input_embeddings(ModelContext &ctx, const core::Tensor &ids) override;

    core::Tensor get_logits(ModelContext &ctx, const core::Tensor &hidden, bool ln_input) override;

    // core::Tensor get_logits(ModelContext &ctx, const core::Tensor &hidden, bool ln_input) override;
};

} // namespace model
