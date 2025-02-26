#pragma once

#include <bmengine/core/core.h>
#include <bmengine/core/context.h>
#include <vector>

namespace model {
class ModelContext;
} // namespace model

namespace beam_utility {

using namespace bmengine;
template <class>
class BeamBufferManager;

core::Tensor log_softmax_bias(const core::Context &ctx,
                              const core::Tensor &logits, // half (batch, dim_logits)
                              const core::Tensor &bias);  // float32 (batch)

void log_softmax_bias(const core::Context &ctx,
                      const core::Tensor &logits, // half (batch, dim_logits)
                      const core::Tensor &bias,   // float32 (batch)
                      float temperature,
                      core::Tensor *out);

core::Tensor log_softmax_bias(const core::Context &ctx,
                              const core::Tensor &logits, // half (batch, dim_logits)
                              const core::Tensor &bias,   // float32 (batch)
                              float temperature);

core::Tensor gather_logits(
    const core::Context &ctx, const core::Tensor &indexes, const core::Tensor &logits);

core::Tensor apply_gumbel_softmax(
    const core::Context &ctx, curandGenerator_t &gen, const core::Tensor &logits);

void scatter_update(const core::Context &ctx,
                    const std::vector<float> &value,
                    const std::vector<int32_t> &token_ids, // indices[i]
                    const std::vector<int32_t> &batch_ids, // indices[0]
                    core::Tensor &logits);

void apply_beam_repetition_penalty(model::ModelContext &ctx,
                                   const BeamBufferManager<int> &bm,
                                   const std::vector<int> &hypotheses_last_pos,
                                   float ngram_penalty,
                                   float repetition_penalty,
                                   core::Tensor *logits_all);

void random_sampler_gpu(const core::Context &ctx,
                        curandGenerator_t &gen,
                        core::Tensor &probs,  // (..., n_classes)
                        core::Tensor &select, // (...)
                        float top_p = 1.0f,
                        int top_k = 0,
                        int num_samples = 1);
} // namespace beam_utility
