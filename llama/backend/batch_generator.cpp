#include "backend/batch_generator.h"

namespace batch_generator {

BatchGenerator::BatchGenerator(DynBatchConfig config,
                               model::ModelBase *par_model,
                               bmengine::core::Engine *engine) :
    config(config),
    par_model_(par_model),
    engine_(engine),
    queue_(config.task_queue_size) {
    BM_ASSERT(config.max_total_token % 64 == 0, "max_total_token should align to 64");
    BM_ASSERT(!par_model_.empty(), "No model");
}

} // namespace batch_generator
