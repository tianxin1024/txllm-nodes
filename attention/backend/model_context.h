#pragma once
#include <bmengine/core/core.h>
#include "backend/model.h"
#include "backend/model_config.h"

namespace model {

using namespace bmengine;

/**
 * Extend Context to hold more info for LLM model inference
 */
class ModelContext : public bmengine::core::Context {
public:
    const ModelConfig cfg;

private:
    const ModelBase &model_;
    bool parallel_;

public:
    ModelContext(bmengine::core::Context &&ctx,
                 const ModelBase &md,
                 int batch_size = 1,
                 bool parallel = false,
                 bool BSHD = false);
    ~ModelContext() override;
    ModelContext(ModelContext &&) = default;

}; // end of class ModelContext

} // namespace model
