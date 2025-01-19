#pragma once
#include <memory>
#include <functional>
#include <queue>
#include <condition_variable>
#include <bmengine/core/core.h>
#include "backend/dyn_batch_context.h"
#include "backend/model.h"

namespace batch_generator {

using namespace bmengine;
using model::DynBatchConfig;
using model::ModelBase;

struct SearchTask_ {
    std::vector<int32_t> input_tokens;
}; // end of struct SearchTask_

class TaskQueue {
    int max_size_;
    std::mutex mutex_;
    std::condition_variable can_push_cond_;
    std::condition_variable can_pop_cond_;
    std::queue<SearchTask> queue_;

public:
    explicit TaskQueue(int max_size);
}; // end of class TaskQueue

class BatchGenerator {
    DynBatchConfig config;

    ModelBase *model_;
    model::ModelBase *par_model_;
    bmengine::core::Engine *engine_;

    TaskQueue queue_;

public:
    BatchGenerator(DynBatchConfig config,
                   model::ModelBase *par_models,
                   bmengine::core::Engine *engine);
    ~BatchGenerator();

    void start();
    void stop();
    void run();

}; // end of class Batchenerator

} // namespace batch_generator
