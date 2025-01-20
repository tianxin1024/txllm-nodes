#pragma once
#include <memory>
#include <mutex>
#include <functional>
#include <thread>
#include <queue>
#include <condition_variable>
#include <bmengine/core/core.h>
#include "backend/dyn_batch_context.h"
#include "backend/model.h"
#include "backend/generator.h"
#include "backend/utils.h"

namespace batch_generator {

using namespace bmengine;
using model::DynBatchConfig;
using model::ModelBase;

struct SearchTask_ {
    std::vector<int32_t> input_tokens;
    int beam_size;
    int max_length;
    float presence_penalty;
    float repetition_penalty;
    bool diverse;
    int seed;
    float temperature;
    int num_results;
    float top_p;
    int top_k;
    int top_topprobs;
    int stream{0}; // 0: non stream; 1. single stream result; 2: multiple stream result
    utils::TSQueue<generator::SearchResults> res_queue{INT_MAX};

}; // end of struct SearchTask_

typedef std::shared_ptr<SearchTask_> SearchTask;

class TaskQueue {
    int max_size_;
    std::mutex mutex_;
    std::condition_variable can_push_cond_;
    std::condition_variable can_pop_cond_;
    std::queue<SearchTask> queue_;
    volatile bool stopping_{false};

public:
    explicit TaskQueue(int max_size);

    void stop();
    size_t size();

}; // end of class TaskQueue

class BatchGenerator {
    DynBatchConfig config;

    ModelBase *model_;
    model::ModelBase *par_model_;
    bmengine::core::Engine *engine_;

    TaskQueue queue_;

    std::shared_ptr<std::thread> thread_;
    volatile bool stopping_{false};

    std::mutex mutex_;
    std::condition_variable done_cond_;
    std::condition_variable stop_cond_;
    volatile bool stopped_{false};

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