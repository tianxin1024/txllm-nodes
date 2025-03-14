#pragma once
#include <memory>
#include <mutex>
#include <functional>
#include <thread>
#include <queue>
#include <condition_variable>
#include <bmengine/core/core.h>
#include "backend/dyn_batch_context.h"
#include "backend/rag_buffer_context.h"
#include "backend/model.h"
#include "backend/generator.h"
#include "backend/utils.h"
#include "backend/llama.h"

namespace batch_generator {

using namespace bmengine;
using model::DynBatchConfig;
using model::ModelBase;
using model::ModelContext;

struct SearchTask_ {
    std::vector<int32_t> input_tokens;
    int beam_size;
    int max_length;
    float presence_penalty;
    float repetition_penalty;
    float ngram_penalty;
    bool diverse;
    int seed;
    float temperature;
    int num_results;
    float top_p;
    int top_k;
    int top_logprobs;
    int stream{0}; // 0: non stream; 1. single stream result; 2: multiple stream result
    utils::TSQueue<generator::SearchResults> res_queue{INT_MAX};
    std::function<void(const generator::SearchResults &results)> callback;
    volatile bool canceled{false};
    long begin_ts{0};
    long first_token_delay_ms{0};

    std::vector<int> position_ids;           // passed-in position ids of 'PROMPT'
    bmengine::core::Tensor input_embeddings; // passed-in embeddings of 'PROMPT', device=CPU

public:
    void finish(generator::SearchResults &&results);
    void update_stream(const generator::SearchResults &results);
    size_t input_length() const {
        return input_tokens.size();
    }
    size_t full_length() const {
        return input_tokens.size() + size_t(beam_size * max_length);
    }
    bool is_random() const {
        return top_p < 1. or top_k > 0;
    }

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

    bool push(SearchTask task, bool wait, bool notify = true);

    bool empty();
    std::vector<SearchTask> pop_multi(int limit, bool wait, int require, int max_token, bool pre_alloc);
    void stop();
    size_t size();

}; // end of class TaskQueue

template <class, class>
class SearcherImplV1;

class BatchGenerator {
    friend class SearcherImplV1<int, int>;

    DynBatchConfig config;

    ModelBase *model_;
    std::vector<model::ModelBase *> par_models_;
    bmengine::core::Engine *engine_;

    TaskQueue queue_;
    int active_size_;

    std::shared_ptr<std::thread> thread_;
    volatile bool stopping_{false};

    std::mutex mutex_;
    std::condition_variable done_cond_;
    std::condition_variable stop_cond_;
    volatile bool stopped_{false};

public:
    BatchGenerator(DynBatchConfig config,
                   std::vector<model::ModelBase *> par_models,
                   bmengine::core::Engine *engine);
    ~BatchGenerator();

    model::LLaMALike *llama_model() {
        return dynamic_cast<model::LLaMALike *>(model_);
    }

    bool submit(SearchTask task, bool wait, bool notify = true);

    int queue_size() {
        return queue_.size();
    }

    void start();
    void stop();
    void run();

}; // end of class Batchenerator

} // namespace batch_generator
