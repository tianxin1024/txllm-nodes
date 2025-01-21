#include "backend/batch_generator.h"
#include "backend/model_context.h"

namespace batch_generator {

typedef std::unique_lock<std::mutex> Lock;

TaskQueue::TaskQueue(int max_size) :
    max_size_(max_size) {
}

void TaskQueue::stop() {
    Lock lock(mutex_);
    stopping_ = true;
    can_pop_cond_.notify_one();
}

BatchGenerator::BatchGenerator(DynBatchConfig config,
                               model::ModelBase *par_model,
                               bmengine::core::Engine *engine) :
    config(config),
    par_model_(par_model),
    engine_(engine),
    queue_(config.task_queue_size) {
    BM_ASSERT(config.max_total_token % 64 == 0, "max_total_token should align to 64");
    // BM_ASSERT(!par_model_.empty(), "No model");
    // BM_ASSERT(par_model_ == nullptr, "No model");
    model_ = par_model_;
}

BatchGenerator::~BatchGenerator() {
    if (!stopping_) {
        stop();
    }
}

using namespace std::chrono_literals;
void BatchGenerator::stop() {
    stopping_ = true;
    queue_.stop();
    if (thread_) {
        int max_retry = 20;
        for (int i = 0; i < 20; ++i) {
            if (stopped_) {
                break;
            }
            Lock lock(mutex_);
            if (stop_cond_.wait_for(lock, 100ms) == std::cv_status::timeout && i > 0) {
                std::cout << "stop_cond_.wait timeout\n";
            }
            queue_.stop();
        }

        if (stopped_) {
            thread_->join();
        } else {
            std::cerr << "Search thread can not stop!!!\n";
        }
        thread_.reset();
    }
}

void BatchGenerator::run() {
    {
        pthread_setname_np(pthread_self(), "DynBatch");
        // context must create and destroy in the same thread
        model::ModelContext ctx = model::ModelContext::create(
            *engine_, *model_, config, -1, 0);
        // if (llama_model()) {
        if (false) {
            // SearcherImplV1<int, int>(ctx, config, this).batch_search();
        } else if (!model_) {
            throw std::invalid_argument("No model");
        } else {
            throw std::invalid_argument(std::string("Unknown model:") + model_->layer_type());
        }
        Lock lock(mutex_);
        stopped_ = true;
        stop_cond_.notify_one();
        std::cerr << "Exit BeamSearcher::run()" << std::endl;
    }
}

} // namespace batch_generator
