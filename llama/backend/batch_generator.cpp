#include "backend/batch_generator.h"
#include "backend/model_context.h"
#include "backend/beam_result_manager.h"
#include "backend/beam_buffer_manager.h"
#include "backend/dyn_batch_context.h"
#include "backend/matrix.h"
#include "backend/prefix_cache.h"
#include "backend/beam_util.h"

#include <bmengine/core/thread_pool.h>
#include <bmengine/logger/kernel_time_trace.hpp>
#include <bmengine/logger/std_log_op.hpp>
#include "private/allocator.h"
#include <unistd.h>
#include <sys/syscall.h>

#define NO_GCC_OPT __attribute__((optimize("O0"))) /* for gdb */

namespace batch_generator {

using bmengine::core::DataType;
using bmengine::core::TaskThreadPool;
using generator::BeamHypothesis;
using generator::SearchResult;
using generator::SearchResults;
using bmengine::core::Tensor;

using model::RagBufferContext;

using utils::Matrix2D;
using utils::Matrix3D;
typedef utils::Matrix2D<int32_t> Mat2DInt;

typedef std::unique_lock<std::mutex> Lock;
typedef unsigned int len_t;

template <class T>
using RagVector = std::vector<std::vector<T>>; // sub-vector has different size.

len_t round_up_len(len_t len, len_t d = 32) {
    return (len + d - 1) / d * d;
}

void SearchTask_::finish(generator::SearchResults &&results) {
    BM_ASSERT(!results.results.empty(), "finish without result!");
    callback(results);
    res_queue.emplace(std::move(results));
}

void SearchTask_::update_stream(const generator::SearchResults &results) {
    res_queue.push(results, true);
}

TaskQueue::TaskQueue(int max_size) :
    max_size_(max_size) {
}

bool TaskQueue::push(SearchTask task, bool wait, bool notify) {
    Lock lock(mutex_);
    if (!wait && queue_.size() >= max_size_) {
        return false;
    }
    while (queue_.size() >= max_size_) {
        can_push_cond_.wait(lock);
    }
    queue_.push(task);
    if (notify) {
        can_pop_cond_.notify_one();
    }
    return true;
}

bool TaskQueue::empty() {
    Lock lock(mutex_);
    return queue_.empty();
}

std::vector<SearchTask> TaskQueue::pop_multi(int limit, bool wait, int require, int max_token, bool pre_alloc) {
    std::vector<SearchTask> tasks;
    int total_token_len = 0;
    {
        Lock lock(mutex_);
        while (wait && !stopping_ && queue_.size() < require) {
            can_pop_cond_.wait(lock);
            if (!queue_.empty() && queue_.front()->canceled) {
                queue_.pop(); // Drop canceled task
                continue;
            }
        }
        while (!queue_.empty() && tasks.size() < limit) {
            total_token_len += pre_alloc ? round_up_len(queue_.front()->full_length() + 2, 32) : queue_.front()->input_length();
            if (total_token_len > max_token) {
                std::cout << "#### max_token: " << max_token
                          << ", input_length=" << queue_.front()->input_length()
                          << ", beam_size=" << queue_.front()->beam_size
                          << ", max_len=" << queue_.front()->max_length
                          << ", full_length=" << queue_.front()->full_length()
                          << std::endl;
                break;
            }
            tasks.push_back(queue_.front());
            queue_.pop();
        }
    }
    if (!tasks.empty()) {
        can_push_cond_.notify_one();
    }
    return tasks;
}

void TaskQueue::stop() {
    Lock lock(mutex_);
    stopping_ = true;
    can_pop_cond_.notify_one();
}

class RepetitionPenalty {
public:
    float repetition_penalty;
    float ngram_penalty;

    RepetitionPenalty(float repetition_penalty, float ngram_penalty) :
        repetition_penalty(repetition_penalty), ngram_penalty(ngram_penalty) {
    }

    virtual ~RepetitionPenalty() = default;

    virtual void apply_penalty(ModelContext &ctx,
                               const std::vector<int> &hypo_last_poses,
                               Tensor *logits_all) = 0;
}; // end of class RepetitionPenalty

class LlamaRepetitionPenalty : public RepetitionPenalty {
    beam_utility::BeamBufferManager<int> &bm;

public:
    LlamaRepetitionPenalty(float repetition_penalty,
                           float ngram_penalty,
                           beam_utility::BeamBufferManager<int> &bm) :
        RepetitionPenalty(repetition_penalty, ngram_penalty),
        bm(bm) {
    }

    void apply_penalty(ModelContext &ctx,
                       const std::vector<int> &hypo_last_poses,
                       Tensor *logits_all) override {
        apply_beam_repetition_penalty(
            ctx, bm,
            hypo_last_poses,
            ngram_penalty,
            repetition_penalty,
            logits_all);
    }
}; // end of class LlamaRepetitionPenalty

BatchGenerator::BatchGenerator(DynBatchConfig config,
                               std::vector<model::ModelBase *> par_models,
                               bmengine::core::Engine *engine) :
    config(config),
    par_models_(par_models),
    engine_(engine),
    queue_(config.task_queue_size) {
    // CHECK_IS_POWER_OF_2(config.max_beam_size);
    BM_ASSERT(config.max_total_token % 64 == 0, "max_total_token should align to 64");
    BM_ASSERT(!par_models_.empty(), "No model");
    model_ = par_models_[0];
}

BatchGenerator::~BatchGenerator() {
    if (!stopping_) {
        stop();
    }
}

bool BatchGenerator::submit(SearchTask task, bool wait, bool notify) {
    if (task->input_tokens.size() <= 1) {
        throw std::invalid_argument("Empty input");
    }
    int max_input_token = config.max_total_token - task->beam_size;
    if (task->input_length() > max_input_token) {
        throw std::invalid_argument(
            "Input tokens are too long: " + std::to_string(task->input_length()));
    }
    if (task->beam_size > config.max_beam_size || task->beam_size < 1) {
        throw std::invalid_argument("Invalid beam size: " + std::to_string(task->beam_size));
    }
    if (task->is_random()) {
        task->beam_size = task->num_results;
    }
    if (task->is_random() && task->seed == 0) {
        task->seed = rand();
    }
    return queue_.push(task, wait, notify);
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

size_t TaskQueue::size() {
    Lock lock(mutex_);
    return queue_.size();
}

class TopKWrapper {
    model::ModelContext &ctx;
    std::unique_ptr<functions::TopK> topk;
    bool diverse;
    curandGenerator_t gen{nullptr};

    void create_gen() {
        CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MRG32K3A));
        CURAND_CHECK(curandSetStream(gen, ctx.current_stream()->ptr));
        CURAND_CHECK(curandSetGeneratorOrdering(gen, CURAND_ORDERING_PSEUDO_BEST));
    }

public:
    explicit TopKWrapper(model::ModelContext &ctx) :
        ctx(ctx), topk(std::make_unique<functions::TopK>(ctx)), diverse(false) {
    }

    TopKWrapper(TopKWrapper &&o) :
        ctx(o.ctx), topk(std::move(o.topk)), diverse(o.diverse), gen(o.gen) {
        o.gen = nullptr;
    }

    ~TopKWrapper() {
        if (gen != nullptr) {
            curandDestroyGenerator(gen);
            gen = nullptr;
        }
    }

    curandGenerator_t &generator() {
        return gen;
    }

    void set_seed(bool diverse, int seed) {
        this->diverse = diverse;
        if (diverse) {
            if (gen == nullptr) {
                create_gen();
            }
            CURAND_CHECK(curandSetGeneratorOffset(gen, 0));
            CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, seed));
        }
    }

    static void check_shape(const std::pair<Tensor, Tensor> &top_scores, int top) {
        BM_ASSERT_EQ(top_scores.first.size(-1), top, "Invalid topk size");
        BM_ASSERT_EQ(top_scores.first.numel(), top_scores.second.numel(), "TopK error");
    }

    void get_top_k(Tensor &logits_all, int top, float *top_probs, int *top_ids) {
        if (diverse) {
            // apply gumbel softmax
            auto gumbel_logits = beam_utility::apply_gumbel_softmax(ctx, gen, logits_all);
            auto top_scores = topk->forward(ctx, gumbel_logits, top);
            check_shape(top_scores, top);

            // 还是用 logits_all 里面的分数
            beam_utility::gather_logits(ctx, top_scores.second, logits_all).to_buffer(top_probs);
            top_scores.second.to_buffer(top_ids);
        } else {
            auto top_scores = topk->forward(ctx, logits_all, top);
            check_shape(top_scores, top);

            functions::typecast(ctx, top_scores.first, DataType::kFloat).to_buffer(top_probs);
            top_scores.second.to_buffer(top_ids);
        }
    }

    void get_top_k(Tensor &logits_all, int top, Matrix2D<float> *top_probs, Matrix2D<int> *top_ids) {
        BM_ASSERT_EQ(logits_all.size(0) * top, top_ids->size(), "size mismatch");
        get_top_k(logits_all, top, top_probs->mutable_data(), top_ids->mutable_data());
    }

    void get_top_k(Tensor &logits_all, int top, std::vector<float> *top_probs, std::vector<int> *top_ids) {
        BM_ASSERT_EQ(logits_all.size(0) * top, top_ids->size(), "size mismatch");
        get_top_k(logits_all, top, top_probs->data(), top_ids->data());
    }

}; // end of class TopKWrapper

len_t calc_max_beam_size(std::vector<SearchTask> &tasks, len_t beam_size) {
    for (SearchTask &task : tasks) {
        if (task) {
            beam_size = std::max(beam_size, len_t(task->beam_size));
        }
    }
    return beam_size;
}

static int _get_tid() {
    return syscall(SYS_gettid);
}

struct SwapBuf {
    char *ptr{nullptr};
    size_t len{0};

    SwapBuf() = default;

    explicit SwapBuf(size_t len) :
        len(len) {
        BM_CUDART_ASSERT(cudaMallocHost(&ptr, len));
    }
    SwapBuf(const SwapBuf &) = delete;
    SwapBuf(SwapBuf &&other) {
        ptr = other.ptr;
        len = other.len;
        other.ptr = nullptr;
        other.len = 0;
    }
    SwapBuf &operator=(const SwapBuf &) = delete;
    SwapBuf &operator=(SwapBuf &&other) {
        ptr = other.ptr;
        len = other.len;
        other.ptr = nullptr;
        other.len = 0;
        return *this;
    }

    ~SwapBuf() {
        if (ptr) {
            cudaFreeHost(ptr);
        }
    }
}; // end of struct SwapBuf

static void print_model_dim(const ModelBase &m) {
    std::cout << "num_layers=" << m.num_layers
              << ", dim_model=" << m.dim_model
              << ", dim_ff=" << m.dim_ff
              << ", num_heads=" << m.num_heads
              << ", num_kv_heads=" << m.num_kv_heads
              << ", dim_head=" << m.dim_head
              << ", num_heads*dim_head=" << m.num_heads * m.dim_head
              << std::endl;
}

static void print_config(const DynBatchConfig &config, bool parallel) {
    std::cout << "DynBatch (";
    std::cout << "max_batch=" << config.max_batch;
    std::cout << ", beam_size=" << config.max_beam_size;
    std::cout << ", queue_size=" << config.task_queue_size;
    std::cout << ", bos=" << config.bos_id;
    std::cout << ", eos=" << config.eos_id;
    if (parallel) {
        std::cout << ", parallel=true, reduce_sum=" << (config.nccl ? "NCCL" : "manual");
    }
    std::cout << (config.flash_attention ? ", flash_attention" : "");
    if (config.rag_buffer) {
        std::cout << ", rag_buffer=true, reserved_work_mem=" << config.reserved_work_mem_mb << "MB";
    }
    std::cout << ")" << std::endl;
}

static size_t get_kv_buf_bytes(const ModelBase &m, int len, int rep = 1) {
    int dim_kv = m.num_kv_heads * m.dim_head;
    int buf_size_pt = 2 * m.num_layers * dim_kv;
    return size_t(len) * sizeof(half) * buf_size_pt;
}

template <class T>
Tensor rag_tensor(ModelContext &ctx, const RagVector<T> &v2d) {
    std::vector<T> flat;
    for (const std::vector<T> &v : v2d) {
        flat.insert(flat.end(), v.begin(), v.end());
    }
    return ctx.tensor_of<T>(flat);
}

template <typename TokenT, typename ResultT>
class SearcherImplV1 {
    typedef beam_utility::BeamBufferInfo<TokenT> BeamBufferInfo;
    typedef beam_utility::BeamBufferManager<TokenT> BeamBufferManager;
    typedef generator::BeamSearchResultManager<ResultT> BeamSearchResultManager;

    len_t max_batch;
    len_t max_beam_size;
    model::ModelContext &ctx;
    std::vector<model::ModelContext *> peer_ctx;
    DynBatchConfig config;
    BatchGenerator *searcher;

    std::vector<SearchTask> tasks;
    TopKWrapper topk_all;
    std::vector<std::unique_ptr<TopKWrapper>> topk;
    std::vector<BeamSearchResultManager> result_mgr;
    std::vector<SearchResults> stream_res;

    std::vector<BeamBufferManager> bm;

    std::vector<int> steps;
    std::vector<std::vector<BeamHypothesis>> hypotheses;
    std::vector<std::vector<BeamBufferInfo>> next_tokens;
    std::vector<SearchTask> new_tasks;
    len_t beam_size;
    len_t max_batch_active{0};
    len_t len_buf{0};

    int debug_batch{-1};
    int debug_batch_idx{-1};
    int batch_idx{0};
    int debug_level{0}; // tianx ...
    len_t num_new_tasks{0};
    int dual_stream{false};
    bool pre_alloc{false};
    int chunking_b{-1}; // index of chunking task, usually the last
    len_t chunked{0};   // processed chunked tokens
    bool enabled_chunk_prefill;
    len_t chunk_size;

    std::vector<std::shared_ptr<PrefixCache>> prefix_cache;

    std::vector<TaskThreadPool *> device_threads;

    std::vector<std::vector<SwapBuf>> swapped_buffers;
    len_t swap_count{0};
    len_t kv_buf_btypes;
    len_t total_len_buf{0};
    len_t max_buf_token_num;

    bool in_chunking() {
        return chunking_b != -1;
    }

public:
    // NO_GCC_OPT
    SearcherImplV1(model::ModelContext &ctx, DynBatchConfig config, BatchGenerator *searcher) :
        max_batch(config.max_batch),
        max_beam_size(config.max_beam_size),
        ctx(ctx),
        config(config),
        searcher(searcher),
        tasks(max_batch),
        topk_all(ctx),
        topk(max_batch),
        result_mgr(max_batch, BeamSearchResultManager(0)),
        stream_res(max_batch),
        bm(max_batch, BeamBufferManager(config.chunk_size)),
        steps(max_batch),
        hypotheses(max_batch),
        next_tokens(max_batch),
        swapped_buffers(max_batch),
        beam_size(max_beam_size) {
        // debug_level = utils::get_int_env("DYN_BATCH_DEBUG", utils::get_int_env("BM_DEBUG_LEVEL"));
        debug_level = utils::get_int_env("DYN_BATCH_DEBUG");
        std::cout << ">>>>>>>>>>>> SearcherImplV1 instroust tasks[0] : " << tasks[0] << "tasks.addr" << &tasks[0]
                  << "max_batch: " << max_batch << std::endl;
        if (debug_level) {
            std::cout << "LWP " << _get_tid() << " DynBatch-Main\n";
        }
        topk_all.set_seed(config.seed != 0, config.seed);
        debug_batch_idx = utils::get_int_env("DEBUG_BATCH", -1);
        bool need_dequant_weight = utils::get_int_env("DEED_DEQUANT_WEIGHT", 0) > 0;
        auto dev = ctx.with_device(0);
        dual_stream = utils::get_int_env("DUAL_STREAM", 0) && ctx.get_compute_capability() > 80;
        bool host_reduce = utils::get_int_env("HOST_REDUCE", 0) > 0;
        enabled_chunk_prefill = utils::get_int_env("CHUNKED_PREFILL", 0) > 0;
        chunk_size = utils::get_int_env("CHUNKED_PREFILL_SIZE", 512);
        pre_alloc = utils::get_int_env("PRE_ALLOC_ALL_TOKEN", 1) > 0
                    && config.flash_attention && dual_stream > 0;
        if (pre_alloc) {
            std::cout << ">>> Use pre_alloc: " << pre_alloc << std::endl;
        }
        std::cout << ">>> CHUNKED_PREFILL_SIZE:" << enabled_chunk_prefill << ", SIZE: " << chunk_size << std::endl;

        if (config.enable_prompt_caching) {
            BM_ASSERT_EQ(ctx.devices().size(), 1, "Only support TP");
        }

        // set peer context
        for (int i = 1; i < searcher->par_models_.size(); ++i) {
            device_threads.push_back(new TaskThreadPool(1, i));
        }

        peer_ctx.resize(device_threads.size() + 1);
        peer_ctx[0] = &ctx;
        peer_run([this, searcher](int i) {
            if (i == 0) return;
            std::string name = "DynBatch-TP" + std::to_string(i);
            if (debug_level > 1)
                std::cout << "LWP " << _get_tid() << " " << name << std::endl;
            pthread_setname_np(pthread_self(), name.c_str());
            peer_ctx[i] = new ModelContext(
                ModelContext::create(*searcher->engine_, *searcher->par_models_[i], this->config, i, true));
        },
                 true, false);

        // set peer info for reduce sum
        if (config.nccl == -1) {
            config.nccl = 1;
            if (debug_level) std::cout << "Auto set config.nccl to " << config.nccl << std::endl;
        }
        std::shared_ptr<model::HostAllReducer> host_reducer;
        if (host_reduce && searcher->par_models_.size() > 1) {
            host_reducer.reset(this->ctx.create_host_reducer());
            auto fn = [&](int i) { peer_ctx[i]->set_host_reducer(host_reducer); };
            peer_run(fn, true);
        }
        if (!config.nccl) {
            // TODO ...
        }

        // calc max_buf_token_num
        auto &model = *searcher->model_;
        auto allocator = ctx.get_allocator();
        size_t free_mem = allocator->get_free_memory();
        kv_buf_btypes = get_kv_buf_bytes(model, 1) / ctx.world_size();
        // if (ctx.rag_buffer()->is_cache_quant()) {
        //     kv_buf_bytes /= 2;
        // }
        // if (ctx.latent_cache() && ctx.cfg.kv_lora_rank > 0) {
        //     kv_buf_bytes = model.num_layers * (ctx.cfg.kv_lora_rank + ctx.cfg.qk_rope_head_dim) * sizeof(half);
        // }
        size_t reserve_mem = size_t(config.reserved_work_mem_mb) * 1024U * 1024U;
        if (need_dequant_weight) {
            std::cout << ">>>>>>>>>>>>>> need_dequant_weight " << std::endl;
        }
        if (dual_stream) {
            std::cout << ">>>>>>>>>>> dual stream" << std::endl;
        }
        max_buf_token_num = (free_mem - reserve_mem) / kv_buf_btypes;
        static bool logged = false;

        if (debug_level > 0 && !logged) {
            logged = true;
            print_model_dim(model);
            print_config(config, !peer_ctx.empty());
            std::cout << "free_mem=" << (free_mem / 102400) << "MB, ";
            std::cout << "reserve_mem=" << (reserve_mem / 102400) << "MB, ";
            std::cout << "kv_per_token=" << (kv_buf_btypes / 1024) << "KB, ";
            std::cout << "max_buf_token_num=" << max_buf_token_num << std::endl;
        }
        if (reserve_mem > free_mem) {
            throw std::runtime_error("Not enough memory for workspace");
        }
        if (config.max_total_token > max_buf_token_num) {
            throw std::runtime_error("Not enough memory for max_total_token > " + std::to_string(max_buf_token_num));
        }

        // prefix_cache.resize(device_threads.size() + 1);
        if (config.enable_prompt_caching) {
            // TODO: better config
            max_buf_token_num /= 2; // use half memory for cache
            int block_size = utils::get_int_env("CACHE_BLOCK_SIZE", 128);
            int block_num = max_buf_token_num / block_size;
            int max_prefix = utils::get_int_env("CACHE_MAX_PREFIX", 4096) / block_size;
            std::cout << "cache block_num: " << block_num << std::endl;
        }
    }

    ~SearcherImplV1() {
        try {
            peer_run([this](int i) {
                if (i > 0)
                    delete peer_ctx[i];
            },
                     true, false);
        } catch (const std::exception &e) {
            std::cerr << "Stop error: " << e.what() << std::endl;
        }

        for (auto p : device_threads) {
            delete p;
        }
        device_threads.clear();
    }

    void peer_wait() {
        for (auto &device_thread : device_threads) {
            device_thread->wait();
        }
    }

    void peer_run(std::function<void(int)> fn, bool wait = false, bool with_dev = true) {
        for (int i = 1; i <= device_threads.size(); ++i) {
            if (with_dev) {
                auto fn1 = [=]() {
                    auto dev = peer_ctx[i]->with_device(0);
                    fn(i);
                };
                device_threads[i - 1]->run(fn1);
            } else {
                device_threads[i - 1]->run(std::bind(fn, i));
            }
        }
        fn(0);
        if (wait) {
            peer_wait();
        }
    }

    void batch_search();

    void set_debug_batch(int i, int b) {
        if (batch_idx++ == debug_batch_idx) {
            std::cout << "b=" << b << ", i=" << i
                      << ", batch_idx=" << batch_idx - 1
                      << ", debug_batch_idx=" << debug_batch_idx << std::endl;
            debug_batch = b;
            ctx.dyn_batch()->debug_batch = i;
        }
    }

    void fill_encode_input(std::vector<SearchTask> &new_tasks);

    void fill_search_tokens(Mat2DInt &h_placement, Matrix2D<float> &h_prob_prev);

    void save_prompt_cache() {
    }

    Tensor join_forward(Tensor *hidden);

    void pick_top_k(Tensor logits_all, Tensor hidden, Mat2DInt &h_placement, Matrix2D<float> &h_prob_prev);

    void random_sample(len_t b, Tensor penalised_logits, std::vector<int> *top_ids, std::vector<float> *probs);

    void update_stream(len_t b, int sent_id, int word_id, int last_hypo_pos, float score);

    void apply_repetition_penalty(Tensor &logits_all, Mat2DInt &h_prob_prev);

    len_t assign_free_slot(SearchTask task) {
        for (len_t b = 0; b < tasks.size(); ++b) {
            if (!tasks[b]) {
                return b;
            }
        }
        return -1;
    }

    void resize_task_buf(len_t b, len_t new_len_buf, bool init = false) {
        if (pre_alloc && !init) {
            len_t full_len = ctx.rag_buffer()->get_buf_len(b);
            BM_ASSERT_LE(new_len_buf, full_len, "");
            return;
        }
        auto fn = [=](int i) {
            if (dual_stream) {
                peer_ctx[i]->use_cache_alloc(true);
            }
            peer_ctx[i]->resize_task_buf(b, new_len_buf);
            if (dual_stream) {
                peer_ctx[i]->use_cache_alloc(false);
            }
        };
        peer_run(fn, false); // resize_task_buf
    }

    void init_slot(len_t b, SearchTask task) {
        max_batch_active = std::max(b + 1, max_batch_active);

        tasks[b] = task;
        std::cout << "task: b=" << b << ", random=" << task->is_random() << ", seed=" << task->seed << std::endl;

        if (!topk[b]) {
            topk[b].reset(new TopKWrapper(ctx));
        }
        topk[b]->set_seed(task->diverse || task->is_random(), task->seed);
        result_mgr[b].reset(std::max(task->beam_size, task->num_results));
        if (!config.rag_buffer) {
            bm[b].reset(len_buf); // global len_buf
        } else {
            // individual len_buf
            len_t new_len_buf = round_up_len(task->input_length() + 2, 32);
            bm[b].reset(new_len_buf);
            if (pre_alloc) {
                len_t full_len_buf = round_up_len(task->full_length() + 2, 32);
                resize_task_buf(b, full_len_buf, true);
            } else {
                resize_task_buf(b, new_len_buf); // alloc new
            }
        }

        steps[b] = 0;
        hypotheses[b].clear();
        hypotheses[b].resize(1);
        next_tokens[b].clear();
        stream_res[b].stream.clear();
    }

    len_t get_batch_active() {
        return max_batch_active;
    }

    Tensor log_softmax(const Tensor &logits_all, Matrix2D<float> &h_prob_prev);

    void erase_task(size_t b) {
        tasks.erase(tasks.begin() + b);
        topk.erase(topk.begin() + b);
        result_mgr.erase(result_mgr.begin() + b);
        stream_res.erase(stream_res.begin() + b);
        bm.erase(bm.begin() + b);
        steps.erase(steps.begin() + b);
        hypotheses.erase(hypotheses.begin() + b);
        next_tokens.erase(next_tokens.begin() + b);
        swapped_buffers.erase(swapped_buffers.begin() + b);
        auto fn = [this, b](int i) {
            peer_ctx[i]->free_task_buf(b);
        };
        peer_run(fn, false);
    }

    void pad_task() {
        // padding to max_batch
        while (tasks.size() < max_batch) {
            tasks.emplace_back();
            topk.emplace_back();
            result_mgr.emplace_back(0);
            stream_res.emplace_back();
            bm.emplace_back(0);
            steps.emplace_back();
            hypotheses.emplace_back();
            next_tokens.emplace_back();
            swapped_buffers.emplace_back();
        }
    }

    void move_task(size_t b, size_t e) {
        tasks[b] = std::move(tasks[e]);
        topk[b] = std::move(topk[e]);
        result_mgr[b] = std::move(result_mgr[e]);
        stream_res[b] = std::move(stream_res[e]);
        bm[b] = std::move(bm[e]);
        steps[b] = std::move(steps[e]);
        hypotheses[b] = std::move(hypotheses[e]);
        next_tokens[b] = std::move(next_tokens[e]);
        swapped_buffers[b] = std::move(swapped_buffers[e]);
    }

    void record_top_logprobs(size_t b, const Tensor &logits) {
        Tensor zero_bias = ctx.tensor_of(std::vector<float>(max_beam_size));
        // Tensor log_probs = logits;
        Tensor log_probs = ctx.tensor(logits.shape(), logits.dtype());
        beam_utility::log_softmax_bias(ctx, logits, zero_bias, tasks[b]->temperature, &log_probs);
        functions::TopK tk(ctx);
        auto log_probs_ch = log_probs.chunk();
        BM_ASSERT_EQ(log_probs.size(0), size_t(max_beam_size), "Wrong size");
        BM_ASSERT(!next_tokens[b].empty(), "");
        BM_ASSERT_LE(next_tokens[b].size(), hypotheses[b].size(), "");
        log_probs_ch.resize(next_tokens[b].size());

        int top_num = tasks[b]->top_logprobs;
        for (len_t i = 0; i < next_tokens[b].size(); i++) {
            auto [d_probs, d_ids] = tk.forward(ctx, log_probs.slice_dim0_len(i, 1), top_num);
            std::vector<int> h_ids(top_num);
            std::vector<float> h_probs(top_num);
            d_ids.to_buffer(h_ids.data());
            functions::typecast(ctx, d_probs, DataType::kFloat).to_buffer(h_probs.data());
            BM_ASSERT_EQ(hypotheses[b][i].top_logprobs.size(), steps[b] * top_num, "");

            for (int k = 0; k < top_num; ++k) {
                hypotheses[b][i].top_logprobs.emplace_back(h_ids[k], h_probs[k]);
            }
        }
    }
}; // end of class SearcherImplV1

template <>
void SearcherImplV1<int, int>::fill_encode_input(std::vector<SearchTask> &new_tasks) {
    bool chunking = in_chunking();
    if (chunking) {
        BM_ASSERT(new_tasks.empty(), "");
        new_tasks.push_back(tasks[chunking_b]);
    }

    std::vector<int> v_batch(new_tasks.size());
    std::vector<int> input_lens(new_tasks.size());
    std::vector<int> full_input_lens(new_tasks.size()); // flash attention need real length
    std::vector<int> buf_lens(new_tasks.size());
    // fill matrices of input tokens
    RagVector<int32_t> h_token;
    RagVector<int32_t> h_batch;     // batch in buffer
    RagVector<int32_t> h_placement; // pos in buffer
    RagVector<int32_t> h_position;
    RagVector<int8_t> h_mask;

    // std::cout << ">>>>>>>>>> new_tasks.size: " << new_tasks.size() << std::endl;
    for (size_t i = 0; i < new_tasks.size(); ++i) {
        auto &task = new_tasks[i];
        auto &tokens = task->input_tokens;
        len_t b;
        if (chunking) {
            b = chunking_b;
        } else {
            b = assign_free_slot(task);
            BM_ASSERT(b != len_t(-1), "No free slot");
            set_debug_batch(i, b);
            init_slot(b, task);
        }
        v_batch[i] = b;

        len_t cached_len = chunked;
        if (config.enable_prompt_caching && tokens.size() > 10 && chunked == 0) {
            auto fn = [=, &cached_len](int i) {
                auto rag_buf = peer_ctx[i]->rag_buffer().get();
                int len = prefix_cache[i]->get(
                    *peer_ctx[i], task->input_tokens, {rag_buf->buf_k_[b].get(), rag_buf->buf_v_[b].get()});
                if (i == 0) cached_len = len;
            };
            peer_run(fn, true); // prefix_cache[i]->get
            std::cout << "cached_len: " << cached_len << std::endl;
        }
        len_t token_num = tokens.size() - 1; // Reserve last token for search
        BM_ASSERT_LT(cached_len, token_num, "");
        len_t encode_len = token_num - cached_len;

        std::cout << "encode_len : " << encode_len << std::endl;

        bool first_chunk = false;
        if (!chunking && enabled_chunk_prefill && encode_len > chunk_size) {
            first_chunk = true;
            chunking = true;
            chunking_b = b; // start chunk_prefill
            chunked = cached_len;
            std::cout << " Start chunked prefill b=" << b << std::endl;
        }

        if (!chunking || first_chunk) {
            std::cout << "Init bm[" << b << "] len=" << token_num << std::endl;
            bm[b].init(task->input_tokens, int(token_num));
        }
        if (chunking) {
            len_t beam_size1 = calc_max_beam_size(tasks, 0);
            BM_ASSERT(beam_size1 > 0, "");
            BM_ASSERT(max_batch_active > 0, "");
            len_t chunk_size_adj = chunk_size - chunking_b * beam_size1;
            if (encode_len <= chunk_size_adj) {
                std::cout << "Done chunking b=" << b << ", len=" << (chunked + encode_len) << std::endl;
                max_batch_active = chunking_b + 1;
                chunking_b = -1; // end chunk_prefill
                chunked = 0;
                chunking = false;
            } else {
                std::cout << "Prefill size=" << chunk_size_adj << " chunk=["
                          << chunked << ", " << (chunked + chunk_size_adj) << "] of " << tokens.size() << std::endl;
                encode_len = chunk_size_adj;
                chunked += chunk_size_adj;
            }
        }

        input_lens[i] = encode_len;
        full_input_lens[i] = token_num;
        if (chunking)
            full_input_lens[i] = chunked;
        len_t len1 = round_up_len(encode_len, 32);
        buf_lens[i] = (len1 <= len_buf / 2 || (len1 + 256) <= len_buf) ? len1 : len_buf;
        buf_lens[i] = config.rag_buffer ? bm[b].len_buf : buf_lens[i];

        h_token.emplace_back(encode_len);
        h_batch.emplace_back(encode_len);
        h_placement.emplace_back(encode_len);
        h_position.emplace_back(encode_len);

        for (len_t pos = 0; pos < encode_len; pos++) {
            h_token[i][pos] = tokens[cached_len + pos];
            h_batch[i][pos] = b;
            h_placement[i][pos] = cached_len + pos;
            h_position[i][pos] = cached_len + pos;
        }
        h_mask.emplace_back(encode_len * buf_lens[i]);
        bm[b].mask_input(&h_mask[i][0], encode_len, buf_lens[i], cached_len);
        // set last token to search
        if (!chunking)
            next_tokens[b].emplace_back(tokens[token_num], bm[b].last_input_buf_pos, 0.0, 1);
    }

    auto set_fn = [&](ModelContext &ctx) {
        // convert matrices to tensors
        Tensor e_token = rag_tensor(ctx, h_token);
        Tensor e_placement = rag_tensor(ctx, h_placement);
        Tensor e_mask = rag_tensor(ctx, h_mask);
        Tensor e_position = rag_tensor(ctx, h_position);
        if (!new_tasks[0]->position_ids.empty()) {
            const len_t num_ids = new_tasks[0]->position_ids.size();
            BM_ASSERT_EQ(new_tasks.size(), 1, "multi-task is not supported");
            BM_ASSERT_EQ(num_ids % new_tasks[0]->input_length(), 0, "position_ids size mismatch");
            len_t m = num_ids / new_tasks[0]->input_length();
            e_position = ctx.tensor_of(new_tasks[0]->position_ids);
            // TODO: support prefix cache
            BM_ASSERT(!config.enable_prompt_caching, "");
            e_position = e_position.slice_dim0(0, m * h_position[0].size());
        }

        if (debug_level < 1 && ctx.rank() != 0) { // todo ...
            std::cout << "e_token: " << e_token << std::endl;
            std::cout << "e_placement: " << e_placement << std::endl;
            std::cout << "e_position: " << e_position << std::endl;
            std::cout << "e_mask: " << e_mask << std::endl;
        }

        ctx.dyn_batch()->set_encode(e_token, Tensor(), e_placement, e_position, e_mask);
        ctx.dyn_batch()->set_encode_batch(v_batch, rag_tensor(ctx, h_batch));
        ctx.dyn_batch()->set_encode_len(input_lens, full_input_lens, ctx.tensor_of(input_lens));
        ctx.dyn_batch()->ev_len_buf = buf_lens;
    };
    peer_run([&](int i) { set_fn(*peer_ctx[i]); }, true); // prepare dyn_batch encode
}

template <>
void SearcherImplV1<int, int>::fill_search_tokens(Matrix2D<int32_t> &h_placement, Matrix2D<float> &h_prob_prev) {
    len_t batch_active = get_batch_active();
    if (batch_active == 0) {
        std::cout << "in_chunking: " << in_chunking() << std::endl;
        // BM_ASSERT(in_chunking(), "");
        auto set_fn = [&](ModelContext &ctx) {
            ctx.dyn_batch()->set_search(Tensor(), Tensor(), Tensor(), Tensor(), Tensor());
            ctx.dyn_batch()->sv_len_buf = {};
            ctx.dyn_batch()->s_len_buf = Tensor();
        };
        peer_run([&](int i) { set_fn(*peer_ctx[i]); }, true);
        return;
    }
    BM_ASSERT(batch_active > 0, "");
    Matrix2D<int32_t> h_token(batch_active, max_beam_size);
    Matrix2D<int32_t> h_position(batch_active, max_beam_size); // pos in sentence
    Matrix3D<int8_t> h_mask(batch_active, max_beam_size, len_buf);
    std::vector<int> rag_buf_lens;
    RagVector<int8_t> rag_mask;
    std::vector<int> h_cu_q_seqlens;
    std::vector<int> h_cu_k_seqlens;
    h_cu_q_seqlens.push_back(0);
    h_cu_k_seqlens.push_back(0);
    int sum_q = 0;
    int sum_k = 0;
    int max_hyp_num = 0;
    int max_q_seqlen = 0;
    int max_k_seqlen = 0;

    // fill matrices of searching tokens
    for (size_t b = 0; b < batch_active; ++b) {
        BM_ASSERT(!config.rag_buffer || tasks[b].get(), "rag_buffer with null task.");
        int hyp_num = next_tokens[b].size(); // ~=beam_size; =0 if b is unused; =1 1th round.
        if (hyp_num > max_hyp_num) {
            max_hyp_num = hyp_num;
        }
        int position = tasks[b] ? int(tasks[b]->input_tokens.size()) + steps[b] : 0;
        bool is_random = tasks[b] && tasks[b]->is_random();
        // int cached_len = tasks[b] ? int(tasks[b]->cached_len_) : 0;
        for (int i = 0; i < hyp_num; i++) {
            BeamBufferInfo &hypo_i = next_tokens[b][i];
            h_token(b, i) = hypo_i.token;
            h_placement(b, i) = bm[b].place_token(hypo_i); // place in buffer
            h_position(b, i) = position - 1;               // position in sentence
            h_prob_prev(b, i) = is_random ? 0 : hypo_i.log_prob;
        }
        if (!config.rag_buffer) {
            bm[b].mask_hypotheses(&h_placement(b, 0), hyp_num, &h_mask(b, 0, 0));
        } else {
            size_t len_q = max_beam_size;
            size_t len_buffer = bm[b].len_buf;
            rag_buf_lens.push_back(len_buffer);
            rag_mask.emplace_back(len_q * len_buffer);
            bm[b].mask_hypotheses(&h_placement(b, 0), hyp_num, &rag_mask[b][0]);
            if (ctx.is_BSHD() && hyp_num > 0) {
                int input_len = int(tasks[b]->input_tokens.size()) + steps[b];
                sum_q += hyp_num;
                sum_k += input_len;
                h_cu_q_seqlens.push_back(sum_q);
                h_cu_k_seqlens.push_back(sum_k);
                if (hyp_num > max_q_seqlen) {
                    max_q_seqlen = hyp_num;
                }
                if (input_len > max_k_seqlen) {
                    max_k_seqlen = input_len;
                }
            }
        }
    }
    auto set_fn = [&](ModelContext &ctx) {
        // convert matrices to tensors
        Tensor d_token = h_token.to_tensor(ctx).view({h_token.size()});
        Tensor d_placement = h_placement.to_tensor(ctx);
        Tensor d_position = h_position.to_tensor(ctx).view({h_token.size()});
        Tensor d_mask = config.rag_buffer ? rag_tensor(ctx, rag_mask) : h_mask.to_tensor(ctx);

        ctx.dyn_batch()->set_search(d_token, Tensor(), d_placement, d_position, d_mask);
        ctx.dyn_batch()->sv_len_buf = rag_buf_lens;
        ctx.dyn_batch()->s_len_buf = ctx.tensor_of(rag_buf_lens);
        // TODO: max_hyp_num > 1
        if (max_hyp_num == 1 && h_cu_q_seqlens.size() > 0) {
            ctx.dyn_batch()->cu_q_seqlens = ctx.tensor_of(h_cu_q_seqlens);
            ctx.dyn_batch()->cu_k_seqlens = ctx.tensor_of(h_cu_k_seqlens);
            ctx.dyn_batch()->max_q_seqlen = max_q_seqlen;
            ctx.dyn_batch()->max_k_seqlen = max_k_seqlen;
            ctx.dyn_batch()->total_k = sum_k;
        }
    };
    peer_run([&](int i) { set_fn(*peer_ctx[i]); }, true); // prepare dyn_batch search
}

using functions::concat_tensor;

template <>
Tensor SearcherImplV1<int, int>::join_forward(Tensor *hidden) {
    long ts0 = logger::get_time_us();

    Tensor ret_logits;
    auto peer_fn = [&](int i) {
        auto &ctx1 = *peer_ctx[i];
        model::DynBatchContext *dyn_ctx = ctx1.dyn_batch().get();
        // join encode and search input together to call model->encode()
        Tensor group_token = concat_tensor(ctx1, dyn_ctx->e_token, dyn_ctx->s_token, 0);
        Tensor group_position = concat_tensor(ctx1, dyn_ctx->e_position, dyn_ctx->s_position, 0);

        ctx1.rag_buffer()->skip_last = in_chunking();
        ctx1.rag_buffer()->set_buffer_addr(ctx1);

        Tensor input_embeddings;
        Tensor &task0_emb = tasks[0]->input_embeddings;
        if (!task0_emb.empty() && !dyn_ctx->e_token.empty()) {
            BM_ASSERT_EQ(1, dyn_ctx->s_token.numel(), "Feed embedding decode tasks[0] only");
            BM_ASSERT_EQ(group_token.size(0), task0_emb.size(0), "Feed embedding decode tasks[0] only");
            input_embeddings = ctx1.tensor(task0_emb.shape(), task0_emb.dtype());
            ctx1.assign_or_copy(&input_embeddings, &task0_emb);
        }

        auto md = dynamic_cast<model::LLaMALike *>(searcher->par_models_[i]);
        Tensor hidden_g = md->encode(ctx1,
                                     group_token,
                                     group_position,
                                     Tensor(),
                                     Tensor(),
                                     ctx1.dyn_batch()->s_mask,
                                     ctx1.dyn_batch()->s_placement,
                                     input_embeddings,
                                     false);
        BM_ASSERT_EQ(hidden_g.size(0), group_token.size(0), "encode result dim mismatch");

        if (dyn_ctx->e_token.numel() == group_token.size(0)) {
            // Only chunking. no search tokens. Keep logits_all as empty Tensor()
            BM_ASSERT(in_chunking(), "");
        } else {
            // cut out encoding
            Tensor hidden_search = hidden_g.slice_dim0(dyn_ctx->e_token.numel(), group_token.size(0));
            std::cout << "hidden_search: " << hidden_search.numel() << std::endl;
            Tensor logits = md->get_logits(ctx1, hidden_search, true);
            // assign result in rank 0
            if (i == 0) {
                ret_logits = logits;
            }
        }

        ctx1.clear_identity_cache();
        BM_CUDART_ASSERT(cudaStreamSynchronize(ctx1.current_stream()->ptr));
    };
    peer_run(peer_fn, true); // join_forward

    return ret_logits;
}

template <typename TokenT, typename ResultT>
void SearcherImplV1<TokenT, ResultT>::batch_search() {
    int active_count = 0;
    std::cout << "[cpp] SearcherImplV1<TokenT, ResultT>::batch_search()" << std::endl;

    while (true) {
        if (in_chunking()) {
            max_batch_active = chunking_b + 1;
        }
        if (config.rag_buffer) {
        }
        searcher->active_size_ = active_count;
        if (active_count == 0 && searcher->queue_.size() == 0) {
            searcher->done_cond_.notify_one();
        }
        int max_total_token;
        if (swap_count == 0 && active_count < max_batch && !in_chunking()) {
            int free_token_num = int(max_buf_token_num - total_len_buf - 2);
            if (dual_stream)
                free_token_num -= config.max_total_token - 5000;
            max_total_token = std::min(config.max_total_token, free_token_num);
            if (pre_alloc && dual_stream) {
                auto dev = ctx.with_device(0);
                ctx.use_cache_alloc(true);
                max_total_token = ctx.get_allocator()->get_free_memory() / kv_buf_btypes;
                ctx.use_cache_alloc(false);
            }
            int limit = 1; // dual_stream ? 1 : max_batch - active_count
            new_tasks = searcher->queue_.pop_multi(
                limit, active_count == 0, 1, max_total_token, pre_alloc);
            std::cout << " >>>>> new_tasks size: " << new_tasks.size() << std::endl;
            for (auto task : new_tasks) {
                task->begin_ts = logger::get_time_us();
            }
        }

        if (searcher->stopping_) {
            break;
        } else if (max_batch_active == 0) {
            BM_ASSERT(!new_tasks.empty(), "pop_multi() return 0 tasks.");
        }

        auto dev = ctx.with_device(0);

        // resize fields
        if (!config.rag_buffer) {
            // todo tianx ...
        }
        num_new_tasks = new_tasks.size();
        if (debug_level && num_new_tasks > 0) {
            std::cout << "new_tasks=" << new_tasks.size()
                      << ", active_count=" << active_count << std::endl;
        }
        bool feed_input_embedding = !new_tasks.empty() && !new_tasks[0]->input_embeddings.empty();
        std::cout << ">>>>>>>>>>>>>> tasks[0]: " << tasks[0] << std::endl
                  << std::endl;
        if (feed_input_embedding && tasks[0]) {
            // int b = assign_free_slot(tasks[0]);
            // if (debug_level) std::cout << "Move task 0 to " << b << std::endl;
            // move_task(b, 0);
            // BM_ASSERT(!tasks[0], "Feed input_embeddings need put task at index=0");
        }

        /** -------------------------- Fill Encode Input -------------------------- **/
        if (!new_tasks.empty() || in_chunking()) {
            fill_encode_input(new_tasks); // update max_batch_active in init_slot()
            new_tasks.clear();
        } else {
            peer_run([&](int i) { peer_ctx[i]->dyn_batch()->clear_encode(); });
        }
        max_beam_size = calc_max_beam_size(tasks, 1);

        /** -------------------------- Fill Next Search --------------------------- **/
        if (feed_input_embedding) max_batch_active = 1;
        if (in_chunking()) max_batch_active = chunking_b;
        Matrix2D<int32_t> h_placement(max_batch_active, max_beam_size, -1); // pos in buffer
        Matrix2D<float> h_prob_prev(max_batch_active, max_beam_size, -50000);
        fill_search_tokens(h_placement, h_prob_prev);

        /** -------------------------- Get Search Logits --------------------------- **/
        Tensor hidden;
        Tensor logits_all = join_forward(&hidden);
        if (config.enable_prompt_caching) {
            save_prompt_cache();
        }
        if (logits_all.numel() == 0) {
            BM_ASSERT(in_chunking(), "");
            max_batch_active = 1;
            continue; // no other tasks, goto next chunk directly
        }

        size_t logits_dim = searcher->model_->vocab_size;
        logits_all = logits_all.view({max_batch_active, max_beam_size, logits_dim});
        if (hidden.numel()) {
            hidden = hidden.view({max_batch_active, max_beam_size, hidden.size(-1)});
        }

        /** ------------------------------ Pick top k ------------------------------- **/
        pick_top_k(logits_all, hidden, h_placement, h_prob_prev);

        for (len_t b = 0; b < max_batch_active; ++b) {
            if (!next_tokens[b].empty()) {
                float best_score = next_tokens[b][0].log_prob / float(steps[b] + 1);
                if (ctx.debug() > 1 && b == debug_batch) {
                    std::cout << "b=" << b << ", t=" << steps[b] << ", score=" << best_score << std::endl;
                }
                if (!result_mgr[b].accept_score(best_score)) {
                    // stop search, because current result's score is too small
                    next_tokens[b].clear();
                }
            }
            steps[b]++;
            if (steps[b] == 1 && tasks[b]) {
                tasks[b]->first_token_delay_ms = (logger::get_time_us() - tasks[b]->begin_ts) / 1000;
            }
        }

        /** ------------------------------ Fill result ------------------------------- **/
        active_count = 0;
        len_t max_batch_active1 = max_batch_active;
        max_batch_active = 0;
        for (len_t b = 0; b < max_batch_active1; ++b) {
            if (tasks[b] && next_tokens[b].empty()) {
                if (result_mgr[b].get_current_results() == 0) {
                    std::cerr << "No Result and no next_tokens!" << std::endl;
                }
                int num = std::min(tasks[b]->num_results, result_mgr[b].get_current_results());
                SearchResults results{};
                results.results = result_mgr[b].get_search_results(num);
                results.first_token_delay_ms = tasks[b]->first_token_delay_ms;
                tasks[b]->finish(std::move(results));
                tasks[b].reset();
                if (in_chunking()) {
                    BM_ASSERT(chunking_b > 0, "");
                    BM_ASSERT_LE(b, chunking_b - 1, "");
                    chunking_b--;
                }
                std::cout << "Finish task " << b << std::endl;
            }
            if (tasks[b] && tasks[b]->canceled) {
                tasks[b].reset();
                std::cout << "Cancel search task " << b << std::endl;
            }
            if (tasks[b]) {
                active_count++;
                max_batch_active = b + 1;
            } else if (config.rag_buffer) {
                erase_task(b);
                b--;
                max_batch_active1--;
            }
        }
        std::cout << ">>>>>>>>>>>>>>> current line !!!" << std::endl;
        if (config.rag_buffer) {
            pad_task();
        } else if (searcher->queue_.empty() && active_count < max_batch_active) {
            for (len_t b = 0, e = max_batch_active - 1; b < e && active_count < max_batch_active;) {
                while (tasks[b]) b++;
                move_task(b, e);
                do {
                    e--;
                    max_batch_active--;
                } while (!tasks[e]);
            }
        }

        std::cout << ">>>>>>>>>>>>>>> well done!!!" << std::endl;
    }
}

template <>
void SearcherImplV1<int, int>::apply_repetition_penalty(Tensor &logits_all, // (batch, beam_size, vocab_size)
                                                        Mat2DInt &h_placement) {
    // implement: logits[:, self.tokenizer.bos_token_id] = -float("inf")
    std::vector<int> batch_hypos; // (batch, beam_size)
    for (int b = 0; b < max_batch_active; ++b) {
        if (tasks[b] && steps[b] == 0) {
            float penalty_factor = tasks[b]->repetition_penalty;
            for (len_t h = 0; h < max_beam_size; ++h) {
                batch_hypos.push_back(b * max_beam_size + h);
            }
        }
        if (!batch_hypos.empty()) {
            std::vector<int> bos_ids(batch_hypos.size(), config.bos_id);
            std::vector<int> eos_ids(batch_hypos.size(), config.eos_id);
            std::vector<float> neg_inf(batch_hypos.size(), -50000);
            beam_utility::scatter_update(ctx, neg_inf, bos_ids, batch_hypos, logits_all);
            beam_utility::scatter_update(ctx, neg_inf, eos_ids, batch_hypos, logits_all);
        }
    }

    bool need_ngram_penalty = false;
    for (int b = 0; b < max_batch_active; ++b) {
        if (tasks[b] && tasks[b]->ngram_penalty > 1.0) {
            need_ngram_penalty = true;
            break;
        }
    }
    if (!need_ngram_penalty) {
        std::vector<int> batch_hypos; // (batch, beam_size)
        std::vector<int> tokens;
        std::vector<float> repeat_penalty;
        std::vector<float> presence_penalties;

        for (int b = 0; b < max_batch_active; ++b) {
            if (tasks[b] && (tasks[b]->repetition_penalty != 1.0 || tasks[b]->presence_penalty != 0.)) {
                float penalty_factor = tasks[b]->repetition_penalty;
                float presence_penalty = tasks[b]->repetition_penalty;
                for (size_t h = 0; h < hypotheses[b].size(); ++h) {
                    for (int token_id : hypotheses[b][h].token_id_set) {
                        batch_hypos.push_back(b * max_beam_size + h);
                        tokens.push_back(token_id);
                        repeat_penalty.push_back(penalty_factor);
                        presence_penalties.push_back(presence_penalty);
                    }
                }
            }
        }
        if (!batch_hypos.empty()) {
            // sparsely update
            // beam_utility::beam_repetition_penalty(ctx, repeat_penalty, tokens, batch_hypos, logits_all, presence_penalties);
        }
        return;
    }

    std::vector<Tensor> logits_chunks = logits_all.chunk();
    for (int b = 0; b < max_batch_active; ++b) {
        if (!tasks[b])
            continue;
        size_t hyp_num = next_tokens[b].size();
        BM_ASSERT(hyp_num > 0, "hyp_num > 0");
        std::vector<int> b_placement(&h_placement(b, 0), &h_placement(b, 0) + hyp_num);
        Tensor chunk = logits_chunks[b].slice_dim0(0, hyp_num);
        LlamaRepetitionPenalty(tasks[b]->repetition_penalty, tasks[b]->ngram_penalty, bm[b])
            .apply_penalty(ctx, b_placement, &chunk);
    }
}

template <typename TokenT, typename ResultT>
Tensor SearcherImplV1<TokenT, ResultT>::log_softmax(const Tensor &logits_all, Matrix2D<float> &h_prob_prev) {
    Tensor d_prob_prev = h_prob_prev.to_tensor(ctx);
    Tensor score_all = beam_utility::log_softmax_bias(ctx, logits_all, d_prob_prev);
    return score_all;
}

template <>
void SearcherImplV1<int, int>::update_stream(
    len_t b, int sent_id, int word_id, int last_hypo_pos, float score) {
    len_t t = steps[b];
    stream_res[b].stream.score = score;
    if (sent_id == 0 && t == stream_res[b].stream.step + 1) {
        // increasingly update
        if (word_id != config.eos_id) {
            stream_res[b].stream.append(word_id);
            tasks[b]->update_stream(stream_res[b]);
        }
    } else {
        // full update
        bool is_eos = word_id == config.eos_id && !config.keep_eos;
        auto tmp_res = bm[b].get_hypo_tokens(word_id, is_eos, last_hypo_pos);
        BM_ASSERT(tmp_res.size() > 0, "No results");
        stream_res[b].stream.update(std::move(tmp_res), t);
        tasks[b]->update_stream(stream_res[b]);
    }
}

template <>
void SearcherImplV1<int, int>::pick_top_k(
    Tensor logits_all, Tensor hidden, Mat2DInt &h_placement, Matrix2D<float> &h_prob_prev) {
    // std::cout << ">>>>>>>>>> pick top k >>>>>>>>>>>>>>>>>>" << std::endl;
    size_t logits_dim = searcher->model_->vocab_size;
    this->apply_repetition_penalty(logits_all, h_placement);
    auto penalised_logits = logits_all.chunk();

    // shape (max_batch_active, beam_size, logits_dim)
    Tensor score_all = log_softmax(logits_all, h_prob_prev);

    // calculate top k
    len_t num_top = std::min(max_beam_size * 2, len_t(32)); // * 2 because may have eos_id
    Matrix2D<float> top_probs_all(max_batch_active, num_top);
    Matrix2D<int> top_ids_all(max_batch_active, num_top);
    // 拍平，后面的 top k 对应的是所有 this_step_size 个 hypothesis 的所有 vocab 的打分
    score_all = score_all.view({max_batch_active, max_beam_size * logits_dim});
    topk_all.get_top_k(score_all, num_top, &top_probs_all, &top_ids_all);
    // TODO diverse for every task

    for (size_t b = 0; b < max_batch_active; ++b) {
        BM_ASSERT_EQ(hypotheses[b].size(), next_tokens[b].size(), "size mismatch");
        size_t hyp_num = next_tokens[b].size();
        if (tasks[b] && tasks[b]->top_logprobs > 0) {
            record_top_logprobs(b, penalised_logits[b]);
        }
        std::vector<BeamHypothesis> old_hypotheses = std::move(hypotheses[b]);
        hypotheses[b].clear();
        next_tokens[b].clear();
        if (hyp_num == 0)
            continue;
        len_t t = steps[b];
        std::vector<int> top_ids = top_ids_all.vec(b);
        std::vector<float> top_probs = top_probs_all.vec(b);
        if (tasks[b]->is_random()) {
            random_sample(b, penalised_logits[b], &top_ids, &top_probs);
        }
        // put top k into next_tokens
        for (len_t i = 0; i < top_ids.size(); i++) {
            int word_id = top_ids[i] % logits_dim; // token_id
            int sent_id = top_ids[i] / logits_dim; // hypo id
            if (sent_id >= int(hyp_num)) {
                std::cerr << "[b=" << steps[b] << ", i=" << i << "] send_id out of range!\n";
                continue;
            }
            int last_hypo_pos = h_placement(b, sent_id);
            float score = top_probs[i] / float(t + 1);

            if (i == 0 && tasks[b]->stream) { // && score > stream_res[b].stream_res[b].stream.score
                update_stream(b, sent_id, word_id, last_hypo_pos, score);
            }

            bool is_eos = word_id == config.eos_id;
            if (is_eos || t + 1 == tasks[b]->max_length) {
                bool ignore_eos = config.ignore_eos && (t + 1) < tasks[b]->max_length;
                if (!ignore_eos && result_mgr[b].accept_score(score)) {
                    // add_result if end of beam
                    auto res = bm[b].get_hypo_tokens(word_id, is_eos && !config.keep_eos, last_hypo_pos);
                    int res_idx = result_mgr[b].add_result(res, {}, top_probs[i], score);
                    if (res_idx >= 0 && tasks[b]->top_logprobs > 0) {
                        BM_ASSERT_LE(size_t(sent_id) + 1, old_hypotheses.size(), "");
                        result_mgr[b].set_top_logprobs(
                            res_idx, old_hypotheses[sent_id].get_top_logprobs(tasks[b]->top_logprobs));
                        std::cout << "Add result:" << result_mgr[b].get_current_results() << " at step " << steps[b]
                                  << ", b=" << b << ", beam_size=" << tasks[b]->beam_size << std::endl;
                        if (tasks[b]->is_random()) {
                            if (0 == --tasks[b]->beam_size) {
                                std::cout << "Stop random search" << std::endl;
                                next_tokens[b].clear();
                                break;
                            }
                        }
                    }
                } else {
                    bm[b].increase_buf_ref(last_hypo_pos);
                    if (hyp_num == 1 && tasks[b]->beam_size == 1) {
                        hypotheses[b].emplace_back(std::move(old_hypotheses[sent_id]));
                    } else {
                        hypotheses[b].push_back(old_hypotheses[sent_id]);
                    }
                    hypotheses[b].back().add_token(word_id, top_probs[i]);
                    next_tokens[b].emplace_back(word_id, last_hypo_pos, top_probs[i], 0);
                }

                if (next_tokens[b].size() >= tasks[b]->beam_size) {
                    break;
                }
            }
            bm[b].release_buffer(&h_placement(b, 0), hyp_num);
        }
    }
}

template <typename TokenT, typename ResultT>
void SearcherImplV1<TokenT, ResultT>::random_sample(
    len_t b, Tensor penalised_logits, std::vector<int> *top_ids, std::vector<float> *fake_probs) {
    SearchTask task = tasks[b];
    bool first_step = steps[b] == 0;
    penalised_logits = penalised_logits.slice_dim0(0, first_step ? 1 : task->beam_size);
    Tensor probs = ctx.tensor(penalised_logits.size(), penalised_logits.dtype());
    functions::softmax(ctx, penalised_logits, probs, tasks[b]->temperature);

    core::Tensor selection;

    // sample many results in the first step
    // sample one for each instance after the first step
    int num_samples = first_step ? task->num_results : 1;
    int num_selections = task->beam_size;
    std::cout << "b: " << b << ", step=" << steps[b] << ", num_samples=" << num_samples << ", num_selections=" << num_selections << std::endl;
    selection = ctx.tensor({(size_t)num_selections}, core::DataType::kInt32);
    beam_utility::random_sampler_gpu(ctx, topk[b]->generator(), probs, selection, task->top_p, task->top_k, num_samples);

    top_ids->resize(num_selections);
    selection.to_buffer(top_ids->data());

    fake_probs->clear();
    int logits_dim = probs.size(-1);
    for (int i = 0; i < num_selections; i++) {
        if (!first_step)
            (*top_ids)[i] += logits_dim * i; // set sent_id to id
        fake_probs->push_back(steps[b] * (steps[b] + 1));
    }
}

void BatchGenerator::run() {
    {
        pthread_setname_np(pthread_self(), "DynBatch");
        // context must create and destroy in the same thread
        model::ModelContext ctx = model::ModelContext::create(
            *engine_, *model_, config, par_models_.empty() ? -1 : 0, !par_models_.empty());
        std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>> BatchGenerator run() \n";
        if (llama_model()) {
            SearcherImplV1<int, int>(ctx, config, this).batch_search();
        } else if (!model_) {
            throw std::invalid_argument("No model");
        } else {
            throw std::invalid_argument(std::string("Unknown model:") + model_->layer_type());
        }
        std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>> BatchGenerator end() \n";
        Lock lock(mutex_);
        stopped_ = true;
        stop_cond_.notify_one();
        std::cerr << "Exit BeamSearcher::run()" << std::endl;
    }
}

} // namespace batch_generator
