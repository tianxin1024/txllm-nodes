#include "backend/batch_generator.h"
#include "backend/model_context.h"
#include "backend/beam_result_manager.h"
#include "backend/beam_buffer_manager.h"
#include "backend/dyn_batch_context.h"
#include "backend/matrix.h"

#include <bmengine/core/thread_pool.h>
#include <bmengine/logger/kernel_time_trace.hpp>
#include <bmengine/logger/std_log_op.hpp>
#include "private/allocator.h"
#include <unistd.h>
#include <sys/syscall.h>

#define NO_GCC_OPT __attribute__((optimize("O0"))) /* for gdb */

namespace batch_generator {

using bmengine::core::TaskThreadPool;
using generator::BeamHypothesis;
using generator::SearchResult;
using generator::SearchResults;
using bmengine::core::Tensor;

using model::RagBufferContext;

using utils::Matrix2D;
typedef utils::Matrix2D<int32_t> Mat2DInt;

typedef std::unique_lock<std::mutex> Lock;
typedef unsigned int len_t;

template <class T>
using RagVector = std::vector<std::vector<T>>; // sub-vector has different size.

TaskQueue::TaskQueue(int max_size) :
    max_size_(max_size) {
}

void TaskQueue::stop() {
    Lock lock(mutex_);
    stopping_ = true;
    can_pop_cond_.notify_one();
}

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
    int debug_level{0};
    len_t num_new_tasks{0};
    int dual_stream{false};
    bool pre_alloc{false};
    int chunking_b{-1}; // index of chunking task, usually the last
    bool enabled_chunk_prefill;
    len_t chunk_size;

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
        // TODO memory / 10
        size_t reserve_mem = size_t(config.reserved_work_mem_mb) * 1024U * 1024U / 10;
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
            // throw std::runtime_error("Not enough memory for max_total_token > " + std::to_string(max_buf_token_num));
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

    Tensor join_forward(Tensor *hidden);

    len_t assign_free_slot(SearchTask task) {
        for (len_t b = 0; b < tasks.size(); ++b) {
            if (!tasks[b]) {
                return b;
            }
        }
        return -1;
    }

    void init_slot(len_t b, SearchTask task) {
        max_batch_active = std::max(b + 1, max_batch_active);

        std::cout << "00000000000000000000000000000000000000000000" << std::endl;

        // tasks[b] = task;
        // std::cout << "task: b=" << b << ", random=" << task->is_random() << ", seed=" << task->seed << std::endl;

        // if (!topk[b]) {
        //     topk[b].reset(new TopKWrapper(ctx));
        // }
        // topk[b]->set_seed(task->diverse || task->is_random(), task->seed);
        // result_mgr[b].reset(std::max(task->beam_size, task->num_results));
        // if (!config.rag_buffer) {
        //     bm[b].reset(len_buf); // global len_buf
        // } else {
        //     // individual len_buf
        //     len_t new_len_buf = round_up_len(task->input_length() + 2, 32);
        //     bm[b].reset(new_len_buf);
        //     if (pre_alloc) {
        //         len_t full_len_buf = round_up_len(task->full_length() + 2, 32);
        //         resize_task_buf(b, full_len_buf, true);
        //     } else {
        //         resize_task_buf(b, new_len_buf); // alloc new
        //     }
        // }
    }

    len_t get_batch_active() {
        return max_batch_active;
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
    }

    std::cout << "9999999999999999999999999999999999999999999999" << std::endl;
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
        // TODO task0_emb.empty() 内存访问不到
        // if (!task0_emb.empty() && !dyn_ctx->e_token.empty()) {
        std::cout << "dyn_ctx->s_token.numel(): " << dyn_ctx->s_token.numel() << std::endl;
        if (!dyn_ctx->e_token.empty()) {
            std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> tianx >>>>>>>>>>>>>>>>" << std::endl;
            BM_ASSERT_EQ(1, dyn_ctx->s_token.numel(), "Feed embedding decode tasks[0] only");
            BM_ASSERT_EQ(group_token.size(0), task0_emb.size(0), "Feed embedding decode tasks[0] only");
            input_embeddings = ctx1.tensor(task0_emb.shape(), task0_emb.dtype());
            ctx1.assign_or_copy(&input_embeddings, &task0_emb);
        }

        std::cout << "input_embeddings: " << input_embeddings << std::endl;

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
        std::cout << ">>>>>>>>>>>>>>>>>>>> current line >>>>>>>>>>>>>>>>>>" << std::endl;
        BM_ASSERT_EQ(hidden_g.size(0), group_token.size(0), "encode result dim mismatch");

        if (dyn_ctx->e_token.numel() == group_token.size(0)) {
            // Only chunking. no search tokens. Keep logits_all as empty Tensor()
            BM_ASSERT(in_chunking(), "");
        } else {
            // cut out encoding
            Tensor hidden_search = hidden_g.slice_dim0(dyn_ctx->e_token.numel(), group_token.size(0));
            // Tensor logits = md->get_logits(ctx1, hidden_search, true);

            // assign result in rank 0
            // if (i == 0) ret_logits = logits;
        }

        ctx1.clear_identity_cache();
        BM_CUDART_ASSERT(cudaStreamSynchronize(ctx1.current_stream()->ptr));
    };

    std::cout << "join_forward: " << logger::get_time_us() - ts0 << std::endl;

    peer_run(peer_fn, true); // join_forward

    return ret_logits;
}

template <typename TokenT, typename ResultT>
void SearcherImplV1<TokenT, ResultT>::batch_search() {
    int active_count = 0;

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
            // new_tasks = searcher->queue_.pop_multi(
            //     limit, active_count == 0, 1, max_total_token, pre_alloc);
            // for (auto task : new_tasks) {
            //     task->begin_ts = logger::get_time_us();
            // }
        }

        if (searcher->stopping_) {
            break;
        } else if (max_batch_active == 0) {
            // BM_ASSERT(!new_tasks.empty(), "pop_multi() return 0 tasks.");
        }
        auto dev = ctx.with_device(0);

        // resize fields
        if (!config.rag_buffer) {
            std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>> current \n";
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
            std::cout << "<<<<<<<<<<<<<<<<<<<<< current new_tasks" << std::endl;
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

        exit(0);
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
