#include "backend/batch_generator.h"
#include "backend/model_context.h"
#include "backend/model_context.h"
#include "backend/beam_result_manager.h"
#include "backend/beam_buffer_manager.h"
#include <unistd.h>
#include <sys/syscall.h>

#define NO_GCC_OPT __attribute__((optimize("O0"))) /* for gdb */

namespace batch_generator {

using generator::BeamHypothesis;
using generator::SearchResult;
using generator::SearchResults;

typedef std::unique_lock<std::mutex> Lock;
typedef unsigned int len_t;

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
    int debug_level{0};
    int dual_stream{false};
    bool pre_alloc{false};
    bool enabled_chunk_prefill;
    len_t chunk_size;

    std::vector<std::vector<SwapBuf>> swapped_buffers;

public:
    NO_GCC_OPT
    SearcherImplV1(model::ModelContext &ctx, DynBatchConfig config, BatchGenerator *searcher) :
        max_batch(config.max_batch),
        max_beam_size(config.max_beam_size),
        ctx(ctx),
        config(config),
        searcher(searcher),
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
        debug_level = utils::get_int_env("DYN_BATCH_DEBUG", utils::get_int_env("BM_DEBUG_LEVEL"));
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
        device_threads.push_back(new TaskThreadPool(1, 0));

        peer_ctx.resize(device_threads.size() + 1);
        peer_ctx[0] = &ctx;
    }

}; // end of class SearcherImplV1

void BatchGenerator::run() {
    {
        pthread_setname_np(pthread_self(), "DynBatch");
        // context must create and destroy in the same thread
        model::ModelContext ctx = model::ModelContext::create(
            *engine_, *model_, config, -1, 0);
        if (llama_model()) {
            SearcherImplV1<int, int>(ctx, config, this).batch_search();
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
