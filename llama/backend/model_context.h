#pragma once

#include <bmengine/core/core.h>
#include "backend/model.h"
#include "backend/buffer_context.h"
#include "backend/dyn_batch_context.h"
#include "backend/rag_buffer_context.h"
#include "backend/dyn_batch_context.h"

#include <queue>

namespace model {

using namespace bmengine;
using kvcache::KVCacheConfig;

class ModelBase;
struct DynBatchConfig;
struct DynBatchContext;
class ModelContext;

class HostAllReducer {
public:
    HostAllReducer() = default;
    virtual ~HostAllReducer() = default;

    // virtual core::Tensor reduce_sum(int rank, int layer, core::Tensor &data) = 0;
    // virtual core::Tensor reduce_sum_async(int rank, int layer, core::Tensor &data, core::Tensor &out,
    //                                       cudaStream_t is, cudaStream_t os, bool copy_only = false) = 0;

}; // end of class HostAllReducer

struct ReduceContext {
    std::vector<ModelContext *> peers_;
    std::queue<std::pair<ReduceContext *, core::Tensor>> peer_buffers_;
}; // end of class ReduceContext

/*
*  Extend Context to hold more info for LLM model inference
*/
class ModelContext : public bmengine::core::Context {
public:
    const ModelConfig cfg;

private:
    const ModelBase &model_;
    bool parallel_;

    std::vector<int> layer_devices;
    std::shared_ptr<BufferContext> buf_ctx_;

    std::shared_ptr<DynBatchContext> dyn_batch_;
    std::shared_ptr<RagBufferContext> rag_buffer_;

    std::shared_ptr<ReduceContext> reducer_;

    bool latent_cache_{false};

public:
    ModelContext(core::Context &&ctx,
                 const ModelBase &md,
                 int batch_size = 1,
                 bool parallel = false,
                 bool BSHD = false);
    ~ModelContext() override;
    ModelContext(ModelContext &&) = default;

    static ModelContext create(core::Engine &engine,
                               const ModelBase &md,
                               const DynBatchConfig &config,
                               int dev,
                               bool parallel);

    void set_rag_buffer(const std::shared_ptr<RagBufferContext> &buffer) {
        rag_buffer_ = buffer;
    }

    HostAllReducer *create_host_reducer();
    void set_host_reducer(std::shared_ptr<HostAllReducer> reducer);

private:
    KVCacheConfig get_kv_cache_config();

}; // end of class ModelContext

} // namespace model
