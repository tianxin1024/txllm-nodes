#pragma once
#include <bmengine/core/core.h>
#include "backend/model.h"
#include "backend/kvcache.h"

namespace model {

using namespace kvcache;

/*
* Generation buffers managment
*/
class BufferContext {
protected:
    const ModelBase &model_;
    bool parallel_;

public:
    BufferContext(const ModelBase &md, bool parallel = false) :
        model_(md), parallel_(parallel) {
    }

    ~BufferContext() = default;
    BufferContext(BufferContext &&) = default;

    virtual void set_layer_devices(const std::vector<int> &layer_devices) = 0;

}; // end of class BufferContext

class TransformerBufferContext : public BufferContext {
private:
    bool BSHD;
    bool cache_paged;

    std::shared_ptr<KVCache> buf_k_;
    std::shared_ptr<KVCache> buf_v_;

    size_t kvcache_len_;

public:
    TransformerBufferContext(const ModelBase &md,
                             int batch_size = 1,
                             bool parallel = false,
                             int world_size = 1,
                             bool BSHD = false);
    ~TransformerBufferContext();

    void set_layer_devices(const std::vector<int> &layer_devices) override;

}; // end of class TransformerBufferContext

} // namespace model
