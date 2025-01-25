#pragma once
#include <bmengine/core/core.h>
#include "backend/model.h"
#include "backend/kvcache.h"

namespace model {

using namespace kvcache;
using bmengine::core::Tensor;

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

    virtual Tensor *buf_k(size_t layer) = 0;
    virtual Tensor *buf_v(size_t layer) = 0;

    virtual const Tensor *block_table(size_t layer) = 0;

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

    Tensor *buf_k(size_t layer) override;
    Tensor *buf_v(size_t layer) override;

    const Tensor *block_table(size_t layer) override;

    void set_layer_devices(const std::vector<int> &layer_devices) override;

}; // end of class TransformerBufferContext

} // namespace model
