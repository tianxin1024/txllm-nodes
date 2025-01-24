#pragma once

#include "backend/model_context.h"
#include "backend/transformer_buffer.h"

namespace model {

using kvcache::KVCacheConfig;
struct RagBufferContext {
    typedef std::vector<void *> AddressVector;

    kvcache::KVCacheConfig config_k_;
    kvcache::KVCacheConfig config_v_;
    size_t num_layers;

    bool skip_last{false};

private:
    typedef const Tensor &(TransformerBuffer::*BufPtr)(int i) const;
    AddressVector get_buf_addresses(bool is_k, BufPtr buf_ptr) {
    }

public:
    RagBufferContext(KVCacheConfig config_k, KVCacheConfig config_v) :
        config_k_(config_k), config_v_(config_v) {
        BM_ASSERT_EQ(config_k.num_layers, config_v.num_layers, "num_layers mismatch");
        this->num_layers = config_k.num_layers;
    }

    // void set_buffer_addr(ModelContext &ctx) {
    //     if (active_batch() > 0) {
    //         h_buf_k_addresses = get_buf_addresses(BUF_K, &TransformerBuffer::get_layer);
    //         buf_k_addresses = ctx.tensor_of(h_buf_k_addresses, {num_layers, active_batch()});
    //     }
    // }
}; // end of struct RagBufferContext

} // namespace model
