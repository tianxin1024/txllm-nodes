#pragma once

#include "backend/model_context.h"

namespace model {

using kvcache::KVCacheConfig;
struct RagBufferContext {
    kvcache::KVCacheConfig config_k_;
    kvcache::KVCacheConfig config_v_;
    size_t num_layers;

public:
    RagBufferContext(KVCacheConfig config_k, KVCacheConfig config_v) :
        config_k_(config_k), config_v_(config_v) {
        BM_ASSERT_EQ(config_k.num_layers, config_v.num_layers, "num_layers mismatch");
        this->num_layers = config_k.num_layers;
    }
}; // end of struct RagBufferContext

} // namespace model
