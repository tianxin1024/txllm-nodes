#pragma once

#include <bmengine/core/core.h>
#include <memory>
#include <vector>

namespace kvcache {

using namespace bmengine;

class KVCache {
protected:
    int batch_size;
    size_t num_layers, num_heads, dim_head;
    core::DataType dtype;
    std::vector<int> layer_devices;
    bool parallel;
    bool BSHD;
    bool continuous{false};

public:
    KVCache(int num_layers,
            int num_heads,
            int dim_head,
            core::DataType dtype,
            bool parallel = false,
            bool BSHD = false) :
        KVCache(-1, num_layers, num_heads, dim_head, dtype, parallel, BSHD) {
    }

    KVCache(int batch_size,
            int num_layers,
            int num_heads,
            int dim_head,
            core::DataType dtype,
            bool parallel,
            bool BSHD);

}; // end of class KVCache

} // namespace kvcache
