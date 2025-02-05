#include "backend/kvcache.h"

namespace kvcache {

KVCache::KVCache(int batch_size,
                 int num_layers,
                 int num_heads,
                 int dim_head,
                 core::DataType dtype,
                 bool parallel,
                 bool BSHD) :
    batch_size(batch_size),
    num_layers(num_layers),
    dim_head(dim_head),
    dtype(dtype),
    parallel(parallel),
    BSHD(BSHD),
    layer_devices(num_layers) {
}

} // namespace kvcache
