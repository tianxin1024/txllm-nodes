#pragma once
#include <bmengine/core/core.h>
#include "backend/kvcache.h"

namespace kvcache {

using namespace bmengine;

class TransformerBuffer : public KVCache {
public:
    TransformerBuffer(int batch_size,
                      int num_layers,
                      int num_heads,
                      int dim_head,
                      core::DataType dtype,
                      bool parallel,
                      bool BSHD);
    TransformerBuffer(const KVCacheConfig &config);
    ~TransformerBuffer();

    const core::Tensor &operator[](int i) const override;
    core::Tensor &operator[](int i) override;

    const core::Tensor &get_layer(int i) const {
        return this->operator[](i);
    }

    const core::Tensor &get_scale(int i) const;

}; // end of class TransformerBuffer

} // namespace kvcache
