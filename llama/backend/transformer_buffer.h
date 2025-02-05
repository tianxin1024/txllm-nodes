#pragma once
#include <bmengine/core/core.h>
#include "backend/kvcache.h"

namespace kvcache {

using namespace bmengine;

class TransformerBuffer : public KVCache {
    std::vector<core::Tensor> buffer;
    core::Tensor all_scale_; // scale for all layers. shape: (num_layers, S, H)
    std::vector<core::Tensor> scales_;
    std::shared_ptr<core::DataType> scale_dtype_;

    void check_layer(int i) const;
    bool is_dyn_batch() {
        return batch_size == -1;
    }

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
