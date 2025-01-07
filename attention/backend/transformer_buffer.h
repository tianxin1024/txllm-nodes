#pragma once

#include <memory>
#include <vector>
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
    void resize_dyn_batch(const core::Context &ctx, size_t new_length);

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

    void resize(const core::Context &ctx, size_t new_length) override;
}; // end of class TransformerBuffer

core::Tensor resize_buffer(
    const core::Context &ctx, const core::Tensor &buffer, int dim, size_t nw_length);

} // namespace kvcache
