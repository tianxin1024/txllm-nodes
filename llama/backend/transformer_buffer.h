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
    void resize_dyn_batch(const core::Context &ctx, size_t new_length);
    void resize_multi_layer(const core::Context &ctx, size_t new_length, int begin, int end, int dim);
    void resize_scale(const core::Context &ctx, size_t new_length, int begin, int end, int dim);

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

    void resize(const core::Context &ctx, size_t new_length) override;

    const core::Tensor &get_scale(int i) const;

}; // end of class TransformerBuffer

core::Tensor resize_buffer(const core::Context &ctx, const core::Tensor &buffer, int dim, size_t new_length);

} // namespace kvcache
