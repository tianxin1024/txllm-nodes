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
    core::Tensor copy(const core::Context &ctx,
                      int layer,
                      const core::Tensor &src,
                      const core::Tensor &placement,
                      int start = 0,
                      bool need_dequant = false);

    const core::Tensor &get_scale(int i) const;
    bool is_quantized() const {
        return scale_dtype_.get();
    }

    void load_slice(core::Context &ctx, size_t start, size_t len, const core::Tensor &input);

}; // end of class TransformerBuffer

void copy_to_buffer(int num_heads,
                    int len_kv,
                    int len_buf,
                    int dim_head,
                    const core::Tensor *placement,
                    const core::Tensor &src,
                    const core::Tensor &dst,
                    cudaStream_t stream,
                    bool BSHD = false); // batch, (seqlen, num_heads|num_heads, seqlen), dim_head

core::Tensor resize_buffer(const core::Context &ctx, const core::Tensor &buffer, int dim, size_t new_length);

} // namespace kvcache
