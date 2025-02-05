#include <memory>
#include <assert.h>

#include "backend/transformer_buffer.h"

namespace kvcache {

TransformerBuffer::TransformerBuffer(int batch_size,
                                     int num_layers,
                                     int num_heads,
                                     int dim_head,
                                     core::DataType dtype,
                                     bool parallel,
                                     bool BSHD) :
    KVCache(batch_size, num_layers, num_heads, dim_head, dtype, parallel, BSHD) {
    BM_ASSERT(is_dyn_batch() || batch_size > 0, "batch_size must be greater than 0");
    BM_ASSERT(num_layers > 0, "num_layers must be greater than 0");
    BM_ASSERT(num_heads > 0, "num_heads must be greater than 0");
    BM_ASSERT(dim_head > 0, "dim_head must be greater than 0");
    buffer.resize(num_layers);
}

TransformerBuffer::TransformerBuffer(const KVCacheConfig &c) :
    TransformerBuffer(-1, c.num_layers, c.num_heads, c.dim_head, c.dtype, false, c.BSHD) {
    scale_dtype_ = c.scale_dtype;
    layer_devices = c.layer_devices;
    std::cout << "batch_size=" << batch_size << ", num_layers:" << this->num_layers << ", num_heads=" << this->num_heads
              << ", BSHD=" << this->BSHD << std::endl;
}

TransformerBuffer::~TransformerBuffer() {
}

void TransformerBuffer::check_layer(int i) const {
    BM_ASSERT(i >= 0 && i < num_layers,
              "Invalid layer index: i must in [0, " + std::to_string(num_layers) + "), but got "
                  + std::to_string(i));
}

const core::Tensor &TransformerBuffer::operator[](int i) const {
    check_layer(i);
    return buffer[i];
}

core::Tensor &TransformerBuffer::operator[](int i) {
    check_layer(i);
    return buffer[i];
}

const core::Tensor &TransformerBuffer::get_scale(int i) const {
    BM_ASSERT_EQ(num_layers, scales_.size(), "Wrong scales_ size");
    check_layer(i);
    return scales_[i];
}

} // namespace kvcache
