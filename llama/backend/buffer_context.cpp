#include "backend/buffer_context.h"
#include "backend/kvcache.h"
#include "backend/transformer_buffer.h"

namespace model {

using namespace kvcache;
using kvcache::TransformerBuffer;

TransformerBufferContext::TransformerBufferContext(
    const ModelBase &m, int batch_size, bool parallel, int world_size, bool BSHD) :
    BufferContext(m, parallel),
    BSHD(BSHD) {
    int num_kv_heads = parallel ? m.num_kv_heads / world_size : m.num_kv_heads;

    std::cout << "TransformerBufferContext " << parallel << " " << world_size << std::endl;
    buf_k_ = std::make_shared<TransformerBuffer>(
        batch_size, m.num_layers, num_kv_heads, m.dim_head, m.dtype, parallel, BSHD);
    buf_v_ = std::make_shared<TransformerBuffer>(
        batch_size, m.num_layers, num_kv_heads, m.dim_head, m.dtype, parallel, BSHD);
}

TransformerBufferContext::~TransformerBufferContext() = default;

Tensor *TransformerBufferContext::buf_k(size_t layer) {
    return &(*buf_k_)[layer];
}

Tensor *TransformerBufferContext::buf_v(size_t layer) {
    return &(*buf_v_)[layer];
}

const Tensor *TransformerBufferContext::block_table(size_t layer) {
    return nullptr;
}

void TransformerBufferContext::set_layer_devices(const std::vector<int> &layer_devices) {
}

} // namespace model
