#include "backend/buffer_context.h"
#include "backend/kvcache.h"

namespace model {

using namespace kvcache;

TransformerBufferContext::TransformerBufferContext(
    const ModelBase &m, int batch_size, bool parallel, int world_size, bool BSHD) :
    BufferContext(m, parallel),
    BSHD(BSHD) {
    int num_kv_heads = parallel ? m.num_kv_heads / world_size : m.num_kv_heads;

    std::cout << "TransformerBufferContext " << parallel << " " << world_size << std::endl;
    // buf_k_ = std::make_shared<TransformerBuffer>(
    //     batch_size, m.num_layers, num_kv_heads, m.dim_head, m.dtype, parallel, BSHD);
    // buf_v_ = std::make_shared<TransformerBuffer>(
    //     batch_size, m.num_layers, num_kv_heads, m.dim_head, m.dtype, parallel, BSHD);
}

TransformerBufferContext::~TransformerBufferContext() {
}

void TransformerBufferContext::set_layer_devices(const std::vector<int> &layer_devices) {
}

} // namespace model
