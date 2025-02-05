#pragma once

#include "backend/model_context.h"
#include <bmengine/logger/std_log_op.hpp>
#include "backend/transformer_buffer.h"

#include <memory>

namespace model {

using kvcache::KVCacheConfig;

struct RagBufferContext {
    typedef std::vector<void *> AddressVector;

    kvcache::KVCacheConfig config_k_;
    kvcache::KVCacheConfig config_v_;
    size_t num_layers;
    std::vector<std::unique_ptr<TransformerBuffer>> buf_k_; // buffer per task
    std::vector<std::unique_ptr<TransformerBuffer>> buf_v_; // buffer per task

    AddressVector h_buf_k_addresses;
    AddressVector h_buf_v_addresses;
    AddressVector h_scale_k_addresses;
    AddressVector h_scale_v_addresses;
    Tensor buf_k_addresses; // (num_layers, batch)
    Tensor buf_v_addresses; // (num_layers, batch)
    Tensor scale_k_address; // (num_layers, batch)
    Tensor scale_v_address; // (num_layers, batch)

    bool skip_last{false};

private:
    typedef const Tensor &(TransformerBuffer::*BufPtr)(int i) const;
    static constexpr bool BUF_K = true;
    static constexpr bool BUF_V = false;
    size_t active_batch() const {
        return buf_k_.size() - size_t(skip_last);
    }

    AddressVector get_buf_addresses(bool is_k, BufPtr buf_ptr) {
    }

public:
    RagBufferContext(KVCacheConfig config_k, KVCacheConfig config_v) :
        config_k_(config_k), config_v_(config_v) {
        BM_ASSERT_EQ(config_k.num_layers, config_v.num_layers, "num_layers mismatch");
        this->num_layers = config_k.num_layers;
    }

    void set_buffer_addr(ModelContext &ctx) {
        if (active_batch() > 0) {
            h_buf_k_addresses = get_buf_addresses(BUF_K, &TransformerBuffer::get_layer);
            buf_k_addresses = ctx.tensor_of(h_buf_k_addresses, {num_layers, active_batch()});
            if (config_k_.is_quant()) {
                h_scale_k_addresses = get_buf_addresses(BUF_K, &TransformerBuffer::get_scale);
                scale_k_address = ctx.tensor_of(h_scale_k_addresses, {num_layers, active_batch()});
            }
            std::cout << "set_buffer_addr::buf_k_addresses: " << buf_k_addresses.shape() << std::endl;
        }
    }
}; // end of struct RagBufferContext

} // namespace model
