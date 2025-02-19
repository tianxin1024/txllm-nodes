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
        std::cout << ">>>>>> AddressVector get_buf_addresses " << std::endl;
    }

    bool has_v() const {
        return config_v_.dim_head > 0;
    }

public:
    RagBufferContext(KVCacheConfig config_k, KVCacheConfig config_v) :
        config_k_(config_k), config_v_(config_v) {
        BM_ASSERT_EQ(config_k.num_layers, config_v.num_layers, "num_layers mismatch");
        this->num_layers = config_k.num_layers;
    }

    void check_task_index(int i_task) {
        if (i_task >= buf_k_.size()) {
            throw std::out_of_range("Invalid task index");
        }
    }

    TransformerBuffer &buf_k(int b) {
        check_task_index(b);
        return (*buf_k_[b]);
    }
    TransformerBuffer &buf_v(int b) {
        check_task_index(b);
        return (*buf_v_[b]);
    }
    const Tensor &buf_k(int b, int layer) {
        check_task_index(b);
        return (*buf_k_[b])[layer];
    }
    const Tensor &buf_v(int b, int layer) {
        check_task_index(b);
        return (*buf_v_[b])[layer];
    }

    void resize_task_buf(const core::Context &ctx, int b, size_t new_length) {
        if (buf_k_.size() < b + 1) {
            buf_k_.resize(b + 1);
            buf_v_.resize(b + 1);
        }
        if (!buf_k_[b]) {
            buf_k_[b] = std::make_unique<TransformerBuffer>(config_k_);
            if (has_v()) {
                buf_v_[b] = std::make_unique<TransformerBuffer>(config_v_);
            }
        }
        buf_k_[b]->resize(ctx, new_length);
        if (has_v()) {
            buf_v_[b]->resize(ctx, new_length);
        }
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
        if (active_batch() > 0 && has_v()) {
            h_buf_v_addresses = get_buf_addresses(BUF_V, &TransformerBuffer::get_layer);
            buf_v_addresses = ctx.tensor_of(h_buf_v_addresses, {num_layers, active_batch()});
            if (config_v_.is_quant()) {
                h_scale_v_addresses = get_buf_addresses(BUF_V, &TransformerBuffer::get_scale);
                scale_v_address = ctx.tensor_of(h_scale_v_addresses, {num_layers, active_batch()});
            }
        }
    }

    size_t get_buf_len(size_t b) {
        if (b < buf_k_.size()) {
            auto &t = (*buf_k_[b])[0];
            return t.numel() ? t.size(buf_k_[b]->is_BSHD() ? -3 : -2) : 0;
        }
        return 0;
    }

}; // end of struct RagBufferContext

} // namespace model
