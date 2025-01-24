#pragma once

#include <bmengine/core/core.h>

namespace model {

using bmengine::core::Tensor;
using std::vector;

static inline bool is_power_of_2(int x) {
    return (x > 0) && !(x & (x - 1));
}

#define CHECK_IS_POWER_OF_2(x)                                                              \
    do {                                                                                    \
        if (!is_power_of_2(x)) {                                                            \
            throw std::invalid_argument(#x "=" + std::to_string(x) + " is not power of 2"); \
        }                                                                                   \
    } while (0)

struct DynBatchConfig {
    int max_batch{20};
    int max_beam_size{1 * 8}; // set to n * 8 for best preformance
    int task_queue_size{8};
    int max_total_token{4096}; // input and output tokens
    int chunk_size{256};
    int max_chunk{20};
    int seed{0};
    int eos_id{0};
    int bos_id{6};
    int unk_id{1};
    int first_batch{1};
    int nccl{-1};
    bool rag_buffer{false};
    bool ignore_eos{false};
    bool keep_eos{false};
    int reserved_work_mem_mb{1024};
    int high_precision{0};
    bool enable_prompt_caching{false};
    bool flash_attention{false};
};

struct RopeCache {
    Tensor cos;
    Tensor sin;
    void clear() {
        cos = Tensor();
        sin = Tensor();
    }
};

struct DynBatchContext {
    RopeCache rope_cache;
    // search
    Tensor s_token;
    Tensor s_sub;
    Tensor s_placement;
    Tensor s_position;
    Tensor s_mask;
    vector<Tensor> s_position_buckets; // for CPMBee ragged buffer
    vector<Tensor> s_position_biases;  // for CPMBee ragged buffer
    vector<int> sv_len_buf;            // for ragged buffer
    Tensor s_len_buf;                  // for ragged buffer
    // encode
    Tensor e_token;
    Tensor e_sub;
    Tensor e_placement;
    Tensor e_position;
    Tensor e_mask;
    vector<Tensor> e_position_buckets; // for CPMBee
    vector<Tensor> e_position_biases;  // for CPMBee

    vector<int> ev_batch;
    Tensor e_batch;
    vector<int> ev_input_len;
    vector<int> full_input_len;
    Tensor e_input_len;
    vector<int> ev_len_buf;

    int debug_batch{-1};

    std::vector<void *> host_position_bias_addresses;
    Tensor position_bias_addresses;

    Tensor cu_q_seqlens; // for FlashDecoding
    Tensor cu_k_seqlens; // for FlashDecoding
    size_t total_k{0};
    int max_q_seqlen{0};
    int max_k_seqlen{0};

    std::shared_ptr<Tensor> unquant_key_buf;
    std::shared_ptr<Tensor> unquant_val_buf;
    int input_len_no_split;

    void set_search(const Tensor &token_ids,
                    const Tensor &token_sub,
                    const Tensor &placement,
                    const Tensor &position,
                    const Tensor &mask) {
        s_token = token_ids;
        s_sub = token_sub;
        s_placement = placement;
        s_position = position;
        s_mask = mask;
    }

    void set_encode(const Tensor &token_ids,
                    const Tensor &token_sub,
                    const Tensor &placement,
                    const Tensor &position,
                    const Tensor &mask) {
        e_token = token_ids;
        e_sub = token_sub;
        e_placement = placement;
        e_position = position;
        e_mask = mask;
    }

    void set_encode_batch(const vector<int> &v_batch, const Tensor &batch) {
        this->ev_batch = v_batch;
        this->e_batch = batch;
    }

    void set_encode_len(const vector<int> &v_input_len,
                        const vector<int> &full_input_lens,
                        const Tensor &input_len) {
        this->ev_input_len = v_input_len;
        this->full_input_len = full_input_lens;
        this->e_input_len = input_len;
    }

    void clear_encode() {
        set_encode(Tensor(), Tensor(), Tensor(), Tensor(), Tensor());
        set_encode_batch(vector<int>(), Tensor());
        set_encode_len({}, {}, Tensor());
        ev_len_buf.clear();
    }

}; // end of struct DynBatchContext

} // namespace model
