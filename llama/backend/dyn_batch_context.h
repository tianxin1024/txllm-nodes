#pragma once

#include <bmengine/core/core.h>

namespace model {

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

} // namespace model
