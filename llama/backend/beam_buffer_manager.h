#pragma once

#include "backend/generator.h"
#include <algorithm>
#include <vector>
#include <deque>

namespace beam_utility {

template <typename TokenT>
class BeamBufferInfo {
public:
    TokenT token;
    int prev;
    float log_prob;
    int ref_count;
    int hyp_id;

    BeamBufferInfo(TokenT token, int prev, float log_prob = 0, int ref_count = 0, int hyp_id = 0) :
        token(token), prev(prev), log_prob(log_prob), ref_count(ref_count), hyp_id(hyp_id) {
    }
    BeamBufferInfo() :
        token(), prev(-1), log_prob(0), ref_count(-1) {
    }
    ~BeamBufferInfo() = default;
}; // end of class BeamBufferInfo

// clang-format off
/**
 * 为了实现在单个 buffer 上做 beam search，需要管理 beam_size 个 beam hypotheses sentences 的 pos 分配
 * [示例]
 *   pos:        0, 1, 2, 3, 4, 5, 6, 7
 *   ref_count:  1, 1, 1, 1, 2, 2, 1, 1
 *   inputs:     X, A, B, C
 *   hyp1:                   D, E, F
 *   hyp2:                   D, E,    G
 * hyp1 当前使用了 pos 4,5,6
 * hyp2 当前使用了 pos 4,5,7
 */
// clang-format on

template <typename TokenT>
class BeamBufferManager {
public:
    std::vector<BeamBufferInfo<TokenT>> buf_local;
    int len_buf;
    std::deque<int> unused_buffer_pos;
    int last_input_buf_pos{-1};
    int mask_stride{-1};

    explicit BeamBufferManager(int len_buf) :
        buf_local(len_buf), len_buf(len_buf) {
    }

    BeamBufferManager(const BeamBufferManager &other) = default;

    BeamBufferManager &operator=(BeamBufferManager &&other) {
        // use swap and reset() to reserve other's vector's capacity
        buf_local.swap(other.buf_local);
        len_buf = other.len_buf;
        unused_buffer_pos.swap(other.unused_buffer_pos);
        last_input_buf_pos = other.last_input_buf_pos;
        other.reset(0);
        return *this;
    }

}; // end of class BeamBufferManager

} // namespace beam_utility