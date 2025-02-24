#pragma once

#include <bmengine/core/core.h>
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

    // flying hypo<->pos mapping, tracked postions are valid till next input.
    // picked hypo positions are refcount increased, remains are garbage collected.
    std::vector<int> head_placement_;

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

    int pop_unused_pos() {
        BM_ASSERT(!unused_buffer_pos.empty(), "extend_buffer should be called before pop");
        int pos = unused_buffer_pos.back();
        unused_buffer_pos.pop_back();
        return pos;
    }

    int place_token(const BeamBufferInfo<TokenT> &token) {
        int place = pop_unused_pos();
        buf_local[place] = token;
        return place;
    }
    void init(const std::vector<TokenT> &input, int len_input) {
        // fill inputs
        for (int i = 0; i < len_input; i++) {
            buf_local[i] = BeamBufferInfo<TokenT>(input[i], i - 1, 0.0, 1);
        }
        last_input_buf_pos = len_input - 1;
        // unused pos: len_input, len_input+1 ... len_buf-1 in reverse order
        for (int i = 0; i < len_buf - len_input; i++) {
            unused_buffer_pos.push_back(len_buf - i - 1);
        }
        head_placement_.emplace_back(last_input_buf_pos);
    }

    void reset(int new_len_buf) {
        len_buf = new_len_buf;
        buf_local.clear();
        buf_local.resize(len_buf);
        unused_buffer_pos.clear();
        last_input_buf_pos = -1;
    }

    // mask: matrix: [x, len_buf]
    void mask_hypothesis(int8_t *mask, size_t hyp_i, int last_pos) const {
        // 往回遍历该 hypothesis 所有的 token 对应的 pos，包含 input
        int pos = last_pos;
        size_t stride = mask_stride == -1 ? len_buf : size_t(mask_stride);
        while (pos != -1) {
            mask[hyp_i * stride + pos] = 1; // mask[i][pos]
            pos = buf_local[pos].prev;
        }
    }

    void mask_hypothesis(std::vector<int8_t> &mask, size_t hyp_i, int last_pos) const {
        mask_hypothesis(mask.data(), hyp_i, last_pos);
    }

    void mask_hypotheses(
        const std::vector<int> &hypotheses_end_pos, std::vector<int8_t> &mask) const {
        size_t stride = mask_stride == -1 ? len_buf : size_t(mask_stride);
        mask.resize(hypotheses_end_pos.size() * stride);
        std::fill(mask.begin(), mask.end(), 0);
        for (size_t i = 0; i < hypotheses_end_pos.size(); i++) {
            mask_hypothesis(mask, i, hypotheses_end_pos[i]);
        }
    }

    void mask_hypotheses(const int *hypotheses_last_pos, size_t hyp_num, int8_t *mask) {
        for (size_t i = 0; i < hyp_num; i++) {
            mask_hypothesis(mask, i, hypotheses_last_pos[i]);
            head_placement_.emplace_back(hypotheses_last_pos[i]);
        }
    }

    void mask_input(int8_t *mask, int len_input, int stride = -1, int pos = 0) const {
        stride = stride == -1 ? len_buf : stride;
        for (size_t i = 0; i < len_input; i++) {
            for (size_t j = 0; j < stride; j++) {
                mask[i * stride + j] = (i + pos) < j ? 0 : 1;
            }
        }
    }

}; // end of class BeamBufferManager

} // namespace beam_utility
