#pragma once

#include "backend/transformer_buffer.h"
#include "backend/model_context.h"
#include "backend/lru_cache.h"

namespace batch_generator {

using bmengine::core::Tensor;
using kvcache::TransformerBuffer;
using model::ModelContext;

typedef TransformerBuffer *BufferPtr;
typedef std::pair<BufferPtr, BufferPtr> BufferPair;

// Cache KV by chunk
class PrefixCache {
    typedef std::vector<int> Key; // or std::array?

    size_t block_size;
    size_t max_block;
    utils::LRUCache<Key, int, utils::IntVecHasher> lru_cache;

    Tensor cache_mem;
    std::deque<int> unused_block;

    std::vector<Key> to_blocks(const std::vector<int> &tokens) {
        size_t num_block = (tokens.size() - 2) / block_size; // Reserve last token for search
        num_block = std::min(num_block, max_block);
        std::vector<Key> blocks;
        blocks.reserve(num_block);
        for (size_t i = 0; i < num_block; ++i) {
            auto begin = tokens.begin() + i * block_size;
            blocks.emplace_back(begin, begin + block_size);
            blocks.back().push_back(int(i)); // Add block index into key
        }
        return std::move(blocks);
    }

    void load_block(ModelContext &ctx, const BufferPair &kv_buffer, size_t start, size_t len, int block_id) {
        auto p = cache_mem.slice_dim0_len(block_id * 2, 2).chunk();
        kv_buffer.first->load_slice(ctx, start, len, p[0]);  // k
        kv_buffer.second->load_slice(ctx, start, len, p[1]); // v
    }

public:
    PrefixCache(ModelContext &ctx,
                size_t num_block,
                size_t block_size,
                size_t max_block,
                size_t num_layers,
                size_t num_heads,
                size_t dim_head,
                bmengine::core::DataType dtype) :
        block_size(block_size),
        max_block(max_block),
        lru_cache(num_block) {
        cache_mem = ctx.is_BSHD() ?
                        ctx.tensor({num_block * 2, num_layers, block_size, num_heads, dim_head}, dtype) :
                        ctx.tensor({num_block * 2, num_layers, num_heads, block_size, dim_head}, dtype);
        for (int i = 0; i < num_block; ++i) {
            unused_block.push_back(i);
        }
    }

    int get(ModelContext &ctx, const std::vector<int> &tokens, const BufferPair &kv_buffer) {
        std::vector<Key> blocks = to_blocks(tokens);
        size_t matched_len = 0;
        for (size_t i = 0; i < blocks.size(); ++i) {
            int block_id;
            if (!lru_cache.get(blocks[i], block_id)) {
                break;
            }
            load_block(ctx, kv_buffer, i * block_size, block_size, block_id);
            matched_len += block_size;
        }
        return matched_len;
    }

}; // end of class PrefixCache

} // namespace batch_generator
