// #pragma once

// #include <memory>
// #include <vector>
// #include <bmengine/core/core.h>
// #include "backend/kvcache.h"

// namespace kvcache {

// using namespace bmengine;

// class TransformerBuffer : public KVCache {
//     std::vector<core::Tensor> buffer;
//     core::Tensor all_scale_; // scale for all layers. shape: (num_layers, S, H)
//     std::vector<core::Tensor> scales_;
//     std::shared_ptr<core::DataType> scale_dtype_;

// public:
//     TransformerBuffer(int batch_size,
//                       int num_layers,
//                       int num_heads,
//                       int dim_head,
//                       core::DataType dtype,
//                       bool parallel,
//                       bool BSHD);

//     TransformerBuffer(const KVCacheConfig &config);
//     ~TransformerBuffer();
// };

// } // namespace kvcache
