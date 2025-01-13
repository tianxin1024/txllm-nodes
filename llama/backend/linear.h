#pragma once
#include <bmengine/core/core.h>
#include "backend/model.h"

namespace nn {
using namespace bmengine;

class Linear : public core::Layer {
    Linear() = default;
    BM_LAYER_DEF(Linear);

public:
    Linear(const core::Context &ctx,
           int dim_in,
           int dim_out,
           std::string act_fn_type,
           model::QuantConfig quant,
           bool scale_weights = false,
           bool weight_transposed = true,
           bool parallel = false,
           core::DistLayout dist_layout = core::DistLayout::COLUMNAR,
           core::DataType dtype = core::DataType::kHalf);

    Linear(const core::Context &ctx,
           int dim_in,
           int dim_out,
           model::QuantConfig quant,
           core::DistLayout dist_layout,
           core::DataType dtype = core::DataType::kHalf);

    Linear(const core::Context &ctx,
           const std::string &name,
           const core::Tensor &w);

}; // end of class Linear

} // namespace nn
