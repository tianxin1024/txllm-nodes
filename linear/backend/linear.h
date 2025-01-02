#pragma once
#include <bmengine/core/core.h>

using namespace bmengine;

namespace model {

struct QuantConfig {
    int quant;
    bool scale;
};

} // namespace model

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
           bool parallen = false,
           core::DistLayout dist_layout = core::DistLayout::COLUMNAR,
           core::DataType dtype = core::DataType::kHalf);

    Linear(const core::Context &ctx,
           int dim_in,
           int dim_out,
           model::QuantConfig quant,
           core::DistLayout dist_layout,
           core::DataType dtype = core::DataType::kHalf);
};
