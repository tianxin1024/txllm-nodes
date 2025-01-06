#pragma once
#include <bmengine/core/core.h>
#include "backend/model_config.h"

using namespace bmengine;

class Linear : public core::Layer {
    Linear() = default;
    BM_LAYER_DEF(Linear);

public:
    Linear(
        const core::Context &ctx,
        int dim_in,
        int dim_out,
        std::string act_fn_type,
        model::QuantConfig quant,
        bool scale_weights = false,
        bool weight_transposed = true,
        bool parallel = false,
        core::DistLayout dist_layout = core::DistLayout::COLUMNAR,
        core::DataType dtype = core::DataType::kHalf);

    Linear(
        const core::Context &ctx,
        int dim_in,
        int dim_out,
        model::QuantConfig quant,
        core::DistLayout dist_layout,
        core::DataType dtype = core::DataType::kHalf);

    Linear(
        const core::Context &ctx,
        const std::string &name,
        const core::Tensor &w);

    void move(Linear &other);

    bool support_fuse_gptq_gate_in(const core::Tensor &input);

    core::Tensor forward(const core::Context &ctx,
                         const core::Tensor &x,
                         bool quant_back = true,
                         core::Tensor *output = nullptr);
};
