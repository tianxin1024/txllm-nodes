#pragma once
#include <bmengine/core/core.h>

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
        bool scale_weights = false,
        bool weight_transposed = true,
        bool parallel = false,
        core::DistLayout dist_layout = core::DistLayout::COLUMNAR,
        core::DataType dtype = core::DataType::kHalf);

    Linear(
        const core::Context &ctx,
        int dim_in,
        int dim_out,
        core::DistLayout dist_layout,
        core::DataType dtype = core::DataType::kHalf);

    Linear(
        const core::Context &ctx,
        const std::string &name,
        const core::Tensor &w);

    void move(Linear &other);

    void load_state_dict(
        const core::Context &ctx,
        const std::map<std::string, const core::Tensor> &state_dict,
        const std::string &prefix,
        bool allow_missing) override;
};
