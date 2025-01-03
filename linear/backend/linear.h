#pragma once
#include <bmengine/core/core.h>
#include <ATen/ATen.h>

using namespace bmengine;

template <typename LayerType>
class PyLayerBase {
protected:
    std::shared_ptr<LayerType> layer;
    std::shared_ptr<bmengine::core::Engine> engine;
    std::shared_ptr<bmengine::core::Context> ctx;
    std::shared_ptr<bmengine::core::WithDevice> with_device;

public:
    template <typename... Args>
    PyLayerBase(Args &&... args) {
        std::vector<bmengine::core::DeviceConfiguration> devices;
        devices.emplace_back(0, (size_t)2 << 30);

        engine = std::make_shared<bmengine::core::Engine>(devices);
        ctx = std::make_shared<bmengine::core::Context>(engine->create_context({0}));
        with_device = std::make_shared<bmengine::core::WithDevice>(ctx->with_device(0));

        layer = std::make_shared<LayerType>(*ctx, std::forward<Args>(args)...);
        printf(">>>> PyLayerBase constructor\n");
    }

    void load_state_dict(const std::map<std::string, bmengine::core::Tensor> &state_dict) {
        auto named_params = layer->named_parameters("", true);
        // bind::load_at_state_dict(*ctx, state_dict, named_params);
    }

    std::map<const std::string, at::Tensor> named_parameters() {
        std::map<const std::string, at::Tensor> result;
        // auto named_params = layer->named_parameters("", true);
        // for (auto it : named_params) {
        //     result.emplace(it.first, bind::tensor_to_aten(*ctx, *it.second));
        // }
        // return result;
    }

}; // end of class PyLayerBase

namespace nn {

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

    // void scale_output(float scale);
    // void set_output_type(core::DataType dtype);

    // const core::Tensor &get_weight() const;
    // core::Tensor get_dequant_weight(const core::Context &ctx) const;
    // const core::Tensor *get_weight_scale() const; // for quant

    // core::Tensor forward(
    //     const core::Context &ctx,
    //     const core::Tensor &x,
    //     bool quant_back = true,
    //     core::Tensor *output = nullptr);

    void init_parameters(
        const core::Context &ctx, curandGenerator_t &gen, const std::string &prefix = "") override;

    void load_state_dict(
        const core::Context &ctx,
        const std::map<std::string, const core::Tensor> &state_dict,
        const std::string &prefix,
        bool allow_missing) override;

    // static Linear *fuse(const core::Context &ctx, Linear &a, Linear &b);
    // static Linear *fuse(const core::Context &ctx, Linear &q, Linear &k, Linear &v);
    // static Linear *fuse(const core::Context &ctx, const std::vector<Linear *> &layers);
    // std::vector<Linear *> split(const core::Context &ctx, size_t n_split, bool dim_out);

    // bool support_fuse_gptq_gate_in(const core::Tensor &input);
    // std::tuple<core::Tensor, core::Tensor, core::Tensor, bool> get_gptq_weights();

    // void set_has_bias(bool b = true);

    // void dequant_cache_weight(core::Context &ctx, const core::Tensor &fake_input);
};
} // namespace nn
