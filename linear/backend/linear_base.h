#pragma once
#include <bmengine/core/core.h>

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

}; // end of class PyLayerBase
