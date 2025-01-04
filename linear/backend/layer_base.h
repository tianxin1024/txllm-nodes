#pragma once
#include <bmengine/core/core.h>
#include <pybind11/pybind11.h>
#include <ATen/ATen.h>
#include "backend/py_utils.h"

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

    ~PyLayerBase() {
        // order matters.
        layer = nullptr;
        with_device = nullptr;
        ctx = nullptr;
    }

    PyLayerBase(const PyLayerBase &other) = delete;
    PyLayerBase(PyLayerBase &&other) {
        layer = std::move(other.layer);
        engine = std::move(other.engine);
        ctx = std::move(other.ctx);
        with_device = std::move(other.with_device);
    }

    void load_state_dict(const std::map<std::string, at::Tensor> &state_dict) {
        std::cout << "PyLayerBase: load_state_dict" << std::endl;
        auto named_params = layer->named_parameters("", true);
        bind::load_at_state_dict(*ctx, state_dict, named_params);
    }

    std::map<const std::string, at::Tensor> named_parameters() {
        std::cout << "PyLayerBase: named_parameters" << std::endl;
        std::map<const std::string, at::Tensor> result;
        // auto named_params = layer->named_parameters("", true);
        // for (auto it : named_params) {
        // result.emplace(it.first, bind::tensor_to_aten(*ctx, *it.second));
        // }
        return result;
    }

}; // end of class PyLayerBase
