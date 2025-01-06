#include <bmengine/core/core.h>
#include <pybind11/pybind11.h>

#include "backend/model_config.h"
#include "backend/attention.h"

using namespace bmengine;
namespace py = pybind11;

class PyAttention {
private:
    std::shared_ptr<Attention> md;
    std::shared_ptr<bmengine::core::Engine> engine;
    std::shared_ptr<bmengine::core::Context> ctx;
    std::shared_ptr<bmengine::core::WithDevice> with_device;
    model::ModelConfig model_config;

public:
    PyAttention(model::ModelConfig model_config, bool parallel = false) :
        model_config(model_config) {
        std::vector<bmengine::core::DeviceConfiguration> devices;
        devices.emplace_back(0, (size_t)2 << 30);

        engine = std::make_shared<bmengine::core::Engine>(devices);
        ctx = std::make_shared<bmengine::core::Context>(engine->create_context({0}));
        with_device = std::make_shared<bmengine::core::WithDevice>(ctx->with_device(0));

        md = std::make_shared<Attention>(*ctx, model_config, 0, parallel);
    }

    static PyAttention create(int dim_model, int num_heads, int dim_head, std::string pos_bias_type,
                              int quant, bool scale_weights = false, bool weight_transposed = true,
                              bool parallel = false) {
        model::ModelConfig model_config("", 0, dim_model, num_heads, dim_head, 0, 0, 1e-6,
                                        -1, {}, scale_weights, weight_transposed, 0, 1.0, 1.0,
                                        bmengine::core::DataType::kHalf);
        auto layer = PyAttention(model_config, parallel);
        return layer;
    }

public:
    ~PyAttention() {
        md = nullptr;
    }
};

void define_layer_linear(py::module_ &layers_m) {
    py::class_<PyAttention>(layers_m, "Attention")
        .def(py::init(&PyAttention::create));
    // .def("init_parameters", &PyAttention::init_parameters)
    // .def("load_state_dict", &PyAttention::load_state_dict)
    // .def("named_parameters", &PyAttention::named_parameters)
    // .def("forward", &PyAttention::forward);
}

PYBIND11_MODULE(llm_nodes, handle) {
    py::module_ layers_m = handle.def_submodule("layers", "internal layers for testing.");

    define_layer_linear(layers_m);
}
