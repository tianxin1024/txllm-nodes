#include <bmengine/core/core.h>
#include <pybind11/pybind11.h>

#include "backend/model_config.h"

using namespace bmengine;
namespace py = pybind11;

class PyAttention : public PyLayerBase<Attention> {
private:
    model::ModelConfig model_config;

public:
    PyAttention(model::ModelConfig model_config, bool parallel = false) :
        PyLayerBase<Attention>(model_config, 0, parallel), model_config(model_config) {
    }
    static PyAttention create(int dim_model, int num_heads, int dim_head, std::string pos_bias_type,
                              int quant, bool scale_weights = false, bool weight_transposed = true,
                              bool parallel = false) {
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
