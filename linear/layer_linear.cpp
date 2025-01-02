#include <pybind11/pybind11.h>
#include <bmengine/core/core.h>
#include "backend/linear.h"
#include "backend/linear_base.h"
#include <string>
#include <iostream>

using namespace bmengine;
namespace py = pybind11;

class PyLinear : public PyLayerBase<Linear> {
public:
    PyLinear(
        int dim_model,
        int dim_ff,
        std::string act_fn_type,
        bool quant,
        bool scale,
        bool weight_transposed,
        std::string &dtype_name) :
        PyLayerBase<Linear>(
            dim_model,
            dim_ff,
            act_fn_type,
            scale,
            weight_transposed,
            false,
            core::DistLayout::COLUMNAR,
            core::name_to_data_type(dtype_name)) {
        printf("PyLinear constructor\n");
    }
    static PyLinear create(
        int dim_model,
        int dim_ff,
        std::string act_fn_type,
        bool quant,
        bool scale = false,
        bool weight_transposed = false,
        std::string dtype_name = "half") {
        printf("PyLinear::create\n");
        auto obj = PyLinear(dim_model, dim_ff, act_fn_type, quant, scale, weight_transposed, dtype_name);
        return obj;
    }
}; // class end PyLinear

void define_layer_linear(py::module_ &layers_m) {
    py::class_<PyLinear>(layers_m, "Linear")
        .def(py::init(&PyLinear::create));
    // .def("load_state_dict", &PyLinear::load_state_dict);
}

PYBIND11_MODULE(llm_nodes, handle) {
    py::module_ layers_m = handle.def_submodule("layers", "internal layers for testing.");

    define_layer_linear(layers_m);
}
