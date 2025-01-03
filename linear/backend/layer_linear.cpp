#include <iostream>
#include <bmengine/core/core.h>
#include <pybind11/pybind11.h>
#include "backend/linear.h"
#include "backend/utils.h"

namespace py = pybind11;

using namespace bmengine;

class PyLinear : public PyLayerBase<nn::Linear> {
public:
    PyLinear(
        int dim_model,
        int dim_ff,
        std::string act_fn_type,
        bool quant,
        bool scale,
        bool weight_transposed,
        std::string &dtype_name) :
        PyLayerBase<nn::Linear>(dim_model,
                                dim_ff,
                                act_fn_type,
                                scale,
                                weight_transposed,
                                false,
                                core::DistLayout::COLUMNAR,
                                utils::name_to_data_type(dtype_name)) {
        std::cout << "PyLinear constructor" << std::endl;
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
        .def(py::init(&PyLinear::create))
        .def("load_state_dict", &PyLinear::load_state_dict);
}

PYBIND11_MODULE(llm_nodes, handle) {
    py::module_ layers_m = handle.def_submodule("layers", "internal layers for testing.");

    define_layer_linear(layers_m);
}
