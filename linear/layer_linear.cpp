#include <pybind11/pybind11.h>
#include <bmengine/core/core.h>
#include <string>
#include <iostream>

using namespace bmengine;
namespace py = pybind11;

template <typename LayerType>
class PyLayerBase {
public:
    template <typename... Args>
    PyLayerBase(Args &&... args) {
    }

    ~PyLayerBase() {
        // order matters.
    }
}; // end of class PyLayerBase

class PyLinear : public PyLayerBase<Linear> {
public:
    PyLinear(int dim_model, int dim_ff, std::string act_fn_type,
             int quant, bool scale, bool weight_transposed,
             std::string &dtype_name) :
        PyLayerBase<Linear>(dim_model, dim_ff, act_fn_type,
                            quant, scale, weight_transposed, false,
                            core::DisLayout::COLUMNAR,
                            core::name_to_data_type(dtype_name)) {
        std::cout << " >>>> PyLinear constructor " << dim_model << " " << dim_ff << " " << act_fn_type << std::endl;
    }

    static PyLinear create(int dim_model, int dim_ff, std::string act_fn_type,
                           int quant, bool scale = false, bool weight_transposed = false,
                           std::string dtype_name = "half") {
        std::cout << "PyLinear create " << dim_model << " " << dim_ff << " " << act_fn_type << std::endl;
        auto ff = PyLinear(dim_model, dim_ff, act_fn_type, quant, scale, weight_transposed, dtype_name);
        return ff;
    }

}; // end of class PyLinear

void define_layer_linear(py::module_ &layers_m) {
    py::class_<PyLinear>(layers_m, "Linear")
        .def(py::init(&PyLinear::create));
}

PYBIND11_MODULE(llm_nodes, handle) {
    py::module_ layers_m = handle.def_submodule("layers", "internal layers for testing.");

    define_layer_linear(layers_m);
}
