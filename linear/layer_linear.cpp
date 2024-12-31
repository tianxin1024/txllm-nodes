#include <pybind11/pybind11.h>
#include <string>

namespace py = pybind11;

class PyLinear {
public:
    PyLinear(int dim_model, int dim_ff, std::string act_fn_type,
             int quant, bool scale, bool weight_transposed,
             std::string &dtype_name) {
    }

    static PyLinear create(int dim_model, int dim_ff, std::string act_fn_type,
                           int quant, bool scale = false, bool weight_transposed = false,
                           std::string dtype_name = "half") {
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
