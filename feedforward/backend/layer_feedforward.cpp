#include <iostream>
#include <bmengine/core/core.h>
#include <pybind11/pybind11.h>
#include "backend/feedforward.h"

namespace py = pybind11;

using namespace bmengine;

class PyFeedForward {
private:
    std::vector<std::shared_ptr<FeedForward>> mds;
    std::shared_ptr<bmengine::core::Engine> engine;
    int dim_model;
    int dim_ff;
    std::string act_fn_type;
    int quant;
    bool scale_weights;
    bool weight_transposed;

    PyFeedForward(int dim_model,
                  int dim_ff,
                  std::string act_fn_type,
                  int quant,
                  bool scale_weights = false,
                  bool weight_transposed = true) :
        dim_model(dim_model),
        dim_ff(dim_ff), act_fn_type(act_fn_type),
        quant(quant), scale_weights(scale_weights), weight_transposed(weight_transposed) {
    }

public:
    ~PyFeedForward() {
        mds.clear();
    }

    static PyFeedForward create(int dim_model,
                                int dim_ff,
                                std::string act_fn_type,
                                int quant,
                                bool scale_weights = false,
                                bool weight_transposed = true) {
        auto ff = PyFeedForward(dim_model, dim_ff, act_fn_type, quant, scale_weights, weight_transposed);
        return ff;
    }

}; // end of class PyFeedForward

void define_layer_linear(py::module_ &layers_m) {
    py::class_<PyFeedForward>(layers_m, "Linear")
        .def(py::init(&PyFeedForward::create));
    // .def("load_state_dict", &PyFeedForward::load_state_dict);
    // .def("named_parameters", &PyLinear::named_parameters);
}

PYBIND11_MODULE(llm_nodes, handle) {
    py::module_ layers_m = handle.def_submodule("layers", "internal layers for testing.");

    define_layer_linear(layers_m);
}
