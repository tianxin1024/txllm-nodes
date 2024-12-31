#include <iostream>
#include <bmengine/core/core.h>
#include <pybind11/pybind11.h>
#include "backend/embedding.h"

namespace py = pybind11;

class PyEmbedding {
private:
    std::shared_ptr<Embedding> md;
    std::shared_ptr<bmengine::core::Engine> engine;
    int dim_model;
    int vocab_size;
    bool scale_weights;

    PyEmbedding(int dim_model, int vocab_size, bool scale_weights = false) :
        dim_model(dim_model), scale_weights(scale_weights) {
    }

public:
    ~PyEmbedding() {
        md = nullptr;
    }

    static PyEmbedding create(int dim_model, int vocab_size, bool scale_weights = false) {
        std::cout << ">>>>> PyEmbedding create function" << std::endl;
        auto ff = PyEmbedding(dim_model, vocab_size, scale_weights);
        return ff;
    }

}; // end of class PyEmbedding

void define_layer_embedding(py::module_ &layers_m) {
    py::class_<PyEmbedding>(layers_m, "Embedding")
        .def(py::init(&PyEmbedding::create));
}

PYBIND11_MODULE(llm_nodes, handle) {
    handle.doc() = "This is embedding llm_nodes.";
    py::module_ layers_m = handle.def_submodule("layers", "internal layers for testing.");
    define_layer_embedding(layers_m);
}
