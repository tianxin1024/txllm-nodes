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
        std::vector<bmengine::core::DeviceConfiguration> devices;
        devices.emplace_back(0, (size_t)2 << 28);

        engine = std::make_shared<bmengine::core::Engine>(devices);

        auto ctx = engine->create_context({0});
        {
            auto d = ctx.with_device(0);
            md = std::make_shared<Embedding>(ctx, dim_model, vocab_size, scale_weights);
        }
    }

public:
    ~PyEmbedding() {
        md = nullptr;
    }

    PyEmbedding(const PyEmbedding &other) {
        md = other.md;
        engine = other.engine;
        dim_model = other.dim_model;
        vocab_size = other.vocab_size;
        scale_weights = other.scale_weights;
    }
    PyEmbedding(PyEmbedding &&other) {
        md = std::move(other.md);
        engine = std::move(other.engine);
        dim_model = other.dim_model;
        vocab_size = other.vocab_size;
        scale_weights = other.scale_weights;
    }
    PyEmbedding &operator=(const PyEmbedding &other) {
        md = other.md;
        engine = other.engine;
        dim_model = other.dim_model;
        vocab_size = other.vocab_size;
        scale_weights = other.scale_weights;
        return *this;
    }
    PyEmbedding &operator=(PyEmbedding &&other) {
        md = std::move(other.md);
        engine = std::move(other.engine);
        dim_model = other.dim_model;
        vocab_size = other.vocab_size;
        scale_weights = other.scale_weights;
        return *this;
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
