#include <iostream>
#include <bmengine/core/core.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "backend/embedding.h"
#include "backend/model_utils.h"

namespace py = pybind11;

namespace bind {

void load_state_dict(
    bmengine::core::Context &ctx,
    const std::map<std::string, py::array> &state_dict,
    std::map<const std::string, bmengine::core::Tensor *> named_params,
    bool parallel = false);

void load_state_dict(
    bmengine::core::Context &ctx,
    const std::map<std::string, py::array> &state_dict,
    std::map<const std::string, bmengine::core::Tensor *> named_params,
    bool parallen) {
    for (auto it : named_params) {
        BM_ASSERT(it.second, it.first + std::string(" not in named_params"));
        auto p = state_dict.find(it.first);
        if (p != state_dict.end()) {
            if (!parallen || ctx.rank() == 0) {
                auto buf = p->second.request();
                BM_ASSERT(
                    it.second->ndim() == buf.ndim,
                    it.first + " ndim miss match: " + std::to_string(it.second->ndim())
                        + " != " + std::to_string(buf.ndim));
                for (int i = 0; i < it.second->ndim(); ++i) {
                    std::stringstream ss;
                    ss << "model[" << i << "]=" << it.second->shape()[i] << ", state[" << i
                       << "]=" << buf.shape[i];
                    // std::cout << ss.str() + "=>tianx" << std::endl;
                    BM_ASSERT(
                        it.second->shape()[i] == buf.shape[i],
                        "Parameter `" + it.first + "` has different shape" + ss.str());
                }
                BM_ASSERT(
                    it.second->nbytes() == (buf.size * buf.itemsize),
                    it.first + " size miss match: " + std::to_string(it.second->nbytes())
                        + " != " + std::to_string(buf.size * buf.itemsize));
                ctx.init_parameter(it.first, it.second);
                it.second->from_buffer(buf.ptr);
            } else {
                it.second->from_buffer(nullptr);
            }
        } else {
            std::stringstream ss;
            ss << "state_dict missing: " << it.first;
            throw std::runtime_error(ss.str());
        }
    }
}
} // namespace bind

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

    void load_state_dict(const std::map<std::string, py::array> &state_dict) {
        auto ctx = engine->create_context({0});
        {
            auto d = ctx.with_device(0);
            auto named_params = md->named_parameters("token_embedding", true);
            bind::load_state_dict(ctx, state_dict, named_params);
        }
    }

    std::map<const std::string, py::array_t<float>> named_parameters() {
        std::map<const std::string, py::array_t<float>> result;

        auto ctx = engine->create_context({0});
        {
            auto d = ctx.with_device(0);
            auto named_params = md->named_parameters("token_embedding", true);
            for (auto it : named_params) {
                py::array_t<float> ndarray(it.second->size());
                auto converted = model::convert_fp32(ctx, *it.second);
                converted.to_buffer(ndarray.mutable_data());
                result.emplace(it.first, ndarray);
            }
            return result;
        }
    }

    py::array forward(py::array &ids, py::array &ids_sub) {
        auto ctx = engine->create_context({0});
        {
            auto d = ctx.with_device(0);

            auto ids_buf = ids.request();
            std::vector<size_t> ids_size;
            for (int i = 0; i < ids.ndim(); ++i) {
                ids_size.push_back(ids_buf.shape[i]);
            }
            auto t_ids = ctx.tensor(ids_size, bmengine::core::DataType::kInt32);
            t_ids.from_buffer(ids_buf.ptr);

            auto subs_buf = ids_sub.request();
            std::vector<size_t> subs_size;
            for (int i = 0; i < ids_sub.ndim(); ++i) {
                subs_size.push_back(subs_buf.shape[i]);
            }
            auto t_subs = ctx.tensor(subs_size, bmengine::core::DataType::kInt32);
            t_subs.from_buffer(subs_buf.ptr);

            auto out_data = md->forward(ctx, t_ids, t_subs);
            py::array_t<float> ndarray(out_data.size());
            auto converted = model::convert_fp32(ctx, out_data);
            converted.to_buffer(ndarray.mutable_data());
            return ndarray;
        }
    }
}; // end of class PyEmbedding

void define_layer_embedding(py::module_ &layers_m) {
    py::class_<PyEmbedding>(layers_m, "Embedding")
        .def(py::init(&PyEmbedding::create))
        .def("load_state_dict", &PyEmbedding::load_state_dict)
        .def("named_parameters", &PyEmbedding::named_parameters)
        .def("forward", &PyEmbedding::forward);
}

PYBIND11_MODULE(llm_nodes, handle) {
    handle.doc() = "This is embedding llm_nodes.";
    py::module_ layers_m = handle.def_submodule("layers", "internal layers for testing.");
    define_layer_embedding(layers_m);
}
