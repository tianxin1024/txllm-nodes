#include <iostream>
#include <bmengine/core/core.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <memory>
#include "backend/linear.h"
#include "backend/utils.h"

namespace py = pybind11;
using namespace bmengine;

namespace bind {
void load_state_dict(bmengine::core::Context &ctx,
                     const std::map<std::string, py::array> &state_dict,
                     std::map<const std::string, bmengine::core::Tensor *> named_params,
                     bool parallel = false);

void load_state_dict(bmengine::core::Context &ctx,
                     const std::map<std::string, py::array> &state_dict,
                     std::map<const std::string, bmengine::core::Tensor *> named_params,
                     bool parallel) {
    for (auto it : named_params) {
        BM_ASSERT(it.second, it.first + std::string(" not in named_params"));
        auto p = state_dict.find(it.first);
        if (p != state_dict.end()) {
            if (!parallel || ctx.rank() == 0) {
                auto buf = p->second.request();
                BM_ASSERT(it.second->ndim() == buf.ndim,
                          it.first + " ndim miss match: " + std::to_string(it.second->ndim()) + " != " + std::to_string(buf.ndim));
                for (int i = 0; i < it.second->ndim(); ++i) {
                    std::stringstream ss;

                    ss << "model[" << i << "]=" << it.second->shape()[i] << ", state["
                       << i << "]=" << buf.shape[i];
                    // std::cout << ss.str() + "=>tianx" << std::endl;
                    BM_ASSERT(it.second->shape()[i] == buf.shape[i],
                              "Parameter `" + it.first + "` has different shape" + ss.str());
                }
                BM_ASSERT(it.second->nbytes() == (buf.size * buf.itemsize),
                          it.first + " size miss match: " + std::to_string(it.second->nbytes()) + " != " + std::to_string(buf.size * buf.itemsize));
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

class PyLinear {
private:
    std::shared_ptr<Linear> md;
    std::shared_ptr<bmengine::core::Engine> engine;
    int dim_model;
    int dim_ff;
    std::string act_fn_type;
    int quant;
    bool scale_weights;
    bool weight_transposed;

    PyLinear(
        int dim_model,
        int dim_ff,
        std::string act_fn_type,
        bool quant,
        bool scale_weights,
        bool weight_transposed,
        std::string &dtype_name) :
        dim_model(dim_model),
        dim_ff(dim_ff), act_fn_type(act_fn_type),
        quant(quant), scale_weights(scale_weights), weight_transposed(weight_transposed) {
        std::cout << " >>>>>> PyLinear constructor" << std::endl;
        std::vector<bmengine::core::DeviceConfiguration> devices;
        devices.emplace_back(0, (size_t)2 << 30);

        engine = std::make_shared<bmengine::core::Engine>(devices);
        auto ctx = engine->create_context({0});
        bmengine::core::WithDevice device(ctx, 0);

        md = std::move(std::make_shared<Linear>(ctx, dim_model, dim_ff, act_fn_type, scale_weights,
                                                weight_transposed));
    }

public:
    ~PyLinear() {
        md = nullptr;
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

    void load_state_dict(const std::map<std::string, py::array> &state_dict)
        __attribute__((visibility("hidden"))) {
        std::cout << "PyLinear: load_state_dict" << std::endl;
        auto ctx = engine->create_context({0});
        bmengine::core::WithDevice device(ctx, 0);
        auto named_params = md->named_parameters("", true);
        bind::load_state_dict(ctx, state_dict, named_params);
    }

    std::map<const std::string, py::array_t<float>> named_parameters()
        __attribute__((visibility("hidden"))) {
        std::map<const std::string, py::array_t<float>> result;
        {
            auto ctx = engine->create_context({0});
            bmengine::core::WithDevice device(ctx, 0);
            auto named_params = md->named_parameters("", true);

            for (auto it : named_params) {
                py::array_t<float> ndarray(it.second->size());
                auto converted = model::convert_fp32(ctx, *it.second);
                converted.to_buffer(ndarray.mutable_data());
                result.emplace(it.first, ndarray);
            }
            return result;
        }
    }

}; // class end PyLinear

void define_layer_linear(py::module_ &layers_m) {
    py::class_<PyLinear>(layers_m, "Linear")
        .def(py::init(&PyLinear::create))
        .def("load_state_dict", &PyLinear::load_state_dict)
        .def("named_parameters", &PyLinear::named_parameters);
}

PYBIND11_MODULE(llm_nodes, handle) {
    py::module_ layers_m = handle.def_submodule("layers", "internal layers for testing.");

    define_layer_linear(layers_m);
}
