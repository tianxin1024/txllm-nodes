#include <iostream>
#include <bmengine/core/core.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "backend/feedforward.h"
#include <thread>
#include <csignal>

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
                    std::cout << ss.str() + "=>tianx" << std::endl;
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
        std::vector<bmengine::core::DeviceConfiguration> devices;
        int gpu_num;
        BM_CUDART_ASSERT(cudaGetDeviceCount(&gpu_num));
        for (int i = 0; i < gpu_num; ++i) {
            devices.emplace_back(i, (size_t)2 << 30);
        }
        engine = std::make_shared<bmengine::core::Engine>(devices);

        std::signal(SIGSEGV, [](int sig) { bmengine::print_demangled_trace(25); });
        std::signal(SIGSEGV, [](int sig) { bmengine::print_demangled_trace(25); });

        std::vector<std::thread> threads;
        mds.resize(engine->num_gpus());
        model::ModelConfig model_config{"", 0, dim_model, 0, 0, dim_ff, 0};
        model_config.activate_fn = act_fn_type;
        model_config.scale_weights = scale_weights;
        model_config.weight_transposed = weight_transposed;

        for (int i = 0; i < engine->num_gpus(); ++i) {
            threads.emplace_back(
                [this, i, model_config, quant] {
                    auto ctx = engine->create_context({i});
                    bmengine::core::WithDevice device(ctx, 0);
                    mds[i] = std::move(std::make_shared<FeedForward>(ctx, model_config, quant, true));
                });
        }
        for (auto it = threads.begin(); it != threads.end(); ++it) {
            it->join();
        }
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

    void init_parameters(int seed = 1024) {
        auto ctx = engine->create_context({0});
        {
            auto d = ctx.with_device(0);
            curandGenerator_t gen;
            CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MRG32K3A));
            CURAND_CHECK(curandSetStream(gen, ctx.current_stream()->ptr));
            CURAND_CHECK(curandSetGeneratorOffset(gen, 0));
            CURAND_CHECK(curandSetGeneratorOrdering(gen, CURAND_ORDERING_PSEUDO_BEST));
            CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, seed));
            mds[0]->init_parameters(ctx, gen);
            curandDestroyGenerator(gen);
        }
    }

    void load_state_dict(const std::map<std::string, py::array> &state_dict)
        __attribute__((visibility("hidden"))) {
        std::vector<std::thread> threads;

        for (int i = 0; i < engine->num_gpus(); ++i) {
            threads.emplace_back([this, i, &state_dict] {
                auto ctx = engine->create_context({i});
                bmengine::core::WithDevice device(ctx, 0);
                auto named_params = mds[i]->named_parameters("ff", true);
                bind::load_state_dict(ctx, state_dict, named_params);
            });
        }

        for (auto it = threads.begin(); it != threads.end(); ++it) {
            it->join();
        }
    }

    std::map<const std::string, py::array_t<float>> named_parameters()
        __attribute__((visibility("hidden"))) {
        std::map<const std::string, py::array_t<float>> result;
        std::map<const std::string, bmengine::core::Tensor> res_tensors;
        std::vector<std::thread> threads;

        for (int i = 0; i < engine->num_gpus(); ++i) {
            threads.emplace_back([this, i, &res_tensors] {
                auto ctx = engine->create_context({i});
                bmengine::core::WithDevice device(ctx, 0);

                auto named_params = mds[i]->named_parameters("ff", true);

                for (auto it : named_params) {
                    if (i == 0) {
                        auto converted = model::convert_fp32(ctx, *it.second);
                        res_tensors.emplace(it.first, std::move(converted));
                    }
                }
            });
        }

        for (auto it = threads.begin(); it != threads.end(); ++it) {
            it->join();
        }

        auto ctx = engine->create_context({0});
        bmengine::core::WithDevice device(ctx, 0);
        for (auto it : res_tensors) {
            py::array_t<float> ndarray(it.second.size());
            try {
                it.second.to_buffer(ndarray.mutable_data());
                result.emplace(it.first, std::move(ndarray));
            } catch (const BMEngineException &e) {
                std::cerr << e.what() << std::endl;
                throw std::logic_error(e.what());
            }
        }
        return result;
    }

}; // end of class PyFeedForward

void define_layer_linear(py::module_ &layers_m) {
    py::class_<PyFeedForward>(layers_m, "FeedForward")
        .def(py::init(&PyFeedForward::create))
        .def("load_state_dict", &PyFeedForward::load_state_dict)
        .def("named_parameters", &PyFeedForward::named_parameters);
}

PYBIND11_MODULE(llm_nodes, handle) {
    py::module_ layers_m = handle.def_submodule("layers", "internal layers for testing.");

    define_layer_linear(layers_m);
}
