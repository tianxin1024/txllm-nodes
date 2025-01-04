#include <iostream>
#include <bmengine/core/core.h>
#include <pybind11/pybind11.h>
#include "backend/feedforward.h"
#include <thread>
#include <csignal>

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
    };

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
