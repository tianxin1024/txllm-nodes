#include <iostream>
#include <vector>
#include <pybind11/pybind11.h>
#include <bmengine/core/core.h>
#include "backend/model_config.h"
#include "py_export/py_utils.h"
#include "py_export/py_model_base.h"
#include "backend/utils.h"

namespace py = pybind11;
using model::ModelConfig;
using model::QuantConfig;
using bmengine::core::Engine;
using bmengine::core::DeviceConfiguration;

namespace bind {

model::ModelConfig create_config(
    int num_layers,
    int dim_model,
    int num_heads,
    int dim_head,
    int dim_ff,
    int vocab_size,
    float eps,
    int num_kv_heads,
    const std::vector<std::vector<bool>> &mask_modules,
    bool scale_weights,
    bool weight_transposed,
    const std::string &dtype) {
    auto data_type = bmengine::core::name_to_data_type(dtype);
    return {"cpm_bee", num_layers, dim_model, num_heads, dim_head, dim_ff, vocab_size,
            eps, num_kv_heads, mask_modules, scale_weights, weight_transposed, 0, 1.0, 1.0, data_type};
}

model::ModelConfig create_config_from_dict(py::dict &cfg) {
    return bind::pydict_to_model_config(cfg);
}

QuantConfig create_quant_config(int i_quant_type, bool quant_weight_kv, bool act_order, int group_size, bool sym) {
    QuantConfig config{};
    config.quant_type = static_cast<model::QuantType>(i_quant_type);
    config.quant_weight_kv = quant_weight_kv;
    config.act_order = act_order;
    config.group_size = group_size;
    config.sym = sym;
    return config;
}

void initialize_gemm(ModelConfig model_config, QuantConfig quant_config, int tp, int max_lenght) {
}

static std::vector<DeviceConfiguration> get_all_dev_config(size_t memory_limit) {
    std::vector<DeviceConfiguration> devices;
    int gpu_num;
    BM_CUDART_ASSERT(cudaGetDeviceCount(&gpu_num));
    for (int i = 0; i < gpu_num; ++i) {
        devices.emplace_back(i, memory_limit);
    }
    return devices;
}

std::shared_ptr<Engine> create_engine(int device_id, size_t memory_limit, int tp) {
    if (memory_limit == 0) {
        size_t free, total;
        BM_CUDART_ASSERT(cudaSetDevice(device_id < 0 ? 0 : device_id));
        BM_CUDART_ASSERT(cudaMemGetInfo(&free, &total));
        size_t def_reserve_mem = free > (36UL << 30UL) ? 1700 : 1024;
        size_t reserve_mem = utils::get_int_env("CPM_RESERVE_MEM_MB", def_reserve_mem);
        memory_limit = free - (reserve_mem << 20UL);
    }

    std::vector<DeviceConfiguration> devices;
    if (device_id == -1) {
        // use all available devices
        devices = get_all_dev_config(memory_limit);
    } else {
        devices.emplace_back(device_id, size_t(memory_limit));
    }
    return std::make_shared<Engine>(devices, tp);
}

void define_model_config(py::module_ &handle) {
    py::class_<model::ModelConfig>(handle, "ModelConfig")
        .def(py::init(&create_config))
        .def(py::init(&create_config_from_dict));
}

void define_quant_config(py::module_ &handle) {
    py::class_<QuantConfig>(handle, "QuantConfig")
        .def(py::init(&create_quant_config));
}

void define_cpm_base(py::module_ &handle) {
    py::class_<PyModelBase>(handle, "CPMBase")
        .def("initialize_gemm", &initialize_gemm);
}

void define_engine(py::module_ &handle) {
    py::class_<Engine, std::shared_ptr<Engine>>(handle, "Engine")
        .def(py::init(&create_engine));
}

} // namespace bind
