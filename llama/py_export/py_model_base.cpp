#include <iostream>
#include <vector>
#include <pybind11/pybind11.h>
#include <bmengine/core/core.h>
#include "backend/model_config.h"
#include "py_export/py_utils.h"

namespace py = pybind11;
using model::ModelConfig;

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

void define_model_config(py::module_ &handle) {
    py::class_<model::ModelConfig>(handle, "ModelConfig")
        .def(py::init(&create_config))
        .def(py::init(&create_config_from_dict));
}

} // namespace bind
