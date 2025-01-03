#pragma once

#include <stddef.h>
#include <stdint.h>
#include <map>
#include <bmengine/core/core.h>
#include <pybind11/pybind11.h>
#include <ATen/ATen.h>
// #include <torch/torch.h>

namespace bind {

namespace py = pybind11;

using bmengine::core::Context;
using bmengine::core::DataType;
using bmengine::core::Tensor;

void load_at_state_dict(
    Context &ctx,
    const std::map<std::string, Tensor> &state_dict,
    std::map<const std::string, Tensor *> named_params,
    bool parallel = false);

at::Tensor tensor_to_aten(const Context &ctx, const Tensor &tensor);

} // namespace bind
