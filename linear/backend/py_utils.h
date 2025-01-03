#pragma once

#include <stddef.h>
#include <stdint.h>
#include <map>
#include <bmengine/core/core.h>
#include <pybind11/pybind11.h>
#include <ATen/ATen.h>
#include <ATen/InitialTensorOptions.h>

namespace bind {

using namespace bmengine;

namespace py = pybind11;

using bmengine::core::Context;
using bmengine::core::DataType;

void load_at_state_dict(
    Context &ctx,
    const std::map<std::string, core::Tensor> &state_dict,
    std::map<const std::string, core::Tensor *> named_params,
    bool parallel = false);

const core::Tensor aten_to_tensor(const Context &ctx, const at::Tensor &at_tensor);
at::Tensor tensor_to_aten(const Context &ctx, const core::Tensor &tensor);

} // namespace bind
