#pragma once

#include <stddef.h>
#include <stdint.h>
#include <map>
#include <bmengine/core/core.h>
#include <pybind11/pybind11.h>
#include <ATen/ATen.h>

namespace bind {

namespace py = pybind11;

using bmengine::core::Context;
using bmengine::core::DataType;
using bmengine::core::Tensor;

void load_at_state_dict(
    Context &ctx,
    const std::map<std::string, at::Tensor> &state_dict,
    std::map<const std::string, Tensor *> named_params,
    bool parallel = false);

} // namespace bind
