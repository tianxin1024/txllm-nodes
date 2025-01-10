#pragma once
#include <stddef.h>
#include <stdint.h>
#include <bmengine/core/core.h>
#include <pybind11/pybind11.h>

namespace bind {

namespace py = pybind11;

void define_model_config(py::module_ &handle);

} // namespace bind
