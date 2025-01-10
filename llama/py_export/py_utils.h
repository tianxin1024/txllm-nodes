#pragma once

#include <stddef.h>
#include <stdint.h>
#include <bmengine/core/core.h>
#include <pybind11/numpy.h>
#include <map>
#include "backend/model.h"

namespace bind {

namespace py = pybind11;

model::ModelConfig pydict_to_model_config(py::dict &cfg);

} // namespace bind
