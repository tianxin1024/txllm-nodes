#pragma once
#include <stddef.h>
#include <stdint.h>
#include <bmengine/core/core.h>
#include <pybind11/pybind11.h>

namespace bind {

namespace py = pybind11;

void define_model_config(py::module_ &handle);
void define_quant_config(py::module_ &handle);

void define_engine(py::module_ &handle);

// model
void define_cpm_base(py::module_ &handle);
void define_llama(py::module_ &handle);

void define_dynamic_batch(py::module_ &handle);

std::vector<int> to_int_vector(const py::list &data_list);

} // namespace bind
