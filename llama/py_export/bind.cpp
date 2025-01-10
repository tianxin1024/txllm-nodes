#include "py_export/bind.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(llm_nodes, handle) {
    bind::define_model_config(handle);
}
