#include "py_export/bind.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(llm_nodes, handle) {
    bind::define_model_config(handle);
    bind::define_quant_config(handle);
    bind::define_engine(handle);

    // models
    bind::define_cpm_base(handle);
}
