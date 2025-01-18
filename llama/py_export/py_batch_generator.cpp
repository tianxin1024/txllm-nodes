#include <pybind11/pybind11.h>
#include "py_export/bind.h"

namespace bind {

// class __attribute__ ((visibility("hidden"))) PyBatchGenerator {
class PyBatchGenerator {
public:
    std::shared_ptr<BatchGenerator> searcher_;
};

void define_dynamic_batch(py::module_ &handle) {
    py::class_<PyBatchGenerator, std::shared_ptr<PyBatchGenerator>>(handle, "BatchGenerator")
        .def(py::init(&PyBatchGenerator::create));
}

} // namespace bind
