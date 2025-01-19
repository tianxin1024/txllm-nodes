#include <pybind11/pybind11.h>
#include "py_export/bind.h"
#include "backend/batch_generator.h"
#include "py_export/py_model_base.h"

namespace bind {

using namespace batch_generator;
using model::DynBatchConfig;

// class __attribute__ ((visibility("hidden"))) PyBatchGenerator {
class PyBatchGenerator {
public:
    std::shared_ptr<BatchGenerator> searcher_;

    static std::shared_ptr<PyBatchGenerator> create(DynBatchConfig &config, PyModelBase *py_model) {
        // std::cerr << "py_model->model()->layer_type() " << py_model->model()->layer_type() << "\n";
        std::shared_ptr<PyBatchGenerator> self = std::make_shared<PyBatchGenerator>();
        self->searcher_ = std::make_shared<BatchGenerator>(
            config, py_model->par_models(), py_model->engine());
        return self;
    }

    ~PyBatchGenerator() {
        if (searcher_) {
            std::cerr << "~PyBatchGenerator" << std::endl;
            searcher_->stop();
            searcher_.reset();
        }
    }

    void run() {
        py::gil_scoped_release release;
        searcher_->run();
    }

    void stop() {
        searcher_->stop();
    }
}; // end of class PyBatchGenerator

void define_dynamic_batch(py::module_ &handle) {
    py::class_<PyBatchGenerator, std::shared_ptr<PyBatchGenerator>>(handle, "BatchGenerator")
        .def(py::init(&PyBatchGenerator::create));
}

} // namespace bind
