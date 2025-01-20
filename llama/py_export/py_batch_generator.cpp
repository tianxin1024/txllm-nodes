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
    py::class_<DynBatchConfig>(handle, "DynBatchConfig")
        .def_readwrite("max_batch", &DynBatchConfig::max_batch)
        .def_readwrite("max_beam_size", &DynBatchConfig::max_beam_size)
        .def_readwrite("task_queue_size", &DynBatchConfig::task_queue_size)
        .def_readwrite("max_total_token", &DynBatchConfig::max_total_token)
        .def_readwrite("seed", &DynBatchConfig::seed)
        .def_readwrite("eos_id", &DynBatchConfig::eos_id)
        .def_readwrite("bos_id", &DynBatchConfig::bos_id)
        .def_readwrite("unk_id", &DynBatchConfig::unk_id)
        .def_readwrite("first_batch", &DynBatchConfig::first_batch)
        .def_readwrite("nccl", &DynBatchConfig::nccl)
        .def_readwrite("rag_buffer", &DynBatchConfig::rag_buffer)
        .def_readwrite("ignore_eos", &DynBatchConfig::ignore_eos)
        .def_readwrite("keep_eos", &DynBatchConfig::keep_eos)
        .def_readwrite("reserved_work_mem_mb", &DynBatchConfig::reserved_work_mem_mb)
        .def_readwrite("high_precision", &DynBatchConfig::high_precision)
        .def_readwrite("flash_attention", &DynBatchConfig::flash_attention)
        .def_readwrite("enable_prompt_caching", &DynBatchConfig::enable_prompt_caching)
        .def(py::init());

    py::class_<PyBatchGenerator, std::shared_ptr<PyBatchGenerator>>(handle, "BatchGenerator")
        .def(py::init(&PyBatchGenerator::create));
}

} // namespace bind
