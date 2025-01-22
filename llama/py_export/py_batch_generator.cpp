#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "py_export/bind.h"
#include "backend/batch_generator.h"
#include "py_export/py_model_base.h"

namespace bind {

using namespace batch_generator;
using model::DynBatchConfig;

// class __attribute__ ((visibility("hidden"))) PySearchTask {
class PySearchTask {
public:
    SearchTask task_;
    generator::SearchResults results;
    bool bee_answer_multi_span;

    // static std::shared_ptr<PySearchTask> create(py::object input_tokens_or_str) {
    //     std::cout << ">>>>>>>>>>>>>>> PySearchTask create" << std::endl;
    // }

    static std::shared_ptr<PySearchTask> create(py::object input_tokens_or_str,
                                                int beam_size,
                                                int max_length,
                                                float presence_penalty,
                                                float repetition_penalty,
                                                float ngram_penalty,
                                                bool diverse,
                                                int seed,
                                                float temperature,
                                                int num_results,
                                                float top_p,
                                                int top_k,
                                                bool bee_answer_multi_span,
                                                int top_logprobs,
                                                int stream) {
        SearchTask t = std::make_shared<SearchTask_>();
        t->input_tokens = py::cast<std::vector<int>>(input_tokens_or_str);
        t->beam_size = beam_size;
        t->max_length = max_length;
        t->presence_penalty = presence_penalty;
        t->repetition_penalty = repetition_penalty;
        t->ngram_penalty = ngram_penalty;
        t->diverse = diverse;
        t->seed = seed;
        t->temperature = temperature;
        t->num_results = num_results;
        t->top_p = top_p;
        t->top_k = top_k;
        t->top_logprobs = top_logprobs;
        t->stream = stream;
        t->callback = [=](auto &t) {};
        std::shared_ptr<PySearchTask> py_task = std::make_shared<PySearchTask>();
        py_task->task_ = t;
        py_task->bee_answer_multi_span = bee_answer_multi_span;
        return py_task;
    }

    ~PySearchTask() {
        task_->canceled = true;
    }

    bool has_result() {
        return task_->res_queue.size() > 0;
    }

    py::object get_result(float timeout);

}; // namespace bind

// class __attribute__ ((visibility("hidden"))) PyBatchGenerator {
class PyBatchGenerator {
public:
    std::shared_ptr<BatchGenerator> searcher_;

    static std::shared_ptr<PyBatchGenerator> create(DynBatchConfig &config, PyModelBase *py_model) {
        // std::cerr << "py_model->model()->layer_type() " << py_model->model()->layer_type() << "\n";
        std::cout << ">>>>>>>>>>>>>>>> PyBatchGenerator create " << std::endl;
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

// for Stream API
// update_flag: 1: Incremental; 2: update all; 3: final result
// return py::tuple(update_flag, out_tokens, score)
// if update_flag==3 (final), then
//   out_tokens is list. i.e. len(out_tokens) = num_results
// else
//  out_tokens: CpmBee: str; LLaMA: list[int]
py::object PySearchTask::get_result(float timeout) {
    generator::SearchResults results;
    {
        py::gil_scoped_release release;
        // results = std::move(pop_res(timeout));
    }
    std::cout << "PySearchTask get_result" << std::endl;
    // TODO ...
}

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

    py::class_<PySearchTask, std::shared_ptr<PySearchTask>>(handle, "SearchTask")
        .def("has_result", &PySearchTask::has_result)
        .def("get_result", &PySearchTask::get_result)
        .def(py::init(&PySearchTask::create));

    py::class_<PyBatchGenerator, std::shared_ptr<PyBatchGenerator>>(handle, "BatchGenerator")
        .def(py::init(&PyBatchGenerator::create))
        .def("run", &PyBatchGenerator::run);
}

} // namespace bind
