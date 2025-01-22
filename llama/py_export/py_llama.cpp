#include "backend/nn.h"
#include "py_export/bind.h"
#include "py_export/py_utils.h"
#include "py_export/py_model_base.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <bmengine/core/core.h>
#include "backend/model.h"
#include "backend/llama.h"

namespace py = pybind11;

using bmengine::core::Tensor;
using bmengine::core::DataType;
typedef std::shared_ptr<bmengine::core::Engine> EnginePtr;

class PyLLaMA : public PyModelBase {
private:
    EnginePtr engine_;
    model::ModelConfig model_config_;
    std::vector<model::ModelBase *> models_;

public:
    PyLLaMA(EnginePtr engine,
            model::ModelConfig model_config,
            model::QuantConfig quant_config,
            bool parallel) :
        PyModelBase("llama", parallel),
        engine_(engine), model_config_(model_config) {
        std::cout << model_config.to_string() << std::endl;
        if (!parallel && engine->num_gpus() > 1) {
            throw std::runtime_error("WARNING: Use parallel=false with multiple GPU !!!");
        }
        models_.resize(engine->world_size());
        engine->device_foreach([this, &model_config, quant_config, parallel](int i) {
            auto ctx = engine_->create_context_rank(i);
            auto with_device = ctx.with_device(0);
            models_[i] = new model::LLaMA(ctx, model_config, quant_config, parallel);
        });
    }

    ~PyLLaMA() {
        if (engine_ && !models_.empty()) {
            engine_->device_foreach([this](int i) {
                delete models_[i];
            });
            models_.clear();
        }
    }

    PyLLaMA(const PyLLaMA &other) = default;
    PyLLaMA(PyLLaMA &&other) = default;
    PyLLaMA &operator=(const PyLLaMA &other) = default;
    PyLLaMA &operator=(PyLLaMA &&other) = default;

    static PyLLaMA create(EnginePtr engine,
                          model::ModelConfig &model_config,
                          model::QuantConfig quant_config,
                          bool parallel = false) {
        return PyLLaMA(engine, model_config, quant_config, parallel);
    }

    virtual bmengine::core::Engine *engine() {
        return engine_.get();
    }

    virtual std::vector<model::ModelBase *> par_models() {
        return models_;
    }

    void load_state_dict_1(const std::map<std::string, py::array> &state_dict) {
        auto tensor_dict = bind::numpy_to_tensor(state_dict);
        engine_->device_foreach([this, &tensor_dict](int i) {
            auto ctx = engine_->create_context_rank(i);
            auto with_device = ctx.with_device(0);
            // load params recursively
            models_[i]->load_state_dict(ctx, tensor_dict, prefix);
        });
        on_load();
    }

}; // end of class PyLLaMA

namespace bind {
void define_llama(py::module_ &handle) {
    py::class_<PyLLaMA, PyModelBase>(handle, "LLaMA")
        .def(py::init(&PyLLaMA::create))
        .def("load_state_dict_1", &PyLLaMA::load_state_dict_1);
}

} // namespace bind
