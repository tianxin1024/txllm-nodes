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
    // std::vector<model::ModelBase *> models_;
    model::ModelBase *model_;

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
        // models_.resize(engine->world_size());
        auto ctx = engine_->create_context({0});
        bmengine::core::WithDevice device(ctx, 0);
        model_ = std::move(new model::LLaMA(ctx, model_config, quant_config, parallel));
        // engine->device_foreach([this, &model_config, quant_config, parallel](int i) {
        //     auto ctx = engine_->create_context_rank(i);
        //     auto with_device = ctx.with_device(0);
        //     models_[i] = new model::LLaMA(ctx, model_config, quant_config, parallel);
        // });
    }

    ~PyLLaMA() {
        model_ = nullptr;
        // if (engine_ && !models_.empty()) {
        //     engine_->device_foreach([this](int i) {
        //         delete models_[i];
        //     });
        //     models_.clear();
        // }
    }

    static PyLLaMA create(EnginePtr engine,
                          model::ModelConfig &model_config,
                          model::QuantConfig quant_config,
                          bool parallel = false) {
        return PyLLaMA(engine, model_config, quant_config, parallel);
    }

    virtual bmengine::core::Engine *engine() {
        return engine_.get();
    }

    void load_state_dict_1(const std::map<std::string, py::array> &state_dict) {
        auto tensor_dict = bind::numpy_to_tensor(state_dict);

        auto ctx = engine_->create_context({0});
        bmengine::core::WithDevice device(ctx, 0);
        model_->load_state_dict(ctx, tensor_dict, prefix);
        std::cout << "[py_llama] load_state_dict " << std::endl;
        // engine_->device_foreach([this, &tensor_dict](int i) {
        //     auto ctx = engine_->create_context_rank(i);
        //     auto with_device = ctx.with_device(0);
        //     // load params recursively
        //     model_->load_state_dict(ctx, tensor_dict, prefix);
        // });
        // on_load();
    }

}; // end of class PyLLaMA

namespace bind {
void define_llama(py::module_ &handle) {
    py::class_<PyLLaMA, PyModelBase>(handle, "LLaMA")
        .def(py::init(&PyLLaMA::create))
        .def("load_state_dict_1", &PyLLaMA::load_state_dict_1);
}

} // namespace bind
