class ModelBase : public core::Layer {
public:
    ModelConfig cfg;
    ModelBase(ModelConfig d) {
    }
    ModelBase(const ModelBase &) = delete;
    ModelBase(ModelBase &&) = delete;

}; // end of class ModelBase

class Layer {
public:
    Layer() = default;
    virtual ~Layer() = default;
    Layer(const Layer &) = delete;
    Layer(Layer &&) = delete;
    virtual void load_state_dict() {
    }
};

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
        models_.resize(engine->world_size());
        engine->device_foreach([this, &model_config, quant_config, parallel](int i) {
            auto ctx = engine_->create_context_rank(i);
            auto with_device = ctx.with_device(0);
            models_[i] = new model::LLaMA(ctx, model_config, quant_config, parallel);
        });
    }
    static PyLLaMA create(EnginePtr engine,
                          model::ModelConfig &model_config,
                          model::QuantConfig quant_config,
                          bool parallel = false) {
        return PyLLaMA(engine, model_config, quant_config, parallel);
    }
    void load_state_dict(const std::map<std::string, py::array> &state_dict) {
        auto tensor_dict = bind::numpy_to_tensor(state_dict);
        engine_->device_foreach([this, &tensor_dict](int i) {
            auto ctx = engine_->create_context_rank(i);
            auto with_device = ctx.with_device(0);
            // load params recursively
            // 这里有一个bug, models_[0] 不能跳转到core::Layer的load_state_dict
            models_[i]->load_state_dict();
        });
    }
