#include <bmengine/core/core.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "backend/model_config.h"
#include "backend/attention.h"
#include "backend/utils.h"
#include "backend/transformer_buffer.h"

using namespace bmengine;
using namespace kvcache;
namespace py = pybind11;

namespace bind {

const core::Tensor numpy_to_tensor(const core::Context &ctx, const py::array &arr);
void load_state_dict(bmengine::core::Context &ctx,
                     const std::map<std::string, py::array> &state_dict,
                     std::map<const std::string, bmengine::core::Tensor *> named_params,
                     bool parallel = false);

const core::Tensor numpy_to_tensor(const core::Context &ctx, const py::array &arr) {
    py::buffer_info buf = arr.request();
    if (buf.size == 0) {
        return std::move(core::Tensor());
    }

    std::vector<size_t> size;
    for (int d = 0; d < buf.ndim; ++d) {
        size.push_back(buf.shape[d]);
    }
    auto ret = ctx.tensor(size, bmengine::core::DataType::kHalf);
    ret.from_buffer(buf.ptr);
    return std::move(ret);
}

void load_state_dict(bmengine::core::Context &ctx,
                     const std::map<std::string, py::array> &state_dict,
                     std::map<const std::string, bmengine::core::Tensor *> named_params,
                     bool parallel) {
    for (auto it : named_params) {
        BM_ASSERT(it.second, it.first + std::string(" not in named_params"));
        auto p = state_dict.find(it.first);
        if (p != state_dict.end()) {
            if (!parallel || ctx.rank() == 0) {
                auto buf = p->second.request();
                BM_ASSERT(it.second->ndim() == buf.ndim,
                          it.first + " ndim miss match: " + std::to_string(it.second->ndim()) + " != " + std::to_string(buf.ndim));
                for (int i = 0; i < it.second->ndim(); ++i) {
                    std::stringstream ss;

                    ss << "model[" << i << "]=" << it.second->shape()[i] << ", state["
                       << i << "]=" << buf.shape[i];
                    // std::cout << ss.str() + "=>tianx" << std::endl;
                    BM_ASSERT(it.second->shape()[i] == buf.shape[i],
                              "Parameter `" + it.first + "` has different shape" + ss.str());
                }
                BM_ASSERT(it.second->nbytes() == (buf.size * buf.itemsize),
                          it.first + " size miss match: " + std::to_string(it.second->nbytes()) + " != " + std::to_string(buf.size * buf.itemsize));
                ctx.init_parameter(it.first, it.second);
                it.second->from_buffer(buf.ptr);
            } else {
                it.second->from_buffer(nullptr);
            }
        } else {
            std::stringstream ss;
            ss << "state_dict missing: " << it.first;
            throw std::runtime_error(ss.str());
        }
    }
}
} // namespace bind

class PyAttention {
private:
    std::shared_ptr<Attention> md;
    std::shared_ptr<bmengine::core::Engine> engine;
    int dim_model;
    int dim_head;
    int num_heads;
    std::string pos_bias_type;
    int quant;
    bool scale_weights;
    bool weight_transposed;
    bool parallel;

    // auto attn = PyAttention(dim_model, num_heads, dim_head, pos_bias_type, quant, scale_weights,
    //                         weight_transposed, parallel);
    PyAttention(int dim_model,
                int num_heads,
                int dim_head,
                std::string pos_bias_type,
                int quant,
                bool scale_weights,
                bool weight_transposed,
                bool parallel) :
        dim_model(dim_model),
        dim_head(dim_head), num_heads(num_heads), pos_bias_type(pos_bias_type), quant(quant),
        scale_weights(scale_weights), weight_transposed(weight_transposed), parallel(parallel) {
        std::vector<bmengine::core::DeviceConfiguration> devices;
        devices.emplace_back(0, (size_t)2 << 30);

        model::ModelConfig model_config("", 0, dim_model, num_heads, dim_head, 0, 0, 1e-6,
                                        -1, {}, scale_weights, weight_transposed, 0, 1.0, 1.0,
                                        bmengine::core::DataType::kHalf);

        engine = std::make_shared<bmengine::core::Engine>(devices);
        auto ctx = engine->create_context({0});
        bmengine::core::WithDevice device(ctx, 0);

        md = std::move(std::make_shared<Attention>(ctx, model_config, 0, parallel));
    }

public:
    ~PyAttention() {
        md = nullptr;
    }
    PyAttention(const PyAttention &other) {
        md = other.md;
        engine = other.engine;
        dim_model = other.dim_model;
        dim_head = other.dim_head;
        num_heads = other.num_heads;
        pos_bias_type = other.pos_bias_type;
        quant = other.quant;
        scale_weights = other.scale_weights;
        weight_transposed = other.weight_transposed;
        parallel = other.parallel;
    }

    PyAttention(PyAttention &&other) {
        md = std::move(other.md);
        engine = std::move(other.engine);
        dim_model = other.dim_model;
        dim_head = other.dim_head;
        num_heads = other.num_heads;
        pos_bias_type = other.pos_bias_type;
        quant = other.quant;
        scale_weights = other.scale_weights;
        weight_transposed = other.weight_transposed;
        parallel = other.parallel;
    }

    PyAttention &operator=(const PyAttention &other) {
        md = other.md;
        engine = other.engine;
        dim_model = other.dim_model;
        dim_head = other.dim_head;
        num_heads = other.num_heads;
        pos_bias_type = other.pos_bias_type;
        quant = other.quant;
        scale_weights = other.scale_weights;
        weight_transposed = other.weight_transposed;
        parallel = other.parallel;
        return *this;
    }

    PyAttention &operator=(PyAttention &&other) {
        md = std::move(other.md);
        engine = std::move(other.engine);
        dim_model = other.dim_model;
        dim_head = other.dim_head;
        num_heads = other.num_heads;
        pos_bias_type = other.pos_bias_type;
        quant = other.quant;
        scale_weights = other.scale_weights;
        weight_transposed = other.weight_transposed;
        parallel = other.parallel;
        return *this;
    }

    static PyAttention create(int dim_model, int num_heads, int dim_head, std::string pos_bias_type,
                              int quant, bool scale_weights = false, bool weight_transposed = true,
                              bool parallel = false) {
        model::ModelConfig model_config("", 0, dim_model, num_heads, dim_head, 0, 0, 1e-6,
                                        -1, {}, scale_weights, weight_transposed, 0, 1.0, 1.0,
                                        bmengine::core::DataType::kHalf);
        auto attn = PyAttention(dim_model, num_heads, dim_head, pos_bias_type, quant, scale_weights,
                                weight_transposed, parallel);
        return attn;
    }

    void load_state_dict(const std::map<std::string, py::array> &state_dict)
        __attribute__((visibility("default"))) {
        auto ctx = engine->create_context({0});
        auto named_params = md->named_parameters("", true);
        bmengine::core::WithDevice device(ctx, 0);
        bind::load_state_dict(ctx, state_dict, named_params);
    }

    std::map<const std::string, py::array_t<float>> named_parameters()
        __attribute__((visibility("hidden"))) {
        std::map<const std::string, py::array_t<float>> result;

        {
            auto ctx = engine->create_context({0});
            bmengine::core::WithDevice device(ctx, 0);
            auto named_params = md->named_parameters("", true);

            for (auto it : named_params) {
                py::array_t<float> ndarray(it.second->size());
                auto converted = model::convert_fp32(ctx, *it.second);
                converted.to_buffer(ndarray.mutable_data());
                result.emplace(it.first, ndarray);
            }
            return result;
        }
    }

    py::array forward(py::array &input, py::array &mask, py::array &position,
                      py::array &seqlens_q, py::array &seqlens_kv) __attribute__((visibility("hidden"))) {
        py::array_t<float> output; // out
        auto buf = input.request();
        output.resize(buf.shape);

        auto ctx = engine->create_context({0});
        bmengine::core::WithDevice device(ctx, 0);

        auto t_input = bind::numpy_to_tensor(ctx, input);
        auto t_mask = bind::numpy_to_tensor(ctx, mask);
        auto t_position = bind::numpy_to_tensor(ctx, position);
        auto t_seqlens_q = bind::numpy_to_tensor(ctx, seqlens_q);
        auto t_seqlens_kv = bind::numpy_to_tensor(ctx, seqlens_kv);

        // std::cout << ">>>>>>>>>>>> t_input: >>>>>>>>>>>>>>>>>>>>>>>>>>" << std::endl;
        // std::cout << t_input << std::endl;

        model::ModelConfig model_config("", 0, dim_model, num_heads, dim_head, 0, 0, 1e-6,
                                        -1, {}, scale_weights, weight_transposed, 0, 1.0, 1.0,
                                        bmengine::core::DataType::kHalf);

        std::cout << "batch size: " << t_input.size(0) << std::endl;

        // int len_buf = round_up(t_input.size(1), 32);
        int len_buf = t_mask.size(-1);
        TransformerBuffer buf_k(t_input.size(0), 1, model_config.num_heads, model_config.dim_head,
                                bmengine::core::DataType::kHalf, true, t_seqlens_kv.numel() != 0);
        TransformerBuffer buf_v(t_input.size(0), 1, model_config.num_heads, model_config.dim_head,
                                bmengine::core::DataType::kHalf, true, t_seqlens_kv.numel() != 0);
        buf_k.resize(ctx, len_buf);
        std::cout << ">>>>>>>>>>>> forward >>>>>>>>>>>>>>>>>>>>>>>>>>" << std::endl;
        buf_v.resize(ctx, len_buf);
        // auto res = md->forward(ctx, t_input, t_mask, t_position, t_seqlens_q, &(buf_k[0]),
        //                        &(buf_v[0]), nullptr, nullptr, nullptr);
    }
};

void define_layer_linear(py::module_ &layers_m) {
    py::class_<PyAttention>(layers_m, "Attention")
        .def(py::init(&PyAttention::create))
        .def("load_state_dict", &PyAttention::load_state_dict)
        .def("named_parameters", &PyAttention::named_parameters)
        .def("forward", &PyAttention::forward);
}

PYBIND11_MODULE(llm_nodes, handle) {
    py::module_ layers_m = handle.def_submodule("layers", "internal layers for testing.");

    define_layer_linear(layers_m);
}
