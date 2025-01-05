#include "backend/linear.h"
#include <bmengine/functions/gemm.h>

using bmengine::core::DataType;
using bmengine::core::DistLayout;
using bmengine::core::Tensor;

class Linear::impl {
public:
    class NormalLinear;
    class Int8Linear;
    class Fp8Linear;
    class Int4GPTQ;
    class GPTQMarlin;
    class AWQ;

    uint32_t dim_in;
    uint32_t dim_out;
    core::DistLayout dist_layout;
    std::string act_fn_type;
    bool weight_transposed;
    int quant;
    core::DataType dtype;
    bool has_bias{false};
    std::string prefix;

    impl(uint32_t dim_in, uint32_t dim_out, std::string act_fn, bool w_trans, int quant, DataType dtype) :
        dim_in(dim_in), dim_out(dim_out), act_fn_type(act_fn), weight_transposed(w_trans), quant(quant), dtype(dtype) {
    }
    virtual ~impl() = default;

    // virtual void scale_output(float scale) = 0;
    // virtual void set_output_type(core::DataType dtype) = 0;
    virtual void set_compute_type(cublasComputeType_t compute_type) {
    }

    // virtual core::Tensor forward(
    //     const core::Context &ctx,
    //     const core::Tensor &input,
    //     const std::string &output_name,
    //     bool quant_back,
    //     Tensor *output) = 0;

    virtual core::Tensor &get_weight() = 0;
    virtual core::Tensor get_dequant_weight(const core::Context &ctx) {
        throw std::runtime_error("not supported");
    };
    virtual core::Tensor *get_weight_scale() {
        return nullptr;
    }

    virtual void load_parameters(const core::Context &ctx, const std::string &prefix) {
        throw std::runtime_error("load_parameters for QuantImpl only");
    }

    virtual void load_state_dict(
        const core::Context &ctx,
        const std::map<std::string, const core::Tensor> &state_dict,
        const std::string &prefix,
        bool allow_missing) = 0;

    Tensor activate(const core::Context &ctx, const Tensor &ret) {
        // BM_ASSERT(act_fn_type.empty(), "");
        if (!act_fn_type.empty()) {
            ctx.recordEvent(act_fn_type, 2);
        }
        if (act_fn_type == "gelu") {
            // gelu_inplace(ret, ctx.current_stream()->ptr);
        } else if (act_fn_type == "silu") {
            // silu_inplace(ret, ctx.current_stream()->ptr);
        } else if (act_fn_type != "") {
            throw std::runtime_error(act_fn_type + " activation is not supported");
        }
        return ret;
    }

    virtual void set_has_bias(bool b) {
        if (b) throw std::runtime_error("Bias is not implemented");
    }
    // Tensor add_bias(const core::Context& ctx, const Tensor& t, const Tensor& bias) {
    //     using BinaryOp = bmengine::functions::BinaryElementwiseOp;
    //     BinaryOp add_op(ctx, BinaryOp::Add);
    //     return add_op.broadcast_y(ctx, t, bias);
    // }

    // bool is_attn_proj() {
    //     return prefix.find("attn.project_") != string::npos;
    // }
    // bool is_ff_in() {
    //     return prefix.find(".w_in") != string::npos || prefix.find(".w_gated") != string::npos;
    // }
    // bool is_ff_out() {
    //     return prefix.find(".w_out") != string::npos;
    // }
};

// class Linear::impl {
// public:
//     class NormalLinear;

//     uint32_t dim_in;
//     uint32_t dim_out;
//     core::DistLayout dist_layout;
//     std::string act_fn_type;
//     bool weight_transposed;
//     int quant;
//     core::DataType dtype;
//     bool has_bias{false};
//     std::string prefix;

//     impl(uint32_t dim_in, uint32_t dim_out, std::string act_fn, bool w_trans, int quant, DataType dtype) :
//         dim_in(dim_in), dim_out(dim_out), act_fn_type(act_fn), weight_transposed(w_trans), quant(quant), dtype(dtype) {
//     }

//     virtual ~impl() = default;
// };

// class Linear::impl::NormalLinear : public Linear::impl {
// public:
//     bool parallel;
//     core::DistLayout dist_layout;
//     float scale_factor;
//     std::unique_ptr<Tensor> weight;
//     Tensor bias;

//     NormalLinear(const core::Context &ctx,
//                  uint32_t dim_in,
//                  uint32_t dim_out,
//                  std::string act_fn_type,
//                  bool scale_weights,
//                  bool weight_transposed,
//                  core::DataType dtype,
//                  bool parallel,
//                  core::DistLayout dist_layout) :
//         Linear::impl(dim_in, dim_out, act_fn_type, weight_transposed, 0, dtype),
//         parallel(parallel),
//         dist_layout(weight_transposed ? dist_layout : transpose_layout(dist_layout)),
//         scale_factor(float(scale_weights ? 1.0 / sqrtf(dim_in) : 1.0)) {
//         std::vector<size_t> shape({
//             weight_transposed ? dim_in : dim_out, // W^T
//             weight_transposed ? dim_out : dim_in, // W
//         });
//     }

//     ~NormalLinear() = default;
// };

// =========================== normal linear ===========================
class Linear::impl::NormalLinear : public Linear::impl {
public:
    bool parallel;
    core::DistLayout dist_layout;
    float scale_factor;
    std::unique_ptr<Tensor> weight;
    Tensor bias;
    functions::Gemm gemm_A_B;
    functions::Gemm gemm_A_Btrans;

    NormalLinear(
        const core::Context &ctx,
        uint32_t dim_in,
        uint32_t dim_out,
        std::string act_fn_type,
        bool scale_weights,
        bool weight_transposed,
        core::DataType dtype,
        bool parallel,
        core::DistLayout dist_layout) :
        Linear::impl(dim_in, dim_out, act_fn_type, weight_transposed, 0, dtype),
        parallel(parallel),
        dist_layout(weight_transposed ? dist_layout : transpose_layout(dist_layout)),
        scale_factor(float(scale_weights ? 1.0 / sqrtf(dim_in) : 1.0)),
        gemm_A_B(ctx, dtype, false, false, scale_factor),
        gemm_A_Btrans(ctx, dtype, false, true, scale_factor) {
        std::vector<size_t> shape({
            weight_transposed ? dim_in : dim_out, // W^T
            weight_transposed ? dim_out : dim_in  // W
        });
        weight = std::make_unique<Tensor>(ctx.parameter(shape, dtype));
        if (ctx.high_precision() >= 1) {
            gemm_A_B.set_compute_type(CUBLAS_COMPUTE_32F);
            gemm_A_Btrans.set_compute_type(CUBLAS_COMPUTE_32F);
        }
    }

    ~NormalLinear() = default;

    void set_has_bias(bool b) override {
        has_bias = b;
    }

    void load_state_dict(
        const core::Context &ctx,
        const std::map<std::string, const core::Tensor> &state_dict,
        const std::string &prefix,
        bool allow_missing) override {
        std::vector<size_t> shape({
            weight_transposed ? dim_in : dim_out, // W^T
            weight_transposed ? dim_out : dim_in  // W
        });
        weight = std::make_unique<Tensor>(ctx.parameter(shape, dtype));
        auto name = prefix + ".weight";
        ctx.load_parameter(weight.get(), name, state_dict, parallel, dist_layout);

        auto bias_layout = dist_layout == DistLayout::ROW ? DistLayout::COLUMNAR : DistLayout::REPLICATED;
        if (has_bias) {
            name = prefix + ".bias";
            bias = ctx.parameter({dim_out}, dtype);
            ctx.load_parameter(&bias, name, state_dict, parallel, bias_layout);
        }
    }

    core::Tensor &get_weight() override {
        return *weight;
    }
    core::Tensor get_dequant_weight(const core::Context &ctx) override {
        return *weight;
    }

    // core::Tensor forward(
    //     const core::Context &ctx,
    //     const core::Tensor &input,
    //     const std::string &output_name,
    //     bool quant_back,
    //     Tensor *output) override {
    //     /*
    //     Input: (seq_len, dim_in)
    //     Output: (seq_len, dim_out)
    //     */
    //     BM_ASSERT(input.ndim() == 2 || input.ndim() == 3, "Input must be 2D/3D");
    //     BM_ASSERT_EQ(input.dtype(), weight->dtype(), "Input data type mismatch");
    //     BM_ASSERT_EQ(input.device(), weight->device(), "Input and weight must be on the same device");

    //     core::Tensor ret; // (seq_len, dim_out)
    //     // x @ W^T
    //     if (!weight_transposed) {
    //         ret = gemm_A_Btrans.forward(ctx, input, *weight, output, has_bias ? &bias : nullptr);
    //         //            if (has_bias) {
    //         //                ret = add_bias(ctx, ret, bias);
    //         //            }
    //     } else {
    //         ret = gemm_A_B.forward(ctx, input, *weight);
    //     }

    //     // set name here to avoid memory allocation.
    //     ret.set_name(output_name);
    //     return activate(ctx, ret);
    // }
};

Linear::Linear(const core::Context &ctx,
               int dim_in,
               int dim_out,
               std::string act_fn_type,
               model::QuantConfig quant_config,
               bool scale_weights,
               bool weight_transposed,
               bool parallel,
               core::DistLayout dist_layout,
               core::DataType dtype) :
    Layer() {
    auto tmp = new impl::NormalLinear(
        ctx, dim_in, dim_out, act_fn_type, scale_weights, weight_transposed, dtype, parallel, dist_layout);
    std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>>> Linear::Linear " << std::endl;
    std::cout << "dim_in: " << dim_in << " dim_out: " << dim_out << std::endl;
    add_parameter("weight", *tmp->weight);
    pimpl = std::unique_ptr<impl>((impl *)tmp);

    pimpl->dist_layout = dist_layout;
}

Linear::Linear(const core::Context &ctx,
               int dim_in,
               int dim_out,
               model::QuantConfig quant_config,
               core::DistLayout dist_layout,
               core::DataType dtype) :
    Linear(ctx, dim_in, dim_out, "", quant_config, false, false, ctx.world_size() > 1, dist_layout, dtype) {
}

Linear::Linear(const core::Context &ctx,
               const std::string &name,
               const core::Tensor &w) :
    Linear(ctx, w.size(1), w.size(0), "", 0, false, false, false, DistLayout::REPLICATED, w.dtype()) {
    BM_ASSERT_EQ(w.ndim(), 2, "");
    this->name = name;
    auto ptr = dynamic_cast<impl::NormalLinear *>(pimpl.get());
    BM_ASSERT(ptr, "Not NormalLinear");
    *ptr->weight = w;
}

void Linear::move(Linear &other) {
    pimpl = std::move(other.pimpl);
}

Linear::~Linear() = default;
