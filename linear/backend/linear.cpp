#include "linear.h"
#include <bmengine/core/core.h>
#include <bmengine/functions/gemm.h>
#include <bmengine/logger/std_log_op.hpp>
#include "backend/activation_kernel.h"

using bmengine::core::Tensor;
using bmengine::core::DataType;
using bmengine::core::DistLayout;

const std::string bmengine::core::Context::EMPTY_STR;

class Linear::impl {
public:
    class NormalLinear;

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

    virtual core::Tensor forward(const core::Context &ctx,
                                 const core::Tensor &input,
                                 const std::string &output_name,
                                 bool quant_back,
                                 Tensor *otuput) = 0;

    Tensor activate(const core::Context &ctx, const Tensor &ret) {
        // BM_ASSERT(act_fn_type.empty(), "");
        if (!act_fn_type.empty()) {
            ctx.recordEvent(act_fn_type, 2);
        }
        if (act_fn_type == "gelu") {
            nn::gelu_inplace(ret, ctx.current_stream()->ptr);
        } else if (act_fn_type == "silu") {
            nn::silu_inplace(ret, ctx.current_stream()->ptr);
        } else if (act_fn_type != "") {
            throw std::runtime_error(act_fn_type + " activation is not supported");
        }
        return ret;
    }
};

class Linear::impl::NormalLinear : public Linear::impl {
public:
    bool parallel;
    core::DistLayout dist_layout;
    float scale_factor;
    std::unique_ptr<Tensor> weight;
    Tensor bias;
    functions::Gemm gemm_A_B;
    functions::Gemm gemm_A_Btrans;

    NormalLinear(const core::Context &ctx,
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

    core::Tensor forward(const core::Context &ctx,
                         const core::Tensor &input,
                         const std::string &output_name,
                         bool quant_back,
                         Tensor *output) override {
        // Input: (sel_len, dim_in)
        // Output: (seq_len, dim_out)
        BM_ASSERT(input.ndim() == 2 || input.ndim() == 3, "Input must be 2D/3D");
        BM_ASSERT_EQ(input.dtype(), weight->dtype(), "Input data type mismatch");
        BM_ASSERT_EQ(input.device(), weight->device(), "Input and weight must be on the same device");

        core::Tensor ret; // (seq_len, dim_out)
        // x @ W^T
        if (!weight_transposed) {
            ret = gemm_A_Btrans.forward(ctx, input, *weight, output, has_bias ? &bias : nullptr);
        } else {
            ret = gemm_A_B.forward(ctx, input, *weight);
        }

        // set name here to avoid memory allocation.
        ret.set_name(output_name);
        return activate(ctx, ret);
    }
};

Linear::Linear(
    const core::Context &ctx,
    int dim_in,
    int dim_out,
    std::string act_fn_type,
    bool scale_weights,
    bool weight_transposed,
    bool parallel,
    core::DistLayout dist_layout,
    core::DataType dtype) :
    Layer() {
    std::cout << ">>>>>>>>> Linear::Linear constructor" << std::endl;
    auto tmp = new impl::NormalLinear(ctx, dim_in, dim_out, act_fn_type, scale_weights, weight_transposed, dtype, parallel, dist_layout);
    add_parameter("weight", *tmp->weight);
    pimpl = std::unique_ptr<impl>((impl *)tmp);

    pimpl->dist_layout = dist_layout;
}

Linear::Linear(
    const core::Context &ctx,
    int dim_in,
    int dim_out,
    core::DistLayout dist_layout,
    core::DataType dtype) :
    Linear(ctx, dim_in, dim_out, "", false, false, ctx.world_size() > 1, dist_layout, dtype) {
}

Linear::Linear(
    const core::Context &ctx,
    const std::string &name,
    const core::Tensor &w) :
    Linear(ctx, w.size(1), w.size(0), 0, false, false, false, DistLayout::REPLICATED, w.dtype()) {
    BM_ASSERT_EQ(w.ndim(), 2, "");
    this->name = name;
    // auto ptr = dynamic_cast<impl::NormalLinear*>(pimpl.get());
    // BM_ASSERT(ptr, "Not NormalLinear");
    // *ptr->weight = w;
}

void Linear::move(Linear &other) {
    pimpl = std::move(other.pimpl);
}

Linear::~Linear() = default;

core::Tensor Linear::forward(const core::Context &ctx, const core::Tensor &input, bool quant_back, Tensor *output) {
    size_t K = input.size(-1);
    size_t M = input.numel() / K;
    size_t N = DistLayout::COLUMNAR == pimpl->dist_layout ? pimpl->dim_out / ctx.world_size() : pimpl->dim_out;
    auto name1 = "Linear(" + name + ")[M=";
    auto ev_name = logger::str_cat(name1, M, ",N=", N, ",K=", K, "]");
    size_t flops = 2UL * K * M * N;
    core::EventScope event_scope(ctx, ev_name, 2, flops);

    Tensor ret;

    if (input.ndim() == 2) {
        ret = pimpl->forward(ctx, input, output_name, quant_back, output);
    } else {
        core::Tensor input2d = input.view({input.numel() / input.size(-1), input.size(-1)});
        Tensor ret2d = pimpl->forward(ctx, input2d, output_name, quant_back, output);
        auto out_shape = input.shape();
        out_shape[out_shape.size() - 1] = ret2d.size(-1);
        ret = ret2d.view(out_shape);
    }

    return ret;
}

void Linear::load_state_dict(
    const core::Context &ctx,
    const std::map<std::string, const core::Tensor> &state_dict,
    const std::string &prefix,
    bool allow_missing) {
    this->prefix = prefix;
}
