#include "linear.h"
#include <bmengine/core/core.h>

using bmengine::core::Tensor;
using bmengine::core::DataType;
using bmengine::core::DistLayout;

namespace nn {

class Linear::impl {
public:
    // class NormalLinear;
    // class Int8Linear;
    // class Fp8Linear;
    // class Int4GPTQ;
    // class GPTQMarlin;
    // class AWQ;

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
    // virtual void set_compute_type(cublasComputeType_t compute_type) {
    // }

    // virtual core::Tensor forward(
    //     const core::Context &ctx,
    //     const core::Tensor &input,
    //     const std::string &output_name,
    //     bool quant_back,
    //     Tensor *output) = 0;

    // virtual core::Tensor &get_weight() = 0;
    // virtual core::Tensor get_dequant_weight(const core::Context &ctx) {
    //     throw std::runtime_error("not supported");
    // };
    // virtual core::Tensor *get_weight_scale() {
    //     return nullptr;
    // }

    // virtual void load_parameters(const core::Context &ctx, const std::string &prefix) {
    //     throw std::runtime_error("load_parameters for QuantImpl only");
    // }

    // virtual void load_state_dict(
    //     const core::Context &ctx,
    //     const std::map<std::string, const core::Tensor> &state_dict,
    //     const std::string &prefix,
    //     bool allow_missing) = 0;
    // virtual void set_has_bias(bool b) {
    //     if (b) throw std::runtime_error("Bias is not implemented");
    // }
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
    // auto tmp = new impl::NormalLinear(
    //     ctx, dim_in, dim_out, act_fn_type, scale_weights, weight_transposed, dtype, parallel, dist_layout);
    // add_parameter("weight", *tmp->weight);
    // gemm has no weight; add only for set prefix
    // add_submodule("gemm_A_B", tmp->gemm_A_B);
    // add_submodule("gemm_A_Btrans", tmp->gemm_A_Btrans);
    // pimpl = std::unique_ptr<impl>((impl*) tmp);
    // pimpl->dist_layout = dist_layout;
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

// void Linear::scale_output(float scale) {
//     pimpl->scale_output(scale);
// }
// void Linear::set_output_type(core::DataType dtype) {
//     pimpl->set_output_type(dtype);
// }

void Linear::init_parameters(
    const core::Context &ctx, curandGenerator_t &gen, const std::string &prefix) {
}

void Linear::load_state_dict(
    const core::Context &ctx,
    const std::map<std::string, const core::Tensor> &state_dict,
    const std::string &prefix,
    bool allow_missing) {
    this->prefix = prefix;
    // pimpl->load_state_dict(ctx, state_dict, prefix, allow_missing);

    // bool dequant_desc_act = utils::get_int_env("DEQUANT_DESC_ACT", 0) > 0;
    // impl::Int4GPTQ* q = dynamic_cast<impl::Int4GPTQ*>(pimpl.get());
    // if (dequant_desc_act && q && q->parallel && q->act_order && !q->dim_out_parallel) {
    //     Tensor w = q->get_dequant_weight(ctx);
    //     auto new_p = new impl::NormalLinear(
    //         ctx, q->dim_in, q->dim_out, q->act_fn_type, false, false, q->dtype, q->parallel, q->dist_layout);
    //     new_p->weight = std::make_unique<Tensor>();
    //     *new_p->weight = w;
    //     pimpl.reset(new_p);
    // }
}
} // namespace nn
