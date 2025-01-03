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

void Linear::init_parameters(
    const core::Context &ctx, curandGenerator_t &gen, const std::string &prefix) {
}

void Linear::load_state_dict(
    const core::Context &ctx,
    const std::map<std::string, const core::Tensor> &state_dict,
    const std::string &prefix,
    bool allow_missing) {
    this->prefix = prefix;
}

} // namespace nn
