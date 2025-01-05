#include "backend/linear.h"

using bmengine::core::DataType;
using bmengine::core::DistLayout;
using bmengine::core::Tensor;

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
};

class Linear::impl::NormalLinear : public Linear::impl {
public:
    bool parallel;
    core::DistLayout dist_layout;
    float scale_factor;
    std::unique_ptr<Tensor> weight;
    Tensor bias;

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
        scale_factor(float(scale_weights ? 1.0 / sqrtf(dim_in) : 1.0)) {
        std::vector<size_t> shape({
            weight_transposed ? dim_in : dim_out, // W^T
            weight_transposed ? dim_out : dim_in, // W
        });
    }

    ~NormalLinear() = default;
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
