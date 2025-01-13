#include "backend/linear.h"
#include <bmengine/core/core.h>
#include <bmengine/functions/all.h>

namespace nn {
using namespace bmengine;

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

    impl(uint32_t dim_in, uint32_t dim_out, std::string act_fn, bool w_trans, int quant, core::DataType dtype) :
        dim_in(dim_in), dim_out(dim_out), act_fn_type(act_fn), weight_transposed(w_trans), quant(quant), dtype(dtype) {
    }
    virtual ~impl() = default;

}; // end of class Lienar::impl

class Linear::impl::NormalLinear : public Linear::impl {
public:
    bool parallel;
    core::DistLayout dist_layout;
    float scale_factor;
    std::unique_ptr<core::Tensor> weight;
    core::Tensor bias;
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
        std::vector<size_t> shape({weight_transposed ? dim_in : dim_out,   // W^T
                                   weight_transposed ? dim_out : dim_in}); // W

        weight = std::make_unique<core::Tensor>(ctx.parameter(shape, dtype));
        if (ctx.high_precision() >= 1) {
            gemm_A_B.set_compute_type(CUBLAS_COMPUTE_32F),
                gemm_A_Btrans.set_compute_type(CUBLAS_COMPUTE_32F);
        }
    }

    ~NormalLinear() = default;

}; // end of class Linear::impl::NormalLinear

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
    auto tmp = new impl::NormalLinear(ctx, dim_in, dim_out, act_fn_type, scale_weights,
                                      weight_transposed, dtype, parallel, dist_layout);
    add_parameter("weight", *tmp->weight);
    // gemm has not weight; add only for set prefix
    add_submodule("gemm_A_B", tmp->gemm_A_B);
    add_submodule("gemm_A_Btrans", tmp->gemm_A_Btrans);
    pimpl = std::unique_ptr<impl>((impl *)tmp);

    pimpl->dist_layout = dist_layout;
}

} // namespace nn
