#include "backend/linear.h"
#include "backend/utils.h"
#include <bmengine/core/core.h>
#include <bmengine/functions/all.h>
#include <bmengine/logger/std_log_op.hpp>

namespace nn {
using namespace bmengine;

// tensors will be updated to returned tensor's slice
core::Tensor concat_dim0(const core::Context &ctx, std::vector<core::Tensor *> tensors, bool stack) {
    BM_ASSERT(!tensors.empty(), "");
    auto shape = tensors[0]->shape();
    auto shape_a = tensors[0]->shape();
    shape[0] = 0;
    std::vector<size_t> dim0s;
    std::vector<size_t> bytes;
    std::vector<void *> datas;
    for (core::Tensor *t : tensors) {
        if (stack) {
            BM_ASSERT_EQ(shape_a, t->shape(), "shape mismatch");
        }
        shape[0] += t->size(0);
        dim0s.push_back(t->size(0));
        bytes.push_back(t->nbytes());
        datas.push_back(t->data());
    }
    core::Tensor ret = ctx.tensor(shape, tensors[0]->dtype());

    auto stream = ctx.current_stream()->ptr;
    auto d2d = cudaMemcpyDeviceToDevice;
    char *dst = ret.data<char>();
    size_t dim0 = 0;
    std::vector<core::Tensor *> quant_scales;
    for (size_t i = 0; i < tensors.size(); ++i) {
        BM_CUDART_ASSERT(cudaMemcpyAsync(dst, datas[i], bytes[i], d2d, stream));
        auto name = tensors[i]->name();
        auto quant_scale = tensors[i]->quant_scale;
        if (quant_scale) {
            quant_scales.push_back(quant_scale.get());
        }
        *tensors[i] = ret.slice_dim0_len(dim0, dim0s[i]); // update tensors to slice
        tensors[i]->set_name(name);
        tensors[i]->quant_scale = quant_scale;
        dst += bytes[i];
        dim0 += dim0s[i];
    }
    if (stack) {
        shape[0] /= tensors.size();
        shape.insert(shape.begin(), tensors.size());
    }
    ret = ret.view(shape);
    if (!quant_scales.empty()) {
        core::Tensor fuse_scale = concat_dim0(ctx, quant_scales, stack);
        // int8_op::set_quant_scale(ret, fuse_scale);
    }
    return ret;
}

static core::Tensor concat2_dim0(const core::Context &ctx, core::Tensor &a, core::Tensor &b) {
    return concat_dim0(ctx, {&a, &b}, false);
}

static core::Tensor concat2_dim1(const core::Context &ctx, const core::Tensor &a, const core::Tensor &b) {
    return functions::concat_tensor(ctx, a, b, 1);
}

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

    virtual void set_output_type(core::DataType dtype) = 0;

    virtual void load_state_dict(const core::Context &ctx,
                                 const std::map<std::string, const core::Tensor> &state_dict,
                                 const std::string &prefix,
                                 bool allow_missing) = 0;

    virtual void set_has_bias(bool b) {
        if (b)
            throw std::runtime_error("Bias is not implemented");
    }

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

    void set_output_type(core::DataType dtype) override {
        gemm_A_B.set_output_type(dtype);
        gemm_A_Btrans.set_output_type(dtype);
    }

    void set_has_bias(bool b) override {
        has_bias = b;
    }

    static NormalLinear *fuse(const core::Context &ctx, NormalLinear &a, NormalLinear &b) {
        BM_ASSERT_EQ(a.scale_factor, b.scale_factor, "scale_factor not equal");
        uint32_t dim_out = a.dim_out + b.dim_out;
        NormalLinear *ret = new NormalLinear(
            ctx, a.dim_in, dim_out, "", false, a.weight_transposed, a.dtype, false, core::DistLayout::ROW);
        core::Tensor weight = a.weight_transposed ?
                                  concat2_dim1(ctx, *a.weight, *b.weight) :
                                  concat2_dim0(ctx, *a.weight, *b.weight);
        ret->weight = std::make_unique<core::Tensor>(weight);
        ret->scale_factor = a.scale_factor;
        if (a.weight_transposed) {
            a.weight.reset();
            b.weight.reset();
        } else {
            BM_ASSERT(ret->weight->data() == a.weight->data(), "weight not match.");
        }

        if (a.has_bias) {
            ret->bias = concat2_dim0(ctx, a.bias, b.bias);
        }
        ret->has_bias = a.has_bias;
        return ret;
    }

    void load_state_dict(const core::Context &ctx,
                         const std::map<std::string, const core::Tensor> &state_dict,
                         const std::string &prefix,
                         bool allow_missing) override {
        std::vector<size_t> shape({weight_transposed ? dim_in : dim_out,   // W^T
                                   weight_transposed ? dim_out : dim_in}); // W
        weight = std::make_unique<core::Tensor>(ctx.parameter(shape, dtype));
        auto name = prefix + ".weight";
        ctx.load_parameter(weight.get(), name, state_dict, parallel, dist_layout);

        auto bias_layout = dist_layout == core::DistLayout::ROW ? core::DistLayout::COLUMNAR : core::DistLayout::REPLICATED;
        if (has_bias) {
            name = prefix + ".bias";
            bias = ctx.parameter({dim_out}, dtype);
            ctx.load_parameter(&bias, name, state_dict, parallel, bias_layout);
        }
    }
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
    Linear(ctx, w.size(1), w.size(0), "", 0, false, false, false, core::DistLayout::REPLICATED, w.dtype()) {
    BM_ASSERT_EQ(w.ndim(), 2, "");
    this->name = name;
    auto ptr = dynamic_cast<impl::NormalLinear *>(pimpl.get());
    BM_ASSERT(ptr, "Not NormalLinear");
    *ptr->weight = w;
}

Linear::~Linear() = default;

void Linear::set_output_type(core::DataType dtype) {
    pimpl->set_output_type(dtype);
}

void Linear::load_state_dict(const core::Context &ctx,
                             const std::map<std::string, const core::Tensor> &state_dict,
                             const std::string &prefix,
                             bool allow_missing) {
    this->prefix = prefix;
    pimpl->load_state_dict(ctx, state_dict, prefix, allow_missing);

    bool dequant_desc_act = utils::get_int_env("DEQUANT_DESC_ACT", 0) > 0;
}

Linear *Linear::fuse(const core::Context &ctx, Linear &q, Linear &k) {
    std::unique_ptr<Linear> ret(new Linear());

    if (q.pimpl->quant == 0) {
        auto q_ptr = dynamic_cast<impl::NormalLinear *>(q.pimpl.get());
        auto k_ptr = dynamic_cast<impl::NormalLinear *>(k.pimpl.get());
        auto fused_ptr = impl::NormalLinear::fuse(ctx, *q_ptr, *k_ptr);
        ret->pimpl = std::unique_ptr<impl>(fused_ptr);
    } else {
        return nullptr;
    }

    if (q.name == "w_in") {
        ret->name = "FUSE_ff_in";
    }

    return ret.release();
}

void Linear::set_has_bias(bool b) {
    pimpl->set_has_bias(b);
}

} // namespace nn
