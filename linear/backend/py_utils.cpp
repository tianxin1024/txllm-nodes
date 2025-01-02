#include "backend/py_utils.h"

namespace bind {

DataType aten_typemeta_to_bmengine(caffe2::TypeMeta type_meta) {
    auto scalar_type = type_meta.toScalarType();
    switch (scalar_type) {
    case at::ScalarType::Double: return DataType::kDouble;
    case at::ScalarType::Float: return DataType::kFloat;
    case at::ScalarType::Half: return DataType::kHalf;
    case at::ScalarType::Char: return DataType::kInt8;
    case at::ScalarType::Short: return DataType::kInt16;
    case at::ScalarType::Int: return DataType::kInt32;
    case at::ScalarType::BFloat16: return DataType::kBFloat16;
    default: break;
    }
    std::stringstream ss;
    ss << std::string("can't convert at::Tensor of scalar_type ") << type_meta.name()
       << "The only supported types are: "
          "Double, Float, Half, Chat, Short, Int and BFloat16.";
    throw std::runtime_error(ss.str());
}

const Tensor aten_to_tensor(const Context &ctx, const at::Tensor &at_tensor) {
    if (at_tensor.numel() == 0) {
        return Tensor();
    }
    auto shape = at_tensor.sizes().vec();
    std::vector<size_t> sizes(shape.begin(), shape.end());
    auto dtype = aten_typemeta_to_bmengine(at_tensor.dtype());
    if (at_tensor.is_cpu()) {
        auto tensor = ctx.tensor(sizes, dtype);
        tensor.from_buffer(at_tensor.data_ptr());
        return std::move(tensor);
    } else {
        BM_ASSERT(at_tensor.is_cuda() && at_tensor.get_device() == ctx.active_device(),
                  "tensor device miss match.");
        const auto tensor = Tensor::from_external(
            sizes, dtype, at_tensor.data_ptr(), at_tensor.nbytes(), ctx.active_device());
        return std::move(tensor);
    }
}

void load_at_state_dict(
    bmengine::core::Context &ctx,
    const std::map<std::string, at::Tensor> &state_dict,
    std::map<const std::string, bmengine::core::Tensor *> named_params,
    bool parallel) {
    for (auto it : named_params) {
        auto p = state_dict.find(it.first);
        if (p != state_dict.end()) {
            if (!parallel || ctx.rank() == 0) {
                auto tensor = aten_to_tensor(ctx, p->second);
                *it.second = tensor;
            } else {
                throw std::runtime_error("parallel not supported yet.");
            }

        } else {
            std::stringstream ss;
            ss << "state_dict missing: " << it.first;
            throw std::runtime_error(ss.str());
        }
    }
}
} // namespace bind
