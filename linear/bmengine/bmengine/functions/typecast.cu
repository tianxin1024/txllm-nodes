#include "bmengine/functions/typecast.h"
#include "bmengine/core/tensor.h"
#include <cuda_bf16.hpp>

namespace bmengine {
namespace functions {

template<typename T1, typename T2>
static __global__ void BM_KERNEL(typecast)(size_t n, const T1* in, T2* out) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = T2(in[i]);
    }
}

#define CALL_TYPECAST(old_type, new_type)                                                          \
    {                                                                                              \
        BM_KERNEL(typecast)<old_type, new_type>                                                    \
            <<<gridDim, blockDim, 0, stream>>>(n, in.data<old_type>(), out.data<new_type>());      \
        break;                                                                                     \
    }

template<typename T1, typename T2>
static __global__ void BM_KERNEL(typecast_half)(size_t n, const T1* in, T2* out) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = T2(float(in[i]));
    }
}

#define CALL_TYPECAST_HALF(old_type, new_type)                                                     \
    {                                                                                              \
        BM_KERNEL(typecast_half)<old_type, new_type>                                               \
            <<<gridDim, blockDim, 0, stream>>>(n, in.data<old_type>(), out.data<new_type>());      \
        break;                                                                                     \
    }

core::Tensor typecast(const core::Context& ctx, const core::Tensor& in, core::DataType out_type) {
    if (in.dtype() == out_type) {
        return in;
    }
    auto stream = ctx.current_stream()->ptr;

    size_t n = in.numel();
    int threads = round_up_thread(n);
    dim3 gridDim(round_up(n, threads) / threads, 1, 1);
    dim3 blockDim(threads, 1, 1);
    core::Tensor out = ctx.tensor(in.size(), out_type);

    auto in_type = in.dtype();
    switch (in_type) {
        case core::DataType::kDouble:
            switch (out_type) {
                case core::DataType::kFloat: CALL_TYPECAST(double, float);
                case core::DataType::kHalf: CALL_TYPECAST(double, half);
                case core::DataType::kBFloat16: CALL_TYPECAST(double, nv_bfloat16);
                default: BM_EXCEPTION("Unsupported typecast from " + std::string(get_data_type_name(in_type)) + " to " + std::string(get_data_type_name(out_type)));
            };
            break;
        case core::DataType::kFloat:
            switch (out_type) {
                case core::DataType::kDouble: CALL_TYPECAST(float, double);
                case core::DataType::kHalf: CALL_TYPECAST(float, half);
                case core::DataType::kBFloat16: CALL_TYPECAST(float, nv_bfloat16);
                default: BM_EXCEPTION("Unsupported typecast from " + std::string(get_data_type_name(in_type)) + " to " + std::string(get_data_type_name(out_type)));
            };
            break;
        case core::DataType::kHalf:
            switch (out_type) {
                case core::DataType::kDouble: CALL_TYPECAST(half, double);
                case core::DataType::kFloat: CALL_TYPECAST(half, float);
                case core::DataType::kBFloat16: CALL_TYPECAST_HALF(half, nv_bfloat16);
                default: BM_EXCEPTION("Unsupported typecast from " + std::string(get_data_type_name(in_type)) + " to " + std::string(get_data_type_name(out_type)));
            };
            break;
        case core::DataType::kBFloat16:
            switch (out_type) {
                case core::DataType::kDouble: CALL_TYPECAST(nv_bfloat16, double);
                case core::DataType::kFloat: CALL_TYPECAST(nv_bfloat16, float);
                 case core::DataType::kHalf: CALL_TYPECAST_HALF(nv_bfloat16, half);
                default: BM_EXCEPTION("Unsupported typecast from " + std::string(get_data_type_name(in_type)) + " to " + std::string(get_data_type_name(out_type)));
            };
            break;
        case core::DataType::kInt32:
            switch (out_type) {
                case core::DataType::kInt16: CALL_TYPECAST(int32_t, int16_t);
                case core::DataType::kInt8: CALL_TYPECAST(int32_t, int8_t);
                default: BM_EXCEPTION("Unsupported typecast from " + std::string(get_data_type_name(in_type)) + " to " + std::string(get_data_type_name(out_type)));
            };
            break;
        case core::DataType::kInt16:
            switch (out_type) {
                case core::DataType::kInt32: CALL_TYPECAST(int16_t, int32_t);
                case core::DataType::kInt8: CALL_TYPECAST(int16_t, int8_t);
                default: BM_EXCEPTION("Unsupported typecast from " + std::string(get_data_type_name(in_type)) + " to " + std::string(get_data_type_name(out_type)));
            };
            break;
        case core::DataType::kInt8:
            switch (out_type) {
                case core::DataType::kInt32: CALL_TYPECAST(int8_t, int32_t);
                case core::DataType::kInt16: CALL_TYPECAST(int8_t, int16_t);
                default: BM_EXCEPTION("Unsupported typecast from " + std::string(get_data_type_name(in_type)) + " to " + std::string(get_data_type_name(out_type)));
            };
            break;
        default: BM_EXCEPTION("Unsupported typecast from " + std::string(get_data_type_name(in_type)) + " to " + std::string(get_data_type_name(out_type)));
    }
    BM_CUDART_ASSERT(cudaGetLastError());
    return out;
}

}
}