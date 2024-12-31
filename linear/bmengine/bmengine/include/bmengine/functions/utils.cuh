#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <limits>

namespace bmengine {
namespace functions {

template<typename T> __host__ __device__ inline T Inf();
template<> __host__ __device__ inline constexpr float Inf() { return __builtin_huge_valf(); }
template<> __host__ __device__ inline constexpr half Inf() { const short v = 0x7c00; return *(reinterpret_cast<const half *>(&(v))); }
template<> __host__ __device__ inline constexpr nv_bfloat16 Inf() { const short v = 0x7f80; return *(reinterpret_cast<const nv_bfloat16 *>(&(v))); }
template<> __host__ __device__ inline constexpr double Inf() { return __builtin_huge_val(); }
template<> __host__ __device__ inline constexpr int32_t Inf() { return __INT_MAX__; }
template<> __host__ __device__ inline constexpr int64_t Inf() { return __LONG_MAX__; }
template<> __host__ __device__ inline constexpr int16_t Inf() { return __SHRT_MAX__; }
template<> __host__ __device__ inline constexpr int8_t Inf() { return __SCHAR_MAX__; }


template <typename T>
struct SharedMemory;

template <>
struct SharedMemory <float>
{
    __device__ float *getPointer()
    {
        extern __shared__ float s_float[];
        return s_float;
    }
};

template <>
struct SharedMemory <double>
{
    __device__ double *getPointer()
    {
        extern __shared__ double s_double[];
        return s_double;
    }
};

template <>
struct SharedMemory <half>
{
    __device__ half *getPointer()
    {
        extern __shared__ half s_half[];
        return s_half;
    }
};

template <>
struct SharedMemory <nv_bfloat16>
{
    __device__ nv_bfloat16 *getPointer()
    {
        extern __shared__ nv_bfloat16 s_bfloat[];
        return s_bfloat;
    }
};

template <>
struct SharedMemory <int32_t>
{
    __device__ int32_t *getPointer()
    {
        extern __shared__ int32_t s_int32[];
        return s_int32;
    }
};

template <>
struct SharedMemory <int16_t>
{
    __device__ int16_t *getPointer()
    {
        extern __shared__ int16_t s_int16[];
        return s_int16;
    }
};

template <>
struct SharedMemory <int8_t>
{
    __device__ int8_t *getPointer()
    {
        extern __shared__ int8_t s_int8[];
        return s_int8;
    }
};

} // namespace functions
} // namespace bmengine
