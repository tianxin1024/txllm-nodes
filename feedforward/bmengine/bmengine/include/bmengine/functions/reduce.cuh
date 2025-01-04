#pragma once
#include "utils.cuh"

namespace bmengine{

namespace functions {

template<typename T>
__inline__ __device__ T threadMax(T a, T b) {
    return (a > b) ? a : b;
}

template<typename T>
__inline__ __device__ T warpReduceMax(T x) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) 
        x = threadMax<T>(x, __shfl_down_sync(0xFFFFFFFF, x, offset));
    return x;
}
template<typename T>
__inline__ __device__ T warpReduceMaxB(T x) {
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        x = threadMax<T>(x, __shfl_down_sync(0xFFFFFFFF, x, offset));
    return __shfl_sync(0xFFFFFFFF, x, 0);  // broadcast
}

template<typename T>
__inline__ __device__ T threadMin(T a, T b) {
    return (a > b) ? b : a;
}

template<typename T>
__inline__ __device__ T warpReduceMin(T x) {
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        x = threadMin<T>(x, __shfl_down_sync(0xFFFFFFFF, x, offset));
    return x;
}

template<typename T>
__inline__ __device__ T warpReduceSum(T x) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) 
        x += __shfl_down_sync(0xFFFFFFFF, x, offset);
    return x;
}
template<typename T>
__inline__ __device__ T warpReduceSumB(T x) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        x += __shfl_down_sync(0xFFFFFFFF, x, offset);
    return __shfl_sync(0xFFFFFFFF, x, 0);  // broadcast
}

template<typename T>
__inline__ __device__ T blockReduceMax(T x) {
    static __shared__ T shared[33];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    x = warpReduceMax<T>(x);
    if (lane == 0) shared[wid] = x;
    __syncthreads();
    x = (threadIdx.x < blockDim.x / 32) ? shared[lane] : T(-INFINITY);
    if (wid == 0) {
        x = warpReduceMax<T>(x);
        if (lane == 0) shared[32] = x;
    }
    __syncthreads();
    return shared[32];  // avoid RAW hazard
}

template<typename T>
__inline__ __device__ T blockReduceMin(T x) {
    static __shared__ T shared[33];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    x = warpReduceMin<T>(x);
    if (lane == 0)
        shared[wid] = x;
    __syncthreads();
    x = (threadIdx.x < blockDim.x / 32) ? shared[lane] : T(INFINITY);
    if (wid == 0) {
        x = warpReduceMin<T>(x);
        if (lane == 0)
            shared[32] = x;
    }
    __syncthreads();
    return shared[32]; // avoid RAW hazard
}

template<typename T>
__inline__ __device__ T blockReduceSum(T x) {
    static __shared__ T shared[33];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    x = warpReduceSum<T>(x);
    if (lane == 0) shared[wid] = x;
    __syncthreads();
    x = (threadIdx.x < blockDim.x / 32) ? shared[lane] : T(0.);
    if (wid == 0) {
        x = warpReduceSum<T>(x);
        if (lane == 0) shared[32] = x;
    }
    __syncthreads();
    return shared[32];  // avoid RAW hazard
}

} // namespace functions

} // namespace bmengine
