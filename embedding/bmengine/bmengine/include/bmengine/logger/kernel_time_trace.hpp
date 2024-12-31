#pragma once

#include <cuda_runtime.h>
#include <sys/time.h>

namespace bmengine {
namespace logger {

inline void createStartEvent(
    bool cond, cudaEvent_t* start, cudaEvent_t* stop, cudaStream_t stream) {
    if (cond) {
        cudaEventCreate(start);
        cudaEventCreate(stop);
        cudaEventRecord(*start, stream);
    }
}

inline float destroyDiffEvent(cudaEvent_t start, cudaEvent_t stop, cudaStream_t stream) {
    float elapsed_ms;
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return elapsed_ms;
}

static inline long get_time_us() {
    timeval now;
    gettimeofday(&now, nullptr);
    return static_cast<long>(now.tv_sec * 1000 * 1000 + now.tv_usec);
}

} // namespace logger
} // namespace bmengine {
