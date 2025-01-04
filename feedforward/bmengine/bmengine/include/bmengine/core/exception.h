#pragma once
#include "bmengine/core/export.h"
#include <iostream>
#include <stdexcept>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <string>

#if defined(__GNUC__)
static inline bool bm_likely(bool x) {
    return __builtin_expect((x), true);
}
static inline bool bm_unlikely(bool x) {
    return __builtin_expect((x), false);
}
#else
static inline bool bm_likely(bool x) {
    return x;
}
static inline bool bm_unlikely(bool x) {
    return x;
}
#endif

class BMEngineException : public std::runtime_error {
    const char* file;
    int line;
    const char* func;
    std::string info;
    std::string error_msg;

public:
    BMEngineException(
        const std::string& msg,
        const char* file_,
        int line_,
        const char* func_,
        const std::string& info_ = "")
        : std::runtime_error(msg), file(file_), line(line_), func(func_), info(info_) {
#ifdef NDEBUG
        error_msg = msg + "\n" + info_ + "\n\n";
#else
        error_msg = std::string("File: ") + file_ + ":" + std::to_string(line_) + " " + func_ + "\n"
                    + msg + "\n" + info_ + "\n\n";
#endif
        // print the message in case that exception is not caught.
        std::cerr << error_msg << std::endl;
    }

    const char* what() const noexcept { return error_msg.c_str(); }
};

#define BM_ASSERT(cond, msg)                                                                       \
    if (bm_unlikely(!(cond))) {                                                                    \
        bmengine::print_demangled_trace(15);                                                       \
        throw BMEngineException(                                                                   \
            "Assertion failed: " #cond, __FILE__, __LINE__, __PRETTY_FUNCTION__, msg);             \
    }
#define BM_ASSERT_EQ(x, y, msg)                                                                    \
    if (bm_unlikely((x) != (y))) {                                                                 \
        bmengine::print_demangled_trace(15);                                                       \
        throw BMEngineException(                                                                   \
            "Assertion failed: " #x " != " #y " i.e. " + std::to_string(x)                         \
                + " != " + std::to_string(y),                                                      \
            __FILE__,                                                                              \
            __LINE__,                                                                              \
            __PRETTY_FUNCTION__,                                                                   \
            msg);                                                                                  \
    }
#define BM_ASSERT_LT(x, y, msg)                                                                    \
    if (bm_unlikely((x) >= (y))) {                                                                 \
        bmengine::print_demangled_trace(15);                                                       \
        throw BMEngineException(                                                                   \
            "Assertion failed: " #x " < " #y " i.e. " + std::to_string(x)                         \
                + " < " + std::to_string(y),                                                      \
            __FILE__,                                                                              \
            __LINE__,                                                                              \
            __PRETTY_FUNCTION__,                                                                   \
            msg);                                                                                  \
    }
#define BM_ASSERT_LE(x, y, msg)                                                                    \
    if (bm_unlikely((x) > (y))) {                                                                 \
        bmengine::print_demangled_trace(15);                                                       \
        throw BMEngineException(                                                                   \
            "Assertion failed: " #x " <= " #y " i.e. " + std::to_string(x)                         \
                + " <= " + std::to_string(y),                                                      \
            __FILE__,                                                                              \
            __LINE__,                                                                              \
            __PRETTY_FUNCTION__,                                                                   \
            msg);                                                                                  \
    }
#define BM_CUDART_ASSERT(status)                                                                   \
    do {                                                                                           \
        cudaError_t _v = (status);                                                                  \
        if (bm_unlikely(_v != cudaSuccess)) {                                                       \
            bmengine::print_demangled_trace(15);                                                   \
            throw BMEngineException(                                                               \
                "CUDA Runtime Error: " #status,                                                    \
                __FILE__,                                                                          \
                __LINE__,                                                                          \
                __PRETTY_FUNCTION__,                                                               \
                cudaGetErrorString(_v));                                                            \
        }                                                                                          \
    } while (0)
#define BM_CUBLAS_ASSERT(status)                                                                   \
    do {                                                                                           \
        cublasStatus_t v = (status);                                                               \
        if (bm_unlikely(v != CUBLAS_STATUS_SUCCESS)) {                                             \
            bmengine::print_demangled_trace(15);                                                   \
            throw BMEngineException(                                                               \
                "CUBLAS Error: " #status,                                                          \
                __FILE__,                                                                          \
                __LINE__,                                                                          \
                __PRETTY_FUNCTION__,                                                               \
                cublasGetErrorString(v));                                                          \
        }                                                                                          \
    } while (0)
#define BM_NCCL_ASSERT(status)                                                                     \
    do {                                                                                           \
        ncclResult_t v = (status);                                                                 \
        if (bm_unlikely(v != ncclSuccess))                                                         \
            throw BMEngineException(                                                               \
                "NCCL Error: " #status,                                                            \
                __FILE__,                                                                          \
                __LINE__,                                                                          \
                __PRETTY_FUNCTION__,                                                               \
                ncclGetErrorString(v));                                                            \
    } while (0)
#define BM_EXCEPTION(msg)                                                                          \
    throw BMEngineException("Exception:\n", __FILE__, __LINE__, __PRETTY_FUNCTION__, msg)

#define CURAND_CHECK(err)                                                                          \
    do {                                                                                           \
        curandStatus_t err_ = (err);                                                               \
        if (err_ != CURAND_STATUS_SUCCESS) {                                                       \
            throw BMEngineException(                                                               \
                "Exception:\n", __FILE__, __LINE__, __PRETTY_FUNCTION__, "curand error");          \
        }                                                                                          \
    } while (0)

namespace bmengine {
BMENGINE_EXPORT void backtrace(int depth);
BMENGINE_EXPORT void print_demangled_trace(int depth);
BMENGINE_EXPORT const char* cublasGetErrorString(cublasStatus_t status);
} // namespace bmengine
