#pragma once
#include "bmengine/core/exception.h"
#include <cuda_runtime.h>
#include <iostream>

namespace bmengine {

namespace core {

class DeviceGuard {
private:
    int old_dev;

public:
    DeviceGuard(int idx) {
        BM_CUDART_ASSERT(cudaGetDevice(&old_dev));
        if (old_dev != idx) {
            BM_CUDART_ASSERT(cudaSetDevice(idx));
        }
    }
    ~DeviceGuard() {
        try {
            if (old_dev != -1) {
                BM_CUDART_ASSERT(cudaSetDevice(old_dev));
            }
        } catch (const BMEngineException& e) { std::cerr << e.what() << std::endl; }
    }
    DeviceGuard(const DeviceGuard&) = delete;
    DeviceGuard(DeviceGuard&&) = delete;
};

template<typename T>
class MemoryGuard {
private:
    T* ptr;

public:
    explicit MemoryGuard(T* p) : ptr(p) { }
    ~MemoryGuard() {
        try {
            if (ptr != nullptr) {
                delete ptr;
            }
        } catch (const BMEngineException& e) { std::cerr << e.what() << std::endl; }
    }
    MemoryGuard(const MemoryGuard&) = delete;
    MemoryGuard(MemoryGuard&&) = delete;
};

template<typename T>
class MemoryArrayGuard {
private:
    T* ptr;

public:
    explicit MemoryArrayGuard(T* p) : ptr(p) { }
    ~MemoryArrayGuard() {
        try {
            if (ptr != nullptr) {
                delete[] ptr;
            }
        } catch (const BMEngineException& e) { std::cerr << e.what() << std::endl; }
    }
    MemoryArrayGuard(const MemoryArrayGuard&) = delete;
    MemoryArrayGuard(MemoryArrayGuard&&) = delete;
};

}

}