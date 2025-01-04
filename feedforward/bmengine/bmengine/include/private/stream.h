#pragma once
#include <cuda_runtime.h>
#include <mutex>
#include <stack>

namespace bmengine {
namespace core {

class StreamAllocator {
private:
    int dev_id;
    std::stack<cudaStream_t> streams;
    std::mutex mutex;

public:
    StreamAllocator(int dev_id);
    ~StreamAllocator();
    cudaStream_t alloc();
    void free(cudaStream_t stream);
};

}
}