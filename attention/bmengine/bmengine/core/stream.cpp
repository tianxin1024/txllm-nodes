#include "bmengine/core/stream.h"
#include "bmengine/core/exception.h"

#include "private/stream.h"
#include "private/guard.h"
#include <iostream>

namespace bmengine {

namespace core {

Stream_::Stream_(cudaStream_t stream, StreamDeleter deleter) : ptr(stream), deleter(deleter) { }
Stream_::~Stream_() {
    deleter(ptr);
}

StreamAllocator::StreamAllocator(int dev_id) : dev_id(dev_id) { }
StreamAllocator::~StreamAllocator() {
    try {
        while (!streams.empty()) {
            cudaStream_t stream = streams.top();
            streams.pop();
            BM_CUDART_ASSERT(cudaStreamDestroy(stream));
        }
    } catch (const BMEngineException& e) { std::cerr << e.what() << std::endl; }
}
cudaStream_t StreamAllocator::alloc() {
    std::lock_guard<std::mutex> lock(mutex);
    if (streams.empty()) {
        DeviceGuard d_guard(dev_id);
        cudaStream_t stream;
        BM_CUDART_ASSERT(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        streams.push(stream);
    }
    cudaStream_t ret = streams.top();
    streams.pop();
    return ret;
}
void StreamAllocator::free(cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(mutex);
    if (streams.size() > 16) {
        if (std::getenv("BM_DEBUG_LEVEL") != nullptr) {
            std::cout << "cudaStreamDestroy " << uint64_t(stream) << "\n";
        }
        BM_CUDART_ASSERT(cudaStreamDestroy(stream));
    } else {
        // reuse
        streams.push(stream);
    }
}

}

}