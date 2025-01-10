#pragma once
#include <cuda_runtime.h>
#include <functional>
#include <memory>

namespace bmengine {

namespace core {

typedef std::function<void(cudaStream_t)> StreamDeleter;

class Stream_ {
public:
    cudaStream_t ptr;
    StreamDeleter deleter;

    Stream_(cudaStream_t ptr, StreamDeleter deleter);
    Stream_(const Stream_&) = delete;
    Stream_(Stream_&&) = delete;
    Stream_& operator=(const Stream_&) = delete;
    Stream_& operator=(Stream_&&) = delete;
    ~Stream_();
};

typedef std::shared_ptr<Stream_> Stream;

}

}