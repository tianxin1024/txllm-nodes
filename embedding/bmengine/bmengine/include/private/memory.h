#pragma once
#include <functional>
#include <memory>
#include <vector>

namespace bmengine {
namespace core {
typedef std::function<void(void*)> MemoryDeleter;
class Memory_ {
public:
    void* ptr;
    int dev;
    size_t num_bytes;
    std::vector<MemoryDeleter> deleters;

    Memory_(void* ptr, int dev, size_t num_bytes, MemoryDeleter deleter);
    Memory_(const Memory_&) = delete;
    Memory_(Memory_&&) = default;
    Memory_& operator=(const Memory_&) = delete;
    Memory_& operator=(Memory_&&) = default;
    ~Memory_();

    void add_deleter(MemoryDeleter deleter);
};

typedef std::shared_ptr<Memory_> Memory;

}
}