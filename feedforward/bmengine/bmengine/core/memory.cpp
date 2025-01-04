#include "private/memory.h"

namespace bmengine {

namespace core {

Memory_::Memory_(void* ptr, int dev, size_t num_bytes, MemoryDeleter deleter)
    : ptr(ptr), dev(dev), num_bytes(num_bytes) {

    deleters.emplace_back(std::move(deleter));
}

Memory_::~Memory_() {
    for (auto& deleter : deleters) {
        deleter(ptr);
    }
}

void Memory_::add_deleter(MemoryDeleter deleter) {
    deleters.emplace_back(std::move(deleter));
}

} // namespace core

} // namespace bmengine
