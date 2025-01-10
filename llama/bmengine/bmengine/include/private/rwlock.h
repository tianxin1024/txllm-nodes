#pragma once
#include <mutex>
#include <condition_variable>

namespace bmengine {

namespace server {

class RWLock;
class ReadGuard {
    RWLock* lock;

public:
    inline ReadGuard(RWLock* lock) noexcept : lock(lock) { }
    ~ReadGuard();
};

class WriteGuard {
    RWLock* lock;

public:
    inline WriteGuard(RWLock* lock) noexcept : lock(lock) { }
    ~WriteGuard();
};

class RWLock {
public:
    std::mutex lock;
    std::condition_variable cond;
    int n_readers, n_writers;

    inline RWLock() noexcept : n_readers(0), n_writers(0) { }
    inline ~RWLock() = default;
    RWLock(const RWLock&) = delete;
    RWLock& operator=(const RWLock&) = delete;
    RWLock(RWLock&&) = delete;
    RWLock& operator=(RWLock&&) = delete;

    inline ReadGuard read() noexcept {
        std::unique_lock<std::mutex> l(lock);
        cond.wait(l, [this] { return n_writers == 0; });
        n_readers++;
        return ReadGuard(this);
    }

    inline WriteGuard write() noexcept {
        std::unique_lock<std::mutex> l(lock);
        cond.wait(l, [this] { return n_writers == 0 && n_readers == 0; });
        n_writers++;
        return WriteGuard(this);
    }
};

inline ReadGuard::~ReadGuard() noexcept {
    std::unique_lock<std::mutex> l(lock->lock);
    lock->n_readers--;
    if (lock->n_writers == 0)
        lock->cond.notify_one();
}

inline WriteGuard::~WriteGuard() noexcept {
    std::unique_lock<std::mutex> l(lock->lock);
    lock->n_writers--;
    if (lock->n_writers == 0)
        lock->cond.notify_all();
}

}

}