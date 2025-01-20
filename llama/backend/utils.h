#pragma once
#include <cstdlib>
#include <condition_variable>
#include <mutex>
#include <queue>

namespace utils {

static inline int get_int_env(const char *name, int def_val = 0) {
    char *env_str = std::getenv(name);
    return env_str != nullptr ? std::atoi(env_str) : def_val;
}

template <typename T>
class TSQueue {
    int max_size_;
    std::mutex mutex_;
    std::condition_variable can_push_cond_;
    std::condition_variable can_pop_cond_;
    std::queue<T> queue_;
    volatile bool stopping_{false};

    typedef std::unique_lock<std::mutex> Lock;

public:
    explicit TSQueue(int max_size) :
        max_size_(max_size) {
    }
}; // end of class TSQueue
} // namespace utils
