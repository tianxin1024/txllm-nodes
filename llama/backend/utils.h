#pragma once
#include <cstdlib>
#include <condition_variable>
#include <mutex>
#include <queue>

namespace utils {

using namespace std::chrono_literals;

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

    size_t size() {
        Lock lock(mutex_);
        return queue_.size();
    }

    T pop_timeout(float timeout) {
        auto now = std::chrono::system_clock::now();
        Lock lock(mutex_);
        if (timeout <= 0) {
            can_pop_cond_.wait(lock, [this] { return !queue_.empty(); });
        } else {
            auto tp = now + int(timeout * 1000) * 1ms;
            if (!can_pop_cond_.wait_until(lock, tp, [this] { return !queue_.empty(); })) {
                throw std::runtime_error("Timeout");
            }
        }
        auto res = std::move(queue_.front());
        queue_.pop();
        return std::move(res);
    }

}; // end of class TSQueue
} // namespace utils
