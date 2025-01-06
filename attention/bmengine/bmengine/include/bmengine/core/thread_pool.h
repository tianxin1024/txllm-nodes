#pragma once
#include "bmengine/core/export.h"
#include <exception>
#include <future>
#include <queue>
#include <thread>

namespace bmengine {

namespace core {

class TaskThreadPool {
protected:
    std::queue<std::function<void()>> tasks_;
    volatile long task_count_;
    std::vector<std::thread> threads_;
    std::mutex mutex_;
    std::condition_variable task_notifier_;
    std::condition_variable stop_notifier_;
    std::exception_ptr e_ptr;

    bool finished_;
    void execution_loop();

public:
    explicit TaskThreadPool(size_t num_threads = 1, int cpu_offset = 0);
    ~TaskThreadPool();
    void run(std::function<void()>);
    void runSync(std::function<void()>);
    void wait();
};

}

}