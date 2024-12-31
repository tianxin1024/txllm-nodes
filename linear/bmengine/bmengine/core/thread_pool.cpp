#include "bmengine/core/thread_pool.h"
#include <sched.h>
#include <pthread.h>
#include <iostream>
namespace bmengine {

namespace core {

TaskThreadPool::TaskThreadPool(size_t num_threads, int cpu_offset)
    : threads_(num_threads), task_count_(0), finished_(false) {
    for (int i = 0; i < num_threads; ++i) {
        threads_[i] = std::thread([this, i, cpu_offset] {
            std::string name = "ZhiLight-TP" + std::to_string(cpu_offset);
            pthread_setname_np(pthread_self(), name.c_str());
            this->execution_loop();
        });
    }
}

TaskThreadPool::~TaskThreadPool() {
    std::unique_lock<std::mutex> lock(mutex_);
    finished_ = true;
    task_notifier_.notify_all();
    lock.unlock();

    for (auto& it : threads_) {
        it.join();
    }
}

void TaskThreadPool::wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    stop_notifier_.wait(lock, [this] { return task_count_ == 0; });
    if (e_ptr) {
        auto tmp_ptr = e_ptr;
        e_ptr = std::exception_ptr();
        std::rethrow_exception(tmp_ptr);
    }
}

void TaskThreadPool::execution_loop() {
    std::unique_lock<std::mutex> lock(mutex_);
    for (;;) {
        task_notifier_.wait(lock, [this] { return !this->tasks_.empty() || finished_; });
        if (finished_) {
            stop_notifier_.notify_all();
            return;
        }
        auto task = std::move(tasks_.front());
        tasks_.pop();
        lock.unlock();
        try {
            // std::cout << "Begin execution_loop \n";
            task();
        } catch (...) {
            e_ptr = std::current_exception();
        }
        lock.lock();

        if (--task_count_ == 0) {
            stop_notifier_.notify_all();
        }
    }
}

void TaskThreadPool::run(std::function<void()> task) {
    std::unique_lock<std::mutex> lock(mutex_);
    tasks_.emplace(std::move(task));
    task_count_++;
    task_notifier_.notify_one();
}

void TaskThreadPool::runSync(std::function<void()> task) {

    std::promise<void> prom;
    auto future = prom.get_future();
    run([task, &prom]() {
        task();
        prom.set_value();
    });
    future.wait();
}

}
}