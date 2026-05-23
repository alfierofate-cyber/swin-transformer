#pragma once

#include <queue>
#include <mutex>
#include <condition_variable>
#include <optional>

namespace rtdet {

// 线程安全的有界队列（生产者-消费者模型）
// 面试重点：mutex + condition_variable + 移动语义
template <typename T>
class ThreadSafeQueue {
public:
    explicit ThreadSafeQueue(size_t max_size = 8) : max_size_(max_size) {}

    // 禁止拷贝，允许移动
    ThreadSafeQueue(const ThreadSafeQueue&) = delete;
    ThreadSafeQueue& operator=(const ThreadSafeQueue&) = delete;

    // 生产者：移动语义入队，队满时阻塞等待
    void push(T&& item) {
        std::unique_lock<std::mutex> lock(mutex_);
        // 背压机制：队列满时阻塞生产者，防止 OOM
        not_full_.wait(lock, [this] { return queue_.size() < max_size_ || stopped_; });
        if (stopped_) return;
        queue_.push(std::move(item));
        not_empty_.notify_one();
    }

    // 消费者：阻塞等待取出元素
    // 返回 std::optional，队列关闭时返回 nullopt
    std::optional<T> wait_and_pop() {
        std::unique_lock<std::mutex> lock(mutex_);
        not_empty_.wait(lock, [this] { return !queue_.empty() || stopped_; });
        if (queue_.empty()) return std::nullopt; // 队列已关闭且为空
        T item = std::move(queue_.front());
        queue_.pop();
        not_full_.notify_one();
        return item;
    }

    // 非阻塞尝试取出
    std::optional<T> try_pop() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.empty()) return std::nullopt;
        T item = std::move(queue_.front());
        queue_.pop();
        not_full_.notify_one();
        return item;
    }

    // 优雅关闭：通知所有等待线程退出
    void stop() {
        std::lock_guard<std::mutex> lock(mutex_);
        stopped_ = true;
        not_empty_.notify_all();
        not_full_.notify_all();
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }

    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }

private:
    std::queue<T> queue_;
    mutable std::mutex mutex_;
    std::condition_variable not_empty_;  // 队列非空时通知消费者
    std::condition_variable not_full_;   // 队列未满时通知生产者
    size_t max_size_;
    bool stopped_ = false;
};

} // namespace rtdet
