// Copyright © 2026 MLX Vulkan Backend

#include "mlx/backend/gpu/eval.h"
#include "mlx/event.h"
#include "mlx/scheduler.h"

#include <condition_variable>
#include <mutex>

namespace mlx::core {

namespace {

struct EventCounter {
  uint64_t value{0};
  std::mutex mtx;
  std::condition_variable cv;
};

} // namespace

Event::Event(Stream stream) : stream_(stream) {
  auto dtor = [](void* ptr) { delete static_cast<EventCounter*>(ptr); };
  event_ = std::shared_ptr<void>(new EventCounter{}, dtor);
}

void Event::wait() {
  auto* ec = static_cast<EventCounter*>(event_.get());
  std::unique_lock<std::mutex> lk(ec->mtx);
  if (ec->value >= value()) {
    return;
  }
  ec->cv.wait(lk, [this, ec] { return ec->value >= value(); });
}

void Event::wait(Stream stream) {
  if (stream.device == Device::cpu) {
    scheduler::enqueue(stream, [*this]() mutable { wait(); });
    return;
  }
  wait();
}

void Event::signal(Stream stream) {
  if (stream.device == Device::cpu) {
    scheduler::enqueue(stream, [*this]() mutable {
      auto* ec = static_cast<EventCounter*>(event_.get());
      {
        std::lock_guard<std::mutex> lk(ec->mtx);
        ec->value = value();
      }
      ec->cv.notify_all();
    });
    return;
  }

  gpu::finalize(stream);
  gpu::synchronize(stream);
  auto* ec = static_cast<EventCounter*>(event_.get());
  {
    std::lock_guard<std::mutex> lk(ec->mtx);
    ec->value = value();
  }
  ec->cv.notify_all();
}

bool Event::is_signaled() const {
  auto* ec = static_cast<EventCounter*>(event_.get());
  std::lock_guard<std::mutex> lk(ec->mtx);
  return ec->value >= value();
}

} // namespace mlx::core
