// Copyright © 2026 MLX Vulkan Backend

#include "mlx/backend/gpu/eval.h"
#include "mlx/fence.h"
#include "mlx/scheduler.h"

#include <condition_variable>
#include <mutex>

namespace mlx::core {

namespace {

struct FenceImpl {
  uint32_t count{0};
  uint32_t value{0};
  std::mutex mtx;
  std::condition_variable cv;
};

} // namespace

Fence::Fence(Stream) {
  auto dtor = [](void* ptr) { delete static_cast<FenceImpl*>(ptr); };
  fence_ = std::shared_ptr<void>(new FenceImpl{}, dtor);
}

void Fence::wait(Stream stream, const array&) {
  auto* f = static_cast<FenceImpl*>(fence_.get());
  uint32_t target = 0;
  {
    std::lock_guard<std::mutex> lk(f->mtx);
    target = f->count;
  }

  if (stream.device == Device::cpu) {
    scheduler::enqueue(stream, [fence_ = fence_, target]() mutable {
      auto* f = static_cast<FenceImpl*>(fence_.get());
      std::unique_lock<std::mutex> lk(f->mtx);
      if (f->value >= target) {
        return;
      }
      f->cv.wait(lk, [f, target] { return f->value >= target; });
    });
    return;
  }

  std::unique_lock<std::mutex> lk(f->mtx);
  if (f->value >= target) {
    return;
  }
  f->cv.wait(lk, [f, target] { return f->value >= target; });
}

void Fence::update(Stream stream, const array&, bool) {
  auto* f = static_cast<FenceImpl*>(fence_.get());
  uint32_t target = 0;
  {
    std::lock_guard<std::mutex> lk(f->mtx);
    f->count++;
    target = f->count;
  }

  if (stream.device == Device::cpu) {
    scheduler::enqueue(stream, [fence_ = fence_, target]() mutable {
      auto* f = static_cast<FenceImpl*>(fence_.get());
      {
        std::lock_guard<std::mutex> lk(f->mtx);
        f->value = target;
      }
      f->cv.notify_all();
    });
    return;
  }

  gpu::finalize(stream);
  gpu::synchronize(stream);
  {
    std::lock_guard<std::mutex> lk(f->mtx);
    f->value = target;
  }
  f->cv.notify_all();
}

} // namespace mlx::core
