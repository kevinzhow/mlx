// Copyright © 2026 MLX Vulkan Backend
#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>

namespace mlx::core::vulkan {

// Runtime switch: set MLX_VK_PROFILE=1 to enable op aggregation.
bool profile_enabled();

class OpProfileScope {
 public:
  explicit OpProfileScope(const char* op_name);
  ~OpProfileScope();

  void mark_fallback();
  void mark_sync(uint64_t count = 1);
  void add_copy_bytes(size_t bytes);

 private:
  const char* op_name_;
  bool enabled_;
  bool fallback_{false};
  uint64_t sync_count_{0};
  uint64_t copy_bytes_{0};
  std::chrono::steady_clock::time_point start_time_;
};

} // namespace mlx::core::vulkan
