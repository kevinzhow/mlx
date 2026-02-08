// Copyright © 2026 MLX Vulkan Backend
// Buffer management using Kompute - Aligned with Metal's memory model

#pragma once

#include <kompute/Kompute.hpp>
#include <mlx/array.h>
#include <mutex>
#include <unordered_map>
#include <memory>

namespace mlx::core::vulkan {

// Forward declarations
class Device;

// ============================================================================
// Buffer - Wrapper around kp::Tensor for MLX integration
// ============================================================================

class Buffer {
 public:
  // Create buffer from existing array data
  static std::shared_ptr<Buffer> from_array(
      const array& arr,
      std::shared_ptr<kp::Manager> manager);

  // Create empty buffer with given size
  static std::shared_ptr<Buffer> empty(
      size_t size,
      std::shared_ptr<kp::Manager> manager);

  // Constructor/destructor
  Buffer(std::shared_ptr<kp::Tensor> tensor, size_t size);
  ~Buffer();

  // Disable copy
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;

  // Accessors
  std::shared_ptr<kp::Tensor> tensor() const { return tensor_; }
  size_t size() const { return size_; }
  void* mapped_ptr() { return mapped_ptr_; }
  bool is_unified() const { return is_unified_; }

  // Data transfer
  void upload(const void* data, size_t size, size_t offset = 0);
  void download(void* data, size_t size, size_t offset = 0);

  // Sync operations
  void sync_to_device(std::shared_ptr<kp::Sequence> seq = nullptr);
  void sync_to_host(std::shared_ptr<kp::Sequence> seq = nullptr);
  void mark_dirty() { dirty_ = true; }
  bool is_dirty() const { return dirty_; }

  // Get raw data pointer (after sync_to_host)
  float* data();

 private:
  std::shared_ptr<kp::Tensor> tensor_;
  size_t size_;
  void* mapped_ptr_;
  bool is_unified_;
  bool dirty_;
};

// ============================================================================
// BufferManager - Global buffer management
// ============================================================================

class BufferManager {
 public:
  static BufferManager& instance();

  void initialize(std::shared_ptr<kp::Manager> manager);
  void shutdown();

  // Get or create buffer for array
  std::shared_ptr<Buffer> get_buffer(const array& arr);
  
  // Create temporary buffer
  std::shared_ptr<Buffer> create_temp_buffer(size_t size);

  // Get Kompute manager
  std::shared_ptr<kp::Manager> manager() { return manager_; }

 private:
  BufferManager() = default;
  
  std::shared_ptr<kp::Manager> manager_;
  std::mutex mutex_;
  std::unordered_map<const void*, std::weak_ptr<Buffer>> array_buffers_;
  std::unordered_map<std::shared_ptr<kp::Tensor>, std::shared_ptr<Buffer>> tensor_buffers_;
};

// ============================================================================
// Helper functions
// ============================================================================

// Check if device supports unified memory
bool supports_unified_memory(std::shared_ptr<kp::Manager> manager);

} // namespace mlx::core::vulkan
