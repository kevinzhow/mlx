// Copyright © 2026 MLX Vulkan Backend
// Buffer implementation using Kompute

#include "mlx/backend/vulkan/buffer.h"
#include <cstring>
#include <iostream>

namespace mlx::core::vulkan {

// ============================================================================
// Buffer Implementation
// ============================================================================

std::shared_ptr<Buffer> Buffer::from_array(
    const array& arr,
    std::shared_ptr<kp::Manager> manager) {
  
  if (!manager) {
    throw std::runtime_error("Manager is null in Buffer::from_array");
  }
  
  // Create tensor from array data
  // Note: This assumes the array data is float32 for now
  size_t num_elements = arr.size();
  
  // Create tensor with initial data
  const float* data = arr.data<float>();
  std::vector<float> vec(data, data + num_elements);
  
  auto tensor = manager->tensor(vec);
  
  return std::shared_ptr<Buffer>(new Buffer(tensor, arr.nbytes()));
}

std::shared_ptr<Buffer> Buffer::empty(
    size_t size,
    std::shared_ptr<kp::Manager> manager) {
  
  if (!manager) {
    throw std::runtime_error("Manager is null in Buffer::empty");
  }
  
  // Create tensor with zeros
  size_t num_floats = size / sizeof(float);
  std::vector<float> zeros(num_floats, 0.0f);
  
  auto tensor = manager->tensor(zeros);
  
  return std::shared_ptr<Buffer>(new Buffer(tensor, size));
}

Buffer::Buffer(std::shared_ptr<kp::Tensor> tensor, size_t size)
    : tensor_(tensor),
      size_(size),
      mapped_ptr_(nullptr),
      is_unified_(true),  // Kompute uses unified memory by default
      dirty_(false) {
  
  if (tensor_) {
    // Get raw data pointer from tensor
    // Note: This is only valid after sync_to_host
    mapped_ptr_ = tensor_->data<float>();
  }
}

Buffer::~Buffer() {
  // Tensor cleanup is handled by shared_ptr
  tensor_.reset();
}

void Buffer::upload(const void* data, size_t size, size_t offset) {
  if (!tensor_) return;
  
  // Copy data to tensor
  float* tensor_data = tensor_->data<float>();
  if (tensor_data) {
    std::memcpy(tensor_data + offset / sizeof(float), data, size);
    dirty_ = true;
  }
}

void Buffer::download(void* data, size_t size, size_t offset) {
  if (!tensor_) return;
  
  // Copy data from tensor
  float* tensor_data = tensor_->data<float>();
  if (tensor_data) {
    std::memcpy(data, tensor_data + offset / sizeof(float), size);
  }
}

void Buffer::sync_to_device(std::shared_ptr<kp::Sequence> seq) {
  if (!tensor_) return;
  
  if (seq) {
    seq->record<kp::OpTensorSyncDevice>({tensor_});
  } else {
    // Create temporary sequence
    auto manager = BufferManager::instance().manager();
    if (manager) {
      auto tmp_seq = manager->sequence();
      tmp_seq->record<kp::OpTensorSyncDevice>({tensor_});
      tmp_seq->eval();
    }
  }
  dirty_ = false;
}

void Buffer::sync_to_host(std::shared_ptr<kp::Sequence> seq) {
  if (!tensor_) return;
  
  if (seq) {
    seq->record<kp::OpTensorSyncLocal>({tensor_});
  } else {
    // Create temporary sequence
    auto manager = BufferManager::instance().manager();
    if (manager) {
      auto tmp_seq = manager->sequence();
      tmp_seq->record<kp::OpTensorSyncLocal>({tensor_});
      tmp_seq->eval();
    }
  }
  // Update mapped_ptr after sync
  mapped_ptr_ = tensor_->data<float>();
}

float* Buffer::data() {
  if (tensor_) {
    return tensor_->data<float>();
  }
  return nullptr;
}

// ============================================================================
// BufferManager Implementation
// ============================================================================

BufferManager& BufferManager::instance() {
  static BufferManager instance;
  return instance;
}

void BufferManager::initialize(std::shared_ptr<kp::Manager> manager) {
  manager_ = manager;
}

void BufferManager::shutdown() {
  array_buffers_.clear();
  tensor_buffers_.clear();
  manager_.reset();
}

std::shared_ptr<Buffer> BufferManager::get_buffer(const array& arr) {
  std::lock_guard<std::mutex> lock(mutex_);
  
  auto it = array_buffers_.find(arr.data<void>());
  if (it != array_buffers_.end()) {
    if (auto buffer = it->second.lock()) {
      return buffer;
    }
  }
  
  // Create new buffer from array
  auto buffer = Buffer::from_array(arr, manager_);
  if (buffer) {
    array_buffers_[arr.data<void>()] = buffer;
    tensor_buffers_[buffer->tensor()] = buffer;
  }
  
  return buffer;
}

std::shared_ptr<Buffer> BufferManager::create_temp_buffer(size_t size) {
  return Buffer::empty(size, manager_);
}

// ============================================================================
// Helper Functions
// ============================================================================

bool supports_unified_memory(std::shared_ptr<kp::Manager> manager) {
  // Kompute typically uses unified memory (host-visible device memory)
  // unless configured otherwise
  if (!manager) {
    return false;
  }
  
  // Try to create a small tensor to test
  try {
    auto tensor = manager->tensor(std::vector<float>{1.0f});
    // Check if tensor is valid by checking if it's initialized
    return tensor && tensor->isInit();
  } catch (...) {
    return false;
  }
}

} // namespace mlx::core::vulkan
