// Copyright © 2026 MLX Vulkan Backend
// Buffer implementation using Kompute

#include "mlx/backend/vulkan/buffer.h"
#include "mlx/allocator.h"
#include <cstring>

namespace mlx::core::vulkan {

namespace {

kp::Tensor::TensorDataTypes to_kompute_dtype(Dtype dtype) {
  switch (dtype) {
    case bool_:
      return kp::Tensor::TensorDataTypes::eBool;
    case uint8:
    case uint16:
    case uint32:
    case uint64:
      return kp::Tensor::TensorDataTypes::eUnsignedInt;
    case int8:
    case int16:
    case int32:
    case int64:
      return kp::Tensor::TensorDataTypes::eInt;
    case float16:
    case float32:
    case bfloat16:
    case complex64:
      return kp::Tensor::TensorDataTypes::eFloat;
    case float64:
      return kp::Tensor::TensorDataTypes::eDouble;
  }
  return kp::Tensor::TensorDataTypes::eFloat;
}

} // namespace

// ============================================================================
// Buffer Implementation
// ============================================================================

std::shared_ptr<Buffer> Buffer::from_array(
    const array& arr,
    std::shared_ptr<kp::Manager> manager) {
  
  if (!manager) {
    throw std::runtime_error("Manager is null in Buffer::from_array");
  }

  auto& mutable_arr = const_cast<array&>(arr);
  if (!mutable_arr.data_shared_ptr()) {
    mutable_arr.set_data(allocator::malloc(mutable_arr.nbytes()));
  }

  auto tensor = manager->tensor(
      const_cast<void*>(arr.data<void>()),
      static_cast<uint32_t>(arr.size()),
      static_cast<uint32_t>(arr.itemsize()),
      to_kompute_dtype(arr.dtype()));
  
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
    mapped_ptr_ = tensor_->rawData();
  }
}

Buffer::~Buffer() {
  // Tensor cleanup is handled by shared_ptr
  tensor_.reset();
}

void Buffer::upload(const void* data, size_t size, size_t offset) {
  if (!tensor_) return;
  
  // Copy data to tensor
  auto* tensor_data = static_cast<char*>(tensor_->rawData());
  if (tensor_data) {
    std::memcpy(tensor_data + offset, data, size);
    dirty_ = true;
  }
}

void Buffer::download(void* data, size_t size, size_t offset) {
  if (!tensor_) return;
  
  // Copy data from tensor
  auto* tensor_data = static_cast<char*>(tensor_->rawData());
  if (tensor_data) {
    std::memcpy(data, tensor_data + offset, size);
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
  mapped_ptr_ = tensor_->rawData();
}

float* Buffer::data() {
  if (tensor_) {
    return static_cast<float*>(tensor_->rawData());
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
  
  // Create new buffer from array. 
  // Caching by raw data pointer is unsafe because MLX reuses memory addresses.
  return Buffer::from_array(arr, manager_);
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
