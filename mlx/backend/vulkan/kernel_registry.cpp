// Copyright © 2026 MLX Vulkan Backend
#include "mlx/backend/vulkan/kernel_registry.h"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <sstream>
#include <iomanip>

// 嵌入的 SPIR-V shader
#include "shaders/add_spv.h"
#include "shaders/add_bf16_spv.h"
#include "shaders/mul_bf16_spv.h"
#include "shaders/sin_spv.h"
#include "shaders/cos_spv.h"
#include "shaders/qmm_affine_bf16_t4_g128_spv.h"
#include "shaders/rmsnorm_bf16_spv.h"
#include "shaders/rope_bf16_t1_spv.h"
#include "shaders/rope_bf16_freqs_spv.h"
#include "shaders/sdpa_bf16_decode_q1_spv.h"
#include "shaders/sdpa_bf16_decode_q1_d128_spv.h"
#include "shaders/sdpa_bf16_decode_q1_d128_k32_spv.h"
#include "shaders/sdpa_bf16_decode_splitk_stage1_spv.h"
#include "shaders/sdpa_bf16_decode_splitk_reduce_spv.h"
#include "shaders/sdpa_bf16_prefill_q1_spv.h"
#include "shaders/sdpa_bf16_prefill_splitk_stage1_spv.h"
#include "shaders/sdpa_bf16_prefill_splitk_reduce_spv.h"

namespace mlx::core::vulkan {

// 静态 kernel 名称定义
const char* KernelRegistry::ADD_F32 = "add_f32";
const char* KernelRegistry::ADD_F16 = "add_f16";
const char* KernelRegistry::ADD_BF16 = "add_bf16";
const char* KernelRegistry::MUL_F32 = "mul_f32";
const char* KernelRegistry::MUL_BF16 = "mul_bf16";
const char* KernelRegistry::SUB_F32 = "sub_f32";
const char* KernelRegistry::DIV_F32 = "div_f32";
const char* KernelRegistry::SIN_F32 = "sin_f32";
const char* KernelRegistry::COS_F32 = "cos_f32";
const char* KernelRegistry::QMM_AFFINE_BF16_T4_G128 =
    "qmm_affine_bf16_t4_g128";
const char* KernelRegistry::RMSNORM_BF16 = "rmsnorm_bf16";
const char* KernelRegistry::ROPE_BF16_T1 = "rope_bf16_t1";
const char* KernelRegistry::ROPE_BF16_FREQS = "rope_bf16_freqs";
const char* KernelRegistry::SDPA_BF16_DECODE_Q1 = "sdpa_bf16_decode_q1";
const char* KernelRegistry::SDPA_BF16_DECODE_Q1_D128 = "sdpa_bf16_decode_q1_d128";
const char* KernelRegistry::SDPA_BF16_DECODE_Q1_D128_K32 =
    "sdpa_bf16_decode_q1_d128_k32";
const char* KernelRegistry::SDPA_BF16_DECODE_SPLITK_STAGE1 =
    "sdpa_bf16_decode_splitk_stage1";
const char* KernelRegistry::SDPA_BF16_DECODE_SPLITK_REDUCE =
    "sdpa_bf16_decode_splitk_reduce";
const char* KernelRegistry::SDPA_BF16_PREFILL_Q1 = "sdpa_bf16_prefill_q1";
const char* KernelRegistry::SDPA_BF16_PREFILL_SPLITK_STAGE1 =
    "sdpa_bf16_prefill_splitk_stage1";
const char* KernelRegistry::SDPA_BF16_PREFILL_SPLITK_REDUCE =
    "sdpa_bf16_prefill_splitk_reduce";

KernelRegistry& KernelRegistry::instance() {
  static KernelRegistry registry;
  registry.initialize();
  return registry;
}

void KernelRegistry::initialize() {
  // 只初始化一次
  static bool initialized = false;
  if (initialized) return;
  
  // 注册内置 shader
  register_builtin_shaders();
  
  initialized = true;
}

void KernelRegistry::register_builtin_shaders() {
  // 将嵌入的 add_spv 转换为 vector<uint32_t>
  std::vector<uint32_t> add_spirv((add_spv_len + 3) / 4);
  std::memcpy(add_spirv.data(), add_spv, add_spv_len);
  shaders_[ADD_F32] = std::move(add_spirv);

  std::vector<uint32_t> add_bf16_spirv((add_bf16_spv_len + 3) / 4);
  std::memcpy(add_bf16_spirv.data(), add_bf16_spv, add_bf16_spv_len);
  shaders_[ADD_BF16] = std::move(add_bf16_spirv);

  std::vector<uint32_t> mul_bf16_spirv((mul_bf16_spv_len + 3) / 4);
  std::memcpy(mul_bf16_spirv.data(), mul_bf16_spv, mul_bf16_spv_len);
  shaders_[MUL_BF16] = std::move(mul_bf16_spirv);
  
  // 注册 sin_spv
  std::vector<uint32_t> sin_spirv((mlx_backend_vulkan_shaders_sin_spv_len + 3) / 4);
  std::memcpy(sin_spirv.data(), mlx_backend_vulkan_shaders_sin_spv, mlx_backend_vulkan_shaders_sin_spv_len);
  shaders_[SIN_F32] = std::move(sin_spirv);
  
  // 注册 cos_spv
  std::vector<uint32_t> cos_spirv((cos_spv_len + 3) / 4);
  std::memcpy(cos_spirv.data(), cos_spv, cos_spv_len);
  shaders_[COS_F32] = std::move(cos_spirv);

  // 注册 qmm_affine_bf16_t4_g128_spv
  std::vector<uint32_t> qmm_spirv(
      (qmm_affine_bf16_t4_g128_spv_len + 3) / 4);
  std::memcpy(
      qmm_spirv.data(),
      qmm_affine_bf16_t4_g128_spv,
      qmm_affine_bf16_t4_g128_spv_len);
  shaders_[QMM_AFFINE_BF16_T4_G128] = std::move(qmm_spirv);

  std::vector<uint32_t> rmsnorm_spirv((rmsnorm_bf16_spv_len + 3) / 4);
  std::memcpy(rmsnorm_spirv.data(), rmsnorm_bf16_spv, rmsnorm_bf16_spv_len);
  shaders_[RMSNORM_BF16] = std::move(rmsnorm_spirv);

  std::vector<uint32_t> rope_spirv((rope_bf16_t1_spv_len + 3) / 4);
  std::memcpy(rope_spirv.data(), rope_bf16_t1_spv, rope_bf16_t1_spv_len);
  shaders_[ROPE_BF16_T1] = std::move(rope_spirv);

  std::vector<uint32_t> rope_freqs_spirv((rope_bf16_freqs_spv_len + 3) / 4);
  std::memcpy(
      rope_freqs_spirv.data(), rope_bf16_freqs_spv, rope_bf16_freqs_spv_len);
  shaders_[ROPE_BF16_FREQS] = std::move(rope_freqs_spirv);

  std::vector<uint32_t> sdpa_spirv((sdpa_bf16_decode_q1_spv_len + 3) / 4);
  std::memcpy(
      sdpa_spirv.data(),
      sdpa_bf16_decode_q1_spv,
      sdpa_bf16_decode_q1_spv_len);
  shaders_[SDPA_BF16_DECODE_Q1] = std::move(sdpa_spirv);

  std::vector<uint32_t> sdpa_d128_spirv((sdpa_bf16_decode_q1_d128_spv_len + 3) / 4);
  std::memcpy(
      sdpa_d128_spirv.data(),
      sdpa_bf16_decode_q1_d128_spv,
      sdpa_bf16_decode_q1_d128_spv_len);
  shaders_[SDPA_BF16_DECODE_Q1_D128] = std::move(sdpa_d128_spirv);

  std::vector<uint32_t> sdpa_d128_k32_spirv(
      (sdpa_bf16_decode_q1_d128_k32_spv_len + 3) / 4);
  std::memcpy(
      sdpa_d128_k32_spirv.data(),
      sdpa_bf16_decode_q1_d128_k32_spv,
      sdpa_bf16_decode_q1_d128_k32_spv_len);
  shaders_[SDPA_BF16_DECODE_Q1_D128_K32] = std::move(sdpa_d128_k32_spirv);

  std::vector<uint32_t> sdpa_prefill_spirv((sdpa_bf16_prefill_q1_spv_len + 3) / 4);
  std::memcpy(
      sdpa_prefill_spirv.data(),
      sdpa_bf16_prefill_q1_spv,
      sdpa_bf16_prefill_q1_spv_len);
  shaders_[SDPA_BF16_PREFILL_Q1] = std::move(sdpa_prefill_spirv);

  std::vector<uint32_t> sdpa_splitk_stage1_spirv(
      (sdpa_bf16_decode_splitk_stage1_spv_len + 3) / 4);
  std::memcpy(
      sdpa_splitk_stage1_spirv.data(),
      sdpa_bf16_decode_splitk_stage1_spv,
      sdpa_bf16_decode_splitk_stage1_spv_len);
  shaders_[SDPA_BF16_DECODE_SPLITK_STAGE1] = std::move(sdpa_splitk_stage1_spirv);
  std::vector<uint32_t> sdpa_prefill_splitk_stage1_spirv(
      (sdpa_bf16_prefill_splitk_stage1_spv_len + 3) / 4);
  std::memcpy(
      sdpa_prefill_splitk_stage1_spirv.data(),
      sdpa_bf16_prefill_splitk_stage1_spv,
      sdpa_bf16_prefill_splitk_stage1_spv_len);
  shaders_[SDPA_BF16_PREFILL_SPLITK_STAGE1] =
      std::move(sdpa_prefill_splitk_stage1_spirv);

  std::vector<uint32_t> sdpa_splitk_reduce_spirv(
      (sdpa_bf16_decode_splitk_reduce_spv_len + 3) / 4);
  std::memcpy(
      sdpa_splitk_reduce_spirv.data(),
      sdpa_bf16_decode_splitk_reduce_spv,
      sdpa_bf16_decode_splitk_reduce_spv_len);
  shaders_[SDPA_BF16_DECODE_SPLITK_REDUCE] = std::move(sdpa_splitk_reduce_spirv);
  std::vector<uint32_t> sdpa_prefill_splitk_reduce_spirv(
      (sdpa_bf16_prefill_splitk_reduce_spv_len + 3) / 4);
  std::memcpy(
      sdpa_prefill_splitk_reduce_spirv.data(),
      sdpa_bf16_prefill_splitk_reduce_spv,
      sdpa_bf16_prefill_splitk_reduce_spv_len);
  shaders_[SDPA_BF16_PREFILL_SPLITK_REDUCE] =
      std::move(sdpa_prefill_splitk_reduce_spirv);
}

const std::vector<uint32_t>& KernelRegistry::get_shader(const std::string& name) {
  auto it = shaders_.find(name);
  if (it == shaders_.end()) {
    throw std::runtime_error("Shader not found: " + name);
  }
  return it->second;
}

std::shared_ptr<kp::Algorithm> KernelRegistry::get_algorithm(
    const std::string& kernel_name,
    kp::Manager& manager,
    const std::vector<std::shared_ptr<kp::Tensor>>& params,
    const kp::Workgroup& workgroup,
    const std::vector<uint32_t>& push_consts) {
  
  // 构建 cache key
  std::stringstream cache_key_ss;
  cache_key_ss
      << build_algorithm_key(kernel_name, params.size(), workgroup, push_consts);
  // Algorithm objects capture descriptor sets bound to specific Tensor objects.
  // Include Tensor identity in cache key to avoid reusing pipelines with stale
  // buffer bindings across different arrays.
  for (const auto& tensor : params) {
    cache_key_ss << "_t" << std::hex
                 << reinterpret_cast<std::uintptr_t>(tensor.get());
  }
  std::string cache_key = cache_key_ss.str();
  
  {
    std::lock_guard<std::mutex> lock(algorithms_mutex_);
    
    // 检查缓存
    auto it = algorithms_.find(cache_key);
    if (it != algorithms_.end()) {
      if (auto algo = it->second.lock()) {
        return algo;
      }
      // 已过期，从缓存中移除
      algorithms_.erase(it);
    }
  }
  
  // 获取 shader
  const auto& spirv = get_shader(kernel_name);
  
  // 创建新的 Algorithm
  std::shared_ptr<kp::Algorithm> algo;
  if (push_consts.empty()) {
    algo = manager.algorithm(params, spirv, workgroup);
  } else {
    algo = manager.algorithm(params, spirv, workgroup, std::vector<float>{}, push_consts);
  }
  
  // 缓存 algorithm
  {
    std::lock_guard<std::mutex> lock(algorithms_mutex_);
    algorithms_[cache_key] = algo;
  }
  
  return algo;
}

void KernelRegistry::clear_cache() {
  std::lock_guard<std::mutex> lock(algorithms_mutex_);
  algorithms_.clear();
}

std::vector<uint32_t> KernelRegistry::load_spirv_from_file(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Cannot open SPIR-V file: " + path);
  }
  
  // 获取文件大小
  file.seekg(0, std::ios::end);
  size_t size = file.tellg();
  file.seekg(0, std::ios::beg);
  
  // 读取数据
  std::vector<char> buffer(size);
  file.read(buffer.data(), size);
  
  // 转换为 uint32_t 数组
  std::vector<uint32_t> spirv(size / 4);
  std::memcpy(spirv.data(), buffer.data(), size);
  
  return spirv;
}

// ============================================================================
// Helper Functions
// ============================================================================

std::string build_algorithm_key(
    const std::string& kernel_name,
    size_t num_params,
    const kp::Workgroup& workgroup,
    const std::vector<uint32_t>& push_consts) {
  std::stringstream ss;
  ss << kernel_name << "_" << num_params << "_"
     << workgroup[0] << "_" << workgroup[1] << "_" << workgroup[2];
  for (uint32_t u : push_consts) {
    ss << "_" << std::hex << std::setw(8) << std::setfill('0') << u;
  }
  return ss.str();
}

} // namespace mlx::core::vulkan
