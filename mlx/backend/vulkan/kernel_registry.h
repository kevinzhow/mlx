// Copyright © 2026 MLX Vulkan Backend
#pragma once

#include <kompute/Kompute.hpp>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace mlx::core::vulkan {

/**
 * KernelRegistry - 管理 SPIR-V shader 的加载和缓存
 * 
 * 单例模式，负责:
 * 1. 加载和缓存 SPIR-V shader
 * 2. 创建和管理 kp::Algorithm 实例
 * 3. 管理 descriptor set 和 pipeline cache
 */
class KernelRegistry {
 public:
  // 获取单例实例
  static KernelRegistry& instance();
  
  // 禁止拷贝
  KernelRegistry(const KernelRegistry&) = delete;
  KernelRegistry& operator=(const KernelRegistry&) = delete;
  
  // 初始化 shader 库
  void initialize();
  
  // 获取 shader 的 SPIR-V 代码
  const std::vector<uint32_t>& get_shader(const std::string& name);
  
  // 获取或创建 Algorithm
  // params: 输入输出 buffer 列表
  // workgroup: 工作组大小
  // push_consts: push constant 数据 (可选)
  std::shared_ptr<kp::Algorithm> get_algorithm(
      const std::string& kernel_name,
      kp::Manager& manager,
      const std::vector<std::shared_ptr<kp::Tensor>>& params,
      const kp::Workgroup& workgroup = kp::Workgroup({256, 1, 1}),
      const std::vector<uint32_t>& push_consts = {});
  
  // 清理缓存
  void clear_cache();
  
  // 内置 kernel 名称
  static const char* ADD_F32;
  static const char* ADD_F16;
  static const char* ADD_BF16;
  static const char* MUL_F32;
  static const char* MUL_BF16;
  static const char* SILU_MUL_BF16;
  static const char* SUB_F32;
  static const char* SUB_F32_SCALAR;
  static const char* SUB_BF16;
  static const char* SUB_BF16_SCALAR;
  static const char* DIV_F32;
  static const char* SIN_F32;
  static const char* COS_F32;
  static const char* ARGMAX_F32_LASTDIM;
  static const char* ARGMAX_BF16_LASTDIM;
  static const char* LOGSUMEXP_F32;
  static const char* LOGSUMEXP_BF16_ROW1;
  static const char* QMM_AFFINE_BF16_T4_G128;
  static const char* QMM_AFFINE_BF16_T4_G128_M1;
  static const char* QMM_AFFINE_BF16_T4_G128_M2;
  static const char* QMM_AFFINE_BF16_T4_G128_M4;
  static const char* RMSNORM_BF16;
  static const char* ROPE_BF16_T1;
  static const char* ROPE_BF16_FREQS;
  static const char* SDPA_BF16_DECODE_Q1;
  static const char* SDPA_BF16_DECODE_Q1_D128;
  static const char* SDPA_BF16_DECODE_Q1_D128_K32;
  static const char* SDPA_BF16_DECODE_Q1_D128_K64;
  static const char* SDPA_BF16_DECODE_Q1_D128_K128;
  static const char* SDPA_BF16_DECODE_SPLITK_STAGE1;
  static const char* SDPA_BF16_DECODE_SPLITK_REDUCE;
  static const char* SDPA_BF16_DECODE_SPLITK_REDUCE_SUBGROUP;
  static const char* SDPA_BF16_DECODE_SPLITK_REDUCE_L32;
  static const char* SDPA_BF16_PREFILL_Q1;
  static const char* SDPA_BF16_PREFILL_SPLITK_STAGE1;
  static const char* SDPA_BF16_PREFILL_SPLITK_REDUCE;
  
 private:
  KernelRegistry() = default;
  
  // 从文件加载 SPIR-V
  std::vector<uint32_t> load_spirv_from_file(const std::string& path);
  
  // 从内存加载 SPIR-V (用于嵌入)
  void register_builtin_shaders();
  
  // Shader 缓存
  std::unordered_map<std::string, std::vector<uint32_t>> shaders_;
  
  // Algorithm 缓存 (key: kernel_name + params signature)
  std::unordered_map<std::string, std::weak_ptr<kp::Algorithm>> algorithms_;
  std::mutex algorithms_mutex_;
};

// 辅助函数: 构建 algorithm cache key
std::string build_algorithm_key(
    const std::string& kernel_name,
    size_t num_params,
    const kp::Workgroup& workgroup,
    const std::vector<uint32_t>& push_consts = {});

} // namespace mlx::core::vulkan
