# MLX Vulkan Backend 架构设计文档

## 1. 架构概览

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Python Layer                                    │
│  import mlx.core as mx                                                      │
│  mx.set_default_device(mx.gpu)  ─────────────────────────────────────────┐  │
│  a = mx.array([1.0, 2.0, 3.0])                                           │  │
│  b = a + b  # 触发 Lazy Evaluation                                       │  │
│  print(b)   # 触发 eval()                                                │  │
└──────────────────────────────────────────────────────────────────────────┼──┘
                                                                           │
┌──────────────────────────────────────────────────────────────────────────┼──┐
│                           MLX Core (C++)                                │  │
│                                                                          │  │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐      │  │
│  │   Device         │  │   Stream         │  │   Array          │      │  │
│  │   (device.h)     │  │   (stream.h)     │  │   (array.h)      │      │  │
│  └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘      │  │
│           │                     │                     │                │  │
│           ▼                     ▼                     ▼                │  │
│  ┌─────────────────────────────────────────────────────────────────┐   │  │
│  │                    Scheduler (scheduler.h)                       │   │  │
│  │                                                                  │   │  │
│  │  new_stream(Device) ────────► gpu::new_stream() ──────────────┐ │   │  │
│  │  eval(array) ───────────────► gpu::eval() ────────────────────┼─┼───┘  │
│  │  synchronize(Stream) ───────► gpu::synchronize() ─────────────┘ │      │
│  │                                                                  │      │
│  └─────────────────────────────────────────────────────────────────┘      │
│                                    │                                       │
└────────────────────────────────────┼───────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           GPU Backend Interface                              │
│                                                                              │
│  namespace mlx::core::gpu {                                                  │
│      bool is_available();           ───► 检测设备可用                        │
│      int device_count();            ───► 获取设备数量                        │
│      void new_stream(Stream);       ───► 创建 Stream                         │
│      void eval(array&);             ───► 执行计算                            │
│      void finalize(Stream);         ───► 完成 Stream                         │
│      void synchronize(Stream);      ───► 同步 Stream                         │
│  }                                                                           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Vulkan Backend Implementation                        │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    VulkanDevice (device.h/cpp)                       │    │
│  │  ┌─────────────────────────────────────────────────────────────┐    │    │
│  │  │  kp::Manager* manager_                                       │    │    │
│  │  │  unordered_map<int, unique_ptr<VulkanStream>> streams_      │    │    │
│  │  │                                                              │    │    │
│  │  │  static VulkanDevice& instance()                             │    │    │
│  │  │  void new_queue(int index) ────────► manager->sequence()    │    │    │
│  │  │  VulkanStream& get_stream(int index)                        │    │    │
│  │  │  void synchronize()                                          │    │    │
│  │  └─────────────────────────────────────────────────────────────┘    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    VulkanStream (device.h/cpp)                       │    │
│  │  ┌─────────────────────────────────────────────────────────────┐    │    │
│  │  │  shared_ptr<kp::Sequence> sequence_                          │    │    │
│  │  │  vector<array> temporaries_                                  │    │    │
│  │  │  bool async_pending_                                         │    │    │
│  │  │                                                              │    │    │
│  │  │  void record(Operation*)     ───► Lazy Evaluation 核心      │    │    │
│  │  │  void eval()                 ───► 执行序列                  │    │    │
│  │  │  void eval_async()           ───► 异步执行                  │    │    │
│  │  │  void synchronize()          ───► 等待完成                  │    │    │
│  │  └─────────────────────────────────────────────────────────────┘    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    VulkanBuffer (buffer.h/cpp)                       │    │
│  │  ┌─────────────────────────────────────────────────────────────┐    │    │
│  │  │  shared_ptr<kp::Tensor> tensor_                              │    │    │
│  │  │  void* host_ptr (可选，用于 unified memory)                  │    │    │
│  │  │                                                              │    │    │
│  │  │  static VulkanBuffer from_array(const array& a)             │    │    │
│  │  │  void to_array(array& a)      ───► 复制回 host               │    │    │
│  │  │  void sync_to_device()        ───► Host → GPU                │    │    │
│  │  │  void sync_to_host()          ───► GPU → Host                │    │    │
│  │  └─────────────────────────────────────────────────────────────┘    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Kompute Framework                               │
│                                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │   Manager    │  │   Sequence   │  │   Tensor     │  │  Algorithm   │    │
│  │              │  │              │  │              │  │              │    │
│  │  - Device    │  │  - Commands  │  │  - Buffer    │  │  - Shader    │    │
│  │  - Queues    │  │  - Recording │  │  - Memory    │  │  - Pipeline  │    │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Vulkan Driver                                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 2. 关键接口定义

### 2.1 GPU Backend Interface

```cpp
// mlx/backend/gpu/eval.h
namespace mlx::core::gpu {

// 设备管理
bool is_available();
int device_count();
const unordered_map<string, variant<string, size_t>>& device_info(int device_index);

// Stream 管理
void new_stream(Stream stream);      // Scheduler 创建 stream 时调用
void eval(array& arr);               // 执行 array 的计算
void finalize(Stream s);             // 完成 stream 中所有操作
void synchronize(Stream s);          // 同步 stream

} // namespace mlx::core::gpu
```

### 2.2 Vulkan Backend Classes

```cpp
// mlx/backend/vulkan/device.h
namespace mlx::core::vulkan {

class VulkanDevice {
public:
    static VulkanDevice& instance();
    
    // Stream 管理
    void new_queue(int index);                    // 对应 gpu::new_stream
    VulkanStream& get_stream(int index);
    void synchronize();                            // 对应 gpu::synchronize
    
    // Buffer 管理
    shared_ptr<kp::Tensor> create_tensor(size_t size, size_t itemsize);
    
    // 底层访问
    kp::Manager& manager() { return *manager_; }
    
private:
    unique_ptr<kp::Manager> manager_;
    unordered_map<int, unique_ptr<VulkanStream>> streams_;
};

class VulkanStream {
public:
    explicit VulkanStream(kp::Manager& manager);
    
    // Lazy Evaluation 核心
    void record_begin();                           // 开始录制
    void record_end();                             // 结束录制
    void eval();                                   // 执行录制的操作
    void synchronize();                            // 等待完成
    
    // 操作录制
    template<typename Op, typename... Args>
    void record_op(Args&&... args);
    
private:
    kp::Manager& manager_;
    shared_ptr<kp::Sequence> sequence_;
    bool is_recording_ = false;
};

} // namespace mlx::core::vulkan
```

### 2.3 占位符算子接口

```cpp
// mlx/backend/vulkan/primitives/placeholder.h
namespace mlx::core {

// 最简单的占位符算子 - 用于验证链路
class VulkanPlaceholder : public UnaryPrimitive {
public:
    explicit VulkanPlaceholder(Stream stream) : UnaryPrimitive(stream) {}
    
    void eval_cpu(const vector<array>& inputs, array& out) override;
    void eval_gpu(const vector<array>& inputs, array& out) override;
    
    // 打印调试信息
    static void log(const string& stage, const array& input);
};

} // namespace mlx::core
```

## 3. 调用流程

### 3.1 Stream 创建流程

```
Python: mx.set_default_device(mx.gpu)
    │
    ▼
C++:    set_default_device(Device::gpu)
    │
    ▼
Scheduler::Scheduler() 构造函数
    │
    ├─► is_available(Device::gpu)  ───────► gpu::is_available()
    │                                          │
    │                                          ▼
    │                                  VulkanDevice::is_available()
    │                                          │
    │                                          ▼
    │                                  kp::Manager 测试创建
    │
    ▼
Scheduler::new_stream(Device::gpu)
    │
    ├─► streams_.emplace_back(...)  创建 Stream 对象
    │
    ├─► threads_.push_back(nullptr)  GPU stream 不需要 CPU thread
    │
    └─► gpu::new_stream(stream)  ─────────► VulkanDevice::new_queue(index)
                                                │
                                                ▼
                                        streams_[index] = make_unique<VulkanStream>(manager)
                                                │
                                                ▼
                                        sequence_ = manager.sequence()
```

### 3.2 Lazy Evaluation 执行流程

```
Python: c = a + b
    │
    ├─► 构建计算图，不执行
    │
    ▼
Python: print(c)  # 需要结果
    │
    ├─► c.eval() 触发
    │
    ▼
C++:    array::eval()
    │
    ├─► 遍历计算图，找到需要执行的节点
    │
    ▼
Scheduler::enqueue(stream, task)
    │
    ├─► 对于 GPU stream: 直接调用 task() (同步执行)
    │
    ▼
task():
    ├─► gpu::eval(arr)  ───────────────────► Vulkan 执行逻辑
    │                                          │
    │                                          ▼
    │                                  VulkanDevice& device = VulkanDevice::instance()
    │                                          │
    │                                          ▼
    │                                  VulkanStream& stream = device.get_stream(s.index)
    │                                          │
    │                                          ▼
    │                                  stream.record_begin()
    │                                          │
    │                                          ▼
    │                                  arr.primitive().eval_gpu(...)
    │                                          │
    │                                          ▼
    │                                  Add::eval_gpu() / Placeholder::eval_gpu()
    │                                          │
    │                                          ▼
    │                                  stream.record_end()
    │                                          │
    │                                          ▼
    │                                  stream.eval()
    │                                          │
    │                                          ▼
    │                                  sequence_->eval()
```

## 4. 与 MLX 集成的关键代码点

### 4.1 文件修改清单

| 文件 | 修改内容 | 目的 |
|-----|---------|------|
| `CMakeLists.txt` | 添加 `MLX_BUILD_VULKAN` 选项 | 控制是否构建 Vulkan backend |
| `mlx/CMakeLists.txt` | 条件包含 `backend/vulkan` | 链接 Vulkan 子目录 |
| `mlx/device.cpp` | `is_available()` 等路由到 `gpu::` | Device 接口统一 |
| `mlx/scheduler.cpp` | 使用 `gpu::new_stream/synchronize` | Scheduler 集成 |

### 4.2 现有 GPU 集成点

```cpp
// mlx/device.cpp
bool is_available(const Device& d) {
  switch (d.type) {
    case Device::cpu:
      return cpu::is_available() && (d.index < cpu::device_count());
    case Device::gpu:
      return gpu::is_available() && (d.index < gpu::device_count());  // ✅ 已实现
  }
}

// mlx/scheduler.cpp
Stream new_stream(Device d) {
  if (!gpu::is_available() && d == Device::gpu) {
    throw invalid_argument("...");
  }
  return scheduler::scheduler().new_stream(d);
}

// mlx/scheduler.h - Scheduler::new_stream
Stream new_stream(const Device& d) {
  streams_.emplace_back(streams_.size(), d);
  if (d == Device::gpu) {
    threads_.push_back(nullptr);        // GPU stream 不需要 thread
    gpu::new_stream(streams_.back());   // ✅ 需要实现
  } else {
    threads_.push_back(new StreamThread{});
  }
  return streams_.back();
}
```

## 5. 验证链路的设计

### 5.1 占位符算子

```cpp
// 最简单的验证方式 - Copy/Passthrough
void Placeholder::eval_gpu(const vector<array>& inputs, array& out) {
    log("BEGIN", inputs[0]);
    
    // 获取 stream
    auto& device = VulkanDevice::instance();
    auto& stream = device.get_stream(out.primitive().stream().index);
    
    log("STREAM_ACQUIRED", out);
    
    // 创建 buffer
    auto buffer = VulkanBuffer::from_array(inputs[0]);
    
    log("BUFFER_CREATED", out);
    
    // 简单的 passthrough (copy)
    buffer.to_array(out);
    
    log("END", out);
}
```

### 5.2 预期调试输出

```
[VulkanPlaceholder] BEGIN: array([1.0, 2.0, 3.0])
[VulkanPlaceholder] STREAM_ACQUIRED: stream_index=0
[VulkanPlaceholder] BUFFER_CREATED: size=3, itemsize=4
[VulkanPlaceholder] END: array([1.0, 2.0, 3.0])
```

## 6. 下一步算子填充接口

### 6.1 算子开发者接口

```cpp
// mlx/backend/vulkan/primitives/binary.h
namespace mlx::core {

// 算子开发者只需实现这个模板
class Add : public UnaryPrimitive {
public:
    void eval_gpu(const vector<array>& inputs, array& out) override {
        // 1. 获取资源
        auto& device = vulkan::VulkanDevice::instance();
        auto& stream = device.get_stream(out.primitive().stream().index);
        
        // 2. 创建 buffers
        auto a = vulkan::VulkanBuffer::from_array(inputs[0]);
        auto b = vulkan::VulkanBuffer::from_array(inputs[1]);
        auto c = vulkan::VulkanBuffer::empty(out.shape(), out.dtype());
        
        // 3. 获取/创建 shader
        auto& registry = vulkan::KernelRegistry::instance();
        auto algo = registry.get_algorithm("add_f32", ...);
        
        // 4. 录制执行
        stream.record_begin();
        stream.record_op<kp::OpTensorSyncDevice>({a.tensor(), b.tensor()});
        stream.record_op<kp::OpAlgoDispatch>(algo);
        stream.record_op<kp::OpTensorSyncLocal>({c.tensor()});
        stream.record_end();
        
        // 5. 执行
        stream.eval();
        
        // 6. 输出
        c.to_array(out);
    }
};

} // namespace mlx::core
```

### 6.2 添加新算子的步骤

1. **创建 shader** (`shaders/<op_name>.comp`)
2. **编译 SPIR-V** (自动通过 CMake)
3. **注册 shader** (`kernel_registry.cpp`)
4. **实现 eval_gpu** (`primitives/<category>.cpp`)

## 7. 架构验证检查清单

- [ ] `gpu::is_available()` 正确返回 true/false
- [ ] `gpu::device_count()` 返回正确的设备数量
- [ ] `Scheduler::new_stream()` 成功创建 VulkanStream
- [ ] `gpu::eval()` 被正确调用
- [ ] `gpu::synchronize()` 等待完成
- [ ] 占位符算子输出调试信息
- [ ] Lazy Evaluation 工作（操作延迟到 eval 执行）

## 8. QuantizedMatmul（当前 Vulkan 首版覆盖）

### 8.1 入口与调度
- 入口：`mlx/backend/vulkan/primitives/fallback.cpp` 中 `QuantizedMatmul::eval_gpu(...)`。
- 先执行 stream-aware 输入就绪（避免 `async_eval` 同轮 event 自等待），再走分支：
  - 命中原生 Vulkan 条件：调度 `KernelRegistry::QMM_AFFINE_BF16_T4_G128`。
  - 未命中：显式回退到 CPU fallback（保持正确性与 stream 安全）。

### 8.2 首版原生覆盖条件
- `mode=Affine`
- `bits=4`
- `group_size=128`
- `transpose=true`
- `x/scales/biases/out` 为 `bfloat16`，`w` 为 `uint32`
- `w/scales/biases` 为 2D 且行连续

### 8.3 Shader 与注册
- Shader：`mlx/backend/vulkan/shaders/qmm_affine_bf16_t4_g128.comp`
- Registry 常量：`KernelRegistry::QMM_AFFINE_BF16_T4_G128`
- 注册位置：`mlx/backend/vulkan/kernel_registry.cpp`

### 8.4 当前限制
- 非上述量化配置（例如其他 `bits/group_size/mode`）仍走 CPU fallback。
- 当前实现以“先可用、再扩覆盖”为策略，优先降低实际推理中的 fallback 占比。

## 9. Binary（bf16）当前覆盖

### 9.1 已原生化算子
- `Add::eval_gpu`：已支持 `float32` 与 `bfloat16` 行连续同形状输入。
- `Multiply::eval_gpu`：已支持 `bfloat16` 行连续同形状输入。

### 9.2 Shader
- `mlx/backend/vulkan/shaders/add_bf16.comp`
- `mlx/backend/vulkan/shaders/mul_bf16.comp`

### 9.3 策略
- 命中条件时走 Vulkan kernel；
- 不命中条件时保持显式 CPU fallback，确保语义与正确性优先。

## 10. Fast Primitive（当前 Vulkan 覆盖）

### 10.1 已原生化路径
- `fast::RMSNorm`：`KernelRegistry::RMSNORM_BF16`
- `fast::RoPE`：`KernelRegistry::ROPE_BF16_T1`（名称沿用，实际已支持 `T>=1`）
- `fast::RoPE (freqs)`：`KernelRegistry::ROPE_BF16_FREQS`
- `fast::ScaledDotProductAttention`（首版）：`KernelRegistry::SDPA_BF16_DECODE_Q1`

### 10.2 覆盖条件
- `RMSNorm`
  - `x/w/out = bfloat16`
  - `x` 行连续且输出同形状
  - `axis_size` 为偶数
  - `w` 为标量或 1D 连续向量
- `RoPE`
  - `bfloat16`
  - `dims == D`，`T >= 1`
  - `base` 路径支持 `traditional=true/false`
  - `offset` 支持标量与长度为 `B` 的 1D 向量（`int32` 连续）
  - `base` 路径：按 `exp2(-p/half_dims * log2(base))` 计算频率
  - `freqs` 路径：`traditional=true/false`，`freqs=float32` 且 1D 连续（长度 `dims/2`），`offset` 支持标量/向量
  - 位置索引按 `offset + (row % T)` 计算，覆盖 decode 与 prefill 常见形态
- `ScaledDotProductAttention`（当前首版）
  - `dtype=bfloat16`
  - `q/out` 为 4D 密集行主序
  - `k/v` 支持两类布局：
    - 4D 密集行主序
    - KV cache-view（`stride[-1]==1`，`batch/head` 维紧邻，`seq` 维可跨大步长，允许 `data_size != size`）
  - `Q_len` 支持范围：`1..MLX_VK_SDPA_MAX_Q_LEN`（默认 `16`）
  - decode/vector 场景支持三类 mask 语义：
    - `mask=None`
    - `mask="causal"`（`Q_len<=K_len`）
    - `mask_mode="array"`（additive mask；支持 broadcast 到 `[B,Hq,Q,K]`）
  - bool mask 在 native 前重编码为 `uint32`（`mask_mode=2`），在 kernel 中按布尔语义直接屏蔽
  - `sinks` / training 仍走 fallback
  - `q_len<=MLX_VK_SDPA_MAX_Q_LEN`（默认 `16`，可通过环境变量调节）
  - `k_len` 门禁支持分段：
    - decode：`k_len<=MLX_VK_SDPA_MAX_K_LEN_DECODE`（默认 `16`）
    - prefill：`k_len<=MLX_VK_SDPA_MAX_K_LEN_PREFILL`（默认继承全局）
    - 兼容全局：`MLX_VK_SDPA_MAX_K_LEN`
  - `qk_dim<=256`，`v_dim<=256`
  - 内核算法：
    - `sdpa_bf16_decode_q1` 与 `split-k stage1` 已切换到单遍 online-softmax（不再双遍重复 QK）

### 10.3 仍走 fallback 的场景
- `RoPE` 的非连续/非 1D `freqs` 布局
- `ScaledDotProductAttention` 的未覆盖场景（training、sinks、`Q_len>MLX_VK_SDPA_MAX_Q_LEN`，或 `k_len>MLX_VK_SDPA_MAX_K_LEN`）

### 10.4 SDPA v3 方案（Metal 对齐 + Ollama/ggml Vulkan 参考）

#### 目标
- 直接对齐 Metal 的 `vector/full` 双路径，不再用单一路径硬门限（如固定 `k_len<=8`）承载主推理流。
- decode 与 prefill 都走 native Vulkan；fallback 仅用于明确未覆盖语义（训练、`output_logsumexp` 等）。
- 在当前 `Device -> CommandEncoder` 与 lazy scheduling 契约下实现，不引入 side path。

#### Metal 对齐结论（作为设计约束）
- Metal 明确分路：
  - `Q_len <= 8`：`sdpa_vector` / `sdpa_vector_2pass`
  - `Q_len > 8`：`sdpa_full_self_attention_*`
- `use_fallback` 的“全局拒绝条件”仅保留语义性条件（如 training / logsumexp），不是把 `mask/causal` 永久挡在 native 外。
- vector 路径按设备与序列长度在 1-pass/2-pass 间切换，full 路径走 tile 化 attention。

#### Ollama/ggml Vulkan 借鉴点
- 不是单一 pipeline：按 `HSK/HSV/small_rows/aligned/f32acc/flags` 建立 variant cache。
- path 选择包含 `scalar/coopmat1/coopmat2`，并在不满足共享内存或特性时自动回退到更稳路径。
- 长 `KV` 场景使用 `split_k`（先算局部 `O/L/M`，后 reduce）。
- 大 mask 场景先做 `mask_opt`（tile 级 all-`-inf` / all-`0` / mixed），减少主 kernel 无效访存。

#### 目标执行架构（Vulkan）
- `Path A: Decode Vector`（`Q_len <= 8`，优先 `Q_len=1`）
  - `A1`: subgroup online-softmax 单 pass（替换当前单线程 decode kernel）
  - `A2`: 2-pass/split-k（`KV` 大或 occupancy 低时启用）
  - `A3`: 完整语义增量：`GQA -> causal -> bool/float mask -> sinks`
- `Path B: Prefill Full`（`Q_len > 8`）
  - `B1`: `Br x Bc` tiled flash-attn（online `M/L`）
  - `B2`: `causal + array mask`（广播步幅与 Metal 行为对齐）
  - `B3`: `split_k` + reduce（长上下文）
  - `B4`: `mask_opt`（仅大 mask 启用）

#### Gate 策略（新）
- 保留全局 fallback：
  - `stream.device == cpu`
  - `detail::in_tracing()`
  - `is_training`
  - `output_logsumexp`（未实现前）
- 从全局 reject 中移除长期限制：
  - `has_mask`
  - `do_causal`
  - 大部分 `k_len` 上限限制（改为路径内策略）
- 路径内 gate（首批建议）：
  - dtype：`bf16` 输入，`f32` 累加，`bf16` 输出
  - `head_dim`：首批 `64/80/96/128/256`
  - layout：行连续优先，必要时 copy-unless（对齐 Metal 处理方式）

#### Pipeline Key（建议）
- `path` (`decode_1pass` / `decode_2pass_stage1` / `decode_2pass_reduce` / `prefill_tile` / `prefill_reduce` / `mask_opt`)
- `qk_dim`, `v_dim`, `dtype`, `mask_type`, `causal`, `has_sinks`, `aligned`, `acc_mode`, `small_cache`, `split_k_on`
- 仅在 key 变化时编译/缓存，避免 runtime 反复建 pipeline。

#### 数值与语义要求
- softmax 统一 online 形式：`M_new = max(M_old, rowmax)`，`L_new = exp(M_old-M_new)*L_old + sum(exp(S-M_new))`。
- `mask` 行为与 Metal 对齐：
  - bool mask -> `-inf` 屏蔽
  - additive mask -> 直接加到 logits
  - causal 在 `q_len <= k_len` 下与 Metal 同步偏移规则
- 当前 decode 实现已支持 `supports_bool_mask() == true`：
  bool mask 在 native 前重编码为 `uint32`（`mask_mode=2`）并由 kernel 直接按布尔语义屏蔽；
  additive mask 继续使用 `mask_mode=1`。训练/VJP 继续 fallback，确保梯度语义不回退。

#### 分阶段落地（执行顺序）
1. `A1`：替换当前 `sdpa_bf16_decode_q1` 为 subgroup 版本，先解除 `k_len<=8` 的性能阻塞。
2. `A2`：加 decode split-k reduce，覆盖 `KV` 长场景。
3. `A3`：补 `causal/mask/sinks` 语义，去掉 `global_gate` 对应拒绝。
4. `B1/B2`：落地 prefill tile kernel 与 mask/causal。
5. `B3/B4`：补 `split_k` 与 `mask_opt`，提升长上下文稳定吞吐。
6. `coopmat`（可选）：仅作为可探测加速 path，不替代 scalar/subgroup 兜底。

#### 验收口径
- 正确性：
  - `python/tests/test_fast_sdpa.py`
  - `Qwen3` 中英 prompt（1/10 token）
  - `ctest` 全量
- 覆盖率：
  - `MLX_VK_DEBUG_SDPA_REJECT` 中 `global_gate/k_len_cap` 明显下降
- 性能：
  - decode `20/40` token 吞吐相对当前基线提升且不出现 timeout
  - prefill（`Q_len>8`）相对 fallback 有稳定收益
