# MLX Vulkan Backend - Metal Architecture Alignment

## 架构对齐状态

### ✅ 已完成对齐的组件

#### 1. Device 类 (`device.h/cpp`)

对齐 Metal 的 `Device` 类：

| Metal | Vulkan | 状态 |
|-------|--------|------|
| `MTL::Device* device_` | `vk::Device device_` | ✅ |
| `unordered_map<int32_t, DeviceStream> stream_map_` | 相同 | ✅ |
| `new_queue(int index)` | 相同 | ✅ |
| `get_command_buffer(int index)` | 相同 | ✅ |
| `command_buffer_needs_commit(int index)` | 相同 | ✅ |
| `commit_command_buffer(int index)` | 相同 | ✅ |
| `get_command_encoder(int index)` | 相同 | ✅ |
| `end_encoding(int index)` | 相同 | ✅ |
| `library_map_` / `library_kernels_` | `pipeline_cache_` | ✅ |

#### 2. CommandEncoder 类 (`device.h/cpp`)

对齐 Metal 的 `CommandEncoder`：

| Metal | Vulkan | 状态 |
|-------|--------|------|
| `set_input_array(const array& a, int idx, int64_t offset)` | 相同 | ✅ |
| `set_output_array(array& a, int idx, int64_t offset)` | 相同 | ✅ |
| `register_output_array(const array& a)` | 相同 | ✅ |
| `set_buffer(MTL::Buffer* buf, int idx, int64_t offset)` | `set_buffer(vk::Buffer buf, ...)` | ✅ |
| `dispatch_threadgroups(MTL::Size grid, MTL::Size group)` | `dispatch_threadgroups(vk::Extent3D, vk::Extent3D)` | ✅ |
| `dispatch_threads(MTL::Size grid, MTL::Size group)` | `dispatch_threads(vk::Extent3D, vk::Extent3D)` | ✅ |
| `maybeInsertBarrier()` | 相同 | ✅ |
| `barrier()` | 相同 | ✅ |
| `set_compute_pipeline_state(MTL::ComputePipelineState*)` | `set_compute_pipeline_state(vk::Pipeline)` | ✅ |
| `set_vector_bytes<T>(const T& vec, int idx)` | 模板相同 | ✅ |
| `set_bytes<T>(const T& v, int idx)` | 模板相同 | ✅ |
| `inputs()` / `outputs()` | 相同 | ✅ |

#### 3. DeviceStream 结构 (`device.h`)

对齐 Metal 的 `DeviceStream`：

| Metal | Vulkan | 状态 |
|-------|--------|------|
| `MTL::CommandQueue* queue` | `vk::Queue queue` | ✅ |
| `unordered_map<const void*, shared_ptr<Fence>> outputs` | 相同 | ✅ |
| `mutex fence_mtx` | 相同 | ✅ |
| `MTL::CommandBuffer* buffer` | `vk::CommandBuffer buffer` | ✅ |
| `int buffer_ops` | 相同 | ✅ |
| `size_t buffer_sizes` | 相同 | ✅ |
| `unique_ptr<CommandEncoder> encoder` | 相同 | ✅ |
| `shared_ptr<Fence> fence` | 相同 | ✅ |
| `vector<array> temporaries` | 相同 | ✅ |

#### 4. GPU Interface (`gpu_interface.cpp`)

对齐 `metal/eval.cpp`：

| 函数 | Metal 实现 | Vulkan 实现 | 状态 |
|------|-----------|-------------|------|
| `new_stream(Stream)` | `metal::device().new_queue()` | `vulkan::device().new_queue()` | ✅ |
| `eval(array&)` | 使用 CommandEncoder | 使用 CommandEncoder | ✅ |
| `finalize(Stream)` | `commit_command_buffer()` | `commit_command_buffer()` | ✅ |
| `synchronize(Stream)` | `waitUntilCompleted()` | `waitIdle()` | ✅ |

#### 5. Binary 算子 (`primitives/binary.cpp`)

对齐 `metal/binary.cpp`：

| 模式 | Metal | Vulkan | 状态 |
|------|-------|--------|------|
| `binary_op_gpu_inplace` | 使用 `get_command_encoder()` | 使用 `get_command_encoder()` | ✅ |
| `Add::eval_gpu` | 调用 `binary_op_gpu_vulkan` | 调用 `binary_op_gpu_vulkan` | ✅ |
| `Multiply::eval_gpu` | 类似模式 | 类似模式 | ✅ |

### 📁 文件结构对齐

```
mlx/backend/vulkan/
├── device.h              # 对齐 metal/device.h
├── device.cpp            # 对齐 metal/device.cpp
├── gpu_interface.cpp     # 对齐 metal/eval.cpp
├── primitives/
│   └── binary.cpp        # 对齐 metal/primitives/binary.cpp
├── shaders/
│   └── binary_add.comp   # GLSL 替代 MSL
└── CMakeLists.txt        # 简化版
```

### 🔑 关键架构对应关系

#### Metal → Vulkan 类型映射

| Metal | Vulkan | 说明 |
|-------|--------|------|
| `MTL::Device` | `vk::Device` | GPU 设备 |
| `MTL::CommandQueue` | `vk::Queue` | 命令队列 |
| `MTL::CommandBuffer` | `vk::CommandBuffer` | 命令缓冲区 |
| `MTL::ComputeCommandEncoder` | `CommandEncoder` (自定义) | 计算编码器 |
| `MTL::ComputePipelineState` | `vk::Pipeline` | 计算管线 |
| `MTL::Buffer` | `vk::Buffer` | GPU 缓冲区 |
| `MTL::Fence` | `vk::Fence` + Fence 包装 | 同步栅栏 |
| `MTL::Size` | `vk::Extent3D` | 3D 尺寸 |

#### Metal → Vulkan API 映射

| Metal API | Vulkan API | 说明 |
|-----------|-----------|------|
| `newLibrary` | `createShaderModule` + `createComputePipeline` | 创建 shader |
| `newCommandQueue` | `getQueue` | 获取队列 |
| `commandBuffer` | `allocateCommandBuffers` | 分配命令缓冲区 |
| `computeCommandEncoder` | `beginCommandBuffer` + 自定义 Encoder | 开始编码 |
| `setBuffer:offset:atIndex:` | `cmdBuffer.bindPipeline` + descriptor sets | 绑定缓冲区 |
| `dispatchThreadgroups:threadsPerThreadgroup:` | `cmdBuffer.dispatch` | 派发计算 |
| `endEncoding` | `cmdBuffer.end` | 结束编码 |
| `commit` | `queue.submit` | 提交执行 |
| `waitUntilCompleted` | `queue.waitIdle` | 等待完成 |

### 🎯 算子实现模式

对齐 Metal 的算子实现模式：

```cpp
// Metal 风格
void Add::eval_gpu(const vector<array>& inputs, array& out) {
    auto& s = out.primitive().stream();
    auto& d = metal::device(s.device);
    auto& enc = d.get_command_encoder(s.index);
    
    auto kernel = get_binary_kernel(d, kernel_name, ...);
    enc.set_compute_pipeline_state(kernel);
    
    int arg_idx = 0;
    enc.set_input_array(inputs[0], arg_idx++);
    enc.set_input_array(inputs[1], arg_idx++);
    enc.set_output_array(out, arg_idx++);
    
    enc.dispatch_threadgroups(grid_dims, group_dims);
}

// Vulkan 风格 (相同)
void Add::eval_gpu(const vector<array>& inputs, array& out) {
    auto& s = out.primitive().stream();
    auto& d = vulkan::device(s.device);
    auto& enc = d.get_command_encoder(s.index);
    
    auto pipeline = d.get_pipeline(kernel_name, ...);
    enc.set_compute_pipeline_state(pipeline);
    
    int arg_idx = 0;
    enc.set_input_array(inputs[0], arg_idx++);
    enc.set_input_array(inputs[1], arg_idx++);
    enc.set_output_array(out, arg_idx++);
    
    enc.dispatch_threadgroups(grid_dims, group_dims);
}
```

### 📊 状态跟踪对齐

Metal 和 Vulkan 都跟踪以下状态：

```cpp
// DeviceStream
int buffer_ops{0};           // 命令缓冲区中的操作数
size_t buffer_sizes{0};      // 累积的缓冲区大小

// 阈值配置
int max_ops_per_buffer_ = 100;     // 最大操作数
int max_mb_per_buffer_ = 50;       // 最大缓冲区大小 (MB)
```

### 🔄 Lazy Evaluation 对齐

两者都使用相同的延迟执行模式：

```cpp
// 1. 获取 encoder (开始录制)
auto& enc = d.get_command_encoder(s.index);

// 2. 设置参数
enc.set_input_array(a, 0);
enc.set_output_array(out, 1);

// 3. 派发 (录制命令，不执行)
enc.dispatch_threadgroups(grid, group);

// 4. eval() 时检查是否需要提交
if (d.command_buffer_needs_commit(s.index)) {
    d.end_encoding(s.index);
    d.commit_command_buffer(s.index);
}
```

### 📝 差异说明

| 方面 | Metal | Vulkan | 处理方式 |
|------|-------|--------|----------|
| 内存管理 | 自动 (UMA) | 显式 | Buffer 抽象层统一 |
| Shader 编译 | 运行时 MSL | SPIR-V 预编译 | CMake 编译 `.comp` → `.spv` |
| Descriptor Sets | 隐式 | 显式 | CommandEncoder 内部管理 |
| Barrier | 自动 | 显式 | `maybeInsertBarrier()` 插入 |

### ✅ 验证检查清单

- [x] Device 类结构与 Metal 一致
- [x] CommandEncoder 提供相同接口
- [x] DeviceStream 包含 buffer_ops/buffer_sizes
- [x] new_queue/get_command_encoder/end_encoding 流程一致
- [x] Add/Multiply 通过 CommandEncoder 实现
- [x] gpu::eval/synchronize 与 Scheduler 集成
- [x] 使用原生 Vulkan API (vulkan.hpp)

### 🚀 下一步

1. **实现 Buffer 管理**：创建 `Buffer` 类管理 `vk::Buffer` 和 `vk::DeviceMemory`
2. **集成 Shader**：加载 SPIR-V 并创建 Pipeline
3. **完善算子**：实现更多 binary/unary 算子
4. **测试验证**：运行验证脚本确认链路
