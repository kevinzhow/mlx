# Vulkan Backend - Operator Implementation Guide

## 概述

本文档定义了算子开发者如何为 MLX Vulkan Backend 添加新的算子实现。

## 前提条件

- 理解 MLX 的 `Primitive` 和 `UnaryPrimitive`/`BinaryPrimitive` 体系
- 了解基本的 GLSL 计算 shader 编写
- 熟悉 Kompute 的基本 API (`kp::Tensor`, `kp::Sequence`, `kp::Algorithm`)

## 快速开始：添加一个新算子

### 步骤 1: 创建 GLSL Shader

在 `mlx/backend/vulkan/shaders/` 创建 `<op_name>.comp`:

```glsl
// shaders/mul_f32.comp
#version 450
layout(local_size_x = 256) in;

layout(set = 0, binding = 0) readonly buffer BufferA {
    float data[];
} buffer_a;

layout(set = 0, binding = 1) readonly buffer BufferB {
    float data[];
} buffer_b;

layout(set = 0, binding = 2) writeonly buffer BufferC {
    float data[];
} buffer_c;

layout(push_constant) uniform PushConstants {
    uint size;
} params;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= params.size) return;
    buffer_c.data[idx] = buffer_a.data[idx] * buffer_b.data[idx];
}
```

### 步骤 2: 注册 Shader

在 `mlx/backend/vulkan/kernel_registry.cpp`:

```cpp
// 添加 kernel 名称
const char* KernelRegistry::MUL_F32 = "mul_f32";

// 在 register_builtin_shaders() 中添加
void KernelRegistry::register_builtin_shaders() {
    // ... existing shaders ...
    
    // Add mul_f32
    std::vector<uint32_t> mul_spirv((mul_f32_spv_len + 3) / 4);
    std::memcpy(mul_spirv.data(), mul_f32_spv, mul_f32_spv_len);
    shaders_[MUL_F32] = std::move(mul_spirv);
}
```

### 步骤 3: 实现 eval_gpu

在 `mlx/backend/vulkan/primitives/binary.cpp`:

```cpp
void Multiply::eval_gpu(const std::vector<array>& inputs, array& out) {
    // 1. Check Vulkan availability
    if (!vulkan::VulkanDevice::is_available()) {
        throw std::runtime_error("Vulkan backend not available");
    }
    
    // 2. Type check (can delegate to CPU for unsupported types)
    if (inputs[0].dtype() != float32) {
        eval_cpu(inputs, out);
        return;
    }
    
    // 3. Get device and stream
    auto& device = vulkan::VulkanDevice::instance();
    auto& stream = device.get_stream(out.primitive().stream().index);
    
    // 4. Create buffers
    auto a = vulkan::VulkanBuffer::from_array(inputs[0]);
    auto b = vulkan::VulkanBuffer::from_array(inputs[1]);
    auto c = vulkan::VulkanBuffer::empty(out.shape(), out.dtype());
    
    // 5. Get shader algorithm
    auto& registry = vulkan::KernelRegistry::instance();
    std::vector<std::shared_ptr<kp::Tensor>> params = {
        a.tensor(), b.tensor(), c.tensor()
    };
    auto workgroup = vulkan::compute_workgroup(out.size());
    std::vector<float> push_consts = {static_cast<float>(out.size())};
    
    auto algo = registry.get_algorithm(
        vulkan::KernelRegistry::MUL_F32,
        device.manager(),
        params,
        workgroup,
        push_consts
    );
    
    // 6. Record operations
    stream.record_begin();
    stream.sequence()->record<kp::OpTensorSyncDevice>(params);
    stream.sequence()->record<kp::OpAlgoDispatch>(algo);
    stream.sequence()->record<kp::OpTensorSyncLocal>({c.tensor()});
    stream.record_end();
    
    // 7. Execute
    stream.eval();
    
    // 8. Copy result
    c.to_array(out);
}
```

### 步骤 4: 重新构建

```bash
cd /home/kevinzhow/clawd/mlx/build
make -j4
```

## 标准算子模板

### 一元算子模板

```cpp
// primitives/unary.cpp
void <OpName>::eval_gpu(const std::vector<array>& inputs, array& out) {
    if (!vulkan::VulkanDevice::is_available()) {
        eval_cpu(inputs, out);
        return;
    }
    
    auto& device = vulkan::VulkanDevice::instance();
    auto& stream = device.get_stream(out.primitive().stream().index);
    
    // Input buffer
    auto in = vulkan::VulkanBuffer::from_array(inputs[0]);
    auto out_buffer = vulkan::VulkanBuffer::empty(out.shape(), out.dtype());
    
    // Get algorithm
    auto& registry = vulkan::KernelRegistry::instance();
    auto algo = registry.get_algorithm(
        vulkan::KernelRegistry::<OP_NAME>,
        device.manager(),
        {in.tensor(), out_buffer.tensor()},
        vulkan::compute_workgroup(out.size()),
        {static_cast<float>(out.size())}
    );
    
    // Record and execute
    stream.record_begin();
    stream.sequence()->record<kp::OpTensorSyncDevice>({in.tensor()});
    stream.sequence()->record<kp::OpAlgoDispatch>(algo);
    stream.sequence()->record<kp::OpTensorSyncLocal>({out_buffer.tensor()});
    stream.record_end();
    stream.eval();
    
    out_buffer.to_array(out);
}
```

### 二元算子模板

```cpp
// primitives/binary.cpp
void <OpName>::eval_gpu(const std::vector<array>& inputs, array& out) {
    if (!vulkan::VulkanDevice::is_available()) {
        eval_cpu(inputs, out);
        return;
    }
    
    auto& device = vulkan::VulkanDevice::instance();
    auto& stream = device.get_stream(out.primitive().stream().index);
    
    auto a = vulkan::VulkanBuffer::from_array(inputs[0]);
    auto b = vulkan::VulkanBuffer::from_array(inputs[1]);
    auto c = vulkan::VulkanBuffer::empty(out.shape(), out.dtype());
    
    auto& registry = vulkan::KernelRegistry::instance();
    auto algo = registry.get_algorithm(
        vulkan::KernelRegistry::<OP_NAME>,
        device.manager(),
        {a.tensor(), b.tensor(), c.tensor()},
        vulkan::compute_workgroup(out.size()),
        {static_cast<float>(out.size())}
    );
    
    stream.record_begin();
    stream.sequence()->record<kp::OpTensorSyncDevice>({a.tensor(), b.tensor()});
    stream.sequence()->record<kp::OpAlgoDispatch>(algo);
    stream.sequence()->record<kp::OpTensorSyncLocal>({c.tensor()});
    stream.record_end();
    stream.eval();
    
    c.to_array(out);
}
```

### 归约算子模板

```cpp
// primitives/reduce.cpp
void <OpName>::eval_gpu(const std::vector<array>& inputs, array& out) {
    // Reduction requires multi-pass or shared memory
    // See shaders/reduce_sum.comp for example
}
```

## GLSL Shader 模板

### 一元操作

```glsl
#version 450
layout(local_size_x = 256) in;

layout(set = 0, binding = 0) readonly buffer Input {
    float data[];
} input_buffer;

layout(set = 0, binding = 1) writeonly buffer Output {
    float data[];
} output_buffer;

layout(push_constant) uniform PushConstants {
    uint size;
} params;

// Operation function
float op(float x) {
    // Example: relu
    return max(x, 0.0);
    
    // Example: sigmoid
    // return 1.0 / (1.0 + exp(-x));
    
    // Example: sqrt
    // return sqrt(x);
}

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= params.size) return;
    
    output_buffer.data[idx] = op(input_buffer.data[idx]);
}
```

### 二元操作

```glsl
#version 450
layout(local_size_x = 256) in;

layout(set = 0, binding = 0) readonly buffer BufferA {
    float data[];
} buffer_a;

layout(set = 0, binding = 1) readonly buffer BufferB {
    float data[];
} buffer_b;

layout(set = 0, binding = 2) writeonly buffer BufferC {
    float data[];
} buffer_c;

layout(push_constant) uniform PushConstants {
    uint size;
} params;

// Operation function
float op(float a, float b) {
    // Example: add
    return a + b;
    
    // Example: multiply
    // return a * b;
    
    // Example: maximum
    // return max(a, b);
}

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= params.size) return;
    
    buffer_c.data[idx] = op(buffer_a.data[idx], buffer_b.data[idx]);
}
```

## 数据类型支持

### 支持的类型映射

| MLX Dtype | GLSL Type | Kompute Template |
|-----------|-----------|------------------|
| float32 | float | `kp::Tensor` (default) |
| float16 | float16 (extension) | `kp::TensorT<float16>` |
| int32 | int | `kp::TensorT<int32_t>` |
| int64 | long | `kp::TensorT<int64_t>` |
| uint32 | uint | `kp::TensorT<uint32_t>` |

### 多类型支持

为支持多类型，可以使用 C++ 模板：

```cpp
template<typename T>
void eval_gpu_typed(const std::vector<array>& inputs, array& out) {
    // Implementation with type T
}

void OpName::eval_gpu(const std::vector<array>& inputs, array& out) {
    switch (inputs[0].dtype()) {
        case float32:
            eval_gpu_typed<float>(inputs, out);
            break;
        case float16:
            eval_gpu_typed<half>(inputs, out);
            break;
        case int32:
            eval_gpu_typed<int32_t>(inputs, out);
            break;
        default:
            eval_cpu(inputs, out);
    }
}
```

## 调试技巧

### 添加调试输出

```cpp
void OpName::eval_gpu(const std::vector<array>& inputs, array& out) {
    std::cerr << "[OpName] eval_gpu: BEGIN" << std::endl;
    std::cerr << "[OpName]   input_shape=" << inputs[0].shape() << std::endl;
    std::cerr << "[OpName]   input_dtype=" << inputs[0].dtype() << std::endl;
    
    // ... implementation ...
    
    std::cerr << "[OpName] eval_gpu: END" << std::endl;
}
```

### 验证 Shader 加载

```cpp
// In kernel_registry.cpp
const std::vector<uint32_t>& KernelRegistry::get_shader(const std::string& name) {
    auto it = shaders_.find(name);
    if (it == shaders_.end()) {
        std::cerr << "[KernelRegistry] Shader not found: " << name << std::endl;
        throw std::runtime_error("Shader not found: " + name);
    }
    std::cerr << "[KernelRegistry] Loaded shader: " << name 
              << " (" << it->second.size() * 4 << " bytes)" << std::endl;
    return it->second;
}
```

## 性能优化

### 1. 使用 Tensor Pool

```cpp
// Instead of creating new tensors every time
auto tensor = device.create_tensor(size, itemsize);

// Use a pool (future enhancement)
auto tensor = device.tensor_pool().acquire(size, itemsize);
// ... use tensor ...
device.tensor_pool().release(tensor);
```

### 2. 批量操作

```cpp
// Instead of multiple eval() calls
stream.record_begin();
stream.sequence()->record<kp::OpTensorSyncDevice>({a, b, c});
stream.sequence()->record<kp::OpAlgoDispatch>(algo1);
stream.sequence()->record<kp::OpAlgoDispatch>(algo2);
stream.sequence()->record<kp::OpTensorSyncLocal>({out});
stream.record_end();
stream.eval();  // Single eval for multiple operations
```

### 3. 异步执行

```cpp
// For independent operations
stream.eval_async();
// ... do other work ...
stream.await();
```

## 测试新算子

```python
import mlx.core as mx

mx.set_default_device(mx.gpu)

# Test your new operator
a = mx.array([1.0, 2.0, 3.0, 4.0, 5.0])
b = mx.array([2.0, 2.0, 2.0, 2.0, 2.0])

# Replace with your operation
# c = mx.multiply(a, b)  # or mx.add, mx.maximum, etc.
# print(c)  # Should print correct result

# Verify against CPU
mx.set_default_device(mx.cpu)
c_cpu = mx.multiply(a, b)

assert mx.allclose(c, c_cpu)
print("Test passed!")
```

## 常见问题

### Q: Shader 编译失败
A: 检查 GLSL 语法，确保使用了正确的版本和扩展

### Q: Runtime error "Shader not found"
A: 确保在 `register_builtin_shaders()` 中注册了 shader，并重新构建

### Q: Wrong results
A: 检查 workgroup 计算是否正确，确保边界检查在 shader 中

### Q: Performance is poor
A: 检查数据同步频率，尽量减少 host-device 数据传输

## 示例：完整添加 Multiply 算子

参见 `examples/adding_multiply_op.md` 获取完整示例。
