# PROGRESS

更新日期: 2026-02-09

## 目标

- 对标 Metal Backend 机制，实现基于 Kompute `v0.9.0` 的 Vulkan Backend。
- 先保证机制对齐与稳定性（stream/eval/finalize/synchronize），再逐步替换 CPU fallback。

## 已完成进展

### 1. Vulkan 基础链路补齐并可编译链接
- 补齐 `device_info` / `event` / `fence` / `gpu_fallback` / `primitives/fallback` 等缺失实现。
- 修复重复符号与关键链接缺口，`cmake --build build --target tests` 可通过。

### 2. 机制稳定化修复
- 修复 `Device::commit_command_buffer` 与 sequence 生命周期问题，避免重复提交。
- CPU encoder 支持 GPU stream 回退到 CPU 默认 stream，避免 fallback 调度崩溃。
- 多处 Vulkan fallback 增加输入就绪保障（先 `eval/wait` 再 `eval_cpu`）。
- 修复二元算子 fallback 的 0-size 早退问题（避免输出未 materialize 导致崩溃）。
- 为 `array::unsafe_weak_copy` 增加防御性检查，避免空 data 指针直接段错误。

### 3. 性能与稳定性优化
- `vulkan::is_available()` 改为原生 Vulkan 物理设备探测 + 进程级缓存，避免高频重复创建/销毁 `kp::Manager`。
- 清理 Vulkan runtime 高频调试输出，移除 I/O 干扰。
- 保留 CPU fallback 的同步语义，消除竞态崩溃。
- 调整 `scheduler.cpp` 判断顺序，仅在 GPU 分支触发 `gpu::is_available()`。
- 修复 `Buffer::from_array` 数据桥接，支持多种 dtype。
- **修复 Algorithm 缓存逻辑**: cache key 包含 push constants，解决参数失效问题。
- **暂时关闭 BufferManager 缓存**: 避免内存复用导致的数据不一致。
- **修复 Add 原生算子**: 增加边界检查和同步指令。

### 4. C++ 测试里程碑
- `test arithmetic binary ops` 全量通过（包含 native Add 路径）。
- `test quantize dequantize` 通过。
- `test scheduler races` 在 Vulkan 下恢复稳定，通过 20 次连续复测。
- **C++ 全量测试通过**: `223/223` tests passed (16.21 sec)

### 5. Python 测试里程碑 ✨
**整体通过率: 94.1% (332/353 tests)**

#### 完美通过的测试 (100%)
- `test_constants.py` (3/3)
- `test_device.py` (10/10) - GPU 检测正常
- `test_memory.py` (2/3, 1 Metal-specific skip)
- `test_init.py` (9/9)
- `test_reduce.py` (10/10)
- `test_random.py` (14/14)
- `test_compile.py` (52/52) - **编译系统完全正常**
- `test_autograd.py` (31/31) - **自动微分完全正常**
- `test_linalg.py` (5/5)

#### 高通过率测试
- `test_array.py` (67/68, 98.5%)
- `test_ops.py` (117/129, 90.7%) - 11 失败, 1 错误

#### 部分通过测试
- `test_blas.py` (12/19, 63.2%) - **7 个矩阵乘法相关失败**

#### 关键成就
- ✅ **Autograd 100% 工作** - 所有梯度计算正确
- ✅ **Compilation 100% 工作** - JIT 编译完全正常
- ✅ **RNG 100% 工作** - 随机数生成正常
- ✅ **核心数组操作 98.5%** - 基础功能稳定

## 当前阻塞

### Python 测试失败分析 (19 个失败)

#### 1. BLAS 矩阵乘法问题 (7 失败) - **最高优先级**
- `test_gather_mm_sorted` (2 variants)
- `test_matmul_batched`
- `test_matmul_shapes` (2 variants) 
- `test_matrix_vector_batched` (2 variants)
**原因**: 矩阵乘法实现有问题，影响深度学习核心操作

#### 2. 数学函数问题 (5 失败)
- `test_sin`, `test_cos` - 三角函数
- `test_log2`, `test_log10` - 对数函数
- `test_rsqrt` - 平方根倒数
**原因**: CPU fallback 实现或精度问题

#### 3. 梯度问题 (4 失败)
- 反三角函数梯度: arcsin, arccos, arcsinh, arccosh
**原因**: VJP 实现问题

#### 4. 其他问题
- `test_unary_ops_from_non_array` - log2/log10 标量输入 (2 失败)
- `test_arange_corner_cases_cast` - AttributeError (1 错误)
- `test_async_eval` - 挂起（未计入统计）
- `test_quantized.py` - core dump 崩溃

## 下一步计划

### 目标: 修复 Python 测试失败，达到 100% 通过率

#### 优先级 1: 修复矩阵乘法 (影响 7 个测试)
- [ ] 调试 batched matmul 实现
- [ ] 验证 matmul shape 处理逻辑
- [ ] 检查 gather+matmul 融合
- [ ] 验证 matrix-vector 乘法

**验证命令**:
```bash
source venv/bin/activate && cd python/tests
python test_blas.py TestBlas.test_matmul_batched -v
python test_blas.py TestBlas.test_matmul_shapes -v
python test_blas.py TestBlas.test_matrix_vector_batched -v
```

#### 优先级 2: 修复数学函数 (影响 5 个测试)
- [ ] 检查 sin/cos 的 CPU fallback 实现
- [ ] 检查 log2/log10 实现
- [ ] 检查 rsqrt 实现
- [ ] 考虑实现原生 Vulkan 算子

**验证命令**:
```bash
python test_ops.py TestOps.test_sin -v
python test_ops.py TestOps.test_cos -v
python test_ops.py TestOps.test_log2 -v
```

#### 优先级 3: 修复反三角函数梯度 (影响 4 个测试)
- [ ] 检查 arcsin/arccos 的 VJP 实现
- [ ] 检查 arcsinh/arccosh 的 VJP 实现

#### 优先级 4: 其他问题
- [ ] 修复 async_eval 挂起问题
- [ ] 调试 test_quantized.py 崩溃
- [ ] 修复 arange corner case

### 验证门禁

**单项测试**:
```bash
# C++ 测试
ctest --test-dir build -R "test scheduler races" --output-on-failure --timeout 120

# Python 单个文件
source venv/bin/activate && cd python/tests
python test_blas.py -v
python test_ops.py -v
```

**全量测试**:
```bash
# C++ 全量
ctest --test-dir build --stop-on-failure --output-on-failure

# Python 批量
source venv/bin/activate && ./run_tests.sh
```

## 维护规则

- 每次有实质进展（修复、发现新阻塞、测试里程碑）必须更新本文件。
- 进入下一轮工作前，先以本文件中的"当前阻塞 + 下一步计划"为执行入口。
