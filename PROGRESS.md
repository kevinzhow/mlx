# PROGRESS

更新日期: 2026-02-11（精简版）

## 目标
- 对标 Metal backend 运行时契约，在 MLX 现有 GPU 路径内完成 Vulkan backend（Kompute v0.9.0）主线化。
- 优先级保持不变：先稳定与正确，再扩原生覆盖，最后做架构级性能提升。
- 开发流程约束：先做命中分析，再做优化；避免在低命中路径上投入主优化成本。

## 当前快照
- 构建状态：`MLX_BUILD_VULKAN=ON` 下 `mlx` 与 `tests` 均可稳定构建链接。
- 回归状态：`ctest` 全量 `223/223` 通过；`python/tests/test_fast_sdpa.py` 通过（`20 passed, 1 skipped`）。
- 模型正确性：Qwen3-0.6B-MLX-4bit（EN/ZH，10 token）输出正常，无乱码/`!!!!!!!!!!` 回归。
- 流程约束（新增）：Qwen3 `mlx_lm generate` 正确性/性能口径需串行执行；任意 C++/Vulkan/shader 变更后，Python 侧验证前必须先执行 `python3 setup.py build_ext --inplace`，避免旧扩展产物导致误判。
- 近期吞吐（实卡 Vulkan，串行口径）：
  - 10-token（EN）：约 `4.51 tok/s`（`MLX_VK_ENABLE_QMM_NATIVE_M1_REDUCE_SUBGROUP=1`，默认 ON）
  - 40-token（EN）：约 `3.58 tok/s`（同口径）
  - 80-token（EN）：约 `2.89 tok/s`（同口径）
- decode 主线 fallback 占比（同口径 profile）：
  - 由 `18.40%` 降到 `13.50%`（D-9.2），再降到 `3.98%`（D-9.11），当前进一步降到 `0.62%`（D-9.13）
- RoPE 回退状态（Qwen3 EN 40-token 样本）：
  - `MLX_VK_ENABLE_ROPE_HS_TRANSPOSED=1` 时 `VulkanRoPEReject` 由 `55` 次降到 `0`。

## 历史完成摘要（已压缩）
1. Vulkan 基础链路与运行时契约已打通  
- 设备识别、stream/eval/sync/fallback 生命周期问题已完成关键修复。  
- 修复过往 async 死锁、host/device 脏数据覆盖、alias cache 不一致等关键问题。

2. 正确性主阻塞已解除  
- Qwen 生成异常（含 `!!!!!!!!!!`）已定位并修复。  
- 当前默认门禁配置下，中英文生成输出稳定。

3. 原生覆盖持续扩展  
- 已落地主线原生路径（含部分窄门禁）：QMM、RMSNorm、RoPE、SDPA（分阶段）、ArgReduce（argmax last-dim）、若干 binary/unary。

4. 可观测性与工程闭环完善  
- 已建立 Vulkan profile/debug 门禁体系。  
- Shader 更新流程已固化为 `.comp -> .spv -> *_spv.h`（同轮闭环）。

5. 文档与运行参数固化  
- `AGENTS.md` 已记录实卡运行环境、标准正确性命令、常用 gate/debug 开关。

## 最近里程碑

### D-9.1：ArgReduce decode 热点 native 化（已完成）
- 目标：移除 decode 主链路 `ArgReduce` CPU fallback 热点。
- 结果：`ArgReduce(argmax, axis=last)` 命中 native 且回归通过。
- 收益：10/40 token 吞吐均有小幅稳定正收益（约 `~1%`）。

### D-9.2：Compiled 热点拆解 + `Sigmoid*Mul*Mul` native 化（已完成）
- 目标：拆解 `Compiled` 高占比来源，并优先移除单一高频融合子图热点。
- 归因：主热点为 `CompiledSigmoidMultiplyMultiply`（bf16，同形状行连续）。
- 变更：新增 `silu_mul_bf16` kernel，并在 `Compiled::eval_gpu` 增加该子图窄门禁 native 桥接。
- profile 收益（10-token 同口径）：
  - `Compiled`：`~660ms -> ~30ms`
  - 总体：`fallback_pct 18.40% -> 13.50%`
- 吞吐 A/B（gate on/off，串行）：
  - 10-token：`3.233 vs 3.192 tok/s`（`+1.3%`）
  - 40-token：`2.701 vs 2.486 tok/s`（`+8.6%`）

### D-9.3：QMM decode `M=1` 专核（已完成）
- 目标：优先吃下 decode 高频形状（`rows==1`）的 `QuantizedMatmul`，降低 QMM 热点指令开销。
- Metal/Ollama 对照启发：
  - Metal 路线强调“高频形状专核 + 最小分支调度”。
  - Ollama/ggml Vulkan 路线强调“解码小 batch 专项优化 + 内核特化”。
  - 本轮采用同类策略：仅对 decode 主形状开窄门禁专核，不改通用路径行为。
- 变更：
  - 新增 shader：`qmm_affine_bf16_t4_g128_m1.comp`（`rows==1` 专核，`local_size_x=128`）。
  - 新增 kernel 注册：`QMM_AFFINE_BF16_T4_G128_M1`。
  - 在 `QuantizedMatmul::eval_gpu` 增加 `rows==1` 动态派发与 gate：`MLX_VK_ENABLE_QMM_NATIVE_M1`（默认 ON）。
  - 按 kernel 工作组大小动态计算 `groups_x`，并修正 `out_words` 为 ceil 形式。
- 回归：
  - `ctest -R "quantize dequantize|arithmetic binary ops|scheduler races|arg reduce"` 通过。
  - `python/tests/test_fast_sdpa.py -v` 通过（`20 passed, 1 skipped`）。
- 吞吐 A/B（Qwen3 EN，实卡 Vulkan，串行）：
  - 40-token：`3.348 vs 2.652 tok/s`（`+26.2%`）
  - 10-token：`4.102 vs 3.201 tok/s`（`+28.1%`）

### E-9.3a：`M=1` 双列同线程累加实验（已回退）
- 假设：同一线程同时累加相邻两列，复用 `x` 解包，可进一步降指令数。
- 实测：`M1=1` 时 40-token 吞吐约 `3.33 -> 2.86 tok/s`（显著回退）。
- 结论：该实现引入了额外分支与寄存器压力，抵消了访存复用收益；已回退到 D-9.3 稳定版本。

### D-9.4：QMM `M>1` 形状分桶（阶段1：`M=2/4/8/16` 已完成）
- 目标：延续 Metal/Ollama 的“高频形状专核”路线，把 QMM 从单点 `M=1` 扩展到小 batch 桶。
- 本轮变更：
  - 新增 shader：`qmm_affine_bf16_t4_g128_m2.comp`（`rows==2` 专核）。
  - 新增 shader：`qmm_affine_bf16_t4_g128_m4.comp`（`rows==4` 专核）。
  - 新增 shader：`qmm_affine_bf16_t4_g128_m8.comp`（`rows==8` 专核）。
  - 新增 shader：`qmm_affine_bf16_t4_g128_m16.comp`（`9<=rows<=16` 专核）。
  - 新增 kernel 注册：`QMM_AFFINE_BF16_T4_G128_M2`。
  - 新增 kernel 注册：`QMM_AFFINE_BF16_T4_G128_M4`。
  - 新增 kernel 注册：`QMM_AFFINE_BF16_T4_G128_M8`。
  - 新增 kernel 注册：`QMM_AFFINE_BF16_T4_G128_M16`。
  - `QuantizedMatmul::eval_gpu` 增加 `rows==2/4/8` 与 `9<=rows<=16` 动态派发与 gate：
    - `MLX_VK_ENABLE_QMM_NATIVE_M2=1`（默认 ON）
    - `MLX_VK_ENABLE_QMM_NATIVE_M4=1`（默认 ON）
    - `MLX_VK_ENABLE_QMM_NATIVE_M8=1`（默认 ON）
    - `MLX_VK_ENABLE_QMM_NATIVE_M16=1`（默认 ON）
  - 新增 QMM 运行统计：`MLX_VK_QMM_STATS=1`
    - 进程退出时打印 native kernel 命中计数（`m1/m2/m4/m8/generic`）与 `rows` 桶分布、fallback 桶分布。
  - 严格执行 shader 闭环：`.comp -> .spv -> *_spv.h` 同轮完成并构建。
- 验证：
  - `ctest --test-dir build_release_vulkan --output-on-failure --timeout 120`：`223/223` 通过。
  - `python/tests/test_fast_sdpa.py -v`：`20 passed, 1 skipped`。
  - Qwen3（EN 40-token）维持高位：约 `3.32~3.43 tok/s`（无正确性回归）。
- 结论：
  - `M=2/4/8/16` 路径已打通，形状分桶框架成立；下一步进入向量化/子组化分支。

### D-9.5：QMM 命中分布可观测性（已完成）
- 目标：在继续做 QMM 架构优化前，先拿到真实命中桶，避免“优化了未命中路径”。
- 变更：
  - `QuantizedMatmul::eval_gpu` 增加 QMM 统计埋点与退出打印。
  - 新增环境变量：`MLX_VK_QMM_STATS=1`。
  - 指标包含：native success/fail、kernel 命中计数、native/fallback 的 rows 桶分布。
- 观测（Qwen3 EN 40-token，实卡 Vulkan）：
  - 优化前：
    - `qmm_affine_bf16_t4_g128_m1`: `8077`
    - `qmm_affine_bf16_t4_g128`(generic): `191`
    - rows 桶：`rows=1` 主导，另有 `rows=9-16` 命中 generic。
  - 优化后（新增 `m16`）：
    - `qmm_affine_bf16_t4_g128_m1`: `8077`
    - `qmm_affine_bf16_t4_g128_m16`: `191`
    - rows 桶仍为 `rows=1` + `rows=9-16`，但 `9-16` 已迁移到专核。
- 结论：
  - 当前工作负载主命中为 `M1`，次热点为 `rows=9-16`，且两者均已专核化。
  - 下一阶段聚焦 `M1` 主核向量化/子组化，避免继续扩低命中桶。

### D-9.6：QMM `M1` 并行归约路径（已完成，默认 ON）
- 目标：直接优化 decode 主命中路径（`rows=1`），提升每 token 生成吞吐。
- 变更：
  - 新增 shader：`qmm_affine_bf16_t4_g128_m1_reduce.comp`。
  - 新增 kernel 注册：`QMM_AFFINE_BF16_T4_G128_M1_REDUCE`。
  - `QuantizedMatmul::eval_gpu` 增加 `rows==1` 归约路径派发。
  - gate：`MLX_VK_ENABLE_QMM_NATIVE_M1_REDUCE`（默认 ON，可关闭回退旧 `M1` 路径）。
- 命中验证（`MLX_VK_QMM_STATS=1`，Qwen3 EN 40-token）：
  - `qmm_affine_bf16_t4_g128_m1_reduce`: `8077`
  - `qmm_affine_bf16_t4_g128_m16`: `191`
  - 说明：主命中 `rows=1` 已切换到新归约核，`rows=9-16` 维持 `m16`。
- 吞吐 A/B（串行）：
  - 40-token：`3.62 / 3.58 tok/s`（`M1_REDUCE=1`） vs `3.34 / 3.35 tok/s`（`M1_REDUCE=0`）
  - 平均提升约 `+8~9%`。
- 稳定性：
  - `ctest` 全量 `223/223` 通过。
  - `python/tests/test_fast_sdpa.py` 通过（`20 passed, 1 skipped`）。
  - Qwen3 EN/ZH 输出正确，无乱码回归。

### E-9.5a：QMM `M=1` 多累加器展开实验（已回退）
- 假设：通过多累加器 + 最内层显式展开降低依赖链，提升 decode 吞吐。
- 实测：Qwen3 EN 40-token 吞吐无稳定正收益（约 `3.32~3.33 tok/s`，未优于基线）。
- 结论：已回退该实现，保持当前稳定核。

### D-9.7：流程护栏修正（已完成）
- 背景：出现过“源码已更新但 Python 扩展未重编，导致运行仍落在旧产物”的问题，干扰性能/正确性判断。
- 本轮修正：
  - 将“Qwen3 `mlx_lm generate` 口径必须串行执行（不并行）”写入 `AGENTS.md` 强制规则。
  - 明确该串行约束仅针对 Qwen3 生成测试，不约束 `ctest`/常规 Python 单测。
  - 将“Python 侧验证前必须先 `python3 setup.py build_ext --inplace`”写入 `AGENTS.md` 强制规则。
  - 将该流程护栏同步记录到 `PROGRESS.md` 当前快照，作为后续默认执行前置条件。
- 影响：
  - 避免并行执行导致的瞬时缓存/产物竞争噪声。
  - 避免旧扩展产物造成的假结论（尤其是 shader/C++ 改动后）。

### D-9.8：RoPE head/seq 转置布局默认放开（已完成）
- 背景：
  - Qwen3 decode 样本中 RoPE fallback 主因集中为 `in_or_out_layout`。
  - 典型输入布局：`shape=[1,8|16,12,128]`，`strides=[B,128,H*128,1]`，即 head/seq 转置视图。
- 变更：
  - 将 `MLX_VK_ENABLE_ROPE_HS_TRANSPOSED` 从默认 OFF 改为默认 ON（仍可 env 关闭）。
  - 不改变 `MLX_VK_ENABLE_ROPE_NATIVE` 总开关语义。
- 验证：
  - `ctest --test-dir build_release_vulkan --output-on-failure --timeout 120`：`223/223` 通过。
  - `PYTHONPATH=python python3 python/tests/test_fast_sdpa.py -v`：`20 passed, 1 skipped`。
  - Qwen3 EN 40-token（实卡 Vulkan，串行）：`VulkanRoPEReject` 由 `55` -> `0`。
  - 吞吐无负收益（40-token A/B 约持平，`~3.63 tok/s`）。
- 结论：
  - RoPE 高频布局回退已消除，后续优化重心继续放在 QMM 主耗时与剩余小算子回退尾部。

### D-9.9：QMM `M1_REDUCE` subgroup 归约路径（已完成，默认 ON）
- 背景（Metal/Ollama 对照）：
  - Metal 路线长期依赖 SIMD-group / wave 级归约减少 workgroup 级 barrier 开销。
  - Ollama/ggml Vulkan 路线同样在 decode 热核优先利用 subgroup/wave 原语，减少 shared-memory 归约树。
  - 当前 QMM `rows=1` 是绝对主命中（`~8k` 次/40-token），适合作为 subgroup 先行落点。
- 变更：
  - 新增 shader：`qmm_affine_bf16_t4_g128_m1_reduce_subgroup.comp`（`subgroupAdd` 两级归约）。
  - 新增 kernel 注册：`QMM_AFFINE_BF16_T4_G128_M1_REDUCE_SUBGROUP`。
  - `QuantizedMatmul::eval_gpu` 新增派发策略：
    - 默认优先 `M1_REDUCE_SUBGROUP`。
    - 若 subgroup kernel dispatch 异常，进程内自动降级关闭该路径并回退到 `M1_REDUCE`，避免重复异常。
  - 新增 gate：`MLX_VK_ENABLE_QMM_NATIVE_M1_REDUCE_SUBGROUP`（默认 ON，可显式关闭）。
  - 严格执行 shader 闭环：`.comp -> .spv -> *_spv.h`。
- 验证：
  - `python3 setup.py build_ext --inplace` 通过。
  - `ctest --test-dir build_release_vulkan --output-on-failure --timeout 120`：`223/223` 通过。
  - `PYTHONPATH=python python3 python/tests/test_fast_sdpa.py -v`：`20 passed, 1 skipped`。
  - Qwen3 ZH 10-token 正确性：输出正常，无乱码，约 `4.648 tok/s`。
- 吞吐 A/B（实卡 Vulkan，串行，EN）：
  - 40-token：`3.646 tok/s`（subgroup ON） vs `3.611 tok/s`（subgroup OFF），`+1.0%`。
  - 80-token：`2.974 tok/s`（subgroup ON） vs `2.924 tok/s`（subgroup OFF），`+1.7%`。
  - 命中确认：`qmm_affine_bf16_t4_g128_m1_reduce_subgroup` 覆盖 `rows=1` 主桶（`8077/15957`）。
- 结论：
  - subgroup 路径带来小幅稳定正收益，说明方向正确。
  - 但距离“性能蜕变”仍远，下一阶段必须进入更激进的架构级重构（不仅是微调）。

### D-9.9a：QMM subgroup 核 `scale/bias` 共享缓存（已完成）
- 背景（Metal/Ollama 对照）：
  - 二者的高性能路径都强调“减少重复全局访存”，尤其在 decode 小 batch 场景下避免同一参数被 lane 重复加载。
  - 原 subgroup 核中同一 `g` 的 `scale/bias` 被 `ww` 维度重复读取（最多 16x 重复）。
- 变更：
  - 在 `qmm_affine_bf16_t4_g128_m1_reduce_subgroup.comp` 引入 workgroup 级 `shared` 缓存：
    - `scale0/bias0/scale1/bias1` 按 `groups_per_col` 预加载一次，再在主循环复用。
  - 对 subgroup 核增加安全条件：仅 `groups_per_col <= 256` 走该路径，避免共享缓存越界。
  - 维持 dispatch 失败自动降级到 `M1_REDUCE` 的安全护栏。
- 验证：
  - `python3 setup.py build_ext --inplace` 通过。
  - `ctest` 全量 `223/223` 通过。
  - `python/tests/test_fast_sdpa.py -v` 通过（`20 passed, 1 skipped`）。
  - Qwen3 ZH 10-token 正确性通过，无乱码。
- 结果：
  - 与 `M1_REDUCE` 对照仍是小幅正收益（约 `+1% ~ +1.7%`），说明访存复用方向有效，但尚不足以带来量级跃迁。

### D-9.10：命令缓冲提交窗口调参验证（已完成）
- 背景（Metal/Ollama 对照）：
  - 两条路线都强调减少不必要的提交/同步开销；因此验证 Kompute 序列提交窗口是否为当前瓶颈。
- 变更：
  - 在 `Device` 构造阶段新增 env 可调参数：
    - `MLX_VK_MAX_OPS_PER_BUFFER`（默认 `100`）
    - `MLX_VK_MAX_MB_PER_BUFFER`（默认 `50`）
  - 保持默认值不变，仅增加可观测与可调能力。
- 吞吐 A/B（实卡 Vulkan，串行，EN）：
  - 40-token：
    - `ops=100`: `3.617 tok/s`
    - `ops=64`: `3.567 tok/s`
    - `ops=32`: `3.473 tok/s`
    - `ops=400`: `3.537 tok/s`
  - 80-token：
    - `ops=100`: `2.901 tok/s`
    - `ops=400`: `2.901 tok/s`
- 结论：
  - 当前阶段“提交窗口”不是主瓶颈，`100` 已在本机口径下接近最优。
  - 后续主线继续回到 QMM/SDPA kernel 级架构优化，而不是 command-buffer 阈值调参。

### D-9.11：Add/Multiply bf16 broadcast-view native 化（已完成）
- 背景（Metal/Ollama 对照）：
  - Metal 与 Ollama 路线都强调“先清理高频小算子 fallback 尾部”，避免 CPU 往返吞掉 decode 吞吐。
  - profile_each 显示 decode 主链路中 `Add/Multiply` 仍有高频 fallback，且形态集中为 bf16 broadcast view（`data_size != size`）。
- 变更：
  - 新增 shader：
    - `add_bf16_scalar.comp`
    - `mul_bf16_scalar.comp`
    - `add_bf16_bcast.comp`
    - `mul_bf16_bcast.comp`
  - 新增 kernel 注册：
    - `ADD_BF16_SCALAR`、`ADD_BF16_BCAST`
    - `MUL_BF16_SCALAR`、`MUL_BF16_BCAST`
  - `binary.cpp` 新增 bf16 广播视图门禁与 push-const 索引映射（`ndim<=4`），并支持附加 push constants 派发。
  - 增加安全护栏：broadcast 维度/stride 必须可安全落在 `uint32` 范围，避免索引溢出。
  - 增加调试开关：`MLX_VK_DEBUG_BINARY_FALLBACK=1`（同签名仅打印一次）。
  - 严格执行 shader 闭环：`.comp -> .spv -> *_spv.h` 同轮完成并构建。
- 验证：
  - `python3 setup.py build_ext --inplace` 通过。
  - `ctest --test-dir build_release_vulkan --output-on-failure --timeout 120`：`223/223` 通过。
  - `PYTHONPATH=python python3 python/tests/test_fast_sdpa.py -v`：`20 passed, 1 skipped`。
  - Qwen3 EN/ZH 10-token 串行正确性通过，无乱码。
- 命中与收益（Qwen3 EN，实卡 Vulkan，串行）：
  - profile_each（40-token）：
    - `Add` fallback：`2392 -> 0`
    - `Multiply` fallback：`42 -> 0`
    - 总 fallback 比例：`15.51% -> 3.98%`
    - 剩余主要 fallback：`BitwiseBinary=672`、`Gather=126`、`fast::Quantize=42`
  - 吞吐 A/B（ON vs `MLX_VK_ENABLE_ADD_BF16=0 MLX_VK_ENABLE_MUL_BF16=0`）：
    - 40-token：`3.543 vs 3.277 tok/s`（`+8.1%`）
    - 80-token：`2.861 vs 2.580 tok/s`（`+10.9%`）

### D-9.12：BitwiseBinary 命中定位 + 小核实验结论（已完成）
- 背景（Metal/Ollama 对照）：
  - 两条路线都强调“先看命中，再决定是否值得做小算子专核”。
  - 本轮先通过签名日志确认 `BitwiseBinary` 的真实形态，再落最小可行 native 路径做 A/B。
- 命中定位（`MLX_VK_DEBUG_BITWISE_FALLBACK=1`）：
  - 主要形态固定为：
    - `LeftShift` / `RightShift`
    - `a/out: uint32` 行连续
    - `b: uint32` 标量广播视图（`strides=[0,0,0]`，`data_size=1`）
  - 典型 shape：`[1,12,128]`、`[1,1,128]`。
- 变更：
  - 新增实验 shader：`lshift_u32_scalar`、`rshift_u32_scalar`（含 `.comp -> .spv -> *_spv.h`）。
  - 新增 kernel 注册与调度路径：
    - gate：`MLX_VK_ENABLE_BITWISE_SHIFT_U32`
    - 默认值设为 `0`（OFF），仅作实验开关保留。
  - 新增调试开关：`MLX_VK_DEBUG_BITWISE_FALLBACK=1`（同签名仅打印一次）。
- A/B 结果（Qwen3 EN，实卡 Vulkan，串行）：
  - 命中：开启后 `BitwiseBinary fallback 672 -> 0`，整体 fallback 比例 `3.98% -> 0.80%`。
  - 吞吐（ON vs OFF）：
    - 40-token：`3.375 vs 3.549 tok/s`（`-4.9%`）
    - 80-token：`2.686 vs 2.863 tok/s`（`-6.2%`）
- 结论：
  - 单独把高频 tiny bitwise op 搬到 Vulkan 小核，受每次 dispatch 固定开销影响，端到端反而回退。
  - 与 Metal/Ollama 启发一致：这类路径更适合“融合到上游/下游大核或减少 launch 次数”，而不是独立微核堆叠。
  - 因此保留实验路径，但默认关闭，避免主线回退。

### D-9.13：`fast::Quantize` dequantize 主形态 native 化（已完成）
- 背景（Metal/Ollama 对照）：
  - Metal 在 `fast::Quantize` 上直接使用 dequantize kernel，而不是退回 CPU；Ollama 路线也将解码量化解包尽量留在 GPU 主链路。
  - 在 D-9.12 定位后确认：`BitwiseBinary` 大量 fallback 源于 `fast::Quantize` 的 fallback 子图，因此优先把该源头 native 化。
- 命中定位（`MLX_VK_DEBUG_FAST_QUANTIZE_FALLBACK=1`）：
  - 主形态固定为：
    - `dequantize=1`
    - `bits=4`
    - `group_size=128`
    - `mode=Affine`
    - `inputs=[uint32 packed_w, bf16 scales, bf16 biases]`
    - `outputs=[bf16 out]`
  - 典型 shape：`w=[1,12,128]` / `out=[1,12,1024]` 与 `w=[1,1,128]` / `out=[1,1,1024]`。
- 变更：
  - 新增 shader：`affine_dequantize_bf16_g128_b4.comp`（仅覆盖上述高频形态）。
  - 新增 kernel 注册：`AFFINE_DEQUANTIZE_BF16_G128_B4`。
  - `fast::Quantize::eval_gpu` 增加 native 派发：
    - gate：`MLX_VK_ENABLE_FAST_QUANTIZE_DEQ_AFFINE_B4`（默认 ON）。
    - 不命中或 dispatch 异常时保持原 fallback，语义不变。
  - 完成 shader 闭环：`.comp -> .spv -> *_spv.h`。
- 命中与收益（Qwen3 EN，实卡 Vulkan，串行）：
  - profile_each（40-token）：
    - `fast::Quantize fallback: 42 -> 0`
    - `BitwiseBinary fallback: 672 -> 0`（源头消除）
    - 总 fallback 比例：`3.98% -> 0.62%`
    - 剩余主要 fallback：`Gather=126`
  - 吞吐 A/B（ON vs `MLX_VK_ENABLE_FAST_QUANTIZE_DEQ_AFFINE_B4=0`）：
    - 40-token：`3.583 vs 3.526 tok/s`（`+1.6%`）
    - 80-token：`2.890 vs 2.882 tok/s`（`+0.3%`）
- 验证：
  - `ctest --test-dir build_release_vulkan --output-on-failure --timeout 120`：`223/223` 通过。
  - `PYTHONPATH=python python3 python/tests/test_fast_sdpa.py -v`：`20 passed, 1 skipped`。
  - Qwen3 EN/ZH 10-token 正确性通过，无乱码回归。

## 当前性能卡点（按优先级）
1. `QuantizedMatmul`  
- 仍是端到端主耗时大头；`M1_REDUCE_SUBGROUP` 仅带来 `~1%` 级增益，说明需要更深层内核重构（访存与解量化融合策略）。

2. `fast::RMSNorm / fast::ScaledDotProductAttention`
- 在 RoPE 回退清理后，二者成为次级原生耗时来源，需继续做 kernel 级效率优化。

3. 高频小算子 fallback 尾部  
- `Gather` 目前是 decode 口径下最主要的剩余 fallback 尾部（profile_each 样本 `126` 次）。
- `BitwiseBinary` 与 `fast::Quantize` 的高频回退已通过 D-9.13 从源头压降。

4. SDPA 长上下文效率  
- 当前 decode-unlimited + split-k 已稳定，但 `k=65+` 的 stage1/reduce 仍有进一步优化空间。

## 下一步（精确执行入口）
1. QMM decode 主核架构升级（对标 Metal/Ollama）  
- 目标：从“微调归约”升级到“访存/解量化/归约一体化”内核，优先作用于 `rows=1` 主桶。
- 方向：`uvec4`/向量化加载、scale/bias 访问复用、减少重复解包与无效算术、压缩指令路径。

2. SDPA decode 次热点并行推进  
- 对照 Metal/Ollama 的 decode attention 内核拆分策略，继续优化 `k=33~128` 区间的 stage1/reduce 配置与核形态。

3. 清理 `Gather` 尾部并推进融合调度  
- 继续沿 Metal/Ollama 的“减少 launch 数量 + 融合高频小算子”路线推进，下一站聚焦 `Gather`，并评估是否可并入上游/下游算子路径。

4. 维持命中优先与串行口径  
- 每轮必须先跑 `MLX_VK_QMM_STATS=1` / `MLX_VK_SDPA_STATS=1` 确认命中，再做优化与 A/B。

5. 门禁与评估口径保持一致  
- 继续用 EN/ZH `10/40/80` 串行压测 + profile 对照，确保“命中提升 = 吞吐提升”。

6. 研究流程约束  
- 每轮方案分析必须对照 Metal 与 Ollama 技术路径，并将结论与目标回写本文件。

## 标准验证命令（保留）
- C++ 全量：
  - `ctest --test-dir build_release_vulkan --output-on-failure --timeout 120`
- Python（SDPA 关键集）：
  - `PYTHONPATH=python python3 python/tests/test_fast_sdpa.py -v`
- Qwen3 正确性（实卡 Vulkan）：
  - `timeout 180s env LD_LIBRARY_PATH=build/temp.linux-x86_64-cpython-312/mlx.core/_deps/kompute-build/src:build/temp.linux-x86_64-cpython-312/mlx.core/_deps/fmt-build:build/lib.linux-x86_64-cpython-312/mlx/lib:$LD_LIBRARY_PATH VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json MESA_VK_DEVICE_SELECT=1002:1900 PYTHONPATH=python TARGET_DEVICE=gpu python3 -m mlx_lm generate --model Qwen/Qwen3-0.6B-MLX-4bit --prompt "你好啊" --max-tokens 10 --temp 0`
  - `timeout 180s env LD_LIBRARY_PATH=build/temp.linux-x86_64-cpython-312/mlx.core/_deps/kompute-build/src:build/temp.linux-x86_64-cpython-312/mlx.core/_deps/fmt-build:build/lib.linux-x86_64-cpython-312/mlx/lib:$LD_LIBRARY_PATH VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json MESA_VK_DEVICE_SELECT=1002:1900 PYTHONPATH=python TARGET_DEVICE=gpu python3 -m mlx_lm generate --model Qwen/Qwen3-0.6B-MLX-4bit --prompt "Hi what is your name" --max-tokens 10 --temp 0`

## 说明
- 旧版超长历史日志已压缩为本摘要；详细过程以 git 历史与相关 commit 为准。
