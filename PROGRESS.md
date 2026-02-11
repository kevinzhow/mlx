# PROGRESS

更新日期: 2026-02-11（精简版）

## 目标
- 对标 Metal backend 运行时契约，在 MLX 现有 GPU 路径内完成 Vulkan backend（Kompute v0.9.0）主线化。
- 优先级保持不变：先稳定与正确，再扩原生覆盖，最后做架构级性能提升。
- 开发流程约束：先做命中分析，再做优化；避免在低命中路径上投入主优化成本。
- 主线升级声明（2026-02-11）：
  - 后续性能目标以“架构升级驱动吞吐提升”为唯一主线，不再以门禁微调/小核试探作为主要提速手段。
  - 120 tok/s 为远期目标，短期评估指标改为“每轮架构变更带来可复现的端到端阶跃提升”。

## 当前快照
- 构建状态：`MLX_BUILD_VULKAN=ON` 下 `mlx` 与 `tests` 均可稳定构建链接。
- 回归状态：`ctest` 全量 `223/223` 通过；`python/tests/test_fast_sdpa.py` 通过（`20 passed, 1 skipped`）。
- 模型正确性：Qwen3-0.6B-MLX-4bit（EN/ZH，10 token）输出正常，无乱码/`!!!!!!!!!!` 回归。
- 流程约束（新增）：Qwen3 `mlx_lm generate` 正确性/性能口径需串行执行；任意 C++/Vulkan/shader 变更后，Python 侧验证前必须先执行 `python3 setup.py build_ext --inplace`，避免旧扩展产物导致误判。
- 近期吞吐（实卡 Vulkan，串行口径）：
  - 10-token（EN）：约 `5.20~5.23 tok/s`（D-9.25 后）
  - 40-token（EN）：约 `4.64~4.73 tok/s`（D-9.26 profiling refresh）
  - 80-token（EN）：约 `3.17~3.34 tok/s`（D-9.23 后）
- 运行时新默认（D-9.21）：
  - `MLX_VK_ENABLE_ALGO_CACHE=0`（默认 OFF）
  - 原因：当前 decode 主线算法缓存命中率为 `0`，默认关闭可减少无效 key/map 开销。
- 运行时新默认（D-9.22）：
  - `MLX_VK_MAX_INFLIGHT_SEQUENCES=8`
  - 语义：Vulkan 提交改为 `evalAsync`，并以 inflight 窗口做受控 `evalAwait`，对齐 Metal“提交异步、同步点显式”的行为模型。
- decode 主线 fallback 占比（同口径 profile）：
  - 由 `18.40%` 降到 `13.50%`（D-9.2），再降到 `3.98%`（D-9.11），当前进一步降到 `0.62%`（D-9.13）
- QMM decode `rows=1` 精确 `gpc` 命中（40-token 样本）：
  - `gpc=8`: `5918`
  - `gpc=16`: `1175`
  - `gpc=24`: `1175`
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

### D-9.14：`Gather` 行拷贝实验路径验证（已完成，默认 OFF）
- 背景（Metal/Ollama 对照）：
  - 在 D-9.13 后 `Gather` 成为剩余高频 fallback，先做最小可行“行拷贝”native 验证是否值得继续。
  - 参考 Metal/Ollama 的经验：若算子粒度过小且 launch 过多，独立 kernel 可能回退。
- 命中定位（`MLX_VK_DEBUG_GATHER_FALLBACK=1`）：
  - 主形态为 2D 行连续源张量按索引行采样：
    - `src`：`[151936, 8]` bf16 或 `[151936, 128]` uint32
    - `indices`：`[1,12]` / `[1,1]`，`int32` 或 `uint32`
    - `out`：`[*,*,1,W]`，行连续
- 变更：
  - 新增实验 shader：`gather_rows_words_i32_idx`（按“word copy”执行行采样）。
  - 新增 kernel 注册与 gate：
    - `MLX_VK_ENABLE_GATHER_ROWS_WORDS`（默认 `0`，OFF）
  - 新增调试开关：`MLX_VK_DEBUG_GATHER_FALLBACK=1`。
- A/B 结果（Qwen3 EN，实卡 Vulkan，串行）：
  - 命中：开启后 `Gather fallback 126 -> 0`。
  - 吞吐（ON vs OFF）：
    - 40-token：`3.317 vs 3.595 tok/s`（`-7.7%`）
  - profile_each（40-token）：
    - `fallback_pct 0.62% -> 0.00%`
    - 但 generation `3.163 vs 3.447 tok/s`（`-8.2%`）
- 结论：
  - 与 D-9.12 一致，独立 tiny gather kernel 受 dispatch 固定开销影响，端到端回退明显。
  - 保留实验路径与诊断能力，默认关闭；后续方向改为“融合/减少 launch”，不做默认独立 gather 专核。

### D-9.15：QMM `M1` 双-word/工作组子组核实验（已完成，默认 OFF）
- 背景（Metal/Ollama 对照）：
  - Metal/ollama 的高性能路径都强调“提高每次 dispatch 的有效工作量”，避免 decode 小 batch 下的重复访存。
  - 针对当前 QMM 主命中 `rows=1`，本轮验证“一个 workgroup 同时计算两个 `out_word`”是否能通过复用 `x` 读取带来收益。
- 变更：
  - 新增 shader：`qmm_affine_bf16_t4_g128_m1_reduce_subgroup_x2.comp`（subgroup kernel，一组计算 2 个 packed 输出 word）。
  - 新增 kernel 注册：`QMM_AFFINE_BF16_T4_G128_M1_REDUCE_SUBGROUP_X2`。
  - `QuantizedMatmul::eval_gpu` 增加分派与降级：
    - gate：`MLX_VK_ENABLE_QMM_NATIVE_M1_REDUCE_SUBGROUP_X2`。
    - 若 x2 kernel dispatch 异常，进程内自动关闭 x2 并回退到 `M1_REDUCE_SUBGROUP`。
  - 严格执行 shader 闭环：`.comp -> .spv -> *_spv.h`。
- 命中与结果（Qwen3 EN，实卡 Vulkan，串行）：
  - 命中确认（10-token, `MLX_VK_QMM_STATS=1`）：
    - `qmm_affine_bf16_t4_g128_m1_reduce_subgroup_x2`: `2167`
    - `qmm_affine_bf16_t4_g128_m16`: `191`
  - A/B（x2 ON vs OFF）：
    - 40-token：`3.583 vs 3.659 tok/s`（最近口径 `-2.1%`，无稳定正收益）
    - 80-token：`2.920 vs 2.916 tok/s`（基本持平）
- 结论：
  - x2 路径命中正常，但吞吐收益不稳定且存在回退窗口。
  - 按“无稳定收益不默认开启”的规则，当前设为默认 OFF（实验门禁保留，用于后续继续迭代）。
- 验证：
  - `python3 setup.py build_ext --inplace` 通过。
  - `ctest --test-dir build_release_vulkan --output-on-failure --timeout 120`：`223/223` 通过。
  - `PYTHONPATH=python python3 python/tests/test_fast_sdpa.py -v`：`20 passed, 1 skipped`。
  - Qwen3 EN 10-token 正确性通过，无乱码回归。

### D-9.16：QMM `M1 + groups_per_col==8` 专核（已完成，默认 ON）
- 背景（Metal/Ollama 对照）：
  - Metal/Ollama 路线都强调“先命中分析，再对主形态做硬特化”。
  - 本轮先扩展 QMM 统计，确认 decode `rows=1` 存在稳定高命中子形态（`groups_per_col==8`），再做专核。
- 命中分析（`MLX_VK_QMM_STATS=1`）：
  - `rows=1` 总命中：`8077`
  - 新专核命中：`5781`（约 `71.6%`）
  - 非 g8 子形态（继续走原 subgroup 核）：`2296`
  - 说明：`rows=1` 内部并非单一形态，`gpc==8` 是当前 Qwen3 decode 的主桶。
- 变更：
  - QMM 统计增强：新增 `NativeShapeBucket/FallbackShapeBucket`（`rows + gpc + k + n` 桶），避免后续“优化错桶”。
  - 新增 shader：`qmm_affine_bf16_t4_g128_m1_reduce_subgroup_g8.comp`（固定 `groups_per_col==8`，去掉动态 group 循环）。
  - 新增 kernel 注册：`QMM_AFFINE_BF16_T4_G128_M1_REDUCE_SUBGROUP_G8`。
  - `QuantizedMatmul::eval_gpu` 增加派发与降级：
    - gate：`MLX_VK_ENABLE_QMM_NATIVE_M1_REDUCE_SUBGROUP_G8`（默认 ON）
    - dispatch 异常时进程内自动关闭 g8 并回退到原 `M1_REDUCE_SUBGROUP`。
  - 严格执行 shader 闭环：`.comp -> .spv -> *_spv.h`。
- 吞吐 A/B（Qwen3 EN，实卡 Vulkan，串行）：
  - 40-token：`3.635 vs 3.531 tok/s`（`+2.9%`）
  - 80-token：`2.941 vs 2.880 tok/s`（`+2.1%`）
- 验证：
  - `python3 setup.py build_ext --inplace` 通过。
  - `ctest --test-dir build_release_vulkan --output-on-failure --timeout 120`：`223/223` 通过。
  - `PYTHONPATH=python python3 python/tests/test_fast_sdpa.py -v`：`20 passed, 1 skipped`。
  - Qwen3 ZH 10-token 正确性通过，无乱码回归。

### D-9.17：QMM `M1 + groups_per_col==16` 专核实验（已完成，默认 OFF）
- 背景（Metal/Ollama 对照）：
  - 在 `g8` 主桶之外，`rows=1` 次桶中 `gpc=9-16` 仍有稳定命中，按“命中优先”策略继续试探第二梯队专核。
- 变更：
  - 新增 shader：`qmm_affine_bf16_t4_g128_m1_reduce_subgroup_g16.comp`。
  - 新增 kernel 注册：`QMM_AFFINE_BF16_T4_G128_M1_REDUCE_SUBGROUP_G16`。
  - `QuantizedMatmul::eval_gpu` 增加 `groups_per_col==16` 派发与异常自动降级。
  - gate：`MLX_VK_ENABLE_QMM_NATIVE_M1_REDUCE_SUBGROUP_G16`。
  - 严格执行 shader 闭环：`.comp -> .spv -> *_spv.h`。
- 结果（Qwen3 EN，实卡 Vulkan，串行）：
  - 命中：`g16` 核可命中（`1148` 次/40-token 样本）。
  - A/B（g16 ON vs OFF）：
    - 40-token：`3.625 vs 3.643 tok/s`（`-0.5%`）
    - 80-token：`2.927 vs 2.936 tok/s`（`-0.3%`）
- 结论：
  - g16 专核当前实现无稳定正收益，默认保持 OFF（实验路径保留，后续可继续重写）。
- 验证：
  - `python3 setup.py build_ext --inplace` 通过。
  - `ctest --test-dir build_release_vulkan --output-on-failure --timeout 120`：`223/223` 通过。
  - `PYTHONPATH=python python3 python/tests/test_fast_sdpa.py -v`：`20 passed, 1 skipped`。
  - Qwen3 ZH 10-token 正确性通过，无乱码回归。

### D-9.18：QMM 精确 `gpc` 统计 + `gpc=24/32` 实验（已完成，默认 OFF）
- 背景（Metal/Ollama 对照）：
  - 对照路线都强调“先拿精确命中，再做特化”。此前 `17-32` 区间过粗，无法判断到底是 `24` 还是 `32` 主导。
- 变更：
  - QMM 统计新增精确 `gpc` 计数（`[VulkanQMMStats][NativeGPC]` / `[FallbackGPC]`）。
  - 新增 `gpc=24` 专核：
    - shader: `qmm_affine_bf16_t4_g128_m1_reduce_subgroup_g24.comp`
    - kernel: `QMM_AFFINE_BF16_T4_G128_M1_REDUCE_SUBGROUP_G24`
    - gate: `MLX_VK_ENABLE_QMM_NATIVE_M1_REDUCE_SUBGROUP_G24`（默认 OFF）
  - 新增 `gpc=32` 专核：
    - shader: `qmm_affine_bf16_t4_g128_m1_reduce_subgroup_g32.comp`
    - kernel: `QMM_AFFINE_BF16_T4_G128_M1_REDUCE_SUBGROUP_G32`
    - gate: `MLX_VK_ENABLE_QMM_NATIVE_M1_REDUCE_SUBGROUP_G32`（默认 OFF）
  - 二者均保留 dispatch 异常自动降级护栏。
  - 严格执行 shader 闭环：`.comp -> .spv -> *_spv.h`。
- 命中分析（40-token，`MLX_VK_QMM_STATS=1`）：
  - `gpc=24` 为真实次桶（`1175` 次）；`gpc=32` 在该样本未形成主命中。
- A/B（Qwen3 EN，实卡 Vulkan，串行）：
  - `gpc24 ON vs OFF`：
    - 40-token：`3.614/3.668` vs `3.649/3.617 tok/s`（无稳定增益）
    - 80-token：`3.039/2.994` vs `2.968/3.008 tok/s`（无稳定增益）
  - 结论：`gpc24` 当前实现不具备稳定收益，默认保持 OFF。
- 验证：
  - `python3 setup.py build_ext --inplace` 通过。
  - `ctest --test-dir build_release_vulkan --output-on-failure --timeout 120`：`223/223` 通过。
  - `PYTHONPATH=python python3 python/tests/test_fast_sdpa.py -v`：`20 passed, 1 skipped`。
  - Qwen3 ZH 10-token 正确性通过，无乱码回归。

### D-9.19：QMM `M1` subgroup 架构化减冗余（已完成，默认 ON）
- 背景（Metal/Ollama 对照）：
  - Metal 与 Ollama 的 decode 热核都强调两点：
    - 波内（subgroup/wave）优先归约，避免不必要的二次归约与 barrier；
    - 对高频形态预展开 dequant，减少内层重复 `scale*q+bias` 标量算术。
  - 当前 Qwen3 decode 主桶仍为 `rows=1 && gpc=8`，因此优先在该路径做“内核级减冗余”而非继续机械分桶。
- 变更：
  - `qmm_affine_bf16_t4_g128_m1_reduce_subgroup_g8.comp`：
    - 新增 `gpc=8` 的 dequant 查表缓存（`q in [0,15]`），将 `scale*q+bias` 从内层循环搬到组级预展开；
    - `x` 读取改为一次 `uvec4` 批量加载并展开累加；
    - 增加 `gl_NumSubgroups==1` 快路径，跳过共享内存二次归约。
  - 同步在 `m1_reduce_subgroup` / `g16` / `g24` / `g32` / `x2` 核增加单-subgroup 快路径，减少 decode 主线上的无效 barrier/二次 `subgroupAdd`。
  - 严格执行 shader 闭环：`.comp -> .spv -> *_spv.h`（6 个 subgroup shader 同轮重编与头文件更新）。
- 命中与收益（Qwen3 EN，实卡 Vulkan，串行）：
  - 命中确认（`MLX_VK_QMM_STATS=1`，40-token）：
    - `qmm_affine_bf16_t4_g128_m1_reduce_subgroup_g8`: `5781`
    - `qmm_affine_bf16_t4_g128_m1_reduce_subgroup`: `2296`
    - `qmm_affine_bf16_t4_g128_m16`: `191`
    - `NativeGPC`: `gpc=8:5918`, `gpc=16:1175`, `gpc=24:1175`
  - A/B（`MLX_VK_ENABLE_QMM_NATIVE_M1_REDUCE_SUBGROUP_G8=1` vs `0`）：
    - 40-token：`3.620/3.689` vs `3.570/3.514 tok/s`（约 `+3.2%`）
    - 80-token：`2.979` vs `2.918 tok/s`（约 `+2.1%`）
- 验证：
  - `python3 setup.py build_ext --inplace` 通过。
  - `ctest --test-dir build_release_vulkan --output-on-failure --timeout 120`：`223/223` 通过。
  - `PYTHONPATH=python python3 python/tests/test_fast_sdpa.py -v`：`20 passed, 1 skipped`。
  - Qwen3 ZH 10-token 正确性通过（无乱码回归）。
- 结论：
  - 方向正确（稳定小幅正收益），但仍未到“量级提升”。
  - 下一步应继续沿 Metal/Ollama 路线做更深层结构改造：把 `rows=1` 次桶（`gpc=16/24`）也并入同类“预展开+向量化”范式，并评估跨层融合减少 decode launch 数。

### E-9.19a：Generic `M1` dequant 查表扩展实验（已回退）
- 背景（Metal/Ollama 对照）：
  - 按“减少内层重复算术”思路，尝试把 `M1` generic 核也改为 `dequant(q=0..15)` 预展开，以提升 `gpc=16/24` 次桶效率。
- 实现与验证：
  - 已实现并完成 `.comp -> .spv -> *_spv.h` 闭环，且 `ctest`/`test_fast_sdpa` 均通过。
  - 但串行 Qwen3 A/B 显示稳定回退：
    - `g8 on`：40-token `3.49~3.58 tok/s`（低于回退前口径）
    - `g8 off`（强制更多 generic 命中）：40-token `~3.05 tok/s`（明显回退）
- 结论与处置：
  - 推断为共享内存占用/寄存器压力抬升导致 occupancy 下降，抵消了算术减冗余收益。
  - 已完整回退 generic 查表改动，仅保留 D-9.19 已验证正收益的改动（`g8` 查表 + 单-subgroup 快路径）。
  - 回退后复测恢复：40-token `3.673 tok/s`，`g8 on/off` 仍保持正向差值（`3.673 vs 3.571`）。
  - 同轮复测 `g16/g24` 默认 OFF 门禁（单次筛查）：
    - 40-token：`g16 on 3.680`、`g24 on 3.680`、`g16+g24 on 3.670` vs `default 3.642 tok/s`
    - 80-token：`g16+g24 on 2.959` vs `default 2.955 tok/s`
    - 结论：当前仅见边际波动，尚不足以改默认门禁，继续保持 `G16/G24=OFF`。

### E-9.19b：`g16/g24` 专核 dequant 查表实验（已回退）
- 背景（Metal/Ollama 对照）：
  - 延续 D-9.19 的思路，尝试将 `g16/g24` 专核也切换为固定小共享内存的 `dequant(q=0..15)` 查表，以提升次桶效率。
- 实验过程：
  - 已完成 `.comp -> .spv -> *_spv.h` 闭环并重编，串行口径下做多轮 Qwen3 EN 对照（40/80 token）。
- 结果（关键样本）：
  - 40-token：
    - `default`: `3.682`, `3.639 tok/s`
    - `g16 on`: `3.667`, `3.660 tok/s`
  - 80-token：
    - `default`: `2.936`, `2.999 tok/s`
    - `g16 on`: `2.912`, `2.949 tok/s`
  - 结论：无稳定正收益，且存在回退窗口。
- 处置：
  - 已回退 `g16/g24` 查表实现，保留现有稳定版本与默认门禁（`G16/G24=OFF`）。

### D-9.20：A1-Phase1 `gpc=8` 双-word tile（`g8_x2`）架构实验（已完成，默认 OFF）
- 背景（Metal/Ollama 对照）：
  - Metal/ollama 的 decode 热核都遵循“每次 launch 做更多有效工作”的原则，核心是复用激活读取、减少碎片化执行。
  - 当前主桶仍是 `rows=1 && gpc=8`，因此 A1 第一阶段选择在该桶验证“一个 workgroup 计算两个 `out_word`”。
- 方案与实现：
  - 新增 shader：`qmm_affine_bf16_t4_g128_m1_reduce_subgroup_g8_x2.comp`。
  - 关键设计：一个 WG 同时计算 2 个 packed `out_word`（4 个 bf16 输出），共享同一批 `x` 读取；`scale/bias` 仍在组内预展开为 `dequant(q=0..15)` 查表。
  - 新增 kernel 注册：`QMM_AFFINE_BF16_T4_G128_M1_REDUCE_SUBGROUP_G8_X2`。
  - 新增 gate：`MLX_VK_ENABLE_QMM_NATIVE_M1_REDUCE_SUBGROUP_G8_X2`（默认 OFF）。
  - 派发策略：仅在 `rows=1 && gpc=8 && out_words>=2` 命中；dispatch 失败时自动禁用该路径并降级回 `g8` 单-word 专核。
  - 严格执行 shader 闭环：`.comp -> .spv -> *_spv.h`。
- 验证：
  - `python3 setup.py build_ext --inplace` 通过。
  - `ctest --test-dir build_release_vulkan --output-on-failure --timeout 120`：`223/223` 通过。
  - `PYTHONPATH=python python3 python/tests/test_fast_sdpa.py -v`：`20 passed, 1 skipped`。
  - Qwen3 ZH/EN 正确性通过，无乱码回归。
- 吞吐 A/B（Qwen3 EN，实卡 Vulkan，串行）：
  - 40-token：`3.649`（ON） vs `3.604 tok/s`（OFF），`+1.2%`
  - 80-token：`2.978`（ON） vs `3.010 tok/s`（OFF），`-1.1%`
  - 结论：无稳定正收益，保持默认 OFF（实验路径保留）。
- 理论结论（A1 阶段性）：
  - `x` 复用方向正确，但当前实现在寄存器/活跃线程占用上的代价抵消了收益。
  - 下一步应优先进入“更低寄存器压力的 tile 方案 + 更少调度碎片”的架构重构，而不是继续做同构分支扩张。

### E-9.20a：A1.2 `g8_x2` 寄存器生命周期重排实验（已完成，默认 OFF）
- 背景（Metal/Ollama 对照）：
  - Metal/ollama 在 decode 热核上都强调“提高每次 launch 有效工作量”的同时控制 occupancy，不接受仅靠局部循环展开造成的寄存器膨胀。
  - D-9.20 的 `g8_x2` 初版已验证命中正确，但收益不稳定；本轮针对其寄存器压力做内核重排。
- 改动：
  - 重写 `qmm_affine_bf16_t4_g128_m1_reduce_subgroup_g8_x2.comp` 内层累加逻辑：
    - 移除 `accumulate_unit` 大量局部临时变量路径。
    - 改为 `unit x t` 双循环中按需解包 nibble，缩短变量 live-range，减少寄存器峰值。
  - 严格执行 shader 闭环：`.comp -> .spv -> *_spv.h` 并重编。
- 验证：
  - `python3 setup.py build_ext --inplace` 通过。
  - `ctest --test-dir build_release_vulkan --output-on-failure --timeout 120`：`223/223` 通过。
  - `PYTHONPATH=python python3 python/tests/test_fast_sdpa.py -v`：`20 passed, 1 skipped`。
  - Qwen3 ZH 10-token 正确性通过（无乱码回归）。
- A/B（Qwen3 EN，实卡 Vulkan，串行）：
  - 40-token：`3.644`（ON） vs `3.639 tok/s`（OFF），近乎持平
  - 80-token：`2.951`（ON） vs `3.010 tok/s`（OFF），约 `-2.0%`
  - 命中确认：`qmm_affine_bf16_t4_g128_m1_reduce_subgroup_g8_x2` 命中计数与预期一致（`gpc=8` 主桶）。
- 结论：
  - 仅靠 `g8_x2` 内核级寄存器重排仍不足以形成端到端稳定收益，默认继续 OFF。
  - 下一步转向 A1.3：先量化 decode token-step 的 CPU/dispatch 固定开销，再决定“提交模型升级/链路融合”的具体落点。

### D-9.21：算法缓存命中观测 + 默认策略切换（已完成）
- 背景（Metal/Ollama 对照）：
  - Metal/Ollama 的高性能路径依赖可复用 pipeline/graph；若缓存层长期 0 命中，则该层应降级为“无缓存直通”避免固定开销。
  - 本轮先补齐可观测性，再依据数据做默认策略切换。
- 变更：
  - `kernel_registry.cpp` 新增 `MLX_VK_ALGO_STATS=1` 统计：
    - `requests/cache_hits/cache_misses`；
    - 按 kernel 的 miss 分布。
  - 新增算法缓存运行开关与策略：
    - `MLX_VK_ENABLE_ALGO_CACHE`（默认改为 `0`，OFF）；
    - `MLX_VK_ENABLE_ALGO_CACHE_AUTO_DISABLE`（默认 `1`）；
    - `MLX_VK_ALGO_CACHE_ZERO_HIT_DISABLE_THRESHOLD`（默认 `2048`）。
- 关键观测（Qwen3 EN，40/80 token）：
  - 算法缓存请求量高（`20k~39k`），但 `cache_hits=0`。
  - miss 主分布：`qmm_affine_bf16_t4_g128_m1_reduce_subgroup_g8`、`rmsnorm_bf16`、`rope_bf16_t1`、`add_bf16` 等 decode 高频核。
- 结论：当前“按 tensor 身份缓存 algorithm”在该主线 workload 上基本无收益。
- 吞吐 A/B（实卡 Vulkan，串行）：
  - 40-token：
    - cache ON：`3.678 tok/s`
    - cache OFF：`3.871 tok/s`（约 `+5.2%`）
  - 80-token：
    - cache ON：`2.974 tok/s`
    - cache OFF：`3.251 tok/s`（约 `+9.3%`）
  - 默认切换后复测（不加额外 env）：
    - 40-token：`3.839 tok/s`
    - 80-token：`3.256 tok/s`
- 验证：
  - `python3 setup.py build_ext --inplace` 通过。
  - `ctest --test-dir build_release_vulkan --output-on-failure --timeout 120`：`223/223` 通过。
  - `PYTHONPATH=python python3 python/tests/test_fast_sdpa.py -v`：`20 passed, 1 skipped`。
  - Qwen3 EN/ZH 正确性通过，无乱码回归。

### D-9.22：提交模型升级（`evalAsync + inflight`，已完成）
- 背景（Metal/Ollama 对照）：
  - Metal 路线中 command buffer 提交是异步的，host 侧只在显式同步点等待；Ollama Vulkan 路线也强调“减少每 token 的同步阻塞”。
  - Vulkan 旧实现在 `commit_command_buffer()` 中使用 `sequence->eval()`（同步），与 Metal 提交语义不一致，且在小核链路下会放大固定等待成本。
- 变更：
  - `Device::commit_command_buffer` 从同步 `eval()` 改为 `evalAsync()`。
  - 新增 `DeviceStream::inflight_sequences`，并新增受控等待：
    - `Device::await_inflight_sequences_()`：仅当 inflight 超过窗口时等待最早批次。
    - `Device::wait_for_stream()`：提交当前录制并等待该 stream 所有 inflight 完成（用于 `synchronize` 和 host copy 前）。
  - `gpu::synchronize` 改为调用 `wait_for_stream()`，保证显式同步点语义清晰。
  - 新增调参环境变量：
    - `MLX_VK_MAX_INFLIGHT_SEQUENCES`（默认 `8`，范围 `1..64`）。
- 验证：
  - `cmake --build build_release_vulkan --target tests -j 4` 通过。
  - `ctest --test-dir build_release_vulkan --output-on-failure --timeout 120`：`223/223` 通过。
  - `python3 setup.py build_ext --inplace` 通过。
  - `PYTHONPATH=python python3 python/tests/test_fast_sdpa.py -v`：`20 passed, 1 skipped`。
  - Qwen3 ZH 10-token 正确性通过，无乱码回归。
- 吞吐结果（Qwen3 EN，实卡 Vulkan，串行）：
  - 默认（`MLX_VK_MAX_INFLIGHT_SEQUENCES=8`）：
    - 40-token：`3.795 / 3.848 tok/s`
    - 80-token：`3.177 / 3.218 tok/s`
  - 对照（本轮快照）：
    - `MLX_VK_MAX_INFLIGHT_SEQUENCES=1`：80-token `3.177 tok/s`
    - `MLX_VK_MAX_INFLIGHT_SEQUENCES=8`：80-token `3.237 tok/s`（单次）
- 结论：
  - 架构契约层面已与 Metal 更一致（异步提交 + 显式同步），并建立了后续“更大提交窗口/融合调度”基础。
  - 端到端吞吐目前仅边际波动，说明“提交同步”不是当前唯一主瓶颈；下一阶段仍需把主力放在“减少 launch 数 + QMM/SDPA 核形态升级”。

### D-9.23：dirty tensor 同步索引化（已完成）
- 背景（Metal/Ollama 对照）：
  - Metal 路线在 host-visible 同步点上倾向于“增量/目标化同步”，避免每次同步遍历全局资源。
  - Vulkan 现状中 `sync_dirty_tensors_for_stream` 每次 `synchronize` 都全量扫描 `tensor_cache_`，decode 多步场景会放大 CPU 固定成本。
- 变更：
  - 在 `Device` 中新增按 stream 的 dirty 索引：
    - `dirty_tensors_by_stream_`（`stream_index -> {keys, key_set}`）。
  - `mark_tensor_host_dirty` 命中条目后同步维护 dirty 索引（支持跨 stream 迁移时反注册旧 stream）。
  - `sync_dirty_tensors_for_stream` 从“全量扫描 tensor_cache_”改为“仅遍历该 stream 的 dirty keys”。
  - `invalidate_tensor` 与 `sync_array_to_host_if_needed` 完成后同步清理 dirty 索引，避免陈旧键累积。
- 验证：
  - `cmake --build build_release_vulkan --target tests -j 4` 通过。
  - `ctest --test-dir build_release_vulkan --output-on-failure --timeout 120`：`223/223` 通过（串行）。
  - `python3 setup.py build_ext --inplace` 通过。
  - `PYTHONPATH=python python3 python/tests/test_fast_sdpa.py -v`：`20 passed, 1 skipped`。
  - Qwen3 ZH 10-token 正确性通过，无乱码回归。
- 吞吐结果（Qwen3 EN，实卡 Vulkan，串行）：
  - 40-token：`3.878 tok/s`
  - 80-token：`3.339 tok/s`
- 结论：
  - 该改动属于“架构级 CPU 固定开销削减”，对 decode 吞吐有可见正收益。
  - 下一步应继续沿同方向推进：减少 `OpAlgoDispatch` 数量（同层合并/链路融合），而不是继续微调单核门禁。

### D-9.24：tensor cache 二级索引化（按底层存储身份）（已完成）
- 背景（Metal/Ollama 对照）：
  - 两条路线都强调 decode 热链路要尽量减少 host 侧运行时管理开销（descriptor/buffer bookkeeping），否则会吞掉小 batch 生成吞吐。
  - Vulkan 现状中 `get_tensor/mark_tensor_host_dirty/tensor_needs_sync_device/sync_array_to_host_if_needed` 在 alias/view 命中时依赖 `tensor_cache_` 全表扫描，decode 多步场景 CPU 固定成本偏高。
- 变更：
  - 在 `Device` 中新增按底层存储身份的二级索引：
    - key：`(data_ptr, data_owner, dtype)`
    - value：候选 tensor-cache keys 列表。
  - 新增统一查找/清理助手：
    - `find_tensor_entry_locked_`：先查直接 key，再走二级索引选最大可用 `elem_count` 候选，替代全表扫描。
    - `erase_tensor_entry_locked_`：统一维护 `tensor_cache_`、`tensor_storage_index_`、dirty-tracker 三者一致性。
  - `invalidate_tensor/mark_tensor_host_dirty/tensor_needs_sync_device/sync_array_to_host_if_needed` 全部切换到新查找路径。
- 验证：
  - `cmake --build build_release_vulkan --target mlx -j 8` 通过。
  - `cmake --build build_release_vulkan --target tests -j 8` 通过。
  - `ctest --test-dir build_release_vulkan --output-on-failure --timeout 120`：`223/223` 通过。
  - `python3 setup.py build_ext --inplace` 通过。
  - `PYTHONPATH=python python3 python/tests/test_fast_sdpa.py -v`：`20 passed, 1 skipped`。
  - Qwen3 EN（实卡 Vulkan，串行）：
    - 10-token：`5.232 / 5.121 tok/s`
    - 40-token：`4.469 tok/s`
- 结论：
  - 本轮属于“运行时架构开销削减”而非门禁调参，端到端吞吐出现可复现阶跃提升（约 `+15%` 量级，相对 D-9.23 口径）。
  - 下一步继续推进 A1.3b/A1.5：直接压缩 decode 热链路 `OpAlgoDispatch` 次数（同层合并/链路融合）。

### D-9.25：tensor sync 批量化（减少碎片化 sync op）（已完成）
- 背景（Metal/Ollama 对照）：
  - 两条路线都强调 decode token-step 中减少碎片化 launch/sync；在小 batch 下，host 侧固定调度成本非常敏感。
  - Vulkan 现状中 `record_tensor_sync_device/local` 对每个 tensor 都单独记录一条 op，导致额外 op 数量膨胀。
- 变更：
  - `CommandEncoder::record_tensor_sync_device`：
    - 从“逐 tensor `OpTensorSyncDevice`”改为“去重后批量一次 `OpTensorSyncDevice`”。
  - `CommandEncoder::record_tensor_sync_local`：
    - 同样改为去重后批量一次 `OpTensorSyncLocal`。
  - `Device::sync_dirty_tensors_for_stream`：
    - host 回拷前的 `OpTensorSyncLocal` 由多条单 tensor 记录改为批量记录。
- 验证：
  - `cmake --build build_release_vulkan --target mlx -j 8` 通过。
  - `cmake --build build_release_vulkan --target tests -j 8` 通过。
  - `ctest --test-dir build_release_vulkan --output-on-failure --timeout 120`：`223/223` 通过。
  - `python3 setup.py build_ext --inplace` 通过。
  - `PYTHONPATH=python python3 python/tests/test_fast_sdpa.py -v`：`20 passed, 1 skipped`。
  - Qwen3 EN（实卡 Vulkan，串行）：
    - 10-token：`5.208 tok/s`
    - 40-token：`4.644 tok/s`
- 结论：
  - 本轮继续验证“减少碎片调度”方向有效；相对 D-9.24 的 40-token `4.469 tok/s`，再提升约 `+3.9%`。
  - 下一步继续围绕 A1.5：把 decode 热链路里的 dispatch 数量进一步压降（优先从高频 `QMM/RMSNorm/RoPE` 周边可融合段入手）。

### D-9.26：decode kernel dispatch 分布复测（已完成）
- 背景：
  - 按 A1.5 路线，继续优化前先刷新当前主线的真实 dispatch 分布，避免盲目改低贡献路径。
- 观测（Qwen3 EN 40-token，实卡 Vulkan，`MLX_VK_ALGO_STATS=1`）：
  - 吞吐：`4.725 tok/s`
  - `VulkanAlgoStats`：
    - `requests=20227`，`cache_hits=0`，`miss_kernels=14`
    - Top miss kernels：
      - `qmm_affine_bf16_t4_g128_m1_reduce_subgroup_g8`: `5781`
      - `rmsnorm_bf16`: `4743`
      - `rope_bf16_t1`: `2351`
      - `add_bf16`: `2350`
      - `qmm_affine_bf16_t4_g128_m1_reduce_subgroup`: `2296`
- 结论：
  - 当前瓶颈仍是“高频小链路重复调度 + QMM/RMSNorm 主核高频调用”，而不是单一 fallback 问题。
  - 下一步执行入口保持 A1.5：优先寻找可稳定落地的链路融合点（先从 `add_bf16 + rmsnorm_bf16` 邻接段可融合性评估与实现切入）。

### D-9.27：RMSNorm 链路观测钩子回退（已完成）
- 背景：
  - 为验证 `add_bf16 + rmsnorm_bf16` 融合价值，尝试在 `fast::RMSNorm::eval_gpu` 内增加输入 primitive 统计钩子。
- 结果：
  - 该 intrusive 钩子在 Qwen3 40-token 流程中触发运行时崩溃（segmentation fault）。
  - 已完整回退该钩子，重新 `build_ext` 验证后，Qwen3 生成恢复稳定。
- 结论：
  - 后续不再在 primitive 热路径引入生命周期敏感统计逻辑。
  - 链路观测改为“非侵入式”方式：`MLX_VK_ALGO_STATS` + 现有 profile/日志口径联合分析。

### D-9.28：`add + RMSNorm` 图融合与原生核闭环（已完成）
- 背景（Metal/Ollama 对照）：
  - Metal/Ollama 都强调先减少 decode 热链路中的小核串联，再看单核效率；本轮先做低风险链路融合验证。
  - 目标：把 `add_bf16 -> rmsnorm_bf16` 的一段邻接链路收敛为单次 dispatch，验证“减少 launch 数量”是否带来端到端收益。
- 实现：
  - fast 图侧：
    - `rms_norm(add(a, b), w)` 自动生成 fused `RMSNorm` primitive（3 输入：`a/b/w`）。
    - `RMSNorm::vjp` 支持 fused-add 形态（`da=dx`, `db=dx`, `dw` 按原公式）。
  - Vulkan 侧：
    - 新增 kernel：`add_rmsnorm_bf16`。
    - 新增 gate：`MLX_VK_ENABLE_ADD_RMSNORM_NATIVE`（默认 ON）。
    - 新增命中判定与 dispatch：`fast::RMSNorm::eval_gpu` 对 3 输入路径优先走 native。
  - shader 闭环（严格执行）：
    - 已完成 `.comp -> .spv -> *_spv.h`：
      - `mlx/backend/vulkan/shaders/add_rmsnorm_bf16.comp`
      - `mlx/backend/vulkan/shaders/add_rmsnorm_bf16.spv`
      - `mlx/backend/vulkan/shaders/add_rmsnorm_bf16_spv.h`
  - 工程接线：
    - `kernel_registry` + `CMakeLists.txt` 已注册新核。
    - 修正 gate 语义：当 `MLX_VK_ENABLE_ADD_RMSNORM_NATIVE=0` 时 fast 图侧不构建 fused primitive，确保 A/B 对照回到旧路径。
- 验证：
  - `ctest --test-dir build_release_vulkan --output-on-failure --timeout 120`：`223/223` 通过。
  - `PYTHONPATH=python python3 python/tests/test_fast_sdpa.py -v`：`20 passed, 1 skipped`。
  - Qwen3 正确性（实卡 Vulkan）：
    - ZH 10-token 正常，无乱码。
  - 命中（EN 10-token, `MLX_VK_ALGO_STATS=1`）：
    - `add_rmsnorm_bf16: 670`（确认命中）
    - `rmsnorm_bf16: 683`（开启融合） vs `1353`（关闭融合）
- 结果与结论：
  - 端到端吞吐未出现稳定提升（EN 40-token：ON `4.665 tok/s` vs OFF `4.719 tok/s`，近似持平）。
  - 说明本轮“减少一段小核 dispatch”方向在当前阶段杠杆不足；主瓶颈仍是 QMM 主耗时与更高层执行图碎片。
  - 下一步回到架构主线：优先推进 QMM 主核与更大粒度 decode 子图融合（而非继续堆小融合核）。

### D-9.29：QMM `g16/g24` 专核默认门禁复测（已完成，维持默认 OFF）
- 背景（命中先行）：
  - `MLX_VK_QMM_STATS=1` 显示 decode 口径下 `groups_per_col=16/24` 具备稳定命中（各约 `~300` 次 / 10-token 样本）。
  - 评估是否将 `MLX_VK_ENABLE_QMM_NATIVE_M1_REDUCE_SUBGROUP_G16/G24` 从默认 OFF 调整为默认 ON。
- 复测结果（实卡 Vulkan，Qwen3 EN 40-token，串行）：
  - ON 样本：`4.761 tok/s`、`4.745 tok/s`
  - OFF 样本：`4.670 tok/s`、`4.658 tok/s`
  - 同轮后续复测出现 `4.635~4.785 tok/s` 波动，整体提升未形成稳定可复现阶跃（约 `<2%` 边界区间）。
- 结论：
  - 按“<~2% 且不稳定不入默认”规则，本轮不调整默认门禁，`G16/G24` 继续作为可选实验开关。
  - 该方向可保留到后续更大改动（如 QMM 主核重构）后再统一复评。

## 当前性能卡点（按优先级）
1. `QuantizedMatmul`  
- 仍是端到端主耗时大头；`g8_x2` A1-Phase1 与 A1.2 重排均未形成稳定收益，后续需转向更深层内核重构（数据布局 + 中间态组织）。

2. 高频 decode 核的 dispatch/创建开销  
- D-9.21 已证明旧 algorithm cache 对主线为 0 命中并已默认关闭。
- D-9.22 已完成“提交异步化 + inflight 窗口”但收益有限；下一阶段需继续减少 `manager.algorithm`/`OpAlgoDispatch` 总次数（链路融合/dispatch 合并）。
- D-9.28 已验证 `add+rmsnorm` 小粒度融合本身不足以带来稳定吞吐提升，后续需转向更大粒度热链路重构。

3. `fast::RMSNorm / fast::ScaledDotProductAttention`
- 在 RoPE 回退清理后，二者仍是次级原生耗时来源，需继续做 kernel 级效率优化。

4. 高频小算子 fallback 尾部  
- `Gather` 目前是 decode 口径下最主要的剩余 fallback 尾部（profile_each 样本 `126` 次）。
- `BitwiseBinary` 与 `fast::Quantize` 的高频回退已通过 D-9.13 从源头压降。
- `Gather` 已验证独立专核会回退，后续应优先寻找融合路径而非继续堆叠 tiny kernel。

5. SDPA 长上下文效率  
- 当前 decode-unlimited + split-k 已稳定，但 `k=65+` 的 stage1/reduce 仍有进一步优化空间。

## 下一步（精确执行入口）
1. 架构主线 A：QMM decode 核重构（最高优先级）  
- 对标 Metal/Ollama：从“单 op 专核”升级为“tile + 预打包 + 最小中间写回”的主核路径。
- 目标：显著降低每 token 的 QMM kernel 数量与每次 dispatch 固定开销，提升单核有效算术密度。
- 实施要点：
  - A1.2：已完成一轮 `g8_x2` 寄存器重排验证（E-9.20a），无稳定收益，保持实验路径 OFF。
  - A1.3：已完成“提交模型升级（D-9.22）”第一阶段；下一步进入 A1.3b，聚焦真正减少 `OpAlgoDispatch` 次数（同层合并/链路融合），而非仅异步化提交。
  - A1.4：已完成算法缓存命中可观测性与默认策略切换（D-9.21）。
  - A1.5：对齐 Metal/Ollama 路线推进“少 dispatch 高负载”执行图，优先把 decode 热链路（QMM->RMSNorm->RoPE->SDPA）的 launch 数量压降到当前的一半量级。
  - A1.6（下一入口）：从“小融合核”切到“QMM 主核末端融合（epilogue）”评估，实现 `QMM + residual add + norm` 的单核化原型，对齐 Metal/Ollama 在主核内完成更多后处理的路线。

2. 架构主线 B：decode 链路融合/提交模型升级  
- 对标 Metal command-buffer 与 Ollama 持续化 decode 路线，减少 token-step 内 kernel 启动碎片。
- 目标：把“多小核串联”改造成“少核高负载”执行图，优先处理 QMM->RMSNorm->RoPE->SDPA 热链路。

3. 架构主线 C：SDPA decode 核形态升级（并行）  
- 在保持正确性的前提下，推进 `k=33~128` 及更长上下文路径的核重构与中间结果压缩，避免 stage1/reduce 过碎调度。

4. 评估与节奏约束（保持）  
- 每轮先跑 `MLX_VK_QMM_STATS=1` / `MLX_VK_SDPA_STATS=1`，再做 EN/ZH `10/40/80` 串行 A/B。
- 若改动仅带来边际波动（<~2%）且不稳定，视为“未达架构升级目标”，不进入默认路径。

5. 研究约束（保持）  
- 每轮方案必须显式对照 Metal 与 Ollama 技术实现，并把结论与下一目标回写本文件。

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
