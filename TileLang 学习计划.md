---
tags: [学习/TileLang, work]
created: 2026-10-31
repo: ~/code/personal/ai/tilelang
---

# TileLang 学习计划

> 目标：会用 TileLang 写高性能算子（GEMM / Attention），理解从 Python DSL → TIR → Pass → CUDA/Ascend 代码生成的完整编译链路，最终能给新硬件后端（如 AGIC）做适配。
>
> 进度标记：`[ ]` 未开始　`[/]` 进行中　`[x]` 完成

## 进度总览

| 阶段 | 主题 | 预计时间 | 状态 |
| --- | --- | --- | --- |
| 0 | 前置知识 | 3-5 天 | 未开始 |
| 1 | 上手使用 | 3 天 | 未开始 |
| 2 | 语言与编程模型 | 1 周 | 未开始 |
| 3 | 经典算子实战 | 1-2 周 | 未开始 |
| 4 | 编译器内部（Pass 链路） | 2 周 | 未开始 |
| 5 | Runtime / JIT / 后端 | 1 周 | 未开始 |
| 6 | 进阶专题 & 输出 | 持续 | 未开始 |

---

## 阶段 0：前置知识

- [ ] GPU 执行模型：Grid / Block / Warp / Thread，SM、Shared Memory、寄存器 → [学习笔记：GPU 执行模型入门](<GPU 执行模型入门.md>)
- [ ] 内存层级与访存优化：coalescing、bank conflict、L2 → [学习笔记：GPU 内存层级与访存优化](<GPU 内存层级与访存优化.md>)
- [ ] Tensor Core / MMA 基本概念（`mma.sync`、`wgmma`、TMA 了解即可） → [学习笔记：Tensor Core 与 MMA 入门](<Tensor Core 与 MMA 入门.md>)
- [ ] TVM 基础：`PrimFunc`、`IRModule`、`Buffer`、TIR Stmt/Expr、Pass 机制
- [ ] 对比阅读：Triton 的编程模型（tile 级别抽象），理解 TileLang 定位

## 阶段 1：上手使用

- [ ] 读 `README.md`、`docs/get_started/overview.md`、`docs/get_started/Installation.md`
- [ ] 编译 / 安装 TileLang（Mac 上无 CUDA，可在 x86 服务器上装）
- [ ] 跑通 `examples/quickstart.py`
- [ ] 跑通 `examples/elementwise`、`examples/gemm` 中的基础例子
- [ ] 学会看生成的代码：`kernel.get_kernel_source()`
- [ ] 读 `docs/tutorials/debug_tools_for_tilelang.md`、`docs/tutorials/logging.md`，学会打印 IR / 调试

## 阶段 2：语言与编程模型

文档：`docs/programming_guides/`

- [ ] `overview.md`、`language_basics.md`：`T.prim_func`、`T.Kernel`、`T.Tensor`
- [ ] `type_system.md`：dtype、shape、动态 shape
- [ ] `instructions.md`：`T.alloc_shared` / `T.alloc_fragment` / `T.copy` / `T.gemm` / `T.reduce` / `T.fill`
- [ ] `control_flow.md`：`T.Parallel`、`T.Pipelined`、`T.serial`、`T.unroll`
- [ ] `software_pipeline.md`：软件流水线（num_stages）原理
- [ ] `cluster_tma.md`：Hopper cluster / TMA
- [ ] `autotuning.md` + `docs/tutorials/auto_tuning.md`：autotuner 用法
- [ ] 对照源码：`tilelang/language/`（`kernel.py`、`copy_op.py`、`gemm_op.py`、`reduce_op.py`、`loop.py`、`allocate.py`）

## 阶段 3：经典算子实战

文档：`docs/deeplearning_operators/`；代码：`examples/`

- [ ] Elementwise（`elementwise.md`，`examples/elementwise`）
- [ ] GEMV（`gemv.md`，`examples/gemv`）
- [ ] GEMM（`matmul.md`，`examples/gemm`）——重点：tile 划分、shared/fragment、pipeline、swizzle
- [ ] GEMM 变种：`gemm_splitk`、`gemm_streamk`、`gemm_fp8`、`dequantize_gemm`
- [ ] Norm / Softmax（`examples/norm`、`examples/online_softmax`）
- [ ] FlashAttention（`examples/flash_attention`）——重点：online softmax + 双 GEMM 融合
- [ ] Flash Decoding（`examples/flash_decoding`）
- [ ] DeepSeek MLA（`deepseek_mla.md`，`examples/deepseek_mla`）
- [ ] Warp Specialize（`examples/warp_specialize`）
- [ ] 选做：`fusedmoe`、`grouped_gemm`、`linear_attention`、`blocksparse_attention`
- [ ] 每个算子：自己从零重写一遍 + 用 `benchmark/` 测性能，和 PyTorch/cuBLAS 对比

## 阶段 4：编译器内部（Pass 链路）

入口：`tilelang/engine/lower.py` → `tilelang/backend/pass_pipeline/pipeline.py`

- [ ] 前端：DSL 如何被解析成 TIR（`tilelang/language/parser`、`tilelang/language/ast`）
- [ ] 编译主流程：`lower()` → `lower_to_host_device_ir()` → host/device 拆分 → codegen
- [ ] Pass 编排：读 `tilelang/backend/pass_pipeline/pipeline.py`，列出 Pass 顺序表
- [ ] Tile Op 抽象：`src/op/`（copy / gemm / reduce 等 TileOperator 的 Lower 与 InferLayout）
- [ ] **Layout 推导**（核心）：`src/layout/`、`src/transform/layout_inference/`，理解 Fragment / Layout
- [ ] `src/transform/lower_tile_op.cc`：Tile 级算子展开为线程级代码
- [ ] 流水线：`pipeline_planning.cc`、`inject_pipeline.cc`
- [ ] 向量化 / 循环：`loop_vectorize.cc`、`loop_partition.cc`、`legalize_vectorized_loop.cc`
- [ ] 内存：`merge_shared_memory_allocations.cc`、`storage_rewrite.cc`、`thread_storage_sync.cc`
- [ ] host/device：`split_host_device.cc`、`make_packed_api.cc`、`lower_device_kernel_launch.cc`
- [ ] 读 `docs/compiler_internals/` 下的专题文档
- [ ] 实践：给 GEMM 例子 dump 每个 Pass 前后的 IR，写一份 Pass 走读笔记

## 阶段 5：Runtime / JIT / 后端

- [ ] JIT：`tilelang/jit/`（`kernel.py`、`adapter/`），理解编译缓存 `tilelang/cache`
- [ ] Runtime：`docs/runtime_internals/stubs.md`、`src/runtime/`
- [ ] CUDA codegen：`src/backend/`、`src/cuda/`、`src/tl_templates/`（CUDA 模板库）
- [ ] 其他后端对比：`src/ascend` / `tilelang/ascend`、`src/rocm`、`src/metal`（`docs/compiler_internals/metal_tilelang_development.md`）
- [ ] 思考题：如果要接入一个新 NPU 后端（AGIC），需要改哪些地方？

## 阶段 6：进阶专题 & 输出

- [ ] Carver（`tilelang/carver`）：硬件感知的 tile 配置推荐
- [ ] Profiler（`tilelang/profiler`）与性能分析
- [ ] 测试体系：`testing/python`、`docs/developer_guide/testing_style.md`
- [ ] 贡献流程：`CONTRIBUTING.md`、`format.sh`，尝试提一个小 PR
- [ ] 输出：在 `代码阅读笔记/` 下沉淀 TileLang 编译链路笔记

---

## 学习日志

| 日期 | 内容 | 收获 / 问题 |
| --- | --- | --- |
| 2026-10-31 | 制定学习计划 | — |
