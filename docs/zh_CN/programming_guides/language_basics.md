# 语言基础

> 英文原文：[Language Basics](../../programming_guides/language_basics.md)。本文为完整中文译文，代码块保留原样。翻译版本及术语说明见[阅读索引](../README.md)。

本文介绍用于编写高性能内核的 TileLang（tile-lang）核心领域专用语言（DSL），重点说明如何定义内核、表达迭代、在不同存储空间之间搬运数据，以及通过 JIT 运行内核。

示例使用以下惯用别名：

```python
import tilelang
import tilelang.language as T
from tilelang import jit
```

## 1. 使用 `@T.prim_func` 定义内核

TileLang 内核是由 `@T.prim_func` 装饰器生成的 TIR（TVM 中间表示）函数。通过 `T.Tensor` 或 `T.Buffer` 为参数标注形状和数据类型。

关于数据类型：

- 数据类型可以使用字符串（例如 `'float32'`）、TileLang 数据类型（例如 `T.float32`），或框架的数据类型（例如 `torch.float32`）。TileLang 会对这些形式进行统一处理。详细信息见英文文档的“Type System”（类型系统）章节。

```python
@T.prim_func
def add_kernel(
    A: T.Tensor((N,), dtype),  # dtype could be 'float32' | T.float32 | torch.float32
    B: T.Tensor((N,), dtype),
    C: T.Tensor((N,), dtype),
): ...  # kernel body
```

- 形状可以使用具体整数，也可以包含符号。可以通过外层 `@jit` 包装函数传入 Python 整数，具体方式见后文；如果需要带名称的符号维度，则可以使用 `T.dyn` 标注。

```python
# Named symbolic dimension (optional)
K = T.dyn["K"]


@T.prim_func
def uses_dyn(A: T.Tensor((K,), "float32")): ...
```

### 引入动态符号维度的两种方式

TileLang 支持两种互补的方式来引入符号维度，也就是动态维度：

- 通过 `T.dyn[...]` 进行类型层面的标注，推荐用于函数签名。
  - 可以直接写在 `T.Tensor((T.dyn['K'], ...), dtype)` 中，也可以像上面的示例一样先绑定，再重复使用。
  - 在内核函数体中，优先从缓冲区的形状读取维度，例如 `M = A.shape[0]`。
- 通过 `T.dynamic(name, dtype)` 创建表达式层面的变量。
  - 它会创建一个 TIR `tir.Var`，可以直接用于表达式和循环。
  - 当函数体中需要引用该维度符号时，这种方式很方便。

```python
# 1) Annotation-only symbol; read the bound size via shape
K = T.dyn["K"]  # dtype defaults to int32


@T.prim_func
def foo(A: T.Tensor((K,), "float32")):
    N = A.shape[0]
    for i in T.serial(N):
        ...


# 2) Explicit Var symbol usable in the body
K = T.dynamic("K", "int32")  # or T.dynamic('K') defaults to int32


@T.prim_func
def bar(A: T.Tensor((K,), "float32")):
    for i in T.serial(K):
        ...
```

注意事项：

- `T.symbolic(name, dtype)` 是 `T.dynamic` 的已弃用别名，将在 v0.1.15 中移除；请优先使用 `T.dynamic`。
- 使用 `@jit` 时，具体尺寸来自首次调用时实际传入的张量参数。
- 标注中的符号不必再作为单独的内核参数传入，TileLang 会根据参数形状绑定它们。

## 2. 使用 `T.Kernel` 启动计算

`with T.Kernel(...)` 声明一个由 tile 程序组成的网格。位置参数给出各个轴上的网格大小，返回的变量则是对应轴上的程序索引。这部分启动语义是各个目标后端共有的：在 CUDA 上，一个程序对应一个线程块，`bx` / `by` 对应 `blockIdx.x` / `blockIdx.y`；在 CPU 上，网格会转换为外层循环。

```python
with T.Kernel(grid_x, grid_y, threads=128) as (bx, by):
    ...  # bx/by are the program indices (blockIdx.x/y on CUDA)
```

关键字参数是启动标注，由后端在确定编译目标后解释。每种语言方言的 `Kernel` 都会用显式关键字参数声明其后端支持的标注，因此在 `T.Kernel` 上悬停或使用自动补全时，只会显示这些参数，其他参数会被拒绝：

- `tilelang.language`（CUDA 方言）提供 `threads`、`prelude` 和 `cluster_dims`。
- `tilelang.rocm.language` / `tilelang.metal.language` 提供 `threads` 和 `prelude`。
- `tilelang.cpu.language` 仅提供 `prelude`。

`threads` 是与 SIMT（单指令、多线程）执行模型对应的参数，表示 GPU 类后端中每个 tile 程序由多少个线程执行。省略时，这些后端使用默认值 128。使用 CUDA 方言编写的内核仍然可以编译到 CPU，CPU 会忽略线程数量。

`T.Kernel` 内部的代码在 tile 程序层面表达计算，因此很少需要直接使用线程索引。大多数内核使用结构化循环（`T.serial`、`T.unroll`、`T.Parallel`、`T.Pipelined`），再由编译器将其映射到线程。对于 SIMT 目标上的线程级代码，`T.get_thread_binding()` 可以取得线程索引；如果目标不支持 SIMT 线程，编译器会拒绝使用该接口的内核。

`T.ClusterKernel(..., cluster_dims=...)` 会添加 CUDA 线程块集群标注，要求 SM90 或更新架构。一个集群对应网格中形状为 `cluster_dims` 的一块区域，所以 `T.get_cluster_id(axis)` 只是对程序索引做普通算术运算（`bx // cluster_dims[axis]`），适用于所有目标；而 `T.block_rank_in_cluster()` 读取的是硬件提供的集群内编号，仅适用于 CUDA。不支持集群的目标会在编译时拒绝 `cluster_dims`。

## 3. 循环与控制流

核心循环结构对应以下常见硬件执行模式：

- `T.serial(start, stop[, step])`：普通 `for` 循环。
- `T.unroll(start, stop[, step])`：展开循环。
- `T.Parallel(ext0, ext1, ...)`：嵌套并行循环，适合逐元素计算。
- `T.Pipelined(iters, num_stages=N)`：用于生产者 / 消费者计算的软件流水线。

```python
for i in T.serial(N):
    ...

for i, j in T.Parallel(M, N):
    C[i, j] = A[i, j] + B[i, j]

for k in T.Pipelined(T.ceildiv(K, BK), num_stages=3):
    # overlap copy/compute across stages
    ...
```

条件分支使用标准 Python `if` / `else`。当问题尺寸不能被 tile 大小整除时，使用条件判断保护边界访问。

## 4. 存储空间与分配

TileLang 暴露了以下由软件管理的关键存储空间：

- Global：设备内存，是 `T.Tensor` 参数默认所在的存储空间。
- Shared：片上、线程块内可见的共享存储，通过 `T.alloc_shared(shape, dtype)` 分配。
- Fragment 与标量：每个线程持有的数据片段和标量变量，但在代码中以共享视图（Shared View）表达，使用 `T.alloc_fragment`、`T.alloc_var` 分配。

```python
A_shared = T.alloc_shared((BM, BK), "float16")
B_shared = T.alloc_shared((BK, BN), "float16")
C_local = T.alloc_fragment((BM, BN), "float32")
T.clear(C_local)  # zero accumulators
```

## 5. 数据搬运：`T.copy`

使用 `T.copy(src, dst)` 在不同存储空间之间搬运 tile。它接受缓冲区、缓冲区区域或缓冲区读取表达式，搬运范围可以自动推断，也可以进行广播。

```python
# Global -> Shared (tile copy), extents inferred from dst
T.copy(A[by * BM, ko * BK], A_shared)
T.copy(B[ko * BK, bx * BN], B_shared)

# Fragment -> Global (store back)
T.copy(C_local, C[by * BM, bx * BN])
```

编译期间，`T.copy` 会进行合并访存处理，并针对具体存储空间执行降低转换。

如果需要显式执行从 global 到 shared 的异步预取，以手动安排流水线，可以使用 `T.async_copy(src, dst)`。它与 `T.copy` 的区别是不会自动插入等待，因此必须在使用 `dst` 中的数据之前显式插入 `T.ptx_wait_group(...)`。如果数据由其他线程使用，仍然需要共享内存屏障；但在大多数 TileLang 程序中，无须手动编写这个屏障，因为 `ThreadSync("shared")` 会在首次读取 `dst` 之前插入必要的 `T.tvm_storage_sync("shared")`。

## 6. 最小完整示例：向量加法

```python
import tilelang
import tilelang.language as T
from tilelang import jit


@jit  # infers target from tensors at first call
def add(N: int, block: int = 256, dtype: str = "float32"):

    @T.prim_func
    def add_kernel(
        A: T.Tensor((N,), dtype),
        B: T.Tensor((N,), dtype),
        C: T.Tensor((N,), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block), threads=block) as bx:
            for i in T.Parallel(block):
                gi = bx * block + i
                # Optional — LegalizeSafeMemoryAccess inserts a guard when an access may be OOB
                C[gi] = A[gi] + B[gi]

    return add_kernel


# Host side (PyTorch shown; NumPy/DLPack also supported)
import torch

N = 1 << 20
A = torch.randn(N, device="cuda", dtype=torch.float32)
B = torch.randn(N, device="cuda", dtype=torch.float32)
C = torch.empty(N, device="cuda", dtype=torch.float32)

kernel = add(N)
kernel(A, B, C)  # runs on GPU
torch.testing.assert_close(C, A + B)
```

注意事项：

- `@jit` 包装函数在首次编译后返回一个可调用的内核。
- 可以通过外层 Python 函数传入编译期可调参数，例如 tile 大小和数据类型，并将这些值固化到生成的 TIR 中。
- 编译期参数及其默认值必须可哈希。列表、字典、集合，以及包含不可哈希元素的元组都会引发 `TypeError`；应改用元组或冻结的数据类等不可变、可哈希的值。影响特化的值，以及这些值的哈希和相等性行为，在缓存内核使用期间必须保持稳定。运行时张量参数不受这一要求限制。

## 7. 分块 GEMM 的基本结构

下面给出一个最小的分块 GEMM 模式，使用共享内存暂存输入，并用 fragment 累加器保存结果。其风格与仓库中的快速入门示例一致。

```python
@T.prim_func
def gemm(
    A: T.Tensor((M, K), "float16"),
    B: T.Tensor((K, N), "float16"),
    C: T.Tensor((M, N), "float16"),
):
    with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=128) as (bx, by):
        A_s = T.alloc_shared((BM, BK), "float16")
        B_s = T.alloc_shared((BK, BN), "float16")
        C_f = T.alloc_fragment((BM, BN), "float32")
        T.clear(C_f)

        for ko in T.Pipelined(T.ceildiv(K, BK), num_stages=3):
            T.copy(A[by * BM, ko * BK], A_s)
            T.copy(B[ko * BK, bx * BN], B_s)
            T.gemm(A_s, B_s, C_f)  # lowered to tensor‑core/ISA specific kernels

        T.copy(C_f, C[by * BM, bx * BN])
```

## 8. 调试与打印

可以在内核中使用 `T.print` 快速查看内部数据。对于 shared / fragment 存储空间，TileLang 会让单个线程执行打印，避免产生大量重复输出。

```python
T.print(C_f, msg="accumulator:")
T.print(A_s, msg="A tile:")
T.print(C[0], msg="C[0] = ")
```

## 9. 后续阅读

- 控制流细节：见英文文档“Programming Guides → Control Flow”（编程指南 → 控制流）。
- 存储相关主题：原文注明编程指南中的 cache / layout 章节已移除，基础内容已在正文中说明。
- tile 大小和映射的自动调优：见英文文档“Programming Guides → Autotuning”（编程指南 → 自动调优）。
- 算子示例，包括 GEMM、GEMV 和 attention：见英文文档“Deep Learning Operators”（深度学习算子）。
