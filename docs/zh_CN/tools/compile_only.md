# 仅编译工具：Compile-Only

> 英文原文：[Compile-Only Tool](../../tools/compile_only.md)。本文为完整中文译文，代码块保留原样。翻译版本及术语说明见[阅读索引](../README.md)。

`python -m tilelang.tools.compile_only` 可以将 TileLang 程序降低为可查看的内核源码，而不运行内核。它不需要 GPU、CUDA 驱动，也不需要编译设备代码：工具在执行 `tilelang.lower(..., enable_device_compile=False)` 后就会停止，并将生成的内核源码写入文件。

该工具为 [Compiler Explorer](https://godbolt.org) 中的 TileLang 集成提供支持，也适合在本地查看 `lower()` 为内核生成的结果。

## 使用条件

该工具包含在标准 TileLang 安装中，没有额外的可选依赖。默认的 `c` 目标适用于所有机器，包括只有 CPU 的主机。

使用 `--target cuda` 时，所安装的 wheel 必须编译进 CUDA 代码生成 FFI。Linux wheel 包含该功能，因此即使机器没有 GPU，也可以生成 CUDA 源码；macOS Metal wheel 则不包含该功能，会给出明确的 `CUDA codegen FFI missing` 提示，而非抛出 `AttributeError`。

## 快速开始

编写一个自包含示例，只定义内核，不启动执行：

```python
# example.py
import tilelang
import tilelang.language as T


@tilelang.jit
def add(A, B):
    N = 64
    A: T.Tensor((N,), T.float16)
    B: T.Tensor((N,), T.float16)
    C = T.empty((N,), T.float16)
    with T.Kernel(1, threads=64):
        for i in T.Parallel(N):
            C[i] = A[i] + B[i]
    return C
```

将它编译为 CPU C 源码：

```bash
python -I -m tilelang.tools.compile_only --output_file out.c example.py
```

生成的内核源码位于 `out.c` 中：

```c
// tilelang target: {"kind":"c","tag":"","keys":["cpu"],"host":{"kind":"c","tag":"","keys":["cpu"]}}
#include <tl_templates/cpp/common.h>

#ifdef __cplusplus
extern "C"
#endif
int32_t add_kernel(half* A, half* B, half* C);
#ifdef __cplusplus
extern "C"
#endif
int32_t add_kernel(half* A, half* B, half* C) {
  for (int32_t i = 0; i < 8; ++i) {
    *(half8*)(C + (i * 8)) = (*(half8*)(A + (i * 8)) + *(half8*)(B + (i * 8)));
  }
  return 0;
}
```

生成 CUDA 源码的方式相同，仍然不需要 GPU；目标架构使用固定值，不执行自动检测：

```bash
python -I -m tilelang.tools.compile_only --target cuda --output_file out.cu example.py
```

```cuda
extern "C" __global__ void __launch_bounds__(64, 1) add_kernel(const half_t* __restrict__ A, const half_t* __restrict__ B, half_t* __restrict__ C) {
  C[((int)threadIdx.x)] = (A[((int)threadIdx.x)] + B[((int)threadIdx.x)]);
}
```

## 输入文件的处理方式

- 输入文件会作为普通 Python 模块导入，因此顶层代码会执行。请让示例只用于编译：定义内核，但不要在导入时启动内核或分配张量。推荐使用 `python -I`（隔离模式）运行命令行工具，Compiler Explorer 也是这样调用的。
- 工具按照模块中的定义顺序，选择第一个 `@tilelang.jit` 函数或 `PrimFunc`。对象按照定义顺序检查，因此，如果 `PrimFunc` 定义在 `@tilelang.jit` 内核之前，就会优先选中该 `PrimFunc`。
- 内核只会执行降低转换，不会编译设备代码，也不会执行，因此不需要张量参数，也不会使用 GPU。

## 命令行参考

```text
python -I -m tilelang.tools.compile_only [--target TARGET] --output_file OUTPUT input_file
```

| 参数 | 含义 |
| --- | --- |
| `input_file` | 仅用于编译的示例文件。会导入，但不会启动其中的内核 |
| `--output_file` | 必填，指定生成内核源码的保存位置。不能与输入文件相同，指向输入文件的符号链接也会被拒绝 |
| `--target` | 显式指定目标，默认为 `c` |

成功时，命令行工具以退出码 `0` 结束，并将源码写入 `--output_file`。

发生任何失败时，工具以退出码 `1` 结束，向标准错误输出一行 `tilelang compile-only error: ...`，且不会留下过期输出：编译前会删除 `--output_file` 位置已有的产物，因此重新运行失败后，不会让 Compiler Explorer 等使用方误读上一次运行的源码。

## 编译目标

必须显式指定目标。由于 `auto` 会执行设备检测，无论以普通字符串还是 JSON 形式传入，都会被拒绝：

```text
$ python -I -m tilelang.tools.compile_only --target auto --output_file out.c example.py
tilelang compile-only error: target must be explicit; do not use auto
```

| `--target` 的值 | 行为 |
| --- | --- |
| `c`（默认） | 生成 CPU C 源码。适用于所有 wheel，不需要 GPU |
| `cuda` | 生成 CUDA 源码，架构固定为 `sm_80`，因此不会检测设备 |
| `cuda -arch=sm_90` | 为指定架构生成 CUDA 源码。选项必须使用 `-key=value` 形式 |
| `{"kind": "cuda", "arch": "sm_90"}` | JSON 形式的目标。仅指定 `{"kind": "cuda"}` 时固定为 `sm_80`；`{"kind": "auto"}` 会被拒绝 |
| `auto` | 拒绝，并提示 `target must be explicit; do not use auto` |

对于不包含 CUDA 代码生成 FFI 的 wheel，例如 macOS Metal wheel，所有 CUDA 目标形式都会以清晰的错误提示结束：`tilelang compile-only error: CUDA codegen FFI missing (e.g. macOS Metal wheel); use default --target c`。

## 错误报告

编译问题会作为诊断信息写入标准错误，退出码为 `1`，不会以缺少 GPU 或驱动的异常堆栈形式报告：

```text
$ python -I -m tilelang.tools.compile_only --output_file out.c broken.py
tilelang compile-only error: no @tilelang.jit kernel or PrimFunc found
```

输入文件中的语法错误、输入文件不存在，以及空内核，也会以相同方式报告。

## 行号指令

如果已安装的构建版本注册了 `tl.emit_line_directives` pass 配置项（`PassConfigKey.TL_EMIT_LINE_DIRECTIVES`），工具会自动启用该选项。生成的源码会带有 `#line <n> "<input_file>"` 指令，将每条语句映射回示例中的原始行号。

Compiler Explorer 使用这些信息关联生成的源码和输入面板。如果旧版 wheel 没有这个配置项，则只会生成不带 `#line` 指令的源码。两种情况都不需要额外设置命令行选项。

## 编程接口

Python 中也可以使用相同功能：

```python
from tilelang.tools.compile_only import compile_kernel_source

source = compile_kernel_source(add.get_tir())  # default target "c"
cuda_source = compile_kernel_source(add.get_tir(), "cuda")  # pinned to sm_80
```

- `compile_kernel_source(func, target="c")`：降低一个 `PrimFunc`，例如 `JITImpl.get_tir()` 返回的对象，并返回非空的内核源码字符串。
- `resolve_target(target)`：将命令行风格的目标字符串转换为传入 `lower()` 的显式目标，并应用上表中的规则。
- `cuda_codegen_available()`：报告已安装的 wheel 是否支持降低到 CUDA 目标。

## 限制

- 只编译模块中的第一个 `@tilelang.jit` 内核或 `PrimFunc`，同一文件中的其他内核会被忽略。
- 输出是降低后的内核源码，而非编译好的二进制文件。工具不会调用 `nvcc`，也不涉及 PTX / SASS。
- 输入文件由调用该工具的 Python 解释器执行，应像对待其他要运行的脚本一样对待它。`python -I` 只隔离运行环境，并不会提供沙箱。
