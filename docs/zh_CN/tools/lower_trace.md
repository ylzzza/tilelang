# IR 降低过程跟踪：IR Lower Trace

> 英文原文：[IR Lower Trace](../../tools/lower_trace.md)。本文为完整中文译文，代码块保留原样，图示说明除外。翻译版本及术语说明见[阅读索引](../README.md)。

IR Lower Trace 用于观察编译 pass 运行时，TileLang 的 TIR 如何变化。它会自动捕获编译流水线中**每个** pass 执行前后的 IR，包括最后生成 C / CUDA / HIP 源码的代码生成步骤，并在终端和 / 或独立的 HTML 页面中生成便于阅读的差异报告。无须修改代码，只要启用一个环境变量即可。

IR Lower Trace 是推荐使用的 [Pass Diff](../../tools/pass_diff.md) 后继工具。与旧版 `TILELANG_PASS_DIFF` 相比，它增加了以下功能：

- **编译阶段上下文**：每个 pass 都标注其所属的流水线阶段，例如 `pipeline_c`、`phase1_...`，方便判断它属于哪个后端编译阶段。
- **捕获代码生成结果**：记录最终从 TIR 到源码的降低过程，并将生成的 C / CUDA / HIP 代码写入磁盘，便于查看或编辑。
- **编辑后重新编译流程**：编辑磁盘上生成的源码并重新运行，修改后的代码会被重新注入编译流程，同时进行冲突检测。
- **按编译任务隔离**：每次内核编译都拥有独立的记录、pass 索引、流水线作用域和报告生命周期。并发编译的内核不会写入同一份共享的内存跟踪记录。
- **原始 `.tir` 快照**：每个 pass 前后的 IR 都会按阶段和 pass 索引写入磁盘。
- **可保留崩溃前结果的增量 HTML**：每个 pass 执行后都会将报告刷新到磁盘，即使进程崩溃，已有结果仍会保留。
- **增强的 HTML 报告**：提供侧边栏 pass 导航、状态圆点、阶段标签页、`j` / `k` 键盘导航、`Shift+E` 全局展开、`F7` 手动对齐，以及深色 / 浅色主题。

## 观察降低流水线

启用 IR Lower Trace 最简单的方式，是在运行脚本之前设置 `TL_LOWER_TRACE` 环境变量：

```bash
# HTML report (default when set to 1/on/true/yes)
TL_LOWER_TRACE=1 python3 my_script.py

# Colored diff printed to the terminal only
TL_LOWER_TRACE=terminal python3 my_script.py

# Both terminal output and HTML report
TL_LOWER_TRACE=both python3 my_script.py

# Disabled (default — zero overhead, no patching)
python3 my_script.py
```

导入 `tilelang` 时，TileLang 会读取该设置，并为后续编译会话注册一个工具工厂。它不会修改 TVM 默认的 `PassContext`，也不会替换全局代码生成函数。如果需要在导入后修改设置，可以通过程序调用 `enable()`。

启用 HTML 输出后，工具会维护一个稳定的符号链接 `<script_dir>/report.html`，使其指向最近一次运行的报告。可以直接在浏览器中打开：

```bash
# typical location
open tmp/lower_trace_dir/my_script/report.html
```

| 变量 | 说明 | 默认值 |
| --- | --- | --- |
| `TL_LOWER_TRACE` | 启用跟踪。可选值：`0` / `off` / `false` / `no` 表示关闭，`1` / `on` / `true` / `yes` 表示 HTML 模式；也可设为 `terminal`、`html`、`both` | `off` |
| `TL_LOWER_TRACE_DIR` | 所有跟踪产物的基础输出目录 | `./tmp/lower_trace_dir` |

### 输出目录结构

一次运行会在 `TL_LOWER_TRACE_DIR` 下生成以下目录结构：

```text
<TL_LOWER_TRACE_DIR>/
└── <script_name>/                      # derived from sys.argv[0], e.g. "my_script"
    ├── report.html                     # symlink → latest run's report
    ├── codegen_device.cpp              # generated device source (editable, see below)
    ├── codegen_device.cpp.original     # device baseline for edit/recompile workflow
    ├── codegen_device.cpp.latest       # latest generated device source
    ├── codegen_host.cpp                # generated CPU wrapper, when host codegen runs
    ├── codegen_host.cpp.original       # independent host baseline
    ├── codegen_host.cpp.latest         # latest generated host source
    └── .run_records/
        └── run_<YYYYMMDD_HHMMSS_ffffff>_<pid>_<unique-id>/
            ├── report.html             # this run's full report
            ├── pipeline_c/             # one subdir per phase (example)
            │   ├── 00_BindTarget_before.tir
            │   ├── 00_BindTarget_after.tir
            │   ├── 01_Simplify_before.tir
            │   └── 01_Simplify_after.tir
            ├── phase2_optimize/        # another phase (illustrative)
            │   └── ...
            ├── codegen/
            │   ├── 42_codegen_device_before.tir
            │   ├── 42_codegen_device_after.cpp
            │   ├── 49_codegen_host_before.tir
            │   └── 49_codegen_host_after.cpp
            └── unscoped/               # passes outside any pipeline window
                └── ...
```

每个阶段都有自己的子目录；pass 使用全局编号（`00_`、`01_` 等），因此顺序明确。代码生成记录会把变换后的产物写为 `.cpp` 文件，即生成的源码；普通 pass 则把变换前后的内容都写为 `.tir` 文件。

设备端和主机端记录都属于 `codegen` 阶段，名称分别为 `codegen_device` 和 `codegen_host`。只有执行后端调用主机端代码生成时，才会出现主机端文件和记录。每个工作文件都有各自的基线、最新输出和备份；主机端代码生成不会覆盖设备端工作文件及其编辑状态。

标准后端包装层会明确标识每个事件属于设备端还是主机端代码生成。如果通用插桩调用省略了 `role`，其角色将保持未指定状态（`None`）：这些事件保留 `codegen` 记录名称，并默认使用 `codegen.cpp` 作为工作文件。对于这类事件，显式指定的 `codegen_output` 路径会直接使用，不做修改。

自定义调用方应提供角色，以区分设备端和主机端输出；未指定角色的事件会共享旧版工作文件和编辑状态。

## 编程接口

为了提供更细粒度的控制，IR Lower Trace 暴露了两层 API。

### 单次调用：`lower_trace()`

可以对 IR 模块应用一组固定的 pass 并比较差异，而无须安装任何全局钩子：

```python
from tilelang.tools import lower_trace as lt
from tilelang import tvm
import tilelang.transform as transform

# Diff a single pass
results = lt.lower_trace(func, transform.Simplify(), mode="terminal")

# Diff a named chain, write an HTML report
results = lt.lower_trace(
    func,
    [
        ("Annotate", tvm.tirx.transform.AnnotateDeviceRegions()),
        ("Split", tvm.tirx.transform.SplitHostDevice()),
        ("ThreadSync", transform.ThreadSync("shared")),
    ],
    mode="both",
    html_path="my_diff.html",
)
```

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `func_or_mod` | 用于执行 pass 的 `PrimFunc` 或 `IRModule` | 必填 |
| `passes` | 单个 pass、pass 列表，或 `(name, pass)` 元组列表 | 必填 |
| `mode` | `"terminal"`、`"html"` 或 `"both"` | `terminal` |
| `context` | 统一差异格式中显示的上下文行数 | `3` |
| `html_path` | HTML 报告的输出路径 | `lower_trace_report.html` |

返回值为 `list[dict]`，每个 pass 步骤对应一个条目，包含 `name`、`before_script`、`after_script`、`diff_lines`、`insertions`、`deletions` 和 `changed`。

### 编译过程插桩：`enable()` / `disable()`

要跟踪真实内核的**整个**编译流水线，可以使用下面的方式。它与设置环境变量的作用相同，只是改为通过程序启用：

```python
from tilelang.tools import lower_trace as lt

# Enable tracing for the rest of the process.
lt.enable(mode="both")

# ... run tilelang.compile() / kernel compilation ...
```

`lt.enable()` 的三个参数都可选。省略 `mode`、`trace_dir` 和 `codegen_output` 时，会根据 `TL_LOWER_TRACE` / `TL_LOWER_TRACE_DIR` 环境变量或合适的默认值确定其取值。

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `mode` | 显式指定跟踪模式：`"terminal"`、`"html"`、`"both"`；设为 `None` 表示关闭 | `TL_LOWER_TRACE` 环境变量 |
| `trace_dir` | 基础输出目录 | 优先使用 `TL_LOWER_TRACE_DIR`，否则使用 `./tmp/lower_trace_dir` |
| `codegen_output` | 设备端工作文件路径；执行主机端代码生成时，会据此推导主机端工作文件路径。显式传入 `None` 可禁止生成这两个工作文件 | `html` / `both` 模式下为 `<script_dir>/codegen_device.cpp` 和 `<script_dir>/codegen_host.cpp`；`terminal` 模式下不生成工作文件 |

显式指定 `codegen_output="foo.cpp"` 时，设备端源码写入 `foo.cpp`，主机端源码写入 `foo_host.cpp`。主机端路径通过在最后一个扩展名之前插入 `_host` 得到：`foo.kernel.cpp` 会变成 `foo.kernel_host.cpp`；没有扩展名的路径，例如 `foo`，则变成 `foo_host`。

每个路径都有各自的 `.original`、`.latest` 和备份文件。设置 `codegen_output=None` 会禁止生成这些稳定的工作文件及其编辑状态，但跟踪中仍然包含代码生成记录，原始快照也仍然可用。

`enable()` 是幂等的，多次调用是安全的。编译开始时会保存这些设置的快照，因此后续修改设置不会影响已经在编译的内核。

#### 何时使用 `disable()`

| 函数 | 调用时机 | 作用 |
| --- | --- | --- |
| `disable()` | 希望在同一进程中**关闭后续编译的跟踪**，例如长期运行的服务只跟踪第一个内核 | 注销工具工厂。正在执行的编译会保留自己的会话，并正常完成 |

```python
from tilelang.tools import lower_trace as lt

lt.enable(mode="both")

# First kernel — traced.
kernel1 = tilelang.compile(func_a)

# Second kernel — traced into its own session and report automatically.
kernel2 = tilelang.compile(func_b)

# Optional: disable tracing for any further compilations.
lt.disable()
```

:::{note}
如果没有调用 `lt.disable()`，后续编译仍会继续跟踪。每份 HTML 报告会在所属编译任务退出时完成收尾，即使该编译因异常退出也是如此。
:::

## HTML 报告功能

HTML 报告是一个独立文件，不依赖外部资源，提供以下功能：

- **侧边栏**：按 pass 导航，显示状态标记（● 有变化 / ○ 无变化 / ✕ 失败 / ◆ 代码生成），以及 `+` / `−` 行数统计。侧边栏可以折叠，也可以拖动调整宽度。
- **阶段标签页**：按流水线阶段筛选 pass。
- **摘要栏**：提供可点击的筛选标记，用于查看有变化、失败或代码生成的记录。
- **并排差异视图**：采用 GitHub 风格配色、词级行内高亮和可折叠上下文；点击 `↑↓ Expand` 按钮可以显示隐藏的相同行。
- **键盘导航**：`j` / `k` 用于切换 pass，`Shift+E` 用于全部展开 / 折叠，`F7` 用于类似 Beyond Compare 的手动对齐。
- **深色 / 浅色主题切换**：通过 `localStorage` 保存选择。
- **复制按钮**：可以复制任意 pass 变换前后的 IR。
- **错误提示框**：某个 pass 失败时，会同时显示异常信息和崩溃前的 IR。

```{figure} ../../_static/img/lower_trace_html.png
:width: 600
:alt: IR Lower Trace HTML 报告截图
:align: center

IR Lower Trace HTML 报告。
```

## 捕获代码生成源码，以及编辑后重新编译流程

启用跟踪后，设备端和主机端的代码生成步骤（TIR → C / CUDA / HIP / …）会被分别拦截。设备端源码运行在 GPU 或 NPU 上，包含计算、设备内存传输和同步逻辑。主机端源码运行在 CPU 上，包装参数检查、输出分配、启动设备内核等运行时操作。

在 `html` / `both` 模式下，生成的源码默认写入 `<script_dir>/codegen_device.cpp` 和 `<script_dir>/codegen_host.cpp`。对于 CUDA 和 Ascend，`tvm_ffi` 会同时调用设备端和主机端代码生成；`nvrtc` 和 `cython` 在这一层不会调用单独的主机端代码生成步骤，但其运行时适配器仍然提供 CPU 端启动逻辑。只有调用主机端代码生成的后端才会生成 `codegen_host.cpp`。

为了支持编辑生成代码，并在重新运行时应用这些修改，IR Lower Trace 会为**每个代码生成角色维护三个配合使用的文件**。下面以设备端文件为例；主机端文件则在 `codegen_host.cpp` 上使用相同后缀，并维护独立的编辑状态：

| 文件 | 作用 |
| --- | --- |
| `codegen_device.cpp` | **工作副本**：用户可以编辑。后端支持下文所述流程时，修改会被重新编译 |
| `codegen_device.cpp.original` | **基线**：工作副本最近一次同步时对应的代码生成快照。只在初始化或重新同步时写入，不会直接无条件覆盖 |
| `codegen_device.cpp.latest` | **最新代码生成输出**：最近一次运行真正生成的代码，每次运行都会覆盖，供比较参考 |

每次运行时，会通过三方比较（基线 / 工作副本 / 当前代码生成输出）决定后续行为：

| 情况 | 工作副本与 `.original` 的关系 | `.latest` 与 `.original` 的关系 | 操作 | 控制台标记 |
| --- | --- | --- | --- | --- |
| 没有变化 | 相同 | 相同 | 按原样编译代码生成输出 | — |
| 只有代码生成结果变化 | 相同 | **不同** | 使用新的代码生成结果重新生成工作副本和 `.original` | `REGENERATED` |
| 只有用户编辑过 | **不同** | 相同 | 注入工作副本（`PATCHED`） | `PATCHED` |
| 两者都变化，工作副本等于最新输出 | **不同** | **不同**，且工作副本与最新输出一致 | 更新基线，使用工作副本 | `SYNCED` |
| 两者都变化，工作副本不等于最新输出 | **不同** | **不同**，且工作副本与最新输出不一致 | **冲突**：将工作副本备份到 `.bak`（冲突备份），将旧基线备份到 `.original.bak`，再根据新的代码生成结果重新生成文件 | `CONFLICT` |
| 首次运行，没有基线 | — | — | 初始化 `.original`，并复制到工作文件 | (init) |
| 工作副本已存在，但没有基线 | — | — | 将已有工作文件备份到 `.bak`（安全备份），再初始化基线 | `INIT-BACKUP` |

> **关于 `.bak` 文件：** `CONFLICT` 和 `INIT-BACKUP` 创建的备份用途不同。`INIT-BACKUP` 用于在跟踪工具接管工作文件之前，保留已存在但来源未知的工作文件；`CONFLICT` 用于在代码生成变化覆盖用户编辑之前，保留用户的修改。设备端工作副本发生 `CONFLICT` 后，可以通过 `diff codegen_device.cpp.original.bak codegen_device.cpp.bak` 找回修改；主机端工作副本则使用对应的 `codegen_host.cpp` 命令。这些备份文件只在需要时创建。

### 典型流程

1. **查看**：设置 `TL_LOWER_TRACE=1` 运行一次，打开 `codegen_device.cpp` 阅读生成的源码。
2. **编辑**：修改 `codegen_device.cpp`，例如添加 `printf` 或调整循环。**不要**修改 `.original`。
3. **重新运行**：由于 `codegen_device.cpp` 与 `.original` 不同，但代码生成输出未变化，控制台会显示 `PATCHED from …/codegen_device.cpp`，并编译修改后的源码。
4. **继续迭代**：反复编辑并重新运行。每次运行都会重新注入工作副本。
5. **如果代码生成结果本身发生变化**，例如修改了 TileLang 程序，则有两种情况：
   - 如果用户修改后的内容恰好与新的代码生成结果一致，状态为 `SYNCED`：更新基线，并保留修改。
   - 如果用户修改和代码生成结果都发生变化，且两者不同，状态为 `CONFLICT`。工作副本会备份到 `codegen_device.cpp.bak`，旧基线会备份到 `codegen_device.cpp.original.bak`。通过 `diff codegen_device.cpp.original.bak codegen_device.cpp.bak` 找回修改，再将其重新应用到刚生成的 `codegen_device.cpp` 上。

:::{note}
**编辑后重新编译所需的后端条件。** 这一流程要求使用会编译源码的执行后端，即 `nvrtc`、`cython` 或 `cutedsl`。这些后端通过 `*_without_compile` 代码生成 FFI 产生只包含源码的模块，然后在运行时通过 NVRTC / Cython / CuTeDSL 编译修改后的源码字符串。

默认的 `tvm_ffi` 后端会在代码生成阶段，将 TIR 预编译为设备代码产物（PTX / hsaco）。如果当前启用的是 `tvm_ffi` 后端，而你修改了 `codegen_device.cpp`，就会看到一条 `NOTE` 消息，说明这些修改已经记录在跟踪中，可用于查看差异，但**没有重新编译**。要使用编辑后重新编译功能，应切换到会编译源码的后端：

```python
# For CUDA targets:
tilelang.compile(..., execution_backend="nvrtc")

# For HIP targets:
tilelang.compile(..., execution_backend="cython")
```

捕获主机端源码并保留主机端修改，并不意味着支持主机端的编辑后重新编译；现有后端要求仍然适用。
:::

## 工作原理

插桩会话由完整的编译入口拥有，例如 `tilelang.compile()`、`tilelang.lower()` 和分组编译。更低层的流水线和代码生成组件会向当前活跃会话提供事件，但不会自行创建独立会话。

IR Lower Trace 通过以下三层协同工作：

1. **TVM `PassInstrument`**：通过执行前 / 执行后回调捕获 `str(mod)`，计算 `+` / `−` 行数，并追加一条 `LowerRecord`。共享的嵌套回调栈会记录深度和父子关系，包括某个 TVM pass 内部调用的其他 pass。流水线窗口之外的 pass 会被标记为 `unscoped` 阶段。
2. **显式的 `PassPipeline.lower` 作用域**：向调用方拥有的会话提供当前上下文中的阶段信息，使后端流水线内的 pass 可以按 `pipeline_c` 等标签分组。流水线不会创建或结束编译会话。这是正常的流水线边界，并非对类进行动态替换（monkey-patch）。
3. **显式的后端代码生成钩子**：TileLang 的设备端 / 主机端代码生成注册表，会将每次 `target.build.*` 调用路由到当前活跃的编译会话。这样既能捕获最终 TIR → 源码的降低过程、标记事件属于设备端还是主机端，也能驱动各个角色的三文件编辑后重新编译流程，而无须替换进程级全局 FFI。没有活跃会话时，注册表会直接调用底层代码生成。

pass 记录在**运行时追加**，而非预先注册。因此，运行时被跳过的条件 pass，例如 `should_force_let_inline()` 为 `False` 时的 `LetInline`，不会出现在报告中，也不会留下不存在或被跳过的占位条目。每个 pass 结束后，HTML 报告都会**增量刷新**，总开销为 O(n)，因此即使进程崩溃或收到 `SIGKILL`，已有结果仍会保留。

如果一次逻辑编译使用了多个 PassContext，例如先进行分组降低，再生成设备端和主机端代码，这些上下文会从同一个编译会话中取得新的回调对象。其记录共享该会话内单调递增的索引。另一次编译则使用全新的记录列表，从索引零开始。

:::{note}
`enable()` 只影响由 TileLang 管理的编译会话，绝不会修改调用启用操作时恰好处于当前状态的 PassContext。对于独立的 TVM PassContext，仍然可以手动使用 `lt.create_pass_instrument()`。

捕获 IR，以及在 HTML 模式下每个 pass 结束后重写报告，都会带来调试开销。正常构建和性能基准测试时，应保持该功能关闭。
:::

## 选择输出模式

- pass 数量较少、需要即时反馈时，使用 `terminal`。
- 降低流水线产生大量变化，或需要比较完整的变换前后代码时，使用 `html`。
- 交互式迭代期间既需要即时反馈，又希望保留可分享的报告时，使用 `both`。

IR Lower Trace 比较的是 `IRModule.script()` 返回的文本形式。文本变化可以作为了解 pass 行为的证据，但文本变化本身不足以证明语义或性能发生了变化。

## 使用建议

- **快速检查时使用 `terminal` 模式**：pass 运行过程中会打印带颜色的差异，可以实时查看变化。
- **深入分析时使用 `html` 模式**：可以在大量 pass 之间导航、展开隐藏上下文，并复制 IR 片段。
- **结合 `TL_LOWER_TRACE_DIR` 使用**：将报告输出到指定位置，例如在 CI 中运行，或比较多次运行结果时。
- **钩子会捕获降低流水线中的所有 pass**，包括由 `tilelang.compile()` 内部触发的 pass，因此适合用来理解完整编译流程。
- **如果此前使用 `TILELANG_PASS_DIFF`，应切换到 `TL_LOWER_TRACE`**：它覆盖了旧工具的全部能力，并提供更多功能，后续改进也将集中在新工具上。
