# 软件流水线标注

> 英文原文：[Software Pipeline Annotations](../../programming_guides/software_pipeline.md)。本文为完整中文译文，代码块保留原样。翻译版本及术语说明见[阅读索引](../README.md)。

TileLang 可以根据 `T.Pipelined(..., num_stages=...)` 推断常见的生产者 / 消费者流水线。对于 GEMM 这类结构规则的内核，优先使用这种方式：

```python
for ko in T.Pipelined(T.ceildiv(K, BK), num_stages=3):
    T.copy(A[by * BM, ko * BK], A_shared)
    T.copy(B[ko * BK, bx * BN], B_shared)
    T.gemm(A_shared, B_shared, C_local)
```

如果内核的循环体具有特殊的执行顺序、额外的后处理步骤，或者需要手动安排异步拷贝分组，可以提供显式流水线标注。本指南介绍如何编写这些标注，并解释为什么可重放的标量 `Bind` 语句不属于用户可见的调度内容。

## 使用者需要理解的模型

流水线标注描述的是参与流水线执行的语句，而非循环体中的每一个 IR 节点。

应为实际执行工作的语句添加标注，包括：

- 拷贝，包括从 global 到 shared 的拷贝和 TMA 拷贝。
- 填充。
- GEMM 或其他 tile 操作。
- 归约。
- 写回和原子操作。
- 手动流水线中的显式等待、提交或同步语句。

不要为可重放的标量别名添加标注：

```python
base: T.int32 = ko * BK
offset: T.int32 = base + tx
```

这些 `Bind` 语句是值的定义，并未将值写入实际存储。编译器会自动在各个使用位置安排它们。

## 阶段与顺序：`stage` 和 `order`

每条参与调度的语句都有两个数字：

- `stage`：逻辑流水线阶段。
- `order`：生成调度后语句时使用的顺序。

阶段编号较小的语句在流水线中更早执行。典型的拷贝 / 计算流水线会把拷贝语句放在阶段 0，把计算语句放在阶段 1：

```python
for ko in T.Pipelined(
    T.ceildiv(K, BK),
    stage=[0, 0, 1],
    order=[0, 1, 2],
):
    T.copy(A[ko * BK], A_shared)
    T.copy(B[ko * BK], B_shared)
    T.gemm(A_shared, B_shared, C_local)
```

这两个数组按照源码顺序，与参与调度的语句一一对应。在上面的示例中，对应关系如下：

```text
statement 0: copy A  -> stage 0, order 0
statement 1: copy B  -> stage 0, order 1
statement 2: gemm    -> stage 1, order 2
```

应用标注后，编译器会检查缓冲区依赖关系。如果一条参与调度的语句为另一条语句生成数据，生产者就不能被安排在消费者之后。具体规则如下：

- 如果生产者与消费者位于同一个阶段，生产者的 `order` 必须小于消费者的 `order`。
- 如果它们位于不同阶段，生产者的 `stage` 必须小于或等于消费者的 `stage`。

手动提供 `stage` 和 `order` 时，常规代码中不要再同时设置 `num_stages`。流水线深度会根据阶段列表推断为 `max(stage) + 1`。需要编译器推断流水线时，单独使用 `num_stages`；需要手动调度流水线时，单独使用 `stage` / `order`。

## 为重排后的流水线添加标注

当需要先发出未来迭代中的生产者操作，再发出当前迭代中的消费者操作时，可以使用 `order` 数组。

```python
for ko in T.Pipelined(
    num_tiles,
    stage=[0, 1],
    order=[1, 0],
):
    T.copy(A[ko * BK], A_shared)
    T.gemm(A_shared, B_shared, C_local)
```

这里声明了两条参与调度的语句：

```text
copy -> stage 0, order 1
gemm -> stage 1, order 0
```

在流水线改写期间，编译器会为每个阶段使用正确的逻辑迭代，生成序幕、稳态循环体和收尾部分。源码顺序仍然决定每个标注条目对应哪条参与调度的语句；完成语句分类后，`order` 决定生成语句的顺序。

## 可重放的标量 `Bind` 语句

可重放的标量 `Bind` 可以作为一条语句出现在循环体中：

```python
for ko in T.Pipelined(
    num_tiles,
    stage=[0, 1],
    order=[1, 0],
):
    base: T.int32 = ko * BK
    T.copy(A[base + tx], A_shared[tx])
    T.gemm(A_shared, B_shared, C_local)
```

`stage` 和 `order` 数组包含两个条目，而非三个。这两个条目分别标注拷贝和 GEMM，特意省略了 `base` 的定义。

这条规则很重要，因为可重放的 `Bind` 没有唯一的流水线位置。它更接近 SSA 值或局部别名，而非一项可执行操作。如果同一个标量被不同阶段的语句使用，每个消费者可能都需要按照不同的逻辑流水线迭代来计算该值。

例如：

```python
for ko in T.Pipelined(
    num_tiles,
    stage=[0, 1],
    order=[1, 0],
):
    base: T.int32 = ko * BK
    T.copy(A[base + tx], A_shared[tx])
    B[base + tx] = A_shared[tx]
```

流水线改写后，拷贝可能需要未来生产者迭代所对应的 `base`，而写回则需要当前消费者迭代所对应的 `base`。如果让用户为 `base` 选择单一的 `stage` / `order`，就可能出现作用域不匹配，或者其中一处使用拿到了错误逻辑迭代的值。

因此，TileLang 将可重放的标量 `Bind` 视为不参与调度的语句：

```text
stage/order annotate only scheduled, effectful statements
replayable Bind definitions are replayed at each consumer
```

重放由使用位置驱动。如果某条消费者语句引用了流水线体中绑定的标量，编译器会在该消费者之前重新生成所需的 `Bind`，并代入该消费者的逻辑访问索引。

可重放的标量 `Bind` 可以包含对某个缓冲区的读取，前提是流水线体内不会写入该缓冲区：

```python
idx: T.int32 = Ids[ko]
T.copy(A[idx], A_shared)
B[idx] = A_shared[0]
```

这里的 `idx` 仍然是值的别名。如果两条参与调度的语句都使用它，TileLang 会在每个消费者处重放 `idx = Ids[logical_ko]`。这可能重复执行对 `Ids` 的读取，但保留了 `Bind` 的别名语义：该值按照消费者的逻辑流水线迭代计算。

这与将生产者结果写入实际存储的情况不同：

```python
idx_shared[tx] = Ids[ko]
T.copy(A[idx_shared[tx]], A_shared)
B[idx_shared[tx]] = A_shared[0]
```

对 `idx_shared` 的写入会产生实际存储，并形成真正的生产者 / 消费者依赖。它应该计入参与调度的语句，并在需要时像其他流水线缓冲区一样使用多个存储版本。

如果 `Bind` 读取的缓冲区在同一流水线体内会被写入，那么它不会被视为可重放的语句：

```python
T.copy(A[ko * BK], A_shared)
value: T.float32 = A_shared[tx]
C[ko * BK + tx] = value
```

对 `A_shared` 的读取依赖流水线中的生产者。TileLang 会把这个 `Bind` 保留在参与调度的语句列表中，因此它需要对应的 `stage` 和 `order` 条目。

由于标量 `Bind` 不具有存储版本管理，参与调度的 `Bind` 必须与所有使用其值的消费者位于同一个阶段。如果希望表达“只读取一次，然后供后续阶段共享”，应通过显式写入缓冲区或寄存器来物化该值，而非依赖 `Bind` 重放。

## `Bind` 的依赖关系

可重放的绑定可以依赖更早定义的其他可重放绑定：

```python
base: T.int32 = ko * BK
offset: T.int32 = base + tx
T.copy(A[offset], A_shared[tx])
```

编译器会先重放依赖项，再重放使用它们的项：

```text
base
offset
consumer statement
```

这样既能让手动标注集中描述实际的流水线操作，又能保留原始循环体中的词法标量依赖关系。

## 旧版标注形式

旧代码可能会把可重放的标量 `Bind` 语句也包含在标注数组中：

```python
for ko in T.Pipelined(
    num_tiles,
    stage=[3, 0, 1],
    order=[1, 2, 0],
):
    base: T.int32 = ko * BK
    T.copy(A[base + tx], A_shared[tx])
    B[base + tx] = A_shared[tx]
```

为保持兼容性，这种形式仍然有效。标量 `Bind` 对应的条目会被忽略，其余条目应用到参与调度的语句上：

```text
base Bind -> ignored
copy      -> stage 0, order 2
store     -> stage 1, order 0
```

如果旧版形式中标注的可重放标量 `Bind` 被多条参与调度的语句使用，TileLang 可能会发出警告，说明标量标注已被忽略，并会在每个使用位置重放该绑定。新代码应优先采用更短的标注数组，完全省略可重放的标量绑定。

## 手动添加 `T.serial` 标注

大多数用户代码应使用 `T.Pipelined`。较低层的测试或编译变换相关代码，也可以直接使用原始循环标注：

```python
for ko in T.serial(
    0,
    num_tiles,
    annotations={
        "software_pipeline_stage": [0, 1],
        "software_pipeline_order": [1, 0],
        "software_pipeline_async_stages": [0],
    },
):
    base: T.int32 = ko * BK
    T.copy(A[base + tx], A_shared[tx])
    B[base + tx] = A_shared[tx]
```

相同规则仍然适用：`software_pipeline_stage` 和 `software_pipeline_order` 只描述参与调度的语句。可重放的标量 `Bind` 语句不需要对应条目。

## 设计理由

流水线 pass 将循环体中的内容分为两类：

```text
scheduled statements: executable operations controlled by stage/order
replayable scalar bindings: local value aliases placed by use-def analysis
```

这种划分既能保持用户 API 稳定，也能避免调度歧义。真正的流水线操作有明确的执行位置，可以读取或写入缓冲区，也可能需要异步拷贝的状态管理或同步。让用户为这类操作指定阶段和顺序是合理的。

可重放的标量 `Bind` 不具备这些特征：

- 它没有副作用。
- 它不拥有缓冲区存储。
- 它可以被多条参与调度的语句使用。
- 它的正确取值可能依赖消费者的逻辑流水线迭代。

强制用户标注这类语句，就等于要求用户在不存在唯一正确答案时仍然选择一个 `stage` / `order`。在每个消费者处重放标量绑定，可以在生成的 IR 中明确表达别名语义，同时让用户可见的调度集中描述实际的流水线工作。

## 检查清单

编写手动流水线标注时：

- 构造 `stage` 和 `order` 时，只统计参与调度的语句。
- 省略 `base = ko * BK` 这类可重放的标量别名；如果流水线内不写入 `Ids`，也应省略 `idx = Ids[ko]`。
- 如果 `Bind` 读取的缓冲区在流水线体内会被写入，应将这个 `Bind` 计入参与调度的语句。
- 保证每条参与调度的语句具有唯一的 `order` 值。
- 按照 `stage` / `order` 的依赖规则，保证生产者先于消费者。
- `software_pipeline_async_producers` 和 `software_pipeline_async_producer_groups` 的长度应与参与调度的语句列表长度相同。
- 新代码中，可重放的别名应优先使用省略绑定条目的形式；只有维护旧内核时，才依赖旧版标注中的绑定占位条目。
