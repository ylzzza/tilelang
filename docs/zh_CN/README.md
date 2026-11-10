# TileLang 中文学习文档

这里收录学习路线中推荐阅读的四篇文档的完整中文译文。

译文基于本地仓库提交 `c58a86f02b456fc3618b0dd86e66d52a0d0f49f7`，翻译日期为 2026 年 10 月 26 日。正文、表格和注意事项已翻译；代码块中的代码、注释、命令、示例输出和目录树保留原样，便于与英文原文对照。每篇译文开头都提供了英文原文链接。

## 建议阅读顺序

| 顺序 | 中文文档 | 学习重点 |
| --- | --- | --- |
| 1 | [语言基础](programming_guides/language_basics.md) | 内核定义、计算网格、循环、存储空间、数据搬运和 JIT |
| 2 | [软件流水线标注](programming_guides/software_pipeline.md) | 自动流水线、手动 `stage` / `order` 标注，以及标量 `Bind` 的处理规则 |
| 3 | [IR 降低过程跟踪：IR Lower Trace](tools/lower_trace.md) | 查看各个编译 pass 前后的 IR，理解代码生成和编辑后重新编译流程 |
| 4 | [仅编译工具：Compile-Only](tools/compile_only.md) | 在不运行内核的情况下生成并查看源码，包括没有 GPU 时的学习方式 |

如果当前没有可用的 GPU，可以在读完语言基础后，先读第 4 篇，用默认 `c` 目标观察编译结果。

## 术语对照

| 英文 | 本组译文中的用法 |
| --- | --- |
| kernel | 内核 |
| tile | 数据块，或保留 `tile` |
| memory scope | 存储空间 |
| fragment | 数据片段，通常保留 `fragment`，以对应 API 名称 |
| layout | 布局 |
| lowering | 降低抽象层级的编译转换，简称“降低” |
| IR | 中间表示 |
| pass | 编译变换，通常保留 `pass`，以便对应源码 |
| producer / consumer | 生产者 / 消费者 |
| replayable Bind | 可重放的 `Bind`，即按使用位置重新生成的标量值绑定 |
| host / device | 主机端 / 设备端 |
| codegen | 代码生成 |

版本号、兼容性和后端限制按对应英文原文翻译；后续学习新版代码时，可以通过原文链接核对变化。
