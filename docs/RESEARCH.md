# 相关工作与设计调研

> 目标不是做“又一个模型图查看器”，而是回答一个具体问题：
> **拿到一个陌生（甚至 AI 生成）的 PyTorch 文件夹，怎样在几分钟内看懂它的空间结构？**

## 1. 已有工具

| 工具 | 输入 | 表示 | 强项 | 对本题的不足 |
|---|---|---|---|---|
| [Netron](https://github.com/lutzroeder/netron) | 模型文件（ONNX/TFLite/PyTorch 等） | 2D 节点-边图 | 格式支持极广、信息密度高、成熟稳定 | 面向已导出的模型文件；没有源码文件夹发现；无张量体积/3D |
| [torchview](https://github.com/mert-kurttutan/torchview) | PyTorch 模型对象 | 2D 计算图（Graphviz） | 递归模块图、可导出、轻量 | 需写 Python 调用；2D DAG，缺少空间/通道体积感 |
| [TorchVista](https://github.com/sachinhosmani/torchvista) | PyTorch 模型对象 | notebook 内交互式前向图 | 单行调用、可导 HTML/SVG、交互好 | 绑定 notebook；非文件夹扫描；2D |
| [visualkeras](https://github.com/paulgavrikov/visualkeras) | Keras 模型 | 分层“2.5D”示意图 | CNN 分层块很有辨识度 | 主要面向 Keras；静态图；不解析源码 |
| [Transformer Explainer](https://poloclub.github.io/transformer-explainer/) | 固定 GPT-2 示例 | 网页交互式解释 | 把 attention/FFN 过程讲得很清楚 | 不是通用模型解析器，不能吃任意文件夹 |

结论：现有工具在“格式覆盖 / 2D 图信息密度 / 单模型调用”上已经很强，
但普遍缺少三件事：

1. **零配置的文件夹级发现**（哪个文件、哪个类才是模型？）；
2. **语义级组件识别**（这是 CNN block 还是 Transformer block？attention 有几个头？）；
3. **同时表达空间张量与组件层级的 3D 交互视图**。

NetPrism 的差异化就押在这三点上。

## 2. 设计原则

### 2.1 先“可信”，再“好看”

模型可视化最容易犯的错是画得很漂亮但形状是猜的。因此：

- 优先 `torch.fx` 真实执行前向图，形状/参数量来自真实张量；
- 静态 AST 是一定要有的回退（陌生代码、必填参数、缺依赖、上传场景），但要在 UI 上明确标注
  `static` 徽章与 warning；
- FLOPs 标为估算，并公开公式（1 MAC = 2 FLOPs），不制造“精确”的假象。

### 2.2 渐进披露（progressive disclosure）

一张图不能同时回答所有问题。界面按三层组织：

1. **全局**：模型摘要、组件分组框、层列表树——先看“骨架”；
2. **局部**：悬停/选中某层，显示形状、卷积核、heads、FFN 扩张比、源码；
3. **细节**：右侧详情面板、源码片段、warnings。

### 2.3 3D 要有信息含义，而不是装饰

每个视觉维度都映射到真实数据，避免“3D 旋转木马”：

| 视觉通道 | 映射 |
|---|---|
| X 轴位置 | 执行顺序（图的拓扑序） |
| 立方体 Y/Z | 特征图 H/W（保持长宽比） |
| 立方体 X 厚度 | 通道数（log 缩放，避免 3→1024 时首尾差 300 倍） |
| 立方体颜色/透明度 | 层 kind（conv/norm/activation/attention/…） |
| 输入侧网格 | 卷积核 `kH×kW`，网格深度反映 `Cin/groups` |
| 层叠薄片 | attention heads（片数 = `numHeads`） |
| 两段板 | FFN 扩张→收缩 |
| 曲线颜色/弧度 | data / residual / skip 边 |
| 组线框 | 组件边界（conv block / transformer block / encoder…） |

### 2.4 3D 里最重要的是“可读”，不是“真实”

- 通道数用 `log` 缩放并 clamp，否则深层网络会变成一排墙；
- 标量/1×1 特征图给最小可见厚度，避免“消失”；
- 默认只显示组标签和少量关键层标签，hover/选中才显示全名；
- 选中时上下游高亮、其余降透明度（focus + context），比单纯放大更利于理解；
- 提供“适应窗口/复位/俯视/正视”预设，3D 场景没有稳定视角很容易迷路。

## 3. 解析层面的调研结论

### 3.1 FX 的 leaf 策略是关键

PyTorch FX 默认把 `torch.nn` 下的模块都当 leaf，导致 `TransformerEncoderLayer` 只显示成一个黑盒。
我们采用**选择性展开**：原子层保留 leaf，复合容器/Transformer 层/自定义 block 展开。
这样既能拿到真实连边，又能在分组阶段恢复“这是一个 Transformer block”的语义。
（参考：[torch.fx 文档](https://pytorch.org/docs/stable/fx.html)）

### 3.2 静态解析必须“宽容”

AI 生成的模型代码通常风格不统一：链式调用、辅助 `_forward_impl`、`ModuleList` 循环、
参数来自 `__init__` 默认值。静态解析器按“能解析多少算多少”设计：保留节点、标注未知维度、
warning 不崩溃——这比“要么全对要么失败”更符合真实使用。

### 3.3 组件识别靠“结构 + 命名”双信号

只靠类名会把所有 `Block` 都叫 block；只靠结构又会在自定义命名上失效。
我们合并两类信号：qualified name 的层级 + 子节点组成模式（attention/FFN/norm/conv/add），
再给出人类可读 label。这一思路与 Netron 的 scope 分组、torchview 的 nested module 图一致，
但额外输出 `kind` 与 `label` 供 3D 场景直接消费。

## 4. 交互调研结论

- **Hover 优于 click**：3D 场景中先给轻量 tooltip，点击才打开详情，避免每次查看都打断视角。
- **Orbit + focus**：拖拽旋转是 3D 的默认心智；双击聚焦/复位是防止“飞丢”的必要安全网。
- **列表与 3D 双向联动**：层列表提供确定性的导航，3D 提供空间关系；只给其中一个都会让用户迷路。
- **动画克制**：数据流播放用于讲清顺序，不应持续闪烁；`prefers-reduced-motion` 时关闭。
- **移动/窄屏**：侧栏可收起，canvas 优先；不做重交互的移动端 3D 编辑。

## 5. 参考链接

- Netron — <https://github.com/lutzroeder/netron>
- torchview — <https://github.com/mert-kurttutan/torchview>
- TorchVista — <https://github.com/sachinhosmani/torchvista> / <https://sachinhosmani.github.io/torchvista/>
- visualkeras — <https://github.com/paulgavrikov/visualkeras>
- Transformer Explainer — <https://poloclub.github.io/transformer-explainer/>
- PyTorch FX — <https://pytorch.org/docs/stable/fx.html>
- PyTorch `torch.fx.passes.shape_prop` — <https://pytorch.org/docs/stable/fx.html#torch.fx.passes.shape_prop.ShapeProp>

> 调研时间：2026-11。工具能力以其官方文档/仓库为准，本文只做定位对比，不评价其实现质量。
