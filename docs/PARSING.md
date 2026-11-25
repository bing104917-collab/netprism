# 解析算法说明

本文解释 `netprism.core` 如何从“一个 Python 文件”得到“带形状、连边、组件分组的图”。
实现可能与本文细节略有出入，但契约与整体策略以本文和 [`SPEC.md`](../SPEC.md) 为准。

## 1. 文件与模型发现

- 递归扫描 `*.py`，跳过 `.git/.venv/venv/node_modules/__pycache__/site-packages/build/dist` 等目录，
  单文件上限 2 MB，默认最多 200 个文件。
- 解析 AST，识别 `class X(nn.Module)`、`class X(torch.nn.Module)`、以及 `Module` 的导入别名。
- 只保留含 `forward` 的类；同时识别模块级**工厂函数**（如 `def ResNet18(): return ResNet(BasicBlock, ...)`），
  无必填参数的工厂会被实际调用并作为模型，图名用工厂名（`ResNet18`），`className` 保留底层类名。
- **顶层模型判定**：
  - 若类 A 的 `__init__`/其他方法里出现 `B(...)` 且 B 也是本文件的 Module 类，则 B 视为积木；
  - 扩展为**跨文件引用图**：扫描整目录导入（`from .unet_parts import *` / 显式 import），
    若 B 在另一个文件 C 的类方法里被实例化为子模块（`self.down = Down(...)`），B 也视为组件，
    整个文件夹扫描时不再作为顶层模型（单文件 `analyze_file` 行为不变）；
  - 若某类没有被引用、且文件内存在其他模型，则作为顶层模型；
  - 兼容 `model = MyNet(...)`、`def build_model()` / `get_model()`。
- 一个文件可以有多个顶层模型；**工厂变体**（ResNet18/34/50…）默认在 CLI/HTTP 展示层折叠为每个类一个代表，
  `--all-variants` 保留全部；`--class ResNet50` 可只解析指定变体。

## 2. Trace 模式（mode=`trace`）

### 2.1 为什么不用默认的 `torch.fx.symbolic_trace`

默认 FX 对 `torch.nn` 命名空间下的模块一律视为 leaf，于是 `TransformerEncoderLayer`、`BasicBlock`
会变成一个黑盒节点——看不到内部的 attention/FFN/卷积。NetPrism 自定义 `Tracer.is_leaf_module`：

| 模块 | 处理 |
|---|---|
| 原子层（Conv/Linear/Norm/激活/池化/Dropout/Embedding/RNN） | leaf，单独成节点 |
| `nn.MultiheadAttention` | leaf（但读取 `num_heads/embed_dim`，前端画多头薄片） |
| `Sequential / ModuleList / ModuleDict` | 展开 |
| `TransformerEncoderLayer / Encoder / Decoder*` | 展开（保留层级路径，供分组识别） |
| 自定义 `nn.Module` | 展开，暴露自定义 block 的内部 |

展开的副作用是图更细，但组件身份可以通过 **qualified name**（如 `encoder.layers.0.self_attn`）
与结构模式恢复；这比只显示一个 “TransformerEncoderLayer” 黑盒更有信息量。

### 2.2 输入推断

Trace 需要一组 dummy 输入。顺序：

1. `--input-shape 1,3,224,224`（用户显式指定）；
2. 模型首层是 `nn.Embedding` → `(1, 16)` `int64`（再试 8/32）；
3. 首层 `Conv2d` → `(1, in_channels, 64, 64)`（失败再试 32/128/224）；
4. `Conv1d` → `(1, C, 64)`；`Conv3d` → `(1, C, 16, 16, 16)`；
5. 首层 `Linear` → `(1, in_features)`；`MultiheadAttention` → `(L, 1, embed_dim)`；
6. 兜底 `(1, 3, 64, 64)`。

每换一组候选就重新 forward；全部失败则走 hook 回退/静态回退，并记录 warning。
对 32×32 的小模型使用 64×64 通常也能跑（卷积/池化形状自适应）；对含固定 flatten 维度的分类头，
若 64 不匹配会再试 32。ViT 类模型会结合 `patch_size`、pos-embedding 形状推断 256/32 等输入尺寸。

### 2.2b 包内相对导入与参数合成

真实仓库的模型常位于包内（`unet/unet_model.py` 里 `from .unet_parts import *`）。导入器会：

- 向上寻找包根（`__init__.py` 链），把包父目录加入 `sys.path`，用 dotted module name 导入；
- 单文件 `analyze_file` 同样可用，不需要用户手动设置工作目录；
- 导入失败时仍按原有链条回退 hook/static。

模型构造参数不足时，`synth.py` 依次尝试：

1. 同文件/同包 dataclass 的默认实例（如 nanoGPT 的 `GPTConfig()`）；
2. 同文件或同目录调用点的字面量（如 `VGG('VGG11')`、`ShuffleNetV2(net_size=0.5)`）；
3. 常见参数名启发（`n_channels=3`、`n_classes/num_classes=10`、`image_size=32`、`vocab_size=128` 等）；
4. 仍无法自信合成（list/callable/未知类型）→ static + 明确 warning，不猜。

所有合成参数都会让 graph 标记 `flopsApprox=true` 并在 `model.warnings` 说明，避免把假设当精确值。

### 2.3 形状记录与连边

- 用 `torch.fx.Interpreter` 子类在 `run_node` 中先 `fetch_args_kwargs_from_env`，抓取每个参数的
  张量形状，再执行 `super().run_node`，得到输出形状；`torch.no_grad()`、CPU、`eval()`。
- 输入/输出形状统一转成 `list[int]`；`-1` 表示动态维。
- 连边：递归检查 node 的 args/kwargs 中引用的上游 FX node。这样残差 `x + self.conv(x)`、
  U-Net `torch.cat([x, skip], 1)` 都会产生真实的跨层边。
- `call_function` / `call_method` 节点：保留有语义的（flatten/reshape/permute/add/cat/relu/softmax/
  dropout/upsample 等）；`getitem`、`get_attr`、纯常量运算尽量折叠/跳过。
- 同一模块多次调用（权重共享）每次调用建一个节点，名称加 `#2` 等后缀。

### 2.4 保底路径

1. FX 失败（数据相关控制流、不可 trace 的 Python 逻辑）→ **eager hook**：
   给原子子模块注册 forward hook，按执行顺序记录 I/O，边退化为顺序边，warning 标注“跳连可能缺失”。
2. 实例化/forward 失败（必填参数、缺依赖、运行时报错）→ **静态 AST**，mode=`static`。

## 3. 静态 AST 模式（mode=`static`）

完全不导入模型代码，只读源码。目标不是 100% 还原，而是“对典型代码给出可读结构”。

### 3.1 `__init__` 层收集

识别：

```python
self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
self.bn1 = nn.BatchNorm2d(64)
self.blocks = nn.Sequential(...)
self.layers = nn.ModuleList([...])
self.attn = SelfAttention(dim, heads)
```

参数解析支持：整数字面量、`__init__` 默认参数、类/模块常量、简单四则运算与 `//`。
解析不了的参数记为不确定（`-1`/warning），但仍保留节点。

### 3.2 `forward` 数据流

支持模式：

```python
x = self.conv1(x)                                  # 顺序
x = F.relu(self.bn1(self.conv1(x)))                # 链式嵌套
x = self.conv(x) + identity                        # 残差
x = torch.cat([x, skip], dim=1)                    # 拼接
x = x.flatten(1) / torch.flatten(x, 1) / x.view(x.size(0), -1)
for block in self.blocks: x = block(x)             # ModuleList 顺序
x = self.encoder(x)                                # 自定义子模块
```

对 `def _forward_impl(...)` 等辅助方法会跟进解析；分支/循环中无法静态确定的情况保留节点并 warning。
当自定义子模块本身可解析时，会将其内部层内联到父图，并用 group 标记组件边界。

### 3.3 形状推导公式

记输入 `(N, C, H, W)`：

- Conv2d：`H' = floor((H + 2p - d(k-1) - 1) / s) + 1`，`C' = out_channels`；
- ConvTranspose2d：`H' = (H-1)*s - 2p + d(k-1) + output_padding + 1`；
- MaxPool/AvgPool：同卷积公式；
- AdaptiveAvgPool：输出为给定的 `output_size`；
- Linear：最后一维变为 `out_features`（前面的维度视为 batch）；
- Flatten：从 `start_dim` 起合并；
- BatchNorm/LayerNorm/激活/Dropout：形状不变；
- Embedding：`(N, L) -> (N, L, D)`；
- MultiheadAttention（batch_first）：`(N, L, D) -> (N, L, D)`；
- LSTM/GRU：`(N, L, D) -> (N, L, H*dirs)`（单层、无初始 hidden）。

动态/无法确定的维度记 `-1`，不编造。

## 4. 参数量与 FLOPs

统一约定：**1 次乘加 = 2 FLOPs**，bias 加法单独计入；只统计主要算子，norm/激活/池化按元素近似，
前端展示为“估算”（`flopsApprox`）。

| 层 | 参数量 | FLOPs |
|---|---|---|
| Conv2d | `Cout*(Cin/groups)*kH*kW + bias*Cout` | `2*N*Hout*Wout*Cout*(Cin/groups)*kH*kW + bias*N*Cout*Hout*Wout` |
| Linear | `Cin*Cout + bias*Cout` | `2*N*Cin*Cout + bias*N*Cout` |
| BatchNorm2d | `2*C`（affine） | `≈4*N*C*H*W` |
| Embedding | `num_embeddings*embedding_dim` | 0（查表） |
| MultiheadAttention | `4*D*D + 4*D`（qkv+out，含 bias） | `4*N*L*D^2 + 2*N*L^2*D` |
| LSTM | `4*((D+H)*H + H)`（单层单向） | `≈2*N*L*4*H*(D+H)` |

Trace 模式直接用 `sum(p.numel())` 与 `requires_grad` 统计参数；FLOPs 用真实形状代入上表。
静态模式用推导形状代入；无法确定的项记 0 并在 warning 里说明。

## 5. op → kind 规则

`kind` 是前端配色/图例/过滤的唯一依据，约 25 类。匹配优先级：

1. 精确类名/后缀（`Conv2d`、`BatchNorm2d`、`MultiheadAttention`、`TransformerEncoderLayer`…）；
2. 关键字包含（`Attention`→`attention`，`Norm`→`norm`，`Pool`→`pool`，`Embedding`→`embedding`）；
3. 自定义结构启发式（含 `FFN|FeedForward`→`ffn`，`Block|ResBlock|Bottleneck`→`block`，
   `Encoder|Decoder`→`block/encoder/decoder`，含 `MLP`→`mlp`）；
4. 兜底 `custom` / `unknown`。

`params` 从模块属性/构造参数提取（如 `kernel_size`、`stride`、`num_heads`），并展开 tuple/list 为数组。

## 6. 组件分组

分组有两类信号：

**层级信号**：模块 qualified name 的父路径天然形成树，例如
`features.0` → `features`；`encoder.layers.0.self_attn` → `encoder` → `encoder.layers.0`。

**结构信号**（在层级内检查）：

- 多个 Conv + Norm/激活 → `conv_block`；
- 出现 residual add / 跨层边 → `residual_block`；
- attention + 2 个 Linear + ≥1 Norm → `transformer_block`；
- attention 但没有 FFN → `attention_block`；
- 连续 Linear → `ffn_block` / `mlp`；
- `encoder`/`decoder` 命名或 U-Net down/up + concat → `encoder`/`decoder`/`unet_block`。

分组约束：node 只属于一个直接 group；`parentId`/`childGroupIds` 双向一致；group 的参数量/FLOPs 是
直接成员 + 子组之和。终端 `inspect` 与前端层列表都直接消费这棵树。

## 7. 失败模式与提示

解析器不会因为一个文件失败而中断整个文件夹扫描；错误进 `index.errors`。常见 warning：

- `FX trace failed: ...` → 已回退 hook/静态，形状可能仍准确（hook）。
- `Could not infer input shape` → 手写 `--input-shape`。
- `Model requires constructor arguments` → 静态解析。
- `Dynamic control flow` / `Unsupported pattern` → 该分支形状未推导。
- `Skip connections may be missing (hook fallback)` → 残差边可能退化为顺序边。
- `FLOPs are approximate` → 注意力/静态推导。

## 8. 校验

`scan` 写盘前调用 `validate_graph`：ID 唯一、边端点存在、group 成员与 `groupId` 一致、
parent/children 双向、无环、形状/数值类型合法。这样前端可以假定数据自洽，不必在每个组件里做防御式解析。
