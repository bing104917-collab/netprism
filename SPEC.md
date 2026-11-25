# NetPrism — 产品与工程规格（v1.0，冻结）

> v2 增量（ONNX/TorchScript/Keras、metrics、report/compare、表格 UI）见 [`SPEC-V2.md`](SPEC-V2.md)；
> 本文件仍是 graph/CLI 的基础契约，v2 只做可选字段加法。

> **一句话**：把任意一个装满 PyTorch 模型代码的文件夹扔给 NetPrism，一条命令解析出网络结构，
> 在浏览器里用可交互 3D 场景看清每一层的张量形状、卷积核、注意力和 FFN。
>
> 代号 **NetPrism / 棱镜**。命令行 `netprism`。UI 全中文，代码与注释英文为主。

本文件是后端、前端、样例/QA 三条线共同的契约。**任何跨模块的字段名、CLI 参数、HTTP 路径都以本文件为准**；
如需变更，先通知 Lead，由 Lead 改本文件并通知所有人。

---

## 0. 环境（重要）

- 工作区根：`C:\Users\hrbkcy2523\Documents\deepseek-harness\default-workspace`
- 项目根：`C:\Users\hrbkcy2523\Documents\deepseek-harness\default-workspace\netprism`
- Python（唯一可用解释器，**不在 PATH 里**）：
  `C:\Users\hrbkcy2523\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe`
  - 已安装 / 正在安装：`torch`（CPU 版）、`pytest`、`jsonschema`。安装完成标志：
    `& $py -c "import torch, pytest, jsonschema; print(torch.__version__)"` 成功。
  - 运行测试一律：`& $py -m pytest tests -q`（在项目根目录）。
- Node：`node -v` 可用（v24），npm/pnpm 可用；前端**不要引入构建步骤**，用原生 ES modules。
- 浏览器：Chrome / Edge 已安装，可做 headless 截图验证。
- 网络可用，但**最终产物运行时不依赖外网**：three.js 必须 vendor 到 `web/vendor/`。
- 平台 Windows，路径用 `pathlib`；shell 是 PowerShell。

---

## 1. 仓库结构（写作用域）

```
netprism/
  SPEC.md                     # 本文件（Lead）
  README.md                   # 用户文档（Lead）
  pyproject.toml              # 打包（Lead）
  netprism/
    __init__.py  __main__.py  # Lead
    schema.py                 # JSON 契约 + 校验（Lead，冻结）
    cli.py  server.py         # CLI + HTTP 服务（Lead）
    core/                     # 解析引擎（backend-dev 独占）
      __init__.py             # 公开 API：AnalyzeOptions / analyze_file / analyze_folder
      discover.py             # 文件与模型发现
      ops.py                  # 算子识别、形状推导、FLOPs/参数量
      astscan.py              # 静态 AST 解析
      tracer.py               # torch.fx / hook 动态追踪
      groups.py               # block / 组件分组
      build.py                # 统一成 graph dict
  web/                        # 前端（frontend-dev 独占）
    index.html
    css/  js/  vendor/  demo/
  samples/                    # 样例模型（samples-qa 独占）
  tests/                      # 测试（按文件分工）
  examples/demo/              # 演示数据（backend-dev 生成；Lead 可放夹具）
  docs/                       # 文档（Lead；frontend-dev 写 web/README.md）
  schema/graph.schema.json    # JSON Schema（Lead）
```

**写作用域纪律**：只改自己拥有的文件/目录；需要改别人的文件时发消息让 owner 改。
`SPEC.md`、`netprism/schema.py`、`schema/graph.schema.json`、`netprism/cli.py`、`netprism/server.py`、
`pyproject.toml`、`README.md`、`docs/` 归 Lead。共享文件只读。

---

## 2. 产品形态

### 2.1 三条使用路径

```powershell
# ① 一键：解析文件夹 + 起服务 + 开浏览器
python -m netprism view .\some_models --open

# ② 只解析，产出 JSON
python -m netprism scan .\some_models --out .\netprism_out

# ③ 起服务看已有结果（或内置 demo）
python -m netprism serve .\netprism_out --port 8765 --open

# 附：终端里快速看结构
python -m netprism inspect .\samples\resnet_mini.py
```

> 真实仓库常有 `ResNet18()/ResNet34()/ResNet50()` 等同文件工厂变体：CLI/Web 默认每个类只展示第一个，
> `--all-variants` 保留全部，`--class ResNet50` 精确选择。折叠只发生在展示层，`analyze_folder` 仍返回全部。

### 2.2 Web 场景必须支持的交互

- 3D 场景：拖拽旋转（OrbitControls）、滚轮缩放、右键/双指平移。
- 悬停任意层 / 卷积核 / 连线 → tooltip：层名、op、kind、输入输出形状、参数量、FLOPs、超参、源码位置。
- 单击选中 → 右侧详情面板 + 高亮上下游；双击层聚焦；双击空白复位。
- 左侧层列表（按 group 分级、可折叠），点击定位并聚焦；搜索框过滤/高亮。
- 图例过滤：按 kind（conv/norm/activation/...）显隐。
- 分组框显示/隐藏；卷积核显示/隐藏；标签显示/隐藏；重置视角；适应窗口。
- 数据流播放：一个小光点沿执行顺序流动，高亮当前层（可暂停/单步）。
- 模型选择器：从 `/api/index` 读取多个模型；显示参数量/FLOPs/模式徽章（trace 精确 / static 静态推断）。
- 拖拽/选择**一个文件夹**上传解析（只接收 `.py`，默认静态解析，不执行代码），解析完直接可视化。
- 空状态/加载状态/错误状态都要有设计，不能白屏。
- 视口尺寸变化自适应；窄屏时侧栏可收起。
- 控制台零报错；不依赖 CDN。

---

## 3. 数据契约（graph JSON v1.0）

**权威定义**：`schema/graph.schema.json`（结构）+ `netprism/schema.py`（语义校验）。
后端产出的每个模型 graph 必须通过 `netprism.schema.validate_graph(graph)`（返回错误列表，必须为空）。
前端只依赖这里写明的字段；`params` 里未知字段可忽略。

### 3.1 顶层

```json
{
  "schemaVersion": "1.0",
  "generator": {"name": "netprism", "version": "0.1.0"},
  "model": {
    "name": "ResNetMini",
    "framework": "pytorch",                 // pytorch | onnx | unknown
    "className": "ResNetMini",              // 可为 null
    "mode": "trace",                         // trace | static | hybrid | onnx
    "source": {"file": "samples/resnet_mini.py", "relPath": "resnet_mini.py"},
    "inputSpec":  [{"name": "input", "shape": [1, 3, 32, 32], "dtype": "float32"}],
    "outputSpec": [{"name": "logits", "shape": [1, 10], "dtype": "float32"}],
    "totalParams": 123456,
    "trainableParams": 123456,
    "totalFlops": 1234567,
    "flopsApprox": true,                     // 是否为估算（静态推导 / 注意力近似）
    "numLayers": 23,                         // 叶子层数量，不含 input/output
    "warnings": ["..."]
  },
  "nodes": [ ... ],
  "edges": [ ... ],
  "groups": [ ... ]
}
```

### 3.2 node

```json
{
  "id": "n7",
  "name": "block1.conv1",
  "op": "Conv2d",
  "kind": "conv",
  "modulePath": "block1.conv1",
  "depth": 1,
  "inShapes": [[1, 32, 32, 32]],
  "outShapes": [[1, 32, 32, 32]],
  "params": {
    "inChannels": 32, "outChannels": 32, "kernelSize": [3, 3],
    "stride": [1, 1], "padding": [1, 1], "dilation": [1, 1],
    "groups": 1, "bias": true
  },
  "numParams": 9248,
  "flops": 18907136,
  "tags": ["spatial", "weighted"],
  "groupId": "g3",
  "source": {"file": "samples/resnet_mini.py", "line": 27, "code": "nn.Conv2d(32, 32, 3, padding=1)"},
  "notes": []
}
```

- `kind` 枚举（前端配色依据）：
  `input, output, conv, deconv, linear, mlp, norm, activation, pool, dropout, flatten, reshape,
   attention, ffn, embedding, positional, add, concat, upsample, rnn, softmax, block, container, custom, unknown`
- `inShapes` / `outShapes` 是**张量列表**（多输入/多输出模块，如 LSTM、MHA 的 k/v）。形状用 int 数组，动态维用 `-1`。
- `params` 必填对象，可含任意 JSON 值；常用键：
  - conv：`inChannels, outChannels, kernelSize, stride, padding, dilation, groups, bias`
  - linear：`inFeatures, outFeatures, bias`
  - norm：`normalizedShape, eps, affine`
  - pool：`kernelSize, stride, padding, type`
  - attention：`embedDim, numHeads, headDim, kdim, vdim, batchFirst, dropout`
  - ffn/mlp：`inFeatures, hiddenFeatures, outFeatures, activation`
  - embedding：`numEmbeddings, embeddingDim`
  - dropout：`p`
- `tags`：`spatial, sequence, weighted, normalized, residual, skip, head, projection, token, channel` 等，前端可用于筛选。
- `groupId`：直接所属分组 id；顶层可为 `null`。
- `source` 可为 `null`；有则尽量给 `line` 与 `code`。
- `notes` 可为空数组。

### 3.3 edge

```json
{"id": "e9", "source": "n3", "target": "n7", "kind": "residual",
 "shapes": [[1, 32, 32, 32]], "label": "skip"}
```

- `kind`：`data | skip | residual | attention | control`。
- 顺序边默认 `data`；跨层直连（residual / U-Net skip）用 `residual` / `skip`；attention 内部可视连边用 `attention`。

### 3.4 group

```json
{
  "id": "g3",
  "name": "layer1.0",
  "label": "Residual Block 1",
  "kind": "residual_block",
  "parentId": "g1",
  "depth": 2,
  "nodeIds": ["n7", "n8", "n9"],
  "childGroupIds": [],
  "numParams": 36992,
  "flops": 37748736,
  "source": {"file": "samples/resnet_mini.py", "line": 22}
}
```

- `kind`：`model, sequential, module, conv_block, residual_block, attention_block, transformer_block,
  ffn_block, mlp, encoder, decoder, unet_block, custom`
- 层级：`parentId` / `childGroupIds`，`depth` 从 0 开始；`nodeIds` 只放**直接**成员。
- 无分组时 `groups: []`，节点 `groupId: null`。
- 约定必有一个顶层 `kind:"model"` 根组（可无直接节点）。

### 3.5 index.json（scan 输出）

```json
{
  "schemaVersion": "1.0",
  "generatedAt": "2026-11-24T12:00:00+08:00",
  "root": "C:/.../samples",
  "models": [
    {
      "id": "resnet_mini__resnetmini",
      "name": "ResNetMini",
      "className": "ResNetMini",
      "file": "resnet_mini__ResNetMini.json",
      "source": "resnet_mini.py",
      "mode": "trace",
      "numLayers": 20,
      "totalParams": 123456,
      "totalFlops": 1234567,
      "warnings": 0
    }
  ],
  "errors": [{"file": "bad.py", "message": "..."}]
}
```

- 每个 graph 单独写 `<id>.json`；`id` 为 `file_stem__ClassName` 的 slug（全小写、非字母数字转 `_`）、去重。
  `scan` 的 `--out` 目录里**只写**这些文件。

---

## 4. 解析后端（backend-dev）

### 4.1 公开 API（不可改签名）

```python
# netprism/core/__init__.py
from dataclasses import dataclass, field

@dataclass
class AnalyzeOptions:
    execute: bool = True              # 允许动态导入并执行模型（trace 模式）
    class_name: str | None = None     # 只解析指定类
    input_shape: tuple[int, ...] | None = None
    max_files: int = 200
    recursive: bool = True
    verbose: bool = False

@dataclass
class FolderResult:
    index: dict                       # 见 3.5
    graphs: dict[str, dict]           # id -> graph

def analyze_file(path: str | Path, options: AnalyzeOptions | None = None) -> list[dict]:
    """返回该文件里发现的所有模型 graph；解析失败抛 AnalyzeError。"""

def analyze_folder(folder: str | Path, options: AnalyzeOptions | None = None) -> FolderResult:
    """扫描文件夹；单个文件失败写入 index.errors，不中断。"""
```

- 不依赖 `netprism.cli` / `netprism.server`。
- **模块顶层不得 import torch**；静态模式必须能在没装 torch 的环境工作。
- 必须使用 `netprism.schema` 的常量与校验（只 import，不改）。
- 输出必须是纯 JSON 可序列化类型（int/float/str/bool/None/list/dict）。

### 4.2 发现规则

- 递归扫描 `*.py`，跳过 `.git, .venv, venv, env, node_modules, __pycache__, site-packages, build, dist, .netprism_out` 和隐藏目录；单文件 < 2 MB；最多 `max_files`。
- AST 找 `class X(nn.Module)` / `(torch.nn.Module)` / 导入别名；必须有 `forward`。
- **顶层模型判定**：文件内若 A 的 `__init__` 或类方法里实例化了 B（同为 Module 类），B 视为积木、不做独立模型；
  文件夹扫描时，若 B 在另一个文件的类方法里被作为子模块实例化（跨文件组件），B 也不做顶层模型；
  模块级无参工厂（`def ResNet18(): return ResNet(...)`）作为顶层模型，图名用工厂名。
- **工厂变体**：同文件多个工厂都解析；CLI/HTTP 展示层默认同类折叠为一个，`--all-variants` 全显。
- 文件里也可有 `model = X(...)` 或 `def build_model()/get_model()`。
- 一个文件可有多个顶层模型（都解析）；`class_name` 指定时只解析该类/工厂。
- 非模型文件（无 Module 类）静默跳过。

### 4.3 动态 trace（首选，mode="trace"）

目标：准确形状 + 真实连边 + 完整层级。

1. 动态导入文件（`importlib.util.spec_from_file_location`，临时把文件父目录加入 `sys.path`），实例化顶层类：
   - 优先 `cls()`；失败时尝试 `build_model()/get_model()` / 模块级 `model`；
   - 只允许无参或全默认参数实例化；需要参数的模型回退静态解析并给 warning。
2. 自动推断 dummy 输入（按顺序尝试，直到 forward 成功）：
   - `AnalyzeOptions.input_shape` 优先；
   - 首层 Embedding → `(1, 16)` int64（再试 8/32）；
   - 首层 Conv2d → `(1, C, 64, 64)`（失败试 32/128/224）；
   - Conv1d → `(1, C, 64)`；Conv3d → `(1, C, 16, 16, 16)`；
   - 首层 Linear → `(1, in_features)`；MHA → `(16, 1, embed_dim)`；
   - 都不行 → `(1, 3, 64, 64)`。
3. 用**自定义 FX Tracer**（`torch.fx.Tracer` 子类）选择性展开：
   - 原子层保持 leaf：Conv/Linear/norm/activation/pool/dropout/embedding/RNN/`nn.MultiheadAttention` 等 torch.nn 原子模块；
   - 容器/复合模块展开：`Sequential, ModuleList, ModuleDict`、`TransformerEncoderLayer/Encoder/Decoder`、
     自定义 nn.Module（这样能看到 block 内部）；
   - 若 `symbolic_trace` 因控制流失败 → 用 eager hook 回退：按执行顺序记录原子模块的 I/O 形状，
     连边退化为顺序边，warning 标注“未识别跳连”。
   - 仍失败（实例化/forward 报错）→ 静态 AST 回退（mode="static"）。
4. 记录每个 FX node 的输入/输出形状：自定义 `torch.fx.Interpreter`（`run_node` 里 `fetch_args_kwargs_from_env`
   后取 shapes，再 `super().run_node`），`torch.no_grad()` + `model.eval()` + CPU。
5. 图节点：input placeholder → `kind:"input"`；`call_module` → 模块节点；`call_function`/`call_method`
   中的有效算子（flatten/view/reshape/permute/add/cat/relu/softmax/dropout 等）→ 节点；`get_attr`、
   常量、`getitem` 跳过（必要时透传依赖）。output → `kind:"output"`。
6. 连边：递归解析每个 node 的 args/kwargs 中引用的上游 FX node，建立边；`kind` 按算子与拓扑判断
   （两输入 add 且一支来自更早层 → `residual`；cat → `skip`；attention 内部 → `attention`）。
7. 命名：`call_module` 用 qualified name（如 `features.0`）；函数节点用 `F.relu` / `torch.flatten` 风格。
8. 记录模块真实参数量；FLOPs 按 4.5 公式（trace 的有形状 → 较准确，`flopsApprox: false`，
   但注意力仍标记估算）。
9. 对同一模块多次调用：每次调用一个 node，名字加 `#2` 后缀。

### 4.4 静态 AST（mode="static"，安全模式 / 回退）

不执行代码。解析 `__init__` 与 `forward`：
- `self.x = nn.Conv2d(3, 64, 3, padding=1)`；支持 `nn.Sequential(...)`、`nn.ModuleList([...])` 递归；
  支持构造参数为字面量/int 常量/`__init__` 默认参数/类属性/简单四则运算。
- forward 支持的典型模式：
  - `x = self.conv1(x)`、链式 `F.relu(self.bn(self.conv(x)))`
  - `x = self.a(x) + self.b(x)`、`x += ...` → residual
  - `x = torch.cat([x, y], dim=1)` / `torch.cat((x, y), 1)` → concat
  - `x = x.flatten(1)` / `torch.flatten(x, 1)` / `x.view(x.size(0), -1)` → flatten
  - `for blk in self.blocks: x = blk(x)` → ModuleList 顺序展开
  - `x = self.encoder(x)`（自定义子模块）→ 若子模块能解析则内联，否则作为 `block/custom` 节点
  - 辅助方法 `def _forward_impl(self, x)`：若 forward 调用它，跟进解析
- 形状推导：conv/pool/linear/flatten/embedding/norm/attention 公式；未知维度用 `-1`，加 warning。
- 静态参数量：按公式精确计算；FLOPs 按公式估算，`flopsApprox: true`。
- 无法静态确定的边/形状：保留节点，warning 说明，不崩溃。
- 源码位置：尽量给 `line` 与 `code`（`ast.get_source_segment`）。

### 4.5 算子识别、参数量、FLOPs

- op → kind 规则表：后缀/包含匹配，覆盖 ResNet / VGG / ViT / Transformer / UNet / GAN / RNN 常用层：
  - `Conv1d/2d/3d`、`ConvTranspose*`、`Linear`、`Bilinear`
  - `BatchNorm*, LayerNorm, GroupNorm, InstanceNorm*, RMSNorm`（自定义 RMSNorm 也认）
  - `ReLU, ReLU6, GELU, SiLU/Swish, LeakyReLU, Sigmoid, Tanh, Softmax, Hardswish, Mish, ELU, PReLU`
  - `MaxPool*, AvgPool*, AdaptiveAvgPool*, AdaptiveMaxPool*, MaxUnpool*`
  - `Dropout*, DropPath, StochasticDepth`
  - `Embedding, EmbeddingBag, PositionalEncoding`（含类名）
  - `MultiheadAttention, *Attention, SelfAttention, CrossAttention`
  - `TransformerEncoderLayer, TransformerDecoderLayer, TransformerEncoder, TransformerDecoder`
  - `LSTM, GRU, RNN, LSTMCell...`
  - `Flatten, Unflatten, Upsample, PixelShuffle, Identity`
  - 自定义类名：含 `Attention` → attention/mha；含 `FFN|FeedForward|MLP` → ffn/mlp；
    含 `Block|ResBlock|Bottleneck|BasicBlock` → block；含 `Encoder|Decoder` → block/encoder。
- 参数量：
  - Conv: `out*(in/groups)*kH*kW + (bias?out:0)`
  - ConvTranspose 同理（按权重形状）
  - Linear: `in*out + (bias?out:0)`
  - Norm: `2*C`（affine）/ `C`（仅 running stats）；Embedding: `num*vocab`
  - 动态模式直接用 `sum(p.numel())` 与 `p.requires_grad` 统计。
- FLOPs（约定 **1 MAC = 2 FLOPs**，bias 加法计入）：
  - Conv: `2*N*Hout*Wout*Cout*(Cin/groups)*kH*kW + N*Cout*Hout*Wout`（bias 时）
  - Linear: `2*N*in*out + N*out`（bias 时）
  - Attention（MHA）：`4*N*L*d^2 + 2*N*L^2*d`（QKVO 投影 + 注意力），按 batch_first 处理，估算
  - norm/activation/pool/softmax: 按其输出元素数近似 1–4 FLOPs/elem（可给 0，说明只统计主要算子）
- 总参数量/FLOPs 与 PyTorch 对照，允许 FLOPs 有 <1% 偏差；测试要覆盖。

### 4.6 分组（高层组件识别）

- 由 module 层级（`features.0`、`encoder.layers.0.self_attn`）建立自然分组；
- 识别人工组件：
  - 父模块下有 ≥2 个 conv + 至少一个 norm/activation → `conv_block`
  - 父模块内出现残差 add/cat（trace 边或 forward AST）→ `residual_block`（或含 bottleneck → 仍 residual_block）
  - 父模块含 attention + 2 个 linear + ≥1 norm → `transformer_block`
  - 父模块含 attention 但无 FFN → `attention_block`
  - 父模块含连续 linear（≥2）→ `mlp` / `ffn_block`
  - 名字含 encoder/decoder → `encoder` / `decoder`；UNet 的 down/up + cat → `unet_block`
  - 其余容器 → `sequential` / `module`
- `label` 人类可读：`layer1.0` → `Residual Block 1`；`encoder.layers.2` → `Transformer Block 3`；
  `features` → `Features`。前端会直接显示 label。
- 分组必须**不重叠**（一个 node 只属于一个直接 group）、层级自洽、node/child group 都属于本组。

### 4.7 演示数据

解析 `samples/` 生成 `examples/demo/`：
- `index.json` + `<id>.json`；
- 至少包含 `cnn_vgg`、`resnet_mini`、`transformer_classifier`、`custom_attention`、`unet_mini`；
- 生成脚本 `scripts/build_demo.py`，能被 Lead 一键重跑；
- 必须全部通过 schema 校验；UI 打开 demo 不能报错。

---

## 5. Web 前端（frontend-dev）

### 5.1 技术约束

- 原生 ES Modules + HTML/CSS，**无打包步骤**；`python -m netprism serve` 直接静态托管 `web/`。
- Three.js 用 npm 安装后把所需文件复制进 `web/vendor/three/`（`three.module.js`、`OrbitControls.js`、
  `CSS2DRenderer.js` 等），`index.html` 用 importmap 指向本地；**禁止 CDN / 外网请求**。
- 不使用 React/Vue；不引入构建链。现代 Chrome/Edge 即可。
- 入口：`web/index.html`，资源相对路径；由 `/` 直接打开。

### 5.2 信息架构

- 顶栏：NetPrism logo / 模型名 / 模式徽章 / 模型切换 / “上传文件夹” / 主题（暗色默认）。
- 左栏：模型摘要卡（参数量、FLOPs、层数、输入输出 shape、warning）+ 搜索 + 层列表树（按 group 折叠）。
- 中区：3D 画布 + 悬浮工具提示 + 底部控制条（播放、单步、重置视角、显示切换）。
- 右栏：选中层详情（op/kind/shape/params/flops/source code/notes）+ 图例过滤。
- 空状态：拖拽/选择文件夹 + 示例模型按钮 + 说明“静态解析，不执行代码”。

### 5.3 3D 视觉与映射（硬性要求）

- **布局**：执行顺序沿 X 轴；节点是代表输出张量的半透明立方体；
  - 4D CNN 张量 `(N,C,H,W)`：Y=H、Z=W（按最大边归一化）、X 厚度=log(C) 映射；
  - 2D `(N,F)`：Y 固定、Z=log(F) 映射（细长板）；
  - 3D 序列 `(N,L,D)`：Y=L、Z=log(D)、X 厚度薄；token/head 有视觉暗示；
  - 1D 序列：细长条；标量：小方块。
  - 同组节点间距更近，组间更远；组用半透明线框包围 + 组名标签。
- **卷积核**：conv 节点在输入侧显示 `kH×kW` 网格小块，深度反映 `Cin/groups`（取 log），
  悬停显示 kernel/stride/padding/groups/dilation；可在设置里关闭。
- **attention**：显示为多 head 的层叠薄片，head 数来自 `params.numHeads`；FFN 用“先扩张后收缩”的
  两段板表示；residual/skip 边用弧形高亮线。
- **连线**：从源节点出口到目标入口的曲线；`data` 灰白、`residual/skip` 橙/紫、`attention` 洋红；
  悬停高亮并显示张量形状；选中节点时上下游高亮、其余降透明度。
- **材质/灯光**：深色背景 + 地面网格 + 柔和阴影/环境光；不要默认灰盒；色彩按 kind 统一（见图例）。
- **标签**：默认只显示组标签和关键节点标签，避免糊成一片；hover/选中显示完整；可一键全部显示。
- **动画**：入场（节点依次浮起）、数据流光点、选中聚焦缓动；`prefers-reduced-motion` 时降级。
- 大模型（>250 节点）自动简化：隐藏部分标签、降低几何段数，并提示。

### 5.4 数据与 API

- 启动：`GET /api/health` → `GET /api/index` → 默认加载第一个模型 `GET /api/model/<id>`；
  失败时显示错误/空状态（demo 模式见下）。
- 上传：`<input type="file" webkitdirectory>` + 拖拽文件夹，收集 `.py` 文本，POST `/api/upload-parse`：
  ```json
  {"files": [{"path": "sub/model.py", "content": "..."}], "execute": false}
  ```
  响应：`{"index": {...}, "graphs": {"<id>": {...}}, "errors": [...]}`；前端直接用返回的 graphs 渲染。
- 路径 `#model=<id>` 可分享当前模型（可选）。
- 若 `/api/index` 返回 404，前端尝试 `web/demo/index.json` + `web/demo/<id>.json`（静态 demo 兜底）。

### 5.5 代码结构（建议）

`js/main.js`（入口/状态）、`js/api.js`、`js/scene.js`（Three 场景/相机/光照）、`js/layout.js`（3D 布局）、
`js/nodes.js`（立方体/卷积核/组框/标签）、`js/edges.js`、`js/interaction.js`（raycaster/hover/select）、
`js/ui.js`（侧栏/列表/搜索/图例/详情）、`js/format.js`（数字/形状/参数格式化）。可合并，但保持可读。
所有面向用户的文案中文；README（`web/README.md`）列出快捷键与交互。

### 5.6 验收

- `python -m netprism serve examples/demo` 打开后 3 个模型可切换，全部正常渲染。
- 用 `samples/` 真实解析结果验证 CNN / Transformer / UNet 三种视觉。
- 拖拽上传 `samples` 文件夹能解析并显示。
- Chrome headless 截图无白屏、无 console error（Lead 会做最终视觉验收）。
- 窄屏 1280×800 下侧栏不遮挡画布，可折叠。

---

## 6. CLI / Server（Lead）

- `scan`：调用 `analyze_folder`，写 index + 每个 graph；终端打印表格（模型、模式、层数、参数、FLOPs、warning）。
- `inspect`：单文件解析，终端树状打印（group → node + shapes + params），`--json` 输出原始 graph。
- `serve`：标准库 `ThreadingHTTPServer`，安全静态托管 web/ + API；`--host 127.0.0.1 --port 8765 --open`；
  默认目录 `examples/demo`。
- `view`：`scan` 到 `--out`（默认 `./netprism_out`）后 `serve` + 打开浏览器。
- API 细节：
  - `GET /api/health` → `{"ok": true, "version": "0.1.0"}`
  - `GET /api/index` → index.json
  - `GET /api/model/<id>` → graph（id 白名单，防目录穿越）
  - `POST /api/upload-parse`（JSON，总计 ≤ 8MB，单文件 ≤ 1.5MB，仅 `.py`；临时目录；默认 `execute=false`；
    若服务以 `--allow-execute` 启动才允许 `execute=true`）
  - `POST /api/scan` `{"path": "...", "execute": false}`（仅本机；目录须存在）
- 所有响应 `Cache-Control: no-store`；错误 JSON `{"error": "..."}` + 4xx/5xx。
- 静态文件路径必须 resolve 后校验在 web 根内。

---

## 7. 样例模型（samples-qa）

每个文件都能在 CPU 上快速实例化/forward（除故意失败的），无外部依赖（只 torch/numpy）：

| 文件 | 类 | 覆盖点 |
|---|---|---|
| `cnn_vgg.py` | `VGGMini` | Sequential、多层 Conv/BN/ReLU/MaxPool/AdaptiveAvgPool/Linear、kernel 3×3/1×1 |
| `resnet_mini.py` | `ResNetMini` | 自定义 `BasicBlock`、残差 add、stride、1×1 投影 shortcut、Layer1/2 |
| `transformer_classifier.py` | `TinyTransformerClassifier` | `nn.Embedding`、位置编码 buffer、`nn.TransformerEncoderLayer`、MHA、FFN、LayerNorm、mean pool、分类头 |
| `custom_attention.py` | `TinyGPT` | 手写 `SelfAttention`（qkv Linear + matmul + softmax + scale）、`FeedForward`、残差、LayerNorm、causal mask |
| `unet_mini.py` | `UNetMini` | Encoder/Decoder、MaxPool、ConvTranspose2d、`torch.cat` skip、双输入 concat |
| `mlp.py` | `MLP` | 纯 Linear/激活/Dropout |
| `needs_args.py` | `NeedsArgs` | `__init__(self, width)` 必填参数 → 只能静态解析 |
| `dynamic_control.py` | `DynamicNet` | forward 里数据相关 `if` → FX trace 失败、hook 回退 |
| `not_a_model.py` | — | 有 nn.Module 但没有 forward / 普通工具函数，应被跳过 |

要求：默认构造（`Cls()`）可工作；forward 输入固定为 `(1,3,32,32)`（transformer/GPT 为 `(1,16)` int64 或按注释），
输出 shape 在文件顶部注释注明；每个文件 < 100 行；风格像真实开源代码，不要写成测试桩。
samples-qa 同时写 `samples/README.md`（每个模型一句话 + 输入/输出）+ `tests/test_samples.py`
（实例化、forward、参数/形状断言）。

---

## 8. 测试与验收

- 统一用 bundled python：`& $py -m pytest tests -q`。
- `tests/test_core.py`（backend-dev）：schema 校验、形状/参数量/FLOPs、残差边、注意力识别、分组、静态回退、
  文件夹扫描、错误隔离。
- `tests/test_samples.py`（samples-qa）：样例自身正确性。
- `tests/test_cli.py`（samples-qa）：`scan` 产出 index/文件、`inspect --json`、server API（起本地随机端口）：
  health/index/model 404/路径穿越/上传静态解析。
- 前端（frontend-dev）：`node --check` 每个 js；用 headless Chrome 至少截一张图（脚本 `web/scripts/screenshot.mjs` 或说明）。
- 最终 Lead 验收：干净环境跑 `scan samples` → schema 全绿 → `serve` → 截图 → README 步骤复现。

**Definition of Done**：三条线各自测试通过 + 集成 `view samples` 端到端可用 + README 可复现 + 无 console 报错。

---

## 9. 分工、任务与协作

- **Lead**：SPEC/schema/CLI/server/文档/集成/最终验收；合代码、修跨模块问题。
- **backend-dev**：`netprism/core/**`、`tests/test_core.py`、`scripts/build_demo.py`、生成 `examples/demo/`。
- **frontend-dev**：`web/**`（含 vendor）、`web/README.md`。
- **samples-qa**：`samples/**`、`tests/test_samples.py`、`tests/test_cli.py`、`samples/README.md`。
- 共享约定：新文件先看 SPEC；跨模块疑问在消息里问 Lead；不要跑 `git` 全局操作；不要删别人的文件；
  测试数据放 `scratch/<name>/` 或临时目录。
- 进度：完成任务后在共享任务板 `complete`，并给 Lead 发一条**简短**报告（做了什么、验证命令/结果、遗留问题）。

---

## 10. 编码规范

- Python 3.12 语法可用（`X | None`、`list[str]`、match），所有文件 `from __future__ import annotations`。
- 后端函数小而可测；异常包装为自定义 `AnalyzeError`（含 file/line/message），不吞掉 traceback（verbose 时可打印）。
- 日志用 `logging`（`netprism.core`），不要在库里 print；CLI 才 print。
- 生成的 JSON 稳定排序（节点按执行序），便于测试 diff。
- 数值格式化在前端做（1.2M / 3.4G）；后端只给整数。
- 不提交大文件（vendor three 除外）；`.gitignore` 忽略 `__pycache__`、`netprism_out/`、`.pytest_cache/`。
