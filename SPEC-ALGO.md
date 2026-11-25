# 算法理解层（v2.1）

目标：不只列层，而是识别**算法结构**。输出必须可被 CLI/报告/UI 直接消费；不写空泛描述。

## 1. 模块 API

```python
# netprism/algorithms.py
def enrich_algorithms(graph: dict, *, force: bool = False) -> dict
    """识别模型家族、组件模式、forward 伪代码，写入 model/group/node.meta。"""

def node_formula(node: dict) -> dict
    """{"formula": str, "semantic": str, "axes": str|None, "tags": [str]}"""

def explain_graph(graph: dict) -> dict
    """{"family","familyLabel","tags","patterns","pseudocode","groups"}；纯 JSON。"""

def render_explain_md(graph: dict) -> str
    """算法理解 Markdown 片段（给 report/CLI 复用）。"""
```

- 纯 stdlib；不执行模型；不改 nodes/edges/params/metrics。
- 幂等；已有 `model.meta.family` 且 force=False 时保留。

## 2. 写入字段（可选 meta，schema 已支持）

```json
"model": {
  "meta": {
    "family": "resnet",
    "familyLabel": "ResNet",
    "algorithmTags": ["cnn", "residual", "conv_bn_relu"],
    "patterns": [
      {"id": "p1", "kind": "residual_block", "label": "Residual Block",
       "nodeIds": ["n1"], "groupIds": ["g3"], "formula": "y = F(x) + x",
       "confidence": 0.95, "evidence": ["add 节点有跨层 skip 边"]}
    ],
    "pseudocode": ["x = Input(1,3,32,32)", "x = ResidualBlock(x)", "..."]
  }
}
```
```json
"group": {"meta": {"pattern": {"kind": "residual_block", "label": "...",
  "formula": "y = F(x) + x", "confidence": 0.9, "tags": ["residual"]}}}
"node": {"meta": {"formula": "y = ReLU(x)", "semantic": "逐元素非线性",
  "axes": null, "tags": ["activation"]}}
```

## 3. 必须识别的模式

| kind | 判定要点 | formula |
|---|---|---|
| `conv_bn_act` | conv→norm→activation 链 | `y = Act(BN(Conv(x)))` |
| `residual_block` | add/concat 合并跨层分支，且组内有 conv | `y = F(x) + x` |
| `bottleneck` | 1×1 → 3×3 → 1×1 + shortcut | `y = x + W3(W2(W1(x)))` |
| `depthwise_separable` | depthwise conv (groups=Cin) + 1×1 conv | `y = PW(DW(x))` |
| `inverted_residual` | 1×1 扩张 → depthwise → 1×1 投影 + residual | MobileNetV2 block |
| `squeeze_excitation` | global pool → FC↓ → act → FC↑ → sigmoid → multiply | `y = x · σ(W2δ(W1·GAP(x)))` |
| `inception` | 同一输入多分支不同 kernel，最后 concat | `y = cat(B1(x), B2(x), ...)` |
| `dense_connection` | 多层 concat 复用前序特征 | `x_l = H_l([x_0,...,x_{l-1}])` |
| `unet_skip` | encoder 特征 concat/add 到 decoder | `y = Dec(cat(x_enc, up(x)))` |
| `transformer_block` | MHA + FFN + norm + residual | `x = x + MHA(LN(x)); x = x + FFN(LN(x))` |
| `self_attention` | MHA 的 Q/K/V 同源 | `softmax(QKᵀ/√d)V` |
| `cross_attention` | Q 与 K/V 不同源 | 同上，K/V 来自另一分支 |
| `ffn` | Linear 扩张 → 激活 → Linear 收缩 | `W2·Act(W1·x)` |
| `gated_mlp` | 两线性分支做 sigmoid/乘 | `y = W2(Act(W1x) ⊙ σ(W3x))` |
| `pre_norm` / `post_norm` | norm 在 attention/FFN 前/后 | Pre-LN / Post-LN |
| `weight_tying` | 不同节点共享 modulePath/参数 | 同一权重复用 |
| `rnn_block` | LSTM/GRU/RNN | 序列状态更新 |

模型家族：`cnn, vgg, resnet, densenet, mobilenet, mobilenet_v2, inception, unet,
transformer_encoder, transformer_decoder, vit, mlp, rnn, hybrid, unknown`。
分类以结构证据优先，类名/模型名仅作辅助，给 confidence。

## 4. 节点公式（至少覆盖）

Conv/DepthwiseConv/ConvTranspose、Linear、BatchNorm/LayerNorm、ReLU/GELU/SiLU/Softmax、
MaxPool/AvgPool/GlobalPool、Embedding、MultiHeadAttention、Add/Concat、Dropout、Flatten/Reshape、
LSTM/GRU。公式用 ASCII/Unicode 简洁表达，带轴说明（NCHW/NHWC/NLC）与关键超参。
不写“该层很重要”之类空话。

## 5. 伪代码

按执行顺序与 group 聚合，输出可读行，含输入输出形状注释；示例：

```
x = Input(1,3,32,32)
x = Conv2d(16,3x3,s1,p1)(x)   # -> (1,16,32,32)
x = ResidualBlock(16)(x)      # -> (1,16,32,32)
x = GlobalAvgPool()(x)
logits = Linear(10)(x)
```

- 顶层 group 优先折叠为 block 调用；游离节点按顺序。
- 每行可附 shape；不编造未知 shape（用 `?`）。

## 6. CLI / 报告 / UI

```powershell
netprism explain <graph.json|scan_dir|model_file> [--model ID] [--json]
```

- 默认输出：家族、关键模式（含公式）、forward 伪代码。
- `report` 的 md/html 增加「算法理解」小节（Lead 集成）。
- Web 新增「算法」面板/tab：家族徽章、模式卡（label/formula/confidence）、伪代码块；
  层详情显示 node.meta.formula/semantic；不改现有 3D/表格功能。
- 新增 DOM id：`#algo-toggle`、`#algo-view`、`#algo-family`、`#algo-patterns`、`#algo-pseudocode`。

## 7. 测试

- 对 samples 的 VGG/ResNet/UNet/TinyGPT/Transformer 断言 family、patterns、公式、伪代码；
- 构造 depthwise/inverted residual/SE/Inception/Dense/UNet/self/cross attention 小图；
- 幂等、无 meta 兼容、未知图不崩、纯 JSON、不依赖 torch/onnx；
- report/CLI/UI 集成由 QA 覆盖。

## 8. 分工

- algo-dev：`netprism/algorithms.py`、`tests/test_algorithms.py`、`tests/fixtures/algorithm_patterns.json`（可选）。
- Lead：schema、CLI `explain`、report 集成、README/docs、EXE。
- frontend-dev：`web/**` 算法面板/公式展示。
