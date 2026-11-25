# NetPrism v2 契约（简洁版）

目标：让科研工作者用一个工具完成多格式模型解析、结构检查、复杂度分析、对比与报告。
v2 只做加法；现有 PyTorch/CLI/UI/EXE 行为保持兼容。

## 1. 新增格式

| 后缀 | 格式 | 依赖 |
|---|---|---|
| `.onnx` | ONNX 模型（`onnx` 包，shape inference） | 运行时可选；EXE 内置 |
| `.pt` / `.pth` | TorchScript archive（`torch.jit.load`；普通 state_dict 报可读错误） | torch |
| `.keras` | Keras v3 模型（zip 内 `config.json`，解析 `Sequential`/`Functional`，不依赖 TensorFlow） | 无（标准库 zip/json） |

`scan`/`view` 自动发现文件夹中的这些文件；`inspect` 支持单文件；Web 端仍以服务端预扫描结果为主。

## 2. graph 增量字段（schema 1.0 向后兼容）

新增可选字段：
- `model.framework`: 扩展 `keras` / `torchscript`
- `model.mode`: 扩展 `keras` / `torchscript`
- `model.meta`: object，格式元信息（opset、keras version、torch version、source format 等）
- `model.metrics`: object：
  ```json
  {"paramBytes": 0, "activationBytes": 0, "peakActivationBytes": 0,
   "maxReceptiveField": 1, "macs": 0}
  ```
- `node.metrics`: object：
  ```json
  {"activationBytes": 0, "receptiveField": 1, "macs": 0}
  ```
- `node.meta`: object，格式私有信息（如 ONNX domain/op_type）。

`activationBytes` 按输出张量 float32 估算；`peakActivationBytes` 为可选峰值估算；
`receptiveField` 对 CNN/池化传播，分支取最大；`macs = flops / 2`。
所有估算值必须在 UI/报告中标注“估算”，不得伪造精确值。

## 3. 适配器 API

```python
# netprism/adapters/__init__.py
FORMAT_SUFFIXES: dict[str, str] = {".onnx": "onnx", ".pt": "torchscript", ...}

def is_format_file(path) -> bool
def analyze_format_file(path, *, max_nodes: int = 20000) -> list[dict]
    """返回 1 个 graph（少数格式可多个）；失败抛 AnalyzeError；输出必须过 validate_graph。"""
```

- 不修改 `netprism/core/**`。
- 顶层不得强依赖 onnx/torch；惰性 import，缺依赖时抛 `AnalyzeError` 带安装提示。
- graph 必须含 `inputSpec`/`outputSpec`、节点 `inShapes`/`outShapes`、`numParams`、`flops`、`groups` 根组。
- 节点 id `n*`，边 `e*`，组 `g*`；source 可为 null（无源码时）。
- ONNX 形状来自 `onnx.shape_inference.infer_shapes`；缺失维度用 `-1`。
- Keras 参数/FLOPs 从层配置与输入形状推导，`flopsApprox=true`；无权重时不计算 weight bytes。
- TorchScript 先 `torch.jit.load`；用 hook 或 aten 图记录节点与形状；无法得到逐层形状时用 `-1` 并 warning。

## 4. 分析 / 报告 / 对比 API

```python
# netprism/metrics.py
def enrich_graph(graph: dict) -> dict          # 计算并写入 metrics，返回同一对象
def layer_rows(graph: dict) -> list[dict]      # 报告/表格共用行

# netprism/report.py
def render_report(graph: dict, fmt: str = "md") -> str   # md | csv | html
def write_report(graph: dict, out: Path, fmt: str) -> None

# netprism/compare.py
def compare_graphs(a: dict, b: dict) -> dict
def render_compare(result: dict, fmt: str = "md") -> str  # md | json | csv
```

报告内容：model summary（格式/模式/params/FLOPs/activation/RF/warnings）、group 汇总、
layer 表（id/name/op/kind/in/out/params/FLOPs/activation/RF/source）、可选 code block。
对比：按 `modulePath`（fallback name）匹配；分类 added/removed/changed（op/shape/params）；
summary 给两次 params/FLOPs/activation 与差值；不得崩溃。

## 5. CLI 增量

```powershell
netprism scan <folder>                # 自动解析 .py + .onnx/.pt/.keras
netprism inspect <file>               # 支持上述格式
netprism report <graph.json|scan_dir> [--model ID] [--format md|csv|html] [--out PATH]
netprism compare <a> <b> [--format md|json|csv] [--out PATH]
```

- `report`/`compare` 的 source 支持：graph JSON、含 index.json 的 scan 目录、模型文件。
- `report` 默认 md 输出到 stdout；`--out` 写文件。
- `compare` 默认 md；JSON 便于科研脚本二次处理。
- `scan --all-variants` 等现有参数不变。

## 6. Web UI 增量

- 摘要/详情显示 framework 徽章、activation 显存、感受野；标注“估算”。
- 新增「表格」视图：可排序 layer 表（name/op/kind/shape/params/FLOPs/activation/RF），从 graph 客户端生成 CSV 下载。
- 不改现有 3D 交互与快捷键；新格式加载走 `/api/model/<id>`。
- 大模型（>250 节点）表格仍需可用，导出不受影响。

## 7. 分工 / 写作用域

| 角色 | 写作用域 |
|---|---|
| Lead | `netprism/schema.py`、`schema/`、`netprism/cli.py`、`netprism/server.py`、`netprism/selection.py`、`README.md`、`docs/`、`packaging/`、`dist/`、`pyproject.toml` |
| formats-dev | `netprism/adapters/**`、`tests/test_adapters.py`、`tests/fixtures/formats/**`、`scripts/make_format_fixtures.py` |
| analysis-dev | `netprism/metrics.py`、`netprism/report.py`、`netprism/compare.py`、`tests/test_analysis.py` |
| frontend-dev | `web/**` |
| qa-dev | `tests/test_v2_integration.py`、`tests/test_ui_v2.py`、`scratch/qa2/**` |

共享文件只读；跨模块字段以本文件 + `schema.py` 为准。完成标准：`pytest tests -q` 全绿；
`scripts/verify.py samples` 通过；真实 ONNX/Keras/TorchScript 文件可解析；`report`/`compare` 有测试；
UI Playwright 0 error；EXE 后续重建能加载新格式。
