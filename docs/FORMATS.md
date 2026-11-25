# 多格式与分析

## 格式

| 输入 | 适配器 | 结构 | 形状 | 参数/FLOPs |
|---|---|---|---|---|
| `.py` PyTorch | core | trace / static | 真实 / AST 推导 | 真实 / 推导 |
| `.onnx` | `adapters/onnx_adapter.py` | ONNX graph + shape inference | 真实（缺失为 -1） | initializer + op 公式 |
| `.pt/.pth` TorchScript | `adapters/torchscript_adapter.py` | `torch.jit.load` + 执行记录 | 解析式传播 | state_dict 精确 |
| `.keras` Keras v3 | `adapters/keras_adapter.py` | zip 内 `config.json` | 配置传播 | 配置推导 |

```powershell
python -m netprism scan .\models --out .\out          # 自动混扫
python -m netprism scan .\models --no-formats          # 只扫 .py
python -m netprism inspect .\model.onnx
python -m netprism view .\model.onnx --open            # 单文件直接 3D
python -m netprism serve examples\multiformat --open    # ONNX/TS/Keras demo
```

## 指标

- `params`：PyTorch `sum(p.numel())`；ONNX initializer；TorchScript state_dict；Keras 由层配置和输入形状推导。
- `FLOPs`：1 MAC = 2 FLOPs；Conv/Linear/MatMul 精确到形状，BN/Pool/激活按元素数估算；`macs = flops / 2`。
- `activationBytes`：每节点首个输出张量的元素数 × 4（float32）；`peakActivationBytes` 用 last-use 活跃集估算。
- `receptiveField`：沿拓扑传播，`rf += (k-1)*d*acc_stride`，`acc_stride *= stride`；分支取 max；未识别算子保守不变。
- 以上除参数量外均为**估算**，报告和 UI 标注「估算」。

## 报告 / 对比

```powershell
python -m netprism report .\out --model resnet18 --format md   --out report.md
python -m netprism report .\model.onnx            --format html --out report.html
python -m netprism compare .\out .\out --model-a resnet18 --model-b resnet34 --format json
```

- `report`：summary + group 表 + layer 表；md/csv/html。CSV 带 UTF-8 BOM、首行 ASCII 摘要注释。
- `compare`：按 `modulePath`（空回退 name）匹配；added/removed/changed；md/json/csv。
- Web：`T` 打开层表格，排序/搜索/导出 CSV；`#framework-badge` 显示格式；`#metric-activation`、`#metric-rf`。

## 限制

- ONNX 默认常量折叠会把 BN 折进 Conv，节点/参数与训练代码不同；需要逐位一致时导出用 `do_constant_folding=False`。
- TorchScript 不保存算子级 shape，逐层形状由输入与 op 常量解析传播；Transformer 动态 view 可能为 -1。
- Keras 不加载权重、不执行模型；嵌套子模型/自定义层保留为 custom；group 仅根组。
- 格式文件目前通过 CLI / 文件夹扫描解析；Web 上传仅支持 `.py`（二进制格式请用 `view`/`scan`）。

## 契约

graph 的可选字段与适配器 API 见 [`SPEC-V2.md`](../SPEC-V2.md)。
