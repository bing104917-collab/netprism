# NetPrism 架构

## 1. 目标与边界

**目标**：输入一个包含 PyTorch 源码的文件夹，输出一份可交互 3D 可视化的网络结构图；覆盖典型 CNN、
残差网络、Transformer/FFN、U-Net 等真实结构，并且对 AI 生成/风格不统一的代码有容错。

**非目标**（v1）：不训练、不推理业务数据、不做性能剖析、不替代 TensorBoard；不追求完整支持所有框架。
数据契约与解析层刻意解耦，后续可加 ONNX/Keras adapter。

## 2. 总览

```
                        ┌─────────────────────────────┐
  model folder  ──────► │  netprism.core               │
  (*.py)                │  discover → trace/static     │
                        │  → ops (shapes/params/FLOPs) │
                        │  → groups → graph dicts      │
                        └──────────────┬──────────────┘
                                       │ graph JSON v1.0 (schema.py)
                                       ▼
  netprism scan  ──────────────►  out/index.json + out/<id>.json
                                       │
  netprism serve / view  ─────►  netprism.server (stdlib HTTP)
                                       │  /api/index, /api/model/<id>,
                                       │  /api/upload-parse, /api/scan
                                       ▼
                        ┌─────────────────────────────┐
                        │  web/ (ES modules + Three)   │
                        │  scene / layout / edges /    │
                        │  interaction / sidebar       │
                        └─────────────────────────────┘
```

## 3. 解析流水线

1. **发现（discover）**：递归遍历 `*.py`，跳过虚拟环境、`node_modules`、缓存目录与超大文件；
   AST 找 `nn.Module` 子类（要求有 `forward`）与**模块级工厂函数**（`ResNet18()`），用引用图判断
   真正的顶层模型——被其他类作为 block 实例化的类、被跨文件 import 当子模块的类都会被降级为组件。
2. **实例化与输入推断（trace 分支）**：按包上下文导入（支持 `from .unet_parts import *`），
   尝试类、工厂、`build_model()/get_model()`；参数不足时用 dataclass 默认值 / 调用点字面量 /
   常见参数名启发（`n_channels=3`、`n_classes=10`…）合成并写 warning；根据首层与 patch/pos-embedding
   推断 dummy 输入（Embedding→token ids，Conv2d→`1×C×64×64`，ViT→按 patch_size 推断 256/32）。
3. **图捕获**：自定义 `torch.fx.Tracer` 选择性展开复合模块，`torch.fx.Interpreter` 在真实执行时记录
   每个节点的输入/输出张量形状；从参数的 FX Node 引用恢复连边（含残差与 U-Net skip）。
4. **静态分支**：如果 trace 失败或显式 `--static`，用 AST 解析 `__init__`/`forward`，按算子的形状公式
   推导；支持链式调用、残差 add、`torch.cat`、flatten/view、`ModuleList` 循环等。
5. **算子语义**：`ops.py` 把类名/函数名映射为统一 `kind`，提取超参，计算参数量与 FLOPs。
6. **组件分组**：按模块路径（`encoder.layers.0.self_attn`）与结构模式（attention+FFN+norm、conv+norm+add）
   建立 group 层级，生成人类可读 label。
7. **契约校验**：`schema.validate_graph` 校验字段、ID 唯一性、边端点、分组归属与层级环；
   `scan` 写出前再看一次。
8. **服务**：`server.py` 静态托管 `web/`，把 graph JSON 直接给前端；上传的文件夹先落临时目录再做静态解析。

## 4. 模块职责

| 模块 | 职责 |
|---|---|
| `netprism/schema.py` | 冻结的字段常量、无依赖语义校验、Index 校验；同时给出 JSON Schema 路径 |
| `netprism/core/discover.py` | 文件遍历、AST 类/工厂扫描、跨文件组件引用图、顶层模型判定、包上下文导入 |
| `netprism/core/synth.py` | dataclass config、调用点字面量、常见构造参数合成 + warning |
| `netprism/core/tracer.py` | FX Tracer / Interpreter、dummy 输入、包内相对导入、hook 回退 |
| `netprism/core/astscan.py` | `__init__` 层定义解析、`forward` 数据流解析、静态形状推导 |
| `netprism/core/ops.py` | op→kind 规则、超参提取、形状公式、参数量/FLOPs |
| `netprism/core/groups.py` | 自然分组 + 组件模式识别 + label |
| `netprism/core/build.py` | 统一节点/边/组结构、编号、汇总、schema 合法性 |
| `netprism/cli.py` | `scan` / `inspect` / `serve` / `view` 参数与终端输出；默认折叠同类工厂变体 |
| `netprism/selection.py` | 展示层变体折叠（`ResNet18/34/50` → 默认一个，`--all-variants` 全显示） |
| `netprism/server.py` | HTTP、静态文件、API、上传安全边界；同样对 API 结果做变体折叠 |
| `web/` | Three.js 场景、布局、边、交互、侧栏、数据加载 |

## 5. 数据契约

唯一真相是 [`schema/graph.schema.json`](../schema/graph.schema.json) + [`netprism/schema.py`](../netprism/schema.py)。

- `model`：模型元信息、输入/输出 spec、总参数量/FLOPs、模式与 warning。
- `nodes`：执行顺序的叶子层；形状用 `[[...]]` 表示多张量输入/输出；`params` 存算子超参；
  `source` 尽量带文件和行号。
- `edges`：`data / skip / residual / attention / control`；前端根据 kind 决定弧度与颜色。
- `groups`：组件层级（parent/children + 直接成员），前端据此画 3D 包围框与层列表树。

契约只增不改；新增字段采用可选字段并更新 schema 版本。`schemaVersion` 目前为 `1.0`。

## 6. 服务端与安全

- 纯标准库 `ThreadingHTTPServer`，默认只绑定 `127.0.0.1`。
- 静态文件路径 `resolve()` 后必须位于 `web/` 内；模型 id 用白名单正则，磁盘图文件必须位于数据目录内。
- 上传接口限制：仅 `.py`、总 ≤ 8 MB、单文件 ≤ 1.5 MB、≤ 300 个；路径分段去掉 `..`；临时目录解析后即删。
- 上传/远程扫描默认 `execute=false`（静态解析）；服务需要 `--allow-execute` 才允许执行上传代码。
- `/api/scan` 只接受本机请求；`trace` 模式仍然是“执行任意 Python”，文档与 UI 都明确警示。

## 7. 前端架构

- **无构建**：`index.html` 用 importmap 指向 `web/vendor/three/`；所有模块原生 ESM。
- **数据流**：`api.js` 负责 `/api/index`、`/api/model/<id>`、上传；失败时回退到 `web/demo/` 静态数据。
- **场景**：`scene.js` 管 renderer/camera/controls/lighting；`layout.js` 计算每个节点的位置与尺寸
  （X 轴执行顺序，Y/Z 为张量空间维度，组内紧凑、组间留白）；`nodes.js` 生成张量体积、卷积核网格、
  组线框；`edges.js` 生成曲线与残差高亮。
- **交互**：`interaction.js` 做 raycast、hover、select、focus、键盘快捷键；`ui.js` 渲染侧栏/树/详情/
  图例/搜索/播放控制，并驱动场景显隐。
- **性能**：几何体复用、标签按需创建、>250 节点自动简化；渲染循环只在需要时 requestAnimationFrame。

视觉编码的取舍见 [RESEARCH.md](RESEARCH.md)。

## 8. 关键设计决策

1. **为什么先 trace 再静态？** 真实形状、真实连边、真实参数量是可视化的可信度来源；
   但执行陌生代码有风险且并非总能成功，所以静态 AST 是必备回退，也是上传默认模式。
2. **为什么自己写 FX Tracer？** 默认 FX 会把 `TransformerEncoderLayer` 当成一个 leaf，导致 Transformer
   内部不可见；选择性展开复合模块、保留原子层，才能同时得到真实图与组件语义。
3. **为什么图 JSON 不直接给前端 Three.js 对象？** 数据与渲染解耦：JSON 可存档、可 diff、可导出，
   前端可独立迭代布局/材质，未来也能换渲染器（Canvas/WebGPU）。
4. **为什么前端不打包？** 项目目标是“下载即可跑、无 node 依赖”；原生 ESM 足够支撑单页应用，
   也降低维护成本。
5. **为什么标准库 HTTP？** 避免 Flask/FastAPI 依赖；服务只做静态文件 + 4 个 JSON 接口，标准库足够可靠。

## 9. 测试策略

- `tests/test_schema.py`：契约、JSON Schema、server 基础 API、路径穿越。
- `tests/test_core.py`：解析引擎对自建/样例模型的形状、参数、FLOPs、残差边、分组、静态回退、错误隔离。
- `tests/test_samples.py`：样例模型自身可实例化/前向。
- `tests/test_cli.py`：CLI 端到端、scan 输出、server API、上传静态解析。
- 前端：`node --check`、headless Chrome 截图、console 无错误。
- 最终集成：`scan samples` → schema 全绿 → `serve` → 截图。

## 10. 扩展点

- 新算子：在 `ops.py` 的规则表加映射/形状公式；无需改前端（按 `kind` 上色）。
- 新组件：在 `groups.py` 加模式；前端 group 渲染是通用的。
- 新框架：新增 `core/onnx_adapter.py` 等，把结果转换成同一 graph dict 即可。
- 新导出：JSON 已经是稳定契约，可加 SVG/PDF/自包含 HTML exporter。
- 新布局：前端 `layout.js` 可替换为分层 DAG 布局、时间轴布局或 VR 布局。
