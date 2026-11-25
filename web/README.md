# NetPrism Web · 棱镜 3D 可视化前端

把 NetPrism 解析出的 `graph.json` 变成可交互的 3D 棱镜场景：张量立方体、卷积核、注意力头、分组线框、残差 / 跳连弧线与数据流播放。  
**原生 ES Modules + Three.js，无构建步骤、无外网依赖**，中文 UI。

---

## 1. 运行方式

### 推荐：走本地解析服务（可用模型切换 + 上传解析）

在项目根目录运行：

```powershell
python -m netprism serve examples/demo --port 8765 --open
# 或从自己的模型文件夹开始：
python -m netprism view .\samples --open
```

打开 <http://127.0.0.1:8765/>。前端从 `/api/index` 读取模型列表，上传文件夹后走 `/api/upload-parse`（默认 `execute:false`，只做静态 AST 解析）。

### 纯静态预览（不需要 Python API）

```powershell
python -m http.server 8123 --directory web
```

打开 <http://127.0.0.1:8123/>。`/api/*` 不存在时自动回退到 `web/demo/index.json` 与 `web/demo/<id>.json`，内置 8 个示例模型；此模式下**上传按钮不可用**（会提示需要启动 `netprism serve`）。

> `web/api/health` 是一个静态标记文件：`python -m http.server` 会直接返回它，前端据此跳过 `/api/index` 探测；`netprism serve` 的真实 `/api/health` 路由优先，不受影响。因此两种运行方式下控制台都不会出现 404。

> `web/` 是纯静态目录，Three.js 已 vendor 在 `web/vendor/three/`，运行时不请求 CDN。

### 无头截图（验收）

```powershell
cd web
node scripts/screenshot.mjs --out ..\scratch\frontend\shot.png
```

脚本内置一个零依赖静态服务器（随机端口），自动寻找本机 Chrome / Edge，以 SwiftShader WebGL 无头渲染并等待截图落盘。也可指定已启动的服务：

```powershell
node scripts/screenshot.mjs --url http://127.0.0.1:8765/ --out ..\scratch\frontend\api.png --width 1600 --height 1000
```

---

## 2. 交互与快捷键

### 鼠标

| 操作 | 效果 |
|---|---|
| 左键拖拽 | 旋转视角（OrbitControls） |
| 滚轮 / 双指捏合 | 缩放 |
| 右键拖拽 / 双指拖动 | 平移 |
| 悬停层、卷积核网格、连线 | 跟随鼠标的 tooltip（op、kind、in/out shape、参数量、FLOPs、超参、源码行） |
| 单击层 | 选中：右侧详情、高亮上下游，其余层保留 35% 左右上下文 |
| 双击层 | 聚焦该层（相机缓动推近） |
| 双击空白 | 复位视角并取消选中 |
| 拖入文件夹 | 收集 `.py` 并 POST `/api/upload-parse`，返回的 `{index, graphs}` 直接渲染 |

### 键盘

| 按键 | 效果 |
|---|---|
| `Space` | 播放 / 暂停数据流 |
| `→` / `←` | 数据流单步前进 / 后退 |
| `Esc` | 取消选中 / 关闭帮助弹层 |
| `F` | 适应窗口 |
| `R` | 重置视角（取消选中 + fit） |
| `L` | 显示 / 隐藏层标签 |
| `A` | 打开 / 关闭算法理解面板 |
| `Shift`+`A` | 一键显示全部层标签 |
| `G` | 显示 / 隐藏分组线框 |
| `K` | 显示 / 隐藏卷积核网格 |
| `E` | 显示 / 隐藏连线 |
| `T` | 打开 / 关闭层表格视图 |
| `/` | 聚焦左侧搜索框（表格打开时聚焦表格搜索） |
| `?` | 打开快捷键帮助 |

### 底栏开关

- **播放数据流 / 单步**：光点沿执行顺序流动，底部 HUD 显示当前层与进度；HUD 右上角可暂停。
- **卷积核 / 分组 / 标签 / 全部标签 / 连线**：显隐开关。
- **重置视角 / 适应窗口**：两者使用同一套自动构图；重置会额外取消选中。
- 窄屏时侧栏可点画布边缘的 `‹` / `›` 收起；画布点击也会自动收起抽屉。

### 左栏 / 右栏

- **左栏**：模型摘要（framework 徽章、参数量、FLOPs、叶子层数、场景节点 / 分组数、输入输出、warnings；可选 activation / 感受野 / MACs 指标）、搜索、按 group 折叠的层树。
- **右栏**：选中层的完整详情（形状、计算量、v2 指标、算法公式 / 语义 / 算法 tags、所属 group 模式、全部 `params`、源码位置与代码、tags、notes、上一层/下一层）；下方是 **kind 图例过滤**，点击色块隐藏 / 显示对应算子。
- 搜索会过滤并高亮匹配层；隐藏的 kind 在树中呈半透明，点击层会自动恢复显示。

### v2 格式徽章与指标

- 顶栏 `#framework-badge` 显示 `PyTorch / ONNX / Keras / TorchScript`（缺字段时隐藏，不伪造）；旁边 `#mode-badge` 显示 `trace / static / hybrid / onnx / keras / torchscript`。
- 摘要与层详情读取可选字段 `model.metrics` / `node.metrics`：
  - `activationBytes`（MB，必要时回退 `peakActivationBytes`）
  - `maxReceptiveField` / `receptiveField`
  - `macs`
- 所有估算值均带 **“估算”** 标签；旧 graph 无 metrics 时摘要指标区隐藏，表格对应单元格显示 `—`，不会伪造 0。
- 层详情与 tooltip 在 metrics 存在时自动追加对应指标行。

### 表格视图

- 点击顶栏 **表格** 按钮或按 `T` 打开覆盖式表格；`Esc` / 关闭按钮回到 3D（3D 场景不销毁，选中/相机状态保留）。
- 10 列：层名、算子、Kind、IN、OUT、参数量、FLOPs、激活显存、感受野、分组；列头点击排序（再次点击反向），支持键盘 Enter / Space。
- `#table-search` 联动过滤（层名 / op / kind / group / shape / tags），`#table-export` 按当前筛选与排序导出 CSV（UTF-8，无第三方依赖）。
- 表格使用 `tbody.innerHTML` 批量渲染 + sticky 表头 / 首列，450 行实测可排序、搜索与导出；大模型不需要重新加载 3D 场景。
- CSV 表头：`name,op,kind,in_shapes,out_shapes,params,flops,activation_mb,receptive_field,group`；缺失 metrics 写 `—`。

### 算法理解面板（v2.1）

- 点击顶栏 **算法** 按钮或按 `A` 打开覆盖式面板；与 **表格** 互斥（打开一个会自动关闭另一个），`Esc` / 关闭按钮返回 3D。
- `#algo-family`：展示 `model.meta.familyLabel` / `family`、家族置信度（支持 `familyConfidence` / `confidence`；未提供时回退为最高模式置信度，仍无则 `—`）与 `algorithmTags`。
- `#algo-patterns`：渲染 `model.meta.patterns` 卡片，含 `kind`、`label`、`formula`、confidence 进度条、`evidence` 列表与节点/分组数量。
- `#algo-pseudocode`：等宽前向伪代码，复制按钮写入剪贴板；`model.meta.pseudocode` 缺失时显示「暂无伪代码」。
- 无 `model.meta` 的旧 graph：`#algo-family` 显示 **“暂无算法信息”**，模式/伪代码各自显示空状态，不报错、不伪造。
- 层详情 `#detail-algo`：显示 `node.meta.formula` / `semantic` / `axes` / `tags`；`#detail-group-pattern` 显示 `group.meta.pattern` 的 label / formula / confidence / tags。
- tooltip 只显示 `node.meta.semantic`（截断），不塞长公式；完整公式在层详情与算法面板中。

---

## 3. URL 参数与数据接口

### URL

- `#model=<id>`：直接打开指定模型，例如  
  `http://127.0.0.1:8765/#model=custom_attention__tinygpt`
- 切换模型、上传解析后会自动 `history.replaceState` 更新 hash，可复制分享。

### 启动顺序

1. `GET /api/health`（失败不阻塞）
2. `GET /api/index` → 模型列表
3. 默认加载第一个模型 / hash 指定模型：`GET /api/model/<id>`
4. 失败时回退：
   - `GET web/demo/index.json`
   - `GET web/demo/<file>`（index 里的 `file` 字段）

### 上传

`<input type="file" webkitdirectory>` 或整文件夹拖拽，读取文本后 POST：

```json
{
  "files": [{ "path": "sub/model.py", "content": "import torch ..." }],
  "execute": false
}
```

响应：

```json
{
  "index": { "schemaVersion": "1.0", "models": ["..."], "errors": [] },
  "graphs": { "<id>": { "schemaVersion": "1.0", "...": "..." } },
  "errors": [{ "file": "bad.py", "message": "..." }]
}
```

前端限制（与服务端一致）：只接收 `.py`；单文件 ≤ 1.5 MB；总计 ≤ 8 MB；最多 200 个文件；跳过 `.git / node_modules / __pycache__ / build / dist` 等目录。解析失败的文件会显示在完成提示面板里，不阻塞其他模型。

`graph` 数据契约见 `../schema/graph.schema.json`；前端只依赖契约字段，`params` 中未知字段会以“原始 key”形式容错展示。

---

## 4. 3D 映射规则

- **执行顺序沿 X 轴**；同组间距更近、组间更远。
- 张量尺寸：
  - CNN 4D `(N, C, H, W)`：Y=H、Z=W（按全图最大值归一化），X 厚度≈log(C)。
  - Linear 2D `(N, F)`：细长板，Y 固定，Z≈log(F)。
  - Transformer 3D `(N, L, D)`：Y=L（序列）、Z≈log(D)（特征）、X 薄。
  - 1D / 标量：细条 / 小方块。
- **卷积核**：conv 节点输入侧显示 `kH×kW` 滤波网格，沿 X 的深度≈log(Cin/groups)；悬停显示 kernel/stride/padding/groups/dilation；可用 `K` 关闭。
- **Attention**：按 `params.numHeads` 在 X 方向叠放薄片（最多 16 层），颜色由粉到紫渐变；内部 `kind:"attention"` 连线默认隐藏，选中注意力节点（或与它相邻）时以局部短曲线出现。
- **FFN / MLP**：扩张段 + 收缩段两块板，中间有发光桥接线；扩张比例来自 `hiddenFeatures / outFeatures`。
- **分组**：有直接成员的连续分组画半透明线框 + 标签；跨组的父级 / 非连续分组画底部括号标签；同 label 的重复分组自动附加 module path，例如 `Encoder 1 · Conv Block`。
- **连线**：
  - `data`：细灰白曲线
  - `residual`：紫色高空弧线
  - `skip`：橙色加粗高空弧线（UNet 的 skip 从模型上方走）
  - `attention`：默认隐藏，选中 attention 节点后显示洋红局部连线
  - `control`：虚线
  - 选中层时，与其相连的连线加亮；其余降低透明度。
- **标签分级**：
  - 默认只显示 **depth ≤ 1 的 group 标签**（VGG: `Features/Classifier`；ResNet: `Stem/Layer 1/Layer 2`；UNet: `Encoder 1-3 / Decoder 1-2`；TinyGPT: `Transformer Blocks`）。
  - depth ≥ 2 的 `Residual Block / Shortcut / Attention / Feed Forward / Conv Block` 标签只在 **hover 或选中对应节点** 时显示，或相机拉近到 fit 距离 70% 以内时显示 depth=2 组；子标签只显示末段（如 `Residual Block 1`），完整 label 保留在 tooltip 与侧栏。
  - 默认最多同时显示 8 个 group 标签（按 depth 优先、选中路径优先），再做屏幕碰撞检测；重叠时隐藏低优先级标签，不会压在节点上。
  - 输入 / 输出、各 group 首层与最大参数层等关键节点标签默认显示；`>250` 节点自动简化。
  - `全部标签`（快捷键 `Shift`+`A`）会强制显示所有节点 / group 标签并跳过碰撞裁剪。
  - `hover` / 选中始终显示完整 tooltip。
- **性能 / 可访问性**：>250 节点进入简化模式；`prefers-reduced-motion` 下关闭入场动画、数据流脉冲与镜头缓动；窗口 resize 自动适配。

---

## 5. 目录结构

```
web/
  index.html             # importmap + 全部 UI 骨架（本地 three.js）
  css/app.css            # 暗色玻璃 UI、响应式、动效降级
  js/
    main.js              # 启动、数据源、模型切换、上传、hash、响应式
    api.js               # /api + demo 回退 + 上传
    scene.js             # Three 渲染器、灯光、地面、相机 fit/focus、数据流光点
    layout.js            # 确定性 3D 布局（执行序 / 张量尺寸 / 分组范围 / 标签策略）
    nodes.js             # 张量立方体、卷积核、attention heads、FFN、token 细节
    groups.js            # 分组线框 / 括号 / 标签
    edges.js             # 连线曲线、残差 / skip 弧线、箭头、边标签
    interaction.js       # raycast hover / 选中 / 键盘 / 播放 / 过滤 / 标签避让
    ui.js                # 顶栏、摘要、层树、详情、图例、tooltip、状态、toast
    format.js            # 数字 / 形状 / kind 颜色与中文标签
    labels.js            # CSS2D 标签封装与清理
  demo/                  # 由 examples/demo 同步的静态兜底数据（8 模型）
  api/health             # 静态模式标记（python http.server 下跳过 API 探测）
  vendor/three/          # three.module.js + three.core.js + OrbitControls + CSS2DRenderer
  scripts/screenshot.mjs # 零依赖无头截图脚本
```

### 开发自检

```powershell
# 每个 JS 做语法检查（Node 24 会自动识别 ESM）
cd web
Get-ChildItem js -Filter *.js | ForEach-Object { node --check $_.FullName }

# 无头 WebGL 截图
node scripts/screenshot.mjs --out ..\scratch\frontend\shot.png
```

要求：现代 Chrome / Edge（支持 WebGL2、ES modules、importmap、`color-mix`）。若没有 WebGL，会显示错误面板而不是白屏。

### 标签分级验收点

- ResNet 默认只出现 `Stem / Layer 1 / Layer 2`，不会再出现 8 个 `Layer 1 · Residual Block 1 · Shortcut` 长标签。
- 选中某个 Shortcut 节点后，出现短标签 `Shortcut / Residual Block 1 / Layer 1`，其余标签自动让位。
- 缩放拉近后出现 4 个 `Residual Block` 短标签（总数仍 ≤ 8）。
- TinyGPT 默认只出现 `Transformer Blocks`，选中 block 内节点后出现 `Transformer Block 1/2`。
- `全部标签` 打开时 ResNet 11 个 group 标签 + 全部节点标签强制显示。

---

## 6. 验收 / 已知边界

- API 服务运行时 Playwright 检查：切模型、hover tooltip、单击详情、双击聚焦、数据流、开关、上传 `samples/` 目录（8 个模型全部可切换）、单文件上传均通过；控制台 0 error / 0 warning。
- v2 回归：8 模型表格行数与 graph 节点一致；10 列排序、搜索、CSV 下载、`T`/`Esc` 切换、450 行表格、metrics 缺失 `—` 均通过；framework/mode 四种格式徽章（PyTorch/ONNX/Keras/TorchScript）通过；控制台 0 error / 0 warning。
- v2.1 回归：注入 meta 验证 `#algo-family` / `#algo-patterns` / `#algo-pseudocode`、family confidence、pattern formula/evidence、复制伪代码、层详情公式 / semantic / group pattern、tooltip 只显示 semantic、空 meta 显示“暂无算法信息”、`A`/`Esc`、与表格互斥、1280 宽屏；控制台 0 error / 0 warning。
- 纯静态模式（`python -m http.server`）同样 0 console error（`web/api/health` 静态标记避免 404）；上传解析会提示需要 `netprism serve`。
- three.js 版本：`0.187.0`，license 见 `vendor/three/LICENSE`。
- 数据契约 v1.0 冻结；前端不修改 graph，不执行上传代码。
