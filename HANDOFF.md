# NetPrism 交接文档

生成时间：2026-11-25 15:45（Asia/Shanghai）

> ## ⚠️ 2026-11-25 15:57 方向变更（用户决定）
>
> 用户复测原生桌面 GUI 后明确表态：**“现在的整个 gui 形式存在巨大问题，基本用不了，
> 退回到原来的开启服务在浏览器里看的模式吧。”**
>
> 因此：
>
> - **当前主用形态 = Web 服务 + 浏览器**，不要再把原生 GUI 当交付形态。
> - 一键入口：双击仓库根目录 `start_web.cmd`，或桌面的 `NetPrism Web.cmd`。
>   它在 `127.0.0.1:8765` 起服务（端口被占用自动顺延）并打开浏览器；关窗口即停止。
> - 等价命令：`python -m netprism serve examples\demo --port 8765 --open`。
> - **免安装 EXE（2026-11-25 构建）**：`dist\web\NetPrism.exe`，9.1 MB 单文件，桌面已放一份 `NetPrism.exe`。
>   双击 → 起服务（自动挑空闲端口）+ 自动开浏览器；冷启动到 API 可访问实测 **0.68 s**。
>   实测：`--smoke` 通过（8 模型 211 层）、Playwright 真浏览器验收 `ok:true`、上传 .py 静态解析通过、
>   拖模型文件夹到 exe 上会静态重扫（8 个 samples 模型）。
>   构建：`powershell -File scripts\build_web_exe.ps1`（PyInstaller onefile，故意排除 torch/onnx/PySide6）。
>   ⚠️ 陷阱：从这个 DSH 工作区目录里直接运行 onefile exe 会报 `Could not create temporary directory!`；
>   复制到桌面/普通文件夹就正常，构建脚本的 smoke 已改为从 `%TEMP%` 副本运行。
> - 桌面旧的 GUI `NetPrism.exe` 已改名 `NetPrism-gui-old.exe`（231 MB，可删）。
> - GUI 源码与 `scripts/build_gui.ps1` 保留，但**不要再花时间修 GUI 交互**，除非用户重新要求。
> - 本文档第 1~10 节是 GUI 时代的历史记录；Web 的运行/验收以 `README.md`、
>   `web/README.md` 和 `scripts/ui_check.py` 为准。
>
> Web 端当前实测（2026-11-25 16:0x，Playwright + Chrome，`python scripts/ui_check.py
> --url http://127.0.0.1:8765/`）：`ok: true`，8 个模型可切换，canvas/悬停/选中/播放/
> 图例过滤/上下文字段全部正常，无 console/page/http 错误。EXE 版（8795 端口）复测同样 `ok: true`。
>
> - **2026-11-25 16:4x 算法补丁（trace concat skip）**：trace 模式的 concat 边原来把所有输入都当
>   skip，UNetMini 因此多画 2 条假 skip（up1/up2 主路径被画成跳连）。已按静态解析同样的规则修好：
>   最新生产者 = 主路径（data），只有形状兼容的较早分支 = skip。演示数据已重新扫描，
>   新增 `test_trace_unetmini_marks_only_the_encoder_branch_as_skip` 与 `test_unet_skip_edges_and_unet_groups`
>   的断言；全量测试 434 passed / 1 skipped。EXE 已重建并替换桌面副本。
>
> - **2026-11-26 09:0x 呈现补丁（用户截图反馈）**：
>   1. **层名**：层列表 / 详情 / 3D 标签的主标题改为算子名（`Conv3d`、`GroupNorm`、`SiLU`、`Concat`…），
>      模块路径（`net.0.0`）降到副标题/详情 meta 里 —— 不再把内部编号当名字。
>   2. **跳连目标位置**：接收跳连的 concat 如果位于某个 group 的末尾，会提升到父 group，画在 group
>      盒子的**后面**（静态与 trace 都生效）。原因是只有组输出之后的张量尺寸才和 encoder 特征一致，
>      可以 cat；之前它被画在组内/组前，跳连看起来接错了。
>   3. **导出图片**：Web 顶栏新增「导出图片」按钮，按当前视角导出 1920px 宽 PNG（含分组标签/层标签，
>      由 WebGL 画面 + CSS2D 标签合成）；EXE 同步重建（9.11 MB）。
>   验证：全量 437 passed / 1 skipped；`ui_check.py` 真浏览器 `ok:true`；EXE 端树标题/导出均实测通过。

新对话请先读本文件顶部的方向说明。GUI 时代那句“先检查桌面 EXE 的真实交互”已经过时；
现在请先确认 `start_web.cmd` 能起服务、`scripts/ui_check.py` 浏览器验收通过。

---

## 1. 用户目标与硬约束

项目：**NetPrism**，解析深度学习模型代码并在原生 3D 中可视化层/张量/卷积核/Transformer 结构，同时给出算法理解、指标和报告。

用户明确要求：

- ~~必须是原生 Python GUI（PySide6 / QtWidgets）~~ → **2026-11-25 15:57 已撤销**：
  用户改回 **Web 服务 + 浏览器**（`python -m netprism serve ... --open` / `start_web.cmd`）。
- ~~不要 HTTP 服务~~ → 撤销
- ~~不要浏览器前端~~ → 撤销
- ~~不要自动打开浏览器~~ → 撤销（一键启动器默认会打开浏览器；`--no-open` 可关）
- 最终做成 Windows EXE 发给别人（GUI 时代的目标，现在暂缓；Web 模式先在本机跑通）
- 已在申请 SignPath Foundation 免费签名；目前尚未通过
- 3D 要准确、流畅、美观
- 分组、skip connection、输入形状必须语义正确
- 用户很在意“实际打开后能不能用”，不是测试数字好不好看
- 不要在用户复测前声称已经修复交互问题

---

## 2. 当前仓库状态

仓库：

```text
C:\Users\hrbkcy2523\Documents\deepseek-harness\default-workspace\netprism
```

Git：

```text
分支: main
HEAD: 8e46183  Orbit native 3D around the model center
远端: https://github.com/bing104917-collab/netprism
本地 main 与 origin/main 已同步
工作区干净
```

最近提交：

```text
8e46183 Orbit native 3D around the model center
4e0d3dc Keep native window on active screen and reuse a single instance
9be2656 Polish 3D density and renderer performance
6c07b8b Classify concat skips by compatible shapes and main path
cc03056 Fix static list state for UNet loops; reduce edge arrow noise
```

GitHub 直连有时会 `Connection was reset`，过一会儿重试通常可恢复；最近一次推送成功。

---

## 3. 当前桌面程序

```text
C:\Users\hrbkcy2523\Desktop\NetPrism.exe
```

- 大小：231,261,807 B
- SHA256：`52C86D22FA6D398590C9802EB0AC4EE1F35D324C0B8C37026739A7B82566ED8F`
- 构建时间：2026-11-25 15:39:02
- 单文件 PyInstaller，`--windowed`
- 未签名
- 当前没有 NetPrism 进程运行

注意：这是 **onefile** 构建，每次启动都要解压约 231 MB，首次打开可能需要约 10 秒甚至更久；第二次启动也会解压，只是单实例逻辑会很快退出并把已有窗口置前。

---

## 4. 当前未解决问题（最高优先级）

### 4.1 用户录屏

用户提供的录屏：

```text
C:\Users\hrbkcy2523\AppData\Local\Packages\Microsoft.ScreenSketch_8wekyb3d8bbwe\TempState\Recordings\20261125-0733-17.1077138.mp4
```

录制时间：约 15:33（本地），**早于 15:39 的 pivot 修复构建**。

录屏内容：

- 应用窗口是可见的
- 主要动作是拖拽旋转 3D 模型
- 模型会大幅甩动、突然放大、跑出视口、边缘被裁切
- 用户反馈“基本处于不能用”“交互上存在巨大问题”
- 用户认为“啥也没改”

### 4.2 已经做过的修复

#### 窗口离屏

实际启动时主窗口曾被 Windows 放到副屏负坐标区域：

```text
窗口: (-1687, 35, -231, 974)
主屏: (0, 0, 1600, 1000)
副屏: (-1920, 0, 1920, 1080)
```

已修：

- `netprism/gui/main_window.py`
  - `showEvent()`
  - `ensure_on_active_screen()`
- 启动后检查窗口与鼠标所在屏幕的交集
- 如果窗口几乎全部不可见，则居中到当前屏幕并置顶
- 实测冻结窗口位置恢复到 `(0, 0, 1453, 936)`

#### 单实例

`packaging/gui_launcher.py` 增加 `QLocalServer / QLocalSocket`：

- 再次双击 EXE 不再创建第二个隐藏主窗口
- 会恢复并置前已有窗口
- 第二次启动仍会先解压 onefile，因此仍有一段等待

#### 3D 旋转枢轴

`netprism/gui/scene3d.py` 的 `_compute_layout()`：

- 之前模型 x 从 0 开始，旋转围绕原点（第一个节点）
- 模型越长，拖拽时越容易整条甩出视口
- 现在所有 x 坐标减去模型中心，**绕模型包围盒中心旋转**
- 新增回归测试：所有节点 x 的最小值 + 最大值 ≈ 0

#### QGraphicsView 绘制开关

上一轮为了“流畅”加的：

```python
setOptimizationFlag(DontSavePainterState)
setCacheMode(CacheBackground)
```

已经删除，恢复稳定绘制路径。

### 4.3 尚未验证

**用户还没有复测 15:39 的 pivot 版本。**

不要直接说已经修好。下一轮首先让用户重新打开最新 EXE，确认：

1. 主窗口是否直接出现在当前屏幕
2. 拖拽旋转是否围绕模型中心
3. 模型是否还会被甩出视口
4. 缩放是否仍然跳变
5. 按钮、层列表、打开文件夹是否正常
6. 启动等待是否长到不可接受

如果仍不能用，优先怀疑：

- 实际窗口/对话框仍然跑到副屏或负坐标
- onefile 启动太慢，用户以为没打开
- 拖拽/缩放仍有镜头或枢轴问题
- QGraphicsView 在真实 GPU/DPI 下的重绘问题
- 用户实际点击的是某个工具栏按钮/文件对话框，而不是 3D 拖拽

---

## 5. 已完成功能清单

### 解析

- 静态 AST（`execute=False`）与 trace（`execute=True`）
- PyTorch / ONNX / TorchScript / Keras 适配器
- 多输入、动态 shape、模块级 helper
- 静态 `list/dict` 操作支持
- UNet 的 `skips.append()` 循环状态已修复
- concat 只有 encoder 兼容分支标为 skip，decoder 主路径标为 data
- FLOPs / 参数 / 激活显存 / 感受野估算

### 输入形状

- 静态模式未指定维显示 `?`
- 原生 GUI 工具栏“输入形状”
  - 输入 `1,4,?,?,?`
  - 输入 `1,4,8,16,32`
  - `Ctrl+I`
- CLI / Web 支持 `--unknown-input`

### 3D

- 原生 3D 软件投影
- 通道厚度线性映射：`4→32→64→3` 可见
- 分组间距：原生 `2.6`，Web `2.7`
- skip/residual 用平滑弧线
- 普通数据流不画箭头；箭头只保留给 skip/residual/attention
- 导出 1920×1080 PNG，自动选 16:9 角度
- 双击复位
- 悬停提示、点击选中

### UI

- 原生模型列表
- 层表格：排序、搜索、CSV
- 层详情
- 算法理解：家族、模式、伪代码
- 报告/摘要导出
- 工具栏：打开、重解析、报告、CSV、图片、输入形状

### 层列表文字

- `Conv3d · k3×3×3 · 4→32 · p1×1×1`
- `GroupNorm · G=8 · C=32`
- `SiLU`
- 绿色 `8` 是 group 数，不是通道数

### Web 旧版

仓库仍有 `web/`：

- Three.js 3D
- 分组框、skip 弧线、层树
- DPR 上限 1.5
- 阴影贴图 1024
- 旧扫描数据必须重新 scan 才能看到新形状/skip 逻辑

但用户要求主产品是原生 GUI，不要把 Web 当交付重点。

---

## 6. 关键文件

```text
netprism/gui/main_window.py
  - 主窗口
  - 窗口屏幕检查 / 单实例置前
  - 输入形状
  - 导出图片/CSV/报告

netprism/gui/scene3d.py
  - 软件 3D 投影
  - _compute_layout：组间距、模型中心枢轴
  - 节点/分组/连接线绘制
  - fit_scene / 鼠标拖拽 / 滚轮缩放
  - export_image

netprism/gui/graph_loader.py
  - 文件夹/单文件解析入口
  - static / trace
  - input_shape / unknown_input

netprism/core/astscan.py
  - 静态 AST 执行
  - list/dict 变更
  - shape 推断
  - concat skip 分类
  - unknown_input

netprism/core/ops.py
  - 算子参数、FLOPs、shape 规则

netprism/cli.py
  - scan / inspect / report / view
  - --unknown-input

packaging/gui_launcher.py
  - 冻结 EXE 启动器
  - 单实例 QLocalServer
  - --smoke

scripts/build_gui.ps1
  - PyInstaller onefile 构建
  - 输出 dist/gui/NetPrismGUI.exe

tests/test_gui.py
tests/test_gui_qa.py
  - 原生 GUI 测试

tests/test_static_shapes.py
  - UNet list 状态 / concat skip 回归
```

---

## 7. 验证命令

Python：

```powershell
$py="C:\Users\hrbkcy2523\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe"
$proj="C:\Users\hrbkcy2523\Documents\deepseek-harness\default-workspace\netprism"
Set-Location $proj
```

全量测试：

```powershell
$env:QT_QPA_PLATFORM="offscreen"
& $py -m pytest tests -q
```

当前应得到：

```text
424 passed, 1 skipped
```

GUI / QA：

```powershell
& $py -m pytest tests/test_gui.py tests/test_gui_qa.py -q
```

当前应得到：

```text
28 passed
```

构建：

```powershell
& powershell -NoProfile -ExecutionPolicy Bypass -File scripts/build_gui.ps1 -Python $py -OneFile
Copy-Item dist\gui\NetPrismGUI.exe "$env:USERPROFILE\Desktop\NetPrism.exe" -Force
```

冻结 smoke：

```powershell
$env:QT_QPA_PLATFORM="offscreen"
& "C:\Users\hrbkcy2523\Desktop\NetPrism.exe" --smoke
```

正常应输出：

```text
OK: 8 models, 55 3D items
```

实际窗口检查（不要只跑 offscreen）：

```powershell
Start-Process "C:\Users\hrbkcy2523\Desktop\NetPrism.exe"
# 等待 10 秒后检查窗口矩形是否在 (0,0,1600,1000) 内
```

---

## 8. 已知风险 / 技术债

1. **onefile 启动慢**
   - 231 MB，每次启动解压
   - 单实例只能防止第二个窗口，不能阻止第二次解压
   - 若用户无法接受，考虑 onedir 文件夹版或更小的依赖裁剪

2. **真实交互缺少自动化覆盖**
   - 现有测试是 offscreen
   - 无法覆盖多显示器离屏、DPI、真实鼠标事件
   - 需要真实桌面 smoke / 录屏验证

3. **窗口位置**
   - 已有 active-screen 检查
   - 文件对话框(QFileDialog)没有专门做离屏检查
   - 如果用户点击“打开文件夹”没反应，要检查对话框是否也在副屏

4. **签名**
   - `CODE_SIGNING_POLICY.md`、`.github/workflows/windows-sign.yml`、`scripts/sign_exe.ps1` 已准备
   - SignPath Foundation 申请未完成
   - 当前 EXE 未签名

5. **Web 与原生 UI 并存**
   - 用户抱怨的“层列表”截图来自 Web 旧版
   - 原生 GUI 没有同款左侧层树，只有层表格/3D
   - 不要混淆两者

6. **测试环境字体**
   - offscreen 会有 `FT_New_Face failed` / `QFontDatabase` 警告
   - 不影响通过，但真实桌面上字体才准确

---

## 9. 新对话建议排查顺序

1. **先确认最新 EXE 是否真的被打开**
   - 当前无 NetPrism 进程
   - 双击后等待 onefile 解压
   - 用 Win32 `GetWindowRect` 检查窗口是否在屏幕内
   - 检查是否存在多个 NetPrism 进程

2. **录一段新的拖拽视频**
   - 用最新 15:39 的 EXE
   - 只做“按下左键拖动 3D 模型 2 秒”
   - 观察是否仍甩出视口
   - 旧录屏不能作为最新版本判断依据

3. **若仍异常，量化**
   - 在 `mouseMoveEvent` 打印/记录 `_yaw/_pitch` 和 `_rebuild` 耗时
   - 检查 `fit_scene` 是否在拖拽过程中被意外调用
   - 检查窗口是否在拖动时被系统移动/缩放
   - 检查真实 DPI 下 `frameGeometry` 与屏幕坐标

4. **不要用 offscreen 测试代替真实桌面验证**
   - 测试只证明绘图代码不崩
   - 不能证明窗口可见、可点击、拖拽合理

5. **用户确认前不要关闭问题**
   - 用户已经多次因为“测试通过但实际不能用”而不满
   - 只有用户复测通过才算完成

---

## 10. 用户沟通注意

- 用中文
- 直接说问题和实际验证结果
- 不要用测试数字掩盖实际不可用
- 不要自动打开浏览器
- 不要要求用户看服务地址
- 改完必须构建并替换桌面 EXE
- 不要声称“已修复”，除非真实窗口验证通过
- 如果仍有问题，优先要用户的具体操作/新录屏，而不是继续盲改
