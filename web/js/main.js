/* NetPrism frontend entry: data boot, model switching, uploads, wiring. */

import { Vector3 } from "three";
import {
  ApiError, assertGraph, isUploadUnsupported, loadGraph, loadIndex, uploadParse,
} from "./api.js";
import { SceneManager } from "./scene.js";
import { UI } from "./ui.js";
import { InteractionManager } from "./interaction.js";
import { computeLayout } from "./layout.js";
import { buildNodeContent } from "./nodes.js";
import { buildGroupContent } from "./groups.js";
import { buildEdgeContent } from "./edges.js";

const IGNORED_DIRS = new Set([
  ".git", ".venv", "venv", "env", "node_modules", "__pycache__", "site-packages",
  "build", "dist", ".netprism_out", ".idea", ".vscode",
]);
const MAX_FILES = 200;
const MAX_TOTAL_BYTES = 8 * 1024 * 1024;
const MAX_FILE_BYTES = 1.5 * 1024 * 1024;

const state = {
  index: null,
  source: null,
  entry: null,
  graph: null,
  layout: null,
  uploadGraphs: new Map(),
  apiOnline: false,
  narrow: false,
  booting: false,
  ready: false,
};

const diagnostics = {
  ready: false,
  errors: [],
  model: null,
  webgl: null,
};
window.__NETPRISM__ = diagnostics;

document.documentElement.dataset.appState = "loading";

window.addEventListener("error", (event) => {
  diagnostics.errors.push(String(event.message || event.error || "unknown error"));
  if (!diagnostics.ready) document.documentElement.dataset.appState = "error";
});
window.addEventListener("unhandledrejection", (event) => {
  const reason = event.reason;
  diagnostics.errors.push(String(reason && reason.message ? reason.message : reason));
  if (!diagnostics.ready) document.documentElement.dataset.appState = "error";
});

let scene = null;
let ui = null;
let interaction = null;
let loadSeq = 0;

function boot() {
  ui = new UI();
  ui.bind({
    onModelChange: (id) => switchModel(id),
    onFilesSelected: (files) => handleFileInput(files),
    onKindToggle: (kind) => interaction?.toggleKind(kind),
    onShowAllKinds: () => interaction?.showAllKinds(),
    onTreeSelect: (id) => interaction?.selectNode(id, { focus: true }),
    onNavigate: (delta) => interaction?.selectRelative(delta),
    onPlayToggle: () => interaction?.togglePlay(),
    onStep: (delta) => interaction?.stepFlow(delta),
    onToggle: (name) => interaction?.toggle(name),
    onReset: () => interaction?.resetView(),
    onFit: () => scene?.fit(),
    onTogglePanel: (side) => togglePanel(side),
    onFocusSelected: () => interaction?.focusSelected(),
    onRetry: () => bootstrap(),
    onDemo: () => bootstrap({ forceDemo: true }),
    onStateAction: (action) => handleStateAction(action),
  });

  try {
    if (!window.WebGLRenderingContext) throw new Error("浏览器不支持 WebGL");
    scene = new SceneManager(document.getElementById("viewport"));
  } catch (error) {
    showWebGLError(error);
    return;
  }

  scene.onContextLost = () => {
    ui.toast("error", "WebGL 上下文丢失", "请刷新页面恢复 3D 场景。", 9000);
    document.documentElement.dataset.appState = "error";
  };

  interaction = new InteractionManager(scene, ui, document.getElementById("viewport"));
  setupResponsive();
  setupDropzone();
  window.addEventListener("hashchange", onHashChange);

  exposeDebugApi();
  ui.setState({
    mode: "loading",
    title: "正在连接解析服务",
    message: "GET /api/health → /api/index；若服务未启动，会自动读取内置演示数据。",
  });
  ui.setBusy(true);
  bootstrap();
}

/* -------------------------------------------------------------------------- */
/* Bootstrap & model loading                                                  */
/* -------------------------------------------------------------------------- */

async function bootstrap({ forceDemo = false } = {}) {
  if (state.booting) return;
  state.booting = true;
  ui.setBusy(true);
  ui.setState({
    mode: "loading",
    title: "正在读取模型索引",
    message: forceDemo ? "读取内置演示数据 web/demo/index.json…" : "连接 /api/index 并加载模型列表…",
  });
  try {
    const { index, source, apiError } = await loadIndex({ forceDemo });
    state.index = index;
    state.source = source;
    state.apiOnline = source === "api";
    ui.setIndex(index, source);

    if (!index.models.length) {
      state.booting = false;
      state.ready = false;
      ui.setState(null);
      ui.setBusy(false);
      showEmptyState();
      return;
    }

    const wanted = readHashModel();
    const entry = index.models.find((model) => model.id === wanted) || index.models[0];
    await loadModel(entry, { source, initial: true });

    state.ready = true;
    diagnostics.ready = true;
    document.documentElement.dataset.appState = "ready";
    ui.setState(null);
    ui.setBusy(false);

    if (source === "demo") {
      ui.toast(
        "info",
        "使用内置演示数据",
        apiError
          ? "未连接 /api 解析服务，已加载 web/demo。上传解析需要运行 python -m netprism serve。"
          : "已加载 web/demo 内置示例。",
        7200,
      );
    }
    if (wanted && wanted !== entry.id) {
      ui.toast("warn", "模型不存在", `未找到 #model=${wanted}，已加载「${entry.name}」。`, 6000);
    }
  } catch (error) {
    state.ready = false;
    document.documentElement.dataset.appState = "error";
    diagnostics.errors.push(error && error.message ? error.message : String(error));
    showLoadError(error);
  } finally {
    state.booting = false;
    ui.setBusy(false);
  }
}

async function loadModel(entry, { source = state.source, graph = null, initial = false } = {}) {
  if (!entry || !entry.id) throw new ApiError("模型条目不完整");
  const seq = ++loadSeq;
  let loadedGraph = graph;
  if (!loadedGraph) {
    if (source === "upload") loadedGraph = state.uploadGraphs.get(entry.id);
    if (!loadedGraph) loadedGraph = await loadGraph(entry, source);
  }
  assertGraph(loadedGraph, entry.id);
  if (!Array.isArray(loadedGraph.nodes) || loadedGraph.nodes.length === 0) {
    throw new ApiError(`模型「${entry.name || entry.id}」没有任何层`);
  }
  if (seq !== loadSeq) return null;

  const layout = computeLayout(loadedGraph);
  const nodeContent = buildNodeContent(layout, loadedGraph);
  const groupContent = buildGroupContent(layout);
  const edgeContent = buildEdgeContent(layout, loadedGraph);

  const content = {
    nodesRoot: nodeContent.root,
    kernelsRoot: nodeContent.kernelsRoot,
    groupsRoot: groupContent.root,
    edgesRoot: edgeContent.root,
    nodeViews: nodeContent.nodeViews,
    kernelViews: nodeContent.kernelViews,
    groupViews: groupContent.groupViews,
    edgeViews: edgeContent.edgeViews,
    pickables: [...nodeContent.pickables, ...edgeContent.pickables],
  };

  scene.setContent(content, layout, loadedGraph);
  interaction.setModel({
    layout,
    graph: loadedGraph,
    nodeViews: nodeContent.nodeViews,
    kernelViews: nodeContent.kernelViews,
    groupViews: groupContent.groupViews,
    edgeViews: edgeContent.edgeViews,
    edgesRoot: edgeContent.root,
    pickables: content.pickables,
  });

  state.entry = entry;
  state.graph = loadedGraph;
  state.layout = layout;
  state.source = source;
  state.uploadGraphs = state.uploadGraphs || new Map();

  ui.setHiddenKinds(interaction.getHiddenKinds());
  ui.setModel(loadedGraph, entry, layout, source);
  ui.setToggleButtons(interaction.getToggles());
  ui.refreshTreeFiltered(interaction.getHiddenKinds());
  interaction.refreshVisibility();
  interaction.refreshMaterials();
  scene.startIntro();
  writeHash(entry.id);
  onboardAfterLoad();

  diagnostics.model = {
    id: entry.id,
    name: loadedGraph.model?.name || entry.name,
    mode: loadedGraph.model?.mode,
    source,
    nodes: layout.nodeCount,
    edges: layout.edgeCount,
    groups: layout.groups.length,
    big: layout.big,
  };
  requestAnimationFrame(() => {
    if (!scene) return;
    diagnostics.webgl = {
      calls: scene.renderer.info.render.calls,
      triangles: scene.renderer.info.render.triangles,
      nodes: layout.nodeCount,
    };
    window.__NETPRISM__.getNodeScreenPosition = (id) => getNodeScreenPosition(id);
    window.__NETPRISM__.listNodes = () => listNodePositions();
  });
  return { layout, graph: loadedGraph };
}

function onboardAfterLoad() {
  window.__NETPRISM__.getNodeScreenPosition = (id) => getNodeScreenPosition(id);
  window.__NETPRISM__.listNodes = () => listNodePositions();
}

async function switchModel(id) {
  if (!id || !state.index) return;
  if (state.entry && state.entry.id === id) return;
  const entry = state.index.models.find((model) => model.id === id);
  if (!entry) {
    ui.toast("error", "找不到模型", `索引中没有 ${id}`);
    if (state.entry && ui.el.modelSelect) ui.el.modelSelect.value = state.entry.id;
    return;
  }
  ui.setBusy(true);
  ui.hideNotice();
  try {
    if (state.source === "upload") {
      const graph = state.uploadGraphs.get(id);
      if (!graph) throw new ApiError(`上传结果中缺少模型 ${id}`);
      await loadModel(entry, { source: "upload", graph });
    } else {
      await loadModel(entry, { source: state.source });
    }
    document.documentElement.dataset.appState = "ready";
  } catch (error) {
    diagnostics.errors.push(error.message || String(error));
    ui.toast("error", "模型加载失败", error.message || String(error));
    ui.showNotice({
      type: "error",
      title: "无法切换到该模型",
      message: error.message || String(error),
      items: [],
    });
    if (state.entry && ui.el.modelSelect) ui.el.modelSelect.value = state.entry.id;
  } finally {
    ui.setBusy(false);
  }
}

function findCurrentSourceGraph(entry) {
  if (state.source === "upload") return state.uploadGraphs.get(entry.id) || null;
  return null;
}

/* -------------------------------------------------------------------------- */
/* Hash routing                                                               */
/* -------------------------------------------------------------------------- */

function readHashModel() {
  const match = /(?:^#|&)model=([^&]+)/.exec(window.location.hash || "");
  if (!match) return null;
  try {
    return decodeURIComponent(match[1]);
  } catch {
    return match[1];
  }
}

function writeHash(id) {
  if (!id) return;
  const next = `#model=${encodeURIComponent(id)}`;
  if (window.location.hash === next) return;
  try {
    window.history.replaceState(null, "", next);
  } catch {
    window.location.hash = next;
  }
}

function onHashChange() {
  if (!state.index || state.booting) return;
  const id = readHashModel();
  if (!id) return;
  if (state.entry && state.entry.id === id) return;
  const entry = state.index.models.find((model) => model.id === id);
  if (entry) switchModel(id);
  else ui.toast("warn", "模型不存在", `索引中没有 #model=${id}`);
}

/* -------------------------------------------------------------------------- */
/* Upload & drag/drop                                                         */
/* -------------------------------------------------------------------------- */

function setupDropzone() {
  let depth = 0;
  const overlay = (visible) => ui.setDropOverlay(visible);
  const hasFiles = (event) => {
    const dt = event.dataTransfer;
    if (!dt) return false;
    if (dt.types && Array.from(dt.types).includes("Files")) return true;
    return Boolean(dt.items && Array.from(dt.items).some((item) => item.kind === "file"));
  };

  window.addEventListener("dragenter", (event) => {
    if (!hasFiles(event)) return;
    event.preventDefault();
    depth += 1;
    if (depth === 1) overlay(true);
  });
  window.addEventListener("dragover", (event) => {
    if (!hasFiles(event)) return;
    event.preventDefault();
    if (event.dataTransfer) event.dataTransfer.dropEffect = "copy";
  });
  window.addEventListener("dragleave", (event) => {
    if (!hasFiles(event)) return;
    depth = Math.max(0, depth - 1);
    if (depth === 0) overlay(false);
  });
  window.addEventListener("drop", async (event) => {
    if (!hasFiles(event)) return;
    event.preventDefault();
    depth = 0;
    overlay(false);
    if (state.booting) {
      ui.toast("warn", "正在加载", "请等待当前模型加载完成后再上传。");
      return;
    }
    try {
      const items = await collectDroppedFiles(event.dataTransfer);
      handleFiles(items);
    } catch (error) {
      ui.toast("error", "无法读取拖入的文件", error.message || String(error));
    }
  });
}

async function collectDroppedFiles(dataTransfer) {
  if (!dataTransfer) return [];
  const items = dataTransfer.items ? Array.from(dataTransfer.items) : [];
  const entries = [];
  if (items.length && typeof items[0].webkitGetAsEntry === "function") {
    for (const item of items) {
      if (item.kind !== "file") continue;
      const entry = item.webkitGetAsEntry();
      if (entry) entries.push(entry);
    }
  }
  if (entries.length) {
    const collected = [];
    for (const entry of entries) await walkEntry(entry, collected);
    return collected;
  }
  return Array.from(dataTransfer.files || []).map((file) => ({
    file,
    path: file.webkitRelativePath || file.name,
  }));
}

function shouldIgnoreDir(name) {
  if (!name) return true;
  if (IGNORED_DIRS.has(name)) return true;
  return name.startsWith(".");
}

function pathIgnored(path) {
  const segments = String(path || "").split(/[\\/]/).filter(Boolean);
  return segments.some((segment, index) => index < segments.length - 1 && shouldIgnoreDir(segment));
}

function readEntries(reader) {
  return new Promise((resolve, reject) => reader.readEntries(resolve, reject));
}

async function walkEntry(entry, out) {
  if (!entry || out.length >= MAX_FILES) return;
  if (entry.isFile) {
    const name = entry.name || "";
    if (!/\.py$/i.test(name)) return;
    const path = String(entry.fullPath || name).replace(/^\/+/, "");
    if (pathIgnored(path)) return;
    const file = await new Promise((resolve, reject) => entry.file(resolve, reject));
    out.push({ file, path });
    return;
  }
  if (entry.isDirectory) {
    if (shouldIgnoreDir(entry.name)) return;
    const reader = entry.createReader();
    while (out.length < MAX_FILES) {
      const batch = await readEntries(reader);
      if (!batch.length) break;
      for (const child of batch) await walkEntry(child, out);
    }
  }
}

function handleFileInput(fileList) {
  const files = Array.from(fileList || []).map((file) => ({
    file,
    path: file.webkitRelativePath || file.name,
  }));
  handleFiles(files);
}

function normalizeFileItems(items) {
  return (items || [])
    .filter((item) => item && item.file)
    .filter((item) => /\.py$/i.test(item.file.name || item.path || ""))
    .filter((item) => !pathIgnored(item.path));
}

async function handleFiles(rawItems) {
  const items = normalizeFileItems(rawItems);
  if (!items.length) {
    ui.toast("warn", "没有 .py 文件", "选择一个包含 PyTorch 模型源码的文件夹（只读取 .py）。");
    ui.showNotice({
      type: "warn",
      title: "没有找到可解析的文件",
      message: "NetPrism 只接收 .py 文件，并跳过 .git / node_modules / __pycache__ 等目录。",
    });
    return;
  }
  if (items.length > MAX_FILES) {
    ui.toast("error", "文件过多", `最多一次解析 ${MAX_FILES} 个 .py 文件，当前 ${items.length} 个。`);
    return;
  }
  let total = 0;
  for (const item of items) {
    const size = item.file.size || 0;
    if (size > MAX_FILE_BYTES) {
      ui.toast("error", "单文件过大", `${item.path} 超过 1.5 MB 限制。`);
      return;
    }
    total += size;
  }
  if (total > MAX_TOTAL_BYTES) {
    ui.toast("error", "文件夹过大", `上传总计 ${(total / 1024 / 1024).toFixed(1)} MB，超过 8 MB 限制。`);
    return;
  }

  ui.setUploading(true);
  ui.setBusy(true);
  ui.hideNotice();
  try {
    const payload = await Promise.all(items.map(async (item) => ({
      path: item.path,
      content: await item.file.text(),
    })));
    const result = await uploadParse(payload);
    const normalized = normalizeUploadResult(result);
    if (!normalized.models.length) {
      throw new ApiError(
        "没有解析出模型。请确认文件夹里有继承 nn.Module 且包含 forward() 的类。",
      );
    }
    state.source = "upload";
    state.index = normalized.index;
    state.uploadGraphs = normalized.graphs;
    ui.setIndex(normalized.index, "upload");
    const entry = normalized.models[0];
    await loadModel(entry, {
      source: "upload",
      graph: normalized.graphs.get(entry.id),
    });
    document.documentElement.dataset.appState = "ready";
    ui.toast(
      "success",
      "解析完成",
      `已载入 ${normalized.models.length} 个模型${normalized.errors.length ? `，${normalized.errors.length} 个文件失败` : ""}。`,
      6000,
    );
    if (normalized.errors.length) {
      ui.showNotice({
        type: "warn",
        title: `解析完成 ${normalized.models.length} 个模型，${normalized.errors.length} 个文件失败`,
        message: "失败文件已跳过，不影响已解析模型的可视化。",
        items: normalized.errors.map((error) => `${error.file || "未知文件"}: ${error.message || ""}`),
      });
    }
  } catch (error) {
    const message = error && error.message ? error.message : String(error);
    diagnostics.errors.push(message);
    ui.toast("error", "上传解析失败", message);
    if (isUploadUnsupported(error)) {
      ui.showNotice({
        type: "warn",
        title: "当前为静态演示模式",
        message:
          "上传解析需要后端服务。请在项目根目录运行：python -m netprism serve examples/demo --open（或 python -m netprism view <文件夹>）",
        items: [
          "静态托管（python -m http.server）只提供页面，不提供 /api/upload-parse。",
          "上传只做静态 AST 解析，默认不执行模型代码。",
        ],
      });
    } else {
      ui.showNotice({
        type: "error",
        title: "无法解析上传的文件夹",
        message,
      });
    }
  } finally {
    ui.setUploading(false);
    ui.setBusy(false);
  }
}

function normalizeUploadResult(result) {
  const rawGraphs = result && result.graphs && typeof result.graphs === "object" ? result.graphs : {};
  const graphs = new Map();
  for (const [id, graph] of Object.entries(rawGraphs)) {
    if (graph && Array.isArray(graph.nodes) && Array.isArray(graph.edges) && Array.isArray(graph.groups)) {
      graphs.set(id, graph);
    }
  }
  const errors = [];
  const rawErrors = [
    ...(Array.isArray(result?.errors) ? result.errors : []),
    ...(Array.isArray(result?.index?.errors) ? result.index.errors : []),
  ];
  for (const error of rawErrors) {
    if (!error) continue;
    if (typeof error === "string") errors.push({ file: "", message: error });
    else errors.push({ file: error.file || "", message: error.message || String(error) });
  }

  let models = Array.isArray(result?.index?.models) ? result.index.models : [];
  models = models
    .filter((model) => model && model.id)
    .map((model) => {
      const graph = graphs.get(model.id);
      const meta = graph?.model || {};
      return {
        ...model,
        name: model.name || meta.name || model.id,
        className: model.className ?? meta.className ?? null,
        file: model.file || `${model.id}.json`,
        mode: model.mode || meta.mode || "static",
        numLayers: Number(model.numLayers ?? meta.numLayers) || 0,
        totalParams: Number(model.totalParams ?? meta.totalParams) || 0,
        totalFlops: Number(model.totalFlops ?? meta.totalFlops) || 0,
        warnings: Number(model.warnings ?? meta.warnings?.length) || 0,
      };
    })
    .filter((model) => graphs.has(model.id));

  // Backend versions may return graphs without an index. Derive entries.
  if (!models.length) {
    for (const [id, graph] of graphs) {
      const meta = graph.model || {};
      models.push({
        id,
        name: meta.name || id,
        className: meta.className ?? null,
        file: `${id}.json`,
        source: meta.source?.relPath || "",
        mode: meta.mode || "static",
        numLayers: Number(meta.numLayers) || 0,
        totalParams: Number(meta.totalParams) || 0,
        totalFlops: Number(meta.totalFlops) || 0,
        warnings: Array.isArray(meta.warnings) ? meta.warnings.length : 0,
      });
    }
  }

  const index = {
    schemaVersion: result?.index?.schemaVersion || "1.0",
    generatedAt: result?.index?.generatedAt || new Date().toISOString(),
    root: result?.index?.root || "upload",
    models,
    errors,
  };
  return { index, graphs, models, errors };
}

/* -------------------------------------------------------------------------- */
/* States & responsive                                                        */
/* -------------------------------------------------------------------------- */

function showEmptyState() {
  ui.setState({
    mode: "empty",
    title: "还没有可展示的模型",
    message:
      "把装满 PyTorch 代码的文件夹拖进页面，或上传一个文件夹。静态解析，不执行模型代码。",
    actions: [
      { action: "upload", label: "上传文件夹", kind: "primary", icon: "upload" },
      { action: "demo", label: "加载内置示例", icon: "cube" },
      { action: "help", label: "使用说明", icon: "help" },
    ],
  });
}

function showLoadError(error) {
  const message = error && error.message ? error.message : String(error);
  const detail = [message];
  if (error && error.status) detail.push(`HTTP ${error.status}`);
  if (error && error.url) detail.push(String(error.url));
  ui.setState({
    mode: "error",
    title: "无法加载模型",
    message:
      "没有连接到 NetPrism 解析服务，内置演示数据也读取失败。请启动本地服务，或重新选择模型文件夹。",
    detail: detail.join("  ·  "),
    actions: [
      { action: "retry", label: "重新加载", kind: "primary", icon: "reset" },
      { action: "upload", label: "上传文件夹", icon: "upload" },
      { action: "help", label: "使用说明", icon: "help" },
    ],
  });
}

function showWebGLError(error) {
  diagnostics.errors.push(error && error.message ? error.message : String(error));
  document.documentElement.dataset.appState = "error";
  if (!ui) return;
  ui.setState({
    mode: "error",
    title: "无法初始化 3D 场景",
    message:
      "浏览器没有可用的 WebGL 上下文。请开启硬件加速，或使用最新版 Chrome / Edge 再试一次。",
    detail: error && error.message ? error.message : String(error),
    actions: [{ action: "retry", label: "重试", kind: "primary", icon: "reset" }],
  });
}

function handleStateAction(action) {
  switch (action) {
    case "retry":
      bootstrap();
      break;
    case "demo":
      bootstrap({ forceDemo: true });
      break;
    case "upload":
      ui.el.fileInput?.click();
      break;
    case "help":
      ui.openHelp();
      break;
    default:
      break;
  }
}

function setupResponsive() {
  const mq = window.matchMedia("(max-width: 1120px)");
  const apply = (narrow, initial) => {
    state.narrow = narrow;
    if (narrow) {
      if (initial) {
        ui.setPanelCollapsed("left", true);
        ui.setPanelCollapsed("right", true);
      }
    } else if (initial) {
      ui.setPanelCollapsed("left", false);
      ui.setPanelCollapsed("right", false);
    }
  };
  apply(mq.matches, true);
  const onChange = (event) => {
    apply(event.matches, true);
    scene?.fit({ instant: true });
  };
  if (mq.addEventListener) mq.addEventListener("change", onChange);
  else mq.addListener(onChange);

  document.getElementById("viewport")?.addEventListener(
    "pointerdown",
    () => {
      if (!state.narrow) return;
      if (!ui.isPanelCollapsed("left")) ui.setPanelCollapsed("left", true);
      if (!ui.isPanelCollapsed("right")) ui.setPanelCollapsed("right", true);
    },
    { capture: true },
  );
}

function togglePanel(side) {
  const collapsed = ui.isPanelCollapsed(side);
  if (collapsed) {
    if (state.narrow) {
      const other = side === "left" ? "right" : "left";
      if (!ui.isPanelCollapsed(other)) ui.setPanelCollapsed(other, true);
    }
    ui.setPanelCollapsed(side, false);
  } else {
    ui.setPanelCollapsed(side, true);
  }
}

/* -------------------------------------------------------------------------- */
/* Debug helpers (used by automated UI checks)                                */
/* -------------------------------------------------------------------------- */

function exposeDebugApi() {
  const api = {
    selectNode: (id) => interaction?.selectNode(id),
    clearSelection: () => interaction?.clearSelection(),
    toggle: (name) => interaction?.toggle(name),
    setToggle: (name, value) => interaction?.setToggle(name, value),
    toggleKind: (kind) => interaction?.toggleKind(kind),
    play: () => interaction?.togglePlay(),
    step: () => interaction?.stepFlow(1),
    fit: () => scene?.fit(),
    reset: () => interaction?.resetView(),
    getState: () => ({
      ready: state.ready,
      source: state.source,
      model: diagnostics.model,
      toggles: interaction?.getToggles(),
      hiddenKinds: [...(interaction?.getHiddenKinds() || [])],
      selected: interaction?.state?.selectedId || null,
      nodes: state.layout?.nodeCount || 0,
    }),
    getGraph: () => state.graph,
    getNodeScreenPosition: (id) => getNodeScreenPosition(id),
    listNodes: () => listNodePositions(),
  };
  Object.assign(diagnostics, api);
}

function getNodeScreenPosition(id) {
  if (!scene || !state.layout) return null;
  const item = state.layout.byId.get(id);
  if (!item) return null;
  const canvas = scene.renderer.domElement;
  const rect = canvas.getBoundingClientRect();
  const point = new Vector3(item.x, 0, 0).project(scene.camera);
  return {
    id,
    x: rect.left + ((point.x + 1) / 2) * rect.width,
    y: rect.top + ((1 - point.y) / 2) * rect.height,
    visible: point.z > -1 && point.z < 1,
  };
}

function listNodePositions() {
  if (!state.layout) return [];
  return state.layout.nodes.map((item) => {
    const position = getNodeScreenPosition(item.id);
    return {
      id: item.id,
      name: item.node.name,
      kind: item.kind,
      order: item.order,
      screen: position,
    };
  });
}

/* -------------------------------------------------------------------------- */

boot();
