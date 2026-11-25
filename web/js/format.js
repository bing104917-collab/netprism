/* Formatting helpers and the shared visual vocabulary (kind → color / label). */

export const KIND_META = {
  input: { label: "输入", color: "#67e8f9" },
  output: { label: "输出", color: "#f0abfc" },
  conv: { label: "卷积", color: "#38bdf8" },
  deconv: { label: "转置卷积", color: "#22d3ee" },
  linear: { label: "线性层", color: "#818cf8" },
  mlp: { label: "多层感知机", color: "#a78bfa" },
  norm: { label: "归一化", color: "#34d399" },
  activation: { label: "激活", color: "#fbbf24" },
  pool: { label: "池化", color: "#2dd4bf" },
  dropout: { label: "Dropout", color: "#94a3b8" },
  flatten: { label: "展平", color: "#8fa3bf" },
  reshape: { label: "变形", color: "#7dd3fc" },
  attention: { label: "注意力", color: "#f472b6" },
  ffn: { label: "前馈网络", color: "#fb923c" },
  embedding: { label: "词嵌入", color: "#c084fc" },
  positional: { label: "位置编码", color: "#e879f9" },
  add: { label: "相加", color: "#fb7185" },
  concat: { label: "拼接", color: "#5eead4" },
  upsample: { label: "上采样", color: "#4ade80" },
  rnn: { label: "循环层", color: "#60a5fa" },
  softmax: { label: "Softmax", color: "#eab308" },
  block: { label: "模块块", color: "#cbd5e1" },
  container: { label: "容器", color: "#94a3b8" },
  custom: { label: "自定义", color: "#9ca3af" },
  unknown: { label: "未知", color: "#64748b" },
};

export const KIND_ORDER = [
  "input", "output", "conv", "deconv", "linear", "mlp", "norm", "activation",
  "pool", "dropout", "flatten", "reshape", "attention", "ffn", "embedding",
  "positional", "add", "concat", "upsample", "rnn", "softmax", "block",
  "container", "custom", "unknown",
];

export const EDGE_META = {
  data: { label: "数据流", color: "#8ea4c8" },
  residual: { label: "残差", color: "#c084fc" },
  skip: { label: "跳连", color: "#fb923c" },
  attention: { label: "注意力", color: "#f472b6" },
  control: { label: "控制", color: "#64748b" },
};

export const GROUP_KIND_META = {
  model: { label: "模型", color: "#67e8f9" },
  sequential: { label: "顺序块", color: "#60a5fa" },
  module: { label: "模块", color: "#94a3b8" },
  conv_block: { label: "卷积块", color: "#38bdf8" },
  residual_block: { label: "残差块", color: "#fb923c" },
  attention_block: { label: "注意力块", color: "#f472b6" },
  transformer_block: { label: "Transformer 块", color: "#f472b6" },
  ffn_block: { label: "前馈块", color: "#fb923c" },
  mlp: { label: "MLP", color: "#a78bfa" },
  encoder: { label: "编码器", color: "#34d399" },
  decoder: { label: "解码器", color: "#4ade80" },
  unet_block: { label: "UNet 块", color: "#2dd4bf" },
  custom: { label: "自定义分组", color: "#9ca3af" },
};

export const MODE_META = {
  trace: { label: "trace 精确", short: "TRACE" },
  static: { label: "static 静态", short: "STATIC" },
  hybrid: { label: "hybrid 混合", short: "HYBRID" },
  onnx: { label: "ONNX 图", short: "ONNX" },
  keras: { label: "Keras", short: "KERAS" },
  torchscript: { label: "TorchScript", short: "TORCHSCRIPT" },
};

export const FRAMEWORK_META = {
  pytorch: { label: "PyTorch", short: "PYTORCH" },
  onnx: { label: "ONNX", short: "ONNX" },
  keras: { label: "Keras", short: "KERAS" },
  torchscript: { label: "TorchScript", short: "TORCHSCRIPT" },
  unknown: { label: "未知格式", short: "UNKNOWN" },
};

const FRAMEWORK_ALIASES = {
  torch: "pytorch",
  pt: "pytorch",
  "torch-script": "torchscript",
  torch_script: "torchscript",
  tf: "keras",
  tensorflow: "keras",
  "keras3": "keras",
  keras_3: "keras",
};

export function kindMeta(kind) {
  return KIND_META[kind] || KIND_META.unknown;
}

export function kindColor(kind) {
  return kindMeta(kind).color;
}

export function kindLabel(kind) {
  return kindMeta(kind).label;
}

export function groupMeta(kind) {
  return GROUP_KIND_META[kind] || GROUP_KIND_META.custom;
}

export function edgeMeta(kind) {
  return EDGE_META[kind] || EDGE_META.data;
}

export function modeMeta(mode) {
  return MODE_META[mode] || MODE_META.static;
}

export function normalizeFramework(framework) {
  if (!framework) return "";
  const key = String(framework).toLowerCase();
  return FRAMEWORK_ALIASES[key] || key;
}

export function frameworkMeta(framework) {
  const normalized = normalizeFramework(framework);
  return FRAMEWORK_META[normalized] || FRAMEWORK_META.unknown;
}

export function escapeHtml(value) {
  return String(value == null ? "" : value).replace(/[&<>"']/g, (ch) => {
    switch (ch) {
      case "&": return "&amp;";
      case "<": return "&lt;";
      case ">": return "&gt;";
      case '"': return "&quot;";
      default: return "&#39;";
    }
  });
}

/** 1234 → 1.2K, 21103252 → 21.1M */
export function formatCount(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return "—";
  if (Math.abs(n) < 1000) return String(Math.round(n));
  const abs = Math.abs(n);
  const units = [
    { base: 1e12, suffix: "T" },
    { base: 1e9, suffix: "G" },
    { base: 1e6, suffix: "M" },
    { base: 1e3, suffix: "K" },
  ];
  for (const { base, suffix } of units) {
    if (abs >= base) {
      const val = n / base;
      const digits = Math.abs(val) >= 100 ? 0 : Math.abs(val) >= 10 ? 1 : 1;
      return `${trimFixed(val, digits)}${suffix}`;
    }
  }
  return String(Math.round(n));
}

function trimFixed(value, digits) {
  const text = value.toFixed(digits);
  return text.replace(/\.0+$/, "").replace(/(\.\d*[1-9])0+$/, "$1");
}

export function formatInteger(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return "—";
  return Math.round(n).toLocaleString("en-US");
}

export function formatFlops(value) {
  const n = Number(value);
  if (!Number.isFinite(n) || n <= 0) return "0";
  return formatCount(n);
}

export function formatParams(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return "—";
  return formatCount(n);
}

export function formatShape(shape) {
  if (!Array.isArray(shape) || shape.length === 0) return "标量";
  return shape.map((v) => (v === -1 ? "?" : String(v))).join(" × ");
}

export function formatShapeCompact(shape) {
  if (!Array.isArray(shape) || shape.length === 0) return "scalar";
  return shape.map((v) => (v === -1 ? "?" : String(v))).join("×");
}

export function formatShapes(shapes) {
  if (!Array.isArray(shapes) || shapes.length === 0) return ["—"];
  return shapes.map(formatShapeCompact);
}

export function shapePills(shapes, out = false) {
  if (!Array.isArray(shapes) || shapes.length === 0) {
    return `<span class="shape-empty">无</span>`;
  }
  return shapes
    .map((shape) => `<span class="shape-pill${out ? " is-out" : ""}">${escapeHtml(formatShapeCompact(shape))}</span>`)
    .join("");
}

const PARAM_LABELS = {
  inChannels: "输入通道",
  outChannels: "输出通道",
  kernelSize: "卷积核",
  stride: "步长",
  padding: "填充",
  dilation: "空洞",
  groups: "分组",
  bias: "偏置",
  inFeatures: "输入特征",
  outFeatures: "输出特征",
  hiddenFeatures: "隐藏维度",
  numFeatures: "特征数",
  normalizedShape: "归一化形状",
  eps: "eps",
  affine: "仿射",
  type: "类型",
  outputSize: "输出尺寸",
  embedDim: "嵌入维度",
  numHeads: "注意力头",
  headDim: "头维度",
  kdim: "K 维度",
  vdim: "V 维度",
  batchFirst: "batch_first",
  dropout: "dropout",
  activation: "激活函数",
  p: "丢弃率",
  startDim: "起始维度",
  numEmbeddings: "词表大小",
  embeddingDim: "嵌入维度",
  inplace: "inplace",
  track_running_stats: "running stats",
  dim: "维度",
  mode: "模式",
  layers: "层数",
  hiddenSize: "隐藏大小",
  numLayers: "层数",
  sourceDim: "源维度",
  targetDim: "目标维度",
};

const DIM_KEYS = new Set(["kernelSize", "stride", "padding", "dilation", "outputSize", "normalizedShape"]);
const BOOL_KEYS = new Set(["bias", "affine", "batchFirst", "inplace", "track_running_stats"]);
const PARAM_ORDER = [
  "type", "kernelSize", "stride", "padding", "dilation", "groups", "inChannels", "outChannels",
  "bias", "inFeatures", "outFeatures", "hiddenFeatures", "numFeatures", "normalizedShape", "eps",
  "affine", "numHeads", "embedDim", "headDim", "kdim", "vdim", "batchFirst", "dropout",
  "numEmbeddings", "embeddingDim", "p", "startDim", "outputSize", "activation", "inplace",
];

export function paramLabel(key) {
  return PARAM_LABELS[key] || key;
}

export function formatParamValue(key, value) {
  if (value === null || value === undefined) return "—";
  if (typeof value === "boolean") return value ? "是" : "否";
  if (Array.isArray(value)) {
    if (value.length === 0) return "[]";
    if (DIM_KEYS.has(key) && value.every((v) => typeof v === "number")) {
      return value.map((v) => (v === -1 ? "?" : v)).join(" × ");
    }
    if (value.every((v) => typeof v === "number" || typeof v === "string")) {
      return `[${value.join(", ")}]`;
    }
    return truncate(JSON.stringify(value), 90);
  }
  if (typeof value === "object") return truncate(JSON.stringify(value), 90);
  if (typeof value === "number") {
    if (Number.isInteger(value)) return String(value);
    if (Math.abs(value) < 1e-3) return String(value);
    return trimFixed(value, 4);
  }
  return String(value);
}

export function paramEntries(params) {
  if (!params || typeof params !== "object") return [];
  const keys = Object.keys(params);
  const known = PARAM_ORDER.filter((k) => keys.includes(k));
  const extra = keys.filter((k) => !PARAM_ORDER.includes(k)).sort();
  return [...known, ...extra].map((key) => ({
    key,
    label: paramLabel(key),
    value: params[key],
    text: formatParamValue(key, params[key]),
  }));
}

export function truncate(text, max = 60) {
  const s = String(text == null ? "" : text);
  return s.length > max ? `${s.slice(0, max - 1)}…` : s;
}

function normalizeDim(dim) {
  if (Array.isArray(dim)) {
    return dim.length === 1 ? [dim[0], dim[0]] : [dim[0], dim[1] ?? dim[0]];
  }
  if (typeof dim === "number") return [dim, dim];
  return null;
}

/** Human readable hyperparameter summary used by labels and detail panels. */
export function opSummary(node) {
  if (!node) return "";
  const p = node.params || {};
  const kind = node.kind;
  const op = node.op || (kind ? kindMeta(kind).label : "—");
  const outShape = node.outShapes && node.outShapes[0];
  const channel = Array.isArray(outShape) && outShape.length >= 2 ? outShape[1] : null;
  const dims = (value) => {
    if (Array.isArray(value)) return value;
    if (typeof value === "number") return [value];
    return null;
  };
  const dimText = (value) => {
    const list = dims(value);
    return list ? list.join("×") : null;
  };
  switch (kind) {
    case "conv":
    case "deconv": {
      const k = dimText(p.kernelSize);
      const s = dimText(p.stride);
      const pad = dimText(p.padding);
      const parts = [op];
      if (k) parts.push(`k${k}`);
      if (p.inChannels != null && p.outChannels != null) parts.push(`${p.inChannels}→${p.outChannels}`);
      if (p.groups != null && Number(p.groups) > 1) parts.push(`g=${p.groups}`);
      if (s && !/^1(×1)*$/.test(s)) parts.push(`s${s}`);
      if (pad && !/^0(×0)*$/.test(pad)) parts.push(`p${pad}`);
      return parts.join(" · ");
    }
    case "linear":
      return p.inFeatures != null && p.outFeatures != null
        ? `${op} · ${p.inFeatures} → ${p.outFeatures}`
        : op;
    case "attention":
    case "mlp": {
      const heads = p.numHeads ?? p.heads;
      const dim = p.embedDim ?? p.hiddenFeatures ?? p.outFeatures;
      const parts = [op];
      if (heads != null) parts.push(`${heads} heads`);
      if (dim != null) parts.push(`d${dim}`);
      return parts.join(" · ");
    }
    case "ffn":
    case "mlp_block": {
      if (p.inFeatures != null && p.hiddenFeatures != null) {
        return `${op} · ${p.inFeatures} → ${p.hiddenFeatures} → ${p.outFeatures ?? p.inFeatures}`;
      }
      break;
    }
    case "norm": {
      if (/GroupNorm/i.test(op)) {
        const groups = p.numGroups ?? p.groups
          ?? (Array.isArray(p.normalizedShape) ? p.normalizedShape[0] : p.normalizedShape);
        const parts = [`${op} · G=${groups ?? "?"}`];
        if (channel != null) parts.push(`C=${channel}`);
        return parts.join(" · ");
      }
      if (/BatchNorm/i.test(op)) {
        return channel != null ? `${op} · C=${channel}` : op;
      }
      if (/LayerNorm/i.test(op)) {
        const normShape = dimText(p.normalizedShape);
        return normShape ? `${op} · ${normShape}` : op;
      }
      if (p.numFeatures != null) return `${op} · C=${p.numFeatures}`;
      if (p.normalizedShape) return `${op} · ${formatParamValue("normalizedShape", p.normalizedShape)}`;
      return op;
    }
    case "pool": {
      const k = dimText(p.kernelSize);
      const parts = [op];
      if (k) parts.push(`k${k}`);
      if (p.outputSize) parts.push(`out ${formatParamValue("outputSize", p.outputSize)}`);
      if (p.type) parts.push(String(p.type));
      return parts.join(" · ");
    }
    case "activation":
    case "dropout":
    case "softmax":
      return op;
    case "embedding":
      return p.numEmbeddings != null ? `${op} · ${p.numEmbeddings} × ${p.embeddingDim ?? "?"}` : op;
  }
  return outShape ? `${op} · ${formatShapeCompact(outShape)}` : op;
}

export function opDetail(node) {
  if (!node) return "";
  const p = node.params || {};
  switch (node.kind) {
    case "conv":
    case "deconv": {
      const k = normalizeDim(p.kernelSize);
      if (k) return `卷积核 ${k[0]}×${k[1]}`;
      break;
    }
    case "pool": {
      if (p.type) return `池化 ${p.type}`;
      break;
    }
    case "attention": {
      if (p.numHeads != null) return `${p.numHeads} heads`;
      break;
    }
    case "norm": {
      if (p.numFeatures != null) return `norm C=${p.numFeatures}`;
      break;
    }
    case "dropout": {
      if (p.p != null) return `p=${p.p}`;
      break;
    }
  }
  return opSummary(node);
}

export function groupShapeSummary(shapes) {
  if (!Array.isArray(shapes) || !shapes.length) return "—";
  return shapes.slice(0, 2).map(formatShapeCompact).join(" / ");
}

export function clamp(v, min, max) {
  return Math.min(max, Math.max(min, v));
}

export function lerp(a, b, t) {
  return a + (b - a) * t;
}

export function easeInOutCubic(t) {
  return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
}

export function easeOutCubic(t) {
  return 1 - Math.pow(1 - t, 3);
}

export function formatBytes(bytes) {
  const n = Number(bytes) || 0;
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}

/** Optional v2 metric fields must never become fake zeroes. */
export function isMetricNumber(value) {
  return typeof value === "number" && Number.isFinite(value) && value >= 0;
}

/** activationBytes → human readable MB (always MB, per v2 UI contract). */
export function formatActivationMB(bytes) {
  if (!isMetricNumber(bytes)) return "—";
  const mb = bytes / (1024 * 1024);
  if (mb === 0) return "0 MB";
  if (mb >= 1000) return `${Math.round(mb).toLocaleString("en-US")} MB`;
  if (mb >= 100) return `${Math.round(mb)} MB`;
  if (mb >= 10) return `${mb.toFixed(1)} MB`;
  if (mb >= 1) return `${mb.toFixed(2)} MB`;
  return `${mb.toFixed(3)} MB`;
}

/** Raw MB number for CSV, or "" when unavailable. */
export function activationMBValue(bytes) {
  if (!isMetricNumber(bytes)) return "";
  const value = (bytes / (1024 * 1024)).toFixed(6).replace(/\.?0+$/, "");
  return value === "" ? "0" : value;
}

export function formatReceptiveField(value) {
  if (!isMetricNumber(value)) return "—";
  return Math.round(value).toLocaleString("en-US");
}

export function formatMacs(value) {
  if (!isMetricNumber(value)) return "—";
  return formatCount(value);
}

export function modelMetricValue(metrics, key) {
  if (!metrics || typeof metrics !== "object") return null;
  const value = metrics[key];
  return isMetricNumber(value) ? value : null;
}

export function nodeMetricValue(node, key) {
  if (!node || !node.metrics || typeof node.metrics !== "object") return null;
  const value = node.metrics[key];
  return isMetricNumber(value) ? value : null;
}
