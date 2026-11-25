/* Deterministic 3D layout: execution order on X, tensor geometry on Y/Z. */

import { clamp, kindColor } from "./format.js";

export const KERNEL_LEFT_PAD = 1.15;

const SEQ_KINDS = new Set([
  "attention", "ffn", "mlp", "embedding", "positional", "rnn", "softmax",
  "block", "container", "custom", "unknown",
]);

const THIN_KINDS = new Set([
  "norm", "activation", "dropout", "softmax", "pool", "upsample", "flatten",
  "reshape", "add", "concat", "positional",
]);
export { THIN_KINDS };

const IMPORTANT_KINDS = new Set([
  "input", "output", "conv", "deconv", "linear", "attention", "ffn",
  "embedding", "rnn", "block", "upsample", "mlp",
]);

const MAX_DIM = 4.35;

function safeNum(value, fallback = 0) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function validShape(shape) {
  return Array.isArray(shape) && shape.length > 0 && shape.every((v) => Number.isFinite(Number(v)));
}

function normalizeKernelSize(params) {
  const raw = params && params.kernelSize;
  if (Array.isArray(raw) && raw.length) {
    const nums = raw.map((v) => safeNum(v, 1)).filter((v) => v > 0);
    if (nums.length >= 2) return [nums[0], nums[1]];
    if (nums.length === 1) return [nums[0], nums[0]];
  }
  if (typeof raw === "number" && raw > 0) return [raw, raw];
  return null;
}

export function hasKernels(node) {
  if (!node) return false;
  if (node.kind !== "conv" && node.kind !== "deconv") return false;
  return normalizeKernelSize(node.params) != null || /conv/i.test(String(node.op || ""));
}

export function kernelDims(node) {
  const params = (node && node.params) || {};
  return normalizeKernelSize(params) || [3, 3];
}

/** Derive a compact geometry descriptor from a tensor shape. */
export function analyzeShape(node) {
  const out = Array.isArray(node.outShapes) && node.outShapes.length ? node.outShapes[0] : null;
  const inp = Array.isArray(node.inShapes) && node.inShapes.length ? node.inShapes[0] : null;
  const shape = validShape(out) ? out.map(Number) : validShape(inp) ? inp.map(Number) : null;
  const tags = Array.isArray(node.tags) ? node.tags : [];
  const kind = node.kind || "unknown";

  if (!shape) {
    return { family: "scalar", rank: 0, dims: [], shape: null };
  }

  const rank = shape.length;
  const isSeqTag = tags.includes("sequence") || tags.includes("token") || tags.includes("head");

  if (rank >= 5) {
    // (N, C, D, H, W) → H / W faces, C thickness.
    return {
      family: "4d",
      rank,
      c: Math.max(1, shape[1] || 1),
      h: Math.max(1, shape[3] || 1),
      w: Math.max(1, shape[4] || 1),
      shape,
    };
  }
  if (rank === 4) {
    return {
      family: "4d",
      rank,
      c: Math.max(1, shape[1] || 1),
      h: Math.max(1, shape[2] || 1),
      w: Math.max(1, shape[3] || 1),
      shape,
    };
  }
  if (rank === 3) {
    // Almost all 3D tensors in PyTorch models are (N, L, D) sequence × feature.
    // Conv1d / pooling keep channel-major semantics and are drawn as channels.
    const channelKinds = new Set(["conv", "deconv", "pool", "upsample"]);
    const seq = !channelKinds.has(kind) || isSeqTag;
    if (seq) {
      return {
        family: "seq",
        rank,
        l: Math.max(1, shape[1] || 1),
        d: Math.max(1, shape[2] || 1),
        shape,
      };
    }
    return {
      family: "channel_seq",
      rank,
      c: Math.max(1, shape[1] || 1),
      l: Math.max(1, shape[2] || 1),
      shape,
    };
  }
  if (rank === 2) {
    return {
      family: "feature",
      rank,
      f: Math.max(1, shape[1] || 1),
      shape,
    };
  }
  return {
    family: "scalar",
    rank,
    f: Math.max(1, shape[0] || 1),
    shape,
  };
}

function topologyOrder(nodes, edges = []) {
  const indexById = new Map(nodes.map((node, index) => [node.id, index]));
  const adjacency = new Map(nodes.map((node) => [node.id, []]));
  const indegree = new Map(nodes.map((node) => [node.id, 0]));
  for (const edge of edges) {
    if (!edge || !indexById.has(edge.source) || !indexById.has(edge.target)) continue;
    if (edge.source === edge.target) continue;
    adjacency.get(edge.source).push(edge.target);
    indegree.set(edge.target, indegree.get(edge.target) + 1);
  }
  const ready = nodes
    .filter((node) => indegree.get(node.id) === 0)
    .map((node) => node.id)
    .sort((a, b) => indexById.get(a) - indexById.get(b));
  const ordered = [];
  while (ready.length) {
    const id = ready.shift();
    ordered.push(id);
    const targets = adjacency.get(id) || [];
    let inserted = false;
    for (const target of targets) {
      const next = indegree.get(target) - 1;
      indegree.set(target, next);
      if (next === 0) {
        const targetIndex = indexById.get(target);
        let at = ready.length;
        for (let i = 0; i < ready.length; i += 1) {
          if (indexById.get(ready[i]) > targetIndex) {
            at = i;
            break;
          }
        }
        ready.splice(at, 0, target);
        inserted = true;
      }
    }
    if (!inserted) {
      // keep deterministic order even when the queue is not sorted
      ready.sort((a, b) => indexById.get(a) - indexById.get(b));
    }
  }
  if (ordered.length !== nodes.length) {
    const seen = new Set(ordered);
    for (const node of nodes) if (!seen.has(node.id)) ordered.push(node.id);
  }
  return ordered.map((id) => nodes[indexById.get(id)]);
}

function emptyBounds() {
  return {
    minX: -4, maxX: 4, minY: -3, maxY: 3, minZ: -3, maxZ: 3,
    centerX: 0, centerY: 0, centerZ: 0, spanX: 8, spanY: 6, spanZ: 6,
  };
}

function computeBounds(nodes, fallbackPadding = 0.9) {
  if (!nodes.length) return emptyBounds();
  let minX = Infinity, maxX = -Infinity;
  let minY = Infinity, maxY = -Infinity;
  let minZ = Infinity, maxZ = -Infinity;
  for (const item of nodes) {
    minX = Math.min(minX, item.bounds.minX);
    maxX = Math.max(maxX, item.bounds.maxX);
    minY = Math.min(minY, item.bounds.minY);
    maxY = Math.max(maxY, item.bounds.maxY);
    minZ = Math.min(minZ, item.bounds.minZ);
    maxZ = Math.max(maxZ, item.bounds.maxZ);
  }
  minX -= fallbackPadding;
  maxX += fallbackPadding;
  minY -= fallbackPadding;
  maxY += fallbackPadding;
  minZ -= fallbackPadding;
  maxZ += fallbackPadding;
  return {
    minX, maxX, minY, maxY, minZ, maxZ,
    centerX: (minX + maxX) / 2,
    centerY: (minY + maxY) / 2,
    centerZ: (minZ + maxZ) / 2,
    spanX: maxX - minX,
    spanY: maxY - minY,
    spanZ: maxZ - minZ,
  };
}

/**
 * @param {object} graph NetPrism graph (contract v1.0)
 * @returns layout model consumed by the Three.js builders
 */
export function computeLayout(graph) {
  const rawNodes = Array.isArray(graph.nodes) ? graph.nodes.filter((n) => n && n.id) : [];
  const rawEdges = Array.isArray(graph.edges) ? graph.edges : [];
  const rawGroups = Array.isArray(graph.groups) ? graph.groups.filter((g) => g && g.id) : [];
  const nodesById = new Map(rawNodes.map((node) => [node.id, node]));
  const groupsById = new Map(rawGroups.map((group) => [group.id, group]));
  const orderedNodes = topologyOrder(rawNodes, rawEdges);

  const analyses = orderedNodes.map((node) => analyzeShape(node));
  let maxH = 1, maxW = 1, maxL = 1, maxD = 1, maxF = 1, maxC = 1;
  for (const a of analyses) {
    if (a.family === "4d") {
      maxH = Math.max(maxH, a.h);
      maxW = Math.max(maxW, a.w);
      maxC = Math.max(maxC, a.c);
    } else if (a.family === "seq") {
      maxL = Math.max(maxL, a.l);
      maxD = Math.max(maxD, a.d);
    } else if (a.family === "channel_seq") {
      maxL = Math.max(maxL, a.l);
      maxC = Math.max(maxC, a.c);
    } else if (a.family === "feature") {
      maxF = Math.max(maxF, a.f);
    } else if (a.family === "scalar") {
      maxF = Math.max(maxF, 1);
    }
  }

  const log2 = (v) => Math.log2(Math.max(1, v) + 1);
  const maxCLog = log2(maxC);
  const maxFLog = log2(maxF);
  const maxDLog = log2(maxD);

  const nodeItems = orderedNodes.map((node, order) => {
    const analysis = analyses[order];
    const kind = node.kind || "unknown";
    const params = node.params && typeof node.params === "object" ? node.params : {};
    let sx = 0.34;
    let sy = 2.3;
    let sz = 0.9;

    if (analysis.family === "4d") {
      sy = clamp(0.36 + MAX_DIM * (analysis.h / maxH), 0.36, MAX_DIM);
      sz = clamp(0.36 + MAX_DIM * (analysis.w / maxW), 0.36, MAX_DIM);
      sx = clamp(0.28 + 0.38 * log2(analysis.c), 0.34, 2.75);
    } else if (analysis.family === "seq") {
      sy = clamp(0.48 + 3.5 * (analysis.l / maxL), 0.48, 4);
      sz = clamp(0.46 + 2.3 * (log2(analysis.d) / Math.max(1, maxDLog)), 0.46, 3.2);
      sx = 0.26;
    } else if (analysis.family === "channel_seq") {
      sy = clamp(0.42 + 3.2 * (analysis.l / maxL), 0.42, 3.8);
      sz = 0.36;
      sx = clamp(0.24 + 0.34 * log2(analysis.c), 0.28, 2.2);
    } else if (analysis.family === "feature") {
      sy = 2.25;
      sz = clamp(0.55 + 3.5 * (log2(analysis.f) / Math.max(1, maxFLog)), 0.55, 4.1);
      sx = 0.3;
    } else {
      sy = 0.5;
      sz = 0.5;
      sx = 0.5;
    }

    if (kind === "attention") {
      const heads = clamp(Math.round(safeNum(params.numHeads ?? params.heads, 4)), 1, 16);
      sx = clamp(0.6 + heads * 0.055, 0.85, 1.9);
      sy = Math.max(sy, 1.6);
      sz = Math.max(sz, 0.9);
    } else if (kind === "ffn") {
      sx = 1.45;
    } else if (kind === "mlp") {
      sx = 0.42;
    } else if (kind === "input" || kind === "output") {
      sx = Math.max(sx, 0.42);
    } else if (kind === "add" || kind === "concat") {
      sx = 0.34;
    } else if (THIN_KINDS.has(kind)) {
      sx = analysis.family === "seq" ? 0.2 : 0.19;
      if (kind === "pool" || kind === "upsample") sx = 0.22;
      if (kind === "flatten" || kind === "reshape") sx = 0.18;
    } else if (kind === "linear") {
      sx = 0.3;
    } else if (kind === "deconv") {
      sx = clamp(sx * 0.92, 0.34, 2.4);
    }

    const kernels = hasKernels(node);
    const leftPad = kernels ? KERNEL_LEFT_PAD : 0;
    return {
      id: node.id,
      node,
      kind,
      order,
      analysis,
      params,
      hasKernels: kernels,
      kernelSize: kernels ? kernelDims(node) : null,
      color: kindColor(kind),
      groupId: node.groupId || null,
      x: 0,
      y: 0,
      z: 0,
      sx,
      sy,
      sz,
      leftPad,
      bounds: null,
      isBoundary: kind === "input" || kind === "output",
    };
  });

  // --- X placement: execution order, tighter inside groups, wider between ---
  let cursor = 0;
  for (let i = 0; i < nodeItems.length; i += 1) {
    const item = nodeItems[i];
    const prev = nodeItems[i - 1];
    let gap = 1.0;
    if (prev) {
      const groupChanged = prev.groupId !== item.groupId;
      gap = groupChanged ? 4.0 : 0.76;
      if (item.hasKernels) gap = Math.max(gap, 1.58);
      if (prev.hasKernels) gap = Math.max(gap, 1.48);
      if (item.isBoundary || prev.isBoundary) gap = Math.max(gap, 1.65);
      if (prev.sx > 1.5 || item.sx > 1.5) gap += 0.15;
      if (item.kind === "attention" || item.kind === "ffn") gap += 0.08;
    }
    cursor += i === 0 ? 0 : gap;
    item.x = cursor + item.sx / 2;
    cursor += item.sx;
    item.bounds = {
      minX: item.x - item.sx / 2 - item.leftPad,
      maxX: item.x + item.sx / 2,
      minY: -item.sy / 2,
      maxY: item.sy / 2,
      minZ: -item.sz / 2,
      maxZ: item.sz / 2,
    };
  }

  const nodeById = new Map(nodeItems.map((item) => [item.node.id, item]));
  const overall = computeBounds(nodeItems, 1.0);

  // --- groups ---
  const childrenByParent = new Map();
  for (const group of rawGroups) {
    const parentId = group.parentId && groupsById.has(group.parentId) ? group.parentId : null;
    if (!childrenByParent.has(parentId)) childrenByParent.set(parentId, []);
    childrenByParent.get(parentId).push(group);
  }
  const directNodesByGroup = new Map();
  for (const item of nodeItems) {
    const gid = item.groupId && groupsById.has(item.groupId) ? item.groupId : null;
    if (!gid) continue;
    if (!directNodesByGroup.has(gid)) directNodesByGroup.set(gid, []);
    directNodesByGroup.get(gid).push(item);
  }

  const descendantCache = new Map();
  const visiting = new Set();
  function descendantNodes(group) {
    if (descendantCache.has(group.id)) return descendantCache.get(group.id);
    if (visiting.has(group.id)) return [];
    visiting.add(group.id);
    const result = [...(directNodesByGroup.get(group.id) || [])];
    const childIds = new Set([
      ...(Array.isArray(group.childGroupIds) ? group.childGroupIds : []),
      ...(childrenByParent.get(group.id) || []).map((g) => g.id),
    ]);
    for (const childId of childIds) {
      const child = groupsById.get(childId);
      if (child && child.id !== group.id) result.push(...descendantNodes(child));
    }
    visiting.delete(group.id);
    const unique = [...new Map(result.map((item) => [item.id, item])).values()];
    descendantCache.set(group.id, unique);
    return unique;
  }

  const groupItems = rawGroups.map((group) => {
    const direct = directNodesByGroup.get(group.id) || [];
    const descendants = descendantNodes(group);
    const orders = descendants.map((item) => item.order).sort((a, b) => a - b);
    const firstOrder = orders.length ? orders[0] : Number.MAX_SAFE_INTEGER;
    const lastOrder = orders.length ? orders[orders.length - 1] : Number.MAX_SAFE_INTEGER;
    const contiguous = descendants.length > 0 && lastOrder - firstOrder + 1 === descendants.length;
    const childGroups = (childrenByParent.get(group.id) || []).filter(
      (child) => descendantNodes(child).length > 0,
    );
    const groupBounds = computeBounds(descendants, 0.62);
    const directBounds = direct.length ? computeBounds(direct, 0.55) : null;
    const isRoot = !group.parentId || !groupsById.has(group.parentId);
    return {
      group,
      depth: safeNum(group.depth, 0),
      isRoot,
      directNodes: direct,
      descendantNodes: descendants,
      descendantCount: descendants.length,
      firstOrder,
      lastOrder,
      contiguous,
      hasChildren: childGroups.length > 0,
      childGroupIds: childGroups.map((child) => child.id),
      bounds: groupBounds,
      directBounds,
      hasDirect: direct.length > 0,
    };
  });

  const groupOrder = new Map(groupItems.map((item) => [item.group.id, item.firstOrder]));

  // Group labels are often repeated ("Conv Block", "Attention"). The backend
  // is adding unique labels, but the frontend must degrade gracefully: append
  // the module path when two displayed groups share the same label.
  const labelCounts = new Map();
  for (const groupItem of groupItems) {
    if (groupItem.isRoot || groupItem.descendantCount === 0) continue;
    const label = groupItem.group.label || groupItem.group.name || groupItem.group.id;
    labelCounts.set(label, (labelCounts.get(label) || 0) + 1);
  }
  const displayCounts = new Map();
  for (const groupItem of groupItems) {
    const group = groupItem.group;
    const label = group.label || group.name || group.id;
    let display = label;
    if ((labelCounts.get(label) || 0) > 1) {
      const suffix = group.name && group.name !== label ? group.name : group.id;
      display = `${label} · ${suffix}`;
    }
    displayCounts.set(display, (displayCounts.get(display) || 0) + 1);
    groupItem.displayLabel = display;
  }
  for (const groupItem of groupItems) {
    if ((displayCounts.get(groupItem.displayLabel) || 0) > 1) {
      groupItem.displayLabel = `${groupItem.displayLabel} · ${groupItem.group.id}`;
    }
    const rawLabel = String(
      groupItem.group.label || groupItem.group.name || groupItem.group.id,
    );
    const segments = rawLabel.split("·").map((part) => part.trim()).filter(Boolean);
    groupItem.shortLabel =
      segments.length > 1 ? segments[segments.length - 1] : groupItem.displayLabel;
  }

  // --- default labels: only meaningful anchors, never a wall of text ---
  const keyLabelIds = new Set();
  for (const groupItem of groupItems) {
    if (!groupItem.directNodes.length) continue;
    const direct = [...groupItem.directNodes].sort((a, b) => a.order - b.order);
    keyLabelIds.add(direct[0].id);
    let maxParams = direct[0];
    for (const item of direct) {
      if (Number(item.node.numParams) > Number(maxParams.node.numParams)) maxParams = item;
    }
    if (maxParams.id !== direct[0].id) keyLabelIds.add(maxParams.id);
  }
  for (const item of nodeItems) {
    if (item.isBoundary) {
      keyLabelIds.add(item.id);
      continue;
    }
    const heads = Number(item.params.numHeads ?? item.params.heads);
    if ((item.kind === "attention" && Number.isFinite(heads)) || item.kind === "ffn") {
      keyLabelIds.add(item.id);
    }
  }
  const labelCap =
    nodeItems.length <= 16
      ? 999
      : nodeItems.length <= 30
        ? 10
        : nodeItems.length <= 100
          ? 8
          : 6;
  if (keyLabelIds.size > labelCap) {
    const priority = new Map();
    for (const id of keyLabelIds) {
      const item = nodeById.get(id);
      priority.set(id, (item.isBoundary ? 100 : 0) + (item.order === 0 ? 50 : 0));
    }
    const ordered = [...keyLabelIds].sort((a, b) => (priority.get(b) || 0) - (priority.get(a) || 0));
    keyLabelIds.clear();
    for (const id of ordered.slice(0, labelCap)) keyLabelIds.add(id);
  }
  for (const item of nodeItems) {
    item.labelKey = keyLabelIds.has(item.id);
    item.labelDefault =
      nodeItems.length <= 16
        ? true
        : nodeItems.length > 250
          ? item.isBoundary
          : item.labelKey;
  }
  if (nodeItems.length > 16) {
    let rank = 0;
    for (const item of nodeItems) {
      item.labelLift = item.labelDefault && !item.isBoundary ? (rank++ % 3) * 0.3 : 0;
    }
  }
  for (const item of nodeItems) {
    if (!Number.isFinite(item.labelLift)) item.labelLift = 0;
  }

  return {
    graph,
    nodes: nodeItems,
    byId: nodeById,
    ordered: nodeItems,
    groups: groupItems,
    groupsById,
    groupOrder,
    bounds: overall,
    nodeCount: nodeItems.length,
    edgeCount: rawEdges.length,
    big: nodeItems.length > 250,
    huge: nodeItems.length > 500,
    maxima: { maxH, maxW, maxL, maxD, maxF, maxC },
    nodeByIdRaw: nodesById,
    groupIdOf(groupId) {
      return groupsById.get(groupId) || null;
    },
  };
}

/** Default label policy before the user toggles labels. */
export function defaultLabelVisible(item, layout) {
  if (item && typeof item.labelDefault === "boolean") return item.labelDefault;
  if (!layout || layout.big) {
    return item.kind === "input" || item.kind === "output";
  }
  if (layout.nodeCount > 60) {
    return IMPORTANT_KINDS.has(item.kind);
  }
  return true;
}

/** Every label shown when the user clicks “显示全部标签”. */
export function allLabelsVisible() {
  return true;
}

export { topologyOrder };
