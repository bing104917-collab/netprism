/* Node meshes: tensors, convolution filter banks, attention head stacks, FFN slabs. */

import * as THREE from "three";
import { createLabel } from "./labels.js";
import {
  kindColor, kindLabel, opSummary, escapeHtml, clamp, truncate, formatParamValue,
  groupShapeSummary,
} from "./format.js";
import { THIN_KINDS, defaultLabelVisible } from "./layout.js";

const TMP_COLOR = new THREE.Color();
const TMP_WHITE = new THREE.Color(0xffffff);
const TMP_VIOLET = new THREE.Color(0xa78bfa);

function brighten(color, amount = 0.25) {
  return new THREE.Color(color).lerp(TMP_WHITE, amount);
}

function gradientBoxGeometry(w, h, d, segments, color) {
  const geometry = new THREE.BoxGeometry(w, h, d, segments, segments, segments);
  const position = geometry.getAttribute("position");
  const colors = new Float32Array(position.count * 3);
  const base = new THREE.Color(color);
  for (let i = 0; i < position.count; i += 1) {
    const y = position.getY(i);
    const t = clamp(y / Math.max(h, 0.001) + 0.5, 0, 1);
    const shade = 0.38 + t * 1.02;
    TMP_COLOR.copy(base).multiplyScalar(shade);
    TMP_COLOR.lerp(TMP_WHITE, 0.03 + t * 0.06);
    colors[i * 3] = TMP_COLOR.r;
    colors[i * 3 + 1] = TMP_COLOR.g;
    colors[i * 3 + 2] = TMP_COLOR.b;
  }
  geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  return geometry;
}

function trackMaterials(materials, material) {
  materials.push(material);
  return material;
}

function meshWithPick(group, geometry, material, pickId, meshes, materials, shadow) {
  const mesh = new THREE.Mesh(geometry, material);
  mesh.castShadow = Boolean(shadow);
  mesh.receiveShadow = false;
  if (pickId) {
    mesh.userData.pick = { type: "node", id: pickId };
    meshes.push(mesh);
  }
  group.add(mesh);
  materials.push(material);
  return mesh;
}

function addEdges(group, geometry, color, opacity, materials) {
  const lineMaterial = new THREE.LineBasicMaterial({
    color,
    transparent: true,
    opacity,
    depthWrite: false,
  });
  const lines = new THREE.LineSegments(new THREE.EdgesGeometry(geometry, 28), lineMaterial);
  group.add(lines);
  materials.push(lineMaterial);
  return lines;
}

function buildTensorVisual(item, group, materials, meshes, options = {}) {
  const pointwise = THIN_KINDS.has(item.kind);
  const {
    big = false,
    opacity = pointwise ? 0.72 : 0.62,
    segments = big ? 1 : 2,
    addInner = true,
    edgesOpacity = pointwise ? 0.8 : 0.66,
    shadow = true,
    emissive = pointwise ? 0.36 : 0.28,
  } = options;
  const geometry = gradientBoxGeometry(item.sx, item.sy, item.sz, segments, item.color);
  const material = new THREE.MeshPhysicalMaterial({
    color: 0xffffff,
    vertexColors: true,
    transparent: true,
    opacity,
    roughness: 0.26,
    metalness: 0.2,
    clearcoat: big ? 0 : 0.55,
    clearcoatRoughness: 0.4,
    emissive: new THREE.Color(item.color),
    emissiveIntensity: emissive,
    depthWrite: false,
    side: THREE.FrontSide,
  });
  const mesh = meshWithPick(group, geometry, material, item.id, meshes, materials, shadow);
  mesh.renderOrder = 2;
  const edgeColor = brighten(item.color, 0.58);
  addEdges(group, geometry, edgeColor, edgesOpacity, materials);

  if (addInner && !big) {
    const innerGeometry = new THREE.BoxGeometry(item.sx * 0.62, item.sy * 0.62, item.sz * 0.62);
    const innerMaterial = new THREE.MeshBasicMaterial({
      color: item.color,
      transparent: true,
      opacity: 0.055,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      side: THREE.BackSide,
    });
    const inner = new THREE.Mesh(innerGeometry, innerMaterial);
    inner.renderOrder = 1;
    group.add(inner);
    materials.push(innerMaterial);
  }

  // A thin luminous rim on the front face gives the prism its signature glow.
  const rimGeometry = new THREE.PlaneGeometry(item.sx * 0.94, item.sy * 0.94);
  const rimMaterial = new THREE.MeshBasicMaterial({
    color: brighten(item.color, 0.5),
    transparent: true,
    opacity: 0.06,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
    side: THREE.DoubleSide,
  });
  const rim = new THREE.Mesh(rimGeometry, rimMaterial);
  rim.position.set(0, 0, item.sz / 2 + 0.004);
  rim.renderOrder = 3;
  group.add(rim);
  materials.push(rimMaterial);
  return mesh;
}

function buildAttentionVisual(item, group, materials, meshes, big) {
  const heads = clamp(Math.round(Number(item.params.numHeads ?? item.params.heads ?? 4)) || 4, 1, 16);
  const baseMaterial = new THREE.MeshPhysicalMaterial({
    color: item.color,
    transparent: true,
    opacity: 0.12,
    roughness: 0.35,
    metalness: 0.2,
    emissive: new THREE.Color(item.color),
    emissiveIntensity: 0.3,
    depthWrite: false,
    side: THREE.DoubleSide,
  });
  const base = new THREE.Mesh(
    new THREE.BoxGeometry(item.sx, item.sy, item.sz),
    baseMaterial,
  );
  base.userData.pick = { type: "node", id: item.id };
  base.castShadow = !big && false;
  meshes.push(base);
  materials.push(baseMaterial);
  group.add(base);
  addEdges(group, base.geometry, brighten(item.color, 0.4), 0.4, materials);

  const drawHeads = big ? Math.min(heads, 6) : heads;
  const sliceThickness = 0.052;
  const span = Math.max(item.sx * 0.78, drawHeads * sliceThickness * 1.8);
  for (let i = 0; i < drawHeads; i += 1) {
    const t = drawHeads === 1 ? 0.5 : i / (drawHeads - 1);
    const color = new THREE.Color(item.color).lerp(TMP_VIOLET, t * 0.62).lerp(TMP_WHITE, 0.06);
    const material = new THREE.MeshPhysicalMaterial({
      color,
      transparent: true,
      opacity: 0.5,
      roughness: 0.3,
      metalness: 0.22,
      clearcoat: 0.5,
      emissive: color,
      emissiveIntensity: 0.25,
      depthWrite: false,
      side: THREE.DoubleSide,
    });
    const geometry = new THREE.BoxGeometry(sliceThickness, item.sy * 0.9, item.sz * 0.88);
    const slice = new THREE.Mesh(geometry, material);
    slice.position.x = (t - 0.5) * span;
    slice.position.z = (t - 0.5) * 0.06;
    slice.rotation.y = (t - 0.5) * 0.08;
    slice.userData.pick = { type: "node", id: item.id };
    slice.renderOrder = 3;
    group.add(slice);
    meshes.push(slice);
    materials.push(material);
    if (!big && i % 2 === 0) {
      const edgeMaterial = new THREE.LineBasicMaterial({
        color: brighten(color, 0.5),
        transparent: true,
        opacity: 0.42,
        depthWrite: false,
      });
      const lines = new THREE.LineSegments(new THREE.EdgesGeometry(geometry), edgeMaterial);
      lines.position.copy(slice.position);
      lines.rotation.copy(slice.rotation);
      group.add(lines);
      materials.push(edgeMaterial);
    }
  }
  return { heads };
}

function buildFfnVisual(item, group, materials, meshes, big) {
  const analysis = item.analysis || {};
  const outDim = analysis.d || analysis.f || item.sz || 64;
  const hidden = Number(item.params.hiddenFeatures ?? item.params.hidden ?? item.params.hiddenSize);
  const ratio = clamp((Number.isFinite(hidden) && hidden > 0 ? hidden : outDim * 4) / Math.max(outDim, 1), 1.15, 3.2);
  const zExpand = clamp(item.sz * Math.min(ratio, 2.5), 0.5, 4.5);
  const slabX = 0.24;
  const gap = 0.34;

  const makeSlab = (x, z, heightScale, opacity) => {
    const geometry = gradientBoxGeometry(slabX, item.sy * heightScale, z, big ? 1 : 2, item.color);
    const material = new THREE.MeshPhysicalMaterial({
      color: 0xffffff,
      vertexColors: true,
      transparent: true,
      opacity,
      roughness: 0.3,
      metalness: 0.2,
      clearcoat: big ? 0 : 0.5,
      emissive: new THREE.Color(item.color),
      emissiveIntensity: 0.24,
      depthWrite: false,
      side: THREE.FrontSide,
    });
    const mesh = meshWithPick(group, geometry, material, item.id, meshes, materials, !big);
    mesh.position.x = x;
    mesh.renderOrder = 3;
    addEdges(group, geometry, brighten(item.color, 0.45), 0.55, materials).position.x = x;
    return mesh;
  };

  makeSlab(-gap - slabX / 2, zExpand, 0.86, 0.5);
  makeSlab(gap + slabX / 2, item.sz, 1, 0.56);

  // Glowing bridge between the expand / contract halves.
  const bridgeGeometry = new THREE.BoxGeometry(gap * 2 + 0.2, item.sy * 0.16, Math.max(item.sz * 0.26, 0.2));
  const bridgeMaterial = new THREE.MeshBasicMaterial({
    color: brighten(item.color, 0.25),
    transparent: true,
    opacity: 0.07,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
    side: THREE.DoubleSide,
  });
  group.add(new THREE.Mesh(bridgeGeometry, bridgeMaterial));
  materials.push(bridgeMaterial);

  const beamMaterial = new THREE.LineBasicMaterial({
    color: brighten(item.color, 0.35),
    transparent: true,
    opacity: 0.32,
    depthWrite: false,
  });
  const beamPoints = [];
  for (const yFactor of [-0.32, 0, 0.32]) {
    beamPoints.push(
      new THREE.Vector3(-gap, item.sy * yFactor * 0.8, 0),
      new THREE.Vector3(gap, item.sy * yFactor, 0),
    );
  }
  const beamGeometry = new THREE.BufferGeometry().setFromPoints(beamPoints);
  group.add(new THREE.LineSegments(beamGeometry, beamMaterial));
  materials.push(beamMaterial);
  return { heads: 0, ratio, zExpand };
}

function addTokenDots(item, group, materials) {
  const count = clamp(Math.round(item.sy / 0.42), 4, 12);
  const material = new THREE.MeshBasicMaterial({
    color: brighten(item.color, 0.45),
    transparent: true,
    opacity: 0.85,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
  });
  const geometry = new THREE.BoxGeometry(0.065, 0.045, 0.028);
  for (let i = 0; i < count; i += 1) {
    const dot = new THREE.Mesh(geometry, material);
    const t = count === 1 ? 0.5 : i / (count - 1);
    dot.position.set(0, (t - 0.5) * item.sy * 0.72, item.sz / 2 + 0.035);
    group.add(dot);
  }
  materials.push(material);
}

function addBoundaryRing(item, group, materials) {
  const radius = clamp(Math.max(item.sy, item.sz) * 0.64, 0.7, 3.4);
  const material = new THREE.MeshBasicMaterial({
    color: item.color,
    transparent: true,
    opacity: 0.32,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
    side: THREE.DoubleSide,
  });
  const geometry = new THREE.RingGeometry(radius * 0.88, radius, 80);
  const ring = new THREE.Mesh(geometry, material);
  ring.rotation.x = -Math.PI / 2;
  ring.position.y = -item.sy / 2 - 0.6;
  group.add(ring);
  materials.push(material);

  const haloMaterial = new THREE.MeshBasicMaterial({
    color: item.color,
    transparent: true,
    opacity: 0.06,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
    side: THREE.DoubleSide,
  });
  const halo = new THREE.Mesh(new THREE.CircleGeometry(radius * 2.1, 64), haloMaterial);
  halo.rotation.x = -Math.PI / 2;
  halo.position.y = -item.sy / 2 - 0.62;
  group.add(halo);
  materials.push(haloMaterial);
}

function nodeLabelHtml(item) {
  const summary = truncate(opSummary(item.node), 34);
  return (
    `<span class="nl-dot"></span>` +
    `<span class="nl-name">${escapeHtml(item.node.name || item.id)}</span>` +
    (summary ? `<span class="nl-sub">${escapeHtml(summary)}</span>` : "")
  );
}

function makeNodeGroup(item, layout, options) {
  const group = new THREE.Group();
  group.name = `node:${item.id}`;
  group.position.set(item.x, 0, 0);
  group.userData.nodeId = item.id;
  const materials = [];
  const meshes = [];
  const big = layout.big;
  let visualTop = item.sy / 2;
  let extra = null;

  if (item.kind === "attention") {
    extra = buildAttentionVisual(item, group, materials, meshes, big);
    visualTop = item.sy / 2;
  } else if (item.kind === "ffn") {
    extra = buildFfnVisual(item, group, materials, meshes, big);
    visualTop = item.sy / 2;
  } else {
    buildTensorVisual(item, group, materials, meshes, {
      big,
      opacity: item.isBoundary ? 0.72 : undefined,
      addInner: !big,
      edgesOpacity: item.isBoundary ? 0.85 : undefined,
      shadow: true,
    });
  }

  if (item.kind === "embedding" || item.kind === "positional") {
    addTokenDots(item, group, materials);
  }
  if (item.isBoundary) {
    addBoundaryRing(item, group, materials);
  }

  const label = createLabel("node-label", nodeLabelHtml(item), {
    x: 0,
    y: visualTop + (item.kind === "ffn" ? 0.75 : 0.5) + (item.labelLift || 0),
    z: 0,
  });
  label.element.style.setProperty("--c", item.color);
  label.element.dataset.nodeId = item.id;
  group.add(label);

  const labelDefault = defaultLabelVisible(item, layout);

  return {
    item,
    node: item.node,
    group,
    materials,
    baseStates: materials.map((material) => ({
      opacity: material.opacity,
      emissive: Number.isFinite(material.emissiveIntensity) ? material.emissiveIntensity : 0,
    })),
    label,
    labelDefault,
    meshes,
    extra,
    visible: true,
    kind: item.kind,
    id: item.id,
  };
}

function buildKernelBank(item, layout) {
  const node = item.node;
  const params = item.params || {};
  const [kH, kW] = item.kernelSize || [3, 3];
  const cin = Math.max(1, Number(params.inChannels) || 1);
  const groups = Math.max(1, Number(params.groups) || 1);
  const depth = clamp(0.14 + 0.18 * Math.log2(cin / groups + 1), 0.16, 0.62);
  const gapToNode = 0.46;
  const centerX = item.x - item.sx / 2 - gapToNode - depth / 2;
  const group = new THREE.Group();
  group.name = `kernels:${item.id}`;
  group.position.set(centerX, 0, 0);
  group.userData.nodeId = item.id;
  group.userData.isKernel = true;

  const rows = clamp(kH, 1, 6);
  const cols = clamp(kW, 1, 6);
  const size = clamp(Math.min(item.sy, item.sz) * 0.66, 0.8, 1.65);
  const cellGap = 0.052;
  const cell = Math.max(0.06, (size - cellGap * (Math.max(rows, cols) - 1)) / Math.max(rows, cols));
  const totalH = cell * rows + cellGap * (rows - 1);
  const totalW = cell * cols + cellGap * (cols - 1);
  const materials = [];
  const meshes = [];
  const big = layout.big;

  const baseColor = new THREE.Color(item.color);
  const materialA = new THREE.MeshStandardMaterial({
    color: brighten(baseColor, 0.2),
    emissive: baseColor,
    emissiveIntensity: 0.45,
    transparent: true,
    opacity: 0.68,
    roughness: 0.34,
    metalness: 0.22,
    depthWrite: false,
  });
  const materialB = new THREE.MeshStandardMaterial({
    color: brighten(baseColor, 0.42),
    emissive: brighten(baseColor, 0.12),
    emissiveIntensity: 0.55,
    transparent: true,
    opacity: 0.8,
    roughness: 0.3,
    metalness: 0.25,
    depthWrite: false,
  });
  materials.push(materialA, materialB);

  const geometry = new THREE.BoxGeometry(depth, cell, cell);
  for (let r = 0; r < rows; r += 1) {
    for (let c = 0; c < cols; c += 1) {
      const mesh = new THREE.Mesh(geometry, (r + c) % 2 === 0 ? materialA : materialB);
      mesh.position.set(
        0,
        totalH / 2 - cell / 2 - r * (cell + cellGap),
        -(totalW / 2 - cell / 2 - c * (cell + cellGap)),
      );
      mesh.castShadow = false;
      mesh.userData.pick = { type: "kernel", id: item.id };
      group.add(mesh);
      meshes.push(mesh);
    }
  }

  const frameMaterial = new THREE.LineBasicMaterial({
    color: brighten(baseColor, 0.4),
    transparent: true,
    opacity: 0.32,
    depthWrite: false,
  });
  const frameGeometry = new THREE.BoxGeometry(depth + 0.04, totalH + 0.16, totalW + 0.16);
  const frame = new THREE.LineSegments(new THREE.EdgesGeometry(frameGeometry), frameMaterial);
  group.add(frame);
  materials.push(frameMaterial);

  const stemMaterial = new THREE.LineBasicMaterial({
    color: brighten(baseColor, 0.2),
    transparent: true,
    opacity: 0.26,
    depthWrite: false,
  });
  const stemGeometry = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(depth / 2 + 0.02, 0, 0),
    new THREE.Vector3(item.x - item.sx / 2 - centerX + 0.02, 0, 0),
  ]);
  group.add(new THREE.Line(stemGeometry, stemMaterial));
  materials.push(stemMaterial);

  let label = null;
  if (!big && layout.nodeCount <= 12) {
    label = createLabel(
      "edge-tag",
      `k${kH}×${kW}${groups > 1 ? ` · g${groups}` : ""}`,
      { x: 0, y: totalH / 2 + 0.2, z: 0 },
    );
    label.element.style.setProperty("--c", item.color);
    group.add(label);
  }

  const view = {
    type: "kernel",
    item,
    node,
    group,
    materials,
    baseStates: materials.map((material) => ({
      opacity: material.opacity,
      emissive: Number.isFinite(material.emissiveIntensity) ? material.emissiveIntensity : 0,
    })),
    label,
    labelDefault: Boolean(label),
    meshes,
    visible: true,
    id: item.id,
    kernelSize: [kH, kW],
  };
  return view;
}

/**
 * Build all node-related content.
 * @returns {{root: THREE.Group, kernelsRoot: THREE.Group, nodeViews: Array, kernelViews: Array, pickables: Array}}
 */
export function buildNodeContent(layout, graph, options = {}) {
  const root = new THREE.Group();
  root.name = "nodes";
  const kernelsRoot = new THREE.Group();
  kernelsRoot.name = "kernels";
  const nodeViews = [];
  const kernelViews = [];
  const pickables = [];

  for (const item of layout.nodes) {
    const view = makeNodeGroup(item, layout, options);
    root.add(view.group);
    nodeViews.push(view);
    pickables.push(...view.meshes);
  }

  for (const item of layout.nodes) {
    if (!item.hasKernels) continue;
    const view = buildKernelBank(item, layout);
    kernelsRoot.add(view.group);
    kernelViews.push(view);
    pickables.push(...view.meshes);
  }

  return { root, kernelsRoot, nodeViews, kernelViews, pickables };
}

/** Short shape description for a node (used by tooltips / details). */
export function nodeShapeSummary(node) {
  const outputs = Array.isArray(node.outShapes) ? node.outShapes : [];
  return groupShapeSummary(outputs);
}

export { kindLabel, formatParamValue };
