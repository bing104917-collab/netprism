/* Curved connection lines between layers, including residual / skip arcs. */

import * as THREE from "three";
import { createLabel } from "./labels.js";
import { edgeMeta, escapeHtml, formatShapeCompact } from "./format.js";

const UP = new THREE.Vector3(0, 1, 0);

function arrowMesh(curve, color, scale, opacity) {
  const length = 0.18 * scale;
  const radius = 0.055 * scale;
  const geometry = new THREE.ConeGeometry(radius, length, 10, 1);
  const material = new THREE.MeshBasicMaterial({
    color,
    transparent: true,
    opacity,
    depthWrite: false,
  });
  const mesh = new THREE.Mesh(geometry, material);
  const point = curve.getPoint(0.985);
  const tangent = curve.getTangent(0.985).normalize();
  mesh.position.copy(point).addScaledVector(tangent, -length * 0.48);
  mesh.quaternion.setFromUnitVectors(UP, tangent);
  return { mesh, material };
}

function specialArch(layout, source, target, offset, kind) {
  let top = Math.max(source.sy, target.sy) / 2;
  const from = Math.min(source.order, target.order);
  const to = Math.max(source.order, target.order);
  for (const item of layout.nodes) {
    if (item.order > from && item.order < to) top = Math.max(top, item.sy / 2);
  }
  // Route special edges above the tensor row.  Keep the apex low enough to
  // read as an arc rather than a tall spike.
  const base = kind === "skip" ? 2.65 : 1.95;
  return top + base + offset * 0.42;
}

export function buildEdgeContent(layout, graph, options = {}) {
  const root = new THREE.Group();
  root.name = "edges";
  const edgeViews = [];
  const pickables = [];
  const big = layout.big;
  const rawEdges = Array.isArray(graph.edges) ? graph.edges : [];
  const priority = { data: 0, control: 1, residual: 2, skip: 2, attention: 2 };
  const edges = [...rawEdges].sort(
    (a, b) => (priority[a.kind] ?? 0) - (priority[b.kind] ?? 0),
  );
  let specialIndex = 0;
  const segments = big ? 14 : 30;

  for (const edge of edges) {
    const source = layout.byId.get(edge.source);
    const target = layout.byId.get(edge.target);
    if (!source || !target) continue;
    const kind = ["data", "residual", "skip", "attention", "control"].includes(edge.kind)
      ? edge.kind
      : "data";
    const meta = edgeMeta(kind);
    const isAttention = kind === "attention";
    const isSpecial = kind !== "data" && !isAttention;
    const color = meta.color;
    const dx = target.x - source.x;

    let start;
    let end;
    let curve;
    if (isAttention) {
      // Attention-internal edges are only revealed when the user selects an
      // attention node. Keep them as short, local curves inside the block so
      // they never form spaghetti arcs above the tensor row.
      start = new THREE.Vector3(source.x + source.sx / 2 + 0.02, source.sy * 0.18, 0);
      end = new THREE.Vector3(target.x - target.sx / 2 - 0.02, target.sy * 0.18, 0);
      const reach = Math.max(0.45, Math.abs(dx) * 0.28);
      curve = new THREE.CubicBezierCurve3(
        start,
        new THREE.Vector3(start.x + reach, Math.max(source.sy * 0.18, target.sy * 0.18) + 0.55, 0),
        new THREE.Vector3(end.x - reach, Math.max(source.sy * 0.18, target.sy * 0.18) + 0.55, 0),
        end,
      );
    } else if (isSpecial) {
      start = new THREE.Vector3(
        source.x + source.sx / 2 + 0.05,
        source.sy * 0.28,
        0,
      );
      end = new THREE.Vector3(
        target.x - target.sx / 2 - 0.05,
        target.sy * 0.28,
        0,
      );
      const arch = specialArch(layout, source, target, specialIndex, kind);
      specialIndex += 1;
      // A single rounded apex reads as an arc.  Using two cubic control points
      // at the same height made short skips look like a tent/spike.
      const apex = new THREE.Vector3((start.x + end.x) / 2, arch, 0);
      curve = new THREE.QuadraticBezierCurve3(start, apex, end);
    } else {
      start = new THREE.Vector3(source.x + source.sx / 2 + 0.04, 0, 0);
      end = new THREE.Vector3(target.x - target.sx / 2 - 0.04, 0, 0);
      const mid = (start.x + end.x) / 2;
      const bow = Math.min(0.5, Math.max(0.06, Math.abs(dx) * 0.02));
      const c1 = new THREE.Vector3(mid - Math.abs(dx) * 0.2, bow, 0);
      const c2 = new THREE.Vector3(mid + Math.abs(dx) * 0.2, bow, 0);
      curve = new THREE.CubicBezierCurve3(start, c1, c2, end);
    }

    const points = curve.getPoints(segments);
    let line;
    let geometry;
    const materials = [];
    let opacity;
    if (isAttention) {
      opacity = 0.42;
      const lineMaterial = new THREE.LineBasicMaterial({
        color,
        transparent: true,
        opacity,
        depthWrite: false,
      });
      line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(points), lineMaterial);
      line.userData.pick = { type: "edge", id: edge.id };
      line.renderOrder = 3;
      materials.push(lineMaterial);
    } else if (isSpecial) {
      // Special connections deserve to read as real arcs, not 1px hairlines.
      opacity = 0.96;
      const radius = kind === "skip" ? 0.115 : 0.065;
      const tubeGeometry = new THREE.TubeGeometry(curve, Math.max(segments, 36), radius, 7, false);
      const tubeMaterial = new THREE.MeshBasicMaterial({
        color,
        transparent: true,
        opacity,
        depthWrite: false,
      });
      line = new THREE.Mesh(tubeGeometry, tubeMaterial);
      line.userData.pick = { type: "edge", id: edge.id };
      line.renderOrder = 5;
      materials.push(tubeMaterial);
    } else {
      opacity = kind === "control" ? 0.34 : 0.5;
      const lineColor = new THREE.Color(color).lerp(new THREE.Color(0xdbeafe), 0.24);
      geometry = new THREE.BufferGeometry().setFromPoints(points);
      const material = kind === "control"
        ? new THREE.LineDashedMaterial({
            color: lineColor,
            transparent: true,
            opacity,
            depthWrite: false,
            dashSize: 0.28,
            gapSize: 0.18,
          })
        : new THREE.LineBasicMaterial({
            color: lineColor,
            transparent: true,
            opacity,
            depthWrite: false,
          });
      line = new THREE.Line(geometry, material);
      if (kind === "control") line.computeLineDistances();
      line.userData.pick = { type: "edge", id: edge.id };
      line.renderOrder = 4;
      materials.push(material);
    }
    root.add(line);
    pickables.push(line);

    let glow = null;
    if (isSpecial && !big) {
      const glowGeometry = new THREE.TubeGeometry(curve, Math.max(segments, 32), kind === "skip" ? 0.24 : 0.15, 7, false);
      const glowMaterial = new THREE.MeshBasicMaterial({
        color,
        transparent: true,
        opacity: kind === "skip" ? 0.26 : 0.18,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
      });
      glow = new THREE.Mesh(glowGeometry, glowMaterial);
      glow.renderOrder = 4;
      root.add(glow);
      materials.push(glowMaterial);
    }

    // A data edge follows the left-to-right execution order; drawing an
    // arrowhead on every layer connection just adds noise.  Keep arrows for
    // skip/residual/attention where direction is semantically important.
    let arrow = null;
    if (isSpecial || isAttention) {
      const arrowScale = isSpecial ? 1.35 : 0.72;
      arrow = arrowMesh(curve, color, arrowScale, isSpecial ? 0.95 : 0.4);
      arrow.mesh.userData.pick = { type: "edge", id: edge.id };
      arrow.mesh.renderOrder = 5;
      root.add(arrow.mesh);
      materials.push(arrow.material);
      pickables.push(arrow.mesh);
    }

    let label = null;
    if (isSpecial && !big) {
      const at = curve.getPoint(0.5);
      label = createLabel(
        "edge-tag",
        escapeHtml(edge.label || meta.label),
        { x: at.x, y: at.y + 0.22, z: at.z },
      );
      label.element.style.setProperty("--c", color);
      root.add(label);
    }

    edgeViews.push({
      edge,
      kind,
      source,
      target,
      line,
      glow,
      arrow: arrow ? arrow.mesh : null,
      label,
      labelDefault: false,
      isAttention,
      materials,
      baseStates: materials.map((mat) => ({
        opacity: mat.opacity,
        emissive: Number.isFinite(mat.emissiveIntensity) ? mat.emissiveIntensity : 0,
      })),
      baseOpacity: opacity,
      visible: true,
      id: edge.id,
      shapeLabel: Array.isArray(edge.shapes) && edge.shapes.length
        ? formatShapeCompact(edge.shapes[0])
        : "",
    });
  }

  return { root, edgeViews, pickables };
}
