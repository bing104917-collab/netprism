/* Group wireframes, floor strips and labels. */

import * as THREE from "three";
import { createLabel } from "./labels.js";
import { escapeHtml, formatCount, groupMeta, kindMeta, truncate } from "./format.js";

function gradientEdgesGeometry(width, height, depth, color) {
  const box = new THREE.BoxGeometry(width, height, depth);
  const edges = new THREE.EdgesGeometry(box, 20);
  box.dispose();
  const position = edges.getAttribute("position");
  const colors = new Float32Array(position.count * 3);
  const base = new THREE.Color(color);
  const white = new THREE.Color(0xffffff);
  const tmp = new THREE.Color();
  for (let i = 0; i < position.count; i += 1) {
    const y = position.getY(i);
    const t = Math.min(1, Math.max(0, y / Math.max(height, 0.001) + 0.5));
    tmp.copy(base).lerp(white, 0.18 + t * 0.35);
    colors[i * 3] = tmp.r;
    colors[i * 3 + 1] = tmp.g;
    colors[i * 3 + 2] = tmp.b;
  }
  edges.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  return edges;
}

function buildGroupBox(groupItem) {
  const { group } = groupItem;
  const meta = groupMeta(group.kind);
  const color = meta.color;
  const b = groupItem.bounds;
  const width = Math.max(b.spanX, 0.2);
  const height = Math.max(b.spanY, 0.2);
  const depth = Math.max(b.spanZ, 0.2);
  const group3d = new THREE.Group();
  group3d.name = `group:${group.id}`;
  group3d.userData.groupId = group.id;

  const material = new THREE.LineBasicMaterial({
    vertexColors: true,
    transparent: true,
    opacity: 0.5,
    depthWrite: false,
  });
  const geometry = gradientEdgesGeometry(width, height, depth, color);
  const lines = new THREE.LineSegments(geometry, material);
  lines.position.set(b.centerX, b.centerY, b.centerZ);
  group3d.add(lines);

  // Corner accents give the frame a faint “technical drawing” feel.
  const accentMaterial = new THREE.LineBasicMaterial({
    color: new THREE.Color(color).lerp(new THREE.Color(0xffffff), 0.35),
    transparent: true,
    opacity: 0.58,
    depthWrite: false,
  });
  const accentPoints = [];
  const corner = 0.55;
  const cs = [
    [b.minX, b.minY, b.minZ], [b.maxX, b.minY, b.minZ],
    [b.minX, b.minY, b.maxZ], [b.maxX, b.minY, b.maxZ],
  ];
  for (const [x, y, z] of cs) {
    accentPoints.push(new THREE.Vector3(x, y, z), new THREE.Vector3(x, y + corner, z));
    accentPoints.push(new THREE.Vector3(x, y, z), new THREE.Vector3(x, y, z + (z > 0 ? -corner : corner)));
  }
  const accentGeometry = new THREE.BufferGeometry().setFromPoints(accentPoints);
  group3d.add(new THREE.LineSegments(accentGeometry, accentMaterial));

  // Soft colored strip on the floor under the group.
  const stripMaterial = new THREE.MeshBasicMaterial({
    color,
    transparent: true,
    opacity: 0.045,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
    side: THREE.DoubleSide,
  });
  const stripGeometry = new THREE.PlaneGeometry(Math.max(b.spanX - 0.7, 0.3), Math.max(b.spanZ - 0.4, 0.3));
  const strip = new THREE.Mesh(stripGeometry, stripMaterial);
  strip.rotation.x = -Math.PI / 2;
  strip.position.set(b.centerX, b.minY + 0.015, b.centerZ);
  group3d.add(strip);

  const label = createLabel(
    "group-tag",
    `<span class="gt-dot"></span>` +
      `<span class="gt-name">${escapeHtml(
        groupItem.depth <= 1
          ? groupItem.displayLabel || group.label || group.name || group.id
          : groupItem.shortLabel || group.label || group.name || group.id,
      )}</span>` +
      `<span class="gt-meta">${groupItem.descendantCount} 层 · ${formatCount(group.numParams || 0)}</span>`,
    {
      x: b.centerX,
      y: b.maxY + 0.82 + Math.min(groupItem.depth || 0, 5) * 0.52,
      z: 0,
    },
  );
  label.element.style.setProperty("--c", color);
  label.element.dataset.groupId = group.id;
  label.element.title = `${meta.label} · ${group.label || group.name || group.id}`;
  if (groupItem.depth > 1) label.element.classList.add("is-child");
  group3d.add(label);

  return {
    group,
    groupItem,
    group3d,
    materials: [material, accentMaterial, stripMaterial],
    label,
    box: true,
    visible: true,
  };
}

function buildParentLane(groupItem) {
  const { group } = groupItem;
  const meta = groupMeta(group.kind);
  const color = meta.color;
  const b = groupItem.bounds;
  const group3d = new THREE.Group();
  group3d.name = `group-lane:${group.id}`;
  group3d.userData.groupId = group.id;

  const material = new THREE.LineBasicMaterial({
    color,
    transparent: true,
    opacity: 0.28,
    depthWrite: false,
  });
  const y = b.maxY + 0.55;
  const points = [
    new THREE.Vector3(b.minX + 0.2, y, 0),
    new THREE.Vector3(b.maxX - 0.2, y, 0),
  ];
  const geometry = new THREE.BufferGeometry().setFromPoints(points);
  group3d.add(new THREE.Line(geometry, material));

  const tickMaterial = new THREE.LineBasicMaterial({
    color,
    transparent: true,
    opacity: 0.36,
    depthWrite: false,
  });
  const tickPoints = [
    new THREE.Vector3(b.minX + 0.2, y, 0), new THREE.Vector3(b.minX + 0.2, y + 0.24, 0),
    new THREE.Vector3(b.maxX - 0.2, y, 0), new THREE.Vector3(b.maxX - 0.2, y + 0.24, 0),
  ];
  const tickGeometry = new THREE.BufferGeometry().setFromPoints(tickPoints);
  group3d.add(new THREE.LineSegments(tickGeometry, tickMaterial));

  const label = createLabel(
    "group-tag is-parent",
    `<span class="gt-dot"></span>` +
      `<span class="gt-name">${escapeHtml(
        groupItem.depth <= 1
          ? groupItem.displayLabel || group.label || group.name || group.id
          : groupItem.shortLabel || group.label || group.name || group.id,
      )}</span>` +
      `<span class="gt-meta">${meta.label} · ${groupItem.descendantCount} 层</span>`,
    {
      x: b.centerX,
      y: b.maxY + 1.08 + Math.min(groupItem.depth || 0, 5) * 0.26,
      z: 0,
    },
  );
  label.element.style.setProperty("--c", color);
  label.element.dataset.groupId = group.id;
  label.element.title = `${meta.label} · ${group.label || group.name || group.id}`;
  if (groupItem.depth > 1) label.element.classList.add("is-child");

  group3d.add(label);
  return {
    group,
    groupItem,
    group3d,
    materials: [material, tickMaterial],
    label,
    box: false,
    visible: true,
  };
}

export function buildGroupContent(layout) {
  const root = new THREE.Group();
  root.name = "groups";
  const groupViews = [];

  for (const groupItem of layout.groups) {
    if (groupItem.isRoot) continue;
    if (groupItem.descendantCount === 0) continue;
    const drawBox = groupItem.hasDirect && groupItem.contiguous;
    const view = drawBox ? buildGroupBox(groupItem) : buildParentLane(groupItem);
    root.add(view.group3d);
    groupViews.push(view);
  }

  return { root, groupViews };
}

export function groupKindColor(kind) {
  return groupMeta(kind).color;
}

export function groupKindLabel(kind) {
  return groupMeta(kind).label;
}

export { kindMeta, truncate };
