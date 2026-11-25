/* Tiny helper around CSS2DObject so labels can be cleaned up reliably. */

import { CSS2DObject } from "three/addons/renderers/CSS2DRenderer.js";

export function createLabel(className, html, position = { x: 0, y: 0, z: 0 }) {
  const element = document.createElement("div");
  element.className = className;
  element.innerHTML = html;
  const object = new CSS2DObject(element);
  object.position.set(position.x || 0, position.y || 0, position.z || 0);
  object.center.set(0.5, 0.5);
  return object;
}

export function removeLabel(object) {
  if (!object) return;
  if (object.element && object.element.parentNode) {
    object.element.parentNode.removeChild(object.element);
  }
  if (object.parent) object.parent.remove(object);
}

export function disposeLabels(root) {
  if (!root) return;
  const labels = [];
  root.traverse((object) => {
    if (object.isCSS2DObject) labels.push(object);
  });
  for (const label of labels) {
    if (label.element && label.element.parentNode) label.element.parentNode.removeChild(label.element);
  }
}

export function setLabelVisible(element, visible) {
  if (!element) return;
  element.style.display = visible ? "" : "none";
}
