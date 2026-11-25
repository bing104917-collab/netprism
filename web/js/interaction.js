/* Pointer / keyboard interaction: hover, select, focus, filters, data-flow. */

import * as THREE from "three";
import { clamp } from "./format.js";

const NONE = new Set();

function isTypingTarget(target) {
  if (!target) return false;
  const tag = target.tagName;
  return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || target.isContentEditable;
}

export class InteractionManager {
  constructor(sceneManager, ui, container) {
    this.scene = sceneManager;
    this.ui = ui;
    this.container = container;
    this.canvas = sceneManager.renderer.domElement;
    this.raycaster = new THREE.Raycaster();
    this.raycaster.params.Line.threshold = 0.2;
    this.raycaster.params.Points.threshold = 0.15;
    this.pointer = new THREE.Vector2();
    this.ctx = null;
    this.nodeViewById = new Map();
    this.kernelViewById = new Map();
    this.kernelsByNode = new Map();
    this.edgeViewById = new Map();
    this.neighborsById = new Map();
    this.state = {
      hiddenKinds: new Set(),
      toggles: { kernels: true, groups: true, labels: true, edges: true, allLabels: false },
      selectedId: null,
      hoverKey: null,
      hover: null,
      flow: {
        active: false,
        playing: false,
        index: -1,
        sub: 0,
        wait: 0,
        done: false,
      },
    };
    this.dragStart = null;
    this.dragging = false;
    this._pickRaf = 0;
    this._lastPointerEvent = null;
    this._frameHandler = this._onFrame.bind(this);
    this._unsubscribeFrame = sceneManager.addFrameCallback(this._frameHandler);
    this._collisionTimer = null;
    this._cameraVisibilityTimer = null;
    this.focusedGroupIds = new Set();
    this.scene.onCameraSettled = () => {
      this.refreshVisibility();
      this.scheduleLabelCollisions(220);
    };
    this.scene.onCameraMove = (delay) => {
      clearTimeout(this._cameraVisibilityTimer);
      this._cameraVisibilityTimer = setTimeout(() => {
        this.refreshVisibility();
        this.scheduleLabelCollisions(60);
      }, delay || 620);
    };
    this._bind();
  }

  /* ------------------------------------------------------------------ */
  /* Wiring                                                             */
  /* ------------------------------------------------------------------ */

  _bind() {
    this._onPointerMove = (event) => {
      this._lastPointerEvent = event;
      if (!this.ctx) return;
      if (!this._pickRaf) {
        this._pickRaf = requestAnimationFrame(() => {
          this._pickRaf = 0;
          if (this._lastPointerEvent) this._pickAt(this._lastPointerEvent);
        });
      }
      if (this.dragStart && !this.dragging) {
        const dx = event.clientX - this.dragStart.x;
        const dy = event.clientY - this.dragStart.y;
        if (dx * dx + dy * dy > 25) this.dragging = true;
      }
    };
    this._onPointerLeave = () => {
      this.ui.hideTooltip();
      this._clearHover();
      this.canvas.style.cursor = "";
    };
    this._onPointerDown = (event) => {
      if (event.button !== 0) return;
      this.dragStart = { x: event.clientX, y: event.clientY, time: performance.now() };
      this.dragging = false;
    };
    this._onPointerUp = (event) => {
      if (!this.dragStart || event.button !== 0) return;
      const elapsed = performance.now() - this.dragStart.time;
      const moved = Math.abs(event.clientX - this.dragStart.x) + Math.abs(event.clientY - this.dragStart.y);
      this.dragStart = null;
      if (this.dragging || moved > 6 || elapsed > 800) {
        this.dragging = false;
        return;
      }
      this.dragging = false;
      this._handleClick(event);
    };
    this._onDoubleClick = (event) => this._handleDoubleClick(event);
    this._onContextMenu = (event) => event.preventDefault();
    this._onKeyDown = (event) => this._handleKeyDown(event);
    this._onBlur = () => {
      this.ui.hideTooltip();
      this._clearHover();
    };

    this.canvas.addEventListener("pointermove", this._onPointerMove);
    this.canvas.addEventListener("pointerleave", this._onPointerLeave);
    this.canvas.addEventListener("pointerdown", this._onPointerDown);
    this.canvas.addEventListener("pointerup", this._onPointerUp);
    this.canvas.addEventListener("dblclick", this._onDoubleClick);
    this.canvas.addEventListener("contextmenu", this._onContextMenu);
    window.addEventListener("keydown", this._onKeyDown);
    window.addEventListener("blur", this._onBlur);
    this.scene.onInteractionStart = () => {
      this.ui.hideTooltip();
      this._clearHover();
      this.ui.hideOnboarding();
    };
  }

  setModel(ctx) {
    this.stopFlow({ silent: true });
    this.clearSelection({ silent: true });
    this._clearHover();
    this.ui.hideTooltip();
    this.ctx = ctx || null;
    this.nodeViewById = new Map((ctx?.nodeViews || []).map((view) => [view.id, view]));
    this.kernelViewById = new Map((ctx?.kernelViews || []).map((view) => [view.id, view]));
    this.edgeViewById = new Map((ctx?.edgeViews || []).map((view) => [view.id, view]));
    this.kernelsByNode = new Map();
    for (const view of ctx?.kernelViews || []) {
      if (!this.kernelsByNode.has(view.id)) this.kernelsByNode.set(view.id, []);
      this.kernelsByNode.get(view.id).push(view);
    }
    this.neighborsById = new Map();
    for (const view of ctx?.edgeViews || []) {
      if (!this.neighborsById.has(view.source.id)) this.neighborsById.set(view.source.id, new Set());
      if (!this.neighborsById.has(view.target.id)) this.neighborsById.set(view.target.id, new Set());
      this.neighborsById.get(view.source.id).add(view.target.id);
      this.neighborsById.get(view.target.id).add(view.source.id);
    }
    this.state.selectedId = null;
    this.state.hoverKey = null;
    this.state.hover = null;
    this._flowItemsCache = null;
    this.ui.setToggleButtons(this.state.toggles);
    this.refreshVisibility();
    this.refreshMaterials();
    this.scheduleLabelCollisions(140);
  }

  /* ------------------------------------------------------------------ */
  /* Visibility & materials                                             */
  /* ------------------------------------------------------------------ */

  nodeVisible(item) {
    if (!item) return false;
    return !this.state.hiddenKinds.has(item.kind);
  }

  hiddenFromKinds(view) {
    return !this.nodeVisible(view.item);
  }

  _focusedGroupIds() {
    const ids = new Set();
    if (!this.ctx) return ids;
    const addPath = (nodeId) => {
      if (!nodeId) return;
      const item = this.ctx.layout.byId.get(nodeId);
      let groupId = item && item.groupId;
      let guard = 0;
      while (groupId && this.ctx.layout.groupsById.has(groupId) && guard < 24) {
        ids.add(groupId);
        const group = this.ctx.layout.groupsById.get(groupId);
        groupId = group && group.parentId;
        guard += 1;
      }
    };
    if (this.state.selectedId) addPath(this.state.selectedId);
    if (this.state.hover && (this.state.hover.type === "node" || this.state.hover.type === "kernel")) {
      addPath(this.state.hover.id);
    }
    return ids;
  }

  refreshVisibility() {
    if (!this.ctx) return;
    const { toggles } = this.state;
    const selected = this.state.selectedId;
    const hover = this.state.hover;
    this.focusedGroupIds = this._focusedGroupIds();
    for (const view of this.ctx.nodeViews) {
      const visible = this.nodeVisible(view.item);
      view.group.visible = visible;
      if (!visible) continue;
      const hovered = Boolean(
        hover && ((hover.type === "node" && hover.id === view.id) || (hover.type === "kernel" && hover.id === view.id)),
      );
      const labelVisible =
        toggles.labels &&
        (toggles.allLabels || view.labelDefault || selected === view.id || hovered);
      if (view.label) {
        view.label.visible = labelVisible;
        if (labelVisible && (selected === view.id || hovered)) {
          view.label.element.classList.remove("is-collision-hidden");
        }
      }
    }
    for (const view of this.ctx.kernelViews) {
      const nodeVisible = this.nodeVisible(view.item);
      view.group.visible = nodeVisible && toggles.kernels;
      if (view.label) {
        view.label.visible = Boolean(
          nodeVisible && toggles.kernels && toggles.labels && (toggles.allLabels || view.labelDefault),
        );
      }
    }
    const selectedNode = selected ? this.ctx.layout.byId.get(selected) : null;
    const hoveredEdge = hover && hover.type === "edge" ? hover.id : null;
    let specialLabelBudget = selected ? 1 : 0;
    for (const view of this.ctx.edgeViews) {
      const endpointVisible = this.nodeVisible(view.source) && this.nodeVisible(view.target);
      const attentionRevealed =
        view.kind !== "attention" ||
        Boolean(selectedNode && (view.source.id === selected || view.target.id === selected));
      const edgeVisible = Boolean(toggles.edges && endpointVisible && attentionRevealed);
      if (view.line) view.line.visible = edgeVisible;
      if (view.glow) view.glow.visible = edgeVisible;
      if (view.arrow) view.arrow.visible = edgeVisible;
      view.visible = edgeVisible;
      if (view.label) {
        let showLabel = false;
        if (edgeVisible && toggles.labels && view.kind !== "attention") {
          if (hoveredEdge === view.id) {
            showLabel = true;
          } else if (selected && (view.source.id === selected || view.target.id === selected) && specialLabelBudget > 0) {
            showLabel = true;
            specialLabelBudget -= 1;
          }
        }
        view.label.visible = showLabel;
      }
    }
    if (this.ctx.edgesRoot) this.ctx.edgesRoot.visible = toggles.edges;

    // Group labels are tiered: depth 0/1 always on, deeper labels appear when
    // their node is hovered/selected or when the user zooms in. In the default
    // mode only 8 group labels may compete for screen space (collision is
    // resolved afterwards, depth first).
    const zoomReveal = this.scene.isLabelZoomedIn();
    const groupEntries = [];
    for (const view of this.ctx.groupViews) {
      const anyVisible = view.groupItem.descendantNodes.some((item) => this.nodeVisible(item));
      const depth = Number(view.groupItem.depth) || 0;
      const focused = this.focusedGroupIds.has(view.group.id);
      const labelVisible =
        toggles.groups &&
        toggles.labels &&
        anyVisible &&
        (toggles.allLabels || depth <= 1 || focused || (zoomReveal && depth === 2));
      view.group3d.visible = toggles.groups && anyVisible;
      view.visible = toggles.groups && anyVisible;
      if (view.label) {
        groupEntries.push({ view, depth, focused, labelVisible });
        view.label.visible = labelVisible;
      }
    }
    if (!toggles.allLabels && toggles.labels) {
      const visibleGroups = groupEntries
        .filter((entry) => entry.labelVisible)
        .sort((a, b) => Number(b.focused) - Number(a.focused) || a.depth - b.depth || a.view.groupItem.firstOrder - b.view.groupItem.firstOrder);
      const keep = new Set(visibleGroups.slice(0, 8).map((entry) => entry.view.group.id));
      for (const entry of groupEntries) {
        if (entry.labelVisible && !keep.has(entry.view.group.id)) entry.view.label.visible = false;
      }
    }
    this._reconcileFlow();
  }

  _applyScale(view, opacityScale = 1, emissiveScale = 1) {
    if (!view || !view.baseStates) return;
    for (let i = 0; i < view.materials.length; i += 1) {
      const material = view.materials[i];
      const base = view.baseStates[i];
      if (!material || !base) continue;
      material.opacity = clamp(base.opacity * opacityScale, 0, 1);
      if (Number.isFinite(base.emissive) && base.emissive > 0) {
        material.emissiveIntensity = base.emissive * emissiveScale;
      }
    }
  }

  refreshMaterials() {
    if (!this.ctx) return;
    const selected = this.state.selectedId;
    const neighbors = selected ? this.neighborsById.get(selected) || NONE : NONE;
    const hover = this.state.hover;
    const selectedView = selected ? this.nodeViewById.get(selected) : null;

    for (const view of this.ctx.nodeViews) {
      let opacityScale = 1;
      let emissiveScale = 1;
      if (selected) {
        if (view.id === selected) {
          opacityScale = 1.18;
          emissiveScale = 2.2;
        } else if (neighbors.has(view.id)) {
          opacityScale = 0.98;
          emissiveScale = 1.6;
        } else {
          opacityScale = 0.38;
          emissiveScale = 0.95;
        }
      }
      const hovered = Boolean(
        hover && ((hover.type === "node" && hover.id === view.id) || (hover.type === "kernel" && hover.id === view.id)),
      );
      if (hovered) {
        opacityScale = Math.max(opacityScale, view.id === selected ? 1.14 : 1.02);
        emissiveScale = Math.max(emissiveScale, 2.0);
      }
      this._applyScale(view, opacityScale, emissiveScale);
      const labelEl = view.label?.element;
      if (labelEl) {
        labelEl.classList.toggle("is-hovered", hovered);
        labelEl.classList.toggle("is-selected", view.id === selected);
        labelEl.classList.toggle(
          "is-dimmed",
          Boolean(selected && view.id !== selected && !neighbors.has(view.id) && !hovered),
        );
      }
      for (const kernel of this.kernelsByNode.get(view.id) || []) {
        let kScale = opacityScale;
        let kEmissive = emissiveScale;
        if (hovered) {
          kScale = Math.max(kScale, 1.1);
          kEmissive = Math.max(kEmissive, 1.9);
        }
        this._applyScale(kernel, kScale, kEmissive);
      }
    }

    for (const view of this.ctx.edgeViews) {
      let scale = 1;
      if (selected) {
        const connected = view.source.id === selected || view.target.id === selected;
        scale = connected ? 1.35 : 0.28;
      }
      if (hover && hover.type === "edge" && hover.id === view.id) scale = 1.8;
      this._applyScale(view, scale, 1);
    }

    if (selectedView) {
      // Kept for readability; selection ring handled by the scene.
    }
  }

  /* ------------------------------------------------------------------ */
  /* Picking & hover                                                    */
  /* ------------------------------------------------------------------ */

  _setRay(clientX, clientY) {
    const rect = this.canvas.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) return false;
    if (clientX < rect.left || clientX > rect.right || clientY < rect.top || clientY > rect.bottom) return false;
    this.pointer.x = ((clientX - rect.left) / rect.width) * 2 - 1;
    this.pointer.y = -((clientY - rect.top) / rect.height) * 2 + 1;
    this.raycaster.setFromCamera(this.pointer, this.scene.camera);
    return true;
  }

  _ancestorsVisible(object) {
    let current = object;
    while (current) {
      if (!current.visible) return false;
      current = current.parent;
    }
    return true;
  }

  _canPick(pick, object) {
    if (!pick) return false;
    if (!this._ancestorsVisible(object)) return false;
    if (pick.type === "node") {
      const view = this.nodeViewById.get(pick.id);
      return Boolean(view && this.nodeVisible(view.item));
    }
    if (pick.type === "kernel") {
      if (!this.state.toggles.kernels) return false;
      const view = this.kernelViewById.get(pick.id);
      return Boolean(view && this.nodeVisible(view.item));
    }
    if (pick.type === "edge") {
      if (!this.state.toggles.edges) return false;
      const view = this.edgeViewById.get(pick.id);
      return Boolean(
        view && view.visible && this.nodeVisible(view.source) && this.nodeVisible(view.target),
      );
    }
    return false;
  }

  _cast(clientX, clientY) {
    if (!this.ctx || !this._setRay(clientX, clientY)) return null;
    const intersections = this.raycaster.intersectObjects(this.ctx.pickables || [], false);
    for (const hit of intersections) {
      const pick = hit.object.userData.pick;
      if (pick && this._canPick(pick, hit.object)) {
        return { type: pick.type, id: pick.id, object: hit.object, point: hit.point, distance: hit.distance };
      }
    }
    return null;
  }

  _pickAt(event) {
    if (!this.ctx) return;
    if (this.dragging) {
      this.ui.hideTooltip();
      return;
    }
    const hit = this._cast(event.clientX, event.clientY);
    const key = hit ? `${hit.type}:${hit.id}` : null;
    if (key !== this.state.hoverKey) {
      this._applyHover(hit);
    }
    if (hit) {
      if (key === this.state.hoverKey && !this.ui.el.tooltip.hidden) {
        this.ui.positionTooltip(event.clientX, event.clientY);
      } else {
        const payload = this._tooltipPayload(hit);
        if (payload) {
          this.ui.showTooltip(payload, event.clientX, event.clientY);
          this.canvas.style.cursor = "pointer";
        }
      }
    } else {
      this.ui.hideTooltip();
      this.canvas.style.cursor = "";
    }
  }

  _tooltipPayload(hit) {
    if (hit.type === "node") {
      const view = this.nodeViewById.get(hit.id);
      if (!view) return null;
      return { type: "node", node: view.node, item: view.item };
    }
    if (hit.type === "kernel") {
      const view = this.kernelViewById.get(hit.id);
      if (!view) return null;
      return { type: "kernel", node: view.node, item: view.item, kernelSize: view.kernelSize };
    }
    if (hit.type === "edge") {
      const view = this.edgeViewById.get(hit.id);
      if (!view) return null;
      return { type: "edge", edge: view.edge, source: view.source, target: view.target };
    }
    return null;
  }

  _applyHover(hit) {
    this._clearHover();
    if (!hit) return;
    this.state.hoverKey = `${hit.type}:${hit.id}`;
    this.state.hover = { type: hit.type, id: hit.id };
    this.refreshMaterials();
    this.refreshVisibility();
    this.scheduleLabelCollisions(90);
    this.canvas.style.cursor = "pointer";
  }

  _clearHover() {
    if (!this.state.hover) return;
    this.state.hover = null;
    this.state.hoverKey = null;
    this.refreshMaterials();
    this.refreshVisibility();
    this.scheduleLabelCollisions(90);
  }

  /* ------------------------------------------------------------------ */
  /* Click / focus                                                      */
  /* ------------------------------------------------------------------ */

  _handleClick(event) {
    if (!this.ctx) return;
    const hit = this._cast(event.clientX, event.clientY);
    if (!hit) {
      this.clearSelection();
      return;
    }
    if (hit.type === "node" || hit.type === "kernel") {
      this.selectNode(hit.id, { focus: false });
    }
  }

  _handleDoubleClick(event) {
    if (!this.ctx) return;
    const hit = this._cast(event.clientX, event.clientY);
    if (!hit) {
      this.resetView();
      return;
    }
    if (hit.type === "node" || hit.type === "kernel") {
      this.selectNode(hit.id, { focus: true });
    } else if (hit.type === "edge") {
      const view = this.edgeViewById.get(hit.id);
      if (view) this.selectNode(view.source.id, { focus: true });
    }
  }

  selectNode(id, { focus = false } = {}) {
    if (!this.ctx) return;
    const view = this.nodeViewById.get(id);
    if (!view) return;
    if (this.state.hiddenKinds.has(view.item.kind)) {
      this.state.hiddenKinds.delete(view.item.kind);
      this._syncFilters();
    }
    this.state.selectedId = id;
    this.scene.showSelection(view.item);
    this.ui.renderDetail(view.node);
    this.ui.setTreeSelected(id);
    this.ui.hideTooltip();
    this.refreshVisibility();
    this.refreshMaterials();
    this.scheduleLabelCollisions(90);
    if (focus) this.scene.focusItem(view.item);
  }

  clearSelection({ silent = false } = {}) {
    const had = Boolean(this.state.selectedId);
    this.state.selectedId = null;
    this.scene.hideSelection();
    if (!silent) {
      this.ui.renderDetail(null);
      this.ui.setTreeSelected(null);
    } else {
      this.ui.renderDetail(null);
      this.ui.setTreeSelected(null);
    }
    this.refreshVisibility();
    this.refreshMaterials();
    if (had) this.ui.hideTooltip();
    this.scheduleLabelCollisions(100);
  }

  selectRelative(delta) {
    if (!this.ctx || !this.ctx.layout.ordered.length) return;
    const ordered = this.ctx.layout.ordered;
    let index = this.state.selectedId
      ? ordered.findIndex((item) => item.id === this.state.selectedId)
      : -1;
    if (index < 0) index = delta > 0 ? -1 : ordered.length;
    const next = clamp(index + delta, 0, ordered.length - 1);
    const item = ordered[next];
    if (item) this.selectNode(item.id, { focus: false });
  }

  focusSelected() {
    if (!this.ctx) return;
    const item = this.state.selectedId ? this.ctx.layout.byId.get(this.state.selectedId) : null;
    if (item) this.scene.focusItem(item);
    else this.scene.fit();
  }

  resetView() {
    this.clearSelection();
    this.ui.hideTooltip();
    this.ui.hideOnboarding();
    this.scene.fit();
  }

  /* ------------------------------------------------------------------ */
  /* Filters / toggles                                                  */
  /* ------------------------------------------------------------------ */

  _syncFilters() {
    this.ui.setHiddenKinds(this.state.hiddenKinds);
    this.ui.refreshTreeFiltered(this.state.hiddenKinds);
    this.ui.renderLegend();
    this._flowItemsCache = null;
    this.refreshVisibility();
    this.refreshMaterials();
  }

  toggleKind(kind) {
    if (!kind) return;
    if (this.state.hiddenKinds.has(kind)) this.state.hiddenKinds.delete(kind);
    else this.state.hiddenKinds.add(kind);
    const selected = this.state.selectedId ? this.nodeViewById.get(this.state.selectedId) : null;
    if (selected && this.state.hiddenKinds.has(selected.item.kind)) {
      this.clearSelection({ silent: true });
    }
    this._syncFilters();
    this._reconcileFlow();
    this.scheduleLabelCollisions(120);
  }

  showAllKinds() {
    if (!this.state.hiddenKinds.size) return;
    this.state.hiddenKinds.clear();
    this._syncFilters();
    this.scheduleLabelCollisions(120);
  }

  setToggle(name, value) {
    if (!(name in this.state.toggles)) return;
    this.state.toggles[name] = Boolean(value);
    if (name === "allLabels" && value) this.state.toggles.labels = true;
    this.ui.setToggleButtons(this.state.toggles);
    if (name === "edges" && !value && this.state.hover?.type === "edge") {
      this._clearHover();
      this.ui.hideTooltip();
    }
    if (name === "kernels" && !value && this.state.hover?.type === "kernel") {
      this._clearHover();
      this.ui.hideTooltip();
    }
    this.refreshVisibility();
    this.scheduleLabelCollisions(120);
  }

  toggle(name) {
    this.setToggle(name, !this.state.toggles[name]);
  }

  getToggles() {
    return this.state.toggles;
  }

  getHiddenKinds() {
    return this.state.hiddenKinds;
  }

  /* ------------------------------------------------------------------ */
  /* Data flow                                                          */
  /* ------------------------------------------------------------------ */

  _flowItems() {
    if (!this.ctx) return [];
    if (this._flowItemsCache) return this._flowItemsCache;
    const items = this.ctx.layout.ordered.filter((item) => this.nodeVisible(item));
    this._flowItemsCache = items;
    return items;
  }

  _flowPoint(item) {
    return new THREE.Vector3(item.x, item.sy / 2 + 0.55, 0);
  }

  _flowLabel(item, active) {
    if (this._flowLabelEl) this._flowLabelEl.classList.remove("is-flow");
    this._flowLabelEl = null;
    if (!item) return;
    const view = this.nodeViewById.get(item.id);
    if (view?.label?.element) {
      view.label.element.classList.add("is-flow");
      this._flowLabelEl = view.label.element;
    }
  }

  _updateFlowUI(item, done = false) {
    const items = this._flowItems();
    if (!item) {
      this.ui.setPlayState({ playing: false, active: false, index: 0, total: 0, name: "" });
      return;
    }
    const index = Math.max(0, items.findIndex((entry) => entry.id === item.id));
    this.ui.setPlayState({
      playing: this.state.flow.playing,
      active: true,
      index,
      total: items.length,
      name: done ? `数据流完成 · ${item.node.name}` : item.node.name,
    });
  }

  togglePlay() {
    if (!this.ctx) return;
    const items = this._flowItems();
    if (!items.length) {
      this.ui.toast("info", "没有可播放的层", "当前过滤条件下没有可见节点。");
      return;
    }
    if (this.state.flow.playing) {
      this.pauseFlow();
      return;
    }
    if (!this.state.flow.active || this.state.flow.done) {
      this.state.flow = { active: true, playing: true, index: -1, sub: 0, wait: 0, done: false };
      this._advanceFlow(1, true);
      return;
    }
    this.state.flow.playing = true;
    const item = items[clamp(this.state.flow.index, 0, items.length - 1)];
    this.scene.showFlowTarget(item);
    this.scene.showFlowPoint(this._flowPoint(item), true);
    this._flowLabel(item);
    this._updateFlowUI(item);
  }

  pauseFlow() {
    if (!this.state.flow.active) return;
    this.state.flow.playing = false;
    this.ui.setPlayState({
      playing: false,
      active: true,
      index: Math.max(0, this.state.flow.index),
      total: this._flowItems().length,
      name: this._flowItems()[Math.max(0, this.state.flow.index)]?.node?.name || "",
    });
    this._flowLabel(null);
  }

  stopFlow({ silent = false } = {}) {
    this.state.flow = { active: false, playing: false, index: -1, sub: 0, wait: 0, done: false };
    this.scene.showFlowPoint(null, false);
    this.scene.showFlowTarget(null);
    this._flowLabel(null);
    if (!silent) this.ui.setPlayState({ playing: false, active: false, index: 0, total: 0, name: "" });
  }

  stepFlow(delta = 1) {
    if (!this.ctx) return;
    const items = this._flowItems();
    if (!items.length) return;
    if (this.state.flow.playing) {
      this.state.flow.playing = false;
    }
    if (!this.state.flow.active || this.state.flow.done) {
      this.state.flow.active = true;
      this.state.flow.done = false;
      this.state.flow.index = -1;
    }
    let index = this.state.flow.index + delta;
    index = clamp(index, 0, items.length - 1);
    this.state.flow.index = index;
    this.state.flow.sub = 0;
    this.state.flow.wait = 0;
    this.state.flow.done = index >= items.length - 1;
    const item = items[index];
    this.scene.showFlowTarget(item);
    this.scene.showFlowPoint(this._flowPoint(item), true);
    this._flowLabel(item);
    this.ui.setPlayState({
      playing: false,
      active: true,
      index,
      total: items.length,
      name: item.node.name,
    });
  }

  _advanceFlow(delta, instant = false) {
    const items = this._flowItems();
    if (!items.length) {
      this.stopFlow();
      return;
    }
    let index = this.state.flow.index + delta;
    if (index >= items.length) {
      this.state.flow.index = items.length - 1;
      this.state.flow.playing = false;
      this.state.flow.done = true;
      const last = items[items.length - 1];
      this.scene.showFlowPoint(this._flowPoint(last), true);
      this.scene.showFlowTarget(last);
      this._flowLabel(last);
      this._updateFlowUI(last, true);
      return;
    }
    index = Math.max(0, index);
    this.state.flow.index = index;
    this.state.flow.sub = 0;
    this.state.flow.wait = 0;
    const item = items[index];
    this.scene.showFlowTarget(item);
    this.scene.showFlowPoint(this._flowPoint(item), true);
    this._flowLabel(item);
    this._updateFlowUI(item);
  }

  _reconcileFlow() {
    if (!this.state.flow.active) return;
    const items = this._flowItems();
    if (!items.length) {
      this.stopFlow();
      return;
    }
    if (this.state.flow.index >= items.length) {
      this.state.flow.index = items.length - 1;
      this.state.flow.done = true;
    }
    const item = items[Math.max(0, this.state.flow.index)];
    this.scene.showFlowTarget(item);
    if (!this.state.flow.playing) this.scene.showFlowPoint(this._flowPoint(item), true);
    this._updateFlowUI(item);
  }

  _onFrame(dt) {
    if (!this.state.flow.active || !this.state.flow.playing || !this.ctx) return;
    const items = this._flowItems();
    if (!items.length) return;
    if (this.scene.reducedMotion) {
      this.state.flow.wait += dt;
      if (this.state.flow.wait >= 0.85) {
        this.state.flow.wait = 0;
        this._advanceFlow(1, true);
      }
      return;
    }
    const index = clamp(this.state.flow.index, 0, items.length - 1);
    const current = items[index];
    const next = items[index + 1];
    if (!next) {
      if (this.state.flow.index >= items.length - 1) {
        this.state.flow.playing = false;
        this.state.flow.done = true;
        this._updateFlowUI(current, true);
      }
      return;
    }
    this.state.flow.sub += dt * 1.55;
    const t = clamp(this.state.flow.sub, 0, 1);
    const from = this._flowPoint(current);
    const to = this._flowPoint(next);
    const position = from.clone().lerp(to, t);
    this.scene.showFlowPoint(position, true);
    if (this.state.flow.sub >= 1) {
      this.state.flow.sub = 0;
      this._advanceFlow(1);
    }
  }

  /* ------------------------------------------------------------------ */
  /* Label collision management                                         */
  /* ------------------------------------------------------------------ */

  scheduleLabelCollisions(delay = 120) {
    clearTimeout(this._collisionTimer);
    this._collisionTimer = setTimeout(() => {
      requestAnimationFrame(() => requestAnimationFrame(() => this.resolveLabelCollisions()));
    }, delay);
  }

  resolveLabelCollisions() {
    if (!this.ctx) return;
    const all = [];
    const collect = (view, priority, visible, protectedLabel = false) => {
      const element = view && view.label && view.label.element;
      if (!element || !visible) return;
      element.classList.remove("is-collision-hidden");
      all.push({ element, priority, protectedLabel });
    };

    for (const view of this.ctx.groupViews) {
      const depth = Math.min(view.groupItem?.depth || 0, 6);
      const focused = this.focusedGroupIds.has(view.group.id);
      collect(
        view,
        focused ? 720 : 420 - depth * 14,
        view.group3d.visible && view.label?.visible,
        focused,
      );
    }
    for (const view of this.ctx.nodeViews) {
      const selected = this.state.selectedId === view.id;
      const hovered = Boolean(
        this.state.hover &&
          ((this.state.hover.type === "node" && this.state.hover.id === view.id) ||
            (this.state.hover.type === "kernel" && this.state.hover.id === view.id)),
      );
      const visible = view.group.visible && view.label?.visible;
      collect(view, selected ? 640 : hovered ? 560 : 120, visible, selected || hovered);
    }
    for (const view of this.ctx.kernelViews) {
      collect(view, 80, view.group.visible && view.label?.visible);
    }
    for (const view of this.ctx.edgeViews) {
      const hoveredEdge = Boolean(
        this.state.hover &&
          this.state.hover.type === "edge" &&
          this.state.hover.id === view.id,
      );
      collect(view, hoveredEdge ? 520 : 70, view.label?.visible, hoveredEdge);
    }

    if (!this.state.toggles.labels || this.state.toggles.allLabels) {
      for (const entry of all) entry.element.classList.remove("is-collision-hidden");
      return;
    }

    const measured = [];
    for (const entry of all) {
      const rect = entry.element.getBoundingClientRect();
      if (rect.width < 2 || rect.height < 2) continue;
      if (entry.element.style.display === "none") continue;
      measured.push({ ...entry, rect });
    }
    measured.sort((a, b) => b.priority - a.priority || a.rect.top - b.rect.top);
    const accepted = [];
    const overlaps = (a, b, pad = 4) =>
      a.left < b.right + pad && a.right + pad > b.left && a.top < b.bottom + pad && a.bottom + pad > b.top;
    for (const entry of measured) {
      if (entry.protectedLabel) {
        accepted.push(entry.rect);
        continue;
      }
      if (accepted.some((rect) => overlaps(entry.rect, rect))) {
        entry.element.classList.add("is-collision-hidden");
      } else {
        accepted.push(entry.rect);
      }
    }
  }

  /* ------------------------------------------------------------------ */
  /* Keyboard                                                           */
  /* ------------------------------------------------------------------ */

  _handleKeyDown(event) {
    if (event.defaultPrevented) return;
    if (this.ui.isHelpOpen()) {
      if (event.key === "Escape") {
        event.preventDefault();
        this.ui.closeHelp();
      }
      return;
    }
    if (this.ui.isAlgoOpen()) {
      if (isTypingTarget(event.target)) return;
      if (event.key === "Escape") {
        event.preventDefault();
        this.ui.closeAlgo();
        return;
      }
      if (event.key === "a" || event.key === "A") {
        event.preventDefault();
        if (event.shiftKey) this.toggle("allLabels");
        else this.ui.closeAlgo();
        return;
      }
      if (event.key === "t" || event.key === "T") {
        event.preventDefault();
        this.ui.openTable();
        return;
      }
      return;
    }
    if (this.ui.isTableOpen()) {
      if (isTypingTarget(event.target)) return;
      if (event.key === "Escape") {
        event.preventDefault();
        this.ui.closeTable();
        return;
      }
      if (event.key === "t" || event.key === "T") {
        event.preventDefault();
        this.ui.closeTable();
        return;
      }
      if (event.key === "a" || event.key === "A") {
        event.preventDefault();
        if (event.shiftKey) this.toggle("allLabels");
        else this.ui.openAlgo();
        return;
      }
      if (event.key === "/") {
        event.preventDefault();
        this.ui.focusTableSearch();
      }
      return;
    }
    if (isTypingTarget(event.target)) {
      return;
    }
    if (event.ctrlKey || event.metaKey || event.altKey) return;
    if (event.key === "a" || event.key === "A") {
      event.preventDefault();
      if (event.shiftKey) this.toggle("allLabels");
      else this.ui.toggleAlgo();
      return;
    }
    switch (event.key) {
      case "Escape":
        event.preventDefault();
        this.clearSelection();
        this.ui.hideTooltip();
        break;
      case " ":
      case "Spacebar":
        event.preventDefault();
        this.togglePlay();
        break;
      case "ArrowRight":
        event.preventDefault();
        this.stepFlow(1);
        break;
      case "ArrowLeft":
        event.preventDefault();
        this.stepFlow(-1);
        break;
      case "f":
      case "F":
        event.preventDefault();
        this.scene.fit();
        break;
      case "r":
      case "R":
        event.preventDefault();
        this.resetView();
        break;
      case "l":
      case "L":
        event.preventDefault();
        this.toggle("labels");
        break;
      case "g":
      case "G":
        event.preventDefault();
        this.toggle("groups");
        break;
      case "k":
      case "K":
        event.preventDefault();
        this.toggle("kernels");
        break;
      case "e":
      case "E":
        event.preventDefault();
        this.toggle("edges");
        break;
      case "t":
      case "T":
        event.preventDefault();
        this.ui.toggleTable();
        break;
      case "/":
        event.preventDefault();
        this.ui.focusSearch();
        break;
      case "?":
        event.preventDefault();
        this.ui.openHelp();
        break;
      default:
        break;
    }
  }

  dispose() {
    this.stopFlow({ silent: true });
    this._unsubscribeFrame?.();
    this.canvas.removeEventListener("pointermove", this._onPointerMove);
    this.canvas.removeEventListener("pointerleave", this._onPointerLeave);
    this.canvas.removeEventListener("pointerdown", this._onPointerDown);
    this.canvas.removeEventListener("pointerup", this._onPointerUp);
    this.canvas.removeEventListener("dblclick", this._onDoubleClick);
    this.canvas.removeEventListener("contextmenu", this._onContextMenu);
    window.removeEventListener("keydown", this._onKeyDown);
    window.removeEventListener("blur", this._onBlur);
    if (this._pickRaf) cancelAnimationFrame(this._pickRaf);
  }
}
