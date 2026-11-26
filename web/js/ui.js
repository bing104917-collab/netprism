/* DOM layer: top bar, side panels, layer tree, legend, details, tooltip, states. */

import {
  EDGE_META, KIND_META, KIND_ORDER, edgeMeta, escapeHtml, formatCount, formatFlops,
  formatShape, formatShapeCompact, formatParamValue, groupMeta, kindColor, kindLabel,
  modeMeta, opSummary, opName, opParams, paramEntries, truncate, shapePills, formatInteger, clamp,
  activationMBValue, formatActivationMB, formatMacs, formatReceptiveField, frameworkMeta,
  isMetricNumber, modelMetricValue, nodeMetricValue, normalizeFramework,
} from "./format.js";

function icon(name, cls = "icon") {
  return `<svg class="${cls}" aria-hidden="true"><use href="#i-${name}"></use></svg>`;
}

function pct(value) {
  return `${clamp(value, 0, 1) * 100}%`;
}

function basename(path) {
  const text = String(path || "");
  const parts = text.split(/[\\/]/);
  return parts[parts.length - 1] || text;
}

function fileNameLine(source) {
  if (!source) return "";
  const path = source.relPath || source.file || "";
  return source.line ? `${path}:${source.line}` : path;
}

const STATE_ICON = {
  loading: null,
  error: "warning",
  empty: "cube",
};

export class UI {
  constructor() {
    this.el = {
      app: document.getElementById("app"),
      modelTitle: document.getElementById("model-title"),
      modelSubtitle: document.getElementById("model-subtitle"),
      frameworkBadge: document.getElementById("framework-badge"),
      modeBadge: document.getElementById("mode-badge"),
      modelSelect: document.getElementById("model-select"),
      sourceBadge: document.getElementById("source-badge"),
      sourceText: document.querySelector("#source-badge .source-text"),
      uploadBtn: document.getElementById("upload-btn"),
      uploadLabel: document.querySelector("#upload-btn .upload-label"),
      exportBtn: document.getElementById("export-btn"),
      fileInput: document.getElementById("file-input"),
      helpBtn: document.getElementById("help-btn"),
      wrap: document.getElementById("viewport")?.closest(".app"),
      summary: document.getElementById("summary"),
      search: document.getElementById("layer-search"),
      searchClear: document.getElementById("search-clear"),
      tree: document.getElementById("layer-tree"),
      treeCount: document.getElementById("tree-count"),
      detail: document.getElementById("detail-panel"),
      detailMeta: document.getElementById("detail-meta"),
      legend: document.getElementById("legend-panel"),
      legendMeta: document.getElementById("legend-meta"),
      stateOverlay: document.getElementById("state-overlay"),
      dropOverlay: document.getElementById("drop-overlay"),
      tooltip: document.getElementById("tooltip"),
      toastStack: document.getElementById("toast-stack"),
      notice: document.getElementById("notice-panel"),
      helpModal: document.getElementById("help-modal"),
      bottombar: document.getElementById("bottombar"),
      playBtn: document.getElementById("play-btn"),
      playIcon: document.querySelector("#play-btn .bb-play-icon use"),
      playLabel: document.querySelector("#play-btn .bb-label"),
      flowHud: document.getElementById("flow-hud"),
      flowLabel: document.getElementById("flow-label"),
      flowCount: document.getElementById("flow-count"),
      flowFill: document.getElementById("flow-fill"),
      bigBadge: document.getElementById("big-badge"),
      onboarding: document.getElementById("onboarding-hint"),
      stageMeta: document.getElementById("stage-meta"),
      leftToggle: document.getElementById("toggle-left"),
      rightToggle: document.getElementById("toggle-right"),
      topProgress: document.getElementById("top-progress"),
      tableToggle: document.getElementById("table-toggle"),
      tableView: document.getElementById("table-view"),
      tableBody: document.getElementById("table-body"),
      tableSearch: document.getElementById("table-search"),
      tableExport: document.getElementById("table-export"),
      tableClose: document.getElementById("table-close"),
      tableCount: document.getElementById("table-count"),
      tableEmpty: document.getElementById("table-empty"),
      tableFootMeta: document.getElementById("table-foot-meta"),
      tableFramework: document.getElementById("table-framework"),
      algoToggle: document.getElementById("algo-toggle"),
      algoView: document.getElementById("algo-view"),
      algoFamily: document.getElementById("algo-family"),
      algoPatterns: document.getElementById("algo-patterns"),
      algoPseudocode: document.getElementById("algo-pseudocode"),
      algoCopy: document.getElementById("algo-copy"),
      algoClose: document.getElementById("algo-close"),
      algoModelName: document.getElementById("algo-model-name"),
      algoPatternCount: document.getElementById("algo-pattern-count"),
      algoFootMeta: document.getElementById("algo-foot-meta"),
    };
    this.callbacks = {};
    this.current = null;
    this.groupById = new Map();
    this.hiddenKinds = new Set();
    this.searchQuery = "";
    this.collapsedGroups = new Set();
    this.toastTimers = new Map();
    this.noticeTimer = null;
    this.helpOpen = false;
    this.stateMode = null;
    this.playState = { playing: false, active: false, index: 0, total: 0, name: "" };
    this.tableOpen = false;
    this.tableQuery = "";
    this.tableSort = { key: null, dir: 1 };
    this.tableRows = [];
    this.tableFiltered = [];
    this.tableSearchTimer = null;
    this.algoOpen = false;
  }

  bind(callbacks) {
    this.callbacks = callbacks || {};
    const on = (el, event, handler) => el && el.addEventListener(event, handler);

    on(this.el.modelSelect, "change", (event) => {
      const id = event.target.value;
      if (id) this.callbacks.onModelChange?.(id);
    });

    on(this.el.uploadBtn, "click", () => {
      if (this.el.uploadBtn.classList.contains("is-busy")) return;
      this.el.fileInput?.click();
    });

    on(this.el.exportBtn, "click", () => this.callbacks.onExport?.());

    on(this.el.fileInput, "change", (event) => {
      const files = event.target.files;
      if (files && files.length) this.callbacks.onFilesSelected?.(Array.from(files));
      event.target.value = "";
    });

    on(this.el.helpBtn, "click", () => this.openHelp());
    document.querySelectorAll("[data-close-help]").forEach((el) => {
      el.addEventListener("click", () => this.closeHelp());
    });

    on(this.el.tableToggle, "click", () => this.toggleTable());
    on(this.el.tableClose, "click", () => this.closeTable());
    on(this.el.algoToggle, "click", () => this.toggleAlgo());
    on(this.el.algoClose, "click", () => this.closeAlgo());
    on(this.el.algoCopy, "click", () => this.copyPseudocode());
    on(this.el.tableSearch, "input", () => {
      clearTimeout(this.tableSearchTimer);
      this.tableSearchTimer = setTimeout(() => this.filterTable(this.el.tableSearch.value), 45);
    });
    on(this.el.tableSearch, "keydown", (event) => {
      if (event.key === "Escape") {
        event.preventDefault();
        if (this.el.tableSearch.value) {
          this.el.tableSearch.value = "";
          this.filterTable("");
        } else {
          this.el.tableSearch.blur();
          this.closeTable();
        }
      }
    });
    on(this.el.tableBody?.closest("table"), "click", (event) => {
      const header = event.target.closest("th[data-sort]");
      if (header) this.sortTable(header.dataset.sort);
    });
    on(this.el.tableBody?.closest("table"), "keydown", (event) => {
      if (event.key !== "Enter" && event.key !== " ") return;
      const header = event.target.closest("th[data-sort]");
      if (!header) return;
      event.preventDefault();
      this.sortTable(header.dataset.sort);
    });
    on(this.el.tableExport, "click", () => this.exportTableCsv());

    on(this.el.search, "input", () => {
      this.filterTree(this.el.search.value);
    });
    on(this.el.search, "keydown", (event) => {
      if (event.key === "Escape") {
        event.preventDefault();
        this.clearSearch(true);
      }
    });
    on(this.el.searchClear, "click", () => this.clearSearch(true));

    on(this.el.tree, "click", (event) => {
      const toggle = event.target.closest("[data-group-toggle]");
      if (toggle) {
        const section = toggle.closest(".tree-group");
        if (!section) return;
        const id = section.dataset.groupId;
        const collapsed = section.classList.toggle("is-collapsed");
        toggle.setAttribute("aria-expanded", String(!collapsed));
        const key = this.groupKey(id);
        if (collapsed) this.collapsedGroups.add(key);
        else this.collapsedGroups.delete(key);
        return;
      }
      const row = event.target.closest("[data-node-id]");
      if (row) this.callbacks.onTreeSelect?.(row.dataset.nodeId);
    });
    on(this.el.tree, "keydown", (event) => {
      if (event.key !== "Enter" && event.key !== " ") return;
      const toggle = event.target.closest("[data-group-toggle]");
      if (toggle) {
        event.preventDefault();
        toggle.click();
      }
    });

    on(this.el.legend, "click", (event) => {
      const item = event.target.closest("[data-kind]");
      if (item) {
        this.callbacks.onKindToggle?.(item.dataset.kind);
        return;
      }
      if (event.target.closest("[data-legend-reset]")) this.callbacks.onShowAllKinds?.();
    });

    on(this.el.detail, "click", (event) => {
      const nav = event.target.closest("[data-nav]");
      if (nav) {
        this.callbacks.onNavigate?.(Number(nav.dataset.nav));
        return;
      }
      if (event.target.closest("[data-focus]")) this.callbacks.onFocusSelected?.();
      if (event.target.closest("[data-upload-action]")) this.el.fileInput?.click();
    });

    on(this.el.bottombar, "click", (event) => {
      const action = event.target.closest("[data-action]");
      if (action) {
        const name = action.dataset.action;
        if (name === "play") this.callbacks.onPlayToggle?.();
        else if (name === "step") this.callbacks.onStep?.(1);
        else if (name === "reset") this.callbacks.onReset?.();
        else if (name === "fit") this.callbacks.onFit?.();
        return;
      }
      const toggle = event.target.closest("[data-toggle]");
      if (toggle) this.callbacks.onToggle?.(toggle.dataset.toggle);
    });

    on(this.el.leftToggle, "click", () => this.callbacks.onTogglePanel?.("left"));
    on(this.el.rightToggle, "click", () => this.callbacks.onTogglePanel?.("right"));

    on(this.el.stateOverlay, "click", (event) => {
      const action = event.target.closest("[data-state-action]");
      if (action) this.callbacks.onStateAction?.(action.dataset.stateAction);
    });

    on(this.el.notice, "click", (event) => {
      if (event.target.closest("[data-notice-close]")) this.hideNotice();
    });

    on(this.el.toastStack, "click", (event) => {
      const close = event.target.closest("[data-toast-close]");
      if (close) this.dismissToast(close.closest(".toast"));
    });

    on(window, "keydown", (event) => {
      if (event.key === "Escape" && this.helpOpen) {
        event.preventDefault();
        this.closeHelp();
      }
    });
    on(window, "resize", () => this._updatePanelToggleLabels());
    this._updatePanelToggleLabels();
  }

  /* ------------------------------------------------------------------ */
  /* Top bar & model switching                                           */
  /* ------------------------------------------------------------------ */

  setIndex(index, source) {
    this.index = index;
    const models = index && Array.isArray(index.models) ? index.models : [];
    const select = this.el.modelSelect;
    if (!select) return;
    select.innerHTML = "";
    if (!models.length) {
      const option = document.createElement("option");
      option.textContent = "暂无模型";
      option.value = "";
      select.appendChild(option);
      select.disabled = true;
    } else {
      for (const model of models) {
        const option = document.createElement("option");
        option.value = model.id;
        const mode = modeMeta(model.mode).short;
        option.textContent = `${model.name || model.id}${mode ? ` · ${mode}` : ""}`;
        select.appendChild(option);
      }
      select.disabled = false;
    }
    this.setSource(source);
  }

  setSource(source) {
    const meta = {
      api: { text: "本地 API", title: "已连接 NetPrism 解析服务" },
      demo: { text: "演示数据", title: "未连接服务，使用 web/demo 内置示例" },
      upload: { text: "上传解析", title: "当前模型来自刚刚上传的文件夹（静态解析）" },
    }[source] || { text: "未知来源", title: "" };
    if (this.el.sourceBadge) {
      this.el.sourceBadge.dataset.source = source || "demo";
      this.el.sourceBadge.title = meta.title;
    }
    if (this.el.sourceText) this.el.sourceText.textContent = meta.text;
  }

  setBusy(busy) {
    this.el.app?.classList.toggle("is-busy", Boolean(busy));
  }

  setUploading(uploading) {
    if (!this.el.uploadBtn) return;
    this.el.uploadBtn.classList.toggle("is-busy", Boolean(uploading));
    this.el.uploadBtn.disabled = Boolean(uploading);
    if (this.el.uploadLabel) this.el.uploadLabel.textContent = uploading ? "解析中…" : "上传文件夹";
  }

  setModel(graph, entry, layout, source, extras = {}) {
    this.current = { graph, entry, layout, source };
    this.groupById = new Map((graph.groups || []).map((group) => [group.id, group]));
    this.groupInfoById = new Map((layout.groups || []).map((info) => [info.group.id, info]));
    this.searchQuery = "";
    if (this.el.search) this.el.search.value = "";
    if (this.el.searchClear) this.el.searchClear.hidden = true;
    this.el.tree?.classList.remove("is-searching");

    const model = graph.model || {};
    const name = model.name || entry.name || entry.id;
    if (this.el.modelTitle) this.el.modelTitle.textContent = name;
    if (this.el.modelSubtitle) {
      const mode = modeMeta(model.mode || entry.mode).label;
      const file = model.source?.relPath || entry.source || "";
      const className = model.className || entry.className;
      this.el.modelSubtitle.textContent = [className, mode, file].filter(Boolean).join("  ·  ");
      this.el.modelSubtitle.title = model.source?.file || entry.file || "";
    }
    if (this.el.modeBadge) {
      const mode = model.mode || entry.mode || "static";
      this.el.modeBadge.hidden = false;
      this.el.modeBadge.dataset.mode = mode;
      this.el.modeBadge.textContent = modeMeta(mode).label;
    }
    this.setFrameworkBadge(model.framework, model.meta);
    if (this.el.modelSelect && entry.id) this.el.modelSelect.value = entry.id;
    this.setSource(source);
    this.setBigModel(layout);
    this.renderSummary();
    this.renderTree();
    this.renderLegend();
    this.renderDetail(null);
    this.hideNotice();
    this.setState(null);
    this.setPlayState({ playing: false, active: false, index: 0, total: 0, name: "" });
    this.el.detail?.scrollTo?.({ top: 0 });
    document.title = `NetPrism · ${name}`;
    this.hideTooltip();
    if (this.el.onboarding) this.el.onboarding.classList.remove("is-hidden");
    this._autoHideOnboarding();
    this._updatePanelToggleLabels();

    if (this.el.tableToggle) this.el.tableToggle.disabled = false;
    if (this.el.algoToggle) this.el.algoToggle.disabled = false;
    this.tableQuery = "";
    this.tableSort = { key: null, dir: 1 };
    if (this.el.tableSearch) this.el.tableSearch.value = "";
    this.updateTableSortIndicators();
    if (this.tableOpen) this.renderTable();
    if (this.algoOpen) this.renderAlgo();
  }

  setFrameworkBadge(framework, meta) {
    const badge = this.el.frameworkBadge;
    if (!badge) return;
    const normalized = normalizeFramework(framework);
    if (!normalized) {
      badge.hidden = true;
      badge.textContent = "";
      return;
    }
    const info = frameworkMeta(normalized);
    badge.hidden = false;
    badge.dataset.framework = normalized;
    badge.textContent = info.label;
    const metaBits = [];
    if (meta && typeof meta === "object") {
      if (meta.opset != null) metaBits.push(`opset ${meta.opset}`);
      if (meta.kerasVersion || meta.keras_version) {
        metaBits.push(`keras ${meta.kerasVersion || meta.keras_version}`);
      }
      if (meta.torchVersion || meta.torch_version) {
        metaBits.push(`torch ${meta.torchVersion || meta.torch_version}`);
      }
      if (meta.sourceFormat || meta.source_format) {
        metaBits.push(String(meta.sourceFormat || meta.source_format));
      }
    }
    badge.title = metaBits.length ? `${info.label} · ${metaBits.join(" · ")}` : info.label;
  }

  setBigModel(layout) {
    if (!this.el.bigBadge) return;
    if (layout && layout.big) {
      this.el.bigBadge.hidden = false;
      this.el.bigBadge.innerHTML = `${icon("bolt")}<span>大模型模式 · ${layout.nodeCount} 节点</span>`;
      this.el.bigBadge.title = "已自动简化标签与几何，以保持交互流畅";
    } else {
      this.el.bigBadge.hidden = true;
    }
  }

  _autoHideOnboarding() {
    clearTimeout(this._onboardingTimer);
    this._onboardingTimer = setTimeout(() => {
      this.el.onboarding?.classList.add("is-hidden");
    }, 9000);
  }

  hideOnboarding() {
    this.el.onboarding?.classList.add("is-hidden");
  }

  /* ------------------------------------------------------------------ */
  /* Summary                                                            */
  /* ------------------------------------------------------------------ */

  showSkeleton() {
    if (this.el.summary) {
      this.el.summary.innerHTML = `
        <div class="summary-placeholder">
          <div class="placeholder-inner">
            <div class="skeleton" style="height:12px;width:42%"></div>
            <div class="placeholder-row">
              <div class="skeleton" style="height:46px"></div>
              <div class="skeleton" style="height:46px"></div>
              <div class="skeleton" style="height:46px"></div>
              <div class="skeleton" style="height:46px"></div>
            </div>
            <div class="skeleton" style="height:10px;width:82%"></div>
            <div class="skeleton" style="height:10px;width:64%"></div>
          </div>
        </div>`;
    }
    if (this.el.tree) {
      this.el.tree.innerHTML = `<div class="tree-empty">${icon("layers")}正在读取模型结构…</div>`;
    }
    if (this.el.treeCount) this.el.treeCount.textContent = "—";
    if (this.el.legend) {
      this.el.legend.innerHTML = `<div class="legend-note">加载模型后显示图例。</div>`;
    }
    if (this.el.legendMeta) this.el.legendMeta.textContent = "—";
    if (this.el.detail) {
      this.el.detail.innerHTML = `<div class="detail-guide"><h3>准备场景</h3><p>正在构建 3D 张量布局…</p></div>`;
    }
    if (this.el.detailMeta) this.el.detailMeta.textContent = "—";
  }

  renderSummary() {
    if (!this.current) return;
    const { graph, layout, entry } = this.current;
    const model = graph.model || {};
    const warnings = Array.isArray(model.warnings) ? model.warnings : [];
    const inputSpecs = Array.isArray(model.inputSpec) ? model.inputSpec : [];
    const outputSpecs = Array.isArray(model.outputSpec) ? model.outputSpec : [];
    const params = Number(model.totalParams) || 0;
    const trainable = Number(model.trainableParams);
    const flops = Number(model.totalFlops) || 0;
    const layers = Number(model.numLayers) || 0;
    const groups = Array.isArray(graph.groups) ? graph.groups.length : 0;

    const inputHtml = inputSpecs.length
      ? inputSpecs.map((spec) =>
          `<span class="shape-pill" title="${escapeHtml(spec.name || "input")} · ${escapeHtml(spec.dtype || "")}">${escapeHtml(formatShapeCompact(spec.shape))}</span>`,
        ).join("")
      : `<span class="shape-empty">—</span>`;
    const outputHtml = outputSpecs.length
      ? outputSpecs.map((spec) =>
          `<span class="shape-pill is-out" title="${escapeHtml(spec.name || "output")} · ${escapeHtml(spec.dtype || "")}">${escapeHtml(formatShapeCompact(spec.shape))}</span>`,
        ).join("")
      : `<span class="shape-empty">—</span>`;

    const warningsHtml = warnings.length
      ? `<details class="sum-warnings">
          <summary>${icon("warning")}解析警告<span class="warn-count">${warnings.length}</span></summary>
          <ul class="sum-warning-list">${warnings.map((w) => `<li>${escapeHtml(w)}</li>`).join("")}</ul>
        </details>`
      : `<div class="sum-ok">${icon("check")}解析完成，无警告</div>`;

    const file = model.source?.relPath || model.source?.file || entry?.source || entry?.file || "";
    const framework = normalizeFramework(model.framework);
    const frameworkInfo = frameworkMeta(framework);
    const metrics = model.metrics && typeof model.metrics === "object" ? model.metrics : null;
    const activationBytes = isMetricNumber(metrics?.activationBytes)
      ? metrics.activationBytes
      : isMetricNumber(metrics?.peakActivationBytes)
        ? metrics.peakActivationBytes
        : null;
    const activationIsPeak =
      !isMetricNumber(metrics?.activationBytes) && isMetricNumber(metrics?.peakActivationBytes);
    const receptiveField = modelMetricValue(metrics, "maxReceptiveField");
    const macs = modelMetricValue(metrics, "macs");
    const hasModelMetrics = Boolean(
      metrics && (activationBytes != null || receptiveField != null || macs != null),
    );
    const metricChips = `
      <div class="summary-metrics" id="summary-metrics" ${hasModelMetrics ? "" : "hidden"}>
        <div class="summary-metric-chip" id="metric-activation" ${activationBytes == null ? "hidden" : ""}
          title="${activationBytes == null ? "" : `${activationIsPeak ? "peak" : "activation"} ${escapeHtml(String(activationBytes))} bytes`}">
          <span class="smc-k">${activationIsPeak ? "峰值激活" : "激活显存"} <span class="metric-approx">估算</span></span>
          <b>${formatActivationMB(activationBytes)}</b>
        </div>
        <div class="summary-metric-chip" id="metric-rf" ${receptiveField == null ? "hidden" : ""}
          title="${receptiveField == null ? "" : "receptiveField = " + escapeHtml(String(receptiveField))}">
          <span class="smc-k">感受野 <span class="metric-approx">估算</span></span>
          <b>${formatReceptiveField(receptiveField)}</b>
        </div>
        <div class="summary-metric-chip" id="metric-macs" ${macs == null ? "hidden" : ""}
          title="macs = flops / 2">
          <span class="smc-k">MACs <span class="metric-approx">估算</span></span>
          <b>${formatMacs(macs)}</b>
        </div>
      </div>`;

    this.el.summary.innerHTML = `
      <div class="summary-card">
        <div class="summary-head">
          <span class="summary-head-label">${icon("cube")}模型摘要</span>
          <span class="summary-head-tags">
            ${framework ? `<span class="summary-framework" data-framework="${escapeHtml(framework)}" title="${escapeHtml(frameworkInfo.label)}">${escapeHtml(frameworkInfo.label)}</span>` : ""}
            <span class="summary-mode" title="解析模式">${escapeHtml(modeMeta(model.mode || entry?.mode).label)}</span>
          </span>
        </div>
        <div class="metric-grid">
          <div class="sum-metric" data-accent="params">
            <span class="sum-metric-k">参数量</span>
            <span class="sum-metric-v" title="${escapeHtml(formatInteger(params))} 参数">${formatCount(params)}</span>
          </div>
          <div class="sum-metric" data-accent="flops">
            <span class="sum-metric-k">FLOPs</span>
            <span class="sum-metric-v" title="${escapeHtml(formatInteger(flops))} FLOPs">${model.flopsApprox ? "≈" : ""}${formatFlops(flops)}</span>
          </div>
          <div class="sum-metric" data-accent="layers">
            <span class="sum-metric-k">叶子层数</span>
            <span class="sum-metric-v">${layers}<small>层</small></span>
          </div>
          <div class="sum-metric" data-accent="depth">
            <span class="sum-metric-k">场景节点</span>
            <span class="sum-metric-v">${layout.nodeCount}<small>· ${groups} 组</small></span>
          </div>
        </div>
        <div class="sum-specs">
          <div class="sum-spec"><span class="sum-spec-k">输入</span><span class="sum-shapes">${inputHtml}</span></div>
          <div class="sum-spec"><span class="sum-spec-k">输出</span><span class="sum-shapes">${outputHtml}</span></div>
          ${trainable >= 0 && trainable !== params ? `<div class="sum-spec"><span class="sum-spec-k">可训练</span><span class="sum-shapes"><span class="shape-pill">${formatCount(trainable)}</span></span></div>` : ""}
        </div>
        ${metricChips}
        ${file ? `<div class="sum-file" title="${escapeHtml(model.source?.file || file)}">${icon("code")}<span>${escapeHtml(file)}</span></div>` : ""}
        ${warningsHtml}
      </div>`;
  }

  /* ------------------------------------------------------------------ */
  /* Layer tree                                                         */
  /* ------------------------------------------------------------------ */

  groupKey(id) {
    const modelId = this.current?.entry?.id || "model";
    return `${modelId}:${id}`;
  }

  groupDisplayLabel(group) {
    if (!group) return "";
    const info = this.groupInfoById?.get(group.id);
    return info?.displayLabel || group.label || group.name || group.id;
  }

  renderTree() {
    if (!this.current) return;
    const { graph, layout } = this.current;
    const groups = Array.isArray(graph.groups) ? graph.groups : [];
    const groupsById = new Map(groups.map((group) => [group.id, group]));
    const infoById = new Map((layout.groups || []).map((info) => [info.group.id, info]));
    const childrenOf = new Map();
    for (const group of groups) {
      const parentId = group.parentId && groupsById.has(group.parentId) ? group.parentId : null;
      if (!childrenOf.has(parentId)) childrenOf.set(parentId, []);
      childrenOf.get(parentId).push(group);
    }
    for (const list of childrenOf.values()) {
      list.sort((a, b) => (infoById.get(a.id)?.firstOrder ?? 1e9) - (infoById.get(b.id)?.firstOrder ?? 1e9));
    }
    const directByGroup = new Map();
    const ungrouped = [];
    for (const item of layout.nodes) {
      const gid = item.groupId && groupsById.has(item.groupId) ? item.groupId : null;
      if (!gid) ungrouped.push(item);
      else {
        if (!directByGroup.has(gid)) directByGroup.set(gid, []);
        directByGroup.get(gid).push(item);
      }
    }
    for (const list of directByGroup.values()) list.sort((a, b) => a.order - b.order);

    const rootGroups = childrenOf.get(null) || [];
    const entries = [];
    for (const group of rootGroups) {
      if (group.kind === "model") {
        const direct = directByGroup.get(group.id) || [];
        for (const item of direct) entries.push({ type: "node", item, order: item.order });
        for (const child of childrenOf.get(group.id) || []) {
          entries.push({ type: "group", group: child, order: infoById.get(child.id)?.firstOrder ?? 1e9 });
        }
      } else {
        entries.push({ type: "group", group, order: infoById.get(group.id)?.firstOrder ?? 1e9 });
      }
    }
    for (const item of ungrouped) entries.push({ type: "node", item, order: item.order });
    entries.sort((a, b) => a.order - b.order);

    const renderNodeRow = (item) => {
      const node = item.node;
      const kind = item.kind;
      const display = opName(node) || item.id;
      const detail = [node.name, opParams(node)].filter(Boolean).join(" · ");
      const search = `${node.name} ${node.op} ${display} ${kindLabel(kind)} ${item.groupId || ""} ${node.tags?.join(" ") || ""}`.toLowerCase();
      const hidden = this.hiddenKinds.has(kind);
      return `
        <button class="tree-node${hidden ? " is-filtered" : ""}" type="button"
          data-node-id="${escapeHtml(item.id)}" data-kind="${escapeHtml(kind)}"
          data-search="${escapeHtml(search)}" style="--c:${kindColor(kind)}"
          title="${escapeHtml(display)} · ${escapeHtml(node.name || "")}">
          <span class="tn-dot"></span>
          <span class="tn-main">
            <span class="tn-name">${escapeHtml(display)}</span>
            <span class="tn-sub">${escapeHtml(detail)}</span>
          </span>
          ${Number(node.numParams) > 0 ? `<span class="tn-params">${formatCount(node.numParams)}</span>` : ""}
        </button>`;
    };

    const renderGroup = (group, depth) => {
      const info = infoById.get(group.id);
      const childGroups = childrenOf.get(group.id) || [];
      const direct = directByGroup.get(group.id) || [];
      const key = this.groupKey(group.id);
      const collapsed = this.collapsedGroups.has(key);
      const meta = groupMeta(group.kind);
      const body = [
        ...direct.map((item) => ({ order: item.order, html: renderNodeRow(item) })),
        ...childGroups.map((child) => ({
          order: infoById.get(child.id)?.firstOrder ?? 1e9,
          html: renderGroup(child, depth + 1),
        })),
      ]
        .sort((a, b) => a.order - b.order)
        .map((entry) => entry.html)
        .join("");
      if (!body) return "";
      return `
        <section class="tree-group${collapsed ? " is-collapsed" : ""}" data-group-id="${escapeHtml(group.id)}">
          <button class="tg-head" type="button" data-group-toggle aria-expanded="${!collapsed}" title="${escapeHtml(group.name || group.id)}">
            ${icon("chevron", "icon tg-chevron")}
            <span class="tg-dot" style="--c:${meta.color}"></span>
            <span class="tg-label">${escapeHtml(info?.displayLabel || group.label || group.name || group.id)}</span>
            <span class="tg-kind">${escapeHtml(meta.label)}</span>
            <span class="tg-meta">${info?.descendantCount ?? direct.length} 层 · ${formatCount(group.numParams || 0)}</span>
          </button>
          <div class="tg-body">${body}</div>
        </section>`;
    };

    const html = entries.map((entry) =>
      entry.type === "node" ? renderNodeRow(entry.item) : renderGroup(entry.group, 0),
    ).join("");

    this.el.tree.innerHTML =
      html +
      `<div class="tree-empty tree-no-match" hidden>${icon("search")}没有匹配的层<br /><span>试试层名、算子（Conv2d）或 kind</span></div>`;
    this.el.tree.classList.remove("is-searching");
    this.el.treeCount.textContent = `${layout.nodeCount} 层`;
    this.applyTreeSearch(this.searchQuery);
  }

  setTreeSelected(id, { scroll = true } = {}) {
    if (!this.el.tree) return;
    for (const row of this.el.tree.querySelectorAll(".tree-node.is-selected")) {
      row.classList.remove("is-selected");
    }
    if (!id) return;
    const row = this.el.tree.querySelector(`.tree-node[data-node-id="${CSS.escape(id)}"]`);
    if (!row) return;
    row.classList.add("is-selected");
    let section = row.closest(".tree-group");
    while (section) {
      if (section.classList.contains("is-collapsed")) {
        section.classList.remove("is-collapsed");
        const toggle = section.querySelector(":scope > .tg-head");
        toggle?.setAttribute("aria-expanded", "true");
        this.collapsedGroups.delete(this.groupKey(section.dataset.groupId));
      }
      section = section.parentElement?.closest(".tree-group") || null;
    }
    if (scroll) row.scrollIntoView({ block: "nearest" });
  }

  filterTree(query) {
    this.searchQuery = typeof query === "string" ? query : "";
    this.applyTreeSearch(this.searchQuery);
  }

  applyTreeSearch(query) {
    if (!this.el.tree) return;
    const q = String(query || "").trim().toLowerCase();
    const clear = this.el.searchClear;
    if (clear) clear.hidden = !q;
    const rows = Array.from(this.el.tree.querySelectorAll(".tree-node"));
    const total = rows.length;
    if (!q) {
      for (const row of rows) {
        row.classList.remove("is-search-hidden", "is-match");
      }
      for (const section of this.el.tree.querySelectorAll(".tree-group")) {
        section.classList.remove("is-search-hidden");
      }
      const empty = this.el.tree.querySelector(".tree-no-match");
      if (empty) empty.hidden = true;
      this.el.tree.classList.remove("is-searching");
      this.el.treeCount.textContent = `${this.current?.layout?.nodeCount ?? total} 层`;
      return;
    }
    this.el.tree.classList.add("is-searching");
    let matches = 0;
    for (const row of rows) {
      const text = row.dataset.search || "";
      const hit = text.includes(q);
      row.classList.toggle("is-search-hidden", !hit);
      row.classList.toggle("is-match", hit);
      if (hit) matches += 1;
    }
    for (const section of this.el.tree.querySelectorAll(".tree-group")) {
      const visible = section.querySelectorAll(".tree-node:not(.is-search-hidden)").length > 0;
      section.classList.toggle("is-search-hidden", !visible);
    }
    const empty = this.el.tree.querySelector(".tree-no-match");
    if (empty) empty.hidden = matches > 0;
    this.el.treeCount.textContent = `${matches} / ${total} 层`;
  }

  clearSearch(blur = false) {
    if (this.el.search) {
      this.el.search.value = "";
      if (blur) this.el.search.blur();
    }
    this.filterTree("");
  }

  focusSearch() {
    this.el.search?.focus();
    this.el.search?.select();
  }

  setHiddenKinds(hiddenKinds) {
    this.hiddenKinds = new Set(hiddenKinds || []);
  }

  refreshTreeFiltered(hiddenKinds) {
    this.setHiddenKinds(hiddenKinds);
    if (!this.el.tree) return;
    for (const row of this.el.tree.querySelectorAll(".tree-node")) {
      row.classList.toggle("is-filtered", this.hiddenKinds.has(row.dataset.kind));
    }
  }

  /* ------------------------------------------------------------------ */
  /* Legend                                                             */
  /* ------------------------------------------------------------------ */

  renderLegend() {
    if (!this.current) return;
    const { graph } = this.current;
    const counts = new Map();
    for (const node of graph.nodes || []) {
      counts.set(node.kind, (counts.get(node.kind) || 0) + 1);
    }
    const visibleKinds = [...counts.keys()].filter((kind) => !this.hiddenKinds.has(kind)).length;
    if (this.el.legendMeta) {
      this.el.legendMeta.textContent = this.hiddenKinds.size
        ? `${visibleKinds} / ${counts.size} 类可见`
        : "全部可见";
    }
    const kinds = KIND_ORDER.filter((kind) => counts.has(kind));
    for (const kind of counts.keys()) {
      if (!kinds.includes(kind)) kinds.push(kind);
    }
    const rows = kinds.map((kind) => {
      const meta = KIND_META[kind] || KIND_META.unknown;
      const off = this.hiddenKinds.has(kind);
      return `
        <button class="legend-item${off ? " is-off" : ""}" type="button" data-kind="${escapeHtml(kind)}"
          style="--c:${meta.color}" aria-pressed="${!off}" title="点击${off ? "显示" : "隐藏"} ${escapeHtml(meta.label)}">
          <span class="legend-swatch"></span>
          <span class="legend-label">${escapeHtml(meta.label)}</span>
          <span class="legend-code">${escapeHtml(kind)}</span>
          <span class="legend-count">${counts.get(kind)}</span>
        </button>`;
    }).join("");

    const edgeRows = ["data", "residual", "skip", "attention"]
      .map((kind) => {
        const meta = EDGE_META[kind];
        return `<span class="edge-legend-item" style="--c:${meta.color}"><span class="edge-swatch"></span>${escapeHtml(meta.label)}</span>`;
      })
      .join("");

    this.el.legend.innerHTML = `
      <div class="legend-toolbar">
        <span class="legend-note">点击图例，隐藏 / 显示对应算子</span>
        <button class="legend-reset" type="button" data-legend-reset ${this.hiddenKinds.size ? "" : "hidden"}>全部显示</button>
      </div>
      <div class="legend-list">${rows || `<div class="legend-note">没有可显示的层。</div>`}</div>
      <div class="edge-legend">${edgeRows}</div>
      <div class="legend-foot">颜色 = 算子 kind；X = 执行顺序，Y = 高 / 序列长度，Z = 宽 / log(特征)。卷积核在输入侧，注意力按 head 分层。attention 内部连边默认隐藏，选中注意力层后显示。</div>`;
  }

  /* ------------------------------------------------------------------ */
  /* Algorithm understanding panel (v2.1)                               */
  /* ------------------------------------------------------------------ */

  isAlgoOpen() {
    return this.algoOpen;
  }

  toggleAlgo() {
    if (this.algoOpen) this.closeAlgo();
    else this.openAlgo();
  }

  openAlgo() {
    if (!this.current || !this.el.algoView) {
      this.toast("info", "暂无模型", "加载模型后即可查看算法理解。");
      return;
    }
    this.closeTable();
    this.algoOpen = true;
    this.el.algoView.hidden = false;
    this.el.app?.classList.add("is-algo-open");
    this.el.algoToggle?.setAttribute("aria-pressed", "true");
    this.hideTooltip();
    this.el.onboarding?.classList.add("is-hidden");
    this.renderAlgo();
  }

  closeAlgo() {
    this.algoOpen = false;
    if (this.el.algoView) this.el.algoView.hidden = true;
    this.el.app?.classList.remove("is-algo-open");
    this.el.algoToggle?.setAttribute("aria-pressed", "false");
  }

  _algorithmConfidence(value) {
    const number = Number(value);
    if (!Number.isFinite(number)) return null;
    if (number > 1) return clamp(number / 100, 0, 1);
    return clamp(number, 0, 1);
  }

  _asStringList(value) {
    if (Array.isArray(value)) {
      return value.map((entry) => String(entry == null ? "" : entry).trim()).filter(Boolean);
    }
    if (typeof value === "string" && value.trim()) {
      return value.split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
    }
    return [];
  }

  _pseudocodeLines(value) {
    if (Array.isArray(value)) {
      return value.map((line) => String(line == null ? "" : line)).filter((line) => line.trim().length);
    }
    if (typeof value === "string" && value.trim()) {
      return value.replace(/\r\n/g, "\n").split("\n").filter((line) => line.trim().length);
    }
    return [];
  }

  renderAlgo() {
    if (!this.current || !this.el.algoView) return;
    const { graph, entry } = this.current;
    const model = graph.model || {};
    const meta = model.meta && typeof model.meta === "object" ? model.meta : null;
    const modelName = model.name || entry?.name || entry?.id || "模型";
    if (this.el.algoModelName) {
      const framework = normalizeFramework(model.framework);
      this.el.algoModelName.textContent = [modelName, framework ? frameworkMeta(framework).label : "", modeMeta(model.mode).label]
        .filter(Boolean)
        .join("  ·  ");
    }

    const family = meta && typeof meta.family === "string" ? meta.family : "";
    const familyLabel = meta && typeof meta.familyLabel === "string" ? meta.familyLabel : family;
    const tags = this._asStringList(meta?.algorithmTags ?? meta?.tags);
    const patterns = Array.isArray(meta?.patterns) ? meta.patterns.filter((pattern) => pattern && typeof pattern === "object") : [];
    const pseudocode = this._pseudocodeLines(meta?.pseudocode);
    const providedFamilyConfidence = this._algorithmConfidence(
      meta?.familyConfidence ?? meta?.family_confidence ?? meta?.confidence,
    );
    const patternConfidences = patterns
      .map((pattern) => this._algorithmConfidence(pattern.confidence))
      .filter((value) => value != null);
    const familyConfidence = providedFamilyConfidence == null && patternConfidences.length
      ? Math.max(...patternConfidences)
      : providedFamilyConfidence;
    const confidenceTitle = providedFamilyConfidence == null && patternConfidences.length
      ? "家族置信度未提供，显示最高模式置信度"
      : "结构证据置信度";
    const hasAlgorithm = Boolean(
      meta && (family || familyLabel || tags.length || patterns.length || pseudocode.length),
    );

    if (this.el.algoFamily) {
      if (!hasAlgorithm) {
        this.el.algoFamily.innerHTML = `
          <div class="algo-empty">
            <span class="algo-empty-icon">${icon("spark")}</span>
            <strong>暂无算法信息</strong>
            <span>当前 graph 没有 model.meta.family / patterns / pseudocode 字段。</span>
          </div>`;
      } else {
        const familyText = familyLabel || family || "未知家族";
        const percent = familyConfidence == null ? null : Math.round(familyConfidence * 100);
        const tagsHtml = tags.length
          ? `<div class="algo-tags">${tags.map((tag) => `<span class="algo-tag">${escapeHtml(tag)}</span>`).join("")}</div>`
          : "";
        this.el.algoFamily.innerHTML = `
          <div class="algo-family-top">
            <div class="algo-family-heading">
              <span class="algo-kicker">ALGORITHM FAMILY</span>
              <h2>${escapeHtml(familyText)}</h2>
              <span class="algo-family-code">${escapeHtml(family || "unknown")}</span>
            </div>
            <div class="algo-family-confidence" title="${confidenceTitle}">
              <span>置信度</span>
              <strong>${percent == null ? "—" : `${percent}%`}</strong>
            </div>
          </div>
          <div class="algo-confidence-bar"><i style="width:${percent == null ? 0 : percent}%"></i></div>
          <div class="algo-family-meta">
            ${tagsHtml}
            <span class="algo-family-count">${patterns.length} 个模式${pseudocode.length ? ` · ${pseudocode.length} 行伪代码` : ""}</span>
          </div>`;
      }
    }

    if (this.el.algoPatterns) {
      if (!patterns.length) {
        this.el.algoPatterns.innerHTML = `<div class="algo-empty algo-empty-small">${icon("layers")}<span>暂无模式</span></div>`;
      } else {
        this.el.algoPatterns.innerHTML = patterns.map((pattern) => {
          const kind = String(pattern.kind || "unknown");
          const label = String(pattern.label || kind);
          const formula = typeof pattern.formula === "string" ? pattern.formula : "";
          const confidence = this._algorithmConfidence(pattern.confidence);
          const percent = confidence == null ? null : Math.round(confidence * 100);
          const evidence = this._asStringList(pattern.evidence);
          const nodeCount = Array.isArray(pattern.nodeIds) ? pattern.nodeIds.length : 0;
          const groupCount = Array.isArray(pattern.groupIds) ? pattern.groupIds.length : 0;
          const color = groupMeta(kind).color || "#67e8f9";
          return `<article class="algo-pattern" style="--c:${color}">
            <div class="algo-pattern-head">
              <span class="algo-pattern-kind">${escapeHtml(kind)}</span>
              <span class="algo-pattern-conf">${percent == null ? "—" : `${percent}%`}</span>
            </div>
            <h4>${escapeHtml(label)}</h4>
            ${formula ? `<pre class="algo-formula">${escapeHtml(formula)}</pre>` : ""}
            ${percent == null ? "" : `<div class="algo-confidence-bar is-small"><i style="width:${percent}%"></i></div>`}
            ${evidence.length ? `<ul class="algo-evidence">${evidence.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>` : ""}
            ${nodeCount || groupCount ? `<div class="algo-pattern-foot">${nodeCount ? `${nodeCount} 节点` : ""}${nodeCount && groupCount ? " · " : ""}${groupCount ? `${groupCount} 分组` : ""}</div>` : ""}
          </article>`;
        }).join("");
      }
    }

    if (this.el.algoPatternCount) {
      this.el.algoPatternCount.textContent = patterns.length ? `${patterns.length} 项` : "—";
    }

    if (this.el.algoPseudocode) {
      this.el.algoPseudocode.textContent = pseudocode.length
        ? pseudocode.join("\n")
        : "暂无伪代码";
    }
    if (this.el.algoCopy) this.el.algoCopy.disabled = pseudocode.length === 0;
    if (this.el.algoFootMeta) {
      this.el.algoFootMeta.textContent = meta
        ? `${family ? `family=${family}` : "family=—"} · ${patterns.length} 个模式 · ${pseudocode.length} 行伪代码`
        : "暂无算法 meta";
    }
  }

  async copyPseudocode() {
    const text = this.el.algoPseudocode?.textContent?.trim() || "";
    if (!text || text === "暂无伪代码") {
      this.toast("info", "没有可复制的伪代码", "当前 graph 未提供 model.meta.pseudocode。");
      return;
    }
    try {
      await navigator.clipboard.writeText(text);
      this.toast("success", "已复制伪代码", "前向伪代码已写入剪贴板。");
    } catch (error) {
      const textarea = document.createElement("textarea");
      textarea.value = text;
      textarea.style.position = "fixed";
      textarea.style.opacity = "0";
      document.body.appendChild(textarea);
      textarea.select();
      try {
        document.execCommand("copy");
        this.toast("success", "已复制伪代码", "前向伪代码已写入剪贴板。");
      } catch (fallbackError) {
        this.toast("error", "复制失败", "浏览器阻止了剪贴板访问。");
      }
      textarea.remove();
    }
  }

  /* ------------------------------------------------------------------ */
  /* Layer table (v2)                                                   */
  /* ------------------------------------------------------------------ */

  isTableOpen() {
    return this.tableOpen;
  }

  toggleTable() {
    if (this.tableOpen) this.closeTable();
    else this.openTable();
  }

  openTable() {
    if (!this.current || !this.el.tableView) return;
    this.closeAlgo();
    this.tableOpen = true;
    this.el.tableView.hidden = false;
    this.el.app?.classList.add("is-table-open");
    if (this.el.tableToggle) {
      this.el.tableToggle.setAttribute("aria-pressed", "true");
      this.el.tableToggle.disabled = false;
    }
    this.hideTooltip();
    this.el.onboarding?.classList.add("is-hidden");
    this.tableQuery = "";
    this.tableSort = { key: null, dir: 1 };
    if (this.el.tableSearch) this.el.tableSearch.value = "";
    this.updateTableSortIndicators();
    this.renderTable({ resetScroll: true });
  }

  closeTable() {
    this.tableOpen = false;
    if (this.el.tableView) this.el.tableView.hidden = true;
    this.el.app?.classList.remove("is-table-open");
    this.el.tableToggle?.setAttribute("aria-pressed", "false");
    this.el.tableSearch?.blur();
  }

  focusTableSearch() {
    if (!this.tableOpen) this.openTable();
    this.el.tableSearch?.focus();
    this.el.tableSearch?.select();
  }

  tableRowData() {
    if (!this.current) return [];
    const { graph, layout } = this.current;
    const items = layout?.nodes?.length
      ? layout.nodes
      : (graph.nodes || []).map((node, index) => ({ node, id: node.id, kind: node.kind, order: index }));
    const rows = [];
    for (const item of items) {
      const node = item.node || item;
      if (!node || !node.id) continue;
      const kind = node.kind || "unknown";
      const group = node.groupId ? this.groupById.get(node.groupId) : null;
      const groupLabel = this.groupDisplayLabel(group) || "—";
      const inShapes = Array.isArray(node.inShapes) ? node.inShapes : [];
      const outShapes = Array.isArray(node.outShapes) ? node.outShapes : [];
      const inText = inShapes.length ? inShapes.map(formatShapeCompact).join(" | ") : "—";
      const outText = outShapes.length ? outShapes.map(formatShapeCompact).join(" | ") : "—";
      const activation = nodeMetricValue(node, "activationBytes");
      const receptiveField = nodeMetricValue(node, "receptiveField");
      const macs = nodeMetricValue(node, "macs");
      const search = [
        node.id, node.name, node.op, kind, kindLabel(kind), groupLabel,
        inText, outText, node.modulePath || "", (node.tags || []).join(" "),
      ]
        .join(" ")
        .toLowerCase();
      rows.push({
        id: item.id || node.id,
        name: node.name || node.id,
        op: node.op || "—",
        kind,
        kindText: kindLabel(kind),
        inText,
        outText,
        inShapes,
        outShapes,
        params: Number(node.numParams) || 0,
        flops: Number(node.flops) || 0,
        activation,
        activationText: activation != null ? formatActivationMB(activation) : "—",
        rf: receptiveField,
        rfText: receptiveField != null ? formatReceptiveField(receptiveField) : "—",
        macs,
        groupLabel,
        search,
        order: item.order ?? rows.length,
      });
    }
    return rows;
  }

  _tableShapesHtml(shapes) {
    if (!Array.isArray(shapes) || !shapes.length) {
      return `<span class="table-missing">—</span>`;
    }
    return `<span class="table-shapes">${shapes
      .map((shape) => `<span class="table-shape" title="${escapeHtml(formatShapeCompact(shape))}">${escapeHtml(formatShapeCompact(shape))}</span>`)
      .join("")}</span>`;
  }

  _sortTableRows(rows) {
    const { key, dir } = this.tableSort;
    if (!key) return rows;
    const numericKeys = new Set(["params", "flops", "activation", "rf"]);
    const valueOf = (row) => {
      switch (key) {
        case "name": return row.name;
        case "op": return row.op;
        case "kind": return row.kind;
        case "in": return row.inText;
        case "out": return row.outText;
        case "params": return row.params;
        case "flops": return row.flops;
        case "activation": return row.activation;
        case "rf": return row.rf;
        case "group": return row.groupLabel;
        default: return "";
      }
    };
    return rows
      .map((row, index) => ({ row, index }))
      .sort((a, b) => {
        const av = valueOf(a.row);
        const bv = valueOf(b.row);
        const aMissing = av == null;
        const bMissing = bv == null;
        if (aMissing && bMissing) return a.index - b.index;
        if (aMissing) return 1;
        if (bMissing) return -1;
        let result;
        if (numericKeys.has(key)) {
          result = Number(av) - Number(bv);
        } else {
          result = String(av).localeCompare(String(bv), "zh-CN", {
            numeric: true,
            sensitivity: "base",
          });
        }
        return (result * dir) || (a.index - b.index);
      })
      .map((entry) => entry.row);
  }

  updateTableSortIndicators() {
    if (!this.el.tableView) return;
    const { key, dir } = this.tableSort;
    for (const header of this.el.tableView.querySelectorAll("th[data-sort]")) {
      const active = header.dataset.sort === key;
      header.classList.toggle("is-sorted", active);
      header.setAttribute("aria-sort", active ? (dir > 0 ? "ascending" : "descending") : "none");
      const arrow = header.querySelector(".sort-arrow");
      if (arrow) arrow.textContent = active ? (dir > 0 ? "↑" : "↓") : "↕";
    }
  }

  renderTable({ resetScroll = false } = {}) {
    if (!this.el.tableView || !this.current) return;
    this.tableRows = this.tableRowData();
    const query = this.tableQuery.trim().toLowerCase();
    const filtered = query
      ? this.tableRows.filter((row) => row.search.includes(query))
      : [...this.tableRows];
    this.tableFiltered = this._sortTableRows(filtered);

    const framework = normalizeFramework(this.current.graph.model?.framework);
    if (this.el.tableFramework) {
      if (framework) {
        this.el.tableFramework.hidden = false;
        this.el.tableFramework.dataset.framework = framework;
        this.el.tableFramework.textContent = `${frameworkMeta(framework).label} · ${modeMeta(this.current.graph.model?.mode).label}`;
      } else {
        this.el.tableFramework.hidden = true;
      }
    }

    const html = this.tableFiltered.map((row) => {
      const kindStyle = `--c:${kindColor(row.kind)}`;
      return `<tr data-node-id="${escapeHtml(row.id)}">
        <td class="t-name"><span class="t-name-main" title="${escapeHtml(row.name)}">${escapeHtml(row.name)}</span><span class="t-name-id">${escapeHtml(row.id)}</span></td>
        <td class="t-op" title="${escapeHtml(row.op)}">${escapeHtml(row.op)}</td>
        <td class="t-kind"><span class="table-kind" style="${kindStyle}"><i></i>${escapeHtml(row.kindText)}</span></td>
        <td class="t-shape">${this._tableShapesHtml(row.inShapes)}</td>
        <td class="t-shape">${this._tableShapesHtml(row.outShapes)}</td>
        <td class="t-num" title="${escapeHtml(String(row.params))}">${escapeHtml(formatCount(row.params))}</td>
        <td class="t-num" title="${escapeHtml(String(row.flops))}">${escapeHtml(formatFlops(row.flops))}</td>
        <td class="t-num t-activation${row.activation == null ? " is-missing" : ""}" title="${row.activation == null ? "缺少 node.metrics.activationBytes" : "估算值：" + escapeHtml(String(row.activation)) + " bytes"}">${escapeHtml(row.activationText)}</td>
        <td class="t-num t-rf${row.rf == null ? " is-missing" : ""}" title="${row.rf == null ? "缺少 node.metrics.receptiveField" : "估算值"}">${escapeHtml(row.rfText)}</td>
        <td class="t-group" title="${escapeHtml(row.groupLabel)}">${escapeHtml(row.groupLabel)}</td>
      </tr>`;
    }).join("");

    if (this.el.tableBody) this.el.tableBody.innerHTML = html;
    if (this.el.tableEmpty) this.el.tableEmpty.hidden = this.tableFiltered.length > 0;
    if (this.el.tableCount) {
      this.el.tableCount.textContent = query
        ? `${this.tableFiltered.length} / ${this.tableRows.length} 层`
        : `${this.tableRows.length} 层`;
    }
    if (this.el.tableFootMeta) {
      this.el.tableFootMeta.textContent = `共 ${this.tableRows.length} 层 · 显示 ${this.tableFiltered.length} · 激活显存 / 感受野为估算`;
    }
    if (this.el.tableExport) this.el.tableExport.disabled = this.tableRows.length === 0;
    this.updateTableSortIndicators();
    if (resetScroll) {
      const scroll = this.el.tableView.querySelector(".table-scroll");
      if (scroll) scroll.scrollTop = 0;
    }
  }

  filterTable(query) {
    this.tableQuery = String(query || "");
    if (this.tableOpen) this.renderTable();
  }

  sortTable(key) {
    if (!key) return;
    if (this.tableSort.key === key) this.tableSort.dir *= -1;
    else this.tableSort = { key, dir: 1 };
    this.renderTable();
  }

  _csvEscape(value) {
    const text = String(value == null ? "" : value);
    return /[",\r\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
  }

  exportTableCsv() {
    if (!this.current) return;
    if (!this.tableFiltered.length && !this.tableRows.length) return;
    const header = [
      "name", "op", "kind", "in_shapes", "out_shapes", "params", "flops",
      "activation_mb", "receptive_field", "group",
    ];
    const lines = [header.join(",")];
    for (const row of this.tableFiltered) {
      lines.push([
        row.name,
        row.op,
        row.kind,
        row.inText,
        row.outText,
        row.params,
        row.flops,
        row.activation != null ? activationMBValue(row.activation) : "—",
        row.rf != null ? String(row.rf) : "—",
        row.groupLabel,
      ].map((value) => this._csvEscape(value)).join(","));
    }
    const csv = lines.join("\r\n");
    const modelName = this.current.graph.model?.name || this.current.entry?.name || "model";
    const safeName = String(modelName).replace(/[\\/:*?"<>|\s]+/g, "_").slice(0, 80) || "model";
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `${safeName}_layers.csv`;
    anchor.style.display = "none";
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    setTimeout(() => URL.revokeObjectURL(url), 3000);
  }

  /* ------------------------------------------------------------------ */
  /* Details                                                            */
  /* ------------------------------------------------------------------ */

  renderDetail(node) {
    if (!this.el.detail) return;
    if (!node) {
      this.el.detailMeta.textContent = "单击层查看";
      this.el.detail.innerHTML = `
        <div class="detail-guide">
          <div class="dg-icon">${icon("target")}</div>
          <h3>在 3D 场景中选择一层</h3>
          <p>悬停查看关键指标，单击选中层并在右侧查看完整参数、形状与源码位置。</p>
          <div class="dg-list">
            <div class="dg-item"><span>旋转视角</span><span><kbd>左键拖拽</kbd></span></div>
            <div class="dg-item"><span>缩放 / 平移</span><span><kbd>滚轮</kbd> <kbd>右键</kbd></span></div>
            <div class="dg-item"><span>聚焦 / 复位</span><span><kbd>双击层</kbd> <kbd>双击空白</kbd></span></div>
            <div class="dg-item"><span>搜索层</span><span><kbd>/</kbd></span></div>
          </div>
        </div>`;
      return;
    }
    const { graph, layout } = this.current || {};
    if (!graph || !layout) return;
    const item = layout.byId.get(node.id);
    const kind = node.kind || "unknown";
    const color = kindColor(kind);
    const group = node.groupId ? this.groupById.get(node.groupId) : null;
    const order = item ? item.order : 0;
    const index = Math.max(0, layout.ordered.findIndex((entry) => entry.id === node.id));
    const params = paramEntries(node.params);
    const shapesIn = Array.isArray(node.inShapes) ? node.inShapes : [];
    const shapesOut = Array.isArray(node.outShapes) ? node.outShapes : [];
    const tags = Array.isArray(node.tags) ? node.tags : [];
    const notes = Array.isArray(node.notes) ? node.notes : [];
    const nodeMeta = node.meta && typeof node.meta === "object" ? node.meta : null;
    const algorithmFormula = typeof nodeMeta?.formula === "string" ? nodeMeta.formula : "";
    const algorithmSemantic = typeof nodeMeta?.semantic === "string" ? nodeMeta.semantic : "";
    const algorithmAxes = typeof nodeMeta?.axes === "string" ? nodeMeta.axes : "";
    const algorithmTags = this._asStringList(nodeMeta?.tags);
    const hasNodeAlgorithm = Boolean(
      algorithmFormula || algorithmSemantic || algorithmAxes || algorithmTags.length,
    );
    const source = node.source;
    const depth = Number(node.depth) || 0;
    const rank = shapesOut[0] ? shapesOut[0].length : shapesIn[0] ? shapesIn[0].length : 0;
    const framework = normalizeFramework(graph.model?.framework);
    const frameworkInfo = frameworkMeta(framework);
    const activation = nodeMetricValue(node, "activationBytes");
    const receptiveField = nodeMetricValue(node, "receptiveField");
    const macs = nodeMetricValue(node, "macs");
    const hasNodeMetrics = Boolean(
      node.metrics && (activation != null || receptiveField != null || macs != null),
    );
    const detailMetricsHtml = hasNodeMetrics
      ? `<div class="dc-section" id="detail-metrics">
          <div class="dc-section-title">${icon("bolt")}显存 / 感受野 <span class="metric-approx">估算</span></div>
          <div class="dc-metrics">
            ${activation != null ? `<div class="dc-metric"><span>激活显存</span><b>${formatActivationMB(activation)}</b></div>` : ""}
            ${receptiveField != null ? `<div class="dc-metric"><span>感受野</span><b>${formatReceptiveField(receptiveField)}</b></div>` : ""}
            ${macs != null ? `<div class="dc-metric"><span>MACs</span><b title="macs = flops / 2">${formatMacs(macs)}</b></div>` : ""}
          </div>
        </div>`
      : "";

    const paramsHtml = params.length
      ? `<div class="param-table">${params.map((entry) => `
          <div class="param-row">
            <span class="param-key" title="${escapeHtml(entry.key)}">${escapeHtml(entry.label)}<em>${escapeHtml(entry.key)}</em></span>
            <span class="param-val">${escapeHtml(entry.text)}</span>
          </div>`).join("")}</div>`
      : `<div class="param-empty">该层没有参数（${escapeHtml(kindLabel(kind))}）</div>`;

    const sourceHtml = source
      ? `<div class="source-box">
          <div class="source-head">${icon("code")}<span class="source-path" title="${escapeHtml(source.file || "")}">${escapeHtml(source.relPath || source.file || "source")}</span>${source.line ? `<span class="source-line-badge">L${source.line}</span>` : ""}</div>
          ${source.code ? `<pre class="source-code">${escapeHtml(source.code)}</pre>` : ""}
        </div>`
      : `<div class="param-empty">未提供源码位置</div>`;

    const notesHtml = notes.length
      ? `<div class="dc-section"><div class="dc-section-title">${icon("info")}备注</div><div class="dc-notes">${notes.map((note) => `<div class="dc-note">${escapeHtml(note)}</div>`).join("")}</div></div>`
      : "";

    const tagsHtml = tags.length
      ? `<div class="dc-section"><div class="dc-section-title">${icon("tag")}标签</div><div class="tag-row">${tags.map((tag) => `<span class="tag-chip">${escapeHtml(tag)}</span>`).join("")}</div></div>`
      : "";

    const nodeAlgorithmHtml = hasNodeAlgorithm
      ? `<div class="dc-section" id="detail-algo">
          <div class="dc-section-title">${icon("spark")}算法解释 <span class="metric-approx">meta</span></div>
          ${algorithmFormula ? `<pre class="dc-formula">${escapeHtml(algorithmFormula)}</pre>` : ""}
          ${algorithmSemantic ? `<p class="dc-semantic">${escapeHtml(algorithmSemantic)}</p>` : ""}
          ${algorithmAxes ? `<div class="dc-meta-row"><span class="dc-meta-chip">轴 ${escapeHtml(algorithmAxes)}</span></div>` : ""}
          ${algorithmTags.length ? `<div class="tag-row">${algorithmTags.map((tag) => `<span class="tag-chip is-algo">${escapeHtml(tag)}</span>`).join("")}</div>` : ""}
        </div>`
      : "";

    let patternGroup = group;
    let patternGuard = 0;
    while (patternGroup && patternGuard < 20) {
      if (patternGroup.meta && patternGroup.meta.pattern) break;
      patternGroup = patternGroup.parentId ? this.groupById.get(patternGroup.parentId) : null;
      patternGuard += 1;
    }
    const groupPattern =
      patternGroup?.meta?.pattern && typeof patternGroup.meta.pattern === "object"
        ? patternGroup.meta.pattern
        : null;
    const groupPatternConfidence = this._algorithmConfidence(groupPattern?.confidence);
    const groupPatternPercent =
      groupPatternConfidence == null ? null : Math.round(groupPatternConfidence * 100);
    const groupPatternTags = this._asStringList(groupPattern?.tags);
    const groupPatternHtml = groupPattern
      ? `<div class="dc-section" id="detail-group-pattern">
          <div class="dc-section-title">${icon("layers")}所属模式 <span class="metric-approx">group.meta</span></div>
          <div class="group-pattern" style="--c:${groupMeta(groupPattern.kind || group.kind).color}">
            <div class="group-pattern-head">
              <span class="group-pattern-kind">${escapeHtml(String(groupPattern.kind || group.kind || "pattern"))}</span>
              <span class="group-pattern-conf">${groupPatternPercent == null ? "—" : `${groupPatternPercent}%`}</span>
            </div>
            <div class="group-pattern-label">${escapeHtml(groupPattern.label || groupPattern.kind || "模式")}</div>
            ${groupPattern.formula ? `<pre class="dc-formula">${escapeHtml(groupPattern.formula)}</pre>` : ""}
            ${groupPatternTags.length ? `<div class="tag-row">${groupPatternTags.map((tag) => `<span class="tag-chip is-algo">${escapeHtml(tag)}</span>`).join("")}</div>` : ""}
          </div>
        </div>`
      : "";

    this.el.detailMeta.textContent = `第 ${index + 1} / ${layout.nodeCount} 层`;
    this.el.detail.innerHTML = `
      <div class="detail-card" style="--c:${color}">
        <div class="dc-head">
          <span class="dc-kind"><span class="dc-kind-dot"></span>${escapeHtml(kindLabel(kind))}</span>
          <span class="dc-op" title="${escapeHtml(node.op)}">${escapeHtml(node.op)}</span>
          <span class="dc-nav">
            <button class="dc-nav-btn prev" type="button" data-nav="-1" ${index <= 0 ? "disabled" : ""} title="上一层 (←)">${icon("chevron")}</button>
            <button class="dc-nav-btn" type="button" data-nav="1" ${index >= layout.nodeCount - 1 ? "disabled" : ""} title="下一层 (→)">${icon("chevron")}</button>
          </span>
        </div>
        <div class="dc-name">${escapeHtml(opName(node) || node.id)}</div>
        <div class="dc-meta-row">
          ${group ? `<span class="dc-meta-chip" style="--c:${groupMeta(group.kind).color}">${escapeHtml(this.groupDisplayLabel(group))}</span>` : `<span class="dc-meta-chip">未分组</span>`}
          <span class="dc-meta-chip">depth ${depth}</span>
          ${framework ? `<span class="dc-meta-chip framework-mini" data-framework="${escapeHtml(framework)}">${escapeHtml(frameworkInfo.label)}</span>` : ""}
          ${node.name ? `<span class="dc-meta-chip" title="模块路径">${escapeHtml(node.name)}</span>` : ""}
          <span class="dc-meta-chip">${escapeHtml(node.id)}</span>
          ${node.tags?.includes("residual") ? `<span class="dc-meta-chip">残差合并</span>` : ""}
        </div>

        <div class="dc-section">
          <div class="dc-section-title">${icon("cube")}张量形状</div>
          <div class="shape-line"><span class="shape-tag">in</span><span class="shape-pills">${shapePills(shapesIn)}</span></div>
          <div class="shape-line"><span class="shape-tag">out</span><span class="shape-pills">${shapePills(shapesOut, true)}</span></div>
        </div>

        <div class="dc-section">
          <div class="dc-section-title">${icon("bolt")}计算量</div>
          <div class="dc-metrics">
            <div class="dc-metric"><span>参数量</span><b title="${escapeHtml(formatInteger(node.numParams))}">${formatCount(node.numParams)}</b></div>
            <div class="dc-metric"><span>FLOPs</span><b title="${escapeHtml(formatInteger(node.flops))}">${formatFlops(node.flops)}</b></div>
            <div class="dc-metric"><span>张量秩</span><b>${rank || "—"}</b></div>
            <div class="dc-metric"><span>算子种类</span><b>${escapeHtml(kind)}</b></div>
          </div>
        </div>

        ${detailMetricsHtml}
        ${nodeAlgorithmHtml}
        ${groupPatternHtml}

        <div class="dc-section">
          <div class="dc-section-title">${icon("grid")}超参数</div>
          ${paramsHtml}
        </div>

        <div class="dc-section">
          <div class="dc-section-title">${icon("code")}源码位置</div>
          ${sourceHtml}
        </div>

        ${tagsHtml}
        ${notesHtml}

        <div class="dc-section">
          <button class="state-btn is-primary" type="button" data-focus style="width:100%;justify-content:center">${icon("target")}聚焦此层</button>
        </div>
      </div>`;
    this.el.detail.scrollTop = 0;
  }

  /* ------------------------------------------------------------------ */
  /* Tooltip                                                            */
  /* ------------------------------------------------------------------ */

  showTooltip(payload, clientX, clientY) {
    const tooltip = this.el.tooltip;
    if (!tooltip || !payload) return;
    let html = "";
    let color = "#67e8f9";
    let hint = `<kbd>单击</kbd> 查看详情 <span>·</span> <kbd>双击</kbd> 聚焦`;

    if (payload.type === "node") {
      const node = payload.node;
      const item = payload.item;
      const kind = node.kind || "unknown";
      color = kindColor(kind);
      const group = node.groupId ? this.groupById.get(node.groupId) : null;
      const params = paramEntries(node.params).slice(0, 8);
      const source = node.source;
      const shapesIn = Array.isArray(node.inShapes) ? node.inShapes : [];
      const shapesOut = Array.isArray(node.outShapes) ? node.outShapes : [];
      const activation = nodeMetricValue(node, "activationBytes");
      const receptiveField = nodeMetricValue(node, "receptiveField");
      const macs = nodeMetricValue(node, "macs");
      const semantic =
        node.meta && typeof node.meta.semantic === "string" ? node.meta.semantic.trim() : "";
      html = `
        <div class="tt-head"><span class="tt-type" style="--c:${color}">层</span><span class="tt-kind">${escapeHtml(kindLabel(kind))}</span><span class="tt-name">${escapeHtml(node.id)}</span></div>
        <div class="tt-title">${escapeHtml(opName(node) || node.id)}</div>
        <div class="tt-sub">${escapeHtml(node.name || "")}${group ? ` · ${escapeHtml(this.groupDisplayLabel(group))}` : ""}</div>
        ${semantic ? `<div class="tt-semantic">${escapeHtml(truncate(semantic, 180))}</div>` : ""}
        <div class="tt-shapes">
          <div class="tt-shape-row"><span class="tt-shape-k">IN</span><span class="tt-shape-pills">${this._ttPills(shapesIn)}</span></div>
          <div class="tt-shape-row"><span class="tt-shape-k">OUT</span><span class="tt-shape-pills">${this._ttPills(shapesOut, true)}</span></div>
        </div>
        <div class="tt-grid">
          <div class="tt-kv"><span class="k">参数量</span><span class="v" title="${escapeHtml(formatInteger(node.numParams))}">${formatCount(node.numParams)}</span></div>
          <div class="tt-kv"><span class="k">FLOPs</span><span class="v" title="${escapeHtml(formatInteger(node.flops))}">${formatFlops(node.flops)}</span></div>
          <div class="tt-kv"><span class="k">深度</span><span class="v">${escapeHtml(String(node.depth ?? 0))}</span></div>
          <div class="tt-kv"><span class="k">所属组</span><span class="v">${escapeHtml(this.groupDisplayLabel(group) || "—")}</span></div>
          ${activation != null ? `<div class="tt-kv"><span class="k">激活显存</span><span class="v">${formatActivationMB(activation)}</span></div>` : ""}
          ${receptiveField != null ? `<div class="tt-kv"><span class="k">感受野</span><span class="v">${formatReceptiveField(receptiveField)}</span></div>` : ""}
          ${macs != null ? `<div class="tt-kv"><span class="k">MACs</span><span class="v">${formatMacs(macs)}</span></div>` : ""}
        </div>
        ${params.length ? `<div class="tt-params">${params.map((entry) => `<span class="tt-param">${escapeHtml(entry.label)}<b>${escapeHtml(entry.text)}</b></span>`).join("")}</div>` : ""}
        ${source ? `<div class="tt-source"><span>${escapeHtml(fileNameLine(source))}</span>${source.code ? `<span class="code">${escapeHtml(truncate(source.code, 160))}</span>` : ""}</div>` : ""}
        <div class="tt-hint">${hint}</div>`;
    } else if (payload.type === "kernel") {
      const node = payload.node;
      const item = payload.item;
      const params = node.params || {};
      const [kH, kW] = payload.kernelSize || [3, 3];
      color = kindColor(node.kind);
      const stride = params.stride ?? "—";
      const padding = params.padding ?? "—";
      const groups = params.groups ?? 1;
      const dilation = params.dilation ?? "—";
      html = `
        <div class="tt-head"><span class="tt-type" style="--c:${color}">卷积核</span><span class="tt-kind">filter bank</span><span class="tt-name">${escapeHtml(node.id)}</span></div>
        <div class="tt-title">${escapeHtml(opName(node) || node.id)}</div>
        <div class="tt-sub">${escapeHtml(node.name || "")} · ${kH}×${kW} 卷积核网格</div>
        <div class="tt-shapes">
          <div class="tt-shape-row"><span class="tt-shape-k">IN</span><span class="tt-shape-pills">${this._ttPills(node.inShapes || [])}</span></div>
          <div class="tt-shape-row"><span class="tt-shape-k">OUT</span><span class="tt-shape-pills">${this._ttPills(node.outShapes || [], true)}</span></div>
        </div>
        <div class="tt-grid">
          <div class="tt-kv"><span class="k">输入通道</span><span class="v">${escapeHtml(formatParamValue("inChannels", params.inChannels))}</span></div>
          <div class="tt-kv"><span class="k">输出通道</span><span class="v">${escapeHtml(formatParamValue("outChannels", params.outChannels))}</span></div>
          <div class="tt-kv"><span class="k">stride</span><span class="v">${escapeHtml(formatParamValue("stride", stride))}</span></div>
          <div class="tt-kv"><span class="k">padding</span><span class="v">${escapeHtml(formatParamValue("padding", padding))}</span></div>
          <div class="tt-kv"><span class="k">groups</span><span class="v">${escapeHtml(String(groups))}</span></div>
          <div class="tt-kv"><span class="k">dilation</span><span class="v">${escapeHtml(formatParamValue("dilation", dilation))}</span></div>
        </div>
        <div class="tt-hint">${icon("grid")} 网格深度反映 Cin / groups = ${Math.round((Number(params.inChannels) || 1) / Math.max(1, Number(groups) || 1))} · ${hint}</div>`;
      hint = "";
    } else if (payload.type === "edge") {
      const edge = payload.edge;
      const sourceItem = payload.source;
      const targetItem = payload.target;
      const meta = edgeMeta(edge.kind);
      color = meta.color;
      html = `
        <div class="tt-head"><span class="tt-type" style="--c:${color}">连线</span><span class="tt-kind">${escapeHtml(meta.label)}</span><span class="tt-name">${escapeHtml(edge.id)}</span></div>
        <div class="tt-title">${escapeHtml(sourceItem?.node?.name || edge.source)} <span style="color:#5d7592">→</span> ${escapeHtml(targetItem?.node?.name || edge.target)}</div>
        <div class="tt-sub">${escapeHtml(edge.kind)} edge${edge.label ? ` · ${escapeHtml(edge.label)}` : ""}</div>
        <div class="tt-shapes"><div class="tt-shape-row"><span class="tt-shape-k">SHAPE</span><span class="tt-shape-pills">${this._ttPills(edge.shapes || [])}</span></div></div>
        <div class="tt-hint">${icon("link")} 悬停高亮 · 单击层可查看该层上下游</div>`;
      hint = "";
    }

    tooltip.style.setProperty("--c", color);
    tooltip.innerHTML = html;
    this._tooltipType = payload.type;
    tooltip.hidden = false;
    tooltip.setAttribute("aria-hidden", "false");
    this.positionTooltip(clientX, clientY);
  }

  positionTooltip(clientX, clientY) {
    const tooltip = this.el.tooltip;
    if (!tooltip || tooltip.hidden) return;
    const rect = tooltip.getBoundingClientRect();
    const pad = 14;
    const above = this._tooltipType === "node" || this._tooltipType === "kernel";
    let left = clientX + 20;
    let top = above ? clientY - rect.height - 14 : clientY + 16;
    if (left + rect.width > window.innerWidth - pad) left = clientX - rect.width - 20;
    if (top + rect.height > window.innerHeight - pad) top = clientY + 16;
    if (top < pad) top = clientY + 16;
    left = clamp(left, pad, window.innerWidth - rect.width - pad);
    top = clamp(top, pad, window.innerHeight - rect.height - pad);
    tooltip.style.transform = `translate3d(${Math.round(left)}px, ${Math.round(top)}px, 0)`;
  }

  _ttPills(shapes, out = false) {
    if (!Array.isArray(shapes) || !shapes.length) return `<span class="tt-pill">—</span>`;
    return shapes.map((shape) => `<span class="tt-pill${out ? " out" : ""}">${escapeHtml(formatShapeCompact(shape))}</span>`).join("");
  }

  hideTooltip() {
    if (!this.el.tooltip) return;
    this.el.tooltip.hidden = true;
    this.el.tooltip.setAttribute("aria-hidden", "true");
  }

  /* ------------------------------------------------------------------ */
  /* Flow / toggles                                                     */
  /* ------------------------------------------------------------------ */

  setToggleButtons(toggles) {
    if (!this.el.bottombar) return;
    for (const button of this.el.bottombar.querySelectorAll("[data-toggle]")) {
      const name = button.dataset.toggle;
      const value = Boolean(toggles[name]);
      button.setAttribute("aria-pressed", String(value));
    }
  }

  setPlayState(state) {
    this.playState = { ...this.playState, ...state };
    const { playing, active, index, total, name } = this.playState;
    if (this.el.playBtn) {
      this.el.playBtn.classList.toggle("is-playing", Boolean(playing));
      this.el.playBtn.title = playing ? "暂停数据流 (空格)" : "播放数据流 (空格)";
    }
    if (this.el.playIcon) this.el.playIcon.setAttribute("href", playing ? "#i-pause" : "#i-play");
    if (this.el.playLabel) this.el.playLabel.textContent = playing ? "暂停" : active && index >= total - 1 ? "重新播放" : "播放数据流";
    if (this.el.flowHud) {
      this.el.flowHud.hidden = !active;
      this.el.flowHud.classList.toggle("is-paused", Boolean(active && !playing));
      this.el.flowHud.classList.toggle("is-done", Boolean(active && index >= total - 1 && !playing));
    }
    if (this.el.flowLabel) this.el.flowLabel.textContent = name || "数据流";
    if (this.el.flowCount) this.el.flowCount.textContent = `${total ? Math.min(index + 1, total) : 0} / ${total}`;
    if (this.el.flowFill) this.el.flowFill.style.width = total ? pct((index + 1) / total) : "0%";
  }

  /* ------------------------------------------------------------------ */
  /* Toasts / notice / state overlay                                    */
  /* ------------------------------------------------------------------ */

  toast(type, title, message = "", timeout = 5600) {
    if (!this.el.toastStack) return;
    const icons = { error: "warning", warn: "warning", success: "check", info: "info" };
    const toast = document.createElement("div");
    toast.className = "toast";
    toast.dataset.type = type || "info";
    toast.innerHTML = `
      <span class="toast-icon">${icon(icons[type] || "info")}</span>
      <div class="toast-body">
        <div class="toast-title">${escapeHtml(title)}</div>
        ${message ? `<div class="toast-message">${escapeHtml(message)}</div>` : ""}
      </div>
      <button class="toast-close" type="button" data-toast-close aria-label="关闭提示">${icon("close")}</button>`;
    this.el.toastStack.appendChild(toast);
    while (this.el.toastStack.children.length > 4) {
      this.el.toastStack.removeChild(this.el.toastStack.firstElementChild);
    }
    if (timeout > 0) {
      const timer = setTimeout(() => this.dismissToast(toast), timeout);
      this.toastTimers.set(toast, timer);
    }
  }

  dismissToast(toast) {
    if (!toast) return;
    const timer = this.toastTimers.get(toast);
    if (timer) clearTimeout(timer);
    this.toastTimers.delete(toast);
    toast.classList.add("is-leaving");
    setTimeout(() => toast.remove(), 260);
  }

  showNotice({ type = "info", title, message = "", items = [], timeout = 0 }) {
    if (!this.el.notice) return;
    clearTimeout(this.noticeTimer);
    this.el.notice.hidden = false;
    this.el.notice.dataset.type = type;
    const iconName = type === "error" || type === "warn" ? "warning" : "info";
    const list = Array.isArray(items) && items.length
      ? `<ul class="notice-list">${items.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>`
      : "";
    this.el.notice.innerHTML = `
      <div class="notice-head">
        <span class="notice-icon">${icon(iconName)}</span>
        <div class="notice-title">${escapeHtml(title)}</div>
        <button class="notice-close" type="button" data-notice-close aria-label="关闭">${icon("close")}</button>
      </div>
      ${message ? `<div class="notice-message">${escapeHtml(message)}</div>` : ""}
      ${list}`;
    if (timeout > 0) this.noticeTimer = setTimeout(() => this.hideNotice(), timeout);
  }

  hideNotice() {
    clearTimeout(this.noticeTimer);
    if (this.el.notice) this.el.notice.hidden = true;
  }

  setState(state) {
    const overlay = this.el.stateOverlay;
    if (!overlay) return;
    if (!state) {
      overlay.hidden = true;
      this.stateMode = null;
      return;
    }
    const { mode = "loading", title = "", message = "", actions = [], detail = "" } = state;
    this.stateMode = mode;
    overlay.hidden = false;

    if (mode === "loading") {
      overlay.innerHTML = `
        <div class="state-card state-loading">
          <div class="prism-loader" aria-hidden="true">
            <span class="pl-ring pl-ring-a"></span><span class="pl-ring pl-ring-b"></span><span class="pl-ring pl-ring-c"></span><span class="pl-core"></span>
          </div>
          <h2>${escapeHtml(title)}</h2>
          <p class="state-message">${escapeHtml(message)}</p>
          ${detail ? `<div class="state-detail">${escapeHtml(detail)}</div>` : ""}
        </div>`;
      this.showSkeleton();
      return;
    }

    const iconName = STATE_ICON[mode] || "info";
    const iconClass = mode === "error" ? "is-error" : "is-empty";
    const actionHtml = actions.map((action) =>
      `<button class="state-btn${action.kind === "primary" ? " is-primary" : ""}" type="button" data-state-action="${escapeHtml(action.action)}">${action.icon ? icon(action.icon) : ""}${escapeHtml(action.label)}</button>`,
    ).join("");
    overlay.innerHTML = `
      <div class="state-card state-${escapeHtml(mode)}">
        <div class="state-icon ${iconClass}">${icon(iconName)}</div>
        <h2>${escapeHtml(title)}</h2>
        <p class="state-message">${escapeHtml(message)}</p>
        ${detail ? `<div class="state-detail">${escapeHtml(detail)}</div>` : ""}
        ${actionHtml ? `<div class="state-actions">${actionHtml}</div>` : ""}
      </div>`;
  }

  /* ------------------------------------------------------------------ */
  /* Panels / misc                                                      */
  /* ------------------------------------------------------------------ */

  setPanelCollapsed(side, collapsed) {
    const cls = side === "left" ? "is-left-collapsed" : "is-right-collapsed";
    this.el.app?.classList.toggle(cls, Boolean(collapsed));
    this._updatePanelToggleLabels();
  }

  isPanelCollapsed(side) {
    return this.el.app?.classList.contains(side === "left" ? "is-left-collapsed" : "is-right-collapsed");
  }

  togglePanel(side) {
    this.setPanelCollapsed(side, !this.isPanelCollapsed(side));
  }

  _updatePanelToggleLabels() {
    const leftCollapsed = this.el.app?.classList.contains("is-left-collapsed");
    const rightCollapsed = this.el.app?.classList.contains("is-right-collapsed");
    if (this.el.leftToggle) {
      this.el.leftToggle.title = leftCollapsed ? "展开模型面板" : "收起模型面板";
      this.el.leftToggle.setAttribute("aria-label", this.el.leftToggle.title);
    }
    if (this.el.rightToggle) {
      this.el.rightToggle.title = rightCollapsed ? "展开详情面板" : "收起详情面板";
      this.el.rightToggle.setAttribute("aria-label", this.el.rightToggle.title);
    }
  }

  setDropOverlay(visible) {
    if (this.el.dropOverlay) this.el.dropOverlay.hidden = !visible;
  }

  openHelp() {
    this.helpOpen = true;
    if (this.el.helpModal) this.el.helpModal.hidden = false;
  }

  closeHelp() {
    this.helpOpen = false;
    if (this.el.helpModal) this.el.helpModal.hidden = true;
  }

  isHelpOpen() {
    return this.helpOpen;
  }

  setSelectedNodeId(id) {
    this.setTreeSelected(id);
  }

  currentModelName() {
    return this.current?.graph?.model?.name || this.current?.entry?.name || "";
  }

  isSearchFocused() {
    return document.activeElement === this.el.search;
  }
}
