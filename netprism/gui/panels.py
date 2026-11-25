"""Right-hand detail, layer table, algorithm and summary panels."""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Any

from PySide6.QtCore import QItemSelectionModel, QModelIndex, QSortFilterProxyModel, QSignalBlocker, Qt, Signal
from PySide6.QtGui import QColor, QGuiApplication, QStandardItem, QStandardItemModel, QTextCursor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QTableView,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from .theme import (
    esc,
    family_from_graph,
    format_confidence,
    format_count,
    format_int,
    format_mb,
    format_shape,
    group_color,
    group_label,
    kind_color,
    kind_label,
    mode_label,
    framework_label,
    semantic_of,
    formula_of,
    shapes_text,
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _html_page(body: str) -> str:
    return f"""
    <html><head><style>
      body {{ color: #9fb6cc; font-family: 'Microsoft YaHei UI', 'Segoe UI'; font-size: 11.5px; line-height: 1.45; }}
      h2 {{ color: #eaf8ff; font-size: 19px; margin: 4px 0 8px 0; }}
      h3 {{ color: #cbefff; font-size: 12.5px; margin: 12px 0 5px 0; font-weight: 600; }}
      h4 {{ color: #dceeff; font-size: 11.5px; margin: 8px 0 4px 0; }}
      .kicker {{ color: #58718c; font-family: Consolas; font-size: 8px; letter-spacing: 2px; }}
      .muted {{ color: #7289a2; }}
      .dim {{ color: #5e758e; }}
      .accent {{ color: #67e8f9; }}
      .chip {{ display: inline-block; color: #9ed7ed; border: 1px solid #1c4b5e;
               background: #0b202b; border-radius: 5px; padding: 1px 5px; margin: 2px 3px 2px 0;
               font-family: Consolas; font-size: 9px; }}
      .violet {{ color: #d8c6ff; border-color: #4a3a76; background: #1a1532; }}
      .card {{ border: 1px solid #1a2d43; border-radius: 7px; padding: 8px; margin: 6px 0;
               background: #0a1422; }}
      .metric {{ border: 1px solid #173046; border-radius: 6px; padding: 6px 7px; background: #0a1826; }}
      .metric-label {{ color: #6f8aa3; font-size: 8px; }}
      .metric-value {{ color: #dff7ff; font-family: Consolas; font-size: 13px; }}
      .formula {{ color: #a8dcf1; background: #050d17; border: 1px solid #17364a;
                  padding: 6px; font-family: Consolas; font-size: 10px; }}
      .params {{ width: 100%; border-collapse: collapse; }}
      .params td {{ border-bottom: 1px solid #152538; padding: 3px 4px; font-size: 10px; }}
      .params td.key {{ color: #6f89a3; font-family: Consolas; }}
      .params td.val {{ color: #c7eaff; text-align: right; }}
      .source {{ color: #86a0b6; background: #050d17; border: 1px solid #14283a; padding: 6px;
                 font-family: Consolas; font-size: 9px; white-space: pre-wrap; }}
      .warn {{ color: #fbd38d; background: #241b0b; border: 1px solid #4a3a1b;
               padding: 5px; margin: 4px 0; font-size: 10px; }}
      .empty {{ color: #617a92; padding: 18px 8px; text-align: center; }}
      .tag {{ color: #b6a4e8; border: 1px solid #3b3260; background: #15132a;
              border-radius: 4px; padding: 1px 5px; margin: 2px 3px 2px 0; font-size: 9px; }}
      .confidence {{ color: #e7d6ff; font-family: Consolas; }}
      .bar {{ height: 4px; background: #172238; margin: 3px 0 7px 0; }}
      .bar i {{ display: block; height: 4px; background: #8b5cf6; }}
      a {{ color: #67e8f9; }}
    </style></head><body>{body}</body></html>
    """


def _safe_size(value: Any) -> str:
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, (int, float)):
        return format_int(value)
    if isinstance(value, (list, tuple)):
        return " × ".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _group_dict(graph: dict[str, Any], group_id: str | None) -> dict[str, Any] | None:
    for group in graph.get("groups") or []:
        if isinstance(group, dict) and str(group.get("id")) == str(group_id):
            return group
    return None


def _nearest_pattern_group(graph: dict[str, Any], group_id: str | None) -> dict[str, Any] | None:
    group = _group_dict(graph, group_id)
    guard = 0
    while group and guard < 20:
        if isinstance(group.get("meta"), dict) and isinstance(group["meta"].get("pattern"), dict):
            return group
        group = _group_dict(graph, group.get("parentId"))
        guard += 1
    return None


# --------------------------------------------------------------------------- #
# details
# --------------------------------------------------------------------------- #
class DetailsPanel(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("detailsPanel")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(
            "#detailsPanel { background: #0a111d; border: 1px solid #142238; border-radius: 10px; }"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 8)
        layout.setSpacing(7)
        title_row = QHBoxLayout()
        title = QLabel("层详情")
        title.setStyleSheet("color:#eaf8ff; font-size: 13px; font-weight: 600;")
        title_row.addWidget(title)
        title_row.addStretch(1)
        self.position_label = QLabel("—")
        self.position_label.setStyleSheet("color:#57718b; font-family:Consolas; font-size:9px;")
        title_row.addWidget(self.position_label)
        layout.addLayout(title_row)
        self.browser = QTextBrowser()
        self.browser.setObjectName("detailBrowser")
        self.browser.setOpenExternalLinks(False)
        self.browser.setHtml(_html_page('<div class="empty">在 3D 场景中选择一层<br><span class="dim">悬停可查看工具提示</span></div>'))
        layout.addWidget(self.browser, 1)
        self.graph: dict[str, Any] = {}
        self.node: dict[str, Any] | None = None

    def set_graph(self, graph: dict[str, Any]) -> None:
        self.graph = graph if isinstance(graph, dict) else {}
        self.node = None
        self.position_label.setText("—")
        self.browser.setHtml(
            _html_page('<div class="empty">在 3D 场景中选择一层<br><span class="dim">悬停可查看工具提示</span></div>')
        )

    def show_node(self, node: dict[str, Any]) -> None:
        if not isinstance(node, dict):
            return
        self.node = node
        nodes = [n for n in (self.graph.get("nodes") or []) if isinstance(n, dict)]
        index = next((i for i, item in enumerate(nodes) if str(item.get("id")) == str(node.get("id"))), -1)
        if index >= 0:
            self.position_label.setText(f"{index + 1} / {len(nodes)}")
        self.browser.setHtml(_html_page(self._node_html(node)))

    def _node_html(self, node: dict[str, Any]) -> str:
        model = self.graph.get("model") if isinstance(self.graph.get("model"), dict) else {}
        kind = str(node.get("kind") or "unknown")
        color = kind_color(kind)
        group = _group_dict(self.graph, node.get("groupId"))
        group_text = group_label(self.graph, node.get("groupId"))
        meta = node.get("meta") if isinstance(node.get("meta"), dict) else {}
        formula = formula_of(node)
        semantic = semantic_of(node)
        axes = meta.get("axes") if isinstance(meta.get("axes"), str) else ""
        meta_tags = meta.get("tags") if isinstance(meta.get("tags"), list) else []
        node_tags = node.get("tags") if isinstance(node.get("tags"), list) else []
        params = node.get("params") if isinstance(node.get("params"), dict) else {}
        shapes_in = node.get("inShapes") or []
        shapes_out = node.get("outShapes") or []
        metrics = node.get("metrics") if isinstance(node.get("metrics"), dict) else {}
        activation = metrics.get("activationBytes")
        rf = metrics.get("receptiveField")
        macs = metrics.get("macs")
        pieces: list[str] = []
        pieces.append(
            f"""<div><span class="chip" style="color:{color};border-color:{color}55;background:{color}12">{esc(kind_label(kind))}</span>
            <span class="chip">{esc(node.get('op') or '—')}</span></div>
            <h2>{esc(node.get('name') or node.get('id'))}</h2>
            <div class="muted"><span class="chip">{esc(group_text if group_text != '—' else '未分组')}</span>
            &nbsp;<span class="chip">{esc(str(model.get('framework') or 'unknown'))}</span>
            &nbsp;<span class="chip">{esc(str(node.get('id')))}</span></div>"""
        )
        pieces.append(
            f"""<h3>张量与计算</h3>
            <div class="card"><span class="muted">IN</span> &nbsp; <span class="accent">{esc(shapes_text(shapes_in))}</span><br>
            <span class="muted">OUT</span> &nbsp; <span class="accent">{esc(shapes_text(shapes_out))}</span><br>
            <span class="muted">params</span> <b>{esc(format_int(node.get('numParams')))}</b> &nbsp;
            <span class="muted">FLOPs</span> <b>{esc(format_count(node.get('flops')))}</b></div>"""
        )
        metric_cells = []
        if isinstance(activation, (int, float)):
            metric_cells.append(f'<span class="metric"><span class="metric-label">激活显存 · 估算</span><br><span class="metric-value">{esc(format_mb(activation))}</span></span>')
        if isinstance(rf, (int, float)):
            metric_cells.append(f'<span class="metric"><span class="metric-label">感受野 · 估算</span><br><span class="metric-value">{esc(str(int(rf)))}</span></span>')
        if isinstance(macs, (int, float)):
            metric_cells.append(f'<span class="metric"><span class="metric-label">MACs · 估算</span><br><span class="metric-value">{esc(format_count(macs))}</span></span>')
        if metric_cells:
            pieces.append('<div>' + '&nbsp;'.join(metric_cells) + '</div>')

        if formula or semantic or axes or meta_tags:
            pieces.append('<h3>算法解释</h3>')
            if formula:
                pieces.append(f'<div class="formula">{esc(formula)}</div>')
            if semantic:
                pieces.append(f'<div>{esc(semantic)}</div>')
            chips = []
            if axes:
                chips.append(f'<span class="chip">axes {esc(axes)}</span>')
            for tag in meta_tags:
                chips.append(f'<span class="tag">{esc(tag)}</span>')
            if chips:
                pieces.append("<div>" + "&nbsp;".join(chips) + "</div>")

        pattern_group = _nearest_pattern_group(self.graph, node.get("groupId"))
        if pattern_group:
            pattern = pattern_group.get("meta", {}).get("pattern")
            if isinstance(pattern, dict):
                confidence = format_confidence(pattern.get("confidence"))
                pieces.append("<h3>所属模式</h3>")
                pieces.append(
                    f'<div class="card"><span class="chip">{esc(pattern.get("kind") or "pattern")}</span>'
                    f'<span class="confidence"> {confidence if confidence is not None else "—"}%</span><br>'
                    f'<b>{esc(pattern.get("label") or pattern.get("kind") or "模式")}</b>'
                    + (f'<div class="formula">{esc(pattern.get("formula"))}</div>' if pattern.get("formula") else "")
                    + "</div>"
                )

        if params:
            rows = "".join(
                f'<tr><td class="key">{esc(key)}</td><td class="val">{esc(_safe_size(value))}</td></tr>'
                for key, value in list(params.items())[:30]
            )
            pieces.append(f'<h3>超参数</h3><table class="params">{rows}</table>')

        source = node.get("source") if isinstance(node.get("source"), dict) else None
        if source:
            source_text = source.get("relPath") or source.get("file") or ""
            line = source.get("line")
            code = source.get("code") or ""
            pieces.append(
                f'<h3>源码位置</h3><div class="source">{esc(source_text)}'
                + (f" · L{esc(line)}" if line else "")
                + (f"\n{esc(code)}" if code else "")
                + "</div>"
            )
        if node_tags:
            pieces.append("<h3>标签</h3><div>" + "&nbsp;".join(f'<span class="tag">{esc(tag)}</span>' for tag in node_tags) + "</div>")
        notes = node.get("notes") if isinstance(node.get("notes"), list) else []
        if notes:
            pieces.append("<h3>备注</h3>" + "".join(f'<div class="card">{esc(note)}</div>' for note in notes))
        return "".join(pieces)


# --------------------------------------------------------------------------- #
# table
# --------------------------------------------------------------------------- #
class GraphSortProxy(QSortFilterProxyModel):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._query = ""
        self._rows: list[dict[str, Any]] = []
        self.setDynamicSortFilter(True)

    def set_rows(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    def set_filter_text(self, text: str) -> None:
        self._query = str(text or "").strip().lower()
        self.invalidateFilter()

    def filterAcceptsRow(self, source_row: int, source_parent: QModelIndex) -> bool:  # noqa: N802
        if not self._query:
            return True
        if 0 <= source_row < len(self._rows):
            return self._query in str(self._rows[source_row].get("search", "")).lower()
        return True

    def lessThan(self, left: QModelIndex, right: QModelIndex) -> bool:  # noqa: N802
        left_value = left.data(Qt.ItemDataRole.UserRole)
        right_value = right.data(Qt.ItemDataRole.UserRole)
        if left_value is None and right_value is None:
            return left.data(Qt.ItemDataRole.DisplayRole) < right.data(Qt.ItemDataRole.DisplayRole)
        if left_value is None:
            return False
        if right_value is None:
            return True
        try:
            return left_value < right_value
        except TypeError:
            return str(left_value) < str(right_value)


class TablePanel(QWidget):
    nodeSelected = Signal(str)
    searchChanged = Signal(str)

    HEADERS = ["name", "op", "kind", "IN", "OUT", "params", "FLOPs", "activation", "RF", "group"]

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("tablePanel")
        self.graph: dict[str, Any] = {}
        self.rows: list[dict[str, Any]] = []
        self._select_guard = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(7)

        top = QHBoxLayout()
        self.search_edit = QLineEdit()
        self.search_edit.setObjectName("tableSearch")
        self.search_edit.setPlaceholderText("过滤层名 / op / kind / group…")
        self.search_edit.setClearButtonEnabled(True)
        self.count_label = QLabel("0 rows")
        self.count_label.setObjectName("countLabel")
        self.count_label.setStyleSheet("color:#6f849e; font-family:Consolas; font-size:9px;")
        top.addWidget(self.search_edit, 1)
        top.addWidget(self.count_label)
        layout.addLayout(top)

        self.model = QStandardItemModel(self)
        self.model.setHorizontalHeaderLabels(self.HEADERS)
        self.proxy = GraphSortProxy(self)
        self.proxy.setSourceModel(self.model)
        self.table_view = QTableView()
        self.table_view.setObjectName("layerTable")
        self.table = self.table_view
        self.table_view.setModel(self.proxy)
        self.table_view.setSortingEnabled(True)
        self.table_view.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table_view.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table_view.setAlternatingRowColors(True)
        self.table_view.setShowGrid(False)
        self.table_view.verticalHeader().setVisible(False)
        self.table_view.verticalHeader().setDefaultSectionSize(24)
        self.table_view.horizontalHeader().setStretchLastSection(True)
        self.table_view.setWordWrap(False)
        self.table_view.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.table_view.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.table_view.sortByColumn(0, Qt.SortOrder.AscendingOrder)
        widths = [100, 82, 54, 78, 78, 54, 50, 66, 28, 80]
        for index, width in enumerate(widths):
            self.table_view.setColumnWidth(index, width)
        layout.addWidget(self.table_view, 1)

        self.search_edit.textChanged.connect(self._on_search)
        self.table_view.selectionModel().currentRowChanged.connect(self._on_current_row)

    # ------------------------------------------------------------------ #
    def set_graph(self, graph: dict[str, Any]) -> None:
        self.graph = graph if isinstance(graph, dict) else {}
        self._select_guard = True
        self.model.removeRows(0, self.model.rowCount())
        self.rows = []
        for node in self.graph.get("nodes") or []:
            if not isinstance(node, dict):
                continue
            node_id = str(node.get("id"))
            kind = str(node.get("kind") or "unknown")
            params = node.get("numParams")
            flops = node.get("flops")
            metrics = node.get("metrics") if isinstance(node.get("metrics"), dict) else {}
            activation = metrics.get("activationBytes") if isinstance(metrics.get("activationBytes"), (int, float)) else None
            rf = metrics.get("receptiveField") if isinstance(metrics.get("receptiveField"), (int, float)) else None
            group = group_label(self.graph, node.get("groupId"))
            if group == "—":
                group = ""
            group_short = group.split(" / ")[-1] if group else ""
            in_text = shapes_text(node.get("inShapes"))
            out_text = shapes_text(node.get("outShapes"))
            values = {
                "node_id": node_id,
                "name": str(node.get("name") or node_id),
                "op": str(node.get("op") or "—"),
                "kind": kind,
                "in_shapes": in_text,
                "out_shapes": out_text,
                "params": int(params) if isinstance(params, (int, float)) else 0,
                "flops": int(flops) if isinstance(flops, (int, float)) else 0,
                "activation_bytes": activation,
                "activation_mb": f"{activation / (1024 * 1024):.4g}" if isinstance(activation, (int, float)) else "—",
                "rf": int(rf) if isinstance(rf, (int, float)) else None,
                "group": group,
                "search": " ".join(
                    [
                        node_id,
                        str(node.get("name") or ""),
                        str(node.get("op") or ""),
                        kind,
                        kind_label(kind),
                        in_text,
                        out_text,
                        group,
                        " ".join(str(tag) for tag in (node.get("tags") or [])),
                    ]
                ),
            }
            display = [
                values["name"],
                values["op"],
                kind,
                values["in_shapes"],
                values["out_shapes"],
                format_int(values["params"]),
                format_count(values["flops"]),
                f"{format_mb(activation)}" if activation is not None else "—",
                str(values["rf"]) if values["rf"] is not None else "—",
                group_short or "—",
            ]
            sortable = [
                values["name"].lower(),
                values["op"].lower(),
                kind,
                in_text,
                out_text,
                values["params"],
                values["flops"],
                activation if activation is not None else -1,
                values["rf"] if values["rf"] is not None else -1,
                group_short.lower(),
            ]
            row_items: list[QStandardItem] = []
            for col, text in enumerate(display):
                item = QStandardItem(str(text))
                item.setEditable(False)
                item.setToolTip(str(text))
                item.setData(sortable[col], Qt.ItemDataRole.UserRole)
                if col == 0:
                    item.setData(node_id, Qt.ItemDataRole.UserRole + 1)
                if col in (5, 6, 7, 8):
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                row_items.append(item)
            self.model.appendRow(row_items)
            self.rows.append(values)
        self.proxy.set_rows(self.rows)
        self.proxy.sort(0, Qt.SortOrder.AscendingOrder)
        self._select_guard = False
        self._update_count()

    def _on_search(self, text: str) -> None:
        self.proxy.set_filter_text(text)
        self._update_count()
        self.searchChanged.emit(text)

    def set_search(self, text: str) -> None:
        with QSignalBlocker(self.search_edit):
            self.search_edit.setText(text)
        self.proxy.set_filter_text(text)
        self._update_count()

    def _update_count(self) -> None:
        visible = self.proxy.rowCount()
        total = len(self.rows)
        self.count_label.setText(f"{visible} / {total} rows")

    def _on_current_row(self, current: QModelIndex, previous: QModelIndex) -> None:  # noqa: ARG002
        if self._select_guard or not current.isValid():
            return
        source = self.proxy.mapToSource(current)
        item = self.model.item(source.row(), 0)
        if item is None:
            return
        node_id = item.data(Qt.ItemDataRole.UserRole + 1)
        if node_id:
            self.nodeSelected.emit(str(node_id))

    def select_node(self, node_id: str) -> None:
        for row, values in enumerate(self.rows):
            if values.get("node_id") != node_id:
                continue
            source_index = self.model.index(row, 0)
            proxy_index = self.proxy.mapFromSource(source_index)
            if proxy_index.isValid():
                self._select_guard = True
                self.table_view.selectionModel().setCurrentIndex(
                    proxy_index,
                    QItemSelectionModel.SelectionFlag.ClearAndSelect | QItemSelectionModel.SelectionFlag.Rows,
                )
                self.table_view.scrollTo(proxy_index, QAbstractItemView.ScrollHint.EnsureVisible)
                self._select_guard = False
            break

    @property
    def visible_row_count(self) -> int:
        return self.proxy.rowCount()

    @property
    def row_count(self) -> int:
        return len(self.rows)

    def export_csv(self, path: str | Path) -> Path:
        target = Path(path)
        with target.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["name", "op", "kind", "in_shapes", "out_shapes", "params", "flops",
                 "activation_mb", "receptive_field", "group"]
            )
            for row in range(self.proxy.rowCount()):
                source = self.proxy.mapToSource(self.proxy.index(row, 0))
                values = self.rows[source.row()]
                writer.writerow(
                    [
                        values["name"],
                        values["op"],
                        values["kind"],
                        values["in_shapes"],
                        values["out_shapes"],
                        values["params"],
                        values["flops"],
                        values["activation_mb"],
                        values["rf"] if values["rf"] is not None else "—",
                        values["group"] or "—",
                    ]
                )
        return target


# --------------------------------------------------------------------------- #
# algorithm
# --------------------------------------------------------------------------- #
class AlgorithmPanel(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("algorithmTab")
        self.graph: dict[str, Any] = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(7)
        self.browser = QTextBrowser()
        self.browser.setObjectName("algoBrowser")
        self.browser.setHtml(_html_page('<div class="empty">暂无算法信息</div>'))
        layout.addWidget(self.browser, 2)

        code_head = QHBoxLayout()
        code_title = QLabel("FORWARD PSEUDOCODE  ·  前向伪代码")
        code_title.setStyleSheet("color:#6c8aa5; font-family:Consolas; font-size:9px; letter-spacing:1px;")
        code_head.addWidget(code_title)
        code_head.addStretch(1)
        self.copy_button = QPushButton("复制伪代码")
        self.copy_button.setObjectName("copyPseudocodeButton")
        self.copy_button.setEnabled(False)
        self.copy_button.clicked.connect(self.copy_pseudocode)
        code_head.addWidget(self.copy_button)
        layout.addLayout(code_head)
        self.pseudocode = QPlainTextEdit()
        self.pseudocode.setObjectName("algoPseudocode")
        self.pseudocode.setReadOnly(True)
        self.pseudocode.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.pseudocode.setPlaceholderText("暂无伪代码")
        layout.addWidget(self.pseudocode, 2)
    def set_graph(self, graph: dict[str, Any]) -> None:
        self.graph = graph if isinstance(graph, dict) else {}
        meta, family, label, confidence, tags = family_from_graph(self.graph)
        patterns = [
            pattern for pattern in (meta.get("patterns") or [])
            if isinstance(pattern, dict)
        ] if isinstance(meta, dict) else []
        code = self._pseudocode()
        if not meta or not (family or label or tags or patterns or code):
            self.browser.setHtml(_html_page('<div class="empty">暂无算法信息<br><span class="dim">该 graph 没有 model.meta.family / patterns / pseudocode 字段</span></div>'))
            self.pseudocode.setPlainText("")
            self.copy_button.setEnabled(False)
            return

        chips = " ".join(f'<span class="chip violet">{esc(tag)}</span>' for tag in tags)
        body = [
            '<div class="kicker">ALGORITHM FAMILY</div>',
            f'<h2>{esc(label or family or "未知家族")}</h2>',
            f'<div><span class="chip">family {esc(family or "unknown")}</span> '
            f'<span class="confidence"> 置信度 {confidence if confidence is not None else "—"}%</span></div>',
            f'<div style="margin-top:4px">{chips}</div>',
            '<h3>识别模式 · PATTERNS</h3>',
        ]
        if not patterns:
            body.append('<div class="empty" style="padding:10px">暂无模式</div>')
        for pattern in patterns:
            pattern_confidence = format_confidence(pattern.get("confidence"))
            formula = pattern.get("formula") or ""
            evidence = pattern.get("evidence")
            if isinstance(evidence, str):
                evidence = [evidence]
            if not isinstance(evidence, list):
                evidence = []
            evidence_html = "".join(f"<div>• {esc(item)}</div>" for item in evidence[:6])
            body.append(
                '<div class="card">'
                f'<span class="chip">{esc(pattern.get("kind") or "pattern")}</span>'
                f'<span class="confidence"> {pattern_confidence if pattern_confidence is not None else "—"}%</span>'
                f'<h4>{esc(pattern.get("label") or pattern.get("kind") or "Pattern")}</h4>'
                + (f'<div class="formula">{esc(formula)}</div>' if formula else "")
                + (f'<div class="muted" style="margin-top:4px">{evidence_html}</div>' if evidence_html else "")
                + "</div>"
            )
        self.browser.setHtml(_html_page("".join(body)))
        self.pseudocode.setPlainText("\n".join(code))
        self.copy_button.setEnabled(bool(code))

    def _pseudocode(self) -> list[str]:
        meta, _family, _label, _confidence, _tags = family_from_graph(self.graph)
        if not isinstance(meta, dict):
            return []
        code = meta.get("pseudocode")
        if isinstance(code, str):
            return [line for line in code.splitlines() if line.strip()]
        if isinstance(code, list):
            return [str(line) for line in code if str(line).strip()]
        return []

    def copy_pseudocode(self) -> None:
        text = self.pseudocode.toPlainText().strip()
        if not text:
            return
        QGuiApplication.clipboard().setText(text)


# --------------------------------------------------------------------------- #
# summary
# --------------------------------------------------------------------------- #
class SummaryPanel(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("summaryTab")
        self.graph: dict[str, Any] = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        self.browser = QTextBrowser()
        self.browser.setObjectName("summaryBrowser")
        self.browser.setHtml(_html_page('<div class="empty">暂无模型摘要</div>'))
        layout.addWidget(self.browser)

    def set_graph(self, graph: dict[str, Any]) -> None:
        self.graph = graph if isinstance(graph, dict) else {}
        model = self.graph.get("model") if isinstance(self.graph.get("model"), dict) else {}
        metrics = model.get("metrics") if isinstance(model.get("metrics"), dict) else {}
        meta, family, label, confidence, tags = family_from_graph(self.graph)
        warnings = model.get("warnings") if isinstance(model.get("warnings"), list) else []
        inputs = model.get("inputSpec") if isinstance(model.get("inputSpec"), list) else []
        outputs = model.get("outputSpec") if isinstance(model.get("outputSpec"), list) else []
        body = [
            f'<div class="kicker">MODEL SUMMARY</div><h2>{esc(model.get("name") or "未命名模型")}</h2>',
            f'<div><span class="chip">{esc(framework_label(model.get("framework")))}</span> '
            f'<span class="chip">{esc(mode_label(model.get("mode")))}</span> '
            f'<span class="chip violet">{esc(label or "未知家族")}</span></div>',
            '<h3>规模</h3><div class="card">'
            f'<span class="metric-label">参数量</span> <span class="metric-value">{esc(format_int(model.get("totalParams")))}</span> &nbsp;&nbsp;'
            f'<span class="metric-label">FLOPs</span> <span class="metric-value">{esc(format_count(model.get("totalFlops")))}</span> &nbsp;&nbsp;'
            f'<span class="metric-label">层数</span> <span class="metric-value">{esc(str(model.get("numLayers", "—")))}</span>'
            '</div>',
        ]
        cells = []
        if isinstance(metrics.get("activationBytes"), (int, float)):
            cells.append(("激活显存", format_mb(metrics["activationBytes"])))
        if isinstance(metrics.get("maxReceptiveField"), (int, float)):
            cells.append(("感受野", str(int(metrics["maxReceptiveField"]))))
        if isinstance(metrics.get("macs"), (int, float)):
            cells.append(("MACs", format_count(metrics["macs"])))
        if cells:
            body.append('<h3>估算指标</h3><div class="card">' + " &nbsp;&nbsp; ".join(
                f'<span class="metric-label">{esc(key)}</span> <span class="metric-value">{esc(value)}</span>'
                for key, value in cells
            ) + "</div>")
        body.append("<h3>输入 / 输出</h3><div class=\"card\">")
        for spec in inputs[:3]:
            if isinstance(spec, dict):
                body.append(f'<div><span class="muted">IN</span> {esc(spec.get("name") or "x")} <span class="accent">{esc(format_shape(spec.get("shape")))}</span></div>')
        for spec in outputs[:3]:
            if isinstance(spec, dict):
                body.append(f'<div><span class="muted">OUT</span> {esc(spec.get("name") or "y")} <span class="accent">{esc(format_shape(spec.get("shape")))}</span></div>')
        body.append("</div>")
        if family:
            body.append(
                f'<h3>算法家族</h3><div class="card"><b>{esc(label)}</b> '
                f'<span class="confidence">family={esc(family)}</span>'
                + (f' · 置信度 {confidence}%' if confidence is not None else "")
                + (f'<div class="muted" style="margin-top:4px">{" ".join(esc(tag) for tag in tags[:12])}</div>' if tags else "")
                + "</div>"
            )
        if warnings:
            body.append("<h3>Warnings</h3>" + "".join(f'<div class="warn">{esc(str(w))}</div>' for w in warnings[:8]))
        self.browser.setHtml(_html_page("".join(body)))


__all__ = [
    "AlgorithmPanel",
    "DetailsPanel",
    "GraphSortProxy",
    "SummaryPanel",
    "TablePanel",
]
