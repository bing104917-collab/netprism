"""Native main window for NetPrism.

Open a folder / model file -> parse in a QThread -> inspect the model in a
software-projected 3D view, a sortable table, an algorithm panel and a native
detail panel.  Exports are written directly with the existing report library.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from PySide6.QtCore import QSignalBlocker, QSize, Qt, QSettings
from PySide6.QtGui import QFont, QFontDatabase, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..report import render_report
from .graph_loader import ParseError, ParseWorker, demo_dir, load_demo_payload, parse_path_sync
from .panels import AlgorithmPanel, DetailsPanel, SummaryPanel, TablePanel
from .scene3d import Graph3DView
from .theme import (
    apply_style,
    family_from_graph,
    format_count,
    format_int,
    framework_label,
    group_label,
    mode_label,
)


def create_application(argv: list[str] | None = None) -> QApplication:
    """Return the single QApplication, themed for NetPrism."""
    if not getattr(create_application, "_font_dir_ready", False):
        windows_fonts = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
        if windows_fonts.is_dir():
            os.environ.setdefault("QT_QPA_FONTDIR", str(windows_fonts))
        create_application._font_dir_ready = True
    app = QApplication.instance()
    if app is None:
        app = QApplication(argv or [])
    app.setApplicationName("NetPrism")
    app.setApplicationDisplayName("NetPrism · 原生算法透视")
    app.setOrganizationName("NetPrism")
    # Qt's offscreen plugin ships no fonts; explicitly register the Windows CJK
    # UI faces so tests/screenshots render Chinese text instead of tofu.
    for font_name in ("msyh.ttc", "msyhbd.ttc", "msyhl.ttc", "seguisb.ttf", "segoeui.ttf"):
        font_path = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts" / font_name
        if font_path.is_file():
            QFontDatabase.addApplicationFont(str(font_path))
    app.setFont(QFont("Microsoft YaHei UI", 9))
    apply_style(app)
    return app


class MainWindow(QMainWindow):
    def __init__(self, *, load_demo: bool = True, parent=None) -> None:
        super().__init__(parent)
        self.settings = QSettings("NetPrism", "NetPrismGUI")
        self.models: list[dict[str, Any]] = []
        self.graphs: dict[str, dict[str, Any]] = {}
        self.index: dict[str, Any] = {}
        self.current_entry: dict[str, Any] | None = None
        self.current_graph: dict[str, Any] | None = None
        self.current_node_id: str | None = None
        self.source_path: str | None = None
        self.source_kind: str | None = None
        self.parse_worker: ParseWorker | None = None
        self._busy = False
        self._syncing_search = False

        self.setWindowTitle("NetPrism · 原生算法透视")
        self.resize(1440, 900)
        self.setMinimumSize(1080, 700)
        self._build_ui()
        self._build_shortcuts()
        self._set_busy(False)
        if load_demo:
            self.load_demo()

    # ------------------------------------------------------------------ #
    # construction
    # ------------------------------------------------------------------ #
    def _build_ui(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._build_topbar())

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setObjectName("mainSplitter")
        self.splitter.setChildrenCollapsible(False)
        self.splitter.setHandleWidth(4)
        self.splitter.addWidget(self._build_left_panel())
        self.splitter.addWidget(self._build_center())
        self.splitter.addWidget(self._build_right_panel())
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setStretchFactor(2, 0)
        self.splitter.setSizes([232, 840, 350])
        root.addWidget(self.splitter, 1)
        self.setCentralWidget(central)

        status = self.statusBar()
        status.showMessage("就绪")
        self.status_count = QLabel("0 models")
        self.status_count.setObjectName("muted")
        self.status_count.setStyleSheet("color:#58708a; font-family:Consolas; font-size:9px; padding-right:8px;")
        status.addPermanentWidget(self.status_count)
        self.progress = QProgressBar()
        self.progress.setFixedWidth(110)
        self.progress.setFixedHeight(5)
        self.progress.setRange(0, 0)
        self.progress.hide()
        status.addPermanentWidget(self.progress)

        # Readable aliases for tooling/tests that do not know the internal names.
        self.model_list_widget = self.model_list
        self.view = self.view_3d
        self.layer_table = self.table_panel.table_view
        self.algo_panel = self.algorithm_panel
        self.algo_tab = self.algorithm_panel
        self.summary_tab = self.summary_panel
        self.detail_panel = self.details_panel

    def _build_topbar(self) -> QWidget:
        bar = QFrame()
        bar.setObjectName("topBar")
        bar.setFixedHeight(62)
        row = QHBoxLayout(bar)
        row.setContentsMargins(14, 7, 14, 7)
        row.setSpacing(7)

        brand = QVBoxLayout()
        brand.setContentsMargins(0, 0, 10, 0)
        brand.setSpacing(0)
        name = QLabel("NetPrism")
        name.setObjectName("brandName")
        sub = QLabel("NATIVE  ·  ALGORITHM  LENS")
        sub.setObjectName("brandSub")
        brand.addWidget(name)
        brand.addWidget(sub)
        brand_wrap = QWidget()
        brand_wrap.setLayout(brand)
        brand_wrap.setFixedWidth(158)
        row.addWidget(brand_wrap)

        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.VLine)
        divider.setStyleSheet("color: #182b40;")
        row.addWidget(divider)

        self.model_title_label = QLabel("打开模型开始")
        self.model_title_label.setObjectName("modelTitle")
        self.model_title_label.setMinimumWidth(110)
        self.model_title_label.setMaximumWidth(230)
        self.model_title_label.setToolTip("当前模型")
        row.addWidget(self.model_title_label)

        self.open_folder_btn = self._button("打开文件夹", "openFolderButton", "打开模型代码文件夹（trace 执行，可勾选静态解析）")
        self.open_file_btn = self._button("打开模型文件", "openFileButton", "打开 .py / .pt / .pth / .onnx / .keras / graph.json")
        self.reparse_btn = self._button("重新解析", "reparseButton", "用当前模式重新解析来源")
        self.export_report_btn = self._button("导出报告", "exportButton", "导出 Markdown 或 HTML 报告")
        self.export_csv_btn = self._button("导出 CSV", "exportCsvButton", "导出当前层表格为 CSV（UTF-8 BOM）")
        for button in (
            self.open_folder_btn,
            self.open_file_btn,
            self.reparse_btn,
            self.export_report_btn,
            self.export_csv_btn,
        ):
            button.setFixedHeight(30)
            row.addWidget(button)

        self.static_checkbox = QCheckBox("静态解析（安全）")
        self.static_checkbox.setObjectName("staticCheckBox")
        self.static_checkbox.setToolTip("默认执行模型代码做 trace；勾选后只做静态 AST 解析，不执行代码")
        self.static_checkbox.setChecked(False)
        row.addWidget(self.static_checkbox)

        row.addStretch(1)
        self.search_edit = QLineEdit()
        self.search_edit.setObjectName("searchEdit")
        self.search_edit.setPlaceholderText("搜索层 / op / kind / group…")
        self.search_edit.setClearButtonEnabled(True)
        self.search_edit.setFixedWidth(250)
        row.addWidget(self.search_edit)

        self.open_folder_btn.clicked.connect(self.open_folder_dialog)
        self.open_file_btn.clicked.connect(self.open_file_dialog)
        self.reparse_btn.clicked.connect(self.reparse)
        self.export_report_btn.clicked.connect(lambda: self.export_report())
        self.export_csv_btn.clicked.connect(lambda: self.export_csv())
        self.search_edit.textChanged.connect(self._on_global_search)
        return bar

    def _button(self, text: str, object_name: str, tooltip: str) -> QPushButton:
        button = QPushButton(text)
        button.setObjectName(object_name)
        button.setToolTip(tooltip)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        return button

    def _build_left_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("leftPanel")
        panel.setMinimumWidth(196)
        panel.setMaximumWidth(300)
        panel.setStyleSheet(
            "#leftPanel { background: #09111d; border-right: 1px solid #142238; }"
        )
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 12, 10, 10)
        layout.setSpacing(8)
        head = QHBoxLayout()
        title = QLabel("模型")
        title.setStyleSheet("color:#eaf8ff; font-size:13px; font-weight:600;")
        self.model_count_label = QLabel("0")
        self.model_count_label.setObjectName("countLabel")
        head.addWidget(title)
        head.addStretch(1)
        head.addWidget(self.model_count_label)
        layout.addLayout(head)

        self.model_list = QListWidget()
        self.model_list.setObjectName("modelList")
        self.model_list.setSpacing(2)
        self.model_list.setUniformItemSizes(False)
        self.model_list.setWordWrap(True)
        self.model_list.currentRowChanged.connect(self._on_model_row_changed)
        layout.addWidget(self.model_list, 1)

        self.source_label = QLabel("内置演示数据")
        self.source_label.setObjectName("pathLabel")
        self.source_label.setWordWrap(True)
        self.source_label.setStyleSheet("color:#58718c; font-size:9px;")
        layout.addWidget(self.source_label)

        hint = QLabel("拖拽旋转 · 滚轮缩放 · 单击层")
        hint.setObjectName("muted")
        hint.setStyleSheet("color:#3f5872; font-size:9px;")
        layout.addWidget(hint)
        return panel

    def _build_center(self) -> QWidget:
        self.tabs = QTabWidget()
        self.tabs.setObjectName("tabWidget")
        self.tabs.setDocumentMode(True)
        self.view_3d = Graph3DView()
        self.view_3d.nodeSelected.connect(lambda node_id: self.select_node(node_id, sync_view=False, sync_table=True))
        self.table_panel = TablePanel()
        self.table_panel.nodeSelected.connect(self._on_table_node_selected)
        self.table_panel.searchChanged.connect(self._on_table_search)
        self.algorithm_panel = AlgorithmPanel()
        self.summary_panel = SummaryPanel()
        self.tabs.addTab(self.view_3d, "3D 视图")
        self.tabs.addTab(self.table_panel, "层表格")
        self.tabs.addTab(self.algorithm_panel, "算法理解")
        self.tabs.addTab(self.summary_panel, "报告 / 摘要")
        return self.tabs

    def _build_right_panel(self) -> QWidget:
        self.details_panel = DetailsPanel()
        self.details_panel.setMinimumWidth(300)
        self.details_panel.setMaximumWidth(470)
        self.detail_browser = self.details_panel.browser
        return self.details_panel

    def _build_shortcuts(self) -> None:
        QShortcut(QKeySequence("Ctrl+O"), self, activated=self.open_folder_dialog)
        QShortcut(QKeySequence("Ctrl+Shift+O"), self, activated=self.open_file_dialog)
        QShortcut(QKeySequence("Ctrl+R"), self, activated=self.reparse)
        QShortcut(QKeySequence("Ctrl+F"), self, activated=self._focus_search)
        QShortcut(QKeySequence("Ctrl+E"), self, activated=lambda: self.export_csv())

    def _focus_search(self) -> None:
        self.search_edit.setFocus()
        self.search_edit.selectAll()

    # ------------------------------------------------------------------ #
    # model loading
    # ------------------------------------------------------------------ #
    def load_demo(self) -> bool:
        try:
            payload = load_demo_payload()
        except Exception as exc:
            self.statusBar().showMessage(f"内置 demo 载入失败：{exc}")
            return False
        self._apply_payload(payload, source_kind="demo")
        self.statusBar().showMessage("已载入内置 8 模型演示数据")
        return True

    def load_documents(
        self,
        index: dict[str, Any],
        graphs: dict[str, dict[str, Any]],
        *,
        source_path: str | None = None,
        source_kind: str = "memory",
    ) -> None:
        """Public hook used by tests/tools to install already-loaded graphs."""
        self._apply_payload(
            {
                "index": index,
                "graphs": graphs,
                "source_path": source_path,
                "source_kind": source_kind,
            },
            source_kind=source_kind,
        )

    def _apply_payload(self, payload: dict[str, Any], *, source_kind: str | None = None) -> None:
        index = payload.get("index") if isinstance(payload.get("index"), dict) else {}
        graphs = payload.get("graphs") if isinstance(payload.get("graphs"), dict) else {}
        models = [entry for entry in (index.get("models") or []) if isinstance(entry, dict)]
        model_map = {
            str(entry.get("id")): graph
            for entry in models
            for graph in [graphs.get(entry.get("id"))]
            if isinstance(graph, dict)
        }
        models = [entry for entry in models if str(entry.get("id")) in model_map]
        if not models:
            self.statusBar().showMessage("没有找到可显示的模型")
            return
        previous_id = str(self.current_entry.get("id")) if self.current_entry else None
        self.index = index
        self.graphs = {str(key): value for key, value in model_map.items()}
        self.models = models
        self.source_path = payload.get("source_path") or self.source_path
        self.source_kind = source_kind or payload.get("source_kind") or "memory"
        self.current_entry = None
        self.current_graph = None
        self.current_node_id = None

        self.model_list.blockSignals(True)
        self.model_list.clear()
        for entry in models:
            model_id = str(entry.get("id"))
            name = str(entry.get("name") or entry.get("id"))
            mode = mode_label(entry.get("mode"))
            layers = int(entry.get("numLayers") or 0)
            item = QListWidgetItem(f"{name}\n{mode} · {layers} 层")
            item.setData(Qt.ItemDataRole.UserRole, model_id)
            item.setSizeHint(QSize(0, 52))
            item.setToolTip(
                f"{name}\n{entry.get('source') or ''}\nparams {format_count(entry.get('totalParams'))}"
            )
            self.model_list.addItem(item)
        self.model_list.blockSignals(False)
        self.model_count_label.setText(str(len(models)))
        self.status_count.setText(f"{len(models)} models")
        self.source_label.setText(self._source_text())
        self.source_label.setToolTip(str(self.source_path or ""))
        self.reparse_btn.setEnabled(bool(self.source_path))

        row = 0
        if previous_id:
            for index_num, entry in enumerate(models):
                if str(entry.get("id")) == previous_id:
                    row = index_num
                    break
        self.model_list.setCurrentRow(row)
        # blockSignals above skipped the selection signal; display explicitly
        self._on_model_row_changed(row)
        self._set_busy(False)
        self._update_actions()

    def _source_text(self) -> str:
        if self.source_kind == "demo":
            return "内置演示  ·  examples/demo"
        if self.source_path:
            return str(self.source_path)
        return "内存数据"

    def _on_model_row_changed(self, row: int) -> None:
        if row < 0 or row >= len(self.models):
            return
        entry = self.models[row]
        graph = self.graphs.get(str(entry.get("id")))
        if not isinstance(graph, dict):
            return
        self.current_entry = entry
        self.current_graph = graph
        self.current_node_id = None
        self._show_model(entry, graph)

    def _show_model(self, entry: dict[str, Any], graph: dict[str, Any]) -> None:
        name = str(entry.get("name") or entry.get("id"))
        self.model_title_label.setText(name)
        self.model_title_label.setToolTip(str(entry.get("source") or name))
        self.view_3d.set_caption(
            f"{name}  ·  {format_int(entry.get('numLayers'))} layers  ·  {mode_label(entry.get('mode'))}"
        )
        self.view_3d.set_graph(graph)
        self.table_panel.set_graph(graph)
        self.algorithm_panel.set_graph(graph)
        self.summary_panel.set_graph(graph)
        self.details_panel.set_graph(graph)
        self.current_node_id = None
        self._syncing_search = True
        with QSignalBlocker(self.search_edit):
            self.search_edit.clear()
        self._syncing_search = False
        self.table_panel.set_search("")
        self.view_3d.highlight_nodes([])
        model = graph.get("model") if isinstance(graph.get("model"), dict) else {}
        _meta, family, family_label, _confidence, _tags = family_from_graph(graph)
        bits = [
            str(model.get("name") or name),
            framework_label(model.get("framework")),
            mode_label(model.get("mode")),
            f"{format_int(model.get('numLayers'))} 层",
            f"params {format_count(model.get('totalParams'))}",
        ]
        if family:
            bits.append(f"family {family_label or family}")
        self.statusBar().showMessage("  ·  ".join(bits))
        self._update_actions()

    # ------------------------------------------------------------------ #
    # selection / search
    # ------------------------------------------------------------------ #
    def select_node(self, node_id: str, *, sync_view: bool = True, sync_table: bool = True) -> None:
        if not self.current_graph or not node_id:
            return
        node = next(
            (item for item in (self.current_graph.get("nodes") or []) if isinstance(item, dict) and str(item.get("id")) == str(node_id)),
            None,
        )
        if node is None:
            return
        self.current_node_id = str(node_id)
        self.details_panel.show_node(node)
        if sync_view:
            self.view_3d.select_node(str(node_id), notify=False)
        if sync_table:
            self.table_panel.select_node(str(node_id))
        self.statusBar().showMessage(f"层 {node.get('name') or node_id}  ·  {node.get('op') or '—'}")

    def _on_table_node_selected(self, node_id: str) -> None:
        self.select_node(node_id, sync_view=True, sync_table=False)

    def _on_global_search(self, text: str) -> None:
        if self._syncing_search:
            return
        self._syncing_search = True
        self.table_panel.set_search(text)
        self._syncing_search = False
        self._highlight_search(text)

    def _on_table_search(self, text: str) -> None:
        if self._syncing_search:
            return
        self._syncing_search = True
        with QSignalBlocker(self.search_edit):
            self.search_edit.setText(text)
        self._syncing_search = False
        self._highlight_search(text)

    def _highlight_search(self, text: str) -> None:
        query = str(text or "").strip().lower()
        if not query or not self.current_graph:
            self.view_3d.highlight_nodes([])
            return
        ids = [
            str(node.get("id"))
            for node in (self.current_graph.get("nodes") or [])
            if isinstance(node, dict) and query in self._node_search_text(node)
        ]
        self.view_3d.highlight_nodes(ids)
        if ids:
            self.statusBar().showMessage(f"搜索 “{text}” · {len(ids)} 个匹配层")
        else:
            self.statusBar().showMessage(f"搜索 “{text}” · 无匹配层")

    def _node_search_text(self, node: dict[str, Any]) -> str:
        return " ".join(
            [
                str(node.get("id") or ""),
                str(node.get("name") or ""),
                str(node.get("op") or ""),
                str(node.get("kind") or ""),
                str(node.get("groupId") or ""),
                " ".join(str(tag) for tag in (node.get("tags") or [])),
            ]
        ).lower()

    # ------------------------------------------------------------------ #
    # parsing
    # ------------------------------------------------------------------ #
    def open_folder_dialog(self) -> None:
        start = self._recent_dir()
        folder = QFileDialog.getExistingDirectory(self, "打开模型文件夹", start)
        if folder:
            self._remember_dir(Path(folder))
            self.open_path(folder, background=True)

    def open_file_dialog(self) -> None:
        start = self._recent_dir()
        filters = "模型文件 (*.py *.pt *.pth *.onnx *.keras *.json);;所有文件 (*.*)"
        path, _selected = QFileDialog.getOpenFileName(self, "打开模型文件", start, filters)
        if path:
            self._remember_dir(Path(path))
            self.open_path(path, background=True)

    def open_path(self, path: str | Path, *, background: bool = True) -> bool:
        source = Path(path).expanduser().resolve()
        if not source.exists():
            self._show_error("路径不存在", f"{source}")
            return False
        self._remember_dir(source.parent if source.is_file() else source)
        if background:
            self._start_parse(source, background=True)
            return True
        try:
            payload = parse_path_sync(source, static=self.static_checkbox.isChecked())
        except Exception as exc:
            self._show_error("解析失败", str(exc))
            return False
        self._apply_payload(payload, source_kind=payload.get("source_kind"))
        self.statusBar().showMessage(f"解析完成：{source}")
        self.tabs.setCurrentIndex(0)
        return True

    def _start_parse(self, source: Path, *, background: bool) -> None:
        if self._busy:
            return
        self._set_busy(True)
        self.statusBar().showMessage(f"解析中：{source}  ·  {'静态 AST' if self.static_checkbox.isChecked() else 'trace'}")
        self.progress.show()
        worker = ParseWorker(source, static=self.static_checkbox.isChecked(), parent=self)
        self.parse_worker = worker
        worker.succeeded.connect(self._on_parse_succeeded)
        worker.failed.connect(self._on_parse_failed)
        worker.finished.connect(lambda: self._on_worker_finished(worker))
        worker.start()

    def _on_parse_succeeded(self, payload: Any) -> None:
        if not isinstance(payload, dict):
            self._on_parse_failed("解析器返回了无效结果")
            return
        self._apply_payload(payload, source_kind=payload.get("source_kind"))
        self.statusBar().showMessage("解析完成")
        self.tabs.setCurrentIndex(0)
        self._set_busy(False)

    def _on_parse_failed(self, message: str) -> None:
        self._set_busy(False)
        self.statusBar().showMessage("解析失败")
        self._show_error("解析失败", str(message))

    def _on_worker_finished(self, worker: ParseWorker) -> None:
        if self.parse_worker is worker:
            self.parse_worker = None
        self.progress.hide()
        worker.deleteLater()

    def wait_for_parse(self, timeout_ms: int = 120000) -> bool:
        worker = self.parse_worker
        if worker is None:
            return True
        return worker.wait(timeout_ms)

    def reparse(self) -> None:
        if self._busy:
            return
        if self.source_kind == "demo" or not self.source_path:
            if demo_dir() is not None:
                self.load_demo()
            return
        self.open_path(self.source_path, background=True)

    # ------------------------------------------------------------------ #
    # exports
    # ------------------------------------------------------------------ #
    def export_csv(self, path: str | Path | None = None) -> Path | None:
        if not self.current_graph:
            self._show_error("无法导出", "当前没有模型。")
            return None
        target: Path | None
        if path is None:
            start = self._recent_dir()
            suggested = Path(start) / f"{self.current_entry.get('name') if self.current_entry else 'model'}_layers.csv"
            selected, _filter = QFileDialog.getSaveFileName(self, "导出 CSV", str(suggested), "CSV (*.csv)")
            target = Path(selected) if selected else None
        else:
            target = Path(path)
        if target is None:
            return None
        if target.suffix.lower() != ".csv":
            target = target.with_suffix(".csv")
        target.parent.mkdir(parents=True, exist_ok=True)
        self.table_panel.export_csv(target)
        self._remember_dir(target.parent)
        self.statusBar().showMessage(f"CSV 已导出：{target}")
        return target

    def export_report(self, path: str | Path | None = None, fmt: str | None = None) -> Path | None:
        if not self.current_graph:
            self._show_error("无法导出", "当前没有模型。")
            return None
        target: Path | None
        chosen_format = (fmt or "").lower()
        if path is None:
            start = self._recent_dir()
            name = str(self.current_entry.get("name") if self.current_entry else "model")
            selected, selected_filter = QFileDialog.getSaveFileName(
                self,
                "导出报告",
                str(Path(start) / f"{name}_report.md"),
                "Markdown (*.md);;HTML (*.html)",
            )
            target = Path(selected) if selected else None
            if target is not None and not chosen_format:
                chosen_format = "html" if target.suffix.lower() == ".html" else "md"
        else:
            target = Path(path)
        if target is None:
            return None
        if not chosen_format:
            chosen_format = "html" if target.suffix.lower() == ".html" else "md"
        if target.suffix.lower() not in (".md", ".html"):
            target = target.with_suffix(".html" if chosen_format == "html" else ".md")
        target.parent.mkdir(parents=True, exist_ok=True)
        text = render_report(self.current_graph, fmt=chosen_format)
        target.write_text(text, encoding="utf-8")
        self._remember_dir(target.parent)
        self.statusBar().showMessage(f"报告已导出：{target}")
        return target

    # ------------------------------------------------------------------ #
    # options / state
    # ------------------------------------------------------------------ #
    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        for button in (self.open_folder_btn, self.open_file_btn, self.reparse_btn):
            button.setEnabled(not busy)
        self.static_checkbox.setEnabled(not busy)
        self.model_list.setEnabled(not busy)
        if busy:
            self.progress.show()
        else:
            self.progress.hide()
        self._update_actions()

    def _update_actions(self) -> None:
        has_model = self.current_graph is not None
        self.export_report_btn.setEnabled(has_model and not self._busy)
        self.export_csv_btn.setEnabled(has_model and not self._busy)
        self.reparse_btn.setEnabled(bool(self.source_path) and not self._busy)

    def _recent_dir(self) -> str:
        value = self.settings.value("lastDir", "", type=str)
        if value and Path(value).is_dir():
            return value
        home = Path.home()
        return str(home if home.is_dir() else Path.cwd())

    def _remember_dir(self, directory: Path) -> None:
        try:
            if directory.is_file():
                directory = directory.parent
            self.settings.setValue("lastDir", str(directory))
        except Exception:
            pass

    def _show_error(self, title: str, message: str) -> None:
        self.statusBar().showMessage(f"{title}：{message}")
        QMessageBox.critical(self, title, message)

    def closeEvent(self, event) -> None:  # noqa: N802
        worker = self.parse_worker
        if worker is not None and worker.isRunning():
            worker.requestInterruption()
            worker.wait(2500)
        super().closeEvent(event)


__all__ = ["MainWindow", "create_application"]
