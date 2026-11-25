"""Shared helpers and the dark visual language for the native GUI."""

from __future__ import annotations

import html
import math
from typing import Any, Iterable

KIND_COLORS: dict[str, str] = {
    "input": "#22d3ee",
    "output": "#f472b6",
    "conv": "#38bdf8",
    "deconv": "#0ea5e9",
    "norm": "#34d399",
    "activation": "#fbbf24",
    "pool": "#60a5fa",
    "linear": "#a78bfa",
    "attention": "#c084fc",
    "add": "#fb7185",
    "concat": "#f97316",
    "flatten": "#94a3b8",
    "reshape": "#94a3b8",
    "dropout": "#94a3b8",
    "embedding": "#2dd4bf",
    "softmax": "#f59e0b",
    "rnn": "#ec4899",
    "custom": "#a3a3a3",
    "unknown": "#94a3b8",
}

KIND_LABELS: dict[str, str] = {
    "input": "输入",
    "output": "输出",
    "conv": "卷积",
    "deconv": "转置卷积",
    "norm": "归一化",
    "activation": "激活",
    "pool": "池化",
    "linear": "线性层",
    "attention": "注意力",
    "add": "相加",
    "concat": "拼接",
    "flatten": "展平",
    "reshape": "重塑",
    "dropout": "Dropout",
    "embedding": "嵌入",
    "softmax": "Softmax",
    "rnn": "循环层",
    "custom": "自定义",
    "unknown": "未知",
}

GROUP_COLORS = {
    "model": "#64748b",
    "module": "#64748b",
    "sequential": "#64748b",
    "conv_block": "#22d3ee",
    "mlp": "#a78bfa",
    "residual_block": "#fb923c",
    "transformer_block": "#c084fc",
    "attention_block": "#e879f9",
    "ffn_block": "#818cf8",
    "unet_block": "#34d399",
    "custom": "#94a3b8",
}

EDGE_COLORS = {
    "data": "#2f6f9e",
    "residual": "#fb923c",
    "skip": "#f59e0b",
    "attention": "#c084fc",
}

FRAMEWORK_LABELS = {
    "pytorch": "PyTorch",
    "onnx": "ONNX",
    "keras": "Keras",
    "torchscript": "TorchScript",
    "unknown": "未知格式",
}

MODE_LABELS = {
    "trace": "trace",
    "static": "静态",
    "hybrid": "hybrid",
    "onnx": "ONNX",
    "keras": "Keras",
    "torchscript": "TorchScript",
}


def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def kind_color(kind: str | None) -> str:
    return KIND_COLORS.get(str(kind or "unknown").lower(), KIND_COLORS["unknown"])


def group_color(kind: str | None) -> str:
    return GROUP_COLORS.get(str(kind or "custom").lower(), GROUP_COLORS["custom"])


def edge_color(kind: str | None) -> str:
    return EDGE_COLORS.get(str(kind or "data").lower(), EDGE_COLORS["data"])


def kind_label(kind: str | None) -> str:
    key = str(kind or "unknown").lower()
    return KIND_LABELS.get(key, key)


def framework_label(value: Any) -> str:
    key = str(value or "unknown").lower()
    return FRAMEWORK_LABELS.get(key, key or "未知格式")


def mode_label(value: Any) -> str:
    key = str(value or "unknown").lower()
    return MODE_LABELS.get(key, key)


def format_int(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "—"
    return f"{int(value):,}"


def format_count(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "—"
    number = float(value)
    sign = "-" if number < 0 else ""
    number = abs(number)
    for unit in ("", "K", "M", "G", "T"):
        if number < 1000 or unit == "T":
            if unit == "":
                return f"{sign}{int(number)}"
            return f"{sign}{number:.1f}{unit}"
        number /= 1000.0
    return f"{sign}{int(number)}"


def format_mb(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        return "—"
    mb = float(value) / (1024.0 * 1024.0)
    if mb >= 100:
        return f"{mb:.0f} MB"
    if mb >= 10:
        return f"{mb:.1f} MB"
    if mb >= 1:
        return f"{mb:.2f} MB"
    return f"{mb:.3f} MB"


def format_shape(shape: Any) -> str:
    if not isinstance(shape, (list, tuple)) or not shape:
        return "—"
    return "×".join("?" if not isinstance(dim, int) or dim < 0 else str(dim) for dim in shape)


def shapes_text(shapes: Any) -> str:
    if not isinstance(shapes, (list, tuple)) or not shapes:
        return "—"
    return " + ".join(format_shape(shape) for shape in shapes)


def format_confidence(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    if number > 1:
        number /= 100.0
    return max(0, min(100, int(round(number * 100))))


def confidence_title(value: Any) -> str:
    percent = format_confidence(value)
    return "—" if percent is None else f"{percent}%"


def group_label(graph: dict[str, Any], group_id: str | None) -> str:
    if not group_id:
        return "—"
    groups = {
        group.get("id"): group
        for group in graph.get("groups", [])
        if isinstance(group, dict) and isinstance(group.get("id"), str)
    }
    group = groups.get(group_id)
    if not group:
        return group_id
    parts: list[str] = []
    seen: set[str] = set()
    while group and len(parts) < 12:
        gid = group.get("id")
        if not gid or gid in seen:
            break
        seen.add(gid)
        text = group.get("label") or group.get("name") or gid
        parts.append(str(text))
        parent = group.get("parentId")
        group = groups.get(parent)
    parts.reverse()
    return " / ".join(parts) if parts else group_id


def group_path(graph: dict[str, Any], group_id: str | None) -> str:
    return group_label(graph, group_id)


def semantic_of(node: dict[str, Any]) -> str:
    meta = node.get("meta") if isinstance(node.get("meta"), dict) else {}
    value = meta.get("semantic")
    return str(value).strip() if value else ""


def formula_of(node: dict[str, Any]) -> str:
    meta = node.get("meta") if isinstance(node.get("meta"), dict) else {}
    value = meta.get("formula")
    return str(value).strip() if value else ""


def node_tooltip(node: dict[str, Any], group_name: str = "") -> str:
    lines = [
        f"{node.get('name') or node.get('id')}  ·  {node.get('op') or '—'}",
        f"{kind_label(node.get('kind'))}  ·  {shapes_text(node.get('outShapes'))}",
    ]
    if group_name:
        lines.append(group_name)
    lines.append(f"params {format_count(node.get('numParams'))}  ·  FLOPs {format_count(node.get('flops'))}")
    metrics = node.get("metrics") if isinstance(node.get("metrics"), dict) else {}
    if isinstance(metrics.get("activationBytes"), (int, float)):
        lines.append(f"激活显存 {format_mb(metrics.get('activationBytes'))}")
    semantic = semantic_of(node)
    if semantic:
        lines.append(semantic[:160])
    return "\n".join(lines)


def iter_algo_patterns(graph: dict[str, Any]) -> list[dict[str, Any]]:
    model = graph.get("model") if isinstance(graph.get("model"), dict) else {}
    meta = model.get("meta") if isinstance(model.get("meta"), dict) else {}
    patterns = meta.get("patterns")
    return [p for p in patterns if isinstance(p, dict)] if isinstance(patterns, list) else []


def iter_algo_pseudocode(graph: dict[str, Any]) -> list[str]:
    model = graph.get("model") if isinstance(graph.get("model"), dict) else {}
    meta = model.get("meta") if isinstance(model.get("meta"), dict) else {}
    code = meta.get("pseudocode")
    if isinstance(code, str):
        return [line for line in code.splitlines() if line.strip()]
    if isinstance(code, list):
        return [str(line) for line in code if str(line).strip()]
    return []


def family_from_graph(graph: dict[str, Any]) -> tuple[dict[str, Any], str, str, int | None, list[str]]:
    model = graph.get("model") if isinstance(graph.get("model"), dict) else {}
    meta = model.get("meta") if isinstance(model.get("meta"), dict) else {}
    family = str(meta.get("family") or "")
    label = str(meta.get("familyLabel") or family or "未知家族")
    tags = meta.get("algorithmTags") or meta.get("tags") or []
    if not isinstance(tags, list):
        tags = [str(tags)]
    tags = [str(tag) for tag in tags if str(tag)]
    confidence = format_confidence(
        meta.get("familyConfidence", meta.get("family_confidence", meta.get("confidence")))
    )
    if confidence is None:
        confidences = [
            format_confidence(pattern.get("confidence"))
            for pattern in iter_algo_patterns(graph)
            if format_confidence(pattern.get("confidence")) is not None
        ]
        if confidences:
            confidence = max(confidences)  # type: ignore[arg-type]
    return meta, family, label, confidence, tags


def algo_has_content(graph: dict[str, Any]) -> bool:
    meta, family, label, _confidence, tags = family_from_graph(graph)
    return bool(meta and (family or label or tags or iter_algo_patterns(graph) or iter_algo_pseudocode(graph)))


STYLE = """
* {
    font-family: "Microsoft YaHei UI", "Segoe UI", "Noto Sans CJK SC", sans-serif;
}
QWidget {
    background: #070b14;
    color: #dbeafe;
    font-size: 12px;
}
QMainWindow, QDialog {
    background: #070b14;
}
QFrame#topBar {
    background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #0d1627, stop:1 #09111f);
    border-bottom: 1px solid #17263d;
}
QLabel#brandName {
    color: #f0f9ff;
    font-size: 18px;
    font-weight: 700;
    letter-spacing: 1px;
}
QLabel#brandSub {
    color: #536b88;
    font-size: 9px;
    letter-spacing: 2px;
}
QLabel#modelTitle {
    color: #f0f9ff;
    font-size: 14px;
    font-weight: 600;
}
QLabel#pathLabel, QLabel#muted, QLabel#countLabel {
    color: #6f849e;
    font-size: 10px;
}
QPushButton {
    background: #0f1a2b;
    color: #c7d8eb;
    border: 1px solid #20314b;
    border-radius: 7px;
    padding: 6px 10px;
    min-height: 17px;
}
QPushButton:hover {
    background: #14253c;
    border-color: #315579;
}
QPushButton:pressed {
    background: #0b1729;
}
QPushButton:disabled {
    color: #4b5f77;
    border-color: #172436;
    background: #0a111d;
}
QPushButton#openFolderButton, QPushButton#openFileButton {
    background: #0d2a37;
    border-color: #1b596a;
    color: #bdf3ff;
}
QPushButton#openFolderButton:hover, QPushButton#openFileButton:hover {
    background: #10394a;
    border-color: #2b8ba1;
}
QPushButton#exportButton, QPushButton#exportCsvButton {
    background: #151a34;
    border-color: #2a3157;
    color: #cfc8ff;
}
QPushButton#exportButton:hover, QPushButton#exportCsvButton:hover {
    border-color: #5d54a0;
    background: #1c2142;
}
QCheckBox {
    color: #9fb3c9;
    spacing: 7px;
    font-size: 10px;
}
QCheckBox::indicator {
    width: 14px;
    height: 14px;
    border-radius: 4px;
    border: 1px solid #31445e;
    background: #0a1322;
}
QCheckBox::indicator:checked {
    background: #0e7490;
    border-color: #22d3ee;
}
QLineEdit {
    background: #0a1322;
    color: #e3f2ff;
    border: 1px solid #1d2e45;
    border-radius: 7px;
    padding: 6px 10px;
    selection-background-color: #155e75;
}
QLineEdit:focus {
    border-color: #2b7f98;
}
QLineEdit::placeholder {
    color: #50657e;
}
QListWidget {
    background: transparent;
    border: none;
    outline: none;
    padding: 3px;
}
QListWidget::item {
    background: transparent;
    border: 1px solid transparent;
    border-radius: 8px;
    padding: 8px 9px;
    margin: 2px 2px;
    color: #a9bfd4;
}
QListWidget::item:hover {
    background: #0c1929;
    border-color: #172c42;
}
QListWidget::item:selected {
    background: #0d2536;
    border-color: #1c5368;
    color: #e6f8ff;
}
QTabWidget::pane {
    border: 1px solid #16243a;
    border-radius: 9px;
    background: #080e19;
    top: -1px;
}
QTabBar::tab {
    background: transparent;
    color: #748aa4;
    border: 1px solid transparent;
    border-bottom: none;
    padding: 8px 14px;
    margin-right: 3px;
    border-top-left-radius: 7px;
    border-top-right-radius: 7px;
    font-size: 11px;
}
QTabBar::tab:hover {
    color: #b7d3e9;
    background: #0b1625;
}
QTabBar::tab:selected {
    color: #8df1ff;
    background: #0a1726;
    border-color: #1b3a52;
}
QTableView {
    background: #080f1b;
    alternate-background-color: #0a1422;
    gridline-color: #132033;
    border: none;
    selection-background-color: #0d3545;
    selection-color: #e8fbff;
    outline: none;
    font-size: 10px;
}
QTableView::item {
    padding: 4px 7px;
    border-bottom: 1px solid #101c2e;
}
QTableView::item:selected {
    background: #0d3545;
}
QHeaderView::section {
    background: #0c1726;
    color: #7c96b2;
    border: none;
    border-right: 1px solid #142238;
    border-bottom: 1px solid #1b3048;
    padding: 6px 7px;
    font-size: 10px;
    font-weight: 600;
}
QHeaderView::section:hover {
    background: #102136;
    color: #b9e9f7;
}
QTextBrowser {
    background: transparent;
    border: none;
    padding: 2px;
    selection-background-color: #155e75;
}
QPlainTextEdit {
    background: #050b14;
    color: #a9d8ea;
    border: 1px solid #14283d;
    border-radius: 7px;
    padding: 8px;
    font-family: "Cascadia Mono", "Consolas", "JetBrains Mono", monospace;
    font-size: 10px;
    selection-background-color: #155e75;
}
QScrollArea {
    border: none;
    background: transparent;
}
QScrollBar:vertical {
    background: transparent;
    width: 9px;
    margin: 2px;
}
QScrollBar::handle:vertical {
    background: #1b3149;
    border-radius: 4px;
    min-height: 28px;
}
QScrollBar::handle:vertical:hover {
    background: #285273;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
    background: transparent;
    height: 0;
}
QScrollBar:horizontal {
    background: transparent;
    height: 9px;
    margin: 2px;
}
QScrollBar::handle:horizontal {
    background: #1b3149;
    border-radius: 4px;
    min-width: 28px;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal,
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
    background: transparent;
    width: 0;
}
QSplitter::handle {
    background: #0c1625;
}
QSplitter::handle:hover {
    background: #143049;
}
QStatusBar {
    background: #080e19;
    border-top: 1px solid #142033;
    color: #7087a0;
    font-size: 10px;
}
QStatusBar::item {
    border: none;
}
QProgressBar {
    background: #0c1726;
    border: none;
    border-radius: 3px;
    height: 5px;
    text-align: center;
    color: transparent;
}
QProgressBar::chunk {
    background: #22d3ee;
    border-radius: 3px;
}
QToolTip {
    background: #0b1728;
    color: #d6ecff;
    border: 1px solid #2b4c66;
    border-radius: 5px;
    padding: 6px;
    font-size: 10px;
}
QMessageBox {
    background: #0b1422;
}
QFileDialog {
    background: #0b1422;
}
"""


def apply_style(app: Any) -> None:
    """Apply the Fusion style + dark palette + stylesheet."""
    from PySide6.QtGui import QColor, QPalette
    from PySide6.QtWidgets import QApplication

    try:
        app.setStyle("Fusion")
    except Exception:
        pass
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor("#070b14"))
    palette.setColor(QPalette.ColorRole.WindowText, QColor("#dbeafe"))
    palette.setColor(QPalette.ColorRole.Base, QColor("#080f1b"))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor("#0a1422"))
    palette.setColor(QPalette.ColorRole.Text, QColor("#dbeafe"))
    palette.setColor(QPalette.ColorRole.Button, QColor("#0f1a2b"))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor("#c7d8eb"))
    palette.setColor(QPalette.ColorRole.Highlight, QColor("#0e7490"))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#f0fdff"))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor("#0b1728"))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor("#d6ecff"))
    app.setPalette(palette)
    app.setStyleSheet(STYLE)
