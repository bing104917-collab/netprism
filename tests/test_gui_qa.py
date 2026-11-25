"""Independent QA for the native PySide6 GUI (task-18).

The GUI must be a real desktop application: no HTTP server, no browser, no
QWebEngine.  These tests are independent from ``tests/test_gui.py`` (they use
the public window/panel API, a fresh offscreen application and real
subprocesses for the launcher / frozen executable).

Run with the project root on ``PYTHONPATH``:

    python -m pytest tests/test_gui.py tests/test_gui_qa.py -q
"""

from __future__ import annotations

import ast
import copy
import csv
import io
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtGui import QGuiApplication, QImage  # noqa: E402
from PySide6.QtWidgets import QMessageBox  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
GUI_DIR = ROOT / "netprism" / "gui"
LAUNCHER = ROOT / "packaging" / "gui_launcher.py"
SAMPLES = ROOT / "samples"
SCRATCH = ROOT / "scratch" / "qa2"

from netprism.gui import main_window as gui_main  # noqa: E402
from netprism.gui.graph_loader import load_demo_payload, project_root  # noqa: E402
from netprism.gui.main_window import MainWindow, create_application  # noqa: E402
from netprism.gui.panels import AlgorithmPanel, TablePanel  # noqa: E402

CSV_HEADER = [
    "name",
    "op",
    "kind",
    "in_shapes",
    "out_shapes",
    "params",
    "flops",
    "activation_mb",
    "receptive_field",
    "group",
]

FORBIDDEN_IMPORT_ROOTS = {"webbrowser", "socketserver", "http", "socket"}
FORBIDDEN_IMPORT_MODULES = {"netprism.server", "PySide6.QtWebEngineWidgets", "PySide6.QtWebEngineCore"}
FORBIDDEN_CALLS = {"serve", "HTTPServer", "ThreadingHTTPServer"}


# --------------------------------------------------------------------------- #
# environment / fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def app():
    application = create_application([])
    application.setQuitOnLastWindowClosed(False)
    return application


@pytest.fixture()
def window(app):
    win = MainWindow()
    yield win
    win.close()
    win.deleteLater()
    app.processEvents()


def _pump(app, rounds: int = 4) -> None:
    for _ in range(rounds):
        app.processEvents()


def _wait_parse(app, window, timeout: float = 90.0) -> None:
    deadline = time.monotonic() + timeout
    while window.parse_worker is not None and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    _pump(app)
    assert window.parse_worker is None, "background parse worker did not finish in time"


def _row_value(panel: TablePanel, row: int, column: int):
    return panel.proxy.index(row, column).data(Qt.ItemDataRole.UserRole)


def _normalized(text: str) -> str:
    """Collapse whitespace: QTextDocument collapses repeated spaces when rendering."""
    return " ".join(str(text).split())


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONPATH"] = str(ROOT)
    return env


def _copy_sample_to_tmp(tmp_path: Path) -> Path:
    folder = tmp_path / "custom_sample"
    folder.mkdir()
    shutil.copy(SAMPLES / "mlp.py", folder / "mlp.py")
    return folder


# --------------------------------------------------------------------------- #
# 1. source-level: no server / browser / web engine
# --------------------------------------------------------------------------- #
def test_gui_source_has_no_server_or_web_engine_imports() -> None:
    gui_files = sorted(GUI_DIR.glob("*.py"))
    assert len(gui_files) >= 7, f"unexpected GUI package layout: {[p.name for p in gui_files]}"
    assert LAUNCHER.is_file()
    assert "--smoke" in LAUNCHER.read_text(encoding="utf-8")
    sources = gui_files + [LAUNCHER]

    for path in sources:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module = alias.name
                    root = module.split(".")[0]
                    assert module not in FORBIDDEN_IMPORT_MODULES, f"{path.name}: imports {module}"
                    assert root not in FORBIDDEN_IMPORT_ROOTS, f"{path.name}: imports {module}"
                    assert not module.startswith("PySide6.QtWebEngine"), f"{path.name}: imports {module}"
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                root = module.split(".")[0]
                assert module not in FORBIDDEN_IMPORT_MODULES, f"{path.name}: imports {module}"
                assert root not in FORBIDDEN_IMPORT_ROOTS, f"{path.name}: imports {module}"
                assert not module.startswith("PySide6.QtWebEngine"), f"{path.name}: imports {module}"
                if node.level:
                    imported = {alias.name for alias in node.names}
                    assert imported.isdisjoint({"server", "serve"}), (
                        f"{path.name}: relative import of a server module"
                    )
            elif isinstance(node, ast.Call):
                function = node.func
                name = function.id if isinstance(function, ast.Name) else (
                    function.attr if isinstance(function, ast.Attribute) else ""
                )
                assert name not in FORBIDDEN_CALLS, f"{path.name}: calls {name}()"
                assert not name.startswith("QWebEngine"), f"{path.name}: calls {name}()"
            elif isinstance(node, ast.Attribute):
                assert node.attr not in FORBIDDEN_CALLS, f"{path.name}: references {node.attr}"

        lowered = source.lower()
        assert "localhost" not in lowered, f"{path.name}: localhost URL"
        assert "127.0.0.1" not in lowered, f"{path.name}: 127.0.0.1 URL"


def test_gui_runtime_imports_no_server_modules() -> None:
    code = r"""
import sys

import netprism.gui
from netprism.gui.main_window import MainWindow, create_application

app = create_application([])
window = MainWindow(load_demo=True)
print("MODELS", len(window.models))
print("SERVER_MODULE", any(name == "netprism.server" or name.startswith("netprism.server.") for name in sys.modules))
print("HTTP_SERVER_MODULE", "http.server" in sys.modules)
print("WEBBROWSER_MODULE", "webbrowser" in sys.modules)
print("WEBENGINE_MODULE", any(name.startswith("PySide6.QtWebEngine") for name in sys.modules))
print("SOCKETSERVER_MODULE", "socketserver" in sys.modules)
window.close()
"""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=_subprocess_env(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
    )
    assert proc.returncode == 0, f"GUI import failed:\n{proc.stderr[-2000:]}"
    values: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        key, _, value = line.partition(" ")
        values[key] = value.strip()
    assert values.get("MODELS") == "8", proc.stdout
    for marker in (
        "SERVER_MODULE",
        "HTTP_SERVER_MODULE",
        "WEBBROWSER_MODULE",
        "WEBENGINE_MODULE",
        "SOCKETSERVER_MODULE",
    ):
        assert values.get(marker) == "False", f"{marker}={values.get(marker)!r}: a server/browser module was imported"


# --------------------------------------------------------------------------- #
# 2. offscreen window: 8 models, 3D, table, algorithm
# --------------------------------------------------------------------------- #
def test_offscreen_eight_models_3d_table_and_algorithm(app, window) -> None:
    assert len(window.models) == 8
    assert window.model_list.count() == 8
    window.resize(1440, 900)
    window.show()
    _pump(app)

    tab_titles = [window.tabs.tabText(index) for index in range(window.tabs.count())]
    assert all(tab_titles), tab_titles
    for token in ("3D", "表格", "算法", "摘要"):
        assert any(token in title for title in tab_titles), f"missing tab {token!r}: {tab_titles}"
    assert any(ord(char) > 127 for title in tab_titles for char in title), "GUI tabs are not localized"

    for row in range(window.model_list.count()):
        window.model_list.setCurrentRow(row)
        _pump(app)
        graph = window.current_graph
        assert graph is not None
        node_count = len(graph["nodes"])
        assert window.table_panel.row_count == node_count
        assert window.table_panel.visible_row_count == node_count
        assert len(window.view_3d._node_items) == node_count
        assert len(window.view_3d.scene().items()) > node_count
        assert window.algorithm_panel.browser.toPlainText().strip()
        assert window.summary_panel.browser.toPlainText().strip()
    window.model_list.setCurrentRow(0)
    _pump(app)


def test_layer_detail_formula_and_group_pattern(app, window) -> None:
    window.show()
    _pump(app)
    window.select_node("n1")
    _pump(app)
    graph = window.current_graph
    node = next(item for item in graph["nodes"] if item["id"] == "n1")
    detail = window.detail_browser.toPlainText()

    assert node["name"] in detail
    assert node["op"] in detail
    assert "IN" in detail and "OUT" in detail
    assert "params" in detail
    assert f"{node['numParams']:,}" in detail or str(node["numParams"]) in detail
    assert node["meta"]["formula"] in detail or _normalized(node["meta"]["formula"]) in _normalized(detail)
    assert node["meta"]["semantic"] in detail
    assert "算法解释" in detail
    assert "所属模式" in detail  # VGG features.0 lives in a patterned group

    summary = window.summary_panel.browser.toPlainText()
    assert graph["model"]["name"] in summary
    assert "参数" in summary
    assert any(token in summary for token in ("激活显存", "感受野", "估算"))


def test_no_meta_graph_does_not_crash(app) -> None:
    payload = load_demo_payload()
    entry = next(
        model for model in payload["index"]["models"] if model["name"] == "VGGMini"
    )
    graph = copy.deepcopy(payload["graphs"][entry["id"]])
    graph["model"].pop("meta", None)
    for node in graph["nodes"]:
        node.pop("meta", None)
    for group in graph["groups"]:
        group.pop("meta", None)

    window = MainWindow(load_demo=False)
    try:
        window.load_documents(
            {"schemaVersion": "1.0", "models": [entry], "errors": []},
            {entry["id"]: graph},
            source_kind="memory",
        )
        _pump(app)
        assert len(window.models) == 1
        assert window.current_graph is not None
        assert window.table_panel.row_count == len(graph["nodes"])
        assert len(window.view_3d._node_items) == len(graph["nodes"])
        assert "暂无算法信息" in window.algorithm_panel.browser.toPlainText()
        assert not window.algorithm_panel.copy_button.isEnabled()
        assert window.algorithm_panel.pseudocode.toPlainText() == ""
        window.select_node("n1")
        _pump(app)
        assert window.detail_browser.toPlainText().strip()
        assert "算法解释" not in window.detail_browser.toPlainText()

        panel = AlgorithmPanel()
        panel.set_graph(graph)
        assert "暂无算法信息" in panel.browser.toPlainText()
        assert not panel.copy_button.isEnabled()
        panel.deleteLater()
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()


# --------------------------------------------------------------------------- #
# 3. table sort / search / CSV + algorithm tab
# --------------------------------------------------------------------------- #
def test_table_sort_search_and_csv_export(app, window, tmp_path: Path) -> None:
    window.show()
    _pump(app)
    panel = window.table_panel
    graph = window.current_graph
    total = len(graph["nodes"])
    assert panel.row_count == total

    panel.table_view.sortByColumn(5, Qt.SortOrder.DescendingOrder)
    _pump(app)
    descending = [_row_value(panel, row, 5) for row in range(panel.visible_row_count)]
    assert descending == sorted(descending, reverse=True), "params column is not sorted descending"

    panel.table_view.sortByColumn(5, Qt.SortOrder.AscendingOrder)
    _pump(app)
    ascending = [_row_value(panel, row, 5) for row in range(panel.visible_row_count)]
    assert ascending == sorted(ascending), "params column is not sorted ascending"
    assert ascending[0] <= ascending[-1]

    window.search_edit.setText("Conv2d")
    _pump(app)
    assert 0 < panel.visible_row_count < total
    ops = {panel.proxy.index(row, 1).data(Qt.ItemDataRole.DisplayRole) for row in range(panel.visible_row_count)}
    assert ops == {"Conv2d"}, f"search returned non-Conv2d rows: {ops}"

    window.search_edit.setText("features.0")
    _pump(app)
    assert panel.visible_row_count == 1
    assert panel.proxy.index(0, 0).data(Qt.ItemDataRole.DisplayRole) == "features.0"
    window.search_edit.clear()
    _pump(app)
    assert panel.visible_row_count == total

    target = tmp_path / "gui_layers.csv"
    exported = window.export_csv(target)
    assert exported == target
    raw = target.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf"), "GUI CSV export is not UTF-8-SIG"
    rows = list(csv.reader(io.StringIO(raw.decode("utf-8-sig"), newline="")))
    assert rows[0] == CSV_HEADER
    assert len(rows) == total + 1
    assert all(len(row) == len(CSV_HEADER) for row in rows)

    node = next(item for item in graph["nodes"] if item["name"] == "features.0")
    exported_row = next(row for row in rows[1:] if row[0] == "features.0")
    assert exported_row[1] == node["op"]
    assert int(exported_row[5]) == node["numParams"]
    assert int(exported_row[6]) == node["flops"]
    assert exported_row[8] == str(node["metrics"]["receptiveField"])


def test_algorithm_tab_family_patterns_pseudocode(app, window) -> None:
    window.show()
    window.tabs.setCurrentIndex(2)
    _pump(app)
    assert window.tabs.currentIndex() == 2
    panel = window.algorithm_panel
    text = panel.browser.toPlainText()
    for expected in ("ALGORITHM FAMILY", "VGG", "vgg", "Conv-BN-Act", "y = Act(BN(Conv(x)))"):
        assert expected in text, f"algorithm panel missing {expected!r}"
    assert "置信度" in text and "%" in text

    pseudocode = panel.pseudocode.toPlainText()
    assert "x = Input(1, 3, 64, 64)" in pseudocode
    assert pseudocode.count("# -> (") >= 3
    assert panel.copy_button.isEnabled()
    panel.copy_pseudocode()
    clipboard = QGuiApplication.clipboard().text()
    assert clipboard == pseudocode.strip()

    window.tabs.setCurrentIndex(3)
    _pump(app)
    assert "VGGMini" in window.summary_panel.browser.toPlainText()


# --------------------------------------------------------------------------- #
# 4. background / static parsing
# --------------------------------------------------------------------------- #
def test_background_parse_sample_and_static_flag(app, tmp_path: Path, monkeypatch) -> None:
    folder = _copy_sample_to_tmp(tmp_path)

    captured_sync: dict[str, bool] = {}
    real_sync = gui_main.parse_path_sync

    def spy_sync(path, *, static: bool = False):  # noqa: ANN001
        captured_sync["static"] = static
        return real_sync(path, static=static)

    monkeypatch.setattr(gui_main, "parse_path_sync", spy_sync)

    capture_worker: dict[str, bool] = {}
    real_worker = gui_main.ParseWorker

    class SpyWorker(real_worker):  # type: ignore[misc, valid-type]
        def __init__(self, path, *, static: bool = False, parent=None):  # noqa: ANN001
            capture_worker["static"] = static
            super().__init__(path, static=static, parent=parent)

    monkeypatch.setattr(gui_main, "ParseWorker", SpyWorker)

    # Synchronous branch: checkbox -> parse_path_sync(static=True).
    window = MainWindow(load_demo=False)
    try:
        window.static_checkbox.setChecked(True)
        assert window.open_path(folder, background=False)
        _pump(app)
        assert captured_sync.get("static") is True
        assert len(window.models) == 1
        assert window.current_graph["model"]["mode"] == "static"
        assert window.table_panel.row_count == len(window.current_graph["nodes"])
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()

    # Background branch: models list grows from empty and the worker got static=True.
    worker_window = MainWindow(load_demo=False)
    try:
        before = len(worker_window.models)
        worker_window.static_checkbox.setChecked(True)
        assert worker_window.open_path(folder, background=True)
        _wait_parse(app, worker_window)
        assert capture_worker.get("static") is True, "static checkbox was not forwarded to ParseWorker"
        assert len(worker_window.models) == 1
        assert len(worker_window.models) > before
        assert worker_window.current_graph is not None
        assert worker_window.current_graph["model"]["mode"] == "static"
        assert worker_window.table_panel.row_count == len(worker_window.current_graph["nodes"])
        assert worker_window.parse_worker is None
    finally:
        worker_window.close()
        worker_window.deleteLater()
        app.processEvents()


def test_parse_error_paths_do_not_crash(app, tmp_path: Path, monkeypatch) -> None:
    # Never open a modal dialog in a headless test.
    monkeypatch.setattr(
        QMessageBox,
        "critical",
        staticmethod(lambda *args, **kwargs: QMessageBox.StandardButton.Ok),
    )
    window = MainWindow(load_demo=False)
    try:
        missing = window.open_path(tmp_path / "missing_model.py", background=False)
        assert missing is False
        status = window.statusBar().currentMessage()
        assert "不存在" in status or "失败" in status

        bad = tmp_path / "broken_model.py"
        bad.write_text("def broken(:\n", encoding="utf-8")
        assert window.open_path(bad, background=False) is False
        assert "失败" in window.statusBar().currentMessage()

        assert window.open_path(bad, background=True) is True
        _wait_parse(app, window)
        assert window.parse_worker is None
        assert window.models == []
        assert "失败" in window.statusBar().currentMessage()
        assert window.tabs.count() >= 4
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()


# --------------------------------------------------------------------------- #
# 5. launcher / frozen executable smoke
# --------------------------------------------------------------------------- #
_SMOKE_RE = re.compile(r"OK:\s*(\d+)\s+models")


def test_gui_launcher_smoke_subprocess() -> None:
    proc = subprocess.run(
        [sys.executable, str(LAUNCHER), "--smoke"],
        cwd=ROOT,
        env=_subprocess_env(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
    )
    assert proc.returncode == 0, f"gui_launcher --smoke failed:\n{proc.stderr[-2000:]}"
    assert "ERROR" not in proc.stdout, proc.stdout
    match = _SMOKE_RE.search(proc.stdout)
    assert match, f"launcher did not report the demo model count:\n{proc.stdout}"
    assert int(match.group(1)) == 8
    assert "3D" in proc.stdout


def test_frozen_gui_executable_smoke_if_present() -> None:
    candidates = [
        ROOT / "dist" / "gui" / "NetPrismGUI" / "NetPrismGUI.exe",
        ROOT / "dist" / "gui" / "NetPrismGUI.exe",
    ]
    executable = next((path for path in candidates if path.is_file()), None)
    if executable is None:
        pytest.skip("frozen GUI executable not built (dist/gui/NetPrismGUI/NetPrismGUI.exe missing)")
    proc = subprocess.run(
        [str(executable), "--smoke"],
        cwd=ROOT,
        env=_subprocess_env(),
        capture_output=True,
        timeout=600,
    )
    assert proc.returncode == 0, f"frozen GUI --smoke failed:\n{proc.stderr.decode('utf-8', 'replace')[-2000:]}"
    stdout = proc.stdout.decode("utf-8", "replace")
    match = _SMOKE_RE.search(stdout)
    assert match, f"frozen GUI did not report the demo model count:\n{stdout}"
    assert int(match.group(1)) == 8


# --------------------------------------------------------------------------- #
# 6. window screenshot evidence
# --------------------------------------------------------------------------- #
def test_gui_screenshot_is_rendered_and_chinese(app, window) -> None:
    window.resize(1440, 900)
    window.show()
    window.tabs.setCurrentIndex(0)
    window.select_node("n1")
    _pump(app, rounds=8)

    pixmap = window.grab()
    assert not pixmap.isNull()
    image = pixmap.toImage()
    assert image.width() >= 1280 and image.height() >= 800
    assert image.width() == window.width() and image.height() == window.height()

    target = SCRATCH / "gui-qa.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    assert pixmap.save(str(target)), f"could not save screenshot to {target}"
    assert target.stat().st_size > 10_000, "screenshot is suspiciously small"

    loaded = QImage(str(target))
    assert not loaded.isNull()
    assert (loaded.width(), loaded.height()) == (image.width(), image.height())

    colors = {
        loaded.pixelColor(x, y).name()
        for x in range(0, loaded.width(), 37)
        for y in range(0, loaded.height(), 37)
    }
    assert len(colors) > 40, f"screenshot looks blank ({len(colors)} sampled colours)"

    # Chinese UI + the three main views are present in the rendered window.
    assert any(ord(char) > 127 for char in window.tabs.tabText(0))
    assert "VGGMini" in window.summary_panel.browser.toPlainText()
    assert window.table_panel.row_count > 0
    assert len(window.view_3d.scene().items()) > 0
    assert window.algorithm_panel.browser.toPlainText().strip()
