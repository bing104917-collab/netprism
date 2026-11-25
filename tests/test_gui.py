"""Offscreen tests for the native PySide6 NetPrism GUI."""

from __future__ import annotations

import copy
import csv
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QPoint, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QGuiApplication, QImage, QWheelEvent  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from netprism.gui.graph_loader import ParseWorker, parse_path_sync, project_root  # noqa: E402
from netprism.gui.main_window import MainWindow, create_application  # noqa: E402
from netprism.gui.panels import AlgorithmPanel  # noqa: E402
from netprism.gui.scene3d import DEFAULT_PITCH, DEFAULT_YAW, Graph3DView  # noqa: E402


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


def _process(app, rounds: int = 3) -> None:
    for _ in range(rounds):
        app.processEvents()


def test_demo_models_3d_and_detail(app, window):
    assert len(window.models) == 8
    assert window.model_list.count() == 8
    assert window.current_graph is not None
    assert len(window.view_3d.scene().items()) > 0
    assert window.table_panel.row_count == len(window.current_graph["nodes"])

    window.show()
    _process(app)
    window.select_node("n1")
    _process(app)
    detail = window.detail_browser.toPlainText()
    assert "features.0" in detail
    assert "Conv2d" in detail
    assert "算法解释" in detail
    assert "所属模式" in detail
    assert window.current_node_id == "n1"
    assert len(window.view_3d.scene().items()) > 0


def test_eight_models_table_rows(app, window):
    assert len(window.models) == 8
    for row in range(window.model_list.count()):
        window.model_list.setCurrentRow(row)
        _process(app)
        graph = window.current_graph
        assert graph is not None
        assert window.table_panel.row_count == len(graph["nodes"])
        assert len(window.view_3d.scene().items()) > 0
        assert window.algorithm_panel.browser.toPlainText().strip()
    window.model_list.setCurrentRow(0)
    _process(app)


def test_table_sort_search_and_csv(app, window, tmp_path):
    panel = window.table_panel
    total = panel.row_count
    assert total == len(window.current_graph["nodes"])

    panel.table_view.sortByColumn(5, Qt.SortOrder.DescendingOrder)
    _process(app)
    first = int(panel.proxy.index(0, 5).data(Qt.ItemDataRole.DisplayRole).replace(",", ""))
    last = int(panel.proxy.index(panel.proxy.rowCount() - 1, 5).data(Qt.ItemDataRole.DisplayRole).replace(",", ""))
    assert first >= last

    window.search_edit.setText("conv")
    _process(app)
    assert 0 < panel.visible_row_count < total
    window.search_edit.clear()
    _process(app)
    assert panel.visible_row_count == total

    target = tmp_path / "layers.csv"
    exported = window.export_csv(target)
    assert exported == target
    raw = target.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")
    text = raw.decode("utf-8-sig")
    lines = text.splitlines()
    assert lines[0].startswith("name,op,kind,in_shapes,out_shapes,params,flops,activation_mb")
    assert len(lines) == total + 1
    assert any("Conv2d" in line for line in lines[1:])


def test_open_static_file_and_1280_table(app, window):
    source = project_root() / "samples" / "mlp.py"
    window.static_checkbox.setChecked(True)
    assert window.open_path(source, background=False)
    assert window.models
    assert window.current_graph is not None
    assert window.current_graph["model"].get("mode") == "static"
    assert window.table_panel.row_count == len(window.current_graph["nodes"])

    window.resize(1280, 800)
    window.show()
    window.tabs.setCurrentIndex(1)
    _process(app)
    assert window.window().width() >= 1280
    assert window.table_panel.table_view.horizontalScrollBar().maximum() == 0


def test_3d_drag_rotates_and_double_click_resets(app, window):
    window.show()
    window.tabs.setCurrentIndex(0)
    _process(app)
    view = window.view_3d
    center = view.viewport().rect().center()
    before = view._yaw
    QTest.mousePress(view.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, center)
    QTest.mouseMove(view.viewport(), center + QPoint(70, 18), 20)
    QTest.mouseRelease(view.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, center + QPoint(70, 18))
    _process(app)
    assert view._yaw != before
    view._yaw = 0.2
    view._pitch = 0.1
    QTest.mouseDClick(view.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, center)
    _process(app)
    assert abs(view._yaw - DEFAULT_YAW) < 1e-6
    assert abs(view._pitch - DEFAULT_PITCH) < 1e-6


def test_3d_drag_repaints_live_and_keeps_model_in_frame(app):
    """A real drag must repaint continuously and never fling the model out.

    Regression for the 15:39 build: a single-shot timer was re-armed on every
    mouse-move, so a steady drag never repainted, and the model then jumped to a
    wildly different pose on release (users reported "甩动/突然放大/跑出视口").
    """
    from netprism.gui.graph_loader import load_demo_payload

    graph = load_demo_payload()["graphs"]["cnn_vgg__vggmini"]
    view = Graph3DView()
    view.resize(1100, 800)
    view.show()
    view.set_graph(graph)
    _process(app)

    refreshes = {"n": 0}
    original = Graph3DView._refresh_projection

    def counting(self):
        refreshes["n"] += 1
        return original(self)

    Graph3DView._refresh_projection = counting
    try:
        viewport = view.viewport()
        start = QPoint(viewport.width() // 2, viewport.height() // 2)
        QTest.mousePress(viewport, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, start)
        snapshots = []
        for step in range(1, 41):
            QTest.mouseMove(viewport, start + QPoint(10 * step, step // 4))
            QTest.qWait(5)
            if step in (5, 10, 20, 30, 40):
                snapshots.append(
                    (
                        view._yaw,
                        [item._faces[0][0][0].x() for item in view._node_items.values()][:6],
                    )
                )
        QTest.mouseRelease(
            viewport, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
            start + QPoint(400, 10),
        )
        _process(app)
        assert refreshes["n"] >= 8, f"drag repainted only {refreshes['n']} times"
        # The picture really changed while the button was still down.
        assert len({round(yaw, 6) for yaw, _points in snapshots}) == len(snapshots)
        assert len({tuple(round(x, 3) for x in points) for _yaw, points in snapshots}) >= 3

        # And the model stayed inside the viewport the whole time.
        rect = view.scene().itemsBoundingRect()
        top_left = view.mapFromScene(rect.topLeft())
        bottom_right = view.mapFromScene(rect.bottomRight())
        assert top_left.x() >= -2 and top_left.y() >= -2
        assert bottom_right.x() <= viewport.width() + 2
        assert bottom_right.y() <= viewport.height() + 2

        # The model centre is pinned to the viewport centre.
        origin = view.mapFromScene(QPointF(0.0, 0.0))
        assert abs(origin.x() - viewport.width() / 2) <= 2
        assert abs(origin.y() - viewport.height() / 2) <= 2
    finally:
        Graph3DView._refresh_projection = original
        view.close()
        view.deleteLater()


def test_3d_rotation_never_zooms_in_past_default(app):
    """Orbiting may zoom out to stay framed, but never past the loaded size."""
    from netprism.gui.graph_loader import load_demo_payload

    graph = load_demo_payload()["graphs"]["cnn_vgg__vggmini"]
    view = Graph3DView()
    view.resize(1100, 800)
    view.show()
    view.set_graph(graph)
    _process(app)
    base_scale = view.transform().m11()
    viewport = view.viewport()
    for yaw in (DEFAULT_YAW, 0.0, 0.6, 1.5708, 2.4, 3.6, -2.0):
        view._yaw = yaw
        view._refresh_projection()
        _process(app)
        scale = view.transform().m11()
        assert scale <= base_scale + 1e-6, f"yaw {yaw} zoomed in: {scale} > {base_scale}"
        rect = view.scene().itemsBoundingRect()
        top_left = view.mapFromScene(rect.topLeft())
        bottom_right = view.mapFromScene(rect.bottomRight())
        assert top_left.x() >= -2 and top_left.y() >= -2, f"yaw {yaw} clipped at top-left"
        assert bottom_right.x() <= viewport.width() + 2, f"yaw {yaw} clipped at right"
        assert bottom_right.y() <= viewport.height() + 2, f"yaw {yaw} clipped at bottom"
    view.close()
    view.deleteLater()


def test_3d_wheel_zoom_is_smooth_and_reversible(app):
    from netprism.gui.graph_loader import load_demo_payload

    graph = load_demo_payload()["graphs"]["cnn_vgg__vggmini"]
    view = Graph3DView()
    view.resize(1100, 800)
    view.show()
    view.set_graph(graph)
    _process(app)
    viewport = view.viewport()

    def wheel(delta: int) -> None:
        pos = QPoint(320, 240)
        event = QWheelEvent(
            QPointF(pos), QPointF(view.mapToGlobal(pos)), QPoint(0, 0), QPoint(0, delta),
            Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
            Qt.ScrollPhase.NoScrollPhase, False,
        )
        QApplication.sendEvent(viewport, event)
        _process(app)

    before = view.transform().m11()
    wheel(120)
    once = view.transform().m11()
    assert once > before
    assert 1.05 < once / before < 1.25, once / before
    origin = view.mapFromScene(QPointF(0.0, 0.0))
    assert abs(origin.x() - viewport.width() / 2) <= 2
    assert abs(origin.y() - viewport.height() / 2) <= 2
    wheel(-120)
    assert abs(view.transform().m11() - before) < 1e-6
    view.close()
    view.deleteLater()


def test_open_path_background_worker(app, window):
    source = project_root() / "samples" / "mlp.py"
    window.static_checkbox.setChecked(True)
    assert window.open_path(source, background=True)
    deadline = time.monotonic() + 30
    while window.parse_worker is not None and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    _process(app)
    assert window.parse_worker is None
    assert window.current_graph is not None
    assert window.current_graph["model"].get("mode") == "static"
    assert window.table_panel.row_count == len(window.current_graph["nodes"])


def test_3d_mouse_click_selects_layer(app, window):
    window.show()
    window.tabs.setCurrentIndex(0)
    _process(app)
    item = list(window.view_3d._node_items.values())[1]
    scene_point = item.boundingRect().center()
    view_point = window.view_3d.mapFromScene(scene_point)
    QTest.mouseClick(
        window.view_3d.viewport(),
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
        view_point,
    )
    _process(app)
    assert window.current_node_id == item.node_id
    assert item.node.get("name") in window.detail_browser.toPlainText()


def test_algorithm_and_report(app, window, tmp_path):
    text = window.algorithm_panel.browser.toPlainText()
    assert "VGG" in text
    assert "Conv-BN-Act" in text
    assert "Forward" in text or "Conv2d" in window.algorithm_panel.pseudocode.toPlainText()
    assert window.algorithm_panel.copy_button.isEnabled()
    window.algorithm_panel.copy_pseudocode()
    assert "Conv2d" in QGuiApplication.clipboard().text()
    assert "VGGMini" in window.summary_panel.browser.toPlainText()

    report = window.export_report(tmp_path / "report.md")
    assert report is not None and report.exists()
    content = report.read_text(encoding="utf-8")
    assert "分析报告" in content
    assert "算法理解" in content

    graph = copy.deepcopy(window.current_graph)
    graph["model"].pop("meta", None)
    for node in graph.get("nodes") or []:
        node.pop("meta", None)
    for group in graph.get("groups") or []:
        group.pop("meta", None)
    panel = AlgorithmPanel()
    panel.set_graph(graph)
    assert "暂无算法信息" in panel.browser.toPlainText()
    assert not panel.copy_button.isEnabled()


def test_3d_hover_tooltip_semantic():
    from netprism.gui.theme import node_tooltip
    from netprism.gui.graph_loader import load_demo_payload

    graph = load_demo_payload()["graphs"]["cnn_vgg__vggmini"]
    node = graph["nodes"][1]
    tooltip = node_tooltip(node)
    assert node["meta"]["semantic"] in tooltip
    assert node["meta"]["formula"] not in tooltip


def test_400_nodes_stay_usable(app):
    view = Graph3DView()
    nodes = []
    edges = []
    for index in range(450):
        node_id = f"n{index}"
        nodes.append(
            {
                "id": node_id,
                "name": f"layer_{index}",
                "op": "Linear",
                "kind": "linear",
                "inShapes": [[1, 32]],
                "outShapes": [[1, 32]],
                "params": {"inFeatures": 32, "outFeatures": 32},
                "numParams": 1024,
                "flops": 2048,
                "tags": [],
                "groupId": None,
                "source": None,
                "notes": [],
            }
        )
        if index:
            edges.append({"id": f"e{index}", "source": f"n{index - 1}", "target": node_id, "kind": "data", "shapes": [[1, 32]]})
    graph = {
        "model": {"name": "BigSynthetic", "numLayers": 450, "mode": "static", "framework": "pytorch"},
        "nodes": nodes,
        "edges": edges,
        "groups": [{"id": "g0", "name": "All", "label": "All", "kind": "module", "parentId": None, "nodeIds": [node["id"] for node in nodes], "childGroupIds": []}],
    }
    started = time.monotonic()
    view.set_graph(graph)
    _process(app)
    elapsed = time.monotonic() - started
    assert len(view._node_items) == 450
    assert len(view.scene().items()) > 450
    assert elapsed < 15.0
    view.deleteLater()


def test_parse_static_worker(app):
    source = project_root() / "samples" / "mlp.py"
    assert source.exists()
    worker = ParseWorker(source, static=True)
    results: list[dict] = []
    errors: list[str] = []
    worker.succeeded.connect(results.append)
    worker.failed.connect(errors.append)
    worker.start()
    deadline = time.monotonic() + 90
    while worker.isRunning() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.02)
    worker.wait(5000)
    app.processEvents()
    assert not errors, errors
    assert results
    payload = results[0]
    assert payload["models"]
    assert payload["source_path"]
    assert payload["graphs"]


def test_static_unknown_input_and_user_override(app, window):
    source = project_root() / "samples" / "cnn_vgg.py"
    window.static_checkbox.setChecked(True)
    assert window.open_path(source, background=False)
    spec = window.current_graph["model"]["inputSpec"][0]["shape"]
    assert spec[:2] == [1, 3]
    assert spec[-1] == -1
    assert "?" in window.summary_panel.browser.toPlainText()

    window.input_shape = (1, 3, 64, 64)
    assert window.open_path(source, background=False)
    spec = window.current_graph["model"]["inputSpec"][0]["shape"]
    assert spec[:3] == [1, 3, 64]
    assert spec[3] == 64


def test_channel_depth_is_visibly_relative(app):
    graph = {
        "nodes": [
            {"id": "n4", "name": "c4", "op": "Conv2d", "kind": "conv", "outShapes": [[1, 4, 16, 16]]},
            {"id": "n32", "name": "c32", "op": "Conv2d", "kind": "conv", "outShapes": [[1, 32, 16, 16]]},
            {"id": "n64", "name": "c64", "op": "Conv2d", "kind": "conv", "outShapes": [[1, 64, 16, 16]]},
            {"id": "n3", "name": "c3", "op": "Conv2d", "kind": "conv", "outShapes": [[1, 3, 16, 16]]},
        ],
        "edges": [],
        "groups": [],
    }
    view = Graph3DView()
    view.resize(900, 600)
    view.show()
    view.set_graph(graph)
    app.processEvents()
    depth = {node_id: view.positions[node_id][1][2] for node_id in ("n3", "n4", "n32", "n64")}
    assert depth["n64"] > depth["n32"] > depth["n4"] > depth["n3"]
    assert depth["n64"] - depth["n3"] > 2.0
    assert depth["n32"] - depth["n4"] > 0.7
    xs = [view.positions[node_id][0][0] for node_id in view.positions]
    assert abs(min(xs) + max(xs)) < 1e-6, "3D orbit pivot should be the model center"
    view.close()


def test_export_image_uses_presentation_angle_and_restores_view(app, window, tmp_path):
    window.show()
    window.tabs.setCurrentIndex(0)
    _process(app)
    view = window.view_3d
    before_yaw, before_pitch = view._yaw, view._pitch
    target = tmp_path / "graph.png"
    exported = window.export_image(target)
    assert exported == target
    assert target.is_file() and target.stat().st_size > 5000
    image = QImage(str(target))
    assert not image.isNull()
    assert (image.width(), image.height()) == (1920, 1080)
    # The scene rect must be cropped to the projection: exporting against the
    # live (rotation-invariant) rect would shrink the model into a small island.
    background = image.pixelColor(2, 2)
    sampled = 0
    painted = 0
    for y in range(0, image.height(), 6):
        for x in range(0, image.width(), 6):
            sampled += 1
            color = image.pixelColor(x, y)
            if abs(color.red() - background.red()) + abs(color.green() - background.green()) + abs(color.blue() - background.blue()) > 12:
                painted += 1
    assert painted / sampled > 0.10, f"export looks empty: {painted / sampled:.3f}"
    # camera angle is restored after the presentation export
    assert abs(view._yaw - before_yaw) < 1e-6
    assert abs(view._pitch - before_pitch) < 1e-6


def test_smoke_subprocess():
    root = project_root()
    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.run(
        [sys.executable, str(root / "packaging" / "gui_launcher.py"), "--smoke"],
        cwd=str(root),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert "OK:" in proc.stdout


def test_no_http_server_or_web_engine(app, window):
    assert window.models
    gui_dir = project_root() / "netprism" / "gui"
    text = "\n".join(path.read_text(encoding="utf-8") for path in gui_dir.glob("*.py"))
    assert "import webbrowser" not in text
    assert "QWebEngine" not in text
    assert "from netprism.server" not in text
    assert "import netprism.server" not in text
    assert "serve(" not in text
    launcher = (project_root() / "packaging" / "gui_launcher.py").read_text(encoding="utf-8")
    assert "import webbrowser" not in launcher
    assert "from netprism.server" not in launcher
    assert "import netprism.server" not in launcher
