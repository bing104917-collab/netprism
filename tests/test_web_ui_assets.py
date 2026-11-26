"""Static guards for the Web UI's layer naming and image export.

The browser behaviour itself is covered by ``tests/test_v2_ui.py`` (Playwright);
these checks keep the two features from silently disappearing from the bundle
that the EXE serves.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"


def test_layer_tree_uses_operator_names() -> None:
    format_js = (WEB / "js" / "format.js").read_text(encoding="utf-8")
    assert "export function opName(" in format_js
    assert "export function opParams(" in format_js
    ui_js = (WEB / "js" / "ui.js").read_text(encoding="utf-8")
    assert "opName(node)" in ui_js
    assert "<span class=\"tn-name\">${escapeHtml(node.name || item.id)}</span>" not in ui_js
    nodes_js = (WEB / "js" / "nodes.js").read_text(encoding="utf-8")
    assert "opName(item.node)" in nodes_js


def test_web_toolbar_has_image_export() -> None:
    index = (WEB / "index.html").read_text(encoding="utf-8")
    assert 'id="export-btn"' in index
    assert "导出图片" in index
    scene = (WEB / "js" / "scene.js").read_text(encoding="utf-8")
    assert "exportImage(width = 1920)" in scene
    assert "preserveDrawingBuffer: true" in scene
    assert "_compositeLabels(" in scene
    main_js = (WEB / "js" / "main.js").read_text(encoding="utf-8")
    assert "exportCurrentImage" in main_js
