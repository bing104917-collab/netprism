"""Native GUI launcher for the frozen NetPrism desktop executable.

Behavior:
* no arguments -> open the window with the bundled eight-model demo;
* a folder / model file -> open the window and parse that source;
* ``--smoke`` -> construct the window offscreen, load the demo, print OK, exit.

The launcher never starts an HTTP server and never imports webbrowser /
QWebEngine.  The old CLI entry point remains available separately.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

KNOWN_MODEL_SUFFIXES = {".py", ".pt", ".pth", ".onnx", ".keras", ".json"}


def _safe_print(text: str) -> None:
    stream = getattr(sys, "stdout", None)
    if stream is None:
        return
    try:
        stream.write(text + "\n")
        stream.flush()
    except Exception:
        pass


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    smoke = "--smoke" in args
    args = [arg for arg in args if arg != "--smoke"]

    if not getattr(sys, "frozen", False):
        root = Path(__file__).resolve().parents[1]
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        os.environ.setdefault("NETPRISM_DEMO_DIR", str(root / "examples" / "demo"))

    if smoke:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    try:
        from netprism.gui.main_window import MainWindow, create_application
    except Exception as exc:  # pragma: no cover - import/deployment failure
        _safe_print(f"ERROR: cannot import NetPrism GUI: {exc}")
        return 1

    app = create_application([sys.argv[0]])
    window = MainWindow(load_demo=True)
    if not window.models:
        _safe_print("ERROR: built-in demo failed to load")
        return 1
    if smoke:
        app.processEvents()
        window.tabs.setCurrentIndex(0)
        if window.view_3d.scene() is None or not window.view_3d.scene().items():
            _safe_print("ERROR: 3D scene is empty")
            return 1
        _safe_print(
            f"OK: {len(window.models)} models, {len(window.view_3d.scene().items())} 3D items"
        )
        window.close()
        return 0

    window.show()
    if args:
        target = Path(args[0]).expanduser()
        if target.exists() and (target.is_dir() or target.suffix.lower() in KNOWN_MODEL_SUFFIXES):
            from PySide6.QtCore import QTimer

            QTimer.singleShot(0, lambda: window.open_path(target, background=True))
        else:
            _safe_print(f"提示：路径不是可解析的模型：{target}")
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
