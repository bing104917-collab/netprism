"""Offscreen screenshot helper for the native GUI (docs/screenshots/gui.png)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

if not getattr(sys, "frozen", False):
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from .main_window import MainWindow, create_application  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Capture the native NetPrism GUI offscreen")
    parser.add_argument("out", nargs="?", default=None, help="output PNG")
    parser.add_argument("--tab", choices=["3d", "table", "algo", "summary"], default="3d")
    parser.add_argument("--node", default="n1")
    parser.add_argument("--width", type=int, default=1440)
    parser.add_argument("--height", type=int, default=900)
    args = parser.parse_args(argv)

    target = Path(args.out) if args.out else Path(__file__).resolve().parents[2] / "docs" / "screenshots" / "gui.png"
    app = create_application([])
    window = MainWindow()
    window.resize(args.width, args.height)
    window.show()
    app.processEvents()
    window.tabs.setCurrentIndex({"3d": 0, "table": 1, "algo": 2, "summary": 3}[args.tab])
    window.select_node(args.node)
    for _ in range(8):
        app.processEvents()
    target.parent.mkdir(parents=True, exist_ok=True)
    ok = window.grab().save(str(target))
    print(f"GUI screenshot: {target} ({'OK' if ok else 'FAILED'})")
    window.close()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
