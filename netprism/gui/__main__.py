"""``python -m netprism.gui`` convenience entry point."""

from __future__ import annotations

import sys

from .main_window import MainWindow, create_application


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    app = create_application([sys.argv[0]])
    window = MainWindow(load_demo=True)
    window.show()
    if args:
        from PySide6.QtCore import QTimer

        from pathlib import Path

        target = Path(args[0]).expanduser()
        if target.exists():
            QTimer.singleShot(0, lambda: window.open_path(target, background=True))
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
