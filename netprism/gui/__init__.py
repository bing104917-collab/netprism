"""Native PySide6 desktop UI for NetPrism.

The GUI intentionally talks to the parser/library API only: it never starts an
HTTP server, never opens a browser and never imports a web engine.
"""

from __future__ import annotations

__all__ = ["MainWindow", "create_application"]

from .main_window import MainWindow, create_application
