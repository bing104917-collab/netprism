"""The one-click Web launcher is now the primary user entry point.

Guards the two failure modes we actually hit on Windows: a launcher that cannot
be found/started from a double click, and ``.cmd`` files with LF line endings
(cmd.exe silently mis-parses those).
"""

from __future__ import annotations

import importlib.util
import socket
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_launcher():
    path = ROOT / "scripts" / "start_web.py"
    spec = importlib.util.spec_from_file_location("netprism_start_web", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_launcher_entrypoints_exist() -> None:
    assert (ROOT / "examples" / "demo" / "index.json").is_file()
    batch = ROOT / "start_web.cmd"
    assert batch.is_file()
    assert "scripts\\start_web.py" in batch.read_text(encoding="utf-8")


def test_windows_launchers_use_crlf_without_bom() -> None:
    for name in ("start_web.cmd",):
        raw = (ROOT / name).read_bytes()
        assert b"\r\n" in raw, f"{name} must use CRLF line endings for cmd.exe"
        assert not raw.startswith(b"\xef\xbb\xbf"), f"{name} must not start with a UTF-8 BOM"


def test_launcher_picks_a_free_port() -> None:
    launcher = _load_launcher()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        occupied = int(sock.getsockname()[1])
        assert launcher._port_free("127.0.0.1", occupied) is False
        picked = launcher._pick_port("127.0.0.1", occupied)
        assert picked != occupied
        if picked:
            assert launcher._port_free("127.0.0.1", picked)


def test_launcher_is_web_only() -> None:
    launcher = _load_launcher()
    assert launcher.DEMO_DIR.is_dir()
    source = (ROOT / "scripts" / "start_web.py").read_text(encoding="utf-8")
    assert "netprism.gui" not in source
    assert "webbrowser" not in source  # server.serve() owns opening the browser
