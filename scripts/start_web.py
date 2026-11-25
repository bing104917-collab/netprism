#!/usr/bin/env python3
"""One-click launcher for the NetPrism Web mode (local server + browser).

The user-facing entry point is ``start_web.cmd`` in the repository root; this
script is what that window actually runs.  It exists so the URL, the port
fallback and the "close this window to stop" hints are computed in one place.

Usage::

    python scripts/start_web.py                 # serve examples/demo, open browser
    python scripts/start_web.py .\\models        # serve a scan output folder
    python scripts/start_web.py --no-open       # start the server only
"""

from __future__ import annotations

import argparse
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEMO_DIR = ROOT / "examples" / "demo"
DEFAULT_PORT = 8765
PORT_TRIES = 20


def _port_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, port))
        except OSError:
            return False
        return True


def _pick_port(host: str, preferred: int) -> int:
    if _port_free(host, preferred):
        return preferred
    for port in range(preferred + 1, preferred + PORT_TRIES):
        if _port_free(host, port):
            return port
    return 0  # let the OS assign a free port


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="启动 NetPrism Web 服务并在浏览器中打开 3D 视图",
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default=None,
        help="scan 输出目录（默认 examples/demo）",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--no-open", action="store_true", help="只启动服务，不自动打开浏览器")
    args = parser.parse_args(argv)

    data_dir = Path(args.directory).expanduser().resolve() if args.directory else DEMO_DIR
    if not data_dir.exists():
        print(f"[NetPrism] 数据目录不存在：{data_dir}", file=sys.stderr)
        print(
            "[NetPrism] 先解析模型文件夹：python -m netprism scan <模型文件夹> --out <输出目录>",
            file=sys.stderr,
        )
        return 2

    from netprism.server import create_app, serve

    port = _pick_port(args.host, args.port)
    if port != args.port:
        print(f"[NetPrism] 端口 {args.port} 被占用，改用 {port or '系统分配端口'}。")

    app = create_app(data_dir)
    app.build_index()
    print()
    print("=" * 64)
    print("  NetPrism · Web 模式")
    print(f"  数据目录：{data_dir}")
    print("  浏览器会自动打开；没弹出就用下面的地址。")
    print("  关闭这个窗口（或按 Ctrl+C）即停止服务。")
    print("=" * 64)
    print()
    try:
        serve(app, host=args.host, port=port, open_browser=not args.no_open)
    except OSError as exc:
        print(f"[NetPrism] 无法监听 {args.host}:{port} — {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
