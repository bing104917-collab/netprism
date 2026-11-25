"""Frozen entry point for the NetPrism web app: click -> local server -> browser.

Built by ``scripts/build_web_exe.ps1`` into a single ``NetPrism.exe``.  The exe
bundles the current ``web/`` frontend, the pre-parsed ``examples/demo`` data and
the static AST parser, so a double click lands directly on the 3D viewer.

The lean build deliberately leaves torch / onnx / PySide6 out:

* the web upload path is static-AST by design (uploaded code is never executed),
  so the browser experience is complete without the ML stack;
* dropping it keeps the single file around 30 MB and start-up around 2 seconds
  instead of 180 MB / ten seconds.

Dragging a model folder (or a single ``.py``) onto the exe rescans it with the
same static engine and serves the fresh result.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import urllib.request
from pathlib import Path

if not getattr(sys, "frozen", False):
    _project_root = Path(__file__).resolve().parents[1]
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))


def bundle_root() -> Path:
    """Directory that contains the bundled ``web/``, ``examples/`` and ``schema/``."""
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
    return Path(__file__).resolve().parents[1]


def _writable_out(root: Path) -> Path:
    """Pick a writable folder for drag-and-drop scan results."""
    candidates: list[Path] = []
    override = os.environ.get("NETPRISM_OUT")
    if override:
        candidates.append(Path(override).expanduser())
    if getattr(sys, "frozen", False):
        candidates.append(Path(sys.executable).resolve().parent / "netprism_out")
    candidates.append(Path(os.environ.get("LOCALAPPDATA") or Path.home()) / "NetPrism" / "out")
    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            probe = candidate / ".netprism_write_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            return candidate
        except OSError:
            continue
    return candidates[-1]


def _open_later(url: str, delay: float = 0.6) -> None:
    time.sleep(delay)
    try:
        import webbrowser

        webbrowser.open(url)
    except Exception:  # pragma: no cover - best effort
        pass


def _banner(hint: str) -> None:
    print("=" * 66)
    print("  NetPrism · 3D 神经网络透视（Web 版）")
    print("=" * 66)
    print(hint)
    print("  · 浏览器会自动打开；关掉这个窗口即停止服务。")
    print("  · 看自己的模型：把模型文件夹拖到 NetPrism.exe 上，")
    print("    或在网页里点「上传文件夹」——静态 AST 解析，不执行模型代码。")
    print("=" * 66)


def _smoke(web_dir: Path, data_dir: Path) -> int:
    """Start the server in-process, hit the APIs, print OK and exit."""
    from netprism.server import create_app, create_server

    app = create_app(data_dir, web_dir)
    httpd = create_server(app, "127.0.0.1", 0)
    port = int(httpd.server_address[1])
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{port}"
        with urllib.request.urlopen(base + "/api/health", timeout=20) as response:
            health = json.loads(response.read().decode("utf-8"))
        with urllib.request.urlopen(base + "/api/index", timeout=20) as response:
            index = json.loads(response.read().decode("utf-8"))
        with urllib.request.urlopen(base + "/", timeout=20) as response:
            page = response.read().decode("utf-8", "replace")
        models = list(index.get("models") or [])
        layers = sum(int(model.get("numLayers") or 0) for model in models)
        if not health.get("ok") or not models or "<title>" not in page:
            print("FAIL: unexpected smoke payload", file=sys.stderr)
            return 1
        print(f"OK: {len(models)} models, {layers} layers, page {len(page)} bytes")
        return 0
    finally:
        httpd.shutdown()
        httpd.server_close()


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

    root = bundle_root()
    demo_dir = root / "examples" / "demo"
    web_dir = root / "web"
    try:
        default_port = int(os.environ.get("NETPRISM_PORT") or "0")
    except ValueError:
        default_port = 0

    parser = argparse.ArgumentParser(
        prog="NetPrism",
        description="NetPrism Web 版：启动本地服务并在浏览器中打开 3D 视图",
    )
    parser.add_argument("target", nargs="?", default=None,
                        help="模型文件夹或单个 .py（把文件夹拖到 exe 上即可）")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=default_port,
                        help="默认 0 = 自动挑选空闲端口")
    parser.add_argument("--no-open", action="store_true", help="不自动打开浏览器")
    parser.add_argument("--data", default=None, help="直接服务一个已 scan 的输出目录")
    parser.add_argument("--smoke", action="store_true", help="自检：起服务、请求 API 后退出")
    args = parser.parse_args(argv)

    if args.smoke:
        return _smoke(web_dir, Path(args.data).expanduser() if args.data else demo_dir)

    if args.data and args.target:
        print("[NetPrism] --data 和拖入路径只能二选一。", file=sys.stderr)
        return 2

    # Dragging a folder/file onto the exe: static rescan, then the CLI serves it.
    if args.target:
        target = Path(args.target).expanduser()
        if not target.exists():
            print(f"[NetPrism] 路径不存在：{target}", file=sys.stderr)
            return 2
        if target.is_file() and target.suffix.lower() != ".py":
            print(
                f"[NetPrism] 这个版本只解析 .py 源码，不支持 {target.suffix or '该格式'}；"
                "ONNX / TorchScript / Keras 请用源码环境或完整版。",
                file=sys.stderr,
            )
            return 2
        out = _writable_out(root)
        print(f"[NetPrism] 正在静态解析：{target}")
        print(f"[NetPrism] 解析结果目录：{out}")
        from netprism.cli import main as cli_main

        view_args = [
            "view", str(target), "--out", str(out),
            "--no-execute", "--no-formats", "--unknown-input",
            "--host", args.host, "--port", str(args.port),
        ]
        if not args.no_open:
            view_args.append("--open")
        return cli_main(view_args)

    data_dir = Path(args.data).expanduser() if args.data else demo_dir
    if not data_dir.is_dir():
        print(f"[NetPrism] 数据目录不存在：{data_dir}", file=sys.stderr)
        return 2

    from netprism.server import create_app, create_server

    _banner(f"  已载入：{data_dir}")
    app = create_app(data_dir, web_dir)
    try:
        httpd = create_server(app, args.host, args.port)
    except OSError as exc:
        print(f"[NetPrism] 无法监听 {args.host}:{args.port} — {exc}", file=sys.stderr)
        return 2
    port = int(httpd.server_address[1])
    url = f"http://{args.host}:{port}/"
    print()
    print(f"  服务已启动：{url}")
    print()
    if not args.no_open:
        threading.Thread(target=_open_later, args=(url,), daemon=True).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[NetPrism] 已停止。")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
