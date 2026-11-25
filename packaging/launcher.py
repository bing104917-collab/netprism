"""Entry point for the frozen NetPrism Windows executable.

Behavior:
* no arguments  -> start with the bundled 8-model demo and open the browser;
* a folder path  -> treat it as "drag folder onto NetPrism.exe": parse it with
                    the trace engine, serve the result and open the browser;
* a .py file     -> inspect it in the console;
* any CLI command (scan/serve/view/inspect/--version) -> pass through unchanged.

The frozen app is built by ``scripts/build_exe.ps1``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

KNOWN_COMMANDS = {"scan", "serve", "view", "inspect", "--version", "-h", "--help"}

if not getattr(sys, "frozen", False):
    _project_root = Path(__file__).resolve().parents[1]
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))


def app_root() -> Path:
    """Directory that contains the bundled ``web/`` and ``examples/`` assets."""
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
    return Path(__file__).resolve().parents[1]


def _no_browser() -> bool:
    return os.environ.get("NETPRISM_NO_BROWSER", "").strip().lower() in {"1", "true", "yes", "on"}


def _port() -> str:
    raw = os.environ.get("NETPRISM_PORT", "0").strip()
    return raw if raw.isdigit() else "0"


def _writable_output_dir(root: Path) -> Path:
    """Choose a writable directory for drag-and-drop scan results.

    Prefer ``NETPRISM_OUT``, then a folder next to the executable (portable
    installs), then ``%LOCALAPPDATA%\\NetPrism\\out``.  Probing avoids a
    cryptic PermissionError on locked-down machines.
    """
    candidates: list[Path] = []
    override = os.environ.get("NETPRISM_OUT")
    if override:
        candidates.append(Path(override).expanduser())
    if getattr(sys, "frozen", False):
        candidates.append(Path(sys.executable).resolve().parent / "netprism_out")
    else:
        candidates.append(root / ".netprism_out")
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


def _banner() -> None:
    print("=" * 68)
    print("  NetPrism 棱镜 · 3D 神经网络透视")
    print("=" * 68)
    print("  演示数据已载入，浏览器即将自动打开。")
    print("  · 想解析自己的模型：把模型文件夹（或 .py/.onnx/.pt/.keras）拖到 NetPrism.exe 上")
    print("  · 或在网页里点“上传文件夹”（安全静态解析，不执行代码）")
    print("  · 关闭本窗口即可停止服务")
    print("=" * 68)


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

    root = app_root()
    argv = sys.argv[1:]
    browser_args = [] if _no_browser() else ["--open"]

    if argv and argv[0] not in KNOWN_COMMANDS:
        target = Path(argv[0]).expanduser()
        if target.is_dir() or (target.is_file() and target.suffix.lower() in {".py", ".onnx", ".pt", ".pth", ".keras"}):
            out = _writable_output_dir(root)
            from netprism.cli import main as cli_main

            return cli_main(
                [
                    "view",
                    str(target),
                    "--out",
                    str(out),
                    "--port",
                    _port(),
                    *browser_args,
                    *argv[1:],
                ]
            )
        if target.is_file() and target.suffix.lower() == ".py":
            from netprism.cli import main as cli_main

            return cli_main(["inspect", str(target), *argv[1:]])

    from netprism.cli import main as cli_main

    if not argv:
        _banner()
        return cli_main(
            ["serve", str(root / "examples" / "demo"), "--port", _port(), *browser_args]
        )
    return cli_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
