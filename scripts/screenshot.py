#!/usr/bin/env python3
"""Take a headless-Chrome screenshot of the NetPrism UI.

The server must already be running. Example::

    python scripts/screenshot.py --url http://127.0.0.1:8765/ --out docs/screenshots/hero.png
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]


def find_chrome() -> str:
    env = os.environ.get("NETPRISM_CHROME")
    if env and Path(env).exists():
        return env
    for candidate in CHROME_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    raise SystemExit("找不到 Chrome/Edge，可用 NETPRISM_CHROME 指定路径。")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8765/")
    parser.add_argument("--out", default="docs/screenshots/hero.png")
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=1000)
    parser.add_argument("--wait-ms", type=int, default=12000)
    args = parser.parse_args()

    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    chrome = find_chrome()
    profile = Path(os.environ.get("TEMP", ".")) / "netprism-chrome-profile"
    cmd = [
        chrome,
        "--headless=new",
        "--disable-gpu",
        "--enable-unsafe-swiftshader",
        "--use-angle=swiftshader",
        "--hide-scrollbars",
        "--no-first-run",
        "--no-default-browser-check",
        f"--user-data-dir={profile}",
        f"--window-size={args.width},{args.height}",
        f"--virtual-time-budget={args.wait_ms}",
        "--run-all-compositor-stages-before-draw",
        f"--screenshot={out}",
        args.url,
    ]
    print(" ".join(cmd))
    completed = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if completed.returncode != 0:
        sys.stderr.write(completed.stderr[-4000:])
        return completed.returncode
    if not out.exists():
        print("screenshot was not created", file=sys.stderr)
        return 1
    print(f"screenshot: {out} ({out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
