#!/usr/bin/env python3
"""Verify the browser folder-upload flow (static parse) with Playwright.

Usage::

    python scripts/upload_check.py --url http://127.0.0.1:8767/ --folder samples
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
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
    parser.add_argument("--url", default="http://127.0.0.1:8767/")
    parser.add_argument("--folder", default="samples")
    parser.add_argument("--out", default="scratch/lead/upload.png")
    args = parser.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

    from playwright.sync_api import sync_playwright

    folder = Path(args.folder).resolve()
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    page_errors: list[str] = []
    requests: list[str] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path=find_chrome(),
            headless=True,
            args=["--enable-unsafe-swiftshader", "--use-angle=swiftshader", "--hide-scrollbars"],
        )
        page = browser.new_page(viewport={"width": 1600, "height": 1000})
        page.on(
            "console",
            lambda msg: errors.append(f"{msg.type}: {msg.text}") if msg.type == "error" else None,
        )
        page.on("pageerror", lambda exc: page_errors.append(str(exc)))
        page.on("request", lambda req: requests.append(req.url) if "upload-parse" in req.url else None)
        page.goto(args.url, wait_until="networkidle", timeout=45_000)
        page.wait_for_function(
            "() => document.querySelectorAll('#model-select option').length >= 8",
            timeout=30_000,
        )
        before = page.locator("#model-select option").count()
        page.set_input_files("#file-input", str(folder))
        # Upload switches the source to the uploaded models (static). The count
        # may stay the same if the folder contains as many models as the demo,
        # so wait for the mode badge to flip to STATIC instead.
        page.wait_for_function(
            "() => { const el = document.querySelector('#mode-badge');"
            " return el && /STATIC/i.test(el.textContent || ''); }",
            timeout=120_000,
        )
        page.wait_for_timeout(1500)
        options = page.locator("#model-select option").all_inner_texts()
        result = {
            "before": before,
            "after": len(options),
            "upload_requests": requests,
            "options": options,
            "console_errors": errors,
            "page_errors": page_errors,
        }
        page.select_option("#model-select", index=0)
        page.wait_for_timeout(2000)
        result["uploaded_title"] = page.locator("#model-title").inner_text().strip()
        result["uploaded_mode"] = page.locator("#mode-badge").inner_text().strip()
        page.screenshot(path=str(out))
        result["screenshot"] = str(out)
        result["ok"] = (
            len(options) > 0
            and any("STATIC" in option.upper() for option in options)
            and bool(requests)
            and not errors
            and not page_errors
            and result["uploaded_mode"].lower().startswith("static")
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        browser.close()
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
