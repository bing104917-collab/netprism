#!/usr/bin/env python3
"""Headless UI smoke check for NetPrism (Playwright + system Chrome).

Start the server first (``python -m netprism serve examples/demo``), then::

    python scripts/ui_check.py --url http://127.0.0.1:8765/ --out-dir docs/screenshots

It verifies: canvas renders, model loads, hover tooltip works, click selection
works, toggles/playback respond, and no console/page errors occur. Screenshots
are written for visual review.
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
    parser.add_argument("--out-dir", default="docs/screenshots")
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=1000)
    parser.add_argument("--all-models", action="store_true", help="screenshot every model in the selector")
    parser.add_argument("--keep-open", action="store_true")
    args = parser.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

    from playwright.sync_api import sync_playwright

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    chrome = find_chrome()
    console_errors: list[str] = []
    console_warnings: list[str] = []
    page_errors: list[str] = []
    failed_requests: list[str] = []
    http_errors: list[str] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path=chrome,
            headless=True,
            args=[
                "--enable-unsafe-swiftshader",
                "--use-angle=swiftshader",
                "--disable-dev-shm-usage",
                "--hide-scrollbars",
            ],
        )
        page = browser.new_page(viewport={"width": args.width, "height": args.height})
        page.on(
            "console",
            lambda msg: console_errors.append(f"{msg.type}: {msg.text}")
            if msg.type == "error"
            else (console_warnings.append(f"{msg.type}: {msg.text}") if msg.type == "warning" else None),
        )
        page.on("pageerror", lambda exc: page_errors.append(str(exc)))
        page.on(
            "requestfailed",
            lambda req: failed_requests.append(f"{req.url} {req.failure}"),
        )
        page.on(
            "response",
            lambda resp: http_errors.append(f"{resp.status} {resp.url}")
            if resp.status >= 400 and "favicon" not in resp.url.lower()
            else None,
        )

        result: dict = {"url": args.url}
        page.goto(args.url, wait_until="networkidle", timeout=45_000)
        page.wait_for_function(
            "() => { const el = document.querySelector('#model-title');"
            " return el && el.textContent.trim() !== '' && el.textContent.trim() !== '初始化中'; }",
            timeout=30_000,
        )
        page.wait_for_selector("#viewport canvas", state="attached", timeout=20_000)
        page.wait_for_timeout(2500)

        result["title"] = page.locator("#model-title").inner_text().strip()
        result["mode"] = page.locator("#mode-badge").inner_text().strip() if page.locator("#mode-badge").count() else ""
        result["tree_count"] = page.locator("#tree-count").inner_text().strip()
        result["canvas_count"] = page.locator("#viewport canvas").count()
        result["state_overlay_hidden"] = page.locator("#state-overlay").is_hidden()
        page.screenshot(path=str(out_dir / "hero.png"))
        result["hero"] = str(out_dir / "hero.png")

        # Hover sweep: find any layer under the mouse.
        viewport = page.locator("#viewport").bounding_box()
        found = None
        if viewport:
            x0, y0 = viewport["x"], viewport["y"]
            w, h = viewport["width"], viewport["height"]
            step = 70
            for yy in range(int(y0 + h * 0.25), int(y0 + h * 0.8), step):
                for xx in range(int(x0 + w * 0.08), int(x0 + w * 0.95), step):
                    page.mouse.move(xx, yy)
                    page.wait_for_timeout(35)
                    if page.locator("#tooltip").is_visible():
                        found = (xx, yy, page.locator("#tooltip").inner_text().strip())
                        break
                if found:
                    break
        result["hover"] = found
        if found:
            xx, yy, text = found
            page.mouse.move(xx, yy)
            page.wait_for_timeout(200)
            page.screenshot(path=str(out_dir / "hover.png"))
            result["hover_screenshot"] = str(out_dir / "hover.png")
            result["tooltip"] = text[:500]
            page.mouse.click(xx, yy)
            page.wait_for_timeout(500)
            result["detail_meta"] = page.locator("#detail-meta").inner_text().strip()
            result["detail_text"] = page.locator("#detail-panel").inner_text().strip()[:500]
            page.screenshot(path=str(out_dir / "selected.png"))
            result["selected_screenshot"] = str(out_dir / "selected.png")

        # Playback + toggles.
        page.click("#play-btn")
        page.wait_for_timeout(1200)
        result["flow_hud_visible"] = page.locator("#flow-hud").is_visible()
        page.screenshot(path=str(out_dir / "playing.png"))
        result["playing_screenshot"] = str(out_dir / "playing.png")
        page.click("#play-btn")

        for toggle in ("groups", "labels", "kernels"):
            button = page.locator(f'[data-toggle="{toggle}"]')
            before = button.get_attribute("aria-pressed")
            button.click()
            page.wait_for_timeout(250)
            after = button.get_attribute("aria-pressed")
            result[f"toggle_{toggle}"] = [before, after]
            button.click()
            page.wait_for_timeout(150)

        # Model switch (if more than one).
        options = page.locator("#model-select option").all_inner_texts()
        result["models"] = options
        if len(options) > 1:
            page.select_option("#model-select", index=1)
            page.wait_for_timeout(2500)
            result["switched_title"] = page.locator("#model-title").inner_text().strip()
            page.screenshot(path=str(out_dir / "model2.png"))
            result["model2_screenshot"] = str(out_dir / "model2.png")

        if args.all_models and len(options) > 1:
            shots = {}
            for index in range(len(options)):
                page.select_option("#model-select", index=index)
                page.wait_for_timeout(2200)
                title = page.locator("#model-title").inner_text().strip()
                safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in title)[:40] or f"model{index}"
                shot = out_dir / f"model_{index:02d}_{safe}.png"
                page.screenshot(path=str(shot))
                shots[title] = str(shot)
            result["model_screenshots"] = shots

        result["console_errors"] = console_errors
        result["console_warnings"] = console_warnings
        result["page_errors"] = page_errors
        result["http_errors"] = http_errors
        result["failed_requests"] = [
            item for item in failed_requests if "favicon" not in item.lower()
        ]
        result["ok"] = (
            result["canvas_count"] > 0
            and result.get("state_overlay_hidden", False)
            and not page_errors
            and not console_errors
            and not http_errors
            and bool(found)
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if args.keep_open:
            page.wait_for_timeout(60_000)
        browser.close()
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
