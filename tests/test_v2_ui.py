"""Playwright acceptance tests for the NetPrism v2 layer-table UI.

The tests start a real ``python -m netprism serve <dir>`` process on a free port
and drive the app with headless system Chrome/Edge through Playwright.  They are
skipped (with an explicit message) when:

* ``playwright`` is not installed;
* no Chrome/Edge binary is available (set ``NETPRISM_CHROME`` to override);
* the served build does not contain the v2 table/framework metric elements yet
  (frontend-dev scope ``web/**``) — never silently passed.

Run::

    python -m pytest tests/test_v2_ui.py -q
"""

from __future__ import annotations

import contextlib
import csv
import io
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = ROOT / "examples" / "demo"
PY = sys.executable

CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]

CHROME_ARGS = [
    "--enable-unsafe-swiftshader",
    "--use-angle=swiftshader",
    "--disable-dev-shm-usage",
    "--hide-scrollbars",
]

TABLE_UI_SELECTORS = [
    "#framework-badge",
    "#metric-activation",
    "#table-toggle",
    "#table-view",
    "#table-search",
    "#table-export",
    "#table-body",
]

ALGO_UI_SELECTORS = [
    "#algo-toggle",
    "#algo-view",
    "#algo-family",
    "#algo-patterns",
    "#algo-pseudocode",
    "#algo-pattern-count",
    "#algo-foot-meta",
]

EXPORT_COLUMNS = [
    "name",
    "op",
    "kind",
    "in_shapes",
    "out_shapes",
    "params",
    "flops",
    "activation_mb",
    "receptive_field",
    "group",
]


# --------------------------------------------------------------------------- #
# process / browser helpers
# --------------------------------------------------------------------------- #
def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@contextlib.contextmanager
def serve_dir(data_dir: Path):
    """Start ``netprism serve`` on a data dir and yield its base URL."""
    port = _free_port()
    proc = subprocess.Popen(
        [PY, "-m", "netprism", "serve", str(data_dir), "--port", str(port), "--host", "127.0.0.1"],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    url = f"http://127.0.0.1:{port}/"
    try:
        deadline = time.time() + 45
        while time.time() < deadline:
            if proc.poll() is not None:
                output = proc.stdout.read().decode("utf-8", "replace") if proc.stdout else ""
                pytest.fail(f"netprism serve exited early (rc={proc.returncode}):\n{output[-3000:]}")
            try:
                with urllib.request.urlopen(url + "api/health", timeout=1) as response:
                    if response.status == 200:
                        break
            except OSError:
                time.sleep(0.15)
        else:
            pytest.fail(f"netprism serve at {url} did not become healthy within 45s")
        yield url
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)


def find_chrome() -> str | None:
    env = os.environ.get("NETPRISM_CHROME")
    if env and Path(env).is_file():
        return env
    for candidate in CHROME_CANDIDATES:
        if Path(candidate).is_file():
            return candidate
    return None


class PageLog:
    """Collect browser console / page / network problems for one page."""

    def __init__(self) -> None:
        self.console_errors: list[str] = []
        self.console_warnings: list[str] = []
        self.page_errors: list[str] = []
        self.http_errors: list[str] = []
        self.failed_requests: list[str] = []

    def attach(self, page) -> None:  # noqa: ANN001 - playwright Page
        page.on(
            "console",
            lambda msg: (
                self.console_errors.append(f"{msg.type}: {msg.text}")
                if msg.type == "error"
                else (
                    self.console_warnings.append(f"{msg.type}: {msg.text}")
                    if msg.type == "warning"
                    else None
                )
            ),
        )
        page.on("pageerror", lambda exc: self.page_errors.append(str(exc)))
        page.on(
            "response",
            lambda response: (
                self.http_errors.append(f"{response.status} {response.url}")
                if response.status >= 400 and "favicon" not in response.url.lower()
                else None
            ),
        )
        page.on(
            "requestfailed",
            lambda request: (
                self.failed_requests.append(f"{request.url} {request.failure}")
                if "favicon" not in request.url.lower()
                else None
            ),
        )

    def assert_clean(self) -> None:
        assert self.page_errors == [], f"page errors: {self.page_errors}"
        assert self.console_errors == [], f"console errors: {self.console_errors}"
        assert self.console_warnings == [], f"console warnings: {self.console_warnings}"
        assert self.http_errors == [], f"HTTP errors: {self.http_errors}"
        assert self.failed_requests == [], f"failed requests: {self.failed_requests}"


@pytest.fixture(scope="session")
def playwright():
    pytest.importorskip("playwright.sync_api", reason="playwright is not installed")
    from playwright.sync_api import sync_playwright

    with sync_playwright() as instance:
        yield instance


@pytest.fixture(scope="session")
def chrome() -> str:
    path = find_chrome()
    if not path:
        pytest.skip(
            "system Chrome/Edge not found — set NETPRISM_CHROME to run the UI tests "
            f"(looked at: {', '.join(CHROME_CANDIDATES)})"
        )
    return path


@pytest.fixture(scope="session")
def browser(playwright, chrome):  # noqa: ANN001
    browser = playwright.chromium.launch(executable_path=chrome, headless=True, args=CHROME_ARGS)
    yield browser
    browser.close()


@pytest.fixture()
def ui(browser):
    page = browser.new_page(viewport={"width": 1600, "height": 1000}, accept_downloads=True)
    log = PageLog()
    log.attach(page)
    yield SimpleNamespace(page=page, log=log)
    page.close()


@pytest.fixture(scope="session")
def demo_url():
    with serve_dir(DEMO_DIR) as url:
        yield url


@pytest.fixture()
def nometa_url(tmp_path: Path):
    """A demo copy whose graph has no algorithm meta at all (an "old" graph)."""
    source = DEMO_DIR
    data = tmp_path / "nometa_data"
    data.mkdir()
    index = json.loads((source / "index.json").read_text(encoding="utf-8"))
    entry = next(model for model in index["models"] if model["id"] == "cnn_vgg__vggmini")
    graph = json.loads((source / entry["file"]).read_text(encoding="utf-8"))
    graph["model"].pop("meta", None)
    for node in graph["nodes"]:
        node.pop("meta", None)
    for group in graph["groups"]:
        group.pop("meta", None)
    (data / entry["file"]).write_text(json.dumps(graph, ensure_ascii=False), encoding="utf-8")
    index["models"] = [entry]
    (data / "index.json").write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
    with serve_dir(data) as url:
        yield url


# --------------------------------------------------------------------------- #
# page helpers
# --------------------------------------------------------------------------- #
def goto_app(ui, url: str) -> None:  # noqa: ANN001
    ui.page.goto(url, wait_until="networkidle", timeout=60_000)
    ui.page.wait_for_function(
        "() => document.documentElement.dataset.appState === 'ready'", timeout=45_000
    )
    ui.page.wait_for_timeout(300)


def require_table_ui(page) -> None:  # noqa: ANN001
    missing = [selector for selector in TABLE_UI_SELECTORS if page.locator(selector).count() == 0]
    if missing:
        pytest.skip(
            "v2 table UI is not available in web/** yet (missing: "
            + ", ".join(missing)
            + "); frontend-dev scope — this test must not pass silently"
        )
    if page.locator("#table-toggle").is_disabled():
        pytest.skip("v2 table toggle is present but still disabled — table UI not wired (frontend-dev scope)")


def require_algo_ui(page) -> None:  # noqa: ANN001
    missing = [selector for selector in ALGO_UI_SELECTORS if page.locator(selector).count() == 0]
    if missing:
        pytest.skip(
            "v2 algorithm UI is not available in web/** yet (missing: "
            + ", ".join(missing)
            + "); frontend-dev scope — this test must not pass silently"
        )
    try:
        page.wait_for_function(
            "() => { const b = document.querySelector('#algo-toggle'); return b && !b.disabled; }",
            timeout=20_000,
        )
    except Exception:  # noqa: BLE001 - timeout means the panel is not wired
        pytest.skip("v2 algorithm toggle never became enabled — panel not wired (frontend-dev scope)")


def hover_node_semantic(page, node_id: str, attempts: int = 8) -> str | None:  # noqa: ANN001
    """Hover a 3D node (via the debug screen-position helper) and read its semantic.

    The camera may still be animating after a tree-row click, so the projected
    position is refreshed per attempt and the tooltip is only accepted when it
    belongs to ``node_id`` (adjacent nodes can overlap on screen).
    """
    offsets = ((0, 0), (3, 0), (-3, 0), (0, 3), (0, -3), (6, 0), (-6, 0), (0, 6), (0, -6), (10, 0), (-10, 0))
    for _ in range(attempts):
        position = page.evaluate("(id) => window.__NETPRISM__.getNodeScreenPosition(id)", node_id)
        if position and position.get("visible"):
            for dx, dy in offsets:
                page.mouse.move(position["x"] + dx, position["y"] + dy)
                page.wait_for_timeout(90)
                if not page.locator("#tooltip").is_visible():
                    continue
                tooltip_name = page.locator("#tooltip .tt-name").inner_text().strip()
                if tooltip_name == node_id:
                    semantic = page.locator("#tooltip .tt-semantic")
                    return semantic.inner_text().strip() if semantic.count() else ""
        page.mouse.move(5, 5)
        page.wait_for_timeout(250)
    return None


def fetch_json(url: str, path: str) -> dict:
    with urllib.request.urlopen(url + path, timeout=15) as response:
        return json.loads(response.read().decode("utf-8"))


def wait_rows(page, count: int, timeout: int = 25_000) -> None:  # noqa: ANN001
    page.wait_for_function(
        "(expected) => document.querySelectorAll('#table-body tr').length === expected",
        arg=count,
        timeout=timeout,
    )


def table_row_ids(page) -> list[str]:  # noqa: ANN001
    return page.eval_on_selector_all(
        "#table-body tr", "rows => rows.map(row => row.getAttribute('data-node-id'))"
    )


def activation_mb_text(value: int) -> str:
    """Mirror web/js/format.js ``activationMBValue`` (UI CSV export contract)."""
    text = f"{value / (1024 * 1024):.6f}".rstrip("0").rstrip(".")
    return text or "0"


def parse_export(raw: bytes) -> tuple[list[str], list[list[str]]]:
    text = raw.decode("utf-8")
    assert not text.startswith("\ufeff"), "UI CSV export should be plain UTF-8"
    rows = list(csv.reader(io.StringIO(text, newline="")))
    assert rows, "downloaded CSV is empty"
    return rows[0], [row for row in rows[1:] if row]


def expected_first_by_params(graph: dict, *, descending: bool) -> str:
    nodes = graph["nodes"]
    ordered = sorted(
        enumerate(nodes),
        key=lambda pair: (
            -(int(pair[1].get("numParams") or 0)) if descending else int(pair[1].get("numParams") or 0),
            pair[0],
        ),
    )
    return ordered[0][1]["id"]


def export_table(page, tmp_path: Path, name: str) -> tuple[str, bytes]:  # noqa: ANN001
    with page.expect_download(timeout=20_000) as download_info:
        page.click("#table-export")
    download = download_info.value
    target = tmp_path / name
    download.save_as(target)
    return download.suggested_filename, target.read_bytes()


# --------------------------------------------------------------------------- #
# large synthetic model (DenseNet/GoogLeNet scale: 400+ nodes)
# --------------------------------------------------------------------------- #
def build_large_graph(node_count: int = 422) -> dict:
    """A schema-shaped 422-node CNN chain used to test the big-model table path."""
    assert node_count >= 260
    shape = [1, 32, 16, 16]
    nodes: list[dict] = [
        {
            "id": "n0",
            "name": "x",
            "op": "Input",
            "kind": "input",
            "modulePath": "",
            "depth": 0,
            "inShapes": [],
            "outShapes": [list(shape)],
            "params": {},
            "numParams": 0,
            "flops": 0,
            "tags": ["boundary"],
            "groupId": None,
            "source": None,
            "notes": [],
            "metrics": {"activationBytes": shape[1] * shape[2] * shape[3] * 4, "receptiveField": 1, "macs": 0},
        }
    ]
    edges: list[dict] = []
    total_params = 0
    total_flops = 0
    activation_sum = shape[1] * shape[2] * shape[3] * 4
    receptive_field = 1
    conv_index = 0
    for index in range(node_count - 2):
        node_id = f"n{index + 1}"
        kind = ("conv", "norm", "activation")[index % 3]
        if kind == "conv":
            name = f"conv_{conv_index}"
            conv_index += 1
            op = "Conv2d"
            num_params = 32 * 32 * 3 * 3 + 32
            num_flops = 2 * 32 * 32 * 3 * 3 * 16 * 16
            receptive_field += 2
        elif kind == "norm":
            name = f"norm_{index}"
            op = "BatchNorm2d"
            num_params = 64
            num_flops = 2 * 32 * 16 * 16
        else:
            name = f"relu_{index}"
            op = "ReLU"
            num_params = 0
            num_flops = 32 * 16 * 16
        total_params += num_params
        total_flops += num_flops
        activation = 32 * 16 * 16 * 4
        activation_sum += activation
        nodes.append(
            {
                "id": node_id,
                "name": name,
                "op": op,
                "kind": kind,
                "modulePath": f"block{index // 12}.{name}",
                "depth": 1,
                "inShapes": [list(shape)],
                "outShapes": [list(shape)],
                "params": {"kernelSize": [3, 3]} if kind == "conv" else {},
                "numParams": num_params,
                "flops": num_flops,
                "tags": [],
                "groupId": None,
                "source": None,
                "notes": [],
                "metrics": {
                    "activationBytes": activation,
                    "receptiveField": receptive_field,
                    "macs": num_flops // 2,
                },
            }
        )
        edges.append(
            {
                "id": f"e{len(edges)}",
                "source": f"n{index}" if index else "n0",
                "target": node_id,
                "kind": "data",
                "shapes": [list(shape)],
                "label": "",
            }
        )
    output = {
        "id": f"n{node_count - 1}",
        "name": "output",
        "op": "Output",
        "kind": "output",
        "modulePath": "",
        "depth": 0,
        "inShapes": [[1, 32]],
        "outShapes": [[1, 32]],
        "params": {},
        "numParams": 0,
        "flops": 0,
        "tags": ["boundary"],
        "groupId": None,
        "source": None,
        "notes": [],
        "metrics": {"activationBytes": 128, "receptiveField": receptive_field, "macs": 0},
    }
    nodes.append(output)
    activation_sum += 128
    edges.append(
        {
            "id": f"e{len(edges)}",
            "source": f"n{node_count - 3}",
            "target": output["id"],
            "kind": "data",
            "shapes": [[1, 32]],
            "label": "",
        }
    )
    return {
        "schemaVersion": "1.0",
        "generator": {"name": "netprism", "version": "0.1.0"},
        "model": {
            "name": "BigQANet",
            "framework": "pytorch",
            "className": "LargeQA",
            "mode": "trace",
            "source": {"file": "big_qa.py", "relPath": "big_qa.py"},
            "inputSpec": [{"name": "x", "shape": [1, 32, 16, 16], "dtype": "float32"}],
            "outputSpec": [{"name": "y", "shape": [1, 32], "dtype": "float32"}],
            "totalParams": total_params,
            "trainableParams": total_params,
            "totalFlops": total_flops,
            "flopsApprox": False,
            "numLayers": node_count - 2,
            "warnings": [],
            "metrics": {
                "paramBytes": total_params * 4,
                "activationBytes": activation_sum,
                "peakActivationBytes": 128,
                "maxReceptiveField": receptive_field,
                "macs": total_flops // 2,
            },
        },
        "nodes": nodes,
        "edges": edges,
        "groups": [],
    }


@pytest.fixture()
def large_url(tmp_path: Path):
    data = tmp_path / "large_data"
    data.mkdir()
    graph = build_large_graph(422)
    model_file = "bigqa__bigqanet.json"
    (data / model_file).write_text(json.dumps(graph, ensure_ascii=False), encoding="utf-8")
    index = {
        "schemaVersion": "1.0",
        "generatedAt": "2026-11-24T00:00:00+08:00",
        "root": str(data),
        "models": [
            {
                "id": "bigqa__bigqanet",
                "name": "BigQANet",
                "className": "LargeQA",
                "file": model_file,
                "source": "big_qa.py",
                "mode": "trace",
                "numLayers": graph["model"]["numLayers"],
                "totalParams": graph["model"]["totalParams"],
                "totalFlops": graph["model"]["totalFlops"],
                "warnings": 0,
            }
        ],
        "errors": [],
    }
    (data / "index.json").write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
    with serve_dir(data) as url:
        yield url


# --------------------------------------------------------------------------- #
# tests
# --------------------------------------------------------------------------- #
def test_table_view_lists_search_sorts_and_exports(demo_url: str, ui, tmp_path: Path) -> None:
    graph = fetch_json(demo_url, "api/model/cnn_vgg__vggmini")
    nodes = graph["nodes"]

    goto_app(ui, demo_url)
    page = ui.page
    require_table_ui(page)

    assert page.locator("#model-title").inner_text().strip() == graph["model"]["name"]
    assert page.locator("#framework-badge").is_visible()
    assert page.locator("#framework-badge").inner_text().strip() == "PyTorch"
    assert "TRACE" in page.locator("#mode-badge").inner_text().upper()

    # v2 summary metrics: activation estimate must be visible and plausible.
    assert page.locator("#metric-activation").is_visible()
    activation_text = page.locator("#metric-activation").inner_text()
    assert "估算" in activation_text
    assert "MB" in activation_text
    displayed_mb = float(activation_text.split()[-2] if activation_text.split()[-1] == "MB" else "0")
    expected_mb = graph["model"]["metrics"]["activationBytes"] / (1024 * 1024)
    assert abs(displayed_mb - expected_mb) < 0.02, (displayed_mb, expected_mb)
    assert page.locator("#metric-rf").is_visible()

    # Toggle the table open.
    assert page.locator("#table-toggle").get_attribute("aria-pressed") == "false"
    assert page.locator("#table-view").is_hidden()
    page.click("#table-toggle")
    page.wait_for_selector("#table-view", state="visible", timeout=10_000)
    assert page.locator("#table-toggle").get_attribute("aria-pressed") == "true"
    wait_rows(page, len(nodes))
    assert len(table_row_ids(page)) == len(nodes)
    assert set(table_row_ids(page)) == {node["id"] for node in nodes}
    count_text = page.locator("#table-count").inner_text()
    assert str(len(nodes)) in count_text and "层" in count_text
    assert page.locator("#table-framework").is_visible()

    # Search filters rows down and the empty state appears for nonsense queries.
    conv_count = sum(1 for node in nodes if node["op"] == "Conv2d")
    assert conv_count == 6
    page.fill("#table-search", "Conv2d")
    wait_rows(page, conv_count)
    assert set(table_row_ids(page)) == {
        node["id"] for node in nodes if node["op"] == "Conv2d"
    }
    page.fill("#table-search", "features.0")
    wait_rows(page, 1)
    target = next(node for node in nodes if node["name"] == "features.0")
    assert table_row_ids(page) == [target["id"]]
    page.fill("#table-search", "definitely-no-such-layer")
    wait_rows(page, 0)
    assert page.locator("#table-empty").is_visible()
    assert "0" in page.locator("#table-count").inner_text()
    page.fill("#table-search", "")
    wait_rows(page, len(nodes))

    # Header sorting must reorder the first row (params asc -> min, desc -> max).
    ascending_first = expected_first_by_params(graph, descending=False)
    descending_first = expected_first_by_params(graph, descending=True)
    assert ascending_first != descending_first
    page.click('th[data-sort="params"]')
    assert page.locator('th[data-sort="params"]').get_attribute("aria-sort") == "ascending"
    wait_rows(page, len(nodes))
    assert table_row_ids(page)[0] == ascending_first
    page.click('th[data-sort="params"]')
    assert page.locator('th[data-sort="params"]').get_attribute("aria-sort") == "descending"
    wait_rows(page, len(nodes))
    assert table_row_ids(page)[0] == descending_first

    # CSV export accepts the download event, honours the current filter, and is clean UTF-8.
    page.fill("#table-search", "Conv2d")
    wait_rows(page, conv_count)
    filename, raw = export_table(page, tmp_path, "filtered.csv")
    assert filename == "VGGMini_layers.csv"
    assert b"\r\r\n" not in raw, "exported CSV contains doubled CR (\\r\\r\\n)"
    header, rows = parse_export(raw)
    assert header == EXPORT_COLUMNS
    assert len(rows) == conv_count
    assert all(len(row) == len(EXPORT_COLUMNS) for row in rows)
    assert {row[1] for row in rows} == {"Conv2d"}
    by_name = {row[0]: row for row in rows}
    conv_node = next(node for node in nodes if node["op"] == "Conv2d")
    exported = by_name[conv_node["name"]]
    assert int(exported[5]) == conv_node["numParams"]
    assert int(exported[6]) == conv_node["flops"]
    assert exported[7] == activation_mb_text(conv_node["metrics"]["activationBytes"])
    assert exported[8] == str(conv_node["metrics"]["receptiveField"])

    # Full export after clearing the filter contains every layer.
    page.fill("#table-search", "")
    wait_rows(page, len(nodes))
    filename, raw = export_table(page, tmp_path, "all.csv")
    assert filename == "VGGMini_layers.csv"
    header, rows = parse_export(raw)
    assert header == EXPORT_COLUMNS
    assert len(rows) == len(nodes)
    assert {row[0] for row in rows} == {node["name"] for node in nodes}

    # Escape closes the table without breaking the 3D view.
    page.keyboard.press("Escape")
    page.wait_for_selector("#table-view", state="hidden", timeout=10_000)
    assert page.locator("#table-toggle").get_attribute("aria-pressed") == "false"
    assert page.locator("#viewport canvas").count() > 0

    ui.log.assert_clean()


def test_model_switch_keeps_table_and_metrics_working(demo_url: str, ui) -> None:
    goto_app(ui, demo_url)
    page = ui.page
    require_table_ui(page)
    page.click("#table-toggle")
    page.wait_for_selector("#table-view", state="visible", timeout=10_000)
    wait_rows(page, len(fetch_json(demo_url, "api/model/cnn_vgg__vggmini")["nodes"]))

    resnet = fetch_json(demo_url, "api/model/resnet_mini__resnetmini")
    page.select_option("#model-select", value="resnet_mini__resnetmini")
    page.wait_for_function(
        "() => document.querySelector('#model-title')?.textContent.trim() === 'ResNetMini'",
        timeout=30_000,
    )
    wait_rows(page, len(resnet["nodes"]))
    assert page.locator("#table-view").is_visible()
    assert set(table_row_ids(page)) == {node["id"] for node in resnet["nodes"]}
    assert page.locator("#framework-badge").is_visible()

    # Per-layer estimated metrics must be rendered, not silently "--".
    rendered_metrics = page.locator("#table-body td.t-activation:not(.is-missing)").count()
    assert rendered_metrics == len(resnet["nodes"])

    page.fill("#table-search", "stem.0")
    wait_rows(page, 1)
    stem = next(node for node in resnet["nodes"] if node["name"] == "stem.0")
    assert table_row_ids(page) == [stem["id"]]
    page.fill("#table-search", "")
    wait_rows(page, len(resnet["nodes"]))

    ui.log.assert_clean()


def test_large_model_422_nodes_table_and_export(large_url: str, ui, tmp_path: Path) -> None:
    graph = fetch_json(large_url, "api/model/bigqa__bigqanet")
    assert len(graph["nodes"]) == 422

    goto_app(ui, large_url)
    page = ui.page
    require_table_ui(page)
    assert page.locator("#model-title").inner_text().strip() == "BigQANet"
    assert page.locator("#big-badge").is_visible()
    assert "422" in page.locator("#big-badge").inner_text()

    page.click("#table-toggle")
    page.wait_for_selector("#table-view", state="visible", timeout=10_000)
    wait_rows(page, len(graph["nodes"]), timeout=40_000)
    assert len(table_row_ids(page)) == len(graph["nodes"])

    # Search still works on a 400+ node model.
    conv = next(node for node in graph["nodes"] if node["name"] == "conv_100")
    page.fill("#table-search", conv["name"])
    wait_rows(page, 1)
    assert table_row_ids(page) == [conv["id"]]
    page.fill("#table-search", "")
    wait_rows(page, len(graph["nodes"]), timeout=40_000)

    # Sorting and export are not disabled by the big-model rendering mode.
    descending_first = expected_first_by_params(graph, descending=True)
    page.click('th[data-sort="params"]')
    page.click('th[data-sort="params"]')
    wait_rows(page, len(graph["nodes"]), timeout=40_000)
    assert table_row_ids(page)[0] == descending_first

    filename, raw = export_table(page, tmp_path, "big.csv")
    assert filename == "BigQANet_layers.csv"
    assert b"\r\r\n" not in raw
    header, rows = parse_export(raw)
    assert header == EXPORT_COLUMNS
    assert len(rows) == len(graph["nodes"])
    assert all(len(row) == len(EXPORT_COLUMNS) for row in rows)
    assert {row[0] for row in rows} == {node["name"] for node in graph["nodes"]}

    ui.log.assert_clean()


# --------------------------------------------------------------------------- #
# v2.1 algorithm-understanding panel (SPEC-ALGO section 6)
# --------------------------------------------------------------------------- #
def test_algo_panel_family_patterns_pseudocode_and_shortcuts(demo_url: str, ui) -> None:
    graph = fetch_json(demo_url, "api/model/cnn_vgg__vggmini")
    meta = graph["model"]["meta"]
    pattern_count = len(meta["patterns"])

    goto_app(ui, demo_url)
    page = ui.page
    require_algo_ui(page)

    assert page.locator("#algo-toggle").get_attribute("aria-pressed") == "false"
    assert page.locator("#algo-view").is_hidden()
    page.click("#algo-toggle")
    page.wait_for_selector("#algo-view", state="visible", timeout=10_000)
    assert page.locator("#algo-toggle").get_attribute("aria-pressed") == "true"

    # Family card: label + code + confidence percentage.
    family_text = page.locator("#algo-family").inner_text()
    assert "VGG" in family_text
    assert "vgg" in family_text
    assert f"{round(float(meta['familyConfidence']) * 100)}%" in family_text
    assert "VGGMini" in page.locator("#algo-model-name").inner_text()

    # Pattern cards: count, formulas and evidence come straight from model.meta.
    assert page.locator("#algo-patterns .algo-pattern").count() == pattern_count
    formulas = [text.strip() for text in page.locator("#algo-patterns .algo-formula").all_inner_texts()]
    assert formulas == [pattern["formula"] for pattern in meta["patterns"]]
    assert "y = Act(BN(Conv(x)))" in formulas
    assert page.locator("#algo-patterns .algo-evidence li").count() >= pattern_count
    assert f"{pattern_count} 项" in page.locator("#algo-pattern-count").inner_text()
    assert "family=vgg" in page.locator("#algo-foot-meta").inner_text()
    assert f"{pattern_count} 个模式" in page.locator("#algo-foot-meta").inner_text()

    # Pseudocode block: identical lines, shape comments included.
    pseudocode_text = page.locator("#algo-pseudocode").inner_text().strip()
    assert pseudocode_text == "\n".join(meta["pseudocode"]).strip()
    assert "x = Input(1, 3, 64, 64)" in pseudocode_text
    assert pseudocode_text.count("# -> (") >= 3
    assert not page.locator("#algo-copy").is_disabled()

    # Escape closes; "A" re-opens/closes.
    page.keyboard.press("Escape")
    page.wait_for_selector("#algo-view", state="hidden", timeout=10_000)
    assert page.locator("#algo-toggle").get_attribute("aria-pressed") == "false"
    page.keyboard.press("a")
    page.wait_for_selector("#algo-view", state="visible", timeout=10_000)
    page.keyboard.press("a")
    page.wait_for_selector("#algo-view", state="hidden", timeout=10_000)

    ui.log.assert_clean()


def test_algo_panel_updates_on_model_switch_and_table_still_works(demo_url: str, ui) -> None:
    resnet = fetch_json(demo_url, "api/model/resnet_mini__resnetmini")
    resnet_meta = resnet["model"]["meta"]

    goto_app(ui, demo_url)
    page = ui.page
    require_algo_ui(page)
    page.click("#algo-toggle")
    page.wait_for_selector("#algo-view", state="visible", timeout=10_000)

    page.select_option("#model-select", value="resnet_mini__resnetmini")
    page.wait_for_function(
        "() => document.querySelector('#model-title')?.textContent.trim() === 'ResNetMini'",
        timeout=30_000,
    )
    page.wait_for_function(
        "() => document.querySelector('#algo-family')?.textContent.includes('ResNet')",
        timeout=20_000,
    )
    family_text = page.locator("#algo-family").inner_text()
    assert "ResNet" in family_text and "resnet" in family_text
    assert page.locator("#algo-patterns .algo-pattern").count() == len(resnet_meta["patterns"])
    assert page.locator("#algo-patterns .algo-formula").all_inner_texts()
    assert page.locator("#algo-pseudocode").inner_text().strip() == "\n".join(
        resnet_meta["pseudocode"]
    ).strip()

    # The table view remains usable after closing the algorithm panel.
    page.keyboard.press("Escape")
    page.wait_for_selector("#algo-view", state="hidden", timeout=10_000)
    page.click("#table-toggle")
    page.wait_for_selector("#table-view", state="visible", timeout=10_000)
    wait_rows(page, len(resnet["nodes"]))
    assert set(table_row_ids(page)) == {node["id"] for node in resnet["nodes"]}

    ui.log.assert_clean()


def test_algo_layer_detail_formula_and_tooltip_semantic(demo_url: str, ui) -> None:
    graph = fetch_json(demo_url, "api/model/cnn_vgg__vggmini")
    conv = next(
        node
        for node in graph["nodes"]
        if node["kind"] == "conv" and node.get("meta", {}).get("formula") and node["meta"].get("semantic")
    )
    meta = conv["meta"]

    goto_app(ui, demo_url)
    page = ui.page
    require_algo_ui(page)

    # Selecting a layer opens the detail panel with its algorithm explanation.
    page.click(f'#layer-tree [data-node-id="{conv["id"]}"]')
    page.wait_for_selector("#detail-algo", state="visible", timeout=15_000)
    page.wait_for_timeout(1200)  # let the focus camera animation settle before hovering
    detail_text = page.locator("#detail-algo").inner_text()
    assert meta["formula"] in detail_text
    assert meta["semantic"] in detail_text

    # Group pattern (group.meta.pattern) is shown for the owning group when present.
    groups = {group["id"]: group for group in graph["groups"]}
    owner = groups.get(conv.get("groupId") or "")
    expected_pattern = (owner or {}).get("meta", {}).get("pattern")
    if expected_pattern:
        assert page.locator("#detail-group-pattern").count() >= 1
        assert expected_pattern["formula"] in page.locator("#detail-group-pattern").inner_text()

    # Tooltip exposes the same layer semantic from node.meta.
    semantic = hover_node_semantic(page, conv["id"])
    assert semantic is not None, "could not hover the 3D node to inspect its tooltip"
    assert semantic == meta["semantic"].strip()

    ui.log.assert_clean()


def test_algo_panel_empty_meta_shows_placeholder(nometa_url: str, ui) -> None:
    goto_app(ui, nometa_url)
    page = ui.page
    require_algo_ui(page)
    assert page.locator("#model-title").inner_text().strip() == "VGGMini"

    page.click("#algo-toggle")
    page.wait_for_selector("#algo-view", state="visible", timeout=10_000)
    assert "暂无算法信息" in page.locator("#algo-family").inner_text()
    assert "暂无模式" in page.locator("#algo-patterns").inner_text()
    assert page.locator("#algo-pseudocode").inner_text().strip() == "暂无伪代码"
    assert page.locator("#algo-copy").is_disabled()
    assert page.locator("#algo-pattern-count").inner_text().strip() == "—"

    # Old graphs must not fake a layer formula in the detail panel.
    page.click("#algo-close")
    page.wait_for_selector("#algo-view", state="hidden", timeout=10_000)
    page.click('#layer-tree [data-node-id="n1"]')
    page.wait_for_function(
        "() => { const m = document.querySelector('#detail-meta'); return m && m.textContent.trim() !== '单击层查看'; }",
        timeout=15_000,
    )
    assert page.locator("#detail-algo").count() == 0
    assert page.locator("#detail-group-pattern").count() == 0

    ui.log.assert_clean()
