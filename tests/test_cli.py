"""End-to-end tests for the NetPrism CLI and HTTP server.

The real commands are executed through ``python -m netprism`` in a subprocess
(with the interpreter running the tests) so the public contract from SPEC
sections 6 and 8 is exercised: ``scan`` output files + schema validity,
``inspect --json``, and the ``serve`` API (health/index/model/404/path
traversal/upload-parse).

These tests need the parser engine (``netprism.core``); they are not skipped on
purpose -- a missing engine is a hard failure that the Lead must see.
"""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import jsonschema
import pytest

from netprism import schema

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"
PYTHON = sys.executable

EXPECTED_CLASSES = {
    "VGGMini",
    "ResNetMini",
    "TinyTransformerClassifier",
    "TinyGPT",
    "UNetMini",
    "MLP",
    "NeedsArgs",
    "DynamicNet",
}
EXPECTED_MODEL_COUNT = len(EXPECTED_CLASSES)

UPLOAD_SOURCE = '''\
"""Minimal model uploaded by the static /api/upload-parse test."""

import torch.nn as nn


class UploadedTiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(8, 2)

    def forward(self, x):
        return self.fc(x)
'''


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def run_cli(*args: str, timeout: float = 300.0) -> subprocess.CompletedProcess[str]:
    """Run ``python -m netprism <args>`` from the project root."""
    return subprocess.run(
        [PYTHON, "-m", "netprism", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def http_get(base_url: str, path: str, timeout: float = 10.0) -> tuple[int, Any, bytes]:
    """GET ``path``; returns (status, headers, body) even for 4xx/5xx."""
    request = urllib.request.Request(base_url + path, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.headers, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.headers, exc.read()


def http_post_json(base_url: str, path: str, payload: dict[str, Any], timeout: float = 60.0) -> tuple[int, Any, bytes]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        base_url + path,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json", "Content-Length": str(len(body))},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.headers, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.headers, exc.read()


def parse_json(body: bytes, *, where: str) -> Any:
    try:
        return json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        pytest.fail(f"{where}: response is not UTF-8 JSON: {exc}\n{body[:400]!r}")


def graph_schema() -> dict[str, Any]:
    path = ROOT / "schema" / "graph.schema.json"
    assert path.is_file(), f"missing JSON Schema: {path}"
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def scanned(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Run ``netprism scan samples`` once and load the produced JSON files."""
    out_dir = tmp_path_factory.mktemp("netprism_scan")
    proc = run_cli("scan", str(SAMPLES), "--out", str(out_dir), "--quiet")
    assert proc.returncode == 0, (
        "scan failed\n"
        f"exit code: {proc.returncode}\n"
        f"stdout:\n{proc.stdout}\n"
        f"stderr:\n{proc.stderr}"
    )

    index_path = out_dir / "index.json"
    assert index_path.is_file(), f"scan did not write {index_path}\nstderr:\n{proc.stderr}"
    index = json.loads(index_path.read_text(encoding="utf-8"))

    graphs: dict[str, dict[str, Any]] = {}
    for entry in index.get("models", []):
        graph_path = out_dir / str(entry["file"])
        assert graph_path.is_file(), f"missing graph file for {entry['id']}: {graph_path}"
        graphs[entry["id"]] = json.loads(graph_path.read_text(encoding="utf-8"))

    return {"dir": out_dir, "index": index, "graphs": graphs, "stdout": proc.stdout}


@pytest.fixture(scope="session")
def server(scanned: dict[str, Any], tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Start ``netprism serve <scan out>`` on a free port and stop it afterwards."""
    port = free_port()
    log_path = tmp_path_factory.mktemp("server_log") / "server.log"
    log_handle = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        [PYTHON, "-u", "-m", "netprism", "serve", str(scanned["dir"]), "--port", str(port)],
        cwd=ROOT,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    base_url = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 90.0
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                log_handle.flush()
                pytest.fail(
                    f"server exited early (code {proc.returncode}); log:\n{log_path.read_text(encoding='utf-8', errors='replace')}"
                )
            try:
                status, _headers, body = http_get(base_url, "/api/health", timeout=2.0)
                if status == 200 and json.loads(body).get("ok") is True:
                    break
            except (urllib.error.URLError, OSError, json.JSONDecodeError):
                pass
            time.sleep(0.2)
        else:
            log_handle.flush()
            pytest.fail(
                f"server did not become healthy at {base_url}; log:\n"
                f"{log_path.read_text(encoding='utf-8', errors='replace')}"
            )
        yield {"base_url": base_url, "proc": proc, "log": log_path}
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=15)
        log_handle.close()


# --------------------------------------------------------------------------- #
# CLI surface (no parser engine needed)
# --------------------------------------------------------------------------- #
def test_cli_help_lists_commands() -> None:
    proc = run_cli("--help")
    assert proc.returncode == 0, proc.stderr
    for command in ("scan", "inspect", "serve", "view"):
        assert command in proc.stdout, f"--help does not mention {command}"


def test_cli_version() -> None:
    proc = run_cli("--version")
    assert proc.returncode == 0, proc.stderr
    text = (proc.stdout + proc.stderr).strip()
    assert "netprism" in text.lower()
    assert schema.SCHEMA_VERSION in text


# --------------------------------------------------------------------------- #
# scan
# --------------------------------------------------------------------------- #
def test_scan_index_models_and_fields(scanned: dict[str, Any]) -> None:
    index = scanned["index"]
    assert schema.validate_index(index) == [], f"invalid index: {schema.validate_index(index)}"

    models = index["models"]
    assert len(models) == EXPECTED_MODEL_COUNT, (
        f"expected {EXPECTED_MODEL_COUNT} models, got {len(models)}: "
        f"{[(m.get('source'), m.get('className')) for m in models]}"
    )
    assert {model.get("className") for model in models} == EXPECTED_CLASSES
    assert index["errors"] == [], f"scan reported errors: {index['errors']}"
    assert Path(index["root"]).resolve() == SAMPLES.resolve()

    ids = [model["id"] for model in models]
    assert len(set(ids)) == len(ids), f"duplicate model ids: {ids}"
    for model in models:
        for key in ("id", "name", "className", "file", "source", "mode", "numLayers",
                    "totalParams", "totalFlops", "warnings"):
            assert key in model, f"{model.get('id')} missing index field {key!r}"
        assert model["mode"] in schema.MODES
        assert isinstance(model["warnings"], int)


def test_scan_out_dir_contains_only_index_and_graphs(scanned: dict[str, Any]) -> None:
    out_dir: Path = scanned["dir"]
    expected = {"index.json"} | {model["file"] for model in scanned["index"]["models"]}
    actual = {path.name for path in out_dir.iterdir() if path.is_file()}
    assert actual == expected, f"unexpected files in scan output: {sorted(actual ^ expected)}"
    assert not (out_dir / "index.json.tmp").exists()


def test_scan_graphs_match_index_and_schema(scanned: dict[str, Any]) -> None:
    index = scanned["index"]
    graphs = scanned["graphs"]
    json_schema = graph_schema()

    for entry in index["models"]:
        model_id = entry["id"]
        graph = graphs[model_id]
        assert graph["model"]["name"] == entry["name"]
        assert graph["model"]["className"] == entry["className"]
        assert graph["model"]["mode"] == entry["mode"]
        assert graph["model"]["source"]["relPath"] == entry["source"]
        assert graph["model"]["totalParams"] == entry["totalParams"]
        assert graph["model"]["totalFlops"] == entry["totalFlops"]
        assert graph["model"]["numLayers"] == entry["numLayers"]

        errors = schema.validate_graph(graph)
        assert errors == [], f"{model_id} violates netprism.schema: {errors}"
        try:
            jsonschema.validate(instance=graph, schema=json_schema)
        except jsonschema.ValidationError as exc:
            pytest.fail(f"{model_id} violates schema/graph.schema.json: {exc.message} at {list(exc.absolute_path)}")

        node_ids = [node["id"] for node in graph["nodes"]]
        assert len(node_ids) == len(set(node_ids))
        assert len(node_ids) > 1, f"{model_id} has no nodes"
        assert any(node["kind"] == "input" for node in graph["nodes"])
        assert any(node["kind"] == "output" for node in graph["nodes"])
        assert any(group["kind"] == "model" for group in graph["groups"]), f"{model_id} has no model root group"


def test_scan_fallbacks(scanned: dict[str, Any]) -> None:
    by_class = {model["className"]: model for model in scanned["index"]["models"]}

    # NeedsArgs cannot be instantiated -> static AST fallback with a warning.
    needs_args = by_class["NeedsArgs"]
    assert needs_args["mode"] == "static", f"NeedsArgs mode = {needs_args['mode']!r}"
    assert needs_args["warnings"] >= 1, "NeedsArgs should carry a static-fallback warning"

    # DynamicNet cannot be FX-traced -> trace/hook fallback with a warning.
    dynamic = by_class["DynamicNet"]
    assert dynamic["mode"] in ("trace", "hybrid"), f"DynamicNet mode = {dynamic['mode']!r}"
    assert dynamic["warnings"] >= 1, "DynamicNet should warn about the data-dependent branch"


# --------------------------------------------------------------------------- #
# inspect
# --------------------------------------------------------------------------- #
def test_inspect_json_resnet() -> None:
    proc = run_cli("inspect", str(SAMPLES / "resnet_mini.py"), "--json")
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    graph = json.loads(proc.stdout)
    assert schema.validate_graph(graph) == []
    assert graph["model"]["name"] == "ResNetMini"
    assert graph["model"]["mode"] == "trace"

    kinds = {node["kind"] for node in graph["nodes"]}
    assert "conv" in kinds and "linear" in kinds
    conv = next(node for node in graph["nodes"] if node["kind"] == "conv")
    assert len(conv["inShapes"]) >= 1 and len(conv["outShapes"]) >= 1

    residual_edges = [edge for edge in graph["edges"] if edge["kind"] == "residual"]
    assert residual_edges, "ResNetMini should expose residual add edges"
    for edge in residual_edges:
        assert edge["source"] != edge["target"]


def test_inspect_text_output() -> None:
    proc = run_cli("inspect", str(SAMPLES / "resnet_mini.py"))
    assert proc.returncode == 0, proc.stderr
    assert "ResNetMini" in proc.stdout
    assert "Conv2d" in proc.stdout


def test_inspect_static_fallback() -> None:
    proc = run_cli("inspect", str(SAMPLES / "needs_args.py"), "--json")
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    graph = json.loads(proc.stdout)
    assert schema.validate_graph(graph) == []
    assert graph["model"]["name"] == "NeedsArgs"
    assert graph["model"]["mode"] == "static"


def test_inspect_non_model_file_is_skipped() -> None:
    proc = run_cli("inspect", str(SAMPLES / "not_a_model.py"))
    assert proc.returncode == 1, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "没有找到" in (proc.stdout + proc.stderr)


# --------------------------------------------------------------------------- #
# HTTP server
# --------------------------------------------------------------------------- #
def test_server_health(server: dict[str, Any]) -> None:
    status, headers, body = http_get(server["base_url"], "/api/health")
    assert status == 200
    payload = parse_json(body, where="GET /api/health")
    assert payload == {"ok": True, "version": schema.GENERATOR_VERSION, "schemaVersion": schema.SCHEMA_VERSION}
    assert headers.get("Cache-Control") == "no-store"


def test_server_index(server: dict[str, Any]) -> None:
    status, headers, body = http_get(server["base_url"], "/api/index")
    assert status == 200
    index = parse_json(body, where="GET /api/index")
    assert schema.validate_index(index) == [], f"invalid served index: {schema.validate_index(index)}"
    assert len(index["models"]) >= EXPECTED_MODEL_COUNT
    classes = {model.get("className") for model in index["models"]}
    assert EXPECTED_CLASSES <= classes, f"missing models: {EXPECTED_CLASSES - classes}"
    assert headers.get("Cache-Control") == "no-store"


def test_server_model_graph(server: dict[str, Any]) -> None:
    _status, _headers, body = http_get(server["base_url"], "/api/index")
    index = json.loads(body)
    entry = next(model for model in index["models"] if model["className"] == "ResNetMini")

    status, _headers, body = http_get(server["base_url"], f"/api/model/{entry['id']}")
    assert status == 200
    graph = parse_json(body, where=f"GET /api/model/{entry['id']}")
    assert schema.validate_graph(graph) == []
    assert graph["model"]["name"] == "ResNetMini"


def test_server_unknown_model_is_404(server: dict[str, Any]) -> None:
    status, _headers, body = http_get(server["base_url"], "/api/model/definitely-not-a-model")
    assert status == 404
    payload = parse_json(body, where="unknown model")
    assert "error" in payload


def test_server_unknown_api_is_404(server: dict[str, Any]) -> None:
    status, _headers, body = http_get(server["base_url"], "/api/does-not-exist")
    assert status == 404
    payload = parse_json(body, where="unknown api")
    assert "error" in payload


@pytest.mark.parametrize(
    "path",
    [
        "/api/model/..%2f..%2fnetprism%2fschema.py",
        "/api/model/%2e%2e%2f%2e%2e%2fetc%2fpasswd",
        "/api/model/..%5c..%5cwindows%5cwin.ini",
    ],
)
def test_server_rejects_path_traversal(server: dict[str, Any], path: str) -> None:
    status, _headers, body = http_get(server["base_url"], path)
    assert status in (400, 403, 404), f"{path} returned {status}"
    payload = parse_json(body, where=path)
    assert "error" in payload, f"{path} returned JSON without an error field: {payload!r}"
    assert "schemaVersion" not in payload, f"{path} leaked a graph document"


def test_server_upload_parse_static(server: dict[str, Any]) -> None:
    payload = {"files": [{"path": "uploaded/uploaded_tiny.py", "content": UPLOAD_SOURCE}], "execute": False}
    status, _headers, body = http_post_json(server["base_url"], "/api/upload-parse", payload)
    assert status == 200, body[:800]
    result = parse_json(body, where="POST /api/upload-parse")
    for key in ("index", "graphs", "errors"):
        assert key in result, f"upload response missing {key!r}"
    assert result["errors"] == []
    assert len(result["graphs"]) == 1

    model_id, graph = next(iter(result["graphs"].items()))
    assert schema.validate_graph(graph) == []
    assert graph["model"]["className"] == "UploadedTiny"
    assert graph["model"]["mode"] == "static"

    uploaded_classes = {model.get("className") for model in result["index"]["models"]}
    assert "UploadedTiny" in uploaded_classes
    assert EXPECTED_CLASSES <= uploaded_classes

    # The uploaded model is served like any scanned model.
    get_status, _headers, get_body = http_get(server["base_url"], f"/api/model/{model_id}")
    assert get_status == 200
    served = parse_json(get_body, where=f"GET /api/model/{model_id}")
    assert served["model"]["className"] == "UploadedTiny"


def test_server_upload_execute_requires_flag(server: dict[str, Any]) -> None:
    payload = {"files": [{"path": "tiny.py", "content": UPLOAD_SOURCE}], "execute": True}
    status, _headers, body = http_post_json(server["base_url"], "/api/upload-parse", payload)
    assert status == 403, f"execute=true must be rejected without --allow-execute, got {status}: {body[:400]}"
    assert "error" in parse_json(body, where="POST execute=true")
