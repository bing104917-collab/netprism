"""Contract tests for the frozen graph schema and the static HTTP server."""

from __future__ import annotations

import copy
import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from netprism.schema import (
    SCHEMA_PATH,
    load_json_schema,
    validate_graph,
    validate_index,
)
from netprism.server import create_app, create_server

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "examples" / "demo"
FIXTURE = ROOT / "tests" / "fixtures" / "cnn_demo.json"


@pytest.fixture(scope="module")
def demo_graph() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def demo_index() -> dict:
    return json.loads((DEMO / "index.json").read_text(encoding="utf-8"))


def test_demo_graph_matches_json_schema(demo_graph):
    schema = load_json_schema()
    errors = sorted(Draft202012Validator(schema).iter_errors(demo_graph), key=lambda e: list(e.path))
    assert errors == [], [f"{list(e.path)}: {e.message}" for e in errors]


def test_demo_graph_semantic_validation(demo_graph):
    assert validate_graph(demo_graph) == []


def test_demo_index_validation(demo_index):
    assert validate_index(demo_index) == []


def test_rejects_duplicate_node_ids(demo_graph):
    broken = copy.deepcopy(demo_graph)
    broken["nodes"][1]["id"] = broken["nodes"][0]["id"]
    assert any("duplicate" in e.lower() for e in validate_graph(broken))


def test_rejects_dangling_edge(demo_graph):
    broken = copy.deepcopy(demo_graph)
    broken["edges"][0]["target"] = "n999"
    assert any("unknown node" in e for e in validate_graph(broken))


def test_rejects_group_membership_mismatch(demo_graph):
    broken = copy.deepcopy(demo_graph)
    broken["groups"][1]["nodeIds"].append("n10")
    assert any("groupId" in e for e in validate_graph(broken))


def test_rejects_cycle(demo_graph):
    broken = copy.deepcopy(demo_graph)
    broken["groups"][0]["parentId"] = "g4"
    assert any("cycle" in e for e in validate_graph(broken))


def test_accepts_new_format_and_optional_metrics(demo_graph):
    graph = copy.deepcopy(demo_graph)
    graph["model"]["framework"] = "keras"
    graph["model"]["mode"] = "keras"
    graph["model"]["meta"] = {"source_format": "keras-v3"}
    graph["model"]["metrics"] = {
        "paramBytes": 42408,
        "activationBytes": 1024,
        "maxReceptiveField": 5,
        "macs": 1234,
    }
    graph["nodes"][1]["meta"] = {"op_type": "Conv"}
    graph["nodes"][1]["metrics"] = {"activationBytes": 256, "receptiveField": 3, "macs": 512}
    assert validate_graph(graph) == []
    schema = load_json_schema()
    errors = list(Draft202012Validator(schema).iter_errors(graph))
    assert errors == []


def test_rejects_unknown_framework(demo_graph):
    graph = copy.deepcopy(demo_graph)
    graph["model"]["framework"] = "mxnet"
    assert any("model.framework" in e for e in validate_graph(graph))


def test_accepts_group_algorithm_meta(demo_graph):
    graph = copy.deepcopy(demo_graph)
    graph["groups"][1]["meta"] = {
        "pattern": {
            "kind": "residual_block",
            "label": "Residual Block",
            "formula": "y = F(x) + x",
            "confidence": 0.95,
        }
    }
    assert validate_graph(graph) == []
    schema = load_json_schema()
    assert list(Draft202012Validator(schema).iter_errors(graph)) == []


class _Server:
    def __init__(self) -> None:
        self.app = create_app(DEMO)
        self.httpd = create_server(self.app, "127.0.0.1", 0)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self) -> "_Server":
        self.thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)

    def url(self, path: str) -> str:
        return f"http://127.0.0.1:{self.port}{path}"


def _get(url: str) -> tuple[int, dict]:
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def test_server_health_and_index():
    with _Server() as server:
        status, payload = _get(server.url("/api/health"))
        assert status == 200 and payload["ok"] is True
        status, index = _get(server.url("/api/index"))
        assert status == 200
        assert index["schemaVersion"] == "1.0"
        first = index["models"][0]
        model_id = first["id"]
        status, graph = _get(server.url(f"/api/model/{model_id}"))
        assert status == 200
        assert graph["model"]["name"] == first["name"]


def test_server_rejects_unknown_and_traversal():
    with _Server() as server:
        assert _get(server.url("/api/model/does-not-exist"))[0] == 404
        assert _get(server.url("/api/model/..%2F..%2Fetc%2Fpasswd"))[0] == 404
        assert _get(server.url("/api/model/..%5C..%5Csecret"))[0] == 404


def test_schema_file_is_valid_json():
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    assert schema["$id"].endswith("graph-1.0.json")
    assert "node" in schema["$defs"] and "group" in schema["$defs"]
