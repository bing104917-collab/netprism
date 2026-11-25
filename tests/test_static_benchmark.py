"""Independent QA for the static-parser benchmark fix (task-15).

Verification is end-to-end and black box:

* CLI ``scan --no-execute`` over the benchmark archive (five models);
* CLI ``inspect --no-execute`` on synthetic unknown-function models;
* a real ``netprism serve`` process with ``POST /api/upload-parse`` for the GAT
  model, including a proof that ``execute=false`` really does not run code.

The benchmark archive lives in ``tests/fixtures/benchmark/benchmark`` (a
user-supplied copy of the benchmark).  Tests skip with an explicit message when
it is absent.
"""

from __future__ import annotations

import contextlib
import json
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from netprism.schema import validate_graph

ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = ROOT / "examples" / "demo"
BENCHMARK = ROOT / "tests" / "fixtures" / "benchmark" / "benchmark"
PY = sys.executable

BENCHMARK_MODELS = {
    "FNOSensor": "fno_sensor/model.py",
    "SensorGAT": "gat/model.py",
    "MaskedUNet3D": "masked_unet/model.py",
    "SensorTransformer": "sensor_transformer/model.py",
    "VoronoiCNN": "voronoi_cnn/model.py",
}

#: Numbers promised by the static-parser fix.
GAT_NODE_COUNT = 47
GAT_PARAMS = 76_931
FNO_PARAM_NODES = 4
FNO_PARAM_SIZE = 262_144


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def run_cli(*args: str, cwd: Path = ROOT, timeout: int = 300) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [PY, "-m", "netprism", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def node_params(graph: dict) -> int:
    return sum(int(node.get("numParams") or 0) for node in graph["nodes"])


def all_node_shapes(graph: dict) -> list[list[int]]:
    shapes: list[list[int]] = []
    for node in graph["nodes"]:
        for key in ("inShapes", "outShapes"):
            for shape in node.get(key) or []:
                shapes.append(list(shape))
    return shapes


def assert_no_unknown_dimensions(graph: dict, label: str) -> None:
    for shape in all_node_shapes(graph):
        assert -1 not in shape, f"{label}: node shape still contains -1: {shape}"
    for spec in graph["model"]["inputSpec"] + graph["model"]["outputSpec"]:
        assert -1 not in spec["shape"], f"{label}: boundary shape still contains -1: {spec['shape']}"


def assert_clean_model_meta(graph: dict, label: str) -> None:
    model = graph["model"]
    assert model["mode"] == "static", f"{label}: not parsed in static mode"
    assert model["warnings"] == [], f"{label}: unexpected warnings {model['warnings']}"
    lowered = [warning.lower() for warning in model["warnings"]]
    assert not any("could not import module" in warning for warning in lowered), label
    assert not any("treated as identity" in warning for warning in lowered), label
    assert not any(
        node.get("notes") and "static analysis fallback" in node["notes"] for node in graph["nodes"]
    ), f"{label}: fallback block notes remain"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@contextlib.contextmanager
def serve_demo():
    """Run ``netprism serve examples/demo`` on a free port and yield its URL."""
    port = _free_port()
    proc = subprocess.Popen(
        [PY, "-m", "netprism", "serve", str(DEMO_DIR), "--port", str(port), "--host", "127.0.0.1"],
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


def post_upload(url: str, files: list[dict], *, execute: bool) -> tuple[int, object]:
    payload = json.dumps({"files": files, "execute": execute}).encode("utf-8")
    request = urllib.request.Request(
        url + "api/upload-parse",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def benchmark_scan(tmp_path_factory: pytest.TempPathFactory) -> dict:
    if not BENCHMARK.is_dir():
        pytest.skip(f"benchmark archive not present: {BENCHMARK}")
    out = tmp_path_factory.mktemp("benchmark_cli_scan")
    proc = run_cli("scan", str(BENCHMARK), "--out", str(out), "--quiet", "--no-execute")
    assert proc.returncode == 0, proc.stderr
    index = load_json(out / "index.json")
    assert index["errors"] == [], f"benchmark scan errors: {index['errors']}"
    graphs: dict[str, dict] = {}
    entries: dict[str, dict] = {}
    for entry in index["models"]:
        graph = load_json(out / entry["file"])
        assert validate_graph(graph) == [], f"{entry['id']}: invalid graph"
        name = graph["model"]["name"]
        graphs[name] = graph
        entries[name] = entry
    return {"index": index, "graphs": graphs, "entries": entries, "out": out}


@pytest.fixture(scope="module")
def demo_server():
    with serve_demo() as url:
        yield url


def _write_model(tmp_path: Path, name: str, body: str) -> Path:
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return path


def _inspect_static(path: Path) -> dict:
    proc = run_cli("inspect", str(path), "--no-execute", "--json")
    assert proc.returncode == 0, proc.stderr
    assert "\ufffd" not in proc.stdout
    return json.loads(proc.stdout)


# --------------------------------------------------------------------------- #
# 1. CLI scan over the five benchmark models
# --------------------------------------------------------------------------- #
def test_cli_scan_five_benchmark_models_zero_warnings_and_balanced_params(
    benchmark_scan: dict,
) -> None:
    graphs = benchmark_scan["graphs"]
    entries = benchmark_scan["entries"]
    assert set(graphs) == set(BENCHMARK_MODELS), f"unexpected benchmark models: {sorted(graphs)}"

    for name, source in BENCHMARK_MODELS.items():
        graph = graphs[name]
        entry = entries[name]
        assert entry["source"] == source
        assert entry["mode"] == "static"
        assert entry["warnings"] == 0
        assert_clean_model_meta(graph, name)
        assert_no_unknown_dimensions(graph, name)
        assert node_params(graph) == graph["model"]["totalParams"], (
            f"{name}: sum(node.numParams) != model.totalParams"
        )
        assert entry["totalParams"] == graph["model"]["totalParams"]
        assert entry["numLayers"] == graph["model"]["numLayers"]


def test_cli_scan_import_failures_do_not_become_warnings(benchmark_scan: dict) -> None:
    """The benchmark modules import ``benchmark.support``/scipy; static mode ignores that."""
    for name, graph in benchmark_scan["graphs"].items():
        warnings = graph["model"]["warnings"]
        assert warnings == [], f"{name}: import/environment issues leaked into warnings: {warnings}"


# --------------------------------------------------------------------------- #
# 2. GAT-specific promises
# --------------------------------------------------------------------------- #
def test_benchmark_gat_resolved_without_fallback_block(benchmark_scan: dict) -> None:
    graph = benchmark_scan["graphs"]["SensorGAT"]
    model = graph["model"]

    assert [(spec["name"], spec["shape"]) for spec in model["inputSpec"]] == [
        ("q_pos", [1, 1, 3]),
        ("s_pos", [1, 1, 3]),
        ("s_val", [1, 1, 3]),
    ]
    assert model["outputSpec"][0]["shape"] == [1, 1, 3]
    assert len(graph["nodes"]) == GAT_NODE_COUNT
    assert model["totalParams"] == GAT_PARAMS
    assert node_params(graph) == GAT_PARAMS

    ops = [node["op"] for node in graph["nodes"]]
    assert "GATLayer" not in ops, "unresolved GATLayer fallback block remains"
    names = [node["name"] for node in graph["nodes"]]
    assert "gat" not in names, "fallback block would duplicate gat.q/k/v parameters"
    assert "gat.out" in names
    assert not any(
        node.get("notes") and "static analysis fallback" in node["notes"] for node in graph["nodes"]
    )

    topk_nodes = [node for node in graph["nodes"] if "topk" in node["op"]]
    assert len(topk_nodes) == 1
    assert len(topk_nodes[0]["outShapes"]) == 2, "topk must expose values + indices"
    assert all(shape and -1 not in shape for shape in topk_nodes[0]["outShapes"])

    gather_nodes = [node for node in graph["nodes"] if "gather" in node["op"]]
    assert len(gather_nodes) == 2, f"expected gather x2 (features + positions), got {len(gather_nodes)}"
    for node in gather_nodes:
        assert node["outShapes"] and all(-1 not in shape for shape in node["outShapes"])

    incoming = {edge["target"] for edge in graph["edges"]}
    out_node = next(node for node in graph["nodes"] if node["name"] == "gat.out")
    assert out_node["id"] in incoming, "gat.out is disconnected from the graph"


# --------------------------------------------------------------------------- #
# 3. FNO spectral block / UNet / Transformer shape promises
# --------------------------------------------------------------------------- #
def test_benchmark_fno_parameter_nodes_and_spectral_shapes(benchmark_scan: dict) -> None:
    graph = benchmark_scan["graphs"]["FNOSensor"]

    parameter_nodes = [node for node in graph["nodes"] if node["op"] == "Parameter"]
    assert len(parameter_nodes) == FNO_PARAM_NODES
    for node in parameter_nodes:
        assert node["numParams"] == FNO_PARAM_SIZE
        assert node["outShapes"] == [[32, 32, 8, 8, 4]]
    assert sum(node["numParams"] for node in parameter_nodes) == FNO_PARAM_NODES * FNO_PARAM_SIZE
    assert node_params(graph) == graph["model"]["totalParams"]

    einsum_nodes = [node for node in graph["nodes"] if node["op"] == "einsum"]
    assert len(einsum_nodes) == 4
    for node in einsum_nodes:
        shape = node["outShapes"][0]
        assert len(shape) == 5, f"einsum output lost its spectral rank: {shape}"
        assert shape[1] == 32, f"einsum output channel dimension should be 32: {shape}"

    irfftn_nodes = [node for node in graph["nodes"] if node["op"] == "irfftn"]
    assert len(irfftn_nodes) == 4
    for node in irfftn_nodes:
        shape = node["outShapes"][0]
        assert len(shape) == 5, f"irfftn output must be 5D: {shape}"
        assert shape[-3:] == [16, 16, 16]


def test_benchmark_unet_and_transformer_are_not_placeholders(benchmark_scan: dict) -> None:
    unet = benchmark_scan["graphs"]["MaskedUNet3D"]
    assert unet["model"]["outputSpec"][0]["shape"] == [1, 3, 16, 16, 16]
    assert unet["model"]["numLayers"] >= 55, "UNet collapsed into a placeholder block"
    assert len(unet["nodes"]) >= 55
    unet_ops = {node["op"] for node in unet["nodes"]}
    assert {"Conv3d", "ConvTranspose3d", "GroupNorm"} <= unet_ops
    assert "MaskedUNet3D" not in unet_ops, "UNet model class emitted as a fallback block"
    assert node_params(unet) == unet["model"]["totalParams"]

    transformer = benchmark_scan["graphs"]["SensorTransformer"]
    transformer_inputs = [(spec["name"], spec["shape"]) for spec in transformer["model"]["inputSpec"]]
    assert [name for name, _shape in transformer_inputs] == ["q_pos", "s_pos", "s_val"]
    q_pos_shape = dict(transformer_inputs)["q_pos"]
    assert len(q_pos_shape) == 3, f"q_pos should be 3D (B, N, 3), got {q_pos_shape}"
    assert q_pos_shape[-1] == 3
    attention_nodes = [node for node in transformer["nodes"] if node["op"] == "MultiheadAttention"]
    assert len(attention_nodes) == 3, "expected 3 cross-attention blocks"
    assert node_params(transformer) == transformer["model"]["totalParams"]


# --------------------------------------------------------------------------- #
# 4. Generic unknown functions must never degrade to identity
# --------------------------------------------------------------------------- #
def test_unknown_tensor_function_is_custom_with_unknown_shape(tmp_path: Path) -> None:
    single = _write_model(
        tmp_path,
        "single_unknown.py",
        """
import torch
import torch.nn as nn


class SingleUnknown(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(6, 6)

    def forward(self, x):
        x = self.fc(x)
        return torch.nonstandard_op(x)
""",
    )
    graph = _inspect_static(single)
    model = graph["model"]
    assert model["mode"] == "static"
    warnings = model["warnings"]
    assert any("shape unknown for 'torch.nonstandard_op'" in warning for warning in warnings)
    assert not any("treated as identity" in warning for warning in warnings)
    assert not any("could not import module" in warning.lower() for warning in warnings)

    nodes = [node for node in graph["nodes"] if "nonstandard_op" in node["op"]]
    assert nodes, "unknown function node missing from the graph"
    assert nodes[0]["kind"] == "custom"
    assert nodes[0]["outShapes"] == [[-1, -1]]
    assert model["outputSpec"][0]["shape"] == [-1, -1]

    binary = _write_model(
        tmp_path,
        "binary_unknown.py",
        """
import torch
import torch.nn as nn


class BinaryUnknown(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(6, 6)

    def forward(self, x, y):
        return torch.nonstandard_op(self.fc(x), self.fc(y))
""",
    )
    graph = _inspect_static(binary)
    warnings = graph["model"]["warnings"]
    assert any("shape unknown for 'torch.nonstandard_op'" in warning for warning in warnings)
    assert not any("treated as identity" in warning for warning in warnings)
    nodes = [node for node in graph["nodes"] if "nonstandard_op" in node["op"]]
    assert nodes and nodes[0]["kind"] == "custom"
    assert [spec["name"] for spec in graph["model"]["inputSpec"]] == ["x", "y"]
    # Compatible input shapes may still be propagated; the key promise is that
    # the op stays a custom node (never substituted by its input tensor).
    output_shape = graph["model"]["outputSpec"][0]["shape"]
    assert output_shape and all(dim >= 0 or dim == -1 for dim in output_shape)


def test_empty_module_does_not_crash_static_parser(tmp_path: Path) -> None:
    empty = _write_model(
        tmp_path,
        "empty_net.py",
        """
import torch.nn as nn


class EmptyNet(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
""",
    )
    graph = _inspect_static(empty)
    assert graph["model"]["mode"] == "static"
    assert validate_graph(graph) == []
    assert graph["model"]["inputSpec"] and graph["model"]["outputSpec"]


# --------------------------------------------------------------------------- #
# 5. Web upload end-to-end (/api/upload-parse)
# --------------------------------------------------------------------------- #
def test_http_upload_gat_directory_has_zero_warnings(demo_server: str) -> None:
    gat_source = (BENCHMARK / "gat" / "model.py")
    if not gat_source.is_file():
        pytest.skip(f"benchmark GAT model not present: {gat_source}")
    files = [
        {"path": "gat/model.py", "content": gat_source.read_text(encoding="utf-8")},
        # Non-python uploads must be ignored by the endpoint, not error out.
        {"path": "gat/config.yaml", "content": "ignored: true\n"},
    ]
    status, body = post_upload(demo_server, files, execute=False)
    assert status == 200, body
    assert isinstance(body, dict)
    assert body["errors"] == []

    graphs = body["graphs"]
    assert graphs, "upload-parse returned no graphs"
    gat_graphs = [graph for graph in graphs.values() if graph["model"]["name"] == "SensorGAT"]
    assert len(gat_graphs) == 1, f"expected exactly one SensorGAT graph, got {len(gat_graphs)}"
    graph = gat_graphs[0]
    model = graph["model"]
    assert validate_graph(graph) == []
    assert model["mode"] == "static"
    assert model["warnings"] == [], f"upload warnings: {model['warnings']}"
    assert [(spec["name"], spec["shape"]) for spec in model["inputSpec"]] == [
        ("q_pos", [1, 1, 3]),
        ("s_pos", [1, 1, 3]),
        ("s_val", [1, 1, 3]),
    ]
    assert len(graph["nodes"]) == GAT_NODE_COUNT
    assert model["totalParams"] == GAT_PARAMS
    assert node_params(graph) == GAT_PARAMS
    assert model["source"]["relPath"] == "gat/model.py"
    assert not any(node["op"] == "GATLayer" for node in graph["nodes"])

    index_entries = [entry for entry in body["index"]["models"] if entry["name"] == "SensorGAT"]
    assert len(index_entries) == 1
    assert index_entries[0]["warnings"] == 0
    assert index_entries[0]["totalParams"] == GAT_PARAMS


def test_http_upload_execute_false_never_runs_code(demo_server: str, tmp_path: Path) -> None:
    marker = tmp_path / "upload_executed.txt"
    source = (
        "from pathlib import Path\n"
        f"Path(r'{marker}').write_text('executed', encoding='utf-8')\n\n"
        "import torch.nn as nn\n\n"
        "class UploadProbe(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.fc = nn.Linear(4, 2)\n"
        "    def forward(self, x):\n"
        "        return self.fc(x)\n"
    )
    status, body = post_upload(
        demo_server, [{"path": "probe/model.py", "content": source}], execute=False
    )
    assert status == 200, body
    assert body["errors"] == []
    assert not marker.exists(), "upload-parse executed model code despite execute=false"
    probes = [graph for graph in body["graphs"].values() if graph["model"]["name"] == "UploadProbe"]
    assert len(probes) == 1
    assert probes[0]["model"]["mode"] == "static"
    assert probes[0]["model"]["warnings"] == []

    # An execute=true request must be rejected by a server started without
    # --allow-execute, and must still not run the uploaded code.
    status, _body = post_upload(
        demo_server, [{"path": "probe/model.py", "content": source}], execute=True
    )
    assert status == 403, f"execute=true should be rejected, got HTTP {status}"
    assert not marker.exists(), "execute=true request ran code without --allow-execute"
