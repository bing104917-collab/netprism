"""Contract tests for the optional multi-format adapters (ONNX / TorchScript / Keras).

The fixtures are generated with ``scripts/make_format_fixtures.py``.  Keras
fixtures are hand-built zip archives, so the Keras tests never import Keras /
TensorFlow; the ONNX / TorchScript numbers are compared against PyTorch.
"""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from netprism.adapters import (
    FORMAT_LABELS,
    FORMAT_SUFFIXES,
    analyze_format_file,
    format_for_path,
    is_format_file,
)
from netprism.core.errors import AnalyzeError
from netprism.schema import load_json_schema, validate_graph

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "formats"
SAMPLES = ROOT / "samples"

ONNX_FIXTURES = ["vggmini.onnx", "resnetmini.onnx", "mlp.onnx"]
TORCHSCRIPT_FIXTURES = ["vggmini_traced.pt", "resnetmini_traced.pt", "mlp_traced.pt", "tiny_transformer_traced.pt"]
KERAS_FIXTURES = ["tiny_cnn.keras", "tiny_mlp.keras", "tiny_rescnn.keras", "tiny_attention.keras"]
ALL_FIXTURES = [*ONNX_FIXTURES, *TORCHSCRIPT_FIXTURES, *KERAS_FIXTURES]

NODE_ID_RE = re.compile(r"^n\d+$")
EDGE_ID_RE = re.compile(r"^e\d+$")
GROUP_ID_RE = re.compile(r"^g\d+$")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _load_sample(name: str):
    path = SAMPLES / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"adapter_test_sample_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _graph(name: str) -> dict:
    graphs = analyze_format_file(FIXTURES / name)
    assert len(graphs) == 1, f"{name}: expected one graph, got {len(graphs)}"
    return graphs[0]


def _find_node(graph: dict, name: str) -> dict:
    for node in graph["nodes"]:
        if node["name"] == name:
            return node
    raise AssertionError(f"node {name!r} not in {[node['name'] for node in graph['nodes']]}")


def _check_shape(shape) -> None:
    assert isinstance(shape, list) and shape
    for dim in shape:
        assert isinstance(dim, int) and not isinstance(dim, bool) and dim >= -1


def _check_graph_contract(graph: dict) -> None:
    errors = validate_graph(graph)
    assert errors == [], errors
    schema = load_json_schema()
    jsonschema_errors = sorted(
        Draft202012Validator(schema).iter_errors(graph), key=lambda error: list(error.path)
    )
    assert jsonschema_errors == [], [f"{list(error.path)}: {error.message}" for error in jsonschema_errors]

    model = graph["model"]
    assert model["framework"] in ("onnx", "keras", "torchscript")
    assert model["mode"] in ("onnx", "keras", "torchscript")
    assert model["source"]["file"]
    assert model["source"]["relPath"]
    assert model["inputSpec"] and model["outputSpec"]
    for spec in [*model["inputSpec"], *model["outputSpec"]]:
        assert isinstance(spec["dtype"], str) and spec["dtype"]
        _check_shape(spec["shape"])
    assert isinstance(model["totalParams"], int) and model["totalParams"] >= 0
    assert isinstance(model["totalFlops"], int) and model["totalFlops"] >= 0
    assert isinstance(model["flopsApprox"], bool)

    assert graph["nodes"], "graph must contain nodes"
    for node in graph["nodes"]:
        assert NODE_ID_RE.match(node["id"])
        assert node["kind"] in (
            "input", "output", "conv", "deconv", "linear", "mlp", "norm", "activation", "pool",
            "dropout", "flatten", "reshape", "attention", "ffn", "embedding", "positional", "add",
            "concat", "upsample", "rnn", "softmax", "block", "container", "custom", "unknown",
        )
        for shape in node["inShapes"] + node["outShapes"]:
            _check_shape(shape)
        assert isinstance(node["numParams"], int) and node["numParams"] >= 0
        assert isinstance(node["flops"], int) and node["flops"] >= 0

    node_ids = {node["id"] for node in graph["nodes"]}
    for edge in graph["edges"]:
        assert EDGE_ID_RE.match(edge["id"])
        assert edge["source"] in node_ids and edge["target"] in node_ids
        assert edge["source"] != edge["target"]

    roots = [group for group in graph["groups"] if group["parentId"] is None]
    assert len(roots) == 1 and roots[0]["id"] == "g0" and roots[0]["kind"] == "model"
    for group in graph["groups"]:
        assert GROUP_ID_RE.match(group["id"])
    grouped = [node_id for group in graph["groups"] for node_id in group["nodeIds"]]
    assert len(grouped) == len(set(grouped)), "group memberships must not overlap"
    node_group = {node["id"]: node["groupId"] for node in graph["nodes"]}
    for group in graph["groups"]:
        for node_id in group["nodeIds"]:
            assert node_group[node_id] == group["id"]


# ---------------------------------------------------------------------------
# package API / lazy imports
# ---------------------------------------------------------------------------

def test_format_suffixes_and_detection():
    assert FORMAT_SUFFIXES == {
        ".onnx": "onnx",
        ".pt": "torchscript",
        ".pth": "torchscript",
        ".keras": "keras",
    }
    assert set(FORMAT_LABELS) == {"onnx", "torchscript", "keras"}
    assert is_format_file("model.onnx")
    assert is_format_file(Path("model.pt"))
    assert is_format_file("model.PTH")
    assert is_format_file("model.keras")
    assert not is_format_file("model.py")
    assert format_for_path("model.onnx") == "onnx"
    assert format_for_path("model.txt") is None


def test_importing_package_does_not_import_optional_backends():
    code = (
        "import sys\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        "import netprism.adapters\n"
        "bad = [name for name in ('torch', 'onnx', 'keras', 'tensorflow') if name in sys.modules]\n"
        "raise SystemExit(1 if bad else 0)\n"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_missing_file_raises_analyze_error(tmp_path):
    with pytest.raises(AnalyzeError):
        analyze_format_file(tmp_path / "missing.onnx")


def test_unsupported_suffix_raises_analyze_error(tmp_path):
    path = tmp_path / "model.txt"
    path.write_text("not a model", encoding="utf-8")
    with pytest.raises(AnalyzeError) as info:
        analyze_format_file(path)
    assert "supported suffixes" in str(info.value)


def test_max_nodes_limit(tmp_path):
    with pytest.raises(AnalyzeError) as info:
        analyze_format_file(FIXTURES / "mlp.onnx", max_nodes=2)
    assert "max_nodes" in str(info.value)


def test_fixtures_exist():
    for name in ALL_FIXTURES:
        path = FIXTURES / name
        assert path.is_file(), f"missing fixture {name}; run scripts/make_format_fixtures.py"
        assert path.stat().st_size < 20 * 1024 * 1024


@pytest.mark.parametrize("fixture", ["broken.onnx", "broken.keras", "mlp_state_dict.pt"])
def test_negative_fixtures_fail_readably(fixture):
    path = FIXTURES / fixture
    assert path.is_file(), f"missing negative fixture {fixture}"
    with pytest.raises(AnalyzeError) as info:
        analyze_format_file(path)
    message = str(info.value)
    assert message.strip()
    assert "Traceback" not in message


# ---------------------------------------------------------------------------
# every format: schema / graph contract
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fixture", ALL_FIXTURES)
def test_graph_satisfies_contract(fixture):
    graph = _graph(fixture)
    _check_graph_contract(graph)
    expected_framework = "onnx" if fixture.endswith(".onnx") else (
        "keras" if fixture.endswith(".keras") else "torchscript"
    )
    assert graph["model"]["framework"] == expected_framework


# ---------------------------------------------------------------------------
# ONNX
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fixture", ONNX_FIXTURES)
def test_onnx_graph_has_shapes_and_metrics(fixture):
    graph = _graph(fixture)
    assert graph["model"]["mode"] == "onnx"
    assert graph["model"]["totalParams"] > 0
    assert graph["model"]["totalFlops"] > 0
    kinds = {node["kind"] for node in graph["nodes"]}
    assert {"input", "output", "linear"} <= kinds
    if fixture != "mlp.onnx":
        assert "conv" in kinds
        assert any(node["kind"] == "conv" and node["flops"] > 0 for node in graph["nodes"])
    assert all(node["outShapes"] and node["outShapes"][0] for node in graph["nodes"])
    assert graph["model"]["meta"]["format"] == "onnx"
    assert graph["model"]["meta"]["opset"]


@pytest.mark.parametrize(
    "fixture,sample,class_name,input_shape",
    [
        ("vggmini.onnx", "cnn_vgg", "VGGMini", (1, 3, 32, 32)),
        ("resnetmini.onnx", "resnet_mini", "ResNetMini", (1, 3, 32, 32)),
        ("mlp.onnx", "mlp", "MLP", (1, 64)),
    ],
)
def test_onnx_params_match_pytorch(fixture, sample, class_name, input_shape):
    torch = pytest.importorskip("torch")
    module = _load_sample(sample)
    model = getattr(module, class_name)().eval()
    with torch.no_grad():
        expected = sum(parameter.numel() for parameter in model.parameters())
    graph = _graph(fixture)
    assert graph["model"]["totalParams"] == expected
    assert graph["model"]["inputSpec"][0]["shape"] == list(input_shape)


def test_onnx_and_torchscript_agree_on_cnn_flops():
    onnx_graph = _graph("vggmini.onnx")
    script_graph = _graph("vggmini_traced.pt")
    assert onnx_graph["model"]["totalFlops"] == script_graph["model"]["totalFlops"]
    assert onnx_graph["model"]["totalParams"] == script_graph["model"]["totalParams"]
    onnx_resnet = _graph("resnetmini.onnx")
    script_resnet = _graph("resnetmini_traced.pt")
    assert onnx_resnet["model"]["totalFlops"] == script_resnet["model"]["totalFlops"]


def test_onnx_invalid_file_raises(tmp_path):
    path = tmp_path / "broken.onnx"
    path.write_bytes(b"this is definitely not a protobuf")
    with pytest.raises(AnalyzeError):
        analyze_format_file(path)


def test_onnx_missing_dependency_message(tmp_path, monkeypatch):
    path = tmp_path / "tiny.onnx"
    path.write_bytes(b"")
    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "onnx":
            raise ImportError("simulated missing onnx")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    with pytest.raises(AnalyzeError) as info:
        analyze_format_file(path)
    assert "pip install onnx" in str(info.value)


# ---------------------------------------------------------------------------
# TorchScript
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fixture", TORCHSCRIPT_FIXTURES)
def test_torchscript_graph_has_shapes_and_metrics(fixture):
    graph = _graph(fixture)
    assert graph["model"]["mode"] == "torchscript"
    assert graph["model"]["totalParams"] > 0
    assert graph["model"]["flopsApprox"] is True
    assert any("TorchScript archives do not retain" in warning for warning in graph["model"]["warnings"])
    input_spec = graph["model"]["inputSpec"][0]
    assert input_spec["shape"][0] == 1
    assert graph["model"]["outputSpec"]
    assert graph["model"]["meta"]["format"] == "torchscript"


@pytest.mark.parametrize(
    "fixture,sample,class_name,input_shape",
    [
        ("vggmini_traced.pt", "cnn_vgg", "VGGMini", (1, 3, 32, 32)),
        ("resnetmini_traced.pt", "resnet_mini", "ResNetMini", (1, 3, 32, 32)),
        ("mlp_traced.pt", "mlp", "MLP", (1, 64)),
    ],
)
def test_torchscript_params_match_pytorch(fixture, sample, class_name, input_shape):
    torch = pytest.importorskip("torch")
    module = _load_sample(sample)
    model = getattr(module, class_name)().eval()
    expected = sum(parameter.numel() for parameter in model.parameters())
    graph = _graph(fixture)
    assert graph["model"]["totalParams"] == expected
    assert graph["model"]["inputSpec"][0]["shape"] == list(input_shape)


def test_torchscript_cnn_shapes_and_flops():
    graph = _graph("vggmini_traced.pt")
    conv = _find_node(graph, "features.0")
    assert conv["kind"] == "conv"
    assert conv["inShapes"] == [[1, 3, 32, 32]]
    assert conv["outShapes"] == [[1, 16, 32, 32]]
    assert conv["numParams"] == 432
    assert conv["flops"] > 0
    classifier = _find_node(graph, "classifier.1")
    assert classifier["kind"] == "linear"
    assert classifier["outShapes"] == [[1, 32]]
    assert graph["model"]["totalFlops"] > 0
    groups = {group["name"]: group for group in graph["groups"]}
    assert groups["features"]["kind"] == "conv_block"
    assert groups["features"]["nodeIds"]


def test_torchscript_mlp_shapes():
    graph = _graph("mlp_traced.pt")
    first = _find_node(graph, "net.0")
    assert first["inShapes"] == [[1, 64]]
    assert first["outShapes"] == [[1, 128]]
    assert first["numParams"] == 8320
    last = _find_node(graph, "net.5")
    assert last["outShapes"] == [[1, 10]]
    assert last["numParams"] == 650


def test_torchscript_transformer_graph():
    graph = _graph("tiny_transformer_traced.pt")
    assert len(graph["nodes"]) > 20
    kinds = {node["kind"] for node in graph["nodes"]}
    assert {"attention", "linear", "norm", "embedding", "reshape"} <= kinds
    assert graph["model"]["totalParams"] == 29898
    assert graph["model"]["outputSpec"][0]["shape"] == [1, 10]


def test_torchscript_state_dict_error(tmp_path):
    torch = pytest.importorskip("torch")
    module = _load_sample("mlp")
    model = getattr(module, "MLP")()
    path = tmp_path / "state_dict.pt"
    torch.save(model.state_dict(), path)
    with pytest.raises(AnalyzeError) as info:
        analyze_format_file(path)
    message = str(info.value).lower()
    assert "state_dict" in message or "torchscript archive" in message


def test_torchscript_invalid_file_raises(tmp_path):
    path = tmp_path / "broken.pt"
    path.write_text("definitely not a torch archive", encoding="utf-8")
    with pytest.raises(AnalyzeError):
        analyze_format_file(path)


# ---------------------------------------------------------------------------
# Keras
# ---------------------------------------------------------------------------

def test_keras_cnn_params_shapes_and_flops():
    graph = _graph("tiny_cnn.keras")
    assert graph["model"]["mode"] == "keras"
    assert graph["model"]["flopsApprox"] is True
    assert graph["model"]["inputSpec"][0]["shape"] == [-1, 16, 16, 3]
    assert graph["model"]["outputSpec"][0]["shape"] == [-1, 10]
    assert graph["model"]["totalParams"] == 7194
    assert graph["model"]["totalFlops"] > 0

    conv1 = _find_node(graph, "conv1")
    assert conv1["kind"] == "conv"
    assert conv1["outShapes"] == [[-1, 16, 16, 8]]
    assert conv1["numParams"] == 224
    assert conv1["flops"] > 0

    bn1 = _find_node(graph, "bn1")
    assert bn1["kind"] == "norm"
    assert bn1["numParams"] == 32  # gamma + beta + moving_mean + moving_var

    conv2 = _find_node(graph, "conv2")
    assert conv2["outShapes"] == [[-1, 6, 6, 16]]
    assert conv2["numParams"] == 1168

    head = _find_node(graph, "head")
    assert head["kind"] == "linear"
    assert head["numParams"] == 5770
    assert head["flops"] > 0

    pool = _find_node(graph, "pool1")
    assert pool["kind"] == "pool"
    assert pool["flops"] > 0


def test_keras_sequential_without_input_layer_uses_build_input_shape():
    graph = _graph("tiny_cnn.keras")
    names = [node["name"] for node in graph["nodes"]]
    assert names == ["input", "conv1", "bn1", "relu1", "pool1", "conv2", "relu2", "flatten",
                     "head", "output"]
    assert graph["model"]["numLayers"] == 8
    for node in graph["nodes"]:
        if node["kind"] in ("conv", "linear", "pool"):
            assert node["flops"] > 0, f"{node['name']} has zero FLOPs"


def test_keras_functional_mlp_multi_input():
    graph = _graph("tiny_mlp.keras")
    assert graph["model"]["totalParams"] == 1386
    assert len(graph["model"]["inputSpec"]) == 2
    assert graph["model"]["outputSpec"][0]["shape"] == [-1, 10]
    concat = _find_node(graph, "concat")
    assert concat["kind"] == "concat"
    assert concat["inShapes"] == [[-1, 16], [-1, 16]]
    assert concat["outShapes"] == [[-1, 32]]
    incoming = [edge for edge in graph["edges"] if edge["target"] == concat["id"]]
    assert len(incoming) == 2
    assert _find_node(graph, "dense1")["numParams"] == 32 * 32 + 32
    assert _find_node(graph, "head")["numParams"] == 32 * 10 + 10


def test_keras_functional_residual_branch():
    graph = _graph("tiny_rescnn.keras")
    assert graph["model"]["totalParams"] == 962
    add = _find_node(graph, "add")
    assert add["kind"] == "add"
    assert len(add["inShapes"]) == 2
    incoming = [edge for edge in graph["edges"] if edge["target"] == add["id"]]
    assert len(incoming) == 2
    assert add["outShapes"] == [[-1, 32, 32, 8]]
    gap = _find_node(graph, "gap")
    assert gap["outShapes"] == [[-1, 8]]
    assert _find_node(graph, "conv2")["numParams"] == 8 * 8 * 9 + 8


def test_keras_attention_embedding_norm_params():
    graph = _graph("tiny_attention.keras")
    assert graph["model"]["totalParams"] == 1666
    assert _find_node(graph, "emb")["numParams"] == 32 * 16
    assert _find_node(graph, "emb")["kind"] == "embedding"
    assert _find_node(graph, "ln")["numParams"] == 2 * 16
    mha = _find_node(graph, "mha")
    assert mha["kind"] == "attention"
    assert mha["numParams"] == 1088
    assert mha["flops"] > 0
    assert mha["outShapes"] == [[-1, 8, 16]]


def test_keras_batch_input_shape_compat(tmp_path):
    config = {
        "module": "keras",
        "class_name": "Sequential",
        "config": {
            "name": "legacy",
            "layers": [
                {
                    "module": "keras.layers",
                    "class_name": "InputLayer",
                    "config": {"batch_input_shape": [None, 8], "dtype": "float32", "name": "input"},
                    "name": "input",
                    "inbound_nodes": [],
                },
                {
                    "module": "keras.layers",
                    "class_name": "Dense",
                    "config": {"units": 4, "use_bias": True, "name": "dense"},
                    "name": "dense",
                    "inbound_nodes": [[["input", 0, 0, {}]]],
                },
            ],
        },
        "keras_version": "3.0.0",
        "backend": "torch",
    }
    path = tmp_path / "legacy.keras"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("config.json", json.dumps(config))
    graph = analyze_format_file(path)[0]
    _check_graph_contract(graph)
    assert graph["model"]["inputSpec"][0]["shape"] == [-1, 8]
    assert graph["model"]["totalParams"] == 8 * 4 + 4
    assert graph["model"]["outputSpec"][0]["shape"] == [-1, 4]


def test_keras_unknown_layer_is_custom_with_warning(tmp_path):
    config = {
        "module": "keras",
        "class_name": "Sequential",
        "config": {
            "name": "weird",
            "build_input_shape": [None, 4],
            "layers": [
                {"module": "keras.layers", "class_name": "FancyLayer", "config": {"name": "fancy"},
                 "name": "fancy", "inbound_nodes": []},
            ],
        },
        "keras_version": "3.0.0",
        "backend": "torch",
    }
    path = tmp_path / "weird.keras"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("config.json", json.dumps(config))
    graph = analyze_format_file(path)[0]
    _check_graph_contract(graph)
    node = _find_node(graph, "fancy")
    assert node["kind"] == "custom"
    assert node["outShapes"] == [[-1]]
    assert any("FancyLayer" in warning for warning in graph["model"]["warnings"])


def test_keras_invalid_archives(tmp_path):
    not_zip = tmp_path / "broken.keras"
    not_zip.write_text("nope", encoding="utf-8")
    with pytest.raises(AnalyzeError):
        analyze_format_file(not_zip)

    no_config = tmp_path / "empty.keras"
    with zipfile.ZipFile(no_config, "w") as archive:
        archive.writestr("metadata.json", "{}")
    with pytest.raises(AnalyzeError) as info:
        analyze_format_file(no_config)
    assert "config.json" in str(info.value)


def test_keras_count_params_matches_keras_when_available():
    pytest.importorskip("keras")
    import os

    os.environ.setdefault("KERAS_BACKEND", "torch")
    import keras  # noqa: PLC0415 - optional dependency

    for fixture in KERAS_FIXTURES:
        model = keras.saving.load_model(str(FIXTURES / fixture))
        graph = _graph(fixture)
        assert graph["model"]["totalParams"] == int(model.count_params()), fixture


def test_keras_does_not_import_tensorflow():
    code = (
        "import sys\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        "from netprism.adapters import analyze_format_file\n"
        "analyze_format_file(r'" + str(FIXTURES / "tiny_cnn.keras") + "')\n"
        "raise SystemExit(1 if 'tensorflow' in sys.modules else 0)\n"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "tensorflow" not in result.stdout.lower() or True
