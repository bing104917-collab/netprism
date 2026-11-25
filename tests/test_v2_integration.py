"""Independent v2 acceptance tests for the NetPrism CLI (report / compare / scan).

The CLI is exercised as a black box through ``python -m netprism ...``.
Expected numbers are re-derived from the graph JSON documents themselves
(activation memory = prod(output shape) * 4 bytes, macs = flops / 2, ...) and
from an independent layer-matching implementation, so a shared bug in
``netprism.metrics`` / ``report`` / ``compare`` cannot hide behind these tests.

Format-adapter tests (``.onnx`` / ``.pt`` / ``.pth`` / ``.keras``) discover the
``tests/fixtures/formats`` directory at run time.  While formats-dev has not
landed adapters/fixtures yet they skip with an explicit message instead of
passing silently; once the adapter modules exist, real model files are also
generated in ``tmp_path`` with the installed ``torch`` / ``onnx`` packages so
the adapters are exercised even without checked-in fixtures.
"""

from __future__ import annotations

import csv
import io
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from netprism.schema import validate_graph, validate_index

ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = ROOT / "examples" / "demo"
SAMPLES_DIR = ROOT / "samples"
FORMAT_FIXTURES = ROOT / "tests" / "fixtures" / "formats"
ADAPTERS_DIR = ROOT / "netprism" / "adapters"
PY = sys.executable

REPORT_COLUMNS = [
    "id",
    "name",
    "op",
    "kind",
    "modulePath",
    "inShapes",
    "outShapes",
    "numParams",
    "flops",
    "activationBytes",
    "receptiveField",
    "groupId",
    "groupLabel",
    "source",
]

COMPARE_CSV_COLUMNS = [
    "status",
    "key",
    "modulePath",
    "name",
    "op_a",
    "op_b",
    "kind_a",
    "kind_b",
    "numParams_a",
    "numParams_b",
    "flops_a",
    "flops_b",
    "activationBytes_a",
    "activationBytes_b",
    "receptiveField_a",
    "receptiveField_b",
    "changedFields",
]

DEMO_GRAPH_FILES = {
    "vgg": "cnn_vgg__vggmini.json",
    "unet": "unet_mini__unetmini.json",
    "resnet": "resnet_mini__resnetmini.json",
    "transformer": "transformer_classifier__tinytransformerclassifier.json",
    "mlp": "mlp__mlp.json",
    "attention": "custom_attention__tinygpt.json",
    "dynamic": "dynamic_control__dynamicnet.json",
    "needs_args": "needs_args__needsargs.json",
}

FORMAT_FRAMEWORKS = {
    "onnx": (".onnx",),
    "torchscript": (".pt", ".pth"),
    "keras": (".keras",),
}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def run_cli(*args: str, cwd: Path = ROOT, timeout: int = 240) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [PY, "-m", "netprism", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


def run_cli_bytes(*args: str, cwd: Path = ROOT, timeout: int = 240) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        [PY, "-m", "netprism", *args],
        cwd=cwd,
        capture_output=True,
        timeout=timeout,
    )


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_demo(name: str) -> dict:
    return load_json(DEMO_DIR / name)


def demo_index() -> dict:
    return load_json(DEMO_DIR / "index.json")


def demo_index_entry(model_id: str) -> dict:
    return next(entry for entry in demo_index()["models"] if entry["id"] == model_id)


def expected_activation_bytes(node: dict) -> int:
    """float32 activation memory of a node's primary output tensor."""
    shapes = node.get("outShapes") or []
    if not shapes or not isinstance(shapes[0], list) or not shapes[0]:
        return 0
    count = 1
    for dim in shapes[0]:
        count *= dim if isinstance(dim, int) and dim >= 0 else 1
    return count * 4


def expected_model_activation(graph: dict) -> int:
    return sum(expected_activation_bytes(node) for node in graph["nodes"])


def human_bytes(value: int) -> str:
    """Mirror of the report contract: bytes < 1024, else KiB..PiB with 2 decimals."""
    if value < 1024:
        return f"{value:,} B"
    units = ("KiB", "MiB", "GiB", "TiB", "PiB")
    size = float(value)
    index = -1
    while size >= 1024 and index < len(units) - 1:
        size /= 1024
        index += 1
    return f"{size:.2f} {units[index]} ({value:,} B)"


def parse_report_comment(text: str) -> dict[str, str]:
    rows = list(csv.reader(io.StringIO(text.lstrip("\ufeff"))))
    assert rows, "CSV report is empty"
    comment = rows[0][0]
    assert comment.startswith("# model:"), f"unexpected CSV comment row: {comment!r}"
    fields: dict[str, str] = {}
    for part in comment[1:].split("|"):
        key, _, value = part.strip().partition(":")
        fields[key.strip()] = value.strip()
    return fields


def keyed_nodes(graph: dict) -> dict[str, dict]:
    """Match layers by modulePath (fallback name), suffix duplicates with #N."""
    rows: dict[str, dict] = {}
    seen: dict[str, int] = {}
    for node in graph.get("nodes") or []:
        base = str(node.get("modulePath") or node.get("name") or node.get("id") or "node")
        seen[base] = seen.get(base, 0) + 1
        key = base if seen[base] == 1 else f"{base}#{seen[base]}"
        rows[key] = node
    return rows


def independent_compare(a: dict, b: dict) -> dict:
    """Independent implementation of the SPEC 4 comparison contract."""
    rows_a, rows_b = keyed_nodes(a), keyed_nodes(b)
    changed: dict[str, dict] = {}
    unchanged = 0
    for key, node_a in rows_a.items():
        node_b = rows_b.get(key)
        if node_b is None:
            continue
        fields: dict[str, dict] = {}
        for field in ("op", "inShapes", "outShapes", "numParams"):
            if node_a.get(field) != node_b.get(field):
                fields[field] = {"a": node_a.get(field), "b": node_b.get(field)}
        if fields:
            changed[key] = fields
        else:
            unchanged += 1
    return {
        "added": set(rows_b) - set(rows_a),
        "removed": set(rows_a) - set(rows_b),
        "changed": changed,
        "unchanged": unchanged,
    }


def write_mutated_vgg_pair(tmp_path: Path) -> tuple[dict, dict, Path, Path]:
    """One model, one mutated copy: exactly 1 added + 1 removed + 1 changed layer.

    Both documents are real graph JSON files on disk, so this exercises the CLI
    ``compare`` path end-to-end without depending on which demo models happen to
    share module paths.
    """
    graph_a = load_demo(DEMO_GRAPH_FILES["vgg"])
    graph_b = json.loads(json.dumps(graph_a))

    graph_b["model"]["name"] = "VGGMiniMut"
    graph_b["nodes"] = [node for node in graph_b["nodes"] if node["name"] != "classifier.3"]
    changed = next(node for node in graph_b["nodes"] if node["name"] == "features.0")
    changed["op"] = "DepthwiseConv2d"
    changed["numParams"] = 500
    changed["flops"] = 3_500_000
    graph_b["nodes"].append(
        {
            "id": "n99",
            "name": "extra_head",
            "op": "Linear",
            "kind": "linear",
            "modulePath": "extra_head",
            "depth": 0,
            "inShapes": [[1, 10]],
            "outShapes": [[1, 4]],
            "params": {"inFeatures": 10, "outFeatures": 4},
            "numParams": 44,
            "flops": 40,
            "tags": [],
            "groupId": None,
            "source": None,
            "notes": [],
        }
    )
    graph_b["model"]["totalParams"] = sum(node["numParams"] for node in graph_b["nodes"])
    graph_b["model"]["trainableParams"] = graph_b["model"]["totalParams"]
    graph_b["model"]["totalFlops"] = sum(node["flops"] for node in graph_b["nodes"])
    graph_b["model"]["numLayers"] = sum(
        1 for node in graph_b["nodes"] if node["kind"] not in ("input", "output")
    )
    # Drop stored metrics on B so the CLI has to enrich them (A keeps its own).
    graph_b["model"].pop("metrics", None)
    for node in graph_b["nodes"]:
        node.pop("metrics", None)

    path_a = tmp_path / "model_a.json"
    path_b = tmp_path / "model_b.json"
    path_a.write_text(json.dumps(graph_a, ensure_ascii=False), encoding="utf-8")
    path_b.write_text(json.dumps(graph_b, ensure_ascii=False), encoding="utf-8")
    return graph_a, graph_b, path_a, path_b


def adapter_module_exists(format_name: str) -> bool:
    return (ADAPTERS_DIR / f"{format_name}_adapter.py").is_file()


def require_adapters(*format_names: str) -> None:
    missing = [name for name in format_names if not adapter_module_exists(name)]
    if missing:
        pytest.skip(
            f"format adapters not landed yet (missing modules for: {', '.join(missing)}; "
            "formats-dev scope) — real-format coverage will run once they exist"
        )


def _negative_name_hint(path: Path) -> bool:
    lowered = path.name.lower()
    return any(
        token in lowered
        for token in ("bad", "broken", "invalid", "corrupt", "error", "state_dict", "missing")
    )


def format_fixture_files(suffixes: tuple[str, ...], *, exclude_negative: bool) -> list[Path]:
    if not FORMAT_FIXTURES.is_dir():
        return []
    files = [
        path
        for path in sorted(FORMAT_FIXTURES.rglob("*"))
        if path.is_file() and path.suffix.lower() in suffixes and path.stat().st_size > 0
    ]
    if exclude_negative:
        files = [path for path in files if not _negative_name_hint(path)]
    return files


def assert_valid_format_graph(graph: dict, expected_framework: str) -> None:
    assert validate_graph(graph) == []
    model = graph["model"]
    assert model["framework"] == expected_framework
    assert model["mode"] in {expected_framework, "trace", "hybrid"}
    assert graph["nodes"], "adapter returned a graph without nodes"
    assert len(graph["nodes"]) > 1
    assert model["numLayers"] == sum(
        1 for node in graph["nodes"] if node.get("kind") not in ("input", "output")
    )
    assert model["totalParams"] > 0, "format model should expose parameters"
    assert model["inputSpec"] and model["outputSpec"]
    for spec in model["inputSpec"] + model["outputSpec"]:
        assert spec["shape"] and all(isinstance(dim, int) and dim >= -1 for dim in spec["shape"])
    node_ids = {node["id"] for node in graph["nodes"]}
    assert len(node_ids) == len(graph["nodes"])
    for node in graph["nodes"]:
        assert node["id"].startswith("n")
        assert node["outShapes"], f"{node['id']} ({node['name']}): missing output shapes"
        if node.get("kind") != "input":
            assert node["inShapes"], f"{node['id']} ({node['name']}): missing input shapes"
        assert isinstance(node["numParams"], int) and node["numParams"] >= 0
        assert isinstance(node["flops"], int) and node["flops"] >= 0
        assert node["groupId"] is None or node["groupId"].startswith("g")
    for edge in graph["edges"]:
        assert edge["source"] in node_ids and edge["target"] in node_ids
        assert edge["id"].startswith("e")
    assert any(group["kind"] == "model" for group in graph["groups"])


# --------------------------------------------------------------------------- #
# generated real format files (independent from formats-dev fixtures)
# --------------------------------------------------------------------------- #
def build_tiny_onnx(path: Path) -> None:
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    rng = np.random.default_rng(7)
    initializers = [
        numpy_helper.from_array(rng.standard_normal((8, 3, 3, 3)).astype("float32"), "w1"),
        numpy_helper.from_array(np.zeros(8, dtype="float32"), "b1"),
        numpy_helper.from_array(rng.standard_normal((16, 8, 3, 3)).astype("float32"), "w2"),
        numpy_helper.from_array(np.zeros(16, dtype="float32"), "b2"),
        numpy_helper.from_array(rng.standard_normal((16, 4)).astype("float32"), "w3"),
        numpy_helper.from_array(np.zeros(4, dtype="float32"), "b3"),
    ]
    nodes = [
        helper.make_node("Conv", ["x", "w1", "b1"], ["c1"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
        helper.make_node("Relu", ["c1"], ["r1"]),
        helper.make_node("MaxPool", ["r1"], ["p1"], kernel_shape=[2, 2], strides=[2, 2]),
        helper.make_node("Conv", ["p1", "w2", "b2"], ["c2"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
        helper.make_node("Relu", ["c2"], ["r2"]),
        helper.make_node("GlobalAveragePool", ["r2"], ["gap"]),
        helper.make_node("Flatten", ["gap"], ["flat"], axis=1),
        helper.make_node("Gemm", ["flat", "w3", "b3"], ["y"]),
    ]
    graph = helper.make_graph(
        nodes,
        "qa_tiny_onnx",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 8, 8])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])],
        initializer=initializers,
    )
    model = helper.make_model(
        graph, producer_name="netprism-qa", opset_imports=[helper.make_opsetid("", 17)]
    )
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def build_tiny_torchscript(path: Path) -> None:
    import torch
    from torch import nn

    class TinyQA(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
            self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
            self.pool = nn.MaxPool2d(2)
            self.fc = nn.Linear(16 * 2 * 2, 4)

        def forward(self, x):  # noqa: ANN001 - torch module
            x = torch.relu(self.conv1(x))
            x = self.pool(x)
            x = torch.relu(self.conv2(x))
            x = self.pool(x)
            x = torch.flatten(x, 1)
            return self.fc(x)

    traced = torch.jit.trace(TinyQA().eval(), torch.zeros(1, 3, 8, 8))
    traced.save(str(path))


def build_tiny_keras(path: Path) -> None:
    """Write a real Keras v3 archive (zip containing config.json) by hand.

    Mirrors what ``model.save("m.keras")`` stores for a small Sequential CNN;
    no TensorFlow/Keras installation is required.
    """
    config = {
        "module": "keras",
        "class_name": "Sequential",
        "config": {
            "name": "qa_keras_net",
            "layers": [
                {
                    "module": "keras.layers",
                    "class_name": "InputLayer",
                    "config": {"batch_shape": [None, 16, 16, 3], "dtype": "float32", "name": "input_layer"},
                    "registered_name": None,
                },
                {
                    "module": "keras.layers",
                    "class_name": "Conv2D",
                    "config": {
                        "name": "conv1",
                        "filters": 8,
                        "kernel_size": [3, 3],
                        "strides": [1, 1],
                        "padding": "same",
                        "data_format": "channels_last",
                        "dilation_rate": [1, 1],
                        "groups": 1,
                        "activation": "relu",
                        "use_bias": True,
                        "trainable": True,
                    },
                    "registered_name": None,
                },
                {
                    "module": "keras.layers",
                    "class_name": "MaxPooling2D",
                    "config": {
                        "name": "pool1",
                        "pool_size": [2, 2],
                        "strides": [2, 2],
                        "padding": "valid",
                        "data_format": "channels_last",
                    },
                    "registered_name": None,
                },
                {
                    "module": "keras.layers",
                    "class_name": "Flatten",
                    "config": {"name": "flatten", "data_format": "channels_last"},
                    "registered_name": None,
                },
                {
                    "module": "keras.layers",
                    "class_name": "Dense",
                    "config": {
                        "name": "dense1",
                        "units": 4,
                        "activation": "linear",
                        "use_bias": True,
                        "trainable": True,
                    },
                    "registered_name": None,
                },
            ],
            "build_input_shape": [None, 16, 16, 3],
        },
        "registered_name": None,
    }
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("config.json", json.dumps(config))
        archive.writestr("metadata.json", json.dumps({"keras_version": "3.6.0", "backend": "tensorflow"}))


# --------------------------------------------------------------------------- #
# `report` command
# --------------------------------------------------------------------------- #
def test_report_md_from_demo_dir_matches_graph_numbers() -> None:
    graph = load_demo(DEMO_GRAPH_FILES["vgg"])
    model = graph["model"]
    metrics = model["metrics"]
    entry = demo_index_entry("cnn_vgg__vggmini")
    assert entry["totalParams"] == model["totalParams"]
    assert entry["totalFlops"] == model["totalFlops"]

    proc = run_cli("report", "examples/demo", "--model", "cnn_vgg__vggmini", "--format", "md")
    assert proc.returncode == 0, proc.stderr
    text = proc.stdout
    assert "\ufffd" not in text, "report stdout is not valid UTF-8"

    lines = text.splitlines()
    assert lines[0] == f"# {model['name']} 分析报告"
    assert f"**Framework** {model['framework']}" in text
    assert f"**Mode** {model['mode']}" in text
    assert f"**Params** {model['totalParams']:,}" in text
    assert f"**FLOPs** {model['totalFlops']:,}" in text
    assert f"**Activation (估算)** {human_bytes(metrics['activationBytes'])}" in text
    assert f"**Receptive field (估算)** {metrics['maxReceptiveField']:,}" in text
    assert f"**Warnings** {len(model['warnings']):,}" in text
    assert "## 分组汇总 (Groups)" in text
    assert "## 逐层明细 (Layers)" in text
    assert "估算说明" in text
    assert "features.0" in text
    assert text.count("估算") >= 3

    # The layer table must list every graph node exactly once (plus header rows).
    layer_row_lines = [line for line in lines if line.startswith("| n")]
    assert len(layer_row_lines) == len(graph["nodes"])
    for node in graph["nodes"]:
        assert node["name"] in text

    # Root group summary counts must add up to the model totals.
    root_group = next(group for group in graph["groups"] if group["kind"] == "model")
    assert root_group["numParams"] == model["totalParams"]
    assert root_group["flops"] == model["totalFlops"]


def test_report_csv_file_matches_graph_and_has_fixed_columns(tmp_path: Path) -> None:
    graph = load_demo(DEMO_GRAPH_FILES["vgg"])
    model = graph["model"]
    metrics = model["metrics"]
    out = tmp_path / "report.csv"

    proc = run_cli(
        "report",
        "examples/demo",
        "--model",
        "cnn_vgg__vggmini",
        "--format",
        "csv",
        "--out",
        str(out),
    )
    assert proc.returncode == 0, proc.stderr
    raw = out.read_bytes()
    assert raw[:3] == b"\xef\xbb\xbf", "CLI-written CSV must be UTF-8-SIG"
    text = raw.decode("utf-8-sig")
    assert "\ufffd" not in text

    rows = list(csv.reader(io.StringIO(text)))
    assert rows[1] == REPORT_COLUMNS
    data = rows[2:]
    assert len(data) == len(graph["nodes"])
    assert all(len(row) == len(REPORT_COLUMNS) for row in data), "CSV rows have different column counts"

    comment = parse_report_comment(text)
    assert comment["model"] == model["name"]
    assert comment["framework"] == model["framework"]
    assert comment["mode"] == model["mode"]
    assert comment["params"] == str(model["totalParams"])
    assert comment["flops"] == str(model["totalFlops"])
    assert comment["activationBytes"] == str(metrics["activationBytes"])
    assert comment["maxReceptiveField"] == str(metrics["maxReceptiveField"])
    assert comment["warnings"] == str(len(model["warnings"]))

    by_id = {row[0]: row for row in data}
    assert set(by_id) == {node["id"] for node in graph["nodes"]}
    for node in graph["nodes"]:
        row = by_id[node["id"]]
        assert row[1] == node["name"]
        assert row[2] == node["op"]
        assert row[3] == node["kind"]
        assert row[4] == node["modulePath"]
        assert int(row[7]) == node["numParams"]
        assert int(row[8]) == node["flops"]
        assert int(row[9]) == expected_activation_bytes(node)
        assert int(row[10]) == node["metrics"]["receptiveField"]
        assert "\r" not in row[-1]

    assert sum(int(row[9]) for row in data) == metrics["activationBytes"]
    assert sum(int(row[7]) for row in data) == model["totalParams"]
    assert sum(int(row[8]) for row in data) == model["totalFlops"]
    assert max(int(row[10]) for row in data) == metrics["maxReceptiveField"]


def test_report_html_file_is_self_contained_and_renders_numbers(tmp_path: Path) -> None:
    graph = load_demo(DEMO_GRAPH_FILES["resnet"])
    model = graph["model"]
    metrics = model["metrics"]
    out = tmp_path / "report.html"

    proc = run_cli(
        "report",
        "examples/demo",
        "--model",
        "resnet_mini__resnetmini",
        "--format",
        "html",
        "--out",
        str(out),
    )
    assert proc.returncode == 0, proc.stderr
    raw = out.read_bytes()
    assert raw[:3] != b"\xef\xbb\xbf", "HTML report should be plain UTF-8, not UTF-8-SIG"
    text = raw.decode("utf-8")
    assert text.startswith("<!DOCTYPE html>")
    assert "\ufffd" not in text
    assert model["name"] in text
    assert model["framework"] in text
    assert f"{model['totalParams']:,}" in text
    assert f"{model['totalFlops']:,}" in text
    assert human_bytes(metrics["activationBytes"]) in text
    assert f"{metrics['maxReceptiveField']:,}" in text
    assert "估算" in text
    sample_node = next(node for node in graph["nodes"] if node["kind"] not in ("input", "output"))
    assert sample_node["name"] in text
    assert "<script" not in text
    assert "<link" not in text
    assert "http://" not in text and "https://" not in text


def test_report_accepts_a_bare_graph_json_source() -> None:
    model = load_demo(DEMO_GRAPH_FILES["vgg"])["model"]
    proc = run_cli("report", "examples/demo/cnn_vgg__vggmini.json", "--format", "md")
    assert proc.returncode == 0, proc.stderr
    assert f"**Params** {model['totalParams']:,}" in proc.stdout
    assert proc.stdout.startswith(f"# {model['name']} 分析报告")


def test_cli_csv_output_has_no_doubled_carriage_returns() -> None:
    """csv.writer already emits CRLF; the CLI text layer must not add another CR."""
    for args in (
        ("report", "examples/demo", "--model", "cnn_vgg__vggmini", "--format", "csv"),
        (
            "compare",
            "examples/demo",
            "examples/demo",
            "--model-a",
            "cnn_vgg__vggmini",
            "--model-b",
            "resnet_mini__resnetmini",
            "--format",
            "csv",
        ),
    ):
        proc = run_cli_bytes(*args)
        assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")
        assert proc.stdout[:3] == b"\xef\xbb\xbf", f"{args[0]} CSV must start with a UTF-8 BOM"
        assert b"\r\r\n" not in proc.stdout, (
            f"{args[0]} CSV contains doubled CR (\\r\\r\\n): the Windows text layer re-translated "
            "the csv.writer CRLF output"
        )


# --------------------------------------------------------------------------- #
# `compare` command
# --------------------------------------------------------------------------- #
def test_compare_json_matches_independent_classification(tmp_path: Path) -> None:
    graph_a, graph_b, path_a, path_b = write_mutated_vgg_pair(tmp_path)
    expected = independent_compare(graph_a, graph_b)
    assert len(expected["added"]) == 1
    assert len(expected["removed"]) == 1
    assert len(expected["changed"]) == 1
    out = tmp_path / "compare.json"

    proc = run_cli(
        "compare",
        str(path_a),
        str(path_b),
        "--format",
        "json",
        "--out",
        str(out),
    )
    assert proc.returncode == 0, proc.stderr
    raw = out.read_bytes()
    assert raw[:3] != b"\xef\xbb\xbf"
    payload = json.loads(raw.decode("utf-8"))

    assert set(payload) >= {"a", "b", "summary", "counts", "added", "removed", "changed"}
    assert payload["a"]["name"] == graph_a["model"]["name"]
    assert payload["b"]["name"] == graph_b["model"]["name"]
    assert payload["a"]["totalParams"] == graph_a["model"]["totalParams"]
    assert payload["b"]["totalParams"] == graph_b["model"]["totalParams"]

    counts = payload["counts"]
    assert counts == {
        "added": len(expected["added"]),
        "removed": len(expected["removed"]),
        "changed": len(expected["changed"]),
        "unchanged": expected["unchanged"],
    }
    assert counts["added"] > 0 and counts["removed"] > 0 and counts["changed"] > 0
    assert {row["key"] for row in payload["added"]} == expected["added"]
    assert {row["key"] for row in payload["removed"]} == expected["removed"]
    assert {row["key"] for row in payload["changed"]} == set(expected["changed"])
    for entry in payload["changed"]:
        assert entry["fields"] == expected["changed"][entry["key"]]

    params = payload["summary"]["params"]
    flops = payload["summary"]["flops"]
    assert params == {
        "a": graph_a["model"]["totalParams"],
        "b": graph_b["model"]["totalParams"],
        "diff": graph_b["model"]["totalParams"] - graph_a["model"]["totalParams"],
    }
    assert flops == {
        "a": graph_a["model"]["totalFlops"],
        "b": graph_b["model"]["totalFlops"],
        "diff": graph_b["model"]["totalFlops"] - graph_a["model"]["totalFlops"],
    }
    activation = payload["summary"]["activationBytes"]
    assert activation["a"] == expected_model_activation(graph_a)
    assert activation["b"] == expected_model_activation(graph_b)
    assert activation["diff"] == activation["b"] - activation["a"]
    assert payload["summary"]["maxReceptiveField"]["a"] == graph_a["model"]["metrics"]["maxReceptiveField"]

    # Every node lands in exactly one bucket on each side.
    assert (
        counts["unchanged"] + counts["changed"] + counts["removed"] == len(graph_a["nodes"])
    )
    assert counts["unchanged"] + counts["changed"] + counts["added"] == len(graph_b["nodes"])


def test_compare_cross_model_scan_dirs_is_consistent(tmp_path: Path) -> None:
    """Two different demo models from the same scan directory: no crash, exact counts."""
    graph_a = load_demo(DEMO_GRAPH_FILES["vgg"])
    graph_b = load_demo(DEMO_GRAPH_FILES["resnet"])
    expected = independent_compare(graph_a, graph_b)
    out = tmp_path / "cross_models.json"

    proc = run_cli(
        "compare",
        "examples/demo",
        "examples/demo",
        "--model-a",
        "cnn_vgg__vggmini",
        "--model-b",
        "resnet_mini__resnetmini",
        "--format",
        "json",
        "--out",
        str(out),
    )
    assert proc.returncode == 0, proc.stderr
    payload = load_json(out)
    assert payload["counts"] == {
        "added": len(expected["added"]),
        "removed": len(expected["removed"]),
        "changed": len(expected["changed"]),
        "unchanged": expected["unchanged"],
    }
    assert {row["key"] for row in payload["added"]} == expected["added"]
    assert {row["key"] for row in payload["removed"]} == expected["removed"]
    for entry in payload["changed"]:
        assert entry["fields"] == expected["changed"][entry["key"]]
    assert payload["summary"]["params"] == {
        "a": graph_a["model"]["totalParams"],
        "b": graph_b["model"]["totalParams"],
        "diff": graph_b["model"]["totalParams"] - graph_a["model"]["totalParams"],
    }
    assert payload["summary"]["activationBytes"] == {
        "a": expected_model_activation(graph_a),
        "b": expected_model_activation(graph_b),
        "diff": expected_model_activation(graph_b) - expected_model_activation(graph_a),
    }


def test_compare_markdown_and_csv(tmp_path: Path) -> None:
    graph_a, graph_b, path_a, path_b = write_mutated_vgg_pair(tmp_path)
    expected = independent_compare(graph_a, graph_b)

    md_path = tmp_path / "compare.md"
    proc = run_cli(
        "compare",
        str(path_a),
        str(path_b),
        "--format",
        "md",
        "--out",
        str(md_path),
    )
    assert proc.returncode == 0, proc.stderr
    md_text = md_path.read_text(encoding="utf-8")
    assert "\ufffd" not in md_text
    assert md_text.startswith("# 模型对比报告")
    assert graph_a["model"]["name"] in md_text and graph_b["model"]["name"] in md_text
    assert "## 摘要 (Summary)" in md_text
    assert f"## 新增节点 (Added: {len(expected['added'])})" in md_text
    assert f"## 删除节点 (Removed: {len(expected['removed'])})" in md_text
    assert f"## 变更节点 (Changed: {len(expected['changed'])})" in md_text
    assert "对比说明" in md_text
    params_diff = graph_b["model"]["totalParams"] - graph_a["model"]["totalParams"]
    assert f"{params_diff:+,}" in md_text
    for key in list(expected["added"])[:3] + list(expected["removed"])[:3]:
        assert key in md_text

    csv_path = tmp_path / "compare.csv"
    proc = run_cli(
        "compare",
        str(path_a),
        str(path_b),
        "--format",
        "csv",
        "--out",
        str(csv_path),
    )
    assert proc.returncode == 0, proc.stderr
    raw = csv_path.read_bytes()
    assert raw[:3] == b"\xef\xbb\xbf"
    text = raw.decode("utf-8-sig")
    rows = list(csv.reader(io.StringIO(text)))
    assert rows[0] == COMPARE_CSV_COLUMNS
    data = rows[1:]
    assert all(len(row) == len(COMPARE_CSV_COLUMNS) for row in data)
    status_counts = {"added": 0, "removed": 0, "changed": 0}
    for row in data:
        status_counts[row[0]] += 1
    assert status_counts == {
        "added": len(expected["added"]),
        "removed": len(expected["removed"]),
        "changed": len(expected["changed"]),
    }
    assert {row[1] for row in data if row[0] == "added"} == expected["added"]
    assert {row[1] for row in data if row[0] == "removed"} == expected["removed"]

    nodes_a = keyed_nodes(graph_a)
    nodes_b = keyed_nodes(graph_b)
    for row in data:
        status, key = row[0], row[1]
        if status == "added":
            node = nodes_b[key]
            assert row[4] == "", "added row must not have an A-side op"
            assert row[5] == node["op"]
            assert int(row[9]) == node["numParams"]
        elif status == "removed":
            node = nodes_a[key]
            assert row[5] == "", "removed row must not have a B-side op"
            assert row[4] == node["op"]
            assert int(row[8]) == node["numParams"]
        else:
            assert status == "changed"
            assert set(row[16].split("; ")) == set(expected["changed"][key])
        assert "\r" not in row[-1]


def test_compare_identical_graph_is_all_unchanged(tmp_path: Path) -> None:
    graph = load_demo(DEMO_GRAPH_FILES["unet"])
    out = tmp_path / "same.json"
    proc = run_cli(
        "compare",
        "examples/demo",
        "examples/demo",
        "--model-a",
        "unet_mini__unetmini",
        "--model-b",
        "unet_mini__unetmini",
        "--format",
        "json",
        "--out",
        str(out),
    )
    assert proc.returncode == 0, proc.stderr
    payload = load_json(out)
    assert payload["counts"] == {
        "added": 0,
        "removed": 0,
        "changed": 0,
        "unchanged": len(graph["nodes"]),
    }
    for key in ("params", "flops", "activationBytes"):
        assert payload["summary"][key]["diff"] == 0
    assert all(export["diff"] == 0 for export in payload["summary"].values())


def test_compare_accepts_graph_json_and_scan_dir_sources(tmp_path: Path) -> None:
    out = tmp_path / "cross.json"
    proc = run_cli(
        "compare",
        "examples/demo/cnn_vgg__vggmini.json",
        "examples/demo",
        "--model-b",
        "cnn_vgg__vggmini",
        "--format",
        "json",
        "--out",
        str(out),
    )
    assert proc.returncode == 0, proc.stderr
    payload = load_json(out)
    assert payload["counts"] == {
        "added": 0,
        "removed": 0,
        "changed": 0,
        "unchanged": len(load_demo(DEMO_GRAPH_FILES["vgg"])["nodes"]),
    }
    assert payload["a"]["name"] == payload["b"]["name"]


# --------------------------------------------------------------------------- #
# `scan` command + metrics enrichment
# --------------------------------------------------------------------------- #
def test_scan_samples_enriches_metrics_consistently(tmp_path: Path) -> None:
    out = tmp_path / "scan_out"
    proc = run_cli("scan", "samples", "--out", str(out), "--quiet")
    assert proc.returncode == 0, proc.stderr
    index = load_json(out / "index.json")
    assert validate_index(index) == []
    assert index["models"], "scan produced no models"
    assert len(index["models"]) >= 6

    graph_with_rf = 0
    for entry in index["models"]:
        graph = load_json(out / entry["file"])
        assert validate_graph(graph) == []
        model = graph["model"]
        metrics = model.get("metrics")
        assert isinstance(metrics, dict) and metrics, f"{entry['id']}: scan did not add model.metrics"
        assert metrics["paramBytes"] == model["totalParams"] * 4
        assert metrics["macs"] == model["totalFlops"] // 2
        independent = expected_model_activation(graph)
        assert metrics["activationBytes"] == independent
        assert 0 <= metrics["peakActivationBytes"] <= max(1, metrics["activationBytes"])
        assert metrics["maxReceptiveField"] >= 1
        if metrics["maxReceptiveField"] > 1:
            graph_with_rf += 1

        assert model["numLayers"] == sum(
            1 for node in graph["nodes"] if node["kind"] not in ("input", "output")
        )
        assert entry["totalParams"] == model["totalParams"]
        assert entry["totalFlops"] == model["totalFlops"]
        assert entry["numLayers"] == model["numLayers"]

        for node in graph["nodes"]:
            node_metrics = node.get("metrics")
            assert isinstance(node_metrics, dict), f"{entry['id']}/{node['id']}: missing node.metrics"
            assert set(node_metrics) == {"activationBytes", "receptiveField", "macs"}
            assert node_metrics["activationBytes"] == expected_activation_bytes(node)
            assert node_metrics["macs"] == node["flops"] // 2
            assert node_metrics["receptiveField"] >= 1
        assert metrics["activationBytes"] == sum(
            node["metrics"]["activationBytes"] for node in graph["nodes"]
        )
        assert metrics["maxReceptiveField"] == max(
            node["metrics"]["receptiveField"] for node in graph["nodes"]
        )

    assert graph_with_rf >= 1, "no scanned CNN got a receptive field > 1"

    # A scan directory is a valid report source; the reported numbers must be
    # the enriched ones written by that same scan.
    first = index["models"][0]
    graph = load_json(out / first["file"])
    proc = run_cli("report", str(out), "--model", first["id"], "--format", "md")
    assert proc.returncode == 0, proc.stderr
    assert f"**Params** {graph['model']['totalParams']:,}" in proc.stdout
    assert f"**Activation (估算)** {human_bytes(graph['model']['metrics']['activationBytes'])}" in proc.stdout


def test_scan_format_error_isolation_and_no_formats(tmp_path: Path) -> None:
    source = tmp_path / "src"
    source.mkdir()
    shutil.copy(SAMPLES_DIR / "mlp.py", source / "mlp.py")
    (source / "broken.onnx").write_bytes(b"this is definitely not an onnx file")
    (source / "broken.keras").write_bytes(b"not a zip archive either")

    out = tmp_path / "scan_default"
    proc = run_cli("scan", str(source), "--out", str(out), "--quiet")
    assert proc.returncode == 0, proc.stderr
    index = load_json(out / "index.json")
    assert any(entry["id"].startswith("mlp__") for entry in index["models"])
    assert {entry["source"] for entry in index["models"]} == {"mlp.py"}
    errored = {entry["file"]: entry["message"] for entry in index["errors"]}
    assert "broken.onnx" in errored and "broken.keras" in errored
    for message in errored.values():
        assert message.strip()
        assert "Traceback" not in message

    out_no = tmp_path / "scan_no_formats"
    proc = run_cli("scan", str(source), "--out", str(out_no), "--quiet", "--no-formats")
    assert proc.returncode == 0, proc.stderr
    index_no = load_json(out_no / "index.json")
    assert [entry["source"] for entry in index_no["models"]] == ["mlp.py"]
    assert index_no["errors"] == [], "--no-formats must not touch non-Python model files"
    assert validate_index(index_no) == []


def test_bad_format_inspect_fails_with_readable_error(tmp_path: Path) -> None:
    bad = tmp_path / "broken.onnx"
    bad.write_bytes(b"\x00\x01 not onnx")
    proc = run_cli("inspect", str(bad), "--json")
    assert proc.returncode == 2
    assert "Traceback" not in proc.stderr
    assert "错误" in proc.stderr
    assert proc.stdout.strip() == ""


# --------------------------------------------------------------------------- #
# demo graph independent metric consistency (+ hand-computed receptive fields)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("filename", sorted(DEMO_GRAPH_FILES.values()))
def test_demo_metrics_match_independent_activation_and_invariants(filename: str) -> None:
    graph = load_demo(filename)
    model = graph["model"]
    metrics = model["metrics"]
    assert metrics["activationBytes"] == expected_model_activation(graph)
    assert metrics["paramBytes"] == model["totalParams"] * 4
    assert metrics["macs"] == model["totalFlops"] // 2
    assert metrics["maxReceptiveField"] == max(
        node["metrics"]["receptiveField"] for node in graph["nodes"]
    )
    node_activation_sum = 0
    for node in graph["nodes"]:
        assert node["metrics"]["activationBytes"] == expected_activation_bytes(node)
        assert node["metrics"]["macs"] == node["flops"] // 2
        assert node["metrics"]["receptiveField"] >= 1
        node_activation_sum += node["metrics"]["activationBytes"]
    assert node_activation_sum == metrics["activationBytes"]
    node_params = sum(node["numParams"] for node in graph["nodes"])
    node_flops = sum(node["flops"] for node in graph["nodes"])
    if model["warnings"]:
        # An eager-trace fallback can leave unused parameters out of the layer
        # list (DynamicNet only executes one branch); model totals stay the
        # authoritative upper bound and must never be exceeded.
        assert node_params <= model["totalParams"]
        assert node_flops <= model["totalFlops"]
    else:
        assert node_params == model["totalParams"]
        assert node_flops == model["totalFlops"]
    assert 0 <= metrics["peakActivationBytes"] <= max(1, metrics["activationBytes"])
    assert metrics["peakActivationBytes"] >= max(
        node["metrics"]["activationBytes"] for node in graph["nodes"]
    )


def test_vggmini_receptive_fields_hand_calculation() -> None:
    """Hand-computed 3x3 conv / 2x2 stride-2 pool propagation for VGGMini."""
    graph = load_demo(DEMO_GRAPH_FILES["vgg"])
    expected = {
        "x": 1,
        "features.0": 3,   # 1 + (3-1)
        "features.1": 3,   # BatchNorm keeps the field
        "features.3": 5,   # 3 + (3-1)
        "features.5": 6,   # 5 + (2-1)*1, jump becomes 2
        "features.6": 10,  # 6 + (3-1)*2
        "features.9": 10,  # 1x1 conv
        "features.11": 12,  # 10 + (2-1)*2, jump becomes 4
        "features.12": 20,  # 12 + (3-1)*4
        "features.15": 28,  # 20 + (3-1)*4
        "features.18": 28,  # adaptive pool without kernel/stride info
        "classifier.4": 28,
    }
    for name, value in expected.items():
        node = next(node for node in graph["nodes"] if node["name"] == name)
        assert node["metrics"]["receptiveField"] == value, f"{name}: wrong receptive field"
    assert graph["model"]["metrics"]["maxReceptiveField"] == 28


def test_report_numbers_equal_metrics_of_a_freshly_enriched_scan(tmp_path: Path) -> None:
    """End-to-end: scan -> report file must be byte-decodable and keep the numbers."""
    out = tmp_path / "scan"
    proc = run_cli("scan", "samples", "--out", str(out), "--quiet", "--max-files", "12")
    assert proc.returncode == 0, proc.stderr
    index = load_json(out / "index.json")
    vgg_entry = next(entry for entry in index["models"] if entry["name"] == "VGGMini")
    graph = load_json(out / vgg_entry["file"])

    csv_path = tmp_path / "vgg.csv"
    proc = run_cli("report", str(out), "--model", vgg_entry["id"], "--format", "csv", "--out", str(csv_path))
    assert proc.returncode == 0, proc.stderr
    text = csv_path.read_bytes().decode("utf-8-sig")
    rows = list(csv.reader(io.StringIO(text)))
    data = rows[2:]
    assert len(data) == len(graph["nodes"])
    assert sum(int(row[9]) for row in data) == graph["model"]["metrics"]["activationBytes"]
    assert max(int(row[10]) for row in data) == graph["model"]["metrics"]["maxReceptiveField"]


# --------------------------------------------------------------------------- #
# format adapters: checked-in fixtures (formats-dev scope)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("framework", ["onnx", "torchscript", "keras"])
def test_format_fixture_inspect_and_scan(framework: str, tmp_path: Path) -> None:
    suffixes = FORMAT_FRAMEWORKS[framework]
    if not adapter_module_exists(framework):
        pytest.skip(
            f"{framework} adapter not landed yet (netprism/adapters/{framework}_adapter.py missing; "
            "formats-dev scope) — inspect/scan format coverage will run once it exists"
        )
    files = format_fixture_files(suffixes, exclude_negative=True)
    if not files:
        pytest.skip(
            f"no positive {framework} fixture under tests/fixtures/formats "
            f"({', '.join(suffixes)}); formats-dev owns the fixtures"
        )
    fixture = files[0]

    proc = run_cli("inspect", str(fixture), "--json")
    assert proc.returncode == 0, f"inspect failed for {fixture.name}:\n{proc.stderr}"
    graph = json.loads(proc.stdout)
    assert_valid_format_graph(graph, framework)

    # inspect on a format file must work from a bare file path, not only via scan.
    proc_tree = run_cli("inspect", str(fixture))
    assert proc_tree.returncode == 0, proc_tree.stderr
    assert "params=" in proc_tree.stdout
    assert any(node["name"] in proc_tree.stdout for node in graph["nodes"])

    # Copy exactly one valid fixture into an isolated folder and scan it.
    scan_dir = tmp_path / "one"
    scan_dir.mkdir()
    shutil.copy(fixture, scan_dir / fixture.name)
    out = tmp_path / "scan"
    proc = run_cli("scan", str(scan_dir), "--out", str(out), "--quiet")
    assert proc.returncode == 0, proc.stderr
    index = load_json(out / "index.json")
    assert index["errors"] == [], f"valid {framework} fixture was reported as an error: {index['errors']}"
    assert len(index["models"]) == 1
    entry = index["models"][0]
    assert entry["source"] == fixture.name
    scanned = load_json(out / entry["file"])
    assert validate_graph(scanned) == []
    assert scanned["model"]["framework"] == framework
    metrics = scanned["model"]["metrics"]
    assert metrics["activationBytes"] == expected_model_activation(scanned)
    assert metrics["maxReceptiveField"] >= 1
    assert scanned["model"]["totalParams"] == entry["totalParams"]

    # A format model file is a valid report source and is enriched on the fly.
    proc = run_cli("report", str(fixture), "--format", "md")
    assert proc.returncode == 0, proc.stderr
    assert graph["model"]["name"] in proc.stdout
    assert f"**Framework** {framework}" in proc.stdout
    assert "**Params**" in proc.stdout
    assert "Activation (估算)" in proc.stdout
    assert "**Receptive field (估算)**" in proc.stdout

    # Cross-source compare: format file vs. a PyTorch demo graph.
    out_json = tmp_path / "cross.json"
    proc = run_cli(
        "compare",
        str(fixture),
        "examples/demo",
        "--model-b",
        "cnn_vgg__vggmini",
        "--format",
        "json",
        "--out",
        str(out_json),
    )
    assert proc.returncode == 0, proc.stderr
    payload = load_json(out_json)
    assert payload["a"]["framework"] == framework
    assert payload["b"]["framework"] == "pytorch"
    assert payload["summary"]["params"]["diff"] == (
        payload["b"]["totalParams"] - payload["a"]["totalParams"]
    )


def test_scan_all_checked_in_format_fixtures(tmp_path: Path) -> None:
    """Scan the whole fixtures folder: valid models load, broken ones are isolated."""
    require_adapters("onnx", "torchscript", "keras")
    if not FORMAT_FIXTURES.is_dir():
        pytest.skip("tests/fixtures/formats is not ready yet (formats-dev scope)")
    suffixes = (".onnx", ".pt", ".pth", ".keras")
    all_model_files = {
        path.name
        for path in FORMAT_FIXTURES.rglob("*")
        if path.is_file() and path.suffix.lower() in suffixes
    }
    if not all_model_files:
        pytest.skip("no format fixtures under tests/fixtures/formats")
    negatives = {
        path.name
        for path in FORMAT_FIXTURES.rglob("*")
        if path.is_file() and path.suffix.lower() in suffixes and _negative_name_hint(path)
    }
    positives = all_model_files - negatives
    assert positives, "fixtures folder contains no positive model files"
    manifest_path = FORMAT_FIXTURES / "manifest.json"
    if manifest_path.is_file():
        manifest_files = {entry["file"] for entry in load_json(manifest_path)}
        assert manifest_files == all_model_files, "manifest.json does not describe the checked-in fixtures"

    out = tmp_path / "scan_all"
    proc = run_cli("scan", str(FORMAT_FIXTURES), "--out", str(out), "--quiet")
    assert proc.returncode == 0, proc.stderr
    index = load_json(out / "index.json")
    assert validate_index(index) == []
    errored = {entry["file"]: entry["message"] for entry in index["errors"]}
    assert set(errored) == negatives, f"scan error isolation mismatch: {sorted(errored)}"
    for name, message in errored.items():
        assert message.strip() and "Traceback" not in message, f"{name}: unreadable error {message!r}"
    if "mlp_state_dict.pt" in errored:
        assert "state_dict" in errored["mlp_state_dict.pt"].lower()

    assert {entry["source"] for entry in index["models"]} == positives
    assert len(index["models"]) == len(positives)

    suffix_framework = {".onnx": "onnx", ".pt": "torchscript", ".pth": "torchscript", ".keras": "keras"}
    for entry in index["models"]:
        suffix = Path(entry["source"]).suffix.lower()
        framework = suffix_framework[suffix]
        graph = load_json(out / entry["file"])
        assert validate_graph(graph) == [], f"{entry['id']} produced an invalid graph"
        model = graph["model"]
        assert model["framework"] == framework
        assert model["mode"] == framework
        assert model["totalParams"] > 0
        assert len(graph["nodes"]) > 1
        assert len(graph["edges"]) > 0
        assert entry["totalParams"] == model["totalParams"]
        assert entry["numLayers"] == model["numLayers"]
        metrics = model["metrics"]
        assert metrics["activationBytes"] == expected_model_activation(graph)
        assert metrics["paramBytes"] == model["totalParams"] * 4
        assert metrics["maxReceptiveField"] >= 1
        for node in graph["nodes"]:
            if node["kind"] != "input":
                assert node["inShapes"], f"{entry['id']}/{node['id']} missing inShapes"
            assert node["outShapes"], f"{entry['id']}/{node['id']} missing outShapes"

    # `--no-formats` really means "do not parse model files": a fixture-only
    # folder must come back empty, not with the format models above.
    out_no = tmp_path / "scan_no_formats"
    proc = run_cli("scan", str(FORMAT_FIXTURES), "--out", str(out_no), "--quiet", "--no-formats")
    assert proc.returncode == 0, proc.stderr
    index_no = load_json(out_no / "index.json")
    assert index_no["models"] == []
    assert index_no["errors"] == []


def test_pth_suffix_is_treated_as_torchscript(tmp_path: Path) -> None:
    require_adapters("torchscript")
    fixtures = format_fixture_files((".pt",), exclude_negative=True)
    if not fixtures:
        pytest.skip("no .pt fixture under tests/fixtures/formats (formats-dev scope)")
    source = fixtures[0]
    alias = tmp_path / "renamed_model.pth"
    shutil.copy(source, alias)

    proc = run_cli("inspect", str(alias), "--json")
    assert proc.returncode == 0, proc.stderr
    graph = json.loads(proc.stdout)
    assert_valid_format_graph(graph, "torchscript")
    assert graph["model"]["mode"] == "torchscript"
    source_proc = run_cli("inspect", str(source), "--json")
    assert json.loads(source_proc.stdout)["model"]["totalParams"] == graph["model"]["totalParams"]


def test_state_dict_pt_fails_readably_and_scan_isolates_it(tmp_path: Path) -> None:
    require_adapters("torchscript")
    torch = pytest.importorskip("torch")
    from torch import nn

    class Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(4, 2)

        def forward(self, x):  # noqa: ANN001
            return self.fc(x)

    work = tmp_path / "weights"
    work.mkdir()
    torch.save(Tiny().state_dict(), work / "weights.pt")
    proc = run_cli("inspect", str(work / "weights.pt"), "--json")
    assert proc.returncode == 2
    assert proc.stdout.strip() == ""
    assert "Traceback" not in proc.stderr
    stderr = proc.stderr.lower()
    assert "state_dict" in stderr, f"error should explain the state_dict problem:\n{proc.stderr}"
    assert "torch.jit" in stderr or "onnx" in stderr

    fixtures = format_fixture_files((".pt",), exclude_negative=True)
    if not fixtures:
        pytest.skip("no valid .pt fixture to prove scan continues past the bad file")
    shutil.copy(fixtures[0], work / "valid_model.pt")
    out = tmp_path / "scan"
    proc = run_cli("scan", str(work), "--out", str(out), "--quiet")
    assert proc.returncode == 0, proc.stderr
    index = load_json(out / "index.json")
    assert len(index["models"]) == 1
    assert index["models"][0]["source"] == "valid_model.pt"
    errored = {entry["file"]: entry["message"] for entry in index["errors"]}
    assert "weights.pt" in errored
    assert "state_dict" in errored["weights.pt"].lower()


def test_negative_format_fixtures_fail_readably(tmp_path: Path) -> None:
    if not FORMAT_FIXTURES.is_dir():
        pytest.skip("tests/fixtures/formats is not ready yet (formats-dev scope)")
    negatives = [path for path in sorted(FORMAT_FIXTURES.rglob("*")) if path.is_file() and _negative_name_hint(path)]
    if not negatives:
        pytest.skip("no negative fixtures (bad / broken / state_dict) under tests/fixtures/formats")
    for path in negatives:
        proc = run_cli("inspect", str(path), "--json")
        if proc.returncode == 0:
            # A fixture named "state_dict" must not be silently accepted as a model.
            pytest.fail(f"negative fixture {path.name} was parsed as a model instead of failing")
        assert "Traceback" not in proc.stderr, f"{path.name} error is not readable:\n{proc.stderr}"
        assert proc.stderr.strip(), f"{path.name}: expected an error message"


# --------------------------------------------------------------------------- #
# format adapters: real files generated locally (adapter independent)
# --------------------------------------------------------------------------- #
def test_generated_onnx_file_roundtrip(tmp_path: Path) -> None:
    if not adapter_module_exists("onnx"):
        pytest.skip("onnx adapter not landed yet (formats-dev scope)")
    pytest.importorskip("onnx")
    path = tmp_path / "qa_tiny.onnx"
    build_tiny_onnx(path)
    assert path.stat().st_size > 500

    proc = run_cli("inspect", str(path), "--json")
    assert proc.returncode == 0, proc.stderr
    graph = json.loads(proc.stdout)
    assert_valid_format_graph(graph, "onnx")
    assert len(graph["nodes"]) >= 6
    assert graph["model"]["numLayers"] >= 4
    assert graph["model"]["totalParams"] >= 1000

    out = tmp_path / "scan"
    proc = run_cli("scan", str(tmp_path), "--out", str(out), "--quiet")
    assert proc.returncode == 0, proc.stderr
    index = load_json(out / "index.json")
    assert index["errors"] == []
    assert any(entry["id"].startswith("qa_tiny__onnx") for entry in index["models"])

    proc = run_cli("report", str(path), "--format", "csv")
    assert proc.returncode == 0, proc.stderr
    rows = list(csv.reader(io.StringIO(proc.stdout.lstrip("\ufeff"))))
    assert rows[1] == REPORT_COLUMNS
    data = rows[2:]
    assert len(data) == len(graph["nodes"])
    assert all(len(row) == len(REPORT_COLUMNS) for row in data)
    assert sum(int(row[7]) for row in data) == graph["model"]["totalParams"]


def test_generated_torchscript_file_roundtrip(tmp_path: Path) -> None:
    if not adapter_module_exists("torchscript"):
        pytest.skip("torchscript adapter not landed yet (formats-dev scope)")
    pytest.importorskip("torch")
    path = tmp_path / "qa_tiny.pt"
    build_tiny_torchscript(path)
    assert path.stat().st_size > 100

    proc = run_cli("inspect", str(path), "--json")
    assert proc.returncode == 0, f"inspect failed for a traced TorchScript model:\n{proc.stderr}"
    graph = json.loads(proc.stdout)
    assert_valid_format_graph(graph, "torchscript")
    assert len(graph["nodes"]) >= 6
    assert graph["model"]["totalParams"] > 0
    assert graph["model"]["totalFlops"] > 0

    out = tmp_path / "scan"
    proc = run_cli("scan", str(tmp_path), "--out", str(out), "--quiet")
    assert proc.returncode == 0, proc.stderr
    index = load_json(out / "index.json")
    assert index["errors"] == []
    assert any(entry["id"].startswith("qa_tiny__torchscript") for entry in index["models"])


def test_generated_keras_file_roundtrip(tmp_path: Path) -> None:
    if not adapter_module_exists("keras"):
        pytest.skip("keras adapter not landed yet (formats-dev scope)")
    path = tmp_path / "qa_tiny.keras"
    build_tiny_keras(path)
    assert path.stat().st_size > 300

    proc = run_cli("inspect", str(path), "--json")
    assert proc.returncode == 0, f"inspect failed for a Keras v3 archive:\n{proc.stderr}"
    graph = json.loads(proc.stdout)
    assert_valid_format_graph(graph, "keras")
    model = graph["model"]
    assert model["totalParams"] == 2276  # conv 8*16*3*3+8 + dense 512*4+4
    assert model["flopsApprox"] is True
    assert model["totalFlops"] > 0, "Keras FLOPs must be derived from the layer config (SPEC 3)"
    assert [node["name"] for node in graph["nodes"][1:5]] == ["conv1", "pool1", "flatten", "dense1"]
    conv = next(node for node in graph["nodes"] if node["name"] == "conv1")
    assert conv["kind"] == "conv"
    assert conv["numParams"] == 224
    assert conv["outShapes"] == [[-1, 16, 16, 8]]

    out = tmp_path / "scan"
    proc = run_cli("scan", str(tmp_path), "--out", str(out), "--quiet")
    assert proc.returncode == 0, proc.stderr
    index = load_json(out / "index.json")
    assert index["errors"] == []
    assert any(entry["id"].startswith("qa_tiny__keras") for entry in index["models"])
    entry = next(entry for entry in index["models"] if entry["id"].startswith("qa_tiny__keras"))
    scanned = load_json(out / entry["file"])
    assert scanned["model"]["metrics"]["activationBytes"] == expected_model_activation(scanned)
    assert scanned["model"]["metrics"]["maxReceptiveField"] >= 3

    proc = run_cli("report", str(path), "--format", "md")
    assert proc.returncode == 0, proc.stderr
    assert "Activation (估算)" in proc.stdout
    assert "keras" in proc.stdout.lower()
