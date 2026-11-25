"""Tests for the analysis, report and comparison modules (stdlib only).

The demo graphs under ``examples/demo`` are used as real fixtures; synthetic
graphs cover receptive-field arithmetic, cycles and comparison classification.
No torch/onnx is imported here.
"""

from __future__ import annotations

import copy
import csv
import inspect
import io
import json
import math
from pathlib import Path

import pytest

from netprism.compare import compare_graphs, render_compare, write_compare
from netprism.metrics import enrich_graph, layer_rows
from netprism.report import render_report, write_report
from netprism.schema import validate_graph

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "examples" / "demo"

DEMO_FILES = {
    "vgg": "cnn_vgg__vggmini.json",
    "unet": "unet_mini__unetmini.json",
    "resnet": "resnet_mini__resnetmini.json",
    "transformer": "transformer_classifier__tinytransformerclassifier.json",
    "mlp": "mlp__mlp.json",
}

LAYER_COLUMNS = [
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


def load_demo(key: str) -> dict:
    return json.loads((DEMO / DEMO_FILES[key]).read_text(encoding="utf-8"))


def make_graph(
    node_specs: list[dict],
    edge_pairs: list[tuple[str, str]] | None = None,
    *,
    name: str = "Tiny",
    total_params: int | None = None,
    total_flops: int | None = None,
) -> dict:
    """Build a small schema-shaped graph for edge-case tests."""
    nodes: list[dict] = []
    for index, spec in enumerate(node_specs):
        nodes.append(
            {
                "id": f"n{index}",
                "name": spec.get("name", f"n{index}"),
                "op": spec.get("op", "Conv2d"),
                "kind": spec.get("kind", "conv"),
                "modulePath": spec.get("modulePath", ""),
                "depth": 0,
                "inShapes": [list(shape) for shape in spec.get("inShapes", [])],
                "outShapes": [list(shape) for shape in spec.get("outShapes", [])],
                "params": dict(spec.get("params", {})),
                "numParams": spec.get("numParams", 0),
                "flops": spec.get("flops", 0),
                "tags": [],
                "groupId": spec.get("groupId"),
                "source": spec.get("source"),
                "notes": [],
            }
        )
    edges = []
    for index, (source, target) in enumerate(edge_pairs or []):
        edges.append({"id": f"e{index}", "source": source, "target": target, "kind": "data", "shapes": [], "label": ""})
    if total_params is None:
        total_params = sum(node["numParams"] for node in nodes)
    if total_flops is None:
        total_flops = sum(node["flops"] for node in nodes)
    return {
        "schemaVersion": "1.0",
        "generator": {"name": "netprism", "version": "0.1.0"},
        "model": {
            "name": name,
            "framework": "pytorch",
            "className": name,
            "mode": "trace",
            "source": {"file": "tiny.py", "relPath": "tiny.py"},
            "inputSpec": [{"name": "x", "shape": [1, 8, 32, 32], "dtype": "float32"}],
            "outputSpec": [{"name": "y", "shape": [1, 4, 32, 32], "dtype": "float32"}],
            "totalParams": total_params,
            "trainableParams": total_params,
            "totalFlops": total_flops,
            "flopsApprox": False,
            "numLayers": len(nodes),
            "warnings": [],
        },
        "nodes": nodes,
        "edges": edges,
        "groups": [],
    }


def conv_chain(count: int, *, kernel: int = 3, stride: int = 1, dilation: int = 1) -> dict:
    """Input + ``count`` 3x3-style convs, one data edge between neighbours."""
    shape = [1, 8, 32, 32]
    nodes = [{"kind": "input", "op": "Input", "outShapes": [shape]}]
    for index in range(count):
        nodes.append(
            {
                "kind": "conv",
                "op": "Conv2d",
                "modulePath": f"conv{index}",
                "params": {
                    "kernelSize": [kernel, kernel],
                    "stride": [stride, stride],
                    "dilation": [dilation, dilation],
                },
                "outShapes": [shape],
            }
        )
    edges = [(f"n{index}", f"n{index + 1}") for index in range(count)]
    return make_graph(nodes, edges)


def _node_metrics(graph: dict) -> dict[str, dict]:
    return {node["id"]: node.get("metrics", {}) for node in graph["nodes"]}


# --------------------------------------------------------------------------- #
# metrics: demo graphs
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("key", ["vgg", "unet", "resnet", "transformer"])
def test_enrich_demo_graph_metrics(key: str) -> None:
    graph = load_demo(key)
    assert validate_graph(graph) == []
    assert enrich_graph(graph) is graph
    assert validate_graph(graph) == []

    model = graph["model"]
    metrics = model["metrics"]
    assert metrics["paramBytes"] == model["totalParams"] * 4
    assert metrics["macs"] == model["totalFlops"] // 2
    assert metrics["maxReceptiveField"] >= 1
    assert 0 <= metrics["peakActivationBytes"] <= metrics["activationBytes"]
    for field in ("paramBytes", "activationBytes", "peakActivationBytes", "maxReceptiveField", "macs"):
        assert isinstance(metrics[field], int) and not isinstance(metrics[field], bool)
        assert metrics[field] >= 0

    expected_activation = 0
    for node in graph["nodes"]:
        assert set(node["metrics"]) == {"activationBytes", "receptiveField", "macs"}
        assert node["metrics"]["macs"] == node["flops"] // 2
        assert isinstance(node["metrics"]["receptiveField"], int)
        assert node["metrics"]["receptiveField"] >= 1
        first_output = node["outShapes"][0]
        node_bytes = math.prod(first_output) * 4
        assert node["metrics"]["activationBytes"] == node_bytes
        expected_activation += node_bytes
    assert metrics["activationBytes"] == expected_activation


def test_node_activation_bytes_hand_calculation() -> None:
    graph = load_demo("vgg")
    enrich_graph(graph)
    node = next(node for node in graph["nodes"] if node["name"] == "features.0")
    assert node["outShapes"][0] == [1, 16, 64, 64]
    assert node["metrics"]["activationBytes"] == 1 * 16 * 64 * 64 * 4 == 262_144


def test_model_metrics_for_mlp_has_no_spatial_growth() -> None:
    graph = load_demo("mlp")
    enrich_graph(graph)
    assert graph["model"]["metrics"]["maxReceptiveField"] == 1
    assert graph["model"]["metrics"]["activationBytes"] == sum(
        math.prod(node["outShapes"][0]) * 4 for node in graph["nodes"]
    )


def test_unknown_dimensions_are_estimated_conservatively() -> None:
    graph = make_graph(
        [
            {"kind": "input", "op": "Input", "outShapes": [[-1, 3, 8, 8]]},
            {"kind": "conv", "outShapes": [[-1, 4, 8, 8]], "params": {"kernelSize": [3, 3]}},
        ],
        [("n0", "n1")],
    )
    enrich_graph(graph)
    assert graph["nodes"][0]["metrics"]["activationBytes"] == 1 * 3 * 8 * 8 * 4
    assert graph["nodes"][1]["metrics"]["activationBytes"] == 1 * 4 * 8 * 8 * 4
    assert graph["model"]["metrics"]["activationBytes"] == (3 * 8 * 8 + 4 * 8 * 8) * 4


# --------------------------------------------------------------------------- #
# metrics: receptive field propagation
# --------------------------------------------------------------------------- #
def test_receptive_field_three_consecutive_convs() -> None:
    graph = conv_chain(3)
    enrich_graph(graph)
    metrics = _node_metrics(graph)
    assert [metrics[f"n{i}"]["receptiveField"] for i in range(4)] == [1, 3, 5, 7]
    assert graph["model"]["metrics"]["maxReceptiveField"] == 7


def test_receptive_field_respects_stride_and_dilation() -> None:
    graph = conv_chain(3)
    graph["nodes"][1]["params"]["stride"] = [2, 2]  # rf 3, jump 2
    enrich_graph(graph)
    metrics = _node_metrics(graph)
    assert [metrics[f"n{i}"]["receptiveField"] for i in range(4)] == [1, 3, 7, 11]

    dilated = conv_chain(1, dilation=2)
    enrich_graph(dilated)
    assert _node_metrics(dilated)["n1"]["receptiveField"] == 5  # 1 + (3-1)*2
    assert dilated["model"]["metrics"]["maxReceptiveField"] == 5


def test_receptive_field_branch_takes_maximum() -> None:
    graph = make_graph(
        [
            {"kind": "input", "op": "Input", "outShapes": [[1, 8, 32, 32]]},
            {
                "kind": "conv",
                "modulePath": "conv",
                "params": {"kernelSize": [3, 3], "stride": [1, 1]},
                "outShapes": [[1, 8, 32, 32]],
            },
            {
                "kind": "pool",
                "modulePath": "pool",
                "op": "MaxPool2d",
                "params": {"kernelSize": [2, 2], "stride": [2, 2]},
                "outShapes": [[1, 8, 16, 16]],
            },
            {"kind": "add", "modulePath": "add", "op": "add", "outShapes": [[1, 8, 32, 32]]},
            {
                "kind": "conv",
                "modulePath": "conv2",
                "params": {"kernelSize": [3, 3], "stride": [1, 1]},
                "outShapes": [[1, 8, 32, 32]],
            },
        ],
        [("n0", "n1"), ("n0", "n2"), ("n1", "n3"), ("n2", "n3"), ("n3", "n4")],
    )
    enrich_graph(graph)
    metrics = _node_metrics(graph)
    assert metrics["n1"]["receptiveField"] == 3
    assert metrics["n2"]["receptiveField"] == 2
    assert metrics["n3"]["receptiveField"] == 3  # max(3, 2), not the sum
    assert metrics["n4"]["receptiveField"] == 7  # 3 + (3-1)*2
    assert graph["model"]["metrics"]["maxReceptiveField"] == 7


def test_receptive_field_survives_cycle() -> None:
    graph = make_graph(
        [
            {"kind": "conv", "modulePath": "a", "params": {"kernelSize": [3, 3]}, "outShapes": [[1, 4, 8, 8]]},
            {"kind": "conv", "modulePath": "b", "params": {"kernelSize": [3, 3]}, "outShapes": [[1, 4, 8, 8]]},
        ],
        [("n0", "n1"), ("n1", "n0")],
    )
    enrich_graph(graph)  # must not raise
    metrics = _node_metrics(graph)
    assert metrics["n0"]["receptiveField"] >= 1
    assert metrics["n1"]["receptiveField"] >= 1
    assert graph["model"]["metrics"]["maxReceptiveField"] >= 1


def test_unknown_kind_falls_back_to_one_conservatively() -> None:
    graph = make_graph(
        [
            {"kind": "input", "op": "Input", "outShapes": [[1, 4, 8, 8]]},
            {"kind": "conv", "op": "Conv2d", "params": {"kernelSize": [3, 3]}, "outShapes": [[1, 4, 8, 8]]},
            {"kind": "unknown", "op": "Mystery", "outShapes": [[1, 4, 8, 8]]},
            {"kind": "conv", "op": "Conv2d", "params": {"kernelSize": [3, 3]}, "outShapes": [[1, 4, 8, 8]]},
        ],
        [("n0", "n1"), ("n1", "n2"), ("n2", "n3")],
    )
    enrich_graph(graph)
    metrics = _node_metrics(graph)
    assert metrics["n1"]["receptiveField"] == 3
    assert metrics["n2"]["receptiveField"] == 1  # conservative reset
    assert metrics["n3"]["receptiveField"] == 3  # 1 + (3-1)*1
    assert graph["model"]["metrics"]["maxReceptiveField"] == 3


def test_peak_activation_uses_layer_liveness() -> None:
    graph = make_graph(
        [
            {"kind": "input", "op": "Input", "outShapes": [[25]]},
            {"kind": "conv", "outShapes": [[25]]},
            {"kind": "conv", "outShapes": [[25]]},
        ],
        [("n0", "n1"), ("n1", "n2")],
    )
    enrich_graph(graph)
    assert graph["model"]["metrics"]["activationBytes"] == 300
    assert graph["model"]["metrics"]["peakActivationBytes"] == 200


# --------------------------------------------------------------------------- #
# metrics: preservation and layer rows
# --------------------------------------------------------------------------- #
def test_enrich_preserves_existing_metrics_unless_forced() -> None:
    graph = load_demo("vgg")
    graph["model"]["metrics"] = {"paramBytes": 7}
    graph["nodes"][1]["metrics"] = {"activationBytes": 5, "receptiveField": 9, "macs": 1}

    enrich_graph(graph)
    assert graph["model"]["metrics"] == {"paramBytes": 7}
    assert graph["nodes"][1]["metrics"] == {"activationBytes": 5, "receptiveField": 9, "macs": 1}

    enrich_graph(graph, force=True)
    assert graph["model"]["metrics"]["paramBytes"] == graph["model"]["totalParams"] * 4
    assert graph["nodes"][1]["metrics"]["activationBytes"] == 262_144


def test_layer_rows_shape_and_content() -> None:
    graph = load_demo("vgg")
    rows = layer_rows(graph)
    assert len(rows) == len(graph["nodes"])
    assert list(rows[0]) == LAYER_COLUMNS
    conv = next(row for row in rows if row["name"] == "features.0")
    node = next(node for node in graph["nodes"] if node["name"] == "features.0")
    assert conv["op"] == "Conv2d"
    assert conv["kind"] == "conv"
    assert conv["modulePath"] == "features.0"
    assert conv["inShapes"] == [[1, 3, 64, 64]]
    assert conv["outShapes"] == [[1, 16, 64, 64]]
    assert conv["numParams"] == 432
    assert conv["flops"] == 3_538_944
    assert conv["activationBytes"] == 262_144
    assert conv["receptiveField"] == 3
    assert conv["groupId"] == "g2"
    assert conv["groupLabel"] == "Features"
    assert conv["source"] == node["source"]


def test_layer_rows_works_without_metrics_and_does_not_mutate() -> None:
    graph = load_demo("vgg")
    graph["model"].pop("metrics", None)
    for node in graph["nodes"]:
        node.pop("metrics", None)
    rows = layer_rows(graph)  # computes fallbacks in memory only
    assert "metrics" not in graph["model"]
    assert all("metrics" not in node for node in graph["nodes"])
    conv = next(row for row in rows if row["name"] == "features.0")
    assert conv["activationBytes"] == 262_144
    assert conv["receptiveField"] == 3


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def test_report_markdown_summary_groups_and_layers() -> None:
    graph = load_demo("vgg")
    text = render_report(graph, "md")
    assert text.startswith("# VGGMini 分析报告")
    assert "**Framework** pytorch" in text
    assert "**Mode** trace" in text
    assert "**Params** 66,474" in text
    assert "**FLOPs** 63,230,666" in text
    assert "**Activation (估算)**" in text
    assert "**Receptive field (估算)**" in text
    assert "**Warnings** 0" in text
    assert "## 分组汇总 (Groups)" in text
    assert "## 逐层明细 (Layers)" in text
    assert "| Features | conv_block |" in text
    assert "features.0" in text
    assert "估算说明" in text
    assert text.rstrip().endswith("。") or text.rstrip().endswith("。\n")


def test_report_csv_matches_layer_rows_columns() -> None:
    graph = load_demo("vgg")
    text = render_report(graph, "csv")
    assert text.startswith("\ufeff")
    parsed = list(csv.reader(io.StringIO(text.lstrip("\ufeff"))))
    comments = [row for row in parsed if row and row[0].startswith("#")]
    rows = [row for row in parsed if not (row and row[0].startswith("#"))]
    assert len(comments) == 1 and "VGGMini" in comments[0][0]
    assert rows[0] == LAYER_COLUMNS
    assert rows[0] == list(layer_rows(graph)[0])
    assert len(rows) == len(graph["nodes"]) + 1
    assert all(len(row) == len(LAYER_COLUMNS) for row in rows)
    conv = next(row for row in rows[1:] if row[1] == "features.0")
    assert conv[2] == "Conv2d"
    assert conv[4] == "features.0"
    assert conv[5] == "1x3x64x64"
    assert conv[6] == "1x16x64x64"
    assert conv[7] == "432"
    assert conv[9] == "262144"
    assert conv[10] == "3"
    assert conv[13].startswith("samples/cnn_vgg.py:")
    assert "—" not in text  # CSV keeps missing values empty/ASCII for Excel


def test_report_html_is_self_contained_and_scrollable() -> None:
    graph = load_demo("vgg")
    text = render_report(graph, "html")
    assert text.startswith("<!DOCTYPE html>")
    assert "<style>" in text
    assert "overflow" in text
    assert "max-height" in text
    assert "VGGMini" in text
    assert "features.0" in text
    assert "<link" not in text
    assert "<script" not in text
    assert "http://" not in text and "https://" not in text


def test_report_html_escapes_dynamic_text() -> None:
    graph = load_demo("mlp")
    graph["model"]["name"] = "<script>alert(1)</script>"
    graph["nodes"][1]["name"] = "a|b"
    html_text = render_report(graph, "html")
    assert "<script>alert" not in html_text
    assert "&lt;script&gt;" in html_text
    assert "a|b" in html_text  # pipes are safe in HTML
    md_text = render_report(graph, "md")
    assert "a\\|b" in md_text


@pytest.mark.parametrize("fmt", ["md", "csv", "html"])
def test_write_report_writes_utf8_and_csv_bom(tmp_path: Path, fmt: str) -> None:
    graph = load_demo("vgg")
    path = tmp_path / f"report.{fmt}"
    write_report(graph, path, fmt)
    raw = path.read_bytes()
    assert raw
    assert (raw[:3] == b"\xef\xbb\xbf") == (fmt == "csv")
    text = path.read_text(encoding="utf-8-sig")
    assert "features.0" in text
    assert "\ufffd" not in text
    text.encode("utf-8")


def test_report_renders_without_precomputed_metrics() -> None:
    graph = load_demo("unet")
    for node in graph["nodes"]:
        node.pop("metrics", None)
    graph["model"].pop("metrics", None)

    md_text = render_report(graph, "md")
    assert "UNetMini" in md_text
    assert "**Activation (估算)**" in md_text
    assert graph["model"]["metrics"]["activationBytes"] > 0  # rendered report fills metrics

    csv_text = render_report(graph, "csv")
    assert csv_text.startswith("\ufeff")
    html_text = render_report(graph, "html")
    assert "UNetMini" in html_text


def test_empty_report_and_bad_format() -> None:
    for fmt in ("md", "csv", "html"):
        assert isinstance(render_report({}, fmt), str)
    with pytest.raises(ValueError):
        render_report({}, "pdf")


# --------------------------------------------------------------------------- #
# report: algorithm-understanding meta (read-only)
# --------------------------------------------------------------------------- #
def _graph_with_algo_meta() -> dict:
    graph = load_demo("vgg")
    patterns = [
        {
            "id": "p1",
            "kind": "residual_block",
            "label": "Residual Block",
            "formula": "y = F(x) + x",
            "confidence": 0.95,
            "evidence": ["add 节点有跨层 skip 边"],
        }
    ]
    for index in range(1, 12):
        patterns.append(
            {
                "id": f"p{index + 1}",
                "kind": "conv_bn_act",
                "label": f"Pattern {index}",
                "formula": f"y = f{index}(x)",
                "confidence": 0.9,
                "evidence": [f"evidence {index}"],
            }
        )
    graph["model"]["meta"] = {
        "family": "resnet",
        "familyLabel": "ResNet",
        "algorithmTags": ["cnn", "residual"],
        "patterns": patterns,
        "pseudocode": [
            "x = Input(1,3,64,64)",
            "x = ResidualBlock(16)(x)   # -> (1,16,64,64)",
            "logits = Linear(10)(x)",
        ],
    }
    return graph


def test_report_markdown_includes_algorithm_understanding() -> None:
    graph = _graph_with_algo_meta()
    text = render_report(graph, "md")
    assert "## 算法理解" in text
    assert "**家族 (Family)** ResNet" in text
    assert "y = F(x) + x" in text
    assert "0.95" in text
    assert "add 节点有跨层 skip 边" in text
    assert "x = Input(1,3,64,64)" in text
    assert "ResidualBlock(16)" in text
    # After the summary and before the group table.
    assert text.index("**Activation (估算)**") < text.index("## 算法理解") < text.index("## 分组汇总")
    # At most 10 pattern rows are rendered.
    assert "Pattern 9" in text
    assert "Pattern 10" not in text


def test_report_html_includes_algorithm_understanding() -> None:
    graph = _graph_with_algo_meta()
    text = render_report(graph, "html")
    assert 'id="algo-view"' in text
    assert 'class="badge"' in text
    assert "ResNet" in text
    assert "y = F(x) + x" in text
    assert '<pre class="pseudo" id="algo-pseudocode">' in text
    assert "x = Input(1,3,64,64)" in text
    assert "Pattern 9" in text
    assert "Pattern 10" not in text
    assert "<link" not in text
    assert "<script" not in text
    assert "http://" not in text and "https://" not in text


def test_report_algorithm_html_escapes_all_fields() -> None:
    graph = _graph_with_algo_meta()
    pattern = graph["model"]["meta"]["patterns"][0]
    pattern["label"] = "<script>alert(1)</script>"
    pattern["formula"] = "y = a < b & c"
    pattern["evidence"] = ["<img src=x onerror=alert(1)>"]
    graph["model"]["meta"]["pseudocode"] = ["x = Input(1,3,64,64) # <b>raw</b>"]

    text = render_report(graph, "html")
    assert "<script>alert" not in text
    assert "&lt;script&gt;" in text
    assert "y = a &lt; b &amp; c" in text
    assert "&lt;img src=x onerror=alert(1)&gt;" in text
    assert "&lt;b&gt;raw&lt;/b&gt;" in text


def test_report_csv_keeps_layer_columns_with_algorithm_meta() -> None:
    graph = _graph_with_algo_meta()
    text = render_report(graph, "csv")
    parsed = list(csv.reader(io.StringIO(text.lstrip("\ufeff"))))
    comments = [row for row in parsed if row and row[0].startswith("#")]
    rows = [row for row in parsed if not (row and row[0].startswith("#"))]
    assert len(comments) == 1
    assert rows[0] == LAYER_COLUMNS
    assert all(len(row) == len(LAYER_COLUMNS) for row in rows)
    assert len(rows) == len(graph["nodes"]) + 1
    assert "算法理解" not in text
    assert "ResNet" not in text  # algo meta must not add CSV columns


@pytest.mark.parametrize(
    "meta",
    [
        None,
        {},
        {"patterns": []},
        {"family": None, "familyLabel": None, "patterns": [], "pseudocode": []},
    ],
)
def test_report_without_algorithm_meta_keeps_old_output(meta) -> None:
    graph = load_demo("vgg")
    graph["model"].pop("meta", None)
    if meta is not None:
        graph["model"]["meta"] = meta

    md_text = render_report(graph, "md")
    html_text = render_report(graph, "html")
    csv_text = render_report(graph, "csv")
    assert "算法理解" not in md_text
    assert "算法理解" not in html_text
    assert "算法理解" not in csv_text
    assert "## 分组汇总 (Groups)" in md_text
    assert "features.0" in md_text
    assert "<style>" in html_text and "features.0" in html_text
    assert csv_text.startswith("\ufeff")
    assert LAYER_COLUMNS[0] in csv_text


def test_report_algorithm_section_without_family() -> None:
    graph = load_demo("vgg")
    graph["model"]["meta"] = {
        "patterns": [
            {"label": "Block", "kind": "residual_block", "formula": "y = F(x) + x", "confidence": 0.8, "evidence": ["add"]}
        ],
        "pseudocode": [],
    }
    md_text = render_report(graph, "md")
    assert "## 算法理解" in md_text
    assert "y = F(x) + x" in md_text
    assert "**家族 (Family)**" not in md_text
    assert "Forward 伪代码" not in md_text  # empty pseudocode is skipped

    html_text = render_report(graph, "html")
    assert 'id="algo-patterns"' in html_text
    assert 'id="algo-family"' not in html_text
    assert 'id="algo-pseudocode"' not in html_text


# --------------------------------------------------------------------------- #
# compare
# --------------------------------------------------------------------------- #
def _compare_pair() -> tuple[dict, dict]:
    a = make_graph(
        [
            {"kind": "input", "op": "Input", "name": "x", "outShapes": [[1, 4, 8, 8]]},
            {
                "kind": "conv",
                "name": "conv",
                "modulePath": "block.conv",
                "numParams": 100,
                "flops": 200,
                "outShapes": [[1, 4, 8, 8]],
            },
            {"kind": "activation", "name": "act", "modulePath": "block.act", "op": "ReLU", "outShapes": [[1, 4, 8, 8]]},
            {"kind": "linear", "name": "head", "modulePath": "head", "numParams": 50, "flops": 100, "outShapes": [[1, 4]]},
        ],
        [("n0", "n1"), ("n1", "n2"), ("n2", "n3")],
        name="ModelA",
        total_params=150,
        total_flops=300,
    )
    b = make_graph(
        [
            {"kind": "input", "op": "Input", "name": "x", "outShapes": [[1, 4, 8, 8]]},
            {
                "kind": "conv",
                "name": "conv",
                "modulePath": "block.conv",
                "numParams": 120,
                "flops": 220,
                "outShapes": [[1, 4, 4, 4]],
            },
            {"kind": "norm", "name": "bn", "modulePath": "block.bn", "op": "BatchNorm2d", "numParams": 8, "outShapes": [[1, 4, 4, 4]]},
            {"kind": "linear", "name": "head", "modulePath": "head", "numParams": 50, "flops": 100, "outShapes": [[1, 4]]},
        ],
        [("n0", "n1"), ("n1", "n2"), ("n2", "n3")],
        name="ModelB",
        total_params=178,
        total_flops=340,
    )
    return a, b


def test_compare_classifies_added_removed_changed() -> None:
    a, b = _compare_pair()
    result = compare_graphs(a, b)

    assert result["counts"] == {"added": 1, "removed": 1, "changed": 1, "unchanged": 2}
    assert [row["modulePath"] for row in result["added"]] == ["block.bn"]
    assert [row["modulePath"] for row in result["removed"]] == ["block.act"]
    changed = result["changed"][0]
    assert changed["key"] == "block.conv"
    assert changed["fields"]["numParams"] == {"a": 100, "b": 120}
    assert changed["fields"]["outShapes"] == {"a": [[1, 4, 8, 8]], "b": [[1, 4, 4, 4]]}

    assert result["summary"]["params"] == {"a": 150, "b": 178, "diff": 28}
    assert result["summary"]["flops"] == {"a": 300, "b": 340, "diff": 40}
    assert result["summary"]["maxReceptiveField"]["a"] >= 1
    assert result["a"]["name"] == "ModelA"
    assert result["b"]["name"] == "ModelB"
    json.dumps(result)  # must be plain JSON


def test_compare_detects_op_change() -> None:
    a, b = _compare_pair()
    b["nodes"][1]["op"] = "DepthwiseConv2d"
    fields = compare_graphs(a, b)["changed"][0]["fields"]
    assert fields["op"] == {"a": "Conv2d", "b": "DepthwiseConv2d"}


def test_compare_falls_back_to_name_when_module_path_missing() -> None:
    a = make_graph(
        [
            {"kind": "input", "op": "Input", "name": "x", "outShapes": [[1, 8]]},
            {"kind": "linear", "name": "block_a", "numParams": 10, "outShapes": [[1, 8]]},
            {"kind": "linear", "name": "head", "numParams": 20, "outShapes": [[1, 2]]},
        ],
        [("n0", "n1"), ("n1", "n2")],
        name="A",
    )
    b = copy.deepcopy(a)
    b["model"]["name"] = "B"
    b["nodes"][1]["numParams"] = 15

    result = compare_graphs(a, b)
    assert result["counts"] == {"added": 0, "removed": 0, "changed": 1, "unchanged": 2}
    assert result["changed"][0]["key"] == "block_a"
    assert result["changed"][0]["fields"] == {"numParams": {"a": 10, "b": 15}}


def test_compare_matches_duplicate_module_paths() -> None:
    graph = load_demo("resnet")
    result = compare_graphs(graph, copy.deepcopy(graph))
    assert result["counts"] == {"added": 0, "removed": 0, "changed": 0, "unchanged": len(graph["nodes"])}


def test_compare_and_render_empty_graphs() -> None:
    result = compare_graphs({}, {})
    assert result["counts"] == {"added": 0, "removed": 0, "changed": 0, "unchanged": 0}
    assert result["added"] == [] and result["removed"] == [] and result["changed"] == []
    json.dumps(result)

    md_text = render_compare(result, "md")
    assert "无新增节点" in md_text and "无变更节点" in md_text
    assert json.loads(render_compare(result, "json")) == result
    csv_text = render_compare(result, "csv")
    assert csv_text.startswith("\ufeff")
    assert len(list(csv.reader(io.StringIO(csv_text.lstrip("\ufeff"))))) == 1
    with pytest.raises(ValueError):
        render_compare(result, "xml")

    # Missing/None documents must not crash either.
    assert compare_graphs(None, None)["counts"]["unchanged"] == 0
    assert isinstance(render_compare(None, "md"), str)


def test_render_compare_md_and_csv() -> None:
    a, b = _compare_pair()
    result = compare_graphs(a, b)

    md_text = render_compare(result, "md")
    assert "ModelA" in md_text and "ModelB" in md_text
    assert "block.bn" in md_text
    assert "block.act" in md_text
    assert "numParams: 100 → 120" in md_text
    assert "新增节点 (Added: 1)" in md_text

    csv_text = render_compare(result, "csv")
    assert csv_text.startswith("\ufeff")
    rows = list(csv.reader(io.StringIO(csv_text.lstrip("\ufeff"))))
    assert len(rows[0]) == 17
    assert {row[0] for row in rows[1:]} == {"added", "removed", "changed"}
    assert all(len(row) == 17 for row in rows)
    assert "numParams" in rows[-1][-1]

    header = {name: index for index, name in enumerate(rows[0])}
    by_status = {row[0]: row for row in rows[1:]}
    added, removed, changed = by_status["added"], by_status["removed"], by_status["changed"]
    # added: values on the B side only.
    assert added[header["op_a"]] == "" and added[header["op_b"]] == "BatchNorm2d"
    assert added[header["kind_a"]] == "" and added[header["kind_b"]] == "norm"
    assert added[header["numParams_a"]] == "" and added[header["numParams_b"]] == "8"
    # removed: values on the A side only (regression: previously leaked to B).
    assert removed[header["op_a"]] == "ReLU" and removed[header["op_b"]] == ""
    assert removed[header["kind_a"]] == "activation" and removed[header["kind_b"]] == ""
    assert removed[header["numParams_a"]] == "0" and removed[header["numParams_b"]] == ""
    assert removed[header["flops_b"]] == "" and removed[header["activationBytes_b"]] == ""
    # changed: both sides present for the compared fields.
    assert changed[header["numParams_a"]] == "100" and changed[header["numParams_b"]] == "120"


def test_compare_csv_removed_rows_use_a_side_for_demo_graphs() -> None:
    """Regression for QA's VGG-vs-ResNet CSV repro."""
    result = compare_graphs(load_demo("vgg"), load_demo("resnet"))
    text = render_compare(result, "csv")
    rows = list(csv.reader(io.StringIO(text.lstrip("\ufeff"))))
    header = {name: index for index, name in enumerate(rows[0])}
    removed = [row for row in rows[1:] if row[0] == "removed"]
    added = [row for row in rows[1:] if row[0] == "added"]
    assert removed and added
    for row in removed:
        for name in ("op_a", "kind_a", "numParams_a", "flops_a", "activationBytes_a", "receptiveField_a"):
            assert row[header[name]] != "", f"{name} missing on A side: {row}"
        for name in ("op_b", "kind_b", "numParams_b", "flops_b", "activationBytes_b", "receptiveField_b"):
            assert row[header[name]] == "", f"{name} leaked to B side: {row}"
    for row in added:
        for name in ("op_b", "kind_b", "numParams_b", "flops_b", "activationBytes_b", "receptiveField_b"):
            assert row[header[name]] != "", f"{name} missing on B side: {row}"
        for name in ("op_a", "kind_a", "numParams_a", "flops_a", "activationBytes_a", "receptiveField_a"):
            assert row[header[name]] == "", f"{name} leaked to A side: {row}"


def test_write_compare_all_formats(tmp_path: Path) -> None:
    a, b = _compare_pair()
    result = compare_graphs(a, b)
    for fmt in ("md", "json", "csv"):
        path = tmp_path / f"compare.{fmt}"
        write_compare(result, path, fmt)
        raw = path.read_bytes()
        assert raw
        assert (raw[:3] == b"\xef\xbb\xbf") == (fmt == "csv")
    assert json.loads((tmp_path / "compare.json").read_text(encoding="utf-8"))["counts"]["changed"] == 1


# --------------------------------------------------------------------------- #
# purity
# --------------------------------------------------------------------------- #
def test_analysis_modules_do_not_import_torch_or_onnx() -> None:
    import netprism.compare as compare_module
    import netprism.metrics as metrics_module
    import netprism.report as report_module

    for module in (metrics_module, report_module, compare_module):
        source = inspect.getsource(module)
        assert "import torch" not in source
        assert "from torch" not in source
        assert "import onnx" not in source
        assert "from onnx" not in source
