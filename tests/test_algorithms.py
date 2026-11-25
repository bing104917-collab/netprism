"""Tests for the NetPrism algorithm understanding layer.

Covers the five SPEC-ALGO sample models, the node formula catalogue, the
structural pattern detectors (including non-mainstream architectures), the
knowledge-base driven family classifier, idempotency/JSON guarantees and the
"no torch/onnx import" contract.
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from netprism import algorithms
from netprism.algorithms import (
    classify_families,
    enrich_algorithms,
    explain_graph,
    node_formula,
    render_explain_md,
)

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "examples" / "demo"

VGG = "cnn_vgg__vggmini.json"
RESNET = "resnet_mini__resnetmini.json"
UNET = "unet_mini__unetmini.json"
TINYGPT = "custom_attention__tinygpt.json"
TRANSFORMER = "transformer_classifier__tinytransformerclassifier.json"
SAMPLES = [VGG, RESNET, UNET, TINYGPT, TRANSFORMER]


@pytest.fixture(autouse=True)
def _reset_knowledge_cache():
    """Make ``_knowledge()`` re-resolve per test (monkeypatched KBs work)."""
    algorithms._KB_CACHE.update(resolved=False, data=None)
    yield
    algorithms._KB_CACHE.update(resolved=False, data=None)


def load_demo(filename: str) -> dict:
    return json.loads((DEMO / filename).read_text(encoding="utf-8"))


def sample_graph(filename: str) -> dict:
    graph = load_demo(filename)
    enrich_algorithms(graph, force=True)
    return graph


def pattern_kinds(graph: dict) -> set[str]:
    return {pattern["kind"] for pattern in explain_graph(graph)["patterns"]}


def find_pattern(graph: dict, kind: str) -> dict:
    for pattern in explain_graph(graph)["patterns"]:
        if pattern["kind"] == kind:
            return pattern
    raise AssertionError(f"pattern {kind!r} not detected in {explain_graph(graph)['family']}")


# --------------------------------------------------------------------------- #
# synthetic graph factory
# --------------------------------------------------------------------------- #
def N(
    node_id: str,
    kind: str,
    op: str | None = None,
    *,
    name: str | None = None,
    module: str | None = None,
    in_shape=None,
    out_shape=None,
    params: dict | None = None,
    group: str | None = "g1",
    tags=None,
) -> dict:
    module = module or name or f"{kind}.{node_id}"
    return {
        "id": node_id,
        "name": name or module,
        "op": op or kind,
        "kind": kind,
        "depth": 0,
        "modulePath": module,
        "inShapes": [list(in_shape)] if in_shape else [],
        "outShapes": [list(out_shape)] if out_shape else [],
        "params": params or {},
        "numParams": 0,
        "flops": 0,
        "tags": list(tags or []),
        "groupId": group,
        "notes": [],
        "source": None,
    }


def _make_edge(edge_id: str, source: str, target: str, kind: str = "data") -> dict:
    return {"id": edge_id, "source": source, "target": target, "kind": kind, "label": "", "shapes": []}


def make_graph(
    name: str,
    body: list[dict],
    edges,
    *,
    groups: list[dict] | None = None,
    class_name: str | None = None,
    input_shape=(1, 8),
    output_shape=(1, 8),
) -> dict:
    """Assemble a schema-shaped graph around ``body`` nodes.

    ``n0`` is the input and ``n{len(body)+1}`` the output; edges are given as
    ``(source, target)`` or ``(source, target, kind)`` tuples.
    """
    total = len(body)
    input_node = N("n0", "input", "Input", out_shape=list(input_shape), group=None)
    output_node = N(f"n{total + 1}", "output", "Output", out_shape=list(output_shape), group=None)
    nodes = [input_node] + body + [output_node]

    if groups is None:
        group_ids = []
        for node in body:
            gid = node.get("groupId")
            if gid and gid not in group_ids:
                group_ids.append(gid)
        groups = [
            {
                "id": gid,
                "name": gid,
                "label": gid.replace("_", " ").title(),
                "kind": "module",
                "parentId": "g0",
                "depth": 1,
                "nodeIds": [node["id"] for node in body if node.get("groupId") == gid],
                "childGroupIds": [],
                "numParams": 0,
                "flops": 0,
                "source": None,
            }
            for gid in group_ids
        ]
    root = {
        "id": "g0",
        "name": class_name or name,
        "label": name,
        "kind": "model",
        "parentId": None,
        "depth": 0,
        "nodeIds": [],
        "childGroupIds": [group["id"] for group in groups],
        "numParams": 0,
        "flops": 0,
        "source": None,
    }
    edge_docs = []
    for index, spec in enumerate(edges):
        if isinstance(spec, str):
            source, target, kind = spec, None, "data"
        elif len(spec) == 2:
            (source, target), kind = spec, "data"
        else:
            source, target, kind = spec
        edge_docs.append(_make_edge(f"e{index}", source, target, kind))
    model = {
        "name": name,
        "framework": "pytorch",
        "className": class_name or name,
        "mode": "trace",
        "source": {"file": "toy.py", "relPath": "toys/toy.py"},
        "inputSpec": [{"name": "x", "shape": list(input_shape), "dtype": "float32"}],
        "outputSpec": [{"name": "output", "shape": list(output_shape), "dtype": "float32"}],
        "totalParams": 0,
        "trainableParams": 0,
        "totalFlops": 0,
        "flopsApprox": True,
        "numLayers": len(nodes),
        "warnings": [],
    }
    return {
        "schemaVersion": "1.0",
        "generator": {"name": "netprism", "version": "0.1.0"},
        "model": model,
        "nodes": nodes,
        "groups": [root] + groups,
        "edges": edge_docs,
        "warnings": [],
    }


def conv(
    node_id: str,
    cin: int,
    cout: int,
    *,
    kernel=3,
    stride=1,
    groups: int = 1,
    padding: int = 1,
    in_shape=(1, 8, 16, 16),
    out_shape=(1, 8, 16, 16),
    group="g1",
) -> dict:
    k = [kernel, kernel] if isinstance(kernel, int) else list(kernel)
    s = [stride, stride] if isinstance(stride, int) else list(stride)
    return N(
        node_id,
        "conv",
        "Conv2d",
        in_shape=in_shape,
        out_shape=out_shape,
        params={
            "inChannels": cin,
            "outChannels": cout,
            "kernelSize": k,
            "stride": s,
            "padding": [padding, padding],
            "dilation": [1, 1],
            "groups": groups,
            "bias": False,
        },
        group=group,
    )


def bn(node_id: str, channels: int, in_shape, out_shape, group="g1") -> dict:
    return N(
        node_id,
        "norm",
        "BatchNorm2d",
        in_shape=in_shape,
        out_shape=out_shape,
        params={"normalizedShape": [channels], "affine": True},
        group=group,
    )


def act(node_id: str, op: str, shape, group="g1") -> dict:
    return N(node_id, "activation", op, in_shape=shape, out_shape=shape, group=group)


def linear(
    node_id: str,
    cin: int,
    cout: int,
    *,
    in_shape=(1, 8),
    out_shape=(1, 8),
    bias: bool = True,
    group="g1",
    module: str | None = None,
) -> dict:
    return N(
        node_id,
        "linear",
        "Linear",
        module=module,
        in_shape=in_shape,
        out_shape=out_shape,
        params={"inFeatures": cin, "outFeatures": cout, "bias": bias},
        group=group,
    )


def assert_has_pattern(graph: dict, kind: str, *, min_confidence: float = 0.5) -> dict:
    enrich_algorithms(graph, force=True)
    pattern = find_pattern(graph, kind)
    assert pattern["confidence"] >= min_confidence
    assert pattern["nodeIds"]
    assert pattern["evidence"]
    return pattern


# --------------------------------------------------------------------------- #
# sample models: family / patterns / pseudo / node meta
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "filename, expected",
    [
        (VGG, {"vgg", "cnn"}),
        (RESNET, {"resnet"}),
        (UNET, {"unet"}),
        (TINYGPT, {"transformer_decoder", "gpt"}),
        (TRANSFORMER, {"transformer_encoder"}),
    ],
)
def test_sample_family_candidates(filename: str, expected: set[str]) -> None:
    graph = load_demo(filename)
    candidates = classify_families(graph)
    assert 1 <= len(candidates) <= 3
    assert candidates[0]["id"] in expected
    assert candidates[0]["label"]
    assert candidates[0]["evidence"]
    confidences = [candidate["confidence"] for candidate in candidates]
    assert confidences == sorted(confidences, reverse=True)
    assert all(0.0 <= value <= 1.0 for value in confidences)


@pytest.mark.parametrize("filename", SAMPLES)
def test_sample_explain_contract(filename: str) -> None:
    graph = sample_graph(filename)
    explanation = explain_graph(graph)
    for key in (
        "family",
        "familyLabel",
        "familyConfidence",
        "families",
        "tags",
        "patterns",
        "pseudocode",
        "groups",
        "unknown_reason",
    ):
        assert key in explanation, key
    assert isinstance(explanation["familyConfidence"], float)
    assert explanation["familyConfidence"] == explanation["confidence"]
    assert 1 <= len(explanation["families"]) <= 3
    assert explanation["families"][0]["id"] == explanation["family"]
    assert explanation["pseudocode"]
    assert explanation["tags"]
    # pure JSON: no NaN / sets / tuples
    dumped = json.dumps(explanation, ensure_ascii=False, allow_nan=False)
    assert json.loads(dumped)["family"] == explanation["family"]


@pytest.mark.parametrize(
    "filename, required",
    [
        (VGG, {"conv_bn_act", "global_pool", "classifier_head"}),
        (RESNET, {"conv_bn_act", "residual_block", "global_pool", "classifier_head"}),
        (UNET, {"conv_bn_act", "unet_skip"}),
        (
            TINYGPT,
            {
                "transformer_block",
                "self_attention",
                "ffn",
                "pre_norm",
                "weight_tying",
                "positional_encoding",
            },
        ),
        (
            TRANSFORMER,
            {
                "transformer_block",
                "self_attention",
                "ffn",
                "post_norm",
                "global_pool",
                "classifier_head",
            },
        ),
    ],
)
def test_sample_patterns(filename: str, required: set[str]) -> None:
    kinds = pattern_kinds(sample_graph(filename))
    missing = required - kinds
    assert not missing, f"{filename}: missing patterns {sorted(missing)} (got {sorted(kinds)})"


def test_residual_and_cross_attention_are_not_over_reported() -> None:
    # TinyGPT's token+position embedding add is not a residual block.
    assert "residual_block" not in pattern_kinds(sample_graph(TINYGPT))
    # Its attention is causal self-attention, not cross-attention.
    assert "cross_attention" not in pattern_kinds(sample_graph(TINYGPT))
    # TinyTransformer's blocks are self-attention only.
    assert "cross_attention" not in pattern_kinds(sample_graph(TRANSFORMER))


def test_sample_group_patterns() -> None:
    vgg = sample_graph(VGG)
    features = next(group for group in vgg["groups"] if group.get("name") == "features")
    assert features["meta"]["pattern"]["kind"] == "conv_bn_act"

    resnet = sample_graph(RESNET)
    block = next(group for group in resnet["groups"] if group.get("name") == "layer1.0")
    assert block["meta"]["pattern"]["kind"] == "residual_block"

    tinygpt = sample_graph(TINYGPT)
    block = next(group for group in tinygpt["groups"] if group.get("name") == "blocks.0")
    assert block["meta"]["pattern"]["kind"] == "transformer_block"

    unet = sample_graph(UNET)
    pattern_kinds_in_groups = {
        (group.get("meta") or {}).get("pattern", {}).get("kind")
        for group in unet["groups"]
    }
    assert "unet_skip" in pattern_kinds_in_groups


@pytest.mark.parametrize("filename", SAMPLES)
def test_sample_node_meta(filename: str) -> None:
    graph = sample_graph(filename)
    for node in graph["nodes"]:
        meta = node.get("meta")
        assert isinstance(meta, dict), node["id"]
        assert meta.get("formula")
        assert meta.get("semantic")
        assert meta.get("axes") is None or isinstance(meta["axes"], str)
        assert isinstance(meta.get("tags"), list) and meta["tags"]


def test_sample_node_formulas() -> None:
    vgg = sample_graph(VGG)
    nodes = {node["id"]: node for node in vgg["nodes"]}
    conv_formula = node_formula(nodes["n1"])
    assert "Conv" in conv_formula["formula"] or "Σ" in conv_formula["formula"]
    assert conv_formula["axes"] == "NCHW"
    assert "μ_B" in node_formula(nodes["n2"])["formula"]
    assert "max(x, 0)" in node_formula(nodes["n3"])["formula"]
    assert "全局" in node_formula(nodes["n19"])["semantic"]
    assert "Σ_i" in node_formula(nodes["n24"])["formula"]

    resnet = sample_graph(RESNET)
    nodes = {node["id"]: node for node in resnet["nodes"]}
    add_formula = node_formula(nodes["n11"])
    assert "+" in add_formula["formula"]
    assert "residual" in add_formula["tags"]

    unet = sample_graph(UNET)
    nodes = {node["id"]: node for node in unet["nodes"]}
    assert "转置卷积" in node_formula(nodes["n21"])["semantic"]
    assert "cat(" in node_formula(nodes["n22"])["formula"]

    tinygpt = sample_graph(TINYGPT)
    nodes = {node["id"]: node for node in tinygpt["nodes"]}
    assert "E[token" in node_formula(nodes["n1"])["formula"]
    assert "attention" in node_formula(nodes["n15"])["tags"]
    assert "Σ" in node_formula(nodes["n15"])["formula"]
    assert "exp(" in node_formula(nodes["n18"])["formula"]

    transformer = sample_graph(TRANSFORMER)
    nodes = {node["id"]: node for node in transformer["nodes"]}
    assert "softmax" in node_formula(nodes["n4"])["formula"]
    assert "γ[d]" in node_formula(nodes["n7"])["formula"]


# --------------------------------------------------------------------------- #
# sample pseudocode
# --------------------------------------------------------------------------- #
def pseudocode(filename: str) -> list[str]:
    return explain_graph(sample_graph(filename))["pseudocode"]


@pytest.mark.parametrize("filename", SAMPLES)
def test_pseudocode_shape_comments(filename: str) -> None:
    lines = pseudocode(filename)
    assert len(lines) >= 6, lines
    assert lines[0].endswith("Input(1, 16)") or "Input(" in lines[0]
    assert lines[-1].startswith("return") or "# -> " in lines[-1]
    for line in lines:
        if line.startswith("#") or "Input(" in line or line.startswith("return"):
            continue
        assert "# -> " in line, line


def test_vgg_pseudocode_is_staged_not_collapsed() -> None:
    lines = pseudocode(VGG)
    text = "\n".join(lines)
    assert "ConvBlock" not in text
    assert text.count("Conv2d(") >= 4
    assert "MaxPool" in text
    assert "(1, 16, 32, 32)" in text
    assert "(1, 64, 16, 16)" in text
    assert "logits = Linear(10)" in text


def test_resnet_pseudocode_folds_residual_blocks() -> None:
    text = "\n".join(pseudocode(RESNET))
    assert "ResidualBlock" in text
    assert text.count("ResidualBlock") == 4
    assert "GlobalAvgPool" in text
    assert "return logits" in text


def test_unet_pseudocode_keeps_skip_and_blocks() -> None:
    text = "\n".join(pseudocode(UNET))
    assert "EncoderBlock" in text
    assert "DecoderBlock" in text
    assert "Concat(dim=1)" in text
    assert "ConvTranspose2d" in text


def test_tinygpt_pseudocode_blocks_and_shapes() -> None:
    lines = pseudocode(TINYGPT)
    text = "\n".join(lines)
    assert "TransformerBlock(32)" in text
    assert text.count("TransformerBlock(32)") == 2
    positional = next(line for line in lines if "PositionalEmbedding" in line)
    assert "# -> (1, 16, 32)" in positional
    assert "logits = Linear(128)" in text


def test_transformer_pseudocode_positional_encoding() -> None:
    lines = pseudocode(TRANSFORMER)
    text = "\n".join(lines)
    assert "PositionalEncoding(32)" in text
    assert "Mul()" not in text
    assert "Add()" not in text
    assert "TransformerBlock(32)" in text
    assert "logits = Linear(10)" in text


def test_pseudocode_unknown_shape_uses_question_mark() -> None:
    body = [N("n1", "custom", "mystery", name="mystery", in_shape=(1, 8), out_shape=None)]
    graph = make_graph("MysteryNet", body, [("n0", "n1"), ("n1", "n2")])
    enrich_algorithms(graph, force=True)
    lines = explain_graph(graph)["pseudocode"]
    body_lines = [line for line in lines if "Mystery" in line]
    assert body_lines
    assert body_lines[0].endswith("# -> ?")


# --------------------------------------------------------------------------- #
# node formula catalogue
# --------------------------------------------------------------------------- #
def test_node_formula_catalogue_covers_required_ops() -> None:
    conv_params = {
        "inChannels": 3,
        "outChannels": 16,
        "kernelSize": [3, 3],
        "stride": [1, 1],
        "padding": [1, 1],
        "dilation": [1, 1],
        "groups": 1,
        "bias": True,
    }
    cases: list[tuple[dict, str, str | None, str]] = [
        (
            N("n1", "conv", "Conv2d", in_shape=(1, 3, 32, 32), out_shape=(1, 16, 32, 32), params=conv_params),
            "Σ",
            "NCHW",
            "空间卷积",
        ),
        (
            N(
                "n2",
                "conv",
                "Conv2d",
                in_shape=(1, 8, 32, 32),
                out_shape=(1, 8, 32, 32),
                params=dict(conv_params, inChannels=8, outChannels=8, groups=8),
            ),
            "W[c",
            "NCHW",
            "depthwise",
        ),
        (
            N(
                "n3",
                "deconv",
                "ConvTranspose2d",
                in_shape=(1, 8, 16, 16),
                out_shape=(1, 4, 32, 32),
                params=dict(conv_params, inChannels=8, outChannels=4),
            ),
            "Σ",
            "NCHW",
            "转置卷积",  # semantic
        ),
        (
            N("n4", "linear", "Linear", in_shape=(1, 64), out_shape=(1, 10), params={"inFeatures": 64, "outFeatures": 10}),
            "Σ_i",
            "N×F",
            "全连接",
        ),
        (
            N("n5", "norm", "BatchNorm2d", in_shape=(1, 16, 8, 8), out_shape=(1, 16, 8, 8), params={"normalizedShape": [16]}),
            "μ_B",
            "NCHW",
            "批归一化",
        ),
        (
            N("n6", "norm", "LayerNorm", in_shape=(1, 8, 32), out_shape=(1, 8, 32), params={"normalizedShape": [32]}),
            "γ",
            "最后一维",
            "层归一化",
        ),
        (N("n7", "activation", "ReLU", out_shape=(1, 8)), "max(x, 0)", "逐元素", "ReLU"),
        (N("n8", "activation", "GELU", out_shape=(1, 8)), "Φ(x)", "逐元素", "GELU"),
        (N("n9", "activation", "SiLU", out_shape=(1, 8)), "σ(x)", "逐元素", "SiLU"),
        (N("n10", "softmax", "Softmax", out_shape=(1, 8), params={"dim": -1}), "exp(", "dim=-1", "概率"),
        (
            N("n11", "pool", "MaxPool2d", in_shape=(1, 8, 16, 16), out_shape=(1, 8, 8, 8), params={"type": "max", "kernelSize": [2, 2], "stride": [2, 2]}),
            "max_{kh,kw}",
            "NCHW",
            "最大池化",
        ),
        (
            N("n12", "pool", "AdaptiveAvgPool2d", in_shape=(1, 8, 16, 16), out_shape=(1, 8, 1, 1), params={"type": "adaptive", "outputSize": [1, 1]}),
            "mean_{h,w}",
            "NCHW",
            "全局平均池化",
        ),
        (
            N("n13", "embedding", "Embedding", in_shape=(1, 16), out_shape=(1, 16, 32), params={"numEmbeddings": 128, "embeddingDim": 32}),
            "E[token",
            "NLC",
            "嵌入",
        ),
        (
            N(
                "n14",
                "attention",
                "MultiheadAttention",
                in_shape=(1, 16, 32),
                out_shape=(1, 16, 32),
                params={"embedDim": 32, "numHeads": 4, "headDim": 8},
            ),
            "softmax",
            "NLC",
            "多头注意力",
        ),
        (N("n15", "add", "add", in_shape=(1, 32), out_shape=(1, 32)), "x_1 + x_2", "N×F", "相加"),
        (N("n16", "concat", "cat", in_shape=(1, 32), out_shape=(1, 64), params={"dim": 1}), "cat(", "N×F", "拼接"),
        (N("n17", "dropout", "Dropout", out_shape=(1, 8), params={"p": 0.1}), "mask", None, "失活"),
        (N("n18", "flatten", "Flatten", in_shape=(1, 8, 1, 1), out_shape=(1, 8)), "reshape", "N×", "展平"),
        (N("n19", "reshape", "chunk", in_shape=(1, 16), out_shape=(1, 8), params={"dim": -1}), "split", "任意", "切分"),
        (
            N("n20", "rnn", "LSTM", in_shape=(1, 16, 8), out_shape=(1, 16, 8), params={"hiddenFeatures": 8, "numLayers": 1}),
            "LSTM(",
            "NLC",
            "LSTM",
        ),
        (
            N("n21", "rnn", "GRU", in_shape=(1, 16, 8), out_shape=(1, 16, 8), params={"hiddenFeatures": 8}),
            "GRU(",
            "NLC",
            "GRU",
        ),
    ]
    for node, formula_fragment, axes_fragment, semantic_fragment in cases:
        info = node_formula(node)
        assert formula_fragment in info["formula"], (node["op"], info)
        assert semantic_fragment.lower() in info["semantic"].lower(), (node["op"], info["semantic"])
        if axes_fragment is not None:
            assert info["axes"] is not None and axes_fragment in info["axes"], (node["op"], info["axes"])
        assert isinstance(info["tags"], list) and info["tags"]
        assert set(info) == {"formula", "semantic", "axes", "tags"}


def test_node_formula_tolerates_malformed_nodes() -> None:
    for node in ({}, None, {"kind": "custom"}, {"kind": "conv", "params": "not-a-dict"}):
        info = node_formula(node)  # type: ignore[arg-type]
        assert set(info) == {"formula", "semantic", "axes", "tags"}
        assert info["formula"]
        assert info["semantic"]
        assert info["tags"]


# --------------------------------------------------------------------------- #
# constructed structural patterns
# --------------------------------------------------------------------------- #
def test_depthwise_separable_block() -> None:
    body = [
        conv("n1", 8, 8, groups=8),
        bn("n2", 8, (1, 8, 16, 16), (1, 8, 16, 16)),
        act("n3", "ReLU", (1, 8, 16, 16)),
        conv("n4", 8, 16, kernel=1, padding=0, out_shape=(1, 16, 16, 16)),
    ]
    graph = make_graph(
        "DepthwiseNet",
        body,
        [("n0", "n1"), ("n1", "n2"), ("n2", "n3"), ("n3", "n4"), ("n4", "n5")],
        groups=[{"id": "g1", "name": "block", "label": "Block", "kind": "conv_block", "parentId": "g0", "depth": 1, "nodeIds": ["n1", "n2", "n3", "n4"], "childGroupIds": [], "numParams": 0, "flops": 0, "source": None}],
    )
    assert_has_pattern(graph, "depthwise_separable")


def test_inverted_residual_block() -> None:
    body = [
        conv("n1", 8, 32, kernel=1, padding=0, out_shape=(1, 32, 16, 16)),
        bn("n2", 32, (1, 32, 16, 16), (1, 32, 16, 16)),
        act("n3", "ReLU6", (1, 32, 16, 16)),
        conv("n4", 32, 32, groups=32, in_shape=(1, 32, 16, 16), out_shape=(1, 32, 16, 16)),
        bn("n5", 32, (1, 32, 16, 16), (1, 32, 16, 16)),
        act("n6", "ReLU6", (1, 32, 16, 16)),
        conv("n7", 32, 8, kernel=1, padding=0, in_shape=(1, 32, 16, 16), out_shape=(1, 8, 16, 16)),
        bn("n8", 8, (1, 8, 16, 16), (1, 8, 16, 16)),
        N("n9", "add", "add", in_shape=(1, 8, 16, 16), out_shape=(1, 8, 16, 16), tags=["merge", "residual"]),
        act("n10", "ReLU6", (1, 8, 16, 16)),
    ]
    graph = make_graph(
        "MobileNetV2Tiny",
        body,
        [
            ("n0", "n1"),
            ("n1", "n2"),
            ("n2", "n3"),
            ("n3", "n4"),
            ("n4", "n5"),
            ("n5", "n6"),
            ("n6", "n7"),
            ("n7", "n8"),
            ("n0", "n9", "residual"),
            ("n8", "n9"),
            ("n9", "n10"),
            ("n10", "n11"),
        ],
        groups=[{"id": "g1", "name": "block", "label": "Block", "kind": "residual_block", "parentId": "g0", "depth": 1, "nodeIds": [node["id"] for node in body], "childGroupIds": [], "numParams": 0, "flops": 0, "source": None}],
        input_shape=(1, 8, 16, 16),
        output_shape=(1, 8, 16, 16),
    )
    assert_has_pattern(graph, "inverted_residual")
    assert_has_pattern(graph, "residual_block")
    top = classify_families(graph)[0]
    assert top["id"] in {"mobilenet_v2", "mobilenet", "cnn", "vgg"}


def test_squeeze_excitation_block() -> None:
    body = [
        N("n1", "pool", "AdaptiveAvgPool2d", in_shape=(1, 8, 16, 16), out_shape=(1, 8, 1, 1), params={"type": "adaptive", "outputSize": [1, 1]}),
        linear("n2", 8, 2, in_shape=(1, 8), out_shape=(1, 2)),
        act("n3", "ReLU", (1, 2)),
        linear("n4", 2, 8, in_shape=(1, 2), out_shape=(1, 8)),
        act("n5", "Sigmoid", (1, 8)),
        N("n6", "custom", "mul", name="scale", in_shape=(1, 8), out_shape=(1, 8)),
    ]
    graph = make_graph(
        "SENetTiny",
        body,
        [
            ("n0", "n1"),
            ("n1", "n2"),
            ("n2", "n3"),
            ("n3", "n4"),
            ("n4", "n5"),
            ("n5", "n6"),
            ("n0", "n6"),
            ("n6", "n7"),
        ],
    )
    assert_has_pattern(graph, "squeeze_excitation")
    top = classify_families(graph)[0]
    assert top["id"] == "senet"


def test_inception_module() -> None:
    body = [
        conv("n1", 8, 8, kernel=1, padding=0, out_shape=(1, 8, 16, 16)),
        act("n2", "ReLU", (1, 8, 16, 16)),
        conv("n3", 8, 8, kernel=3, out_shape=(1, 8, 16, 16)),
        act("n4", "ReLU", (1, 8, 16, 16)),
        conv("n5", 8, 8, kernel=5, padding=2, out_shape=(1, 8, 16, 16)),
        act("n6", "ReLU", (1, 8, 16, 16)),
        N("n7", "concat", "torch.cat", in_shape=(1, 8, 16, 16), out_shape=(1, 24, 16, 16), params={"dim": 1}),
    ]
    graph = make_graph(
        "InceptionTiny",
        body,
        [
            ("n0", "n1"),
            ("n1", "n2"),
            ("n0", "n3"),
            ("n3", "n4"),
            ("n0", "n5"),
            ("n5", "n6"),
            ("n2", "n7"),
            ("n4", "n7"),
            ("n6", "n7"),
            ("n7", "n8"),
        ],
        input_shape=(1, 8, 16, 16),
        output_shape=(1, 24, 16, 16),
    )
    pattern = assert_has_pattern(graph, "inception")
    assert len(pattern["nodeIds"]) >= 4


def test_dense_connection_block() -> None:
    body = [
        conv("n1", 8, 8, in_shape=(1, 8, 16, 16), out_shape=(1, 8, 16, 16)),
        N("n2", "concat", "torch.cat", in_shape=(1, 8, 16, 16), out_shape=(1, 16, 16, 16), params={"dim": 1}),
        conv("n3", 16, 8, in_shape=(1, 16, 16, 16), out_shape=(1, 8, 16, 16)),
        N("n4", "concat", "torch.cat", in_shape=(1, 8, 16, 16), out_shape=(1, 24, 16, 16), params={"dim": 1}),
    ]
    graph = make_graph(
        "DenseTiny",
        body,
        [
            ("n0", "n1"),
            ("n0", "n2"),
            ("n1", "n2"),
            ("n2", "n3"),
            ("n2", "n4"),
            ("n3", "n4"),
            ("n4", "n5"),
        ],
        input_shape=(1, 8, 16, 16),
        output_shape=(1, 24, 16, 16),
    )
    assert_has_pattern(graph, "dense_connection")
    assert classify_families(graph)[0]["id"] == "densenet"


def test_unet_skip_connection() -> None:
    body = [
        conv("n1", 8, 8, group="g1"),
        N("n2", "deconv", "ConvTranspose2d", in_shape=(1, 8, 8, 8), out_shape=(1, 8, 16, 16), params={"inChannels": 8, "outChannels": 8, "kernelSize": [2, 2], "stride": [2, 2], "padding": [0, 0], "outputPadding": [0, 0]}, group="g2"),
        N("n3", "concat", "torch.cat", in_shape=(1, 8, 16, 16), out_shape=(1, 16, 16, 16), params={"dim": 1}, group="g2"),
        conv("n4", 16, 8, in_shape=(1, 16, 16, 16), out_shape=(1, 8, 16, 16), group="g2"),
    ]
    graph = make_graph(
        "UNetTiny",
        body,
        [
            ("n0", "n1"),
            ("n0", "n2"),
            ("n1", "n3", "skip"),
            ("n2", "n3"),
            ("n3", "n4"),
            ("n4", "n5"),
        ],
        groups=[
            {"id": "g1", "name": "encoder", "label": "Encoder", "kind": "encoder", "parentId": "g0", "depth": 1, "nodeIds": ["n1"], "childGroupIds": [], "numParams": 0, "flops": 0, "source": None},
            {"id": "g2", "name": "decoder", "label": "Decoder", "kind": "decoder", "parentId": "g0", "depth": 1, "nodeIds": ["n2", "n3", "n4"], "childGroupIds": [], "numParams": 0, "flops": 0, "source": None},
        ],
        input_shape=(1, 8, 16, 16),
        output_shape=(1, 8, 16, 16),
    )
    pattern = assert_has_pattern(graph, "unet_skip")
    assert pattern["confidence"] >= 0.9
    assert classify_families(graph)[0]["id"] == "unet"


def test_self_attention_block() -> None:
    attention = N(
        "n1",
        "attention",
        "MultiheadAttention",
        in_shape=(1, 8, 32),
        out_shape=(1, 8, 32),
        params={"embedDim": 32, "numHeads": 4, "headDim": 8},
    )
    graph = make_graph(
        "SelfAttentionNet",
        [attention],
        [("n0", "n1"), ("n1", "n2")],
        groups=[{"id": "g1", "name": "attn", "label": "Attention", "kind": "attention_block", "parentId": "g0", "depth": 1, "nodeIds": ["n1"], "childGroupIds": [], "numParams": 0, "flops": 0, "source": None}],
        input_shape=(1, 8, 32),
        output_shape=(1, 8, 32),
    )
    assert_has_pattern(graph, "self_attention")
    assert "cross_attention" not in pattern_kinds(graph)


def test_cross_attention_block() -> None:
    body = [
        linear("n1", 8, 32, in_shape=(1, 8, 8), out_shape=(1, 8, 32), group="g1"),
        linear("n2", 8, 32, in_shape=(1, 8, 8), out_shape=(1, 8, 32), group="g2"),
        N(
            "n3",
            "attention",
            "MultiheadAttention",
            in_shape=(1, 8, 32),
            out_shape=(1, 8, 32),
            params={"embedDim": 32, "numHeads": 4, "headDim": 8},
            group="g2",
        ),
    ]
    graph = make_graph(
        "CrossAttentionNet",
        body,
        [
            ("n0", "n1"),
            ("n0", "n2"),
            ("n1", "n3", "attention"),
            ("n2", "n3", "attention"),
            ("n3", "n4"),
        ],
        groups=[
            {"id": "g1", "name": "encoder", "label": "Encoder", "kind": "encoder", "parentId": "g0", "depth": 1, "nodeIds": ["n1"], "childGroupIds": [], "numParams": 0, "flops": 0, "source": None},
            {"id": "g2", "name": "decoder", "label": "Decoder", "kind": "decoder", "parentId": "g0", "depth": 1, "nodeIds": ["n2", "n3"], "childGroupIds": [], "numParams": 0, "flops": 0, "source": None},
        ],
        input_shape=(1, 8, 8),
        output_shape=(1, 8, 32),
    )
    pattern = assert_has_pattern(graph, "cross_attention")
    assert len(pattern["nodeIds"]) >= 2


def test_bottleneck_block() -> None:
    body = [
        conv("n1", 32, 8, kernel=1, padding=0, in_shape=(1, 32, 16, 16), out_shape=(1, 8, 16, 16)),
        bn("n2", 8, (1, 8, 16, 16), (1, 8, 16, 16)),
        act("n3", "ReLU", (1, 8, 16, 16)),
        conv("n4", 8, 8, kernel=3, in_shape=(1, 8, 16, 16), out_shape=(1, 8, 16, 16)),
        bn("n5", 8, (1, 8, 16, 16), (1, 8, 16, 16)),
        act("n6", "ReLU", (1, 8, 16, 16)),
        conv("n7", 8, 32, kernel=1, padding=0, in_shape=(1, 8, 16, 16), out_shape=(1, 32, 16, 16)),
        bn("n8", 32, (1, 32, 16, 16), (1, 32, 16, 16)),
        N("n9", "add", "add", in_shape=(1, 32, 16, 16), out_shape=(1, 32, 16, 16), tags=["merge", "residual"]),
        act("n10", "ReLU", (1, 32, 16, 16)),
    ]
    graph = make_graph(
        "BottleneckNet",
        body,
        [
            ("n0", "n1"),
            ("n1", "n2"),
            ("n2", "n3"),
            ("n3", "n4"),
            ("n4", "n5"),
            ("n5", "n6"),
            ("n6", "n7"),
            ("n7", "n8"),
            ("n0", "n9", "residual"),
            ("n8", "n9"),
            ("n9", "n10"),
            ("n10", "n11"),
        ],
        groups=[{"id": "g1", "name": "block", "label": "Block", "kind": "residual_block", "parentId": "g0", "depth": 1, "nodeIds": [node["id"] for node in body], "childGroupIds": [], "numParams": 0, "flops": 0, "source": None}],
        input_shape=(1, 32, 16, 16),
        output_shape=(1, 32, 16, 16),
    )
    assert_has_pattern(graph, "residual_block")
    assert_has_pattern(graph, "bottleneck")


def test_gated_mlp_block() -> None:
    body = [
        linear("n1", 8, 16, in_shape=(1, 8), out_shape=(1, 16)),
        act("n2", "SiLU", (1, 16)),
        linear("n3", 8, 16, in_shape=(1, 8), out_shape=(1, 16)),
        act("n4", "Sigmoid", (1, 16)),
        N("n5", "custom", "mul", name="gate", in_shape=(1, 16), out_shape=(1, 16)),
        linear("n6", 16, 8, in_shape=(1, 16), out_shape=(1, 8)),
    ]
    graph = make_graph(
        "GatedMLPNet",
        body,
        [
            ("n0", "n1"),
            ("n1", "n2"),
            ("n0", "n3"),
            ("n3", "n4"),
            ("n2", "n5"),
            ("n4", "n5"),
            ("n5", "n6"),
            ("n6", "n7"),
        ],
    )
    pattern = assert_has_pattern(graph, "gated_mlp")
    assert "swiglu" in pattern["tags"]


def test_rnn_block() -> None:
    body = [
        N(
            "n1",
            "rnn",
            "LSTM",
            in_shape=(1, 8, 16),
            out_shape=(1, 8, 16),
            params={"inputFeatures": 16, "hiddenFeatures": 16, "numLayers": 1},
        )
    ]
    graph = make_graph(
        "LSTMNet",
        body,
        [("n0", "n1"), ("n1", "n2")],
        input_shape=(1, 8, 16),
        output_shape=(1, 8, 16),
    )
    assert_has_pattern(graph, "rnn_block")
    assert classify_families(graph)[0]["id"] in {"lstm", "rnn"}


def test_moe_block() -> None:
    body = [
        linear("n1", 8, 2, in_shape=(1, 8), out_shape=(1, 2)),
        N("n2", "softmax", "Softmax", in_shape=(1, 2), out_shape=(1, 2), params={"dim": -1}),
        linear("n3", 8, 8, in_shape=(1, 8), out_shape=(1, 8)),
        linear("n4", 8, 8, in_shape=(1, 8), out_shape=(1, 8)),
        N("n5", "custom", "mul", name="weighted0", in_shape=(1, 8), out_shape=(1, 8)),
        N("n6", "custom", "mul", name="weighted1", in_shape=(1, 8), out_shape=(1, 8)),
        N("n7", "add", "add", in_shape=(1, 8), out_shape=(1, 8), tags=["merge"]),
    ]
    graph = make_graph(
        "MoEBlock",
        body,
        [
            ("n0", "n1"),
            ("n1", "n2"),
            ("n0", "n3"),
            ("n0", "n4"),
            ("n3", "n5"),
            ("n2", "n5"),
            ("n4", "n6"),
            ("n2", "n6"),
            ("n5", "n7"),
            ("n6", "n7"),
            ("n7", "n8"),
        ],
    )
    pattern = assert_has_pattern(graph, "moe")
    assert len(pattern["nodeIds"]) >= 3


def test_ssm_mamba_block() -> None:
    body = [
        N(
            "n1",
            "conv",
            "Conv1d",
            in_shape=(1, 8, 16),
            out_shape=(1, 8, 16),
            params={"inChannels": 8, "outChannels": 8, "kernelSize": [4], "stride": [1], "padding": [3], "dilation": [1], "groups": 8, "bias": True},
        ),
        act("n2", "SiLU", (1, 8, 16)),
        linear("n3", 8, 16, in_shape=(1, 8, 16), out_shape=(1, 8, 16)),
        linear("n4", 8, 16, in_shape=(1, 8, 16), out_shape=(1, 8, 16)),
        N("n5", "custom", "mul", name="gate", in_shape=(1, 8, 16), out_shape=(1, 8, 16)),
        linear("n6", 16, 8, in_shape=(1, 8, 16), out_shape=(1, 8, 16)),
    ]
    graph = make_graph(
        "MambaBlock",
        body,
        [
            ("n0", "n1"),
            ("n1", "n2"),
            ("n2", "n3"),
            ("n2", "n4"),
            ("n3", "n5"),
            ("n4", "n5"),
            ("n5", "n6"),
            ("n6", "n7"),
        ],
        input_shape=(1, 8, 16),
        output_shape=(1, 8, 16),
    )
    assert_has_pattern(graph, "ssm")
    assert classify_families(graph)[0]["id"] == "mamba"


def test_rwkv_time_mix_block() -> None:
    body = [
        linear("n1", 8, 8, module="rwkv.time_mix.key", in_shape=(1, 8), out_shape=(1, 8)),
        act("n2", "Sigmoid", (1, 8), group="g1"),
        N("n3", "custom", "mul", name="time_mix", module="rwkv.time_mix", in_shape=(1, 8), out_shape=(1, 8)),
    ]
    graph = make_graph(
        "RWKVBlock",
        body,
        [("n0", "n1"), ("n0", "n2"), ("n1", "n3"), ("n2", "n3"), ("n3", "n4")],
    )
    assert_has_pattern(graph, "rwkv")


def test_gnn_message_passing() -> None:
    body = [
        linear("n1", 8, 8, in_shape=(1, 8), out_shape=(1, 8)),
        N("n2", "norm", "LayerNorm", in_shape=(1, 8), out_shape=(1, 8), params={"normalizedShape": [8]}),
        linear("n3", 8, 8, in_shape=(1, 8), out_shape=(1, 8)),
        N("n4", "add", "add", in_shape=(1, 8), out_shape=(1, 8), tags=["merge"]),
    ]
    graph = make_graph(
        "GCNLayer",
        body,
        [("n0", "n1"), ("n1", "n2"), ("n0", "n3"), ("n2", "n4"), ("n3", "n4"), ("n4", "n5")],
        input_shape=(1, 8),
        output_shape=(1, 8),
    )
    assert_has_pattern(graph, "gnn_message_passing")
    assert classify_families(graph)[0]["id"] in {"gcn", "gin", "graphsage", "mpnn"}


def test_pointnet_mlp() -> None:
    body = [
        linear("n1", 8, 16, in_shape=(1, 8), out_shape=(1, 16)),
        act("n2", "ReLU", (1, 16)),
        N(
            "n3",
            "pool",
            "AdaptiveMaxPool2d",
            in_shape=(1, 16, 4, 4),
            out_shape=(1, 16, 1, 1),
            params={"type": "adaptive", "outputSize": [1, 1]},
        ),
        linear("n4", 16, 4, in_shape=(1, 16), out_shape=(1, 4)),
    ]
    graph = make_graph(
        "PointNet",
        body,
        [("n0", "n1"), ("n1", "n2"), ("n2", "n3"), ("n3", "n4"), ("n4", "n5")],
        input_shape=(1, 8),
        output_shape=(1, 4),
    )
    assert_has_pattern(graph, "point_mlp")


def test_nerf_mlp() -> None:
    body = [
        N("n1", "positional", "PositionalEncoding", in_shape=(1, 3), out_shape=(1, 64)),
        linear("n2", 64, 128, in_shape=(1, 64), out_shape=(1, 128)),
        act("n3", "ReLU", (1, 128)),
        linear("n4", 128, 128, in_shape=(1, 128), out_shape=(1, 128)),
        act("n5", "ReLU", (1, 128)),
        linear("n6", 128, 4, in_shape=(1, 128), out_shape=(1, 4)),
    ]
    graph = make_graph(
        "NeRF",
        body,
        [("n0", "n1"), ("n1", "n2"), ("n2", "n3"), ("n3", "n4"), ("n4", "n5"), ("n5", "n6"), ("n6", "n7")],
        input_shape=(1, 3),
        output_shape=(1, 4),
    )
    assert_has_pattern(graph, "nerf_mlp")


def test_vae_reparameterization() -> None:
    body = [
        linear("n1", 8, 16, in_shape=(1, 8), out_shape=(1, 16)),
        N("n2", "reshape", "chunk", in_shape=(1, 16), out_shape=(1, 8), params={"dim": -1}),
        N("n3", "custom", "mul", name="reparam_mul", in_shape=(1, 8), out_shape=(1, 8)),
        N("n4", "add", "add", in_shape=(1, 8), out_shape=(1, 8), tags=["merge"]),
    ]
    graph = make_graph(
        "VAE",
        body,
        [("n0", "n1"), ("n1", "n2"), ("n2", "n3"), ("n0", "n3"), ("n3", "n4"), ("n4", "n5")],
        input_shape=(1, 8),
        output_shape=(1, 8),
    )
    assert_has_pattern(graph, "vae_reparam")


def test_diffusion_unet() -> None:
    body = [
        conv("n1", 8, 8, out_shape=(1, 8, 16, 16), group="g1"),
        N("n2", "deconv", "ConvTranspose2d", in_shape=(1, 8, 8, 8), out_shape=(1, 8, 16, 16), params={"inChannels": 8, "outChannels": 8, "kernelSize": [2, 2], "stride": [2, 2], "padding": [0, 0], "outputPadding": [0, 0]}, group="g2"),
        N("n3", "concat", "torch.cat", in_shape=(1, 8, 16, 16), out_shape=(1, 16, 16, 16), params={"dim": 1}, group="g2"),
        linear("n4", 8, 16, in_shape=(1, 8), out_shape=(1, 16), group="g1"),
        act("n5", "SiLU", (1, 16), group="g1"),
    ]
    graph = make_graph(
        "DiffusionUNet",
        body,
        [
            ("n0", "n1"),
            ("n0", "n2"),
            ("n1", "n3", "skip"),
            ("n2", "n3"),
            ("n3", "n6"),
            ("n0", "n4"),
            ("n4", "n5"),
        ],
        groups=[
            {"id": "g1", "name": "encoder", "label": "Encoder", "kind": "encoder", "parentId": "g0", "depth": 1, "nodeIds": ["n1", "n4", "n5"], "childGroupIds": [], "numParams": 0, "flops": 0, "source": None},
            {"id": "g2", "name": "decoder", "label": "Decoder", "kind": "decoder", "parentId": "g0", "depth": 1, "nodeIds": ["n2", "n3"], "childGroupIds": [], "numParams": 0, "flops": 0, "source": None},
        ],
        input_shape=(1, 8, 16, 16),
        output_shape=(1, 16, 16, 16),
    )
    assert_has_pattern(graph, "diffusion_unet")


def test_swin_window_attention() -> None:
    body = [
        N(
            "n1",
            "reshape",
            "view",
            name="window_partition",
            module="swin.window_attn.view",
            in_shape=(1, 8, 32),
            out_shape=(1, 2, 4, 32),
        ),
        N(
            "n2",
            "attention",
            "MultiheadAttention",
            module="swin.window_attn.attn",
            in_shape=(1, 2, 4, 32),
            out_shape=(1, 2, 4, 32),
            params={"embedDim": 32, "numHeads": 4, "headDim": 8},
        ),
    ]
    graph = make_graph(
        "SwinTransformer",
        body,
        [("n0", "n1"), ("n1", "n2"), ("n2", "n3")],
        groups=[{"id": "g1", "name": "swin_block", "label": "Swin Block", "kind": "transformer_block", "parentId": "g0", "depth": 1, "nodeIds": ["n1", "n2"], "childGroupIds": [], "numParams": 0, "flops": 0, "source": None}],
        input_shape=(1, 8, 32),
        output_shape=(1, 8, 32),
    )
    assert_has_pattern(graph, "swin_attention")


def test_rope_and_rmsnorm_and_swiglu() -> None:
    body = [
        N("n1", "norm", "RMSNorm", in_shape=(1, 8), out_shape=(1, 8), params={"normalizedShape": [8]}),
        N("n2", "custom", "mul", name="rope_rotate", module="llama.rope.rotate", in_shape=(1, 8), out_shape=(1, 8)),
        linear("n3", 8, 16, in_shape=(1, 8), out_shape=(1, 16)),
        act("n4", "SiLU", (1, 16)),
        linear("n5", 8, 16, in_shape=(1, 8), out_shape=(1, 16)),
        act("n6", "Sigmoid", (1, 16)),
        N("n7", "custom", "mul", name="swiglu", in_shape=(1, 16), out_shape=(1, 16)),
        linear("n8", 16, 8, in_shape=(1, 16), out_shape=(1, 8)),
    ]
    graph = make_graph(
        "LlamaTiny",
        body,
        [
            ("n0", "n1"),
            ("n1", "n2"),
            ("n2", "n3"),
            ("n3", "n4"),
            ("n2", "n5"),
            ("n5", "n6"),
            ("n4", "n7"),
            ("n6", "n7"),
            ("n7", "n8"),
            ("n8", "n9"),
        ],
        input_shape=(1, 8),
        output_shape=(1, 8),
    )
    enrich_algorithms(graph, force=True)
    kinds = pattern_kinds(graph)
    assert "rmsnorm" in kinds
    assert "rope" in kinds
    assert "gated_mlp" in kinds


def test_cbam_and_sk_attention() -> None:
    cbam_body = [
        N("n1", "pool", "AdaptiveAvgPool2d", in_shape=(1, 8, 16, 16), out_shape=(1, 8, 1, 1), params={"type": "adaptive", "outputSize": [1, 1]}),
        N("n2", "pool", "AdaptiveMaxPool2d", in_shape=(1, 8, 16, 16), out_shape=(1, 8, 1, 1), params={"type": "adaptive", "outputSize": [1, 1]}),
        N("n3", "add", "add", in_shape=(1, 8, 1, 1), out_shape=(1, 8, 1, 1), tags=["merge"]),
        linear("n4", 8, 2, in_shape=(1, 8), out_shape=(1, 2)),
        act("n5", "ReLU", (1, 2)),
        linear("n6", 2, 8, in_shape=(1, 2), out_shape=(1, 8)),
        act("n7", "Sigmoid", (1, 8)),
        N("n8", "custom", "mul", name="channel_scale", in_shape=(1, 8), out_shape=(1, 8)),
        conv("n9", 8, 8, kernel=7, padding=3, in_shape=(1, 8, 16, 16), out_shape=(1, 8, 16, 16)),
    ]
    cbam = make_graph(
        "CBAMNet",
        cbam_body,
        [
            ("n0", "n1"),
            ("n0", "n2"),
            ("n1", "n3"),
            ("n2", "n3"),
            ("n3", "n4"),
            ("n4", "n5"),
            ("n5", "n6"),
            ("n6", "n7"),
            ("n7", "n8"),
            ("n0", "n8"),
            ("n0", "n9"),
            ("n8", "n10"),
        ],
        input_shape=(1, 8, 16, 16),
        output_shape=(1, 8, 16, 16),
    )
    assert_has_pattern(cbam, "cbam")

    sk_body = [
        conv("n1", 8, 8, kernel=1, padding=0, out_shape=(1, 8, 16, 16)),
        conv("n2", 8, 8, kernel=3, out_shape=(1, 8, 16, 16)),
        N("n3", "pool", "AdaptiveAvgPool2d", in_shape=(1, 8, 16, 16), out_shape=(1, 8, 1, 1), params={"type": "adaptive", "outputSize": [1, 1]}),
        linear("n4", 8, 4, in_shape=(1, 8), out_shape=(1, 4)),
        linear("n5", 4, 8, in_shape=(1, 4), out_shape=(1, 8)),
        N("n6", "custom", "mul", name="selective", in_shape=(1, 8), out_shape=(1, 8)),
    ]
    sk = make_graph(
        "SKNet",
        sk_body,
        [("n0", "n1"), ("n0", "n2"), ("n0", "n3"), ("n3", "n4"), ("n4", "n5"), ("n5", "n6"), ("n1", "n6"), ("n2", "n6"), ("n6", "n7")],
        input_shape=(1, 8, 16, 16),
        output_shape=(1, 8, 16, 16),
    )
    assert_has_pattern(sk, "sk_attention")


def test_fpn_and_panet() -> None:
    body = [
        conv("n1", 8, 8, out_shape=(1, 8, 16, 16)),
        N("n2", "deconv", "ConvTranspose2d", in_shape=(1, 8, 8, 8), out_shape=(1, 8, 16, 16), params={"inChannels": 8, "outChannels": 8, "kernelSize": [2, 2], "stride": [2, 2], "padding": [0, 0], "outputPadding": [0, 0]}),
        N("n3", "add", "add", in_shape=(1, 8, 16, 16), out_shape=(1, 8, 16, 16), tags=["merge"]),
        conv("n4", 8, 8, in_shape=(1, 8, 16, 16), out_shape=(1, 8, 16, 16)),
        N("n5", "deconv", "ConvTranspose2d", in_shape=(1, 8, 8, 8), out_shape=(1, 8, 16, 16), params={"inChannels": 8, "outChannels": 8, "kernelSize": [2, 2], "stride": [2, 2], "padding": [0, 0], "outputPadding": [0, 0]}),
        N("n6", "add", "add", in_shape=(1, 8, 16, 16), out_shape=(1, 8, 16, 16), tags=["merge"]),
    ]
    edges = [
        ("n0", "n1"),
        ("n0", "n2"),
        ("n1", "n3"),
        ("n2", "n3"),
        ("n3", "n4"),
        ("n4", "n6"),
        ("n0", "n5"),
        ("n5", "n6"),
        ("n6", "n7"),
    ]
    fpn = make_graph("FPN", body, edges, input_shape=(1, 8, 16, 16), output_shape=(1, 8, 16, 16))
    assert_has_pattern(fpn, "fpn")

    panet = make_graph("PANet", body, edges, input_shape=(1, 8, 16, 16), output_shape=(1, 8, 16, 16))
    assert_has_pattern(panet, "panet")


def test_unet_plus_plus_dense_skip() -> None:
    body = [
        conv("n1", 8, 8, group="g1"),
        N("n2", "deconv", "ConvTranspose2d", in_shape=(1, 8, 8, 8), out_shape=(1, 8, 16, 16), params={"inChannels": 8, "outChannels": 8, "kernelSize": [2, 2], "stride": [2, 2], "padding": [0, 0], "outputPadding": [0, 0]}, group="g2"),
        conv("n3", 8, 8, in_shape=(1, 8, 16, 16), out_shape=(1, 8, 16, 16), group="g2"),
        N("n4", "concat", "torch.cat", in_shape=(1, 8, 16, 16), out_shape=(1, 24, 16, 16), params={"dim": 1}, group="g2"),
    ]
    graph = make_graph(
        "UNetPlusPlusTiny",
        body,
        [
            ("n0", "n1"),
            ("n0", "n2"),
            ("n0", "n3"),
            ("n1", "n4", "skip"),
            ("n2", "n4", "skip"),
            ("n3", "n4"),
            ("n4", "n5"),
        ],
        groups=[
            {"id": "g1", "name": "encoder", "label": "Encoder", "kind": "encoder", "parentId": "g0", "depth": 1, "nodeIds": ["n1"], "childGroupIds": [], "numParams": 0, "flops": 0, "source": None},
            {"id": "g2", "name": "decoder", "label": "Decoder", "kind": "decoder", "parentId": "g0", "depth": 1, "nodeIds": ["n2", "n3", "n4"], "childGroupIds": [], "numParams": 0, "flops": 0, "source": None},
        ],
        input_shape=(1, 8, 16, 16),
        output_shape=(1, 24, 16, 16),
    )
    kinds = pattern_kinds(graph)
    assert "unet_skip" in kinds
    assert "unet_plus_plus" in kinds


# --------------------------------------------------------------------------- #
# unknown graph, empty graph, robustness
# --------------------------------------------------------------------------- #
def test_unknown_graph_reports_unknown_and_facts() -> None:
    body = [
        N("n1", "custom", "mystery", name="mystery", in_shape=(1, 8), out_shape=(1, 8)),
        N("n2", "custom", "mystery2", name="mystery2", in_shape=(1, 8), out_shape=(1, 8)),
    ]
    graph = make_graph("MysteryNet", body, [("n0", "n1"), ("n1", "n2"), ("n2", "n3")])
    candidates = classify_families(graph)
    assert all(candidate["id"] != "unknown" for candidate in candidates)
    enrich_algorithms(graph, force=True)
    explanation = explain_graph(graph)
    assert explanation["family"] == "unknown"
    assert explanation["familyConfidence"] <= 0.3
    assert explanation["confidence"] <= 0.3
    assert all(candidate["id"] != "unknown" for candidate in explanation["families"])
    assert explanation["unknown_reason"]
    assert "结构" in explanation["unknown_reason"]
    assert explanation["pseudocode"]
    markdown = render_explain_md(graph)
    assert "算法理解" in markdown
    assert "unknown" in markdown.lower() or "未知" in markdown
    assert "84%" not in markdown


def test_unknown_meta_values_are_low_confidence() -> None:
    body = [N("n1", "custom", "mystery", name="mystery", in_shape=(1, 8), out_shape=(1, 8))]
    graph = make_graph("MysteryNet", body, [("n0", "n1"), ("n1", "n2")])
    enrich_algorithms(graph, force=True)
    meta = graph["model"]["meta"]
    assert meta["family"] == "unknown"
    assert meta["familyLabel"] == "Unknown"
    assert meta["familyConfidence"] <= 0.3
    assert meta["confidence"] <= 0.3
    assert all(candidate["id"] != "unknown" for candidate in meta["families"])
    assert meta["unknownReason"]


def test_empty_graph_is_safe() -> None:
    assert enrich_algorithms({}) == {}
    explanation = explain_graph({})
    assert explanation["family"] == "unknown"
    assert explanation["familyConfidence"] <= 0.3
    assert explanation["families"] == []
    assert explanation["pseudocode"] == []
    assert explanation["patterns"] == []
    assert explanation["groups"] == []
    assert explanation["unknown_reason"]
    markdown = render_explain_md({})
    assert "算法理解" in markdown


def test_malformed_graph_does_not_crash() -> None:
    graph = {"model": {"name": "Broken"}, "nodes": [None, {"id": "n1", "kind": "conv"}], "edges": [None, {"source": "nope", "target": "n1"}]}
    enrich_algorithms(graph, force=True)
    explanation = explain_graph(graph)
    assert isinstance(explanation, dict)
    assert isinstance(explanation["families"], list)


# --------------------------------------------------------------------------- #
# idempotency / preservation
# --------------------------------------------------------------------------- #
def test_enrich_is_idempotent() -> None:
    graph = load_demo(RESNET)
    enrich_algorithms(graph, force=True)
    first = json.dumps(graph, ensure_ascii=False, sort_keys=True)
    enrich_algorithms(graph)
    second = json.dumps(graph, ensure_ascii=False, sort_keys=True)
    assert first == second
    enrich_algorithms(graph, force=True)
    third = json.dumps(graph, ensure_ascii=False, sort_keys=True)
    assert first == third


def test_existing_family_preserved_unless_forced() -> None:
    graph = load_demo(UNET)
    enrich_algorithms(graph, force=True)
    original = graph["model"]["meta"]["family"]
    graph["model"]["meta"]["family"] = "custom_family"
    enrich_algorithms(graph)
    assert graph["model"]["meta"]["family"] == "custom_family"
    enrich_algorithms(graph, force=True)
    assert graph["model"]["meta"]["family"] == original


def test_enrich_preserves_nodes_edges_params_metrics() -> None:
    graph = load_demo(RESNET)
    graph["model"]["metrics"] = {"paramBytes": 123}
    graph["nodes"][1]["metrics"] = {"activationBytes": 7}
    graph["nodes"][1]["meta"] = {"custom": "keep"}
    before_edges = copy.deepcopy(graph["edges"])
    before_nodes = [
        {key: copy.deepcopy(value) for key, value in node.items() if key not in ("meta",)}
        for node in graph["nodes"]
        if isinstance(node, dict)
    ]
    before_metrics = copy.deepcopy([node.get("metrics") for node in graph["nodes"]])
    before_model = {
        key: copy.deepcopy(value)
        for key, value in graph["model"].items()
        if key not in ("meta",)
    }

    enrich_algorithms(graph, force=True)

    assert graph["edges"] == before_edges
    after_nodes = [
        {key: copy.deepcopy(value) for key, value in node.items() if key not in ("meta",)}
        for node in graph["nodes"]
        if isinstance(node, dict)
    ]
    assert after_nodes == before_nodes
    assert [node.get("metrics") for node in graph["nodes"]] == before_metrics
    after_model = {
        key: copy.deepcopy(value)
        for key, value in graph["model"].items()
        if key not in ("meta",)
    }
    assert after_model == before_model
    assert graph["nodes"][1]["meta"]["custom"] == "keep"
    assert graph["nodes"][1]["meta"]["formula"]
    assert isinstance(graph["model"]["meta"], dict)


# --------------------------------------------------------------------------- #
# knowledge-base driven classification
# --------------------------------------------------------------------------- #
SMALL_KB = {
    "version": "test",
    "families": [
        {
            "id": "toy_cnn",
            "label": "Toy CNN",
            "domain": "vision",
            "aliases": ["toynet"],
            "summary": "",
            "components": ["toy_conv", "toy_pool"],
            "required_patterns": ["toy_conv", "toy_pool"],
            "optional_patterns": [],
            "signature_ops": ["conv", "norm", "activation", "pool", "linear"],
            "formula": "y = Head(GAP(Conv(x)))",
            "references": [],
        },
        {
            "id": "toy_mlp",
            "label": "Toy MLP",
            "domain": "general",
            "aliases": ["toymlp"],
            "summary": "",
            "components": ["toy_ffn"],
            "required_patterns": ["toy_ffn"],
            "optional_patterns": [],
            "signature_ops": ["linear", "activation", "dropout"],
            "formula": "y = MLP(x)",
            "references": [],
        },
    ],
    "components": [
        {"id": "toy_conv", "label": "Conv", "formula": "", "description": "", "required_ops": ["conv"], "tags": []},
        {"id": "toy_pool", "label": "Pool", "formula": "", "description": "", "required_ops": ["pool"], "tags": []},
        {"id": "toy_ffn", "label": "FFN", "formula": "", "description": "", "required_ops": ["linear", "activation"], "tags": []},
    ],
}


def _toy_conv_graph(name: str) -> dict:
    body = [
        conv("n1", 3, 8, out_shape=(1, 8, 16, 16)),
        bn("n2", 8, (1, 8, 16, 16), (1, 8, 16, 16)),
        act("n3", "ReLU", (1, 8, 16, 16)),
        N("n4", "pool", "AdaptiveAvgPool2d", in_shape=(1, 8, 16, 16), out_shape=(1, 8, 1, 1), params={"type": "adaptive", "outputSize": [1, 1]}),
        linear("n5", 8, 4, in_shape=(1, 8), out_shape=(1, 4)),
    ]
    return make_graph(
        name,
        body,
        [("n0", "n1"), ("n1", "n2"), ("n2", "n3"), ("n3", "n4"), ("n4", "n5"), ("n5", "n6")],
        input_shape=(1, 3, 16, 16),
        output_shape=(1, 4),
    )


def test_kb_driven_classification(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(algorithms, "_external_knowledge", lambda: SMALL_KB)
    candidates = classify_families(_toy_conv_graph("ToyNet"))
    assert candidates[0]["id"] == "toy_cnn"
    assert candidates[0]["confidence"] > 0.3

    mlp_graph = make_graph(
        "ToyMLP",
        [
            linear("n1", 8, 16, in_shape=(1, 8), out_shape=(1, 16)),
            act("n2", "GELU", (1, 16)),
            linear("n3", 16, 8, in_shape=(1, 16), out_shape=(1, 8)),
        ],
        [("n0", "n1"), ("n1", "n2"), ("n2", "n3"), ("n3", "n4")],
        input_shape=(1, 8),
        output_shape=(1, 8),
    )
    assert classify_families(mlp_graph)[0]["id"] == "toy_mlp"

    unknown_body = [N("n1", "custom", "mystery", in_shape=(1, 8), out_shape=(1, 8))]
    unknown = make_graph("MysteryNet", unknown_body, [("n0", "n1"), ("n1", "n2")])
    unknown_candidates = classify_families(unknown)
    assert all(candidate["id"] != "unknown" for candidate in unknown_candidates)
    unknown_explanation = explain_graph(unknown)
    assert unknown_explanation["family"] == "unknown"
    assert unknown_explanation["familyConfidence"] <= 0.3


def test_builtin_fallback_when_kb_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(algorithms, "_external_knowledge", lambda: None)
    knowledge = algorithms._knowledge()
    assert knowledge["version"] == "fallback"
    assert len(knowledge["families"]) >= 14
    ids = {family["id"] for family in knowledge["families"]}
    assert {"cnn", "vgg", "resnet", "unet", "transformer_encoder", "transformer_decoder", "mlp"} <= ids

    graph = load_demo(VGG)
    candidates = classify_families(graph)
    assert candidates[0]["id"] in {"vgg", "cnn"}
    explanation = explain_graph(graph)
    assert explanation["family"] in {"vgg", "cnn"}


def test_external_knowledge_base_is_loaded() -> None:
    knowledge = algorithms._knowledge()
    assert isinstance(knowledge, dict)
    family_ids = {family["id"] for family in knowledge.get("families") or []}
    assert "resnet" in family_ids
    if knowledge.get("version") != "fallback":
        assert len(family_ids) >= 60


# --------------------------------------------------------------------------- #
# stdlib-only contract
# --------------------------------------------------------------------------- #
def test_no_torch_or_onnx_import() -> None:
    code = (
        "import sys\n"
        "import netprism.algorithms as algorithms\n"
        "algorithms._knowledge()\n"
        "bad = [name for name in ('torch', 'onnx', 'keras') if name in sys.modules]\n"
        "assert not bad, bad\n"
        "print('ok')\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "ok"


def test_no_network_imports_in_algorithms_module() -> None:
    source = (ROOT / "netprism" / "algorithms.py").read_text(encoding="utf-8")
    for forbidden in ("import torch", "import onnx", "import keras", "import numpy"):
        assert forbidden not in source


def test_render_explain_markdown_contract() -> None:
    graph = sample_graph(TINYGPT)
    markdown = render_explain_md(graph)
    assert "### 算法理解" in markdown
    assert "家族候选" in markdown
    assert "| 家族 |" in markdown
    assert "TransformerBlock" in markdown
    assert "```python" in markdown
    assert markdown.endswith("\n")
