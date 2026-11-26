"""Tests for the NetPrism parser core (``netprism.core``).

The models used here are generated into ``tmp_path`` so the suite never depends
on ``samples/`` being present.  Reference shapes / parameters / FLOPs are
computed from live PyTorch models built in the test process.
"""

from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
import textwrap
import uuid
from pathlib import Path

import jsonschema
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from netprism import schema  # noqa: E402
from netprism.core import (  # noqa: E402
    AnalyzeError,
    AnalyzeOptions,
    FolderResult,
    analyze_file,
    analyze_folder,
)

torch = pytest.importorskip("torch", reason="dynamic tracing tests need torch")
nn = torch.nn


# ---------------------------------------------------------------------------
# fixtures & helpers
# ---------------------------------------------------------------------------

CNN_SOURCE = '''
import torch
import torch.nn as nn


class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(16 * 16 * 16, 10),
        )

    def forward(self, x):
        return self.head(self.features(x))
'''

RESIDUAL_SOURCE = '''
import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out + identity)


class ResidualNet(nn.Module):
    def __init__(self, channels=4):
        super().__init__()
        self.stem = nn.Conv2d(3, channels, 3, padding=1)
        self.layer1 = nn.Sequential(BasicBlock(channels), BasicBlock(channels))
        self.layer2 = nn.Sequential(BasicBlock(channels), BasicBlock(channels))

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        return x
'''

ATTENTION_SOURCE = '''
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads, self.head_dim = num_heads, dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        batch, seq, dim = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) * self.scale
        weights = F.softmax(scores, dim=-1)
        out = weights @ v
        out = out.transpose(1, 2).contiguous().view(batch, seq, dim)
        return self.proj(out)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Linear(dim * mult, dim),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = SelfAttention(dim, num_heads)
        self.ln2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.ffn(self.ln2(x))


class AttentionNet(nn.Module):
    def __init__(self, vocab=32, dim=8, num_heads=2, num_layers=2):
        super().__init__()
        self.token_emb = nn.Embedding(vocab, dim)
        self.blocks = nn.ModuleList([Block(dim, num_heads) for _ in range(num_layers)])

    def forward(self, tokens):
        x = self.token_emb(tokens)
        for block in self.blocks:
            x = block(x)
        return x
'''

TRANSFORMER_SOURCE = '''
import torch
import torch.nn as nn


class TinyTransformer(nn.Module):
    def __init__(self, vocab=32, dim=8, heads=2, layers=2, classes=4):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim * 2,
            dropout=0.0, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, classes)

    def forward(self, tokens):
        x = self.embed(tokens)
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.head(self.norm(x))
'''

UNET_SOURCE = '''
import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class TinyUNet(nn.Module):
    def __init__(self, base=4):
        super().__init__()
        self.enc1 = DoubleConv(3, base)
        self.enc2 = DoubleConv(base, base * 2)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.up = nn.ConvTranspose2d(base * 2, base, kernel_size=2, stride=2)
        self.dec = DoubleConv(base * 2, base)
        self.head = nn.Conv2d(base, 1, kernel_size=1)

    def forward(self, x):
        skip = self.enc1(x)
        x = self.enc2(self.pool(skip))
        x = self.up(x)
        x = self.dec(torch.cat([skip, x], dim=1))
        return self.head(x)
'''

TAIL_SKIP_SOURCE = '''
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(cin, cout, 3, padding=1, bias=False),
            nn.GroupNorm(8, cout),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.net(x)


class UpBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.up = nn.ConvTranspose3d(cin, cout, 2, stride=2)
        self.block = ConvBlock(cout, cout)

    def forward(self, x, skip):
        x = self.up(x)
        x = self.block(x)
        return torch.cat([x, skip], dim=1)


class TailSkipNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = ConvBlock(4, 8)
        self.pool = nn.MaxPool3d(2)
        self.mid = ConvBlock(8, 8)
        self.up = UpBlock(8, 4)

    def forward(self, x):
        skip = self.enc(x)
        y = self.mid(self.pool(skip))
        return self.up(y, skip)
'''

DYNAMIC_SOURCE = '''
import torch
import torch.nn as nn


class DynamicNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten(1),
        )
        self.positive = nn.Linear(8 * 16 * 16, 4)
        self.negative = nn.Linear(8 * 16 * 16, 4)

    def forward(self, x):
        x = self.features(x)
        if x.mean() > 0:
            x = torch.relu(self.positive(x))
        else:
            x = torch.relu(self.negative(x))
        return x
'''

NEEDS_ARGS_SOURCE = '''
import torch
import torch.nn as nn


class NeedsArgs(nn.Module):
    def __init__(self, width, classes=10):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(64, width),
            nn.ReLU(inplace=True),
            nn.Linear(width, width),
            nn.ReLU(inplace=True),
            nn.Linear(width, classes),
        )

    def forward(self, x):
        return self.layers(x)
'''

NO_MODEL_SOURCE = '''
import torch
import torch.nn as nn


def helper(x):
    return x + 1


class Cache(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("data", torch.zeros(3))
'''

TWO_MODELS_SOURCE = '''
import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, 1)

    def forward(self, x):
        return self.conv(x)


class ModelA(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = Block()

    def forward(self, x):
        return self.block(x)


class ModelB(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(16, 4)

    def forward(self, x):
        return self.fc(x)
'''


def _write(directory: Path, name: str, source: str) -> Path:
    path = directory / name
    path.write_text(textwrap.dedent(source).lstrip(), encoding="utf-8")
    return path


def _load_reference_model(path: Path, class_name: str) -> torch.nn.Module:
    module_name = f"_netprism_test_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
    model = getattr(module, class_name)()
    return model.eval()


def _assert_schema(graph: dict) -> None:
    errors = schema.validate_graph(graph)
    assert errors == [], f"graph failed semantic validation: {errors[:5]}"
    jsonschema.validate(instance=graph, schema=schema.load_json_schema())


def _assert_group_invariants(graph: dict) -> None:
    nodes = graph["nodes"]
    groups = graph["groups"]
    node_by_id = {node["id"]: node for node in nodes}
    group_by_id = {group["id"]: group for group in groups}
    assert groups and groups[0]["kind"] == "model"
    root = next(group for group in groups if group["kind"] == "model")
    assert root["parentId"] is None and root["depth"] == 0
    labels = [group["label"] for group in groups]
    assert len(labels) == len(set(labels)), f"group labels must be unique: {labels}"

    memberships: dict[str, str] = {}
    for group in groups:
        for node_id in group["nodeIds"]:
            assert node_id in node_by_id
            assert node_by_id[node_id]["groupId"] == group["id"]
            assert node_id not in memberships, f"{node_id} in two groups"
            memberships[node_id] = group["id"]
        for child_id in group["childGroupIds"]:
            assert group_by_id[child_id]["parentId"] == group["id"]
            assert group_by_id[child_id]["depth"] == group["depth"] + 1
        if group["parentId"] is not None:
            assert group["id"] in group_by_id[group["parentId"]]["childGroupIds"]

    def subtree(group: dict) -> list[str]:
        ids = list(group["nodeIds"])
        for child_id in group["childGroupIds"]:
            ids.extend(subtree(group_by_id[child_id]))
        return ids

    for group in groups:
        ids = subtree(group)
        assert group["numParams"] == sum(node_by_id[node_id]["numParams"] for node_id in ids)
        assert group["flops"] == sum(node_by_id[node_id]["flops"] for node_id in ids)


def _find_node(graph: dict, *, op: str | None = None, kind: str | None = None, name: str | None = None):
    for node in graph["nodes"]:
        if op is not None and node["op"] != op:
            continue
        if kind is not None and node["kind"] != kind:
            continue
        if name is not None and node["name"] != name:
            continue
        return node
    raise AssertionError(f"node not found (op={op}, kind={kind}, name={name})")


def _reference_main_flops(model: torch.nn.Module, input_tensor: torch.Tensor) -> int:
    """Conv/Linear FLOPs computed independently from live module outputs."""
    records: list[tuple[torch.nn.Module, torch.Tensor]] = []
    handles = []

    def make_hook(module: torch.nn.Module):
        def hook(_module, _inputs, output):
            if isinstance(output, torch.Tensor):
                records.append((module, output))

        return hook

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            handles.append(module.register_forward_hook(make_hook(module)))
    with torch.no_grad():
        model(input_tensor)
    for handle in handles:
        handle.remove()

    total = 0
    for module, output in records:
        if isinstance(module, nn.Conv2d):
            batch = output.shape[0]
            spatial = int(output.shape[2]) * int(output.shape[3])
            kernel = module.kernel_size[0] * module.kernel_size[1]
            macs = batch * module.out_channels * spatial * (module.in_channels // module.groups) * kernel
            total += 2 * macs + (batch * module.out_channels * spatial if module.bias is not None else 0)
        else:
            batch = int(output.numel() // module.out_features)
            total += 2 * batch * module.in_features * module.out_features
            if module.bias is not None:
                total += batch * module.out_features
    return int(total)


def _main_flops_from_graph(graph: dict) -> int:
    return int(
        sum(node["flops"] for node in graph["nodes"] if node["kind"] in ("conv", "deconv", "linear", "attention"))
    )


# ---------------------------------------------------------------------------
# dynamic trace accuracy
# ---------------------------------------------------------------------------

def test_dynamic_cnn_shapes_params_flops(tmp_path: Path) -> None:
    path = _write(tmp_path, "tiny_cnn.py", CNN_SOURCE)
    graph = analyze_file(path, AnalyzeOptions(input_shape=(1, 3, 32, 32)))[0]
    _assert_schema(graph)

    model = graph["model"]
    assert model["mode"] == "trace"
    assert model["framework"] == "pytorch"
    assert model["className"] == "TinyCNN"
    assert model["flopsApprox"] is False
    assert model["inputSpec"][0]["shape"] == [1, 3, 32, 32]
    assert model["outputSpec"][0]["shape"] == [1, 10]

    reference = _load_reference_model(path, "TinyCNN")
    expected_params = sum(parameter.numel() for parameter in reference.parameters())
    assert model["totalParams"] == expected_params
    assert model["numLayers"] == sum(1 for node in graph["nodes"] if node["kind"] not in ("input", "output"))

    conv = _find_node(graph, op="Conv2d", name="features.0")
    assert conv["inShapes"] == [[1, 3, 32, 32]]
    assert conv["outShapes"] == [[1, 8, 32, 32]]
    assert conv["params"]["kernelSize"] == [3, 3]
    assert conv["params"]["padding"] == [1, 1]
    maxpool = _find_node(graph, op="MaxPool2d", name="features.3")
    assert maxpool["outShapes"] == [[1, 8, 16, 16]]

    reference_flops = _reference_main_flops(reference, torch.zeros(1, 3, 32, 32))
    graph_flops = _main_flops_from_graph(graph)
    assert abs(graph_flops - reference_flops) <= max(1, 0.01 * reference_flops)

    _assert_group_invariants(graph)
    assert _find_node(graph, op="Conv2d")["groupId"] is not None


def test_residual_edges_and_residual_groups(tmp_path: Path) -> None:
    path = _write(tmp_path, "residual_net.py", RESIDUAL_SOURCE)
    graph = analyze_file(path, AnalyzeOptions(input_shape=(1, 3, 16, 16)))[0]
    _assert_schema(graph)

    add_nodes = [node for node in graph["nodes"] if node["kind"] == "add"]
    assert len(add_nodes) == 4
    residual_edges = [edge for edge in graph["edges"] if edge["kind"] == "residual"]
    assert len(residual_edges) == 4
    for edge in residual_edges:
        assert edge["label"] == "identity"
        assert edge["target"] in {node["id"] for node in add_nodes}
    # The residual source is the block input, not the immediately preceding node.
    assert residual_edges[0]["source"] != str(int(residual_edges[0]["target"][1:]) - 1)

    for add in add_nodes:
        assert "residual" in add["tags"]
    block_groups = [group for group in graph["groups"] if group["kind"] == "residual_block"]
    assert {group["name"] for group in block_groups} == {
        "layer1.0",
        "layer1.1",
        "layer2.0",
        "layer2.1",
    }
    group_by_id = {group["id"]: group for group in graph["groups"]}
    by_name = {group["name"]: group for group in graph["groups"]}
    for group in block_groups:
        parent = group_by_id[group["parentId"]]
        assert group["label"].startswith(parent["label"] + " · ")
        node_ids = set(group["nodeIds"])
        assert node_ids & {node["id"] for node in add_nodes}
    assert by_name["layer1"]["label"] == "Layer 1"
    assert by_name["layer2"]["label"] == "Layer 2"
    assert {group["label"] for group in block_groups} == {
        "Layer 1 · Residual Block 1",
        "Layer 1 · Residual Block 2",
        "Layer 2 · Residual Block 1",
        "Layer 2 · Residual Block 2",
    }
    _assert_group_invariants(graph)

    reference = _load_reference_model(path, "ResidualNet")
    assert graph["model"]["totalParams"] == sum(p.numel() for p in reference.parameters())
    reference_flops = _reference_main_flops(reference, torch.zeros(1, 3, 16, 16))
    assert abs(_main_flops_from_graph(graph) - reference_flops) <= max(1, 0.01 * reference_flops)


def test_attention_dynamic_and_static(tmp_path: Path) -> None:
    path = _write(tmp_path, "attention_net.py", ATTENTION_SOURCE)
    reference = _load_reference_model(path, "AttentionNet")
    expected_params = sum(parameter.numel() for parameter in reference.parameters())

    dynamic = analyze_file(path)[0]
    _assert_schema(dynamic)
    assert dynamic["model"]["mode"] == "trace"
    assert dynamic["model"]["totalParams"] == expected_params
    assert dynamic["model"]["flopsApprox"] is True

    attention_nodes = [node for node in dynamic["nodes"] if node["kind"] == "attention"]
    assert attention_nodes, "hand-written attention should expose attention-kind nodes"
    assert any("numHeads" in node["params"] and node["params"]["numHeads"] == 2 for node in attention_nodes)
    assert any(edge["kind"] == "attention" for edge in dynamic["edges"]), "attention internal edges expected"

    attention_groups = [group for group in dynamic["groups"] if group["kind"] == "attention_block"]
    assert attention_groups, "attention block group expected"
    transformer_groups = [group for group in dynamic["groups"] if group["kind"] == "transformer_block"]
    assert len(transformer_groups) == 2
    labels_by_name = {group["name"]: group["label"] for group in dynamic["groups"]}
    assert labels_by_name["blocks"] == "Transformer Blocks"
    assert labels_by_name["blocks.0.attn"] == "Block 1 · Attention"
    assert labels_by_name["blocks.0.ffn"] == "Block 1 · Feed Forward"
    assert labels_by_name["blocks.1.attn"] == "Block 2 · Attention"
    _assert_group_invariants(dynamic)

    static = analyze_file(path, AnalyzeOptions(execute=False))[0]
    _assert_schema(static)
    assert static["model"]["mode"] == "static"
    assert static["model"]["flopsApprox"] is True
    assert static["model"]["totalParams"] == expected_params
    static_attention = [group for group in static["groups"] if group["kind"] == "attention_block"]
    assert static_attention
    _assert_group_invariants(static)


def test_transformer_encoder_expansion(tmp_path: Path) -> None:
    path = _write(tmp_path, "tiny_transformer.py", TRANSFORMER_SOURCE)
    graph = analyze_file(path)[0]
    _assert_schema(graph)
    assert graph["model"]["mode"] == "trace"
    assert graph["model"]["totalParams"] == sum(
        parameter.numel() for parameter in _load_reference_model(path, "TinyTransformer").parameters()
    )

    mha_nodes = [node for node in graph["nodes"] if node["op"] == "MultiheadAttention"]
    assert len(mha_nodes) == 2, "TransformerEncoderLayer must be expanded into its MHA nodes"
    for node in mha_nodes:
        assert node["params"]["numHeads"] == 2
        assert node["params"]["headDim"] == 4
        assert node["params"]["batchFirst"] is True

    group_by_name = {group["name"]: group for group in graph["groups"]}
    assert group_by_name["encoder"]["kind"] == "encoder"
    assert group_by_name["encoder.layers"]["kind"] in ("module", "sequential")
    assert group_by_name["encoder.layers"]["label"] == "Layers"
    assert group_by_name["encoder.layers.0"]["kind"] == "transformer_block"
    assert group_by_name["encoder.layers.0"]["label"] == "Transformer Block 1"
    assert group_by_name["encoder.layers.1"]["kind"] == "transformer_block"
    assert group_by_name["encoder.layers.1"]["label"] == "Transformer Block 2"
    _assert_group_invariants(graph)

    static = analyze_file(path, AnalyzeOptions(execute=False))[0]
    _assert_schema(static)
    assert static["model"]["mode"] == "static"
    static_groups = {group["name"]: group for group in static["groups"]}
    assert static_groups["encoder.layers.0"]["kind"] == "transformer_block"
    _assert_group_invariants(static)


def test_unet_skip_edges_and_unet_groups(tmp_path: Path) -> None:
    path = _write(tmp_path, "tiny_unet.py", UNET_SOURCE)
    graph = analyze_file(path, AnalyzeOptions(input_shape=(1, 3, 16, 16)))[0]
    _assert_schema(graph)

    concat_nodes = [node for node in graph["nodes"] if node["kind"] == "concat"]
    assert len(concat_nodes) == 1
    concat = concat_nodes[0]
    incoming = [edge for edge in graph["edges"] if edge["target"] == concat["id"]]
    assert len(incoming) == 2
    # Older branch = encoder skip; the newest producer is the decoder main path
    # (marking every concat input as skip drew a bogus skip arc on the upsample).
    order = {node["id"]: index for index, node in enumerate(graph["nodes"])}
    incoming.sort(key=lambda edge: order[edge["source"]])
    assert [edge["kind"] for edge in incoming] == ["skip", "data"]
    by_id = {node["id"]: node for node in graph["nodes"]}
    assert by_id[incoming[0]["source"]]["name"].startswith("enc1")
    assert by_id[incoming[1]["source"]]["name"].startswith("up")
    assert concat["outShapes"] == [[1, 8, 16, 16]]

    unet_groups = [group for group in graph["groups"] if group["kind"] == "unet_block"]
    assert {group["name"] for group in unet_groups} == {"enc1", "enc2", "dec"}
    for group in unet_groups:
        assert group["label"].startswith(("Encoder", "Decoder"))
    assert {group["label"] for group in unet_groups} == {"Encoder 1", "Encoder 2", "Decoder"}
    conv_blocks = [group for group in graph["groups"] if group["kind"] == "conv_block"]
    assert len(conv_blocks) == 3
    assert {group["name"] for group in conv_blocks} == {"enc1.block", "enc2.block", "dec.block"}
    conv_labels = {group["name"]: group["label"] for group in conv_blocks}
    assert conv_labels == {
        "enc1.block": "Encoder 1 · Conv Block",
        "enc2.block": "Encoder 2 · Conv Block",
        "dec.block": "Decoder · Conv Block",
    }
    _assert_group_invariants(graph)

    static = analyze_file(path, AnalyzeOptions(execute=False, input_shape=(1, 3, 16, 16)))[0]
    _assert_schema(static)
    assert any(edge["kind"] == "skip" for edge in static["edges"])
    _assert_group_invariants(static)


def test_trailing_concat_merge_is_promoted_out_of_the_group(tmp_path: Path) -> None:
    """A decoder module's output concat belongs *after* the group.

    ``UpBlock.forward`` ends with ``torch.cat([block(x), skip])``; the tensors only
    line up after the block, so the concat must be laid out after the group box
    (regression: it sat inside the group and skip arcs looked like they ended
    before the decoder block).
    """
    path = _write(tmp_path, "tail_skip.py", TAIL_SKIP_SOURCE)
    graph = analyze_file(
        path, AnalyzeOptions(execute=False, input_shape=(1, 4, 8, 8, 8))
    )[0]
    _assert_schema(graph)

    groups = {group["id"]: group for group in graph["groups"]}
    root = graph["groups"][0]
    up_group = next(group for group in graph["groups"] if group["name"] == "up")
    cat = next(node for node in graph["nodes"] if node["kind"] == "concat")

    assert cat["groupId"] == root["id"], "merge node should move to the parent group"
    assert cat["id"] not in up_group["nodeIds"]
    order = {node["id"]: index for index, node in enumerate(graph["nodes"])}
    member_orders = [order[node_id] for node_id in up_group["nodeIds"] if node_id in order]
    assert member_orders and max(member_orders) < order[cat["id"]]

    skips = [edge for edge in graph["edges"] if edge["kind"] == "skip"]
    assert len(skips) == 1
    assert skips[0]["target"] == cat["id"]
    source = next(node for node in graph["nodes"] if node["id"] == skips[0]["source"])
    assert source["outShapes"][0] in cat["inShapes"]
    assert not any(node["id"] in groups[up_group["id"]]["nodeIds"] for node in [cat])
    _assert_group_invariants(graph)


def test_hook_fallback_for_dynamic_control_flow(tmp_path: Path) -> None:
    path = _write(tmp_path, "dynamic_net.py", DYNAMIC_SOURCE)
    graph = analyze_file(path, AnalyzeOptions(input_shape=(1, 3, 16, 16)))[0]
    _assert_schema(graph)
    assert graph["model"]["mode"] == "trace"
    warnings = graph["model"]["warnings"]
    assert any("hook" in warning.lower() for warning in warnings)
    assert any("FX symbolic tracing failed" in warning for warning in warnings)
    # Hook fallback produces an execution-order chain only.
    assert all(edge["kind"] == "data" for edge in graph["edges"])
    for edge in graph["edges"]:
        source = int(edge["source"][1:])
        target = int(edge["target"][1:])
        assert target == source + 1
    _assert_group_invariants(graph)


# ---------------------------------------------------------------------------
# static fallback
# ---------------------------------------------------------------------------

def test_static_fallback_for_required_constructor_args(tmp_path: Path) -> None:
    path = _write(tmp_path, "needs_args.py", NEEDS_ARGS_SOURCE)
    graph = analyze_file(path)[0]
    _assert_schema(graph)
    assert graph["model"]["mode"] == "static"
    assert graph["model"]["className"] == "NeedsArgs"
    assert graph["model"]["warnings"], "static fallback must explain what could not be resolved"
    assert graph["model"]["flopsApprox"] is True
    assert any(node["kind"] == "linear" for node in graph["nodes"])
    _assert_group_invariants(graph)

    forced_static = analyze_file(path, AnalyzeOptions(execute=False, input_shape=(1, 64)))[0]
    _assert_schema(forced_static)
    assert forced_static["model"]["mode"] == "static"
    assert forced_static["model"]["inputSpec"][0]["shape"] == [1, 64]


def test_static_keeps_main_chain_edges(tmp_path: Path) -> None:
    """Regression: static forward parsing must keep the sequential data edges."""
    path = _write(tmp_path, "tiny_cnn.py", CNN_SOURCE)
    static = analyze_file(path, AnalyzeOptions(execute=False, input_shape=(1, 3, 32, 32)))[0]
    dynamic = analyze_file(path, AnalyzeOptions(input_shape=(1, 3, 32, 32)))[0]
    _assert_schema(static)
    assert len(static["edges"]) >= len(dynamic["edges"]) - 1
    assert any(edge["kind"] == "data" for edge in static["edges"])

    # Every non-input node must be reachable from the input node.
    node_ids = [node["id"] for node in static["nodes"]]
    reachable = {node_ids[0]}
    changed = True
    while changed:
        changed = False
        for edge in static["edges"]:
            if edge["source"] in reachable and edge["target"] not in reachable:
                reachable.add(edge["target"])
                changed = True
    assert set(node_ids) <= reachable, f"unreachable nodes: {set(node_ids) - reachable}"


def test_static_output_dtype_is_float_for_token_inputs(tmp_path: Path) -> None:
    path = _write(tmp_path, "tiny_transformer.py", TRANSFORMER_SOURCE)
    graph = analyze_file(path, AnalyzeOptions(execute=False))[0]
    _assert_schema(graph)
    assert graph["model"]["inputSpec"][0]["dtype"] == "int64"
    assert graph["model"]["outputSpec"][0]["dtype"] == "float32"


def test_static_unknown_constructor_args_use_minus_one(tmp_path: Path) -> None:
    path = _write(tmp_path, "needs_args.py", NEEDS_ARGS_SOURCE)
    graph = analyze_file(path, AnalyzeOptions(execute=False, input_shape=(1, 64)))[0]
    _assert_schema(graph)
    assert graph["model"]["mode"] == "static"
    unknown = [
        node
        for node in graph["nodes"]
        if node["kind"] == "linear" and node["outShapes"] == [[-1]]
    ]
    assert unknown, "unresolvable width must produce -1 shapes, not an assumed 64"
    assert all(node["flops"] == 0 for node in unknown)
    assert any("width" in warning for warning in graph["model"]["warnings"])


def test_static_and_dynamic_agree_on_params_and_shapes(tmp_path: Path) -> None:
    path = _write(tmp_path, "tiny_cnn.py", CNN_SOURCE)
    options = AnalyzeOptions(input_shape=(1, 3, 32, 32))
    dynamic = analyze_file(path, options)[0]
    static = analyze_file(path, AnalyzeOptions(input_shape=(1, 3, 32, 32), execute=False))[0]
    assert dynamic["model"]["mode"] == "trace"
    assert static["model"]["mode"] == "static"
    assert dynamic["model"]["totalParams"] == static["model"]["totalParams"]
    assert dynamic["model"]["inputSpec"][0]["shape"] == static["model"]["inputSpec"][0]["shape"]
    assert dynamic["model"]["outputSpec"][0]["shape"] == static["model"]["outputSpec"][0]["shape"]

    def conv_shapes(graph: dict) -> list:
        return sorted(
            (node["name"], node["inShapes"], node["outShapes"])
            for node in graph["nodes"]
            if node["kind"] in ("conv", "linear")
        )

    assert conv_shapes(dynamic) == conv_shapes(static)


# ---------------------------------------------------------------------------
# discovery / folder scanning / errors
# ---------------------------------------------------------------------------

def test_folder_scan_error_isolation(tmp_path: Path) -> None:
    _write(tmp_path, "good.py", CNN_SOURCE.replace("TinyCNN", "GoodCNN"))
    _write(tmp_path, "broken.py", "def broken(:\n    pass\n")
    _write(tmp_path, "no_model.py", NO_MODEL_SOURCE)
    _write(tmp_path, "needs_args.py", NEEDS_ARGS_SOURCE)

    result = analyze_folder(tmp_path, AnalyzeOptions(input_shape=(1, 3, 32, 32)))
    assert isinstance(result, FolderResult)
    assert result.index["schemaVersion"] == "1.0"
    assert result.index["errors"], "syntax error must be reported per file"
    assert any(error["file"] == "broken.py" for error in result.index["errors"])
    assert all(isinstance(error["message"], str) and error["message"] for error in result.index["errors"])
    assert len(result.index["models"]) >= 2
    ids = {entry["id"] for entry in result.index["models"]}
    assert any("good" in model_id for model_id in ids)
    assert any("needs_args" in model_id for model_id in ids)
    for entry in result.index["models"]:
        graph = result.graphs[entry["id"]]
        _assert_schema(graph)
        assert entry["file"] == f"{entry['id']}.json"
        assert entry["numLayers"] == graph["model"]["numLayers"]
        assert entry["totalParams"] == graph["model"]["totalParams"]
    assert schema.validate_index(result.index) == []


def test_no_models_returns_empty_list(tmp_path: Path) -> None:
    path = _write(tmp_path, "tool.py", NO_MODEL_SOURCE)
    assert analyze_file(path) == []


def test_analyze_file_errors(tmp_path: Path) -> None:
    missing = tmp_path / "missing.py"
    with pytest.raises(AnalyzeError):
        analyze_file(missing)

    broken = tmp_path / "broken.py"
    broken.write_text("class X(nn.Module):\n", encoding="utf-8")
    with pytest.raises(AnalyzeError) as excinfo:
        analyze_file(broken)
    assert "broken.py" in str(excinfo.value)

    path = _write(tmp_path, "two.py", TWO_MODELS_SOURCE)
    with pytest.raises(AnalyzeError):
        analyze_file(path, AnalyzeOptions(class_name="DoesNotExist"))


def test_multiple_top_level_models_and_building_block_detection(tmp_path: Path) -> None:
    path = _write(tmp_path, "two_models.py", TWO_MODELS_SOURCE)
    graphs = analyze_file(path, AnalyzeOptions(input_shape=(1, 3, 16, 16)))
    assert len(graphs) == 2
    assert {graph["model"]["className"] for graph in graphs} == {"ModelA", "ModelB"}
    for graph in graphs:
        _assert_schema(graph)

    graphs = analyze_file(path, AnalyzeOptions(class_name="ModelB", input_shape=(1, 16)))
    assert len(graphs) == 1
    assert graphs[0]["model"]["className"] == "ModelB"


def test_folder_scan_static_mode(tmp_path: Path) -> None:
    _write(tmp_path, "tiny_cnn.py", CNN_SOURCE)
    result = analyze_folder(tmp_path, AnalyzeOptions(execute=False, input_shape=(1, 3, 32, 32)))
    assert result.index["errors"] == []
    assert len(result.index["models"]) == 1
    entry = result.index["models"][0]
    assert entry["mode"] == "static"
    graph = result.graphs[entry["id"]]
    _assert_schema(graph)
    assert graph["model"]["mode"] == "static"


def test_output_is_pure_json(tmp_path: Path) -> None:
    path = _write(tmp_path, "tiny_cnn.py", CNN_SOURCE)
    graph = analyze_file(path, AnalyzeOptions(input_shape=(1, 3, 32, 32)))[0]
    payload = json.dumps(graph, ensure_ascii=False)
    assert json.loads(payload) == graph


# ---------------------------------------------------------------------------
# introspection
# ---------------------------------------------------------------------------

def test_core_import_does_not_import_torch() -> None:
    code = (
        "import sys; import netprism.core; "
        "assert 'torch' not in sys.modules, 'netprism.core must not import torch at import time'"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
    assert "AssertionError" not in result.stderr


# ---------------------------------------------------------------------------
# real-world style fixtures (packages, factories, configs)
# ---------------------------------------------------------------------------

FIXTURES = PROJECT_ROOT / "tests" / "fixtures"


def test_factory_functions_are_reported_as_models() -> None:
    path = FIXTURES / "real_style" / "factory_net.py"
    graphs = analyze_file(path)
    names = [graph["model"]["name"] for graph in graphs]
    assert names == ["SmallNet", "DeepNet"]
    for graph in graphs:
        _assert_schema(graph)
        model = graph["model"]
        assert model["mode"] == "trace"
        assert model["className"] == "FactoryNet"
        assert model["name"] in ("SmallNet", "DeepNet")
        assert model["totalParams"] > 0
        assert "FactoryNet" not in names, "the class behind a factory is not a separate model"
    assert graphs[0]["model"]["totalParams"] != graphs[1]["model"]["totalParams"]


def test_package_relative_imports_trace_standalone() -> None:
    path = FIXTURES / "real_style" / "package_model.py"
    graphs = analyze_file(path)
    assert len(graphs) == 1
    graph = graphs[0]
    _assert_schema(graph)
    assert graph["model"]["name"] == "PackageNet"
    assert graph["model"]["mode"] == "trace"
    assert graph["model"]["totalParams"] > 0
    assert any(node["op"] == "Embedding" for node in graph["nodes"])
    _assert_group_invariants(graph)

    # The whole package must scan without import errors and expose the model.
    result = analyze_folder(FIXTURES / "real_style")
    assert result.index["errors"] == []
    scan_names = {entry["name"] for entry in result.index["models"]}
    assert {"PackageNet", "ConfigNet", "SmallNet", "DeepNet", "FineNet", "StringNet"} <= scan_names
    for entry in result.index["models"]:
        _assert_schema(result.graphs[entry["id"]])


def test_dataclass_config_synthesis() -> None:
    path = FIXTURES / "real_style" / "config_net.py"
    graph = analyze_file(path)[0]
    _assert_schema(graph)
    model = graph["model"]
    assert model["name"] == "ConfigNet"
    assert model["className"] == "ConfigNet"
    assert model["mode"] == "trace"
    assert model["flopsApprox"] is True, "synthesized config must not be reported as exact"
    warnings = " ".join(model["warnings"])
    assert "dataclass" in warnings and "synthesized" in warnings
    assert model["totalParams"] > 0


def test_helper_blocks_and_wrappers_are_not_top_models() -> None:
    path = FIXTURES / "real_style" / "noise.py"
    graphs = analyze_file(path)
    names = [graph["model"]["name"] for graph in graphs]
    assert names == ["FineNet"]
    assert "TinyAttention" not in names
    assert "TrainerLike" not in names
    assert "ModelWrapper" not in names
    graph = graphs[0]
    _assert_schema(graph)
    assert graph["model"]["mode"] == "trace"
    assert any(node["op"] == "Linear" for node in graph["nodes"])


def test_call_site_literals_are_used_for_required_strings() -> None:
    path = FIXTURES / "real_style" / "string_model.py"
    graph = analyze_file(path)[0]
    _assert_schema(graph)
    model = graph["model"]
    assert model["name"] == "StringNet"
    assert model["mode"] == "trace"
    assert model["flopsApprox"] is True
    warnings = " ".join(model["warnings"])
    assert "call-site literal" in warnings and "variant='tiny'" in warnings
    assert model["totalParams"] > 0


def test_same_name_models_deduplicated() -> None:
    result = analyze_folder(FIXTURES / "dedupe_pkg")
    assert result.index["errors"] == []
    assert len(result.index["models"]) == 1
    entry = result.index["models"][0]
    assert entry["name"] == "DuplicateNet"
    assert entry["id"].startswith("a__"), "canonical file should win the same-name dedupe"
    _assert_schema(result.graphs[entry["id"]])


def test_cross_file_components_are_not_top_level_models() -> None:
    """Classes instantiated as sub-modules by another file are components."""
    folder = FIXTURES / "cross_file_pkg"
    result = analyze_folder(folder)
    assert result.index["errors"] == []
    names = {entry["name"] for entry in result.index["models"]}
    assert names == {"MainNet", "SecondNet"}, f"unexpected folder models: {names}"
    for banned in ("DownBlock", "OutBlock", "SharedBlock"):
        assert banned not in names
    for entry in result.index["models"]:
        graph = result.graphs[entry["id"]]
        _assert_schema(graph)
        assert graph["model"]["mode"] == "trace"
        assert graph["model"]["totalParams"] > 0

    # ``train.py`` references MainNet outside a class method: must not consume it.
    assert "MainNet" in names

    # Single-file analysis is unchanged: components still form their own graphs.
    single = analyze_file(folder / "components.py")
    assert {graph["model"]["name"] for graph in single} == {"DownBlock", "OutBlock", "SharedBlock"}
    assert [graph["model"]["name"] for graph in analyze_file(folder / "main_model.py")] == ["MainNet"]


def test_missing_third_party_import_falls_back_to_static() -> None:
    path = FIXTURES / "missing_dep_model.py"
    graph = analyze_file(path)[0]
    _assert_schema(graph)
    model = graph["model"]
    assert model["name"] == "MissingDepNet"
    assert model["mode"] == "static"
    assert not any("could not import module" in warning for warning in model["warnings"])


# ---------------------------------------------------------------------------
# samples integration (only when the samples-qa fixtures exist)
# ---------------------------------------------------------------------------

SAMPLES = PROJECT_ROOT / "samples"


@pytest.mark.skipif(not SAMPLES.is_dir(), reason="samples/ not available yet")
def test_samples_folder_integration() -> None:
    result = analyze_folder(SAMPLES)
    assert result.index["errors"] == []
    assert len(result.index["models"]) >= 5
    assert schema.validate_index(result.index) == []
    modes = {entry["id"]: entry["mode"] for entry in result.index["models"]}
    assert any(mode == "static" for mode in modes.values()), "needs_args.py must use the static fallback"
    for entry in result.index["models"]:
        graph = result.graphs[entry["id"]]
        _assert_schema(graph)
        _assert_group_invariants(graph)
