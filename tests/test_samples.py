"""Tests for the NetPrism sample models in ``samples/``.

Each sample file is loaded directly from disk (they are not an importable
package).  The tests pin down the properties the parser/demo rely on:

* every file stays under the 100-line sample budget and only imports
  ``torch``/stdlib,
* default-constructed models produce the documented output shapes on CPU,
* parameter shapes/counts match the architecture,
* the three intentional edge cases behave as designed
  (``needs_args`` cannot be default-constructed, ``dynamic_control`` defeats
  FX tracing but runs eagerly, ``not_a_model`` contains no model).
"""

from __future__ import annotations

import ast
import importlib.util
import math
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch
import torch.fx
import torch.nn as nn
from torch.fx.proxy import TraceError

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"

EXPECTED_FILES = {
    "cnn_vgg.py",
    "resnet_mini.py",
    "transformer_classifier.py",
    "custom_attention.py",
    "unet_mini.py",
    "mlp.py",
    "needs_args.py",
    "dynamic_control.py",
    "not_a_model.py",
}

# (file stem, class name, input shape, input dtype, expected output shape)
MODEL_CASES = [
    ("cnn_vgg", "VGGMini", (1, 3, 32, 32), torch.float32, (1, 10)),
    ("resnet_mini", "ResNetMini", (1, 3, 32, 32), torch.float32, (1, 10)),
    ("transformer_classifier", "TinyTransformerClassifier", (1, 16), torch.int64, (1, 10)),
    ("custom_attention", "TinyGPT", (1, 16), torch.int64, (1, 16, 128)),
    ("unet_mini", "UNetMini", (1, 3, 32, 32), torch.float32, (1, 1, 32, 32)),
    ("mlp", "MLP", (1, 64), torch.float32, (1, 10)),
]

# sum(p.numel() for p in model.parameters()), pinned from the architecture.
EXPECTED_PARAMS = {
    "cnn_vgg": 66474,
    "resnet_mini": 44890,
    "transformer_classifier": 29898,
    "custom_attention": 33920,
    "unet_mini": 117361,
    "mlp": 17226,
}

_LOADED: dict[str, ModuleType] = {}


def load_sample(name: str) -> ModuleType:
    """Import a sample module by file path (cached per session)."""
    if name in _LOADED:
        return _LOADED[name]
    path = SAMPLES / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_netprism_sample_{name}", path)
    assert spec is not None and spec.loader is not None, f"cannot load {path}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _LOADED[name] = module
    return module


def make_input(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    if dtype is torch.int64:
        # token ids: stay inside both embeddings' vocabularies (padding_idx=0 skipped)
        return torch.randint(1, 64, shape, dtype=torch.int64)
    return torch.randn(*shape, dtype=dtype)


def build_model(stem: str, class_name: str) -> nn.Module:
    model = getattr(load_sample(stem), class_name)()
    assert isinstance(model, nn.Module)
    return model.eval()


def num_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# --------------------------------------------------------------------------- #
# File-level hygiene
# --------------------------------------------------------------------------- #


def test_expected_sample_files_exist() -> None:
    actual = {path.name for path in SAMPLES.glob("*.py")}
    assert actual == EXPECTED_FILES, f"unexpected samples: {sorted(actual ^ EXPECTED_FILES)}"
    assert (SAMPLES / "README.md").is_file()


@pytest.mark.parametrize("filename", sorted(EXPECTED_FILES))
def test_sample_file_parses_and_is_small(filename: str) -> None:
    path = SAMPLES / filename
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))

    line_count = len(source.splitlines())
    assert line_count < 100, f"{filename} is {line_count} lines (budget: <100)"

    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert imported <= {"__future__", "math", "torch"}, f"{filename} imports {sorted(imported)}"


def test_model_files_document_io_shapes() -> None:
    for stem, *_ in MODEL_CASES:
        doc = load_sample(stem).__doc__ or ""
        assert "Input" in doc and "Output" in doc, f"{stem}.py must document I/O shapes"


# --------------------------------------------------------------------------- #
# Construction, forward, shapes, parameter counts
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("stem,class_name,input_shape,dtype,output_shape", MODEL_CASES)
def test_default_construction_forward(
    stem: str,
    class_name: str,
    input_shape: tuple[int, ...],
    dtype: torch.dtype,
    output_shape: tuple[int, ...],
) -> None:
    model = build_model(stem, class_name)
    x = make_input(input_shape, dtype)

    with torch.no_grad():
        out = model(x)

    assert tuple(out.shape) == output_shape
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all(), f"{stem}: non-finite outputs"
    assert out.numel() == math.prod(output_shape)


@pytest.mark.parametrize("stem,class_name,input_shape,dtype,output_shape", MODEL_CASES)
def test_parameter_count_and_trainability(
    stem: str,
    class_name: str,
    input_shape: tuple[int, ...],
    dtype: torch.dtype,
    output_shape: tuple[int, ...],
) -> None:
    model = build_model(stem, class_name)
    total = num_params(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    assert total == EXPECTED_PARAMS[stem]
    assert trainable == total, f"{stem}: some parameters are frozen"
    assert len(list(model.parameters())) > 0


@pytest.mark.parametrize("stem,class_name,input_shape,dtype,output_shape", MODEL_CASES)
def test_eval_mode_is_deterministic(
    stem: str,
    class_name: str,
    input_shape: tuple[int, ...],
    dtype: torch.dtype,
    output_shape: tuple[int, ...],
) -> None:
    model = build_model(stem, class_name)
    x = make_input(input_shape, dtype)
    with torch.no_grad():
        first = model(x)
        second = model(x)
    torch.testing.assert_close(first, second, rtol=0, atol=0)


@pytest.mark.parametrize("stem,class_name,input_shape,dtype,output_shape", MODEL_CASES)
def test_batch_dimension_is_supported(
    stem: str,
    class_name: str,
    input_shape: tuple[int, ...],
    dtype: torch.dtype,
    output_shape: tuple[int, ...],
) -> None:
    model = build_model(stem, class_name)
    batched_shape = (2, *input_shape[1:])
    x = make_input(batched_shape, dtype)
    with torch.no_grad():
        out = model(x)
    assert tuple(out.shape) == (2, *output_shape[1:])


# --------------------------------------------------------------------------- #
# Per-model structural assertions
# --------------------------------------------------------------------------- #


def test_vggmini_structure() -> None:
    model = build_model("cnn_vgg", "VGGMini")
    features = model.features
    assert isinstance(features, nn.Sequential)
    assert isinstance(features[0], nn.Conv2d)
    assert features[0].weight.shape == (16, 3, 3, 3)
    assert isinstance(features[9], nn.Conv2d)
    assert features[9].weight.shape == (32, 32, 1, 1), "expected a 1x1 conv"
    assert tuple(model.classifier[1].weight.shape) == (32, 64)
    assert model.classifier[-1].out_features == 10
    assert sum(isinstance(m, nn.BatchNorm2d) for m in model.modules()) == 4


def test_resnetmini_structure_and_residual_add() -> None:
    model = build_model("resnet_mini", "ResNetMini")
    block_type = type(model.layer1[0])
    assert block_type.__name__ == "BasicBlock"

    # layer1 keeps resolution, layer2 downsamples via stride 2.
    assert model.layer1[0].conv1.stride == (1, 1)
    assert model.layer2[0].conv1.stride == (2, 2)
    # 1x1 projection shortcut channel layout (identity / change of channels).
    assert model.layer1[0].shortcut[0].weight.shape == (16, 16, 1, 1)
    assert model.layer2[0].shortcut[0].weight.shape == (32, 16, 1, 1)
    assert model.fc.in_features == 32

    block = model.layer1[0]
    x = torch.randn(1, 16, 32, 32)
    with torch.no_grad():
        residual = block.bn2(block.conv2(block.relu(block.bn1(block.conv1(x)))))
        expected = block.relu(residual + block.shortcut(x))
        out = block(x)
    torch.testing.assert_close(out, expected)


def test_transformer_classifier_structure() -> None:
    model = build_model("transformer_classifier", "TinyTransformerClassifier")
    assert isinstance(model.token_embed, nn.Embedding)
    assert model.token_embed.num_embeddings == 128
    assert model.token_embed.embedding_dim == 32

    assert "pos_encoding" in dict(model.named_buffers())
    assert tuple(model.pos_encoding.shape) == (1, 16, 32)

    encoder = model.encoder
    assert len(encoder.layers) == 2
    layer = encoder.layers[0]
    assert layer.self_attn.batch_first is True
    assert layer.self_attn.embed_dim == 32
    assert layer.self_attn.num_heads == 4
    assert isinstance(layer.linear1, nn.Linear) and isinstance(layer.linear2, nn.Linear)
    assert isinstance(model.norm, nn.LayerNorm)
    assert model.head.out_features == 10


def test_tinygpt_structure_and_causal_mask() -> None:
    model = build_model("custom_attention", "TinyGPT")
    attention = model.blocks[0].attn
    assert attention.qkv.weight.shape == (96, 32)  # 3 * dim
    assert attention.proj.weight.shape == (32, 32)
    assert model.head.weight.shape == (128, 32)

    mask = attention.causal_mask
    assert mask.dtype == torch.bool
    assert tuple(mask.shape) == (1, 1, 16, 16)
    future = torch.triu(torch.ones(16, 16, dtype=torch.bool), diagonal=1)
    assert torch.equal(mask[0, 0], future), "causal mask must hide future positions"

    # Values must not leak from future tokens into earlier positions.
    tokens = torch.randint(1, 128, (1, 16))
    changed = tokens.clone()
    changed[:, 10:] = torch.randint(1, 128, (1, 6))
    with torch.no_grad():
        before = model(tokens)
        after = model(changed)
    delta = (before - after).abs().amax(dim=-1)[0]
    assert torch.all(delta[:10] == 0), "future tokens leaked into the past"
    assert delta[10:].max() > 0


def test_unetmini_structure_and_skip_cats(monkeypatch: pytest.MonkeyPatch) -> None:
    model = build_model("unet_mini", "UNetMini")
    assert isinstance(model.pool, nn.MaxPool2d)
    assert isinstance(model.up1, nn.ConvTranspose2d)
    assert isinstance(model.up2, nn.ConvTranspose2d)
    assert model.up1.kernel_size == (2, 2) and model.up1.stride == (2, 2)
    assert model.head.kernel_size == (1, 1)

    cat_calls: list[tuple[int, int]] = []
    real_cat = torch.cat

    def recording_cat(tensors, dim=0, *, out=None):
        cat_calls.append((len(tensors), dim))
        return real_cat(tensors, dim=dim, out=out) if out is not None else real_cat(tensors, dim=dim)

    monkeypatch.setattr(torch, "cat", recording_cat)
    with torch.no_grad():
        out = model(torch.randn(1, 3, 32, 32))
    monkeypatch.undo()

    assert tuple(out.shape) == (1, 1, 32, 32)
    assert len(cat_calls) == 2, f"expected 2 skip concats, got {cat_calls}"
    assert all(parts == 2 and dim == 1 for parts, dim in cat_calls)


def test_trace_unetmini_marks_only_the_encoder_branch_as_skip() -> None:
    """Regression: trace used to mark *both* concat inputs of UNet as skip.

    The upsampled decoder branch is the main data path; only the encoder branch
    is a skip connection.  Marking both drew a bogus skip arc on UNetMini.
    """
    from netprism.core import AnalyzeOptions, analyze_file

    graph = analyze_file(
        SAMPLES / "unet_mini.py", AnalyzeOptions(input_shape=(1, 3, 64, 64))
    )[0]
    by_id = {node["id"]: node for node in graph["nodes"]}
    skips = [edge for edge in graph["edges"] if edge["kind"] == "skip"]
    assert len(skips) == 2, [
        (by_id[edge["source"]]["name"], by_id[edge["target"]]["name"]) for edge in skips
    ]
    assert sorted(by_id[edge["source"]]["name"] for edge in skips) == [
        "enc1.block.5",
        "enc2.block.5",
    ]
    concat_ids = {node["id"] for node in graph["nodes"] if node["kind"] == "concat"}
    main_path = [
        edge
        for edge in graph["edges"]
        if edge["target"] in concat_ids and by_id[edge["source"]]["kind"] == "deconv"
    ]
    assert len(main_path) == 2
    assert all(edge["kind"] == "data" for edge in main_path)
    for edge in skips:
        source_shape = by_id[edge["source"]]["outShapes"][0]
        assert source_shape in by_id[edge["target"]]["inShapes"]


def test_mlp_structure() -> None:
    model = build_model("mlp", "MLP")
    net = model.net
    assert tuple(net[0].weight.shape) == (128, 64)
    assert tuple(net[3].weight.shape) == (64, 128)
    assert tuple(net[5].weight.shape) == (10, 64)
    assert isinstance(net[2], nn.Dropout)
    assert isinstance(net[1], nn.GELU)


# --------------------------------------------------------------------------- #
# Intentional edge cases
# --------------------------------------------------------------------------- #


def test_needs_args_requires_width_and_static_only() -> None:
    needs_args_cls = getattr(load_sample("needs_args"), "NeedsArgs")

    with pytest.raises(TypeError, match="width"):
        needs_args_cls()

    model = needs_args_cls(width=32).eval()
    assert model.layers[0].weight.shape == (32, 64)
    assert model.layers[2].weight.shape == (32, 32)
    assert model.layers[4].weight.shape == (10, 32)
    with torch.no_grad():
        out = model(torch.randn(1, 64))
    assert tuple(out.shape) == (1, 10)


def test_dynamic_control_runs_eagerly_but_not_under_fx() -> None:
    dynamic_cls = getattr(load_sample("dynamic_control"), "DynamicNet")
    model = dynamic_cls().eval()
    x = torch.randn(1, 3, 32, 32)

    with torch.no_grad():
        out = model(x)
        out_negative = model(-x)
    assert tuple(out.shape) == (1, 10)
    assert tuple(out_negative.shape) == (1, 10)

    with pytest.raises(TraceError):
        torch.fx.symbolic_trace(model)


def test_dynamic_control_branches_on_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dynamic_cls = getattr(load_sample("dynamic_control"), "DynamicNet")
    model = dynamic_cls().eval()

    # Replace the feature extractor so x.mean() has a predictable sign.
    def features_with_value(value: float):
        return lambda x: torch.full((x.shape[0], 32, 1, 1), value)

    calls: list[str] = []
    model.positive.register_forward_hook(lambda *_: calls.append("positive"))
    model.negative.register_forward_hook(lambda *_: calls.append("negative"))

    with torch.no_grad():
        monkeypatch.setattr(model.features, "forward", features_with_value(1.0))
        model(torch.zeros(1, 3, 32, 32))
        monkeypatch.setattr(model.features, "forward", features_with_value(-1.0))
        model(torch.zeros(1, 3, 32, 32))
    monkeypatch.undo()

    assert calls == ["positive", "negative"]


def test_not_a_model_has_no_callable_model() -> None:
    module = load_sample("not_a_model")
    tree = ast.parse((SAMPLES / "not_a_model.py").read_text(encoding="utf-8"))

    model_classes = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            methods = {child.name for child in node.body if isinstance(child, ast.FunctionDef)}
            if "forward" in methods:
                model_classes.append(node.name)
    assert model_classes == [], f"not_a_model.py contains forward methods: {model_classes}"

    # The nn.Module subclass intentionally has no forward, so discovery skips it.
    feature_cache = getattr(module, "FeatureCache")
    assert issubclass(feature_cache, nn.Module)
    assert "forward" not in feature_cache.__dict__

    cache = feature_cache(8)
    assert tuple(cache.features.shape) == (0, 8)

    linear = nn.Linear(4, 3)
    assert module.count_parameters(linear) == 15
    normalized = module.normalize(torch.tensor([[3.0, 4.0]]))
    torch.testing.assert_close(normalized, torch.tensor([[0.6, 0.8]]))
    values, indices = module.top_k(torch.tensor([[0.1, 0.9, 0.5]]), k=2)
    torch.testing.assert_close(values, torch.tensor([[0.9, 0.5]]))
    assert indices.tolist() == [[1, 2]]
    assert module.accuracy(torch.tensor([[0.9, 0.1]]), torch.tensor([0])) == 100.0
