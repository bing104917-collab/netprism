"""Generate the multi-format fixtures under ``tests/fixtures/formats/``.

* ``.onnx`` and TorchScript ``.pt`` archives are exported from the sample models
  with ``torch`` (``do_constant_folding=False`` keeps the ONNX graph identical
  to the module tree, so parameter counts match PyTorch exactly).
* ``.keras`` fixtures are built by hand as Keras v3 zip archives
  (``config.json`` + ``metadata.json``); Keras / TensorFlow are **not** imported,
  which keeps the fixture generation working on machines where the pip install
  of Keras is blocked.

Run from anywhere::

    python scripts/make_format_fixtures.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"
OUT = ROOT / "tests" / "fixtures" / "formats"
MAX_BYTES = 20 * 1024 * 1024


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _load_sample(name: str):
    path = SAMPLES / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"netprism_sample_{name}", path)
    if spec is None or spec.loader is None:  # pragma: no cover - weird environment
        raise RuntimeError(f"cannot load sample {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _report(path: Path) -> dict[str, object]:
    size = path.stat().st_size
    if size > MAX_BYTES:
        raise RuntimeError(f"{path.name} is {size / 1024 / 1024:.1f} MB, above the 20 MB fixture limit")
    payload = {"file": path.name, "format": path.suffix.lstrip("."), "bytes": size}
    print(f"  {path.name:<28} {size / 1024:8.1f} KiB")
    return payload


def _export_onnx(model, dummy, path: Path) -> None:
    import torch

    kwargs = dict(
        input_names=["x"],
        output_names=["output"],
        opset_version=17,
        do_constant_folding=False,
    )
    try:
        torch.onnx.export(model, dummy, str(path), dynamo=False, **kwargs)
    except TypeError:  # pragma: no cover - older torch without ``dynamo``
        torch.onnx.export(model, dummy, str(path), **kwargs)


def _export_torchscript(model, dummy, path: Path) -> None:
    import torch

    traced = torch.jit.trace(model, dummy, check_trace=False, strict=False)
    traced.save(str(path))


def make_negative_fixtures() -> None:
    """Broken inputs used by the readable-error tests.

    The names contain a ``_negative_name_hint`` token (``broken`` /
    ``state_dict``) so QA's fixture discovery can distinguish them from the
    positive fixtures.
    """
    (OUT / "broken.onnx").write_bytes(b"this is definitely not a valid ONNX protobuf\n")
    (OUT / "broken.keras").write_bytes(b"this is not a zip archive either\n")
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is optional
        return
    module = _load_sample("mlp")
    model = getattr(module, "MLP")()
    torch.save(model.state_dict(), OUT / "mlp_state_dict.pt")


# ---------------------------------------------------------------------------
# keras v3 fixtures (hand written config.json; no Keras import)
# ---------------------------------------------------------------------------

KERAS_VERSION = "3.12.0"


def _keras_layer(
    class_name: str,
    config: dict,
    *,
    name: str,
    inbound: list | None = None,
    build_input_shape: list | None = None,
) -> dict:
    entry: dict = {
        "module": "keras.layers",
        "class_name": class_name,
        "config": {"name": name, "trainable": True, "dtype": "float32", **config},
        "registered_name": None,
        "name": name,
        "inbound_nodes": inbound or [],
    }
    if build_input_shape is not None:
        entry["build_config"] = {"input_shape": build_input_shape}
    return entry


def _write_keras(path: Path, class_name: str, inner_config: dict) -> None:
    config = {
        "module": "keras",
        "class_name": class_name,
        "config": inner_config,
        "keras_version": KERAS_VERSION,
        "backend": "torch",
    }
    metadata = {
        "keras_version": KERAS_VERSION,
        "date_saved": "2026-11-24T15:00:00",
    }
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("config.json", json.dumps(config, indent=2))
        archive.writestr("metadata.json", json.dumps(metadata, indent=2))


def _conv2d(name: str, filters: int, kernel=(3, 3), padding="same", strides=(1, 1), *, inbound: list | None = None,
            build_input_shape: list | None = None) -> dict:
    return _keras_layer(
        "Conv2D",
        {
            "filters": filters,
            "kernel_size": list(kernel),
            "strides": list(strides),
            "padding": padding,
            "data_format": "channels_last",
            "dilation_rate": [1, 1],
            "groups": 1,
            "activation": "linear",
            "use_bias": True,
        },
        name=name,
        inbound=inbound,
        build_input_shape=build_input_shape,
    )


def _batchnorm(name: str, *, inbound: list | None = None) -> dict:
    return _keras_layer(
        "BatchNormalization",
        {
            "axis": -1,
            "momentum": 0.99,
            "epsilon": 0.001,
            "center": True,
            "scale": True,
        },
        name=name,
        inbound=inbound,
    )


def _relu(name: str, *, inbound: list | None = None) -> dict:
    return _keras_layer(
        "ReLU",
        {"max_value": None, "negative_slope": 0.0, "threshold": 0.0},
        name=name,
        inbound=inbound,
    )


def _dense(name: str, units: int, *, activation: str = "linear", inbound: list | None = None,
           build_input_shape: list | None = None) -> dict:
    return _keras_layer(
        "Dense",
        {"units": units, "activation": activation, "use_bias": True},
        name=name,
        inbound=inbound,
        build_input_shape=build_input_shape,
    )


def _inbound(*refs: tuple[str, int, int]) -> list:
    """One functional node with one or more tensor references (Keras format)."""
    return [[[name, node, tensor, {}] for name, node, tensor in refs]]


def make_sequential_cnn() -> None:
    """Sequential CNN without an explicit InputLayer (uses build_input_shape)."""
    input_shape = [None, 16, 16, 3]
    inner = {
        "name": "tiny_cnn",
        "trainable": True,
        "dtype": "float32",
        "layers": [
            _conv2d("conv1", 8, build_input_shape=input_shape),
            _batchnorm("bn1"),
            _relu("relu1"),
            _keras_layer("MaxPooling2D", {"pool_size": [2, 2], "padding": "valid", "strides": [2, 2],
                                          "data_format": "channels_last"}, name="pool1"),
            _conv2d("conv2", 16, padding="valid", build_input_shape=[None, 8, 8, 8]),
            _relu("relu2"),
            _keras_layer("Flatten", {"data_format": "channels_last"}, name="flatten"),
            _dense("head", 10, activation="softmax", build_input_shape=[None, 576]),
        ],
        "build_input_shape": input_shape,
    }
    _write_keras(OUT / "tiny_cnn.keras", "Sequential", inner)


def make_functional_mlp() -> None:
    """Functional MLP: two inputs -> Concatenate -> Dense -> ReLU -> Dense."""
    inner = {
        "name": "tiny_mlp",
        "trainable": True,
        "layers": [
            _keras_layer("InputLayer", {"batch_shape": [None, 16], "sparse": False}, name="a"),
            _keras_layer("InputLayer", {"batch_shape": [None, 16], "sparse": False}, name="b"),
            _keras_layer("Concatenate", {"axis": -1}, name="concat", inbound=_inbound(("a", 0, 0), ("b", 0, 0))),
            _dense("dense1", 32, activation="relu", inbound=_inbound(("concat", 0, 0))),
            _dense("head", 10, inbound=_inbound(("dense1", 0, 0))),
        ],
        "input_layers": [["a", 0, 0], ["b", 0, 0]],
        "output_layers": [["head", 0, 0]],
    }
    _write_keras(OUT / "tiny_mlp.keras", "Functional", inner)


def make_functional_rescnn() -> None:
    """Functional ResNet-ish CNN: residual Add branch + global average pooling."""
    inner = {
        "name": "tiny_rescnn",
        "trainable": True,
        "layers": [
            _keras_layer("InputLayer", {"batch_shape": [None, 32, 32, 3], "sparse": False}, name="x"),
            _conv2d("conv1", 8, inbound=_inbound(("x", 0, 0))),
            _batchnorm("bn1", inbound=_inbound(("conv1", 0, 0))),
            _relu("relu1", inbound=_inbound(("bn1", 0, 0))),
            _conv2d("conv2", 8, inbound=_inbound(("relu1", 0, 0))),
            _batchnorm("bn2", inbound=_inbound(("conv2", 0, 0))),
            _keras_layer("Add", {}, name="add", inbound=_inbound(("bn2", 0, 0), ("relu1", 0, 0))),
            _relu("relu2", inbound=_inbound(("add", 0, 0))),
            _keras_layer("GlobalAveragePooling2D", {"data_format": "channels_last"}, name="gap",
                         inbound=_inbound(("relu2", 0, 0))),
            _dense("head", 10, inbound=_inbound(("gap", 0, 0))),
        ],
        "input_layers": [["x", 0, 0]],
        "output_layers": [["head", 0, 0]],
    }
    _write_keras(OUT / "tiny_rescnn.keras", "Functional", inner)


def make_functional_attention() -> None:
    """Functional text model: Embedding -> LayerNorm -> MHA -> Add -> GAP -> Dense."""
    inner = {
        "name": "tiny_attention",
        "trainable": True,
        "layers": [
            _keras_layer("InputLayer", {"batch_shape": [None, 8], "sparse": False, "dtype": "int32"}, name="tokens"),
            _keras_layer("Embedding", {"input_dim": 32, "output_dim": 16}, name="emb",
                         inbound=_inbound(("tokens", 0, 0))),
            _keras_layer("LayerNormalization", {"axis": -1, "epsilon": 1e-5, "center": True, "scale": True},
                         name="ln", inbound=_inbound(("emb", 0, 0))),
            _keras_layer("MultiHeadAttention", {"num_heads": 2, "key_dim": 8, "value_dim": 8,
                                                "use_bias": True, "dropout": 0.0, "output_shape": None},
                         name="mha", inbound=_inbound(("ln", 0, 0))),
            _keras_layer("Add", {}, name="add", inbound=_inbound(("mha", 0, 0), ("emb", 0, 0))),
            _keras_layer("GlobalAveragePooling1D", {"data_format": "channels_last"}, name="gap",
                         inbound=_inbound(("add", 0, 0))),
            _dense("head", 2, inbound=_inbound(("gap", 0, 0))),
        ],
        "input_layers": [["tokens", 0, 0]],
        "output_layers": [["head", 0, 0]],
    }
    _write_keras(OUT / "tiny_attention.keras", "Functional", inner)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-onnx", action="store_true", help="only generate TorchScript / Keras fixtures")
    parser.add_argument("--skip-torchscript", action="store_true", help="only generate ONNX / Keras fixtures")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []

    print("Keras v3 fixtures (zip/json, no Keras import):")
    make_sequential_cnn()
    make_functional_mlp()
    make_functional_rescnn()
    make_functional_attention()
    for name in ("tiny_cnn.keras", "tiny_mlp.keras", "tiny_rescnn.keras", "tiny_attention.keras"):
        manifest.append(_report(OUT / name))

    try:
        import torch
    except ImportError:  # pragma: no cover - fixtures need torch
        print("torch is not installed; skipping .onnx / .pt fixtures", file=sys.stderr)
        torch = None

    if torch is not None and not args.skip_onnx:
        print("ONNX fixtures (torch.onnx.export, do_constant_folding=False):")
        cases = [
            ("cnn_vgg", "VGGMini", (1, 3, 32, 32), "vggmini.onnx"),
            ("resnet_mini", "ResNetMini", (1, 3, 32, 32), "resnetmini.onnx"),
            ("mlp", "MLP", (1, 64), "mlp.onnx"),
        ]
        for sample, class_name, shape, filename in cases:
            module = _load_sample(sample)
            model = getattr(module, class_name)().eval()
            with torch.no_grad():
                dummy = torch.randn(*shape)
            _export_onnx(model, dummy, OUT / filename)
            manifest.append(_report(OUT / filename))

    if torch is not None and not args.skip_torchscript:
        print("TorchScript fixtures (torch.jit.trace):")
        cases = [
            ("cnn_vgg", "VGGMini", lambda t: torch.randn(1, 3, 32, 32), "vggmini_traced.pt"),
            ("resnet_mini", "ResNetMini", lambda t: torch.randn(1, 3, 32, 32), "resnetmini_traced.pt"),
            ("mlp", "MLP", lambda t: torch.randn(1, 64), "mlp_traced.pt"),
            (
                "transformer_classifier",
                "TinyTransformerClassifier",
                lambda t: torch.randint(0, 128, (1, 16), dtype=torch.int64),
                "tiny_transformer_traced.pt",
            ),
        ]
        for sample, class_name, make_input, filename in cases:
            module = _load_sample(sample)
            model = getattr(module, class_name)().eval()
            _export_torchscript(model, make_input(torch), OUT / filename)
            manifest.append(_report(OUT / filename))

    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("Negative fixtures (readable-error tests):")
    make_negative_fixtures()
    for name in ("broken.onnx", "broken.keras", "mlp_state_dict.pt"):
        path = OUT / name
        if path.is_file():
            manifest.append(_report(path))
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(manifest)} fixtures to {OUT}")
    return 0


if __name__ == "__main__":  # pragma: no cover - script entry point
    sys.exit(main())
