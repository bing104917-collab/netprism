"""Keras v3 adapter (``.keras``).

A ``.keras`` file is a zip archive; the architecture lives in ``config.json``.
This adapter parses ``Sequential`` and ``Functional`` configs with the standard
library only (``zipfile`` + ``json``) - it never imports TensorFlow or Keras.

Shapes and parameters are derived from the layer configs and the propagated
input shapes, so ``flopsApprox`` is always ``True`` for Keras graphs.
"""

from __future__ import annotations

import json
import logging
import zipfile
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..core.errors import AnalyzeError
from ._common import (
    assemble_graph,
    make_boundary_node,
    make_edge,
    make_node,
    shape_numel,
    to_shape,
)

LOGGER = logging.getLogger("netprism.adapters.keras")

_ACTIVATION_LAYERS = {
    "Activation",
    "ReLU",
    "LeakyReLU",
    "ELU",
    "SELU",
    "GELU",
    "SiLU",
    "Softmax",
    "Softplus",
    "PReLU",
    "ThresholdedReLU",
}

_KIND_BY_LAYER: dict[str, str] = {
    "Dense": "linear",
    "EinsumDense": "linear",
    "Conv1D": "conv",
    "Conv2D": "conv",
    "Conv3D": "conv",
    "DepthwiseConv2D": "conv",
    "SeparableConv1D": "conv",
    "SeparableConv2D": "conv",
    "Conv1DTranspose": "deconv",
    "Conv2DTranspose": "deconv",
    "Conv3DTranspose": "deconv",
    "BatchNormalization": "norm",
    "LayerNormalization": "norm",
    "GroupNormalization": "norm",
    "UnitNormalization": "norm",
    "MaxPooling1D": "pool",
    "MaxPooling2D": "pool",
    "MaxPooling3D": "pool",
    "AveragePooling1D": "pool",
    "AveragePooling2D": "pool",
    "AveragePooling3D": "pool",
    "GlobalMaxPooling1D": "pool",
    "GlobalMaxPooling2D": "pool",
    "GlobalMaxPooling3D": "pool",
    "GlobalAveragePooling1D": "pool",
    "GlobalAveragePooling2D": "pool",
    "GlobalAveragePooling3D": "pool",
    "Dropout": "dropout",
    "SpatialDropout1D": "dropout",
    "SpatialDropout2D": "dropout",
    "SpatialDropout3D": "dropout",
    "GaussianDropout": "dropout",
    "GaussianNoise": "dropout",
    "Flatten": "flatten",
    "Reshape": "reshape",
    "Permute": "reshape",
    "Embedding": "embedding",
    "MultiHeadAttention": "attention",
    "Add": "add",
    "Subtract": "add",
    "Multiply": "add",
    "Average": "add",
    "Maximum": "add",
    "Minimum": "add",
    "Concatenate": "concat",
    "Identity": "custom",
    "Lambda": "custom",
}

_PARAMETER_FREE = {
    "Activation",
    "ReLU",
    "LeakyReLU",
    "ELU",
    "SELU",
    "GELU",
    "SiLU",
    "Softmax",
    "Softplus",
    "PReLU",
    "Dropout",
    "SpatialDropout1D",
    "SpatialDropout2D",
    "SpatialDropout3D",
    "GaussianDropout",
    "GaussianNoise",
    "Flatten",
    "Reshape",
    "Permute",
    "MaxPooling1D",
    "MaxPooling2D",
    "MaxPooling3D",
    "AveragePooling1D",
    "AveragePooling2D",
    "AveragePooling3D",
    "GlobalMaxPooling1D",
    "GlobalMaxPooling2D",
    "GlobalMaxPooling3D",
    "GlobalAveragePooling1D",
    "GlobalAveragePooling2D",
    "GlobalAveragePooling3D",
    "Add",
    "Subtract",
    "Multiply",
    "Average",
    "Maximum",
    "Minimum",
    "Concatenate",
    "Identity",
}

_NORM_LAYERS = {"BatchNormalization", "LayerNormalization", "GroupNormalization", "UnitNormalization"}


# ---------------------------------------------------------------------------
# config helpers
# ---------------------------------------------------------------------------

def _layer_class(entry: dict[str, Any]) -> str:
    return str(entry.get("class_name") or "")


def _layer_config(entry: dict[str, Any]) -> dict[str, Any]:
    config = entry.get("config")
    return config if isinstance(config, dict) else {}


def _layer_name(entry: dict[str, Any], index: int) -> str:
    config = _layer_config(entry)
    name = entry.get("name") or config.get("name")
    return str(name) if name else f"{_layer_class(entry) or 'layer'}_{index}"


def _is_input_layer(entry: dict[str, Any]) -> bool:
    return _layer_class(entry) == "InputLayer"


def _input_shape(config: dict[str, Any]) -> list[int] | None:
    for key in ("batch_shape", "batch_input_shape", "input_shape"):
        value = config.get(key)
        if isinstance(value, (list, tuple)) and value:
            return to_shape(value)
    shape = config.get("shape")
    if isinstance(shape, (list, tuple)) and shape:
        batch = config.get("batch_size")
        return to_shape([batch if batch is not None else -1, *shape])
    build_config = config.get("build_config")
    if isinstance(build_config, dict):
        value = build_config.get("input_shape")
        if isinstance(value, (list, tuple)) and value:
            return to_shape(value)
    return None


def _iter_inbound_refs(entry: dict[str, Any]) -> list[tuple[str, int, int]]:
    """Extract ``(layer_name, node_index, tensor_index)`` from ``inbound_nodes``.

    Supports the Keras 2/3 list form (``[[name, node, tensor, {}]]``) and the
    Keras 3 dict form (``[{"args": [{"class_name": "__tensor__", "config":
    {"keras_history": [...]}}]}]``).
    """
    refs: list[tuple[str, int, int]] = []
    seen: set[tuple[str, int, int]] = set()

    def add(value: Sequence[Any]) -> None:
        if len(value) < 3 or not isinstance(value[0], str):
            return
        try:
            ref = (str(value[0]), int(value[1]), int(value[2]))
        except (TypeError, ValueError):
            return
        if ref not in seen:
            seen.add(ref)
            refs.append(ref)

    def walk(value: Any, depth: int = 0) -> None:
        if depth > 8:
            return
        if isinstance(value, dict):
            history = value.get("keras_history")
            if isinstance(history, (list, tuple)) and history:
                add(history)
            for item in value.values():
                walk(item, depth + 1)
        elif isinstance(value, (list, tuple)):
            if len(value) >= 3 and isinstance(value[0], str) and isinstance(value[1], int) and isinstance(value[2], int):
                add(value)
                return
            for item in value:
                walk(item, depth + 1)

    walk(entry.get("inbound_nodes") or [])
    return refs


def _read_archive(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
            if "config.json" not in names:
                raise AnalyzeError(
                    "not a valid Keras v3 archive: config.json is missing",
                    file=path,
                )
            try:
                config = json.loads(archive.read("config.json").decode("utf-8"))
            except Exception as exc:  # noqa: BLE001
                raise AnalyzeError(f"invalid Keras config.json ({type(exc).__name__}: {exc})", file=path) from exc
            metadata: dict[str, Any] = {}
            for candidate in ("metadata.json", "config.json"):
                if candidate == "config.json":
                    continue
                if candidate in names:
                    try:
                        metadata = json.loads(archive.read(candidate).decode("utf-8"))
                    except Exception:  # noqa: BLE001 - metadata is optional
                        metadata = {}
    except zipfile.BadZipFile as exc:
        raise AnalyzeError(
            "not a valid Keras v3 archive: expected a zip file containing config.json "
            "(Keras v2 .h5 files are not supported)",
            file=path,
        ) from exc
    if not isinstance(config, dict):
        raise AnalyzeError("invalid Keras config.json: top level must be an object", file=path)
    return config, metadata


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------

def _tuple(value: Any, rank: int, default: int = 1) -> list[int]:
    if value is None:
        return [default] * rank
    if isinstance(value, int):
        return [int(value)] * rank
    if isinstance(value, (list, tuple)):
        values = [int(item) for item in value]
        if len(values) == 1:
            return values * rank
        return values[:rank] + [default] * max(0, rank - len(values))
    return [int(value)] * rank


def _prod(shape: Iterable[int]) -> int:
    total = 1
    for dim in shape:
        if int(dim) < 0:
            return -1
        total *= int(dim)
    return total


def _elems(shape: Sequence[int]) -> int:
    """Element count for FLOPs; an unknown batch dim counts as one sample."""
    if not shape:
        return 0
    total = 1
    for index, dim in enumerate(shape):
        value = int(dim)
        if value < 0:
            if index == 0:
                value = 1  # unknown batch: estimate per-sample FLOPs
            else:
                return 0
        total *= value
    return total


def _conv_out(size: int, kernel: int, stride: int, padding: int, dilation: int = 1) -> int:
    if size < 0:
        return -1
    return (size + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1


def _pool_out(size: int, kernel: int, stride: int) -> int:
    if size < 0:
        return -1
    return (size - kernel) // stride + 1


def _layout(shape: Sequence[int], rank: int, data_format: str) -> tuple[int, int, list[int]] | None:
    """Return ``(batch, channels, spatial)`` for a conv/pool input."""
    shape = list(shape)
    if not shape:
        return None
    batch = int(shape[0])
    if data_format == "channels_first":
        if len(shape) < rank + 2:
            return None
        channels = int(shape[1])
        spatial = [int(dim) for dim in shape[2 : 2 + rank]]
    else:
        if len(shape) < rank + 2:
            return None
        channels = int(shape[-1])
        spatial = [int(dim) for dim in shape[1 : 1 + rank]]
    return batch, channels, spatial


def _spatial_shape(batch: int, channels: int, spatial: Sequence[int], data_format: str) -> list[int]:
    if data_format == "channels_first":
        return [batch, channels, *[int(dim) for dim in spatial]]
    return [batch, *[int(dim) for dim in spatial], channels]


def _channel_axis(shape: Sequence[int], data_format: str) -> int:
    return 1 if data_format == "channels_first" else len(shape) - 1


def _broadcast(left: Sequence[int], right: Sequence[int]) -> list[int]:
    if not left or not right:
        return list(left or right or [-1])
    result: list[int] = []
    for offset in range(1, max(len(left), len(right)) + 1):
        a = left[-offset] if offset <= len(left) else 1
        b = right[-offset] if offset <= len(right) else 1
        if a == 1:
            result.append(b)
        elif b == 1:
            result.append(a)
        elif a < 0 or b < 0:
            result.append(-1)
        else:
            result.append(max(a, b))
    result.reverse()
    return result or [-1]


# ---------------------------------------------------------------------------
# layer inference
# ---------------------------------------------------------------------------

class _LayerResult:
    __slots__ = ("kind", "op", "params", "num_params", "flops", "out_shapes", "notes")

    def __init__(
        self,
        *,
        kind: str,
        op: str,
        params: dict[str, Any] | None = None,
        num_params: int = 0,
        flops: int = 0,
        out_shapes: list[list[int]] | None = None,
        notes: list[str] | None = None,
    ) -> None:
        self.kind = kind
        self.op = op
        self.params = params or {}
        self.num_params = int(num_params)
        self.flops = int(flops)
        self.out_shapes = out_shapes or [[-1]]
        self.notes = notes or []


def _conv_result(
    class_name: str,
    config: dict[str, Any],
    in_shapes: Sequence[list[int]],
    notes: list[str],
) -> _LayerResult:
    rank = 1 if "1D" in class_name else (3 if "3D" in class_name else 2)
    data_format = str(config.get("data_format") or "channels_last")
    filters = int(config.get("filters", 0) or 0)
    kernel = _tuple(config.get("kernel_size"), rank)
    strides = _tuple(config.get("strides"), rank)
    dilation = _tuple(config.get("dilation_rate"), rank)
    padding = str(config.get("padding") or "valid")
    groups = max(1, int(config.get("groups", 1) or 1))
    use_bias = bool(config.get("use_bias", True))
    input_shape = list(in_shapes[0]) if in_shapes and in_shapes[0] else [-1]
    geometry = _layout(input_shape, rank, data_format)
    params: dict[str, Any] = {
        "outChannels": filters,
        "kernelSize": kernel,
        "stride": strides,
        "padding": padding,
        "dilation": dilation,
        "groups": groups,
        "bias": use_bias,
    }
    if geometry is None:
        notes.append(f"{class_name}: unknown input rank, output shape uses -1")
        return _LayerResult(
            kind="conv",
            op=class_name,
            params=params,
            num_params=0,
            out_shapes=[[-1]],
            notes=notes,
        )
    batch, channels, spatial = geometry
    in_channels = channels
    params["inChannels"] = int(in_channels)
    out_spatial: list[int] = []
    if padding == "same":
        out_spatial = [-1 if size < 0 else -(-size // strides[index]) for index, size in enumerate(spatial)]
    elif padding == "causal":
        out_spatial = list(spatial)
    else:
        out_spatial = [
            _conv_out(size, kernel[index], strides[index], 0, dilation[index])
            for index, size in enumerate(spatial)
        ]
    out_shape = _spatial_shape(batch, filters, out_spatial, data_format)
    kernel_prod = _prod(kernel) if _prod(kernel) > 0 else 1
    num_params = kernel_prod * (in_channels // groups if in_channels > 0 else 0) * filters
    if num_params < 0:
        num_params = 0
    if use_bias:
        num_params += filters
    out_elems = _elems(out_shape)
    flops = 0
    if out_elems and in_channels > 0 and kernel_prod > 0:
        flops = 2 * out_elems * (in_channels // groups) * kernel_prod
        if use_bias:
            flops += out_elems
    return _LayerResult(
        kind="conv",
        op=class_name,
        params=params,
        num_params=num_params,
        flops=flops,
        out_shapes=[out_shape],
    )


def _depthwise_conv_result(
    class_name: str,
    config: dict[str, Any],
    in_shapes: Sequence[list[int]],
    notes: list[str],
) -> _LayerResult:
    rank = 1 if "1D" in class_name else 2
    data_format = str(config.get("data_format") or "channels_last")
    kernel = _tuple(config.get("kernel_size"), rank)
    strides = _tuple(config.get("strides"), rank)
    dilation = _tuple(config.get("dilation_rate"), rank)
    padding = str(config.get("padding") or "valid")
    multiplier = max(1, int(config.get("depth_multiplier", 1) or 1))
    use_bias = bool(config.get("use_bias", True))
    input_shape = list(in_shapes[0]) if in_shapes and in_shapes[0] else [-1]
    geometry = _layout(input_shape, rank, data_format)
    if geometry is None:
        notes.append(f"{class_name}: unknown input rank, output shape uses -1")
        return _LayerResult(kind="conv", op=class_name, out_shapes=[[-1]], notes=notes)
    batch, in_channels, spatial = geometry
    out_channels = in_channels * multiplier
    if padding == "same":
        out_spatial = [-1 if size < 0 else -(-size // strides[index]) for index, size in enumerate(spatial)]
    else:
        out_spatial = [_conv_out(size, kernel[index], strides[index], 0, dilation[index]) for index, size in enumerate(spatial)]
    out_shape = _spatial_shape(batch, out_channels, out_spatial, data_format)
    kernel_prod = _prod(kernel) if _prod(kernel) > 0 else 1
    num_params = kernel_prod * max(0, in_channels) * multiplier + (out_channels if use_bias else 0)
    out_elems = _elems(out_shape)
    flops = 2 * out_elems * kernel_prod + (out_elems if use_bias else 0) if out_elems else 0
    params = {
        "inChannels": int(in_channels),
        "outChannels": int(out_channels),
        "kernelSize": kernel,
        "stride": strides,
        "padding": padding,
        "dilation": dilation,
        "depthMultiplier": multiplier,
        "bias": use_bias,
    }
    return _LayerResult(
        kind="conv",
        op=class_name,
        params=params,
        num_params=num_params,
        flops=flops,
        out_shapes=[out_shape],
    )


def _separable_conv_result(
    class_name: str,
    config: dict[str, Any],
    in_shapes: Sequence[list[int]],
    notes: list[str],
) -> _LayerResult:
    """SeparableConv1D / SeparableConv2D: depthwise + pointwise convolution."""
    rank = 1 if "1D" in class_name else 2
    data_format = str(config.get("data_format") or "channels_last")
    filters = int(config.get("filters", 0) or 0)
    kernel = _tuple(config.get("kernel_size"), rank)
    strides = _tuple(config.get("strides"), rank)
    dilation = _tuple(config.get("dilation_rate"), rank)
    padding = str(config.get("padding") or "valid")
    multiplier = max(1, int(config.get("depth_multiplier", 1) or 1))
    use_bias = bool(config.get("use_bias", True))
    input_shape = list(in_shapes[0]) if in_shapes and in_shapes[0] else [-1]
    geometry = _layout(input_shape, rank, data_format)
    if geometry is None:
        notes.append(f"{class_name}: unknown input rank, output shape uses -1")
        return _LayerResult(kind="conv", op=class_name, out_shapes=[[-1]], notes=notes)
    batch, in_channels, spatial = geometry
    if padding == "same":
        out_spatial = [-1 if size < 0 else -(-size // strides[index]) for index, size in enumerate(spatial)]
    elif padding == "causal":
        out_spatial = list(spatial)
    else:
        out_spatial = [_conv_out(size, kernel[index], strides[index], 0, dilation[index]) for index, size in enumerate(spatial)]
    out_shape = _spatial_shape(batch, filters, out_spatial, data_format)
    kernel_prod = _prod(kernel) if _prod(kernel) > 0 else 1
    mid_channels = max(0, in_channels) * multiplier
    depthwise_params = kernel_prod * mid_channels + (mid_channels if use_bias else 0)
    pointwise_params = mid_channels * filters + (filters if use_bias else 0)
    num_params = depthwise_params + pointwise_params
    depthwise_elems = _elems(_spatial_shape(batch, mid_channels, out_spatial, data_format))
    out_elems = _elems(out_shape)
    flops = 2 * depthwise_elems * kernel_prod
    if use_bias:
        flops += depthwise_elems
    flops += 2 * out_elems * mid_channels * filters
    if use_bias:
        flops += out_elems
    params = {
        "inChannels": int(in_channels),
        "outChannels": filters,
        "kernelSize": kernel,
        "stride": strides,
        "padding": padding,
        "dilation": dilation,
        "depthMultiplier": multiplier,
        "bias": use_bias,
        "separable": True,
    }
    return _LayerResult(
        kind="conv",
        op=class_name,
        params=params,
        num_params=num_params,
        flops=int(flops),
        out_shapes=[out_shape],
    )


def _pool_result(class_name: str, config: dict[str, Any], in_shapes: Sequence[list[int]], notes: list[str]) -> _LayerResult:
    rank = 1 if "1D" in class_name else (3 if "3D" in class_name else 2)
    data_format = str(config.get("data_format") or "channels_last")
    input_shape = list(in_shapes[0]) if in_shapes and in_shapes[0] else [-1]
    global_pool = class_name.startswith("Global")
    geometry = _layout(input_shape, rank, data_format)
    params: dict[str, Any] = {"type": "global" if global_pool else ("max" if "Max" in class_name else "avg")}
    if geometry is None:
        notes.append(f"{class_name}: unknown input rank, output shape uses -1")
        return _LayerResult(kind="pool", op=class_name, params=params, out_shapes=[[-1]], notes=notes)
    batch, channels, spatial = geometry
    if global_pool:
        out_shape = [batch, channels]
    else:
        kernel = _tuple(config.get("pool_size"), rank)
        strides = _tuple(config.get("strides"), rank) if config.get("strides") else list(kernel)
        padding = str(config.get("padding") or "valid")
        params["kernelSize"] = kernel
        params["stride"] = strides
        params["padding"] = padding
        out_spatial = [
            (-1 if size < 0 else (-(-size // strides[index]) if padding == "same" else _pool_out(size, kernel[index], strides[index])))
            for index, size in enumerate(spatial)
        ]
        out_shape = _spatial_shape(batch, channels, out_spatial, data_format)
    in_elems = _elems(input_shape)
    return _LayerResult(
        kind="pool",
        op=class_name,
        params=params,
        flops=int(in_elems),
        out_shapes=[out_shape],
    )


def _norm_result(class_name: str, config: dict[str, Any], in_shapes: Sequence[list[int]], notes: list[str]) -> _LayerResult:
    input_shape = list(in_shapes[0]) if in_shapes and in_shapes[0] else [-1]
    data_format = str(config.get("data_format") or "channels_last")
    kind = "norm"
    params: dict[str, Any] = {"eps": float(config.get("epsilon", 1e-3))}
    num_params = 0
    if class_name == "BatchNormalization":
        axis = int(config.get("axis", -1) if config.get("axis") is not None else -1)
        channels = input_shape[axis] if -len(input_shape) <= axis < len(input_shape) else -1
        channels = int(channels) if channels is not None else -1
        scale = bool(config.get("scale", True))
        center = bool(config.get("center", True))
        params.update(
            {
                "normalizedShape": [channels] if channels > 0 else [-1],
                "affine": scale or center,
                "center": center,
                "scale": scale,
            }
        )
        if channels > 0:
            num_params = (channels if scale else 0) + (channels if center else 0) + 2 * channels
    else:
        axis = config.get("axis", -1)
        axes = [axis] if isinstance(axis, int) else list(axis or [-1])
        normalized: list[int] = []
        for item in axes:
            index = int(item)
            if -len(input_shape) <= index < len(input_shape):
                normalized.append(int(input_shape[index]))
        params["normalizedShape"] = normalized or [-1]
        size = _prod(normalized) if normalized else -1
        scale = bool(config.get("scale", True))
        center = bool(config.get("center", True))
        params["affine"] = scale or center
        if size and size > 0:
            num_params = (size if scale else 0) + (size if center else 0)
    out_elems = _elems(input_shape)
    return _LayerResult(
        kind=kind,
        op=class_name,
        params=params,
        num_params=num_params,
        flops=int(out_elems * 4),
        out_shapes=[input_shape],
    )


def _linear_result(config: dict[str, Any], in_shapes: Sequence[list[int]], notes: list[str]) -> _LayerResult:
    units = int(config.get("units", 0) or 0)
    use_bias = bool(config.get("use_bias", True))
    input_shape = list(in_shapes[0]) if in_shapes and in_shapes[0] else [-1]
    in_features = int(input_shape[-1]) if input_shape and input_shape[-1] >= 0 else -1
    params = {"inFeatures": in_features, "outFeatures": units, "bias": use_bias}
    num_params = (in_features * units if in_features > 0 else 0) + (units if use_bias else 0)
    out_shape = [*input_shape[:-1], units] if input_shape else [units]
    flops = 0
    out_elems = _elems(out_shape)
    if in_features > 0:
        flops = 2 * out_elems * in_features + (out_elems if use_bias else 0)
    return _LayerResult(
        kind="linear",
        op="Dense",
        params=params,
        num_params=num_params,
        flops=int(flops),
        out_shapes=[out_shape],
    )


def _embedding_result(config: dict[str, Any], in_shapes: Sequence[list[int]]) -> _LayerResult:
    input_dim = int(config.get("input_dim", 0) or 0)
    output_dim = int(config.get("output_dim", 0) or 0)
    input_shape = list(in_shapes[0]) if in_shapes and in_shapes[0] else [-1]
    out_shape = [*input_shape, output_dim] if output_dim else [-1]
    params = {"numEmbeddings": input_dim, "embeddingDim": output_dim}
    return _LayerResult(
        kind="embedding",
        op="Embedding",
        params=params,
        num_params=input_dim * output_dim,
        flops=0,
        out_shapes=[out_shape],
    )


def _mha_result(config: dict[str, Any], in_shapes: Sequence[list[int]], notes: list[str]) -> _LayerResult:
    num_heads = max(1, int(config.get("num_heads", 1) or 1))
    key_dim = int(config.get("key_dim", 0) or 0)
    value_dim = int(config.get("value_dim", key_dim) or key_dim)
    use_bias = bool(config.get("use_bias", True))
    query_shape = list(in_shapes[0]) if in_shapes and in_shapes[0] else [-1]
    d_model = int(query_shape[-1]) if query_shape and query_shape[-1] >= 0 else -1
    output_shape = config.get("output_shape")
    out_features = int(output_shape[-1]) if isinstance(output_shape, (list, tuple)) and output_shape else d_model
    if d_model < 0:
        notes.append("MultiHeadAttention: unknown embedding dim; parameters/FLOPs are -1/0")
    q_params = d_model * num_heads * key_dim + (num_heads * key_dim if use_bias else 0) if d_model > 0 else 0
    k_params = q_params
    v_params = d_model * num_heads * value_dim + (num_heads * value_dim if use_bias else 0) if d_model > 0 else 0
    out_params = num_heads * value_dim * out_features + (out_features if use_bias else 0) if out_features > 0 else 0
    num_params = q_params * 2 + v_params + out_params
    out_shape = [*query_shape[:-1], out_features] if query_shape and out_features > 0 else [-1]
    params = {
        "embedDim": d_model,
        "numHeads": num_heads,
        "keyDim": key_dim,
        "valueDim": value_dim,
        "bias": use_bias,
    }
    flops = 0
    seq = query_shape[-2] if len(query_shape) >= 2 else -1
    batch = query_shape[0] if query_shape else -1
    if batch < 0:
        batch = 1  # unknown batch: estimate per-sample FLOPs
    if d_model > 0 and seq > 0 and batch > 0:
        flops = 4 * batch * seq * d_model * d_model + 2 * batch * seq * seq * d_model
    return _LayerResult(
        kind="attention",
        op="MultiHeadAttention",
        params=params,
        num_params=num_params,
        flops=flops,
        out_shapes=[out_shape],
    )


def _infer_layer(
    class_name: str,
    config: dict[str, Any],
    in_shapes: Sequence[list[int]],
    notes: list[str],
) -> _LayerResult:
    first = list(in_shapes[0]) if in_shapes and in_shapes[0] else [-1]

    if class_name in ("Dense", "EinsumDense"):
        return _linear_result(config, in_shapes, notes)
    if class_name.startswith("SeparableConv"):
        return _separable_conv_result(class_name, config, in_shapes, notes)
    if class_name.startswith("DepthwiseConv"):
        return _depthwise_conv_result(class_name, config, in_shapes, notes)
    if class_name.startswith("Conv"):
        return _conv_result(class_name, config, in_shapes, notes)
    if class_name in ("MaxPooling1D", "MaxPooling2D", "MaxPooling3D", "AveragePooling1D", "AveragePooling2D",
                      "AveragePooling3D", "GlobalMaxPooling1D", "GlobalMaxPooling2D", "GlobalMaxPooling3D",
                      "GlobalAveragePooling1D", "GlobalAveragePooling2D", "GlobalAveragePooling3D"):
        return _pool_result(class_name, config, in_shapes, notes)
    if class_name in _NORM_LAYERS:
        return _norm_result(class_name, config, in_shapes, notes)
    if class_name == "Embedding":
        return _embedding_result(config, in_shapes)
    if class_name == "MultiHeadAttention":
        return _mha_result(config, in_shapes, notes)
    if class_name == "Dropout" or class_name.startswith("SpatialDropout") or class_name in ("GaussianDropout", "GaussianNoise"):
        rate = config.get("rate", config.get("p", 0.0))
        params = {"p": float(rate)} if rate is not None else {}
        return _LayerResult(kind="dropout", op=class_name, params=params, out_shapes=[first])
    if class_name == "Flatten":
        data_format = str(config.get("data_format") or "channels_last")
        if len(first) >= 2 and all(dim >= 0 for dim in first[1:]):
            flat = 1
            for dim in first[1:]:
                flat *= int(dim)
            out = [first[0], flat]
        else:
            out = [-1]
        return _LayerResult(kind="flatten", op="Flatten", out_shapes=[out])
    if class_name == "Reshape":
        target = config.get("target_shape") or config.get("shape") or []
        target = [int(dim) for dim in target]
        batch = first[0] if first else -1
        out = [batch, *target]
        if target and all(dim >= 0 for dim in first[1:]) and all(dim >= 0 for dim in target):
            total = _prod(first[1:])
            wanted = _prod(target)
            if total > 0 and wanted > 0 and total != wanted:
                notes.append(f"Reshape: target shape {target} does not match input size {total}")
        return _LayerResult(kind="reshape", op="Reshape", params={"shape": out}, out_shapes=[out])
    if class_name in ("Add", "Subtract", "Multiply", "Average", "Maximum", "Minimum"):
        out = list(in_shapes[0]) if in_shapes else [-1]
        for shape in in_shapes[1:]:
            out = _broadcast(out, shape)
        flops = _elems(out)
        return _LayerResult(kind="add", op=class_name, flops=flops, out_shapes=[out])
    if class_name == "Concatenate":
        axis = int(config.get("axis", -1) if config.get("axis") is not None else -1)
        if not in_shapes:
            return _LayerResult(kind="concat", op="Concatenate", out_shapes=[[-1]])
        out = list(in_shapes[0])
        if axis < 0:
            axis += len(out)
        if 0 <= axis < len(out):
            total = 0
            for shape in in_shapes:
                if axis >= len(shape) or shape[axis] < 0:
                    total = -1
                    break
                total += int(shape[axis])
            out[axis] = total
        out_elems = _elems(out)
        return _LayerResult(kind="concat", op="Concatenate", params={"axis": axis}, flops=out_elems, out_shapes=[out])
    if class_name in _ACTIVATION_LAYERS:
        kind = "softmax" if class_name == "Softmax" else "activation"
        out_elems = _elems(first)
        return _LayerResult(
            kind=kind,
            op=class_name,
            params={"activation": str(config.get("activation") or class_name)},
            flops=out_elems if kind == "activation" else 4 * out_elems,
            out_shapes=[first],
        )
    if class_name == "Identity":
        return _LayerResult(kind="custom", op="Identity", out_shapes=[first])
    # Unknown layer: keep it as a custom node with a warning.
    notes.append(f"unsupported Keras layer {class_name!r}; kept as a custom node with shape -1")
    return _LayerResult(kind="custom", op=class_name or "Custom", out_shapes=[[-1]])


# ---------------------------------------------------------------------------
# main analysis
# ---------------------------------------------------------------------------

def analyze_keras(path: str | Path, *, max_nodes: int = 20000) -> list[dict[str, Any]]:
    """Analyze one Keras v3 ``.keras`` archive and return a single graph document."""
    file_path = Path(path).expanduser()
    config, metadata = _read_archive(file_path)
    model_class = str(config.get("class_name") or "Model")
    inner = config.get("config")
    if not isinstance(inner, dict):
        raise AnalyzeError("invalid Keras config.json: missing 'config' object", file=file_path)
    layers = inner.get("layers")
    if not isinstance(layers, list) or not layers:
        raise AnalyzeError("invalid Keras config.json: no layers found", file=file_path)
    if len(layers) + 2 > max_nodes:
        raise AnalyzeError(
            f"model has {len(layers)} Keras layers, above the max_nodes limit ({max_nodes})",
            file=file_path,
        )

    warnings_list: list[str] = []
    layer_entries: list[tuple[int, dict[str, Any]]] = [
        (index, entry) for index, entry in enumerate(layers) if isinstance(entry, dict)
    ]
    if not layer_entries:
        raise AnalyzeError("invalid Keras config.json: no usable layer entries", file=file_path)

    class_names = [_layer_class(entry) for _index, entry in layer_entries]
    entries_by_index: dict[int, dict[str, Any]] = dict(layer_entries)
    if model_class == "Sequential":
        sequential = True
    elif model_class in ("Functional", "Model"):
        sequential = not bool(inner.get("input_layers"))
    else:
        sequential = not any(_iter_inbound_refs(entry) for _index, entry in layer_entries)

    names: dict[int, str] = {}
    for index, entry in layer_entries:
        names[index] = _layer_name(entry, index)
    name_to_index = {name: index for index, name in names.items()}

    # Resolve inbound producers per layer (layer config order index).
    inbound: dict[int, list[int]] = {}
    explicit_input_layers: list[int] = []
    for index, entry in layer_entries:
        if _is_input_layer(entry):
            explicit_input_layers.append(index)
            inbound[index] = []
            continue
        refs = _iter_inbound_refs(entry)
        producers: list[int] = []
        for name, _node_index, _tensor_index in refs:
            producer = name_to_index.get(name)
            if producer is None or producer == index:
                continue
            if producer not in producers:
                producers.append(producer)
        # Keras 3 also stores build_config.input_shape on the first functional layer.
        if not producers:
            build_config = entry.get("build_config")
            if isinstance(build_config, dict) and build_config.get("input_shape") and index not in inbound:
                pass
        inbound[index] = producers

    # Sequential: chain layers (InputLayers, if any, anchor the chain).
    order: list[int] = [index for index, _entry in layer_entries]
    if sequential:
        chain: list[int] = []
        anchor: list[int] = []
        for index in order:
            if _is_input_layer(entries_by_index[index]):
                anchor.append(index)
            else:
                chain.append(index)
        ordered: list[int] = []
        for index in chain:
            ordered.append(index)
        previous: int | None = anchor[0] if anchor else None
        for index in ordered:
            inbound[index] = [previous] if previous is not None else []
            previous = index
        order = anchor + ordered if anchor else ordered
    else:
        # Topological order for Functional configs (creation order is usually
        # already topological, but residual branches may not be).
        ordered: list[int] = []
        remaining = list(order)
        while remaining:
            progressed = False
            for index in list(remaining):
                deps = [dep for dep in inbound.get(index, []) if dep not in ordered and dep != index]
                if not deps:
                    ordered.append(index)
                    remaining.remove(index)
                    progressed = True
            if not progressed:
                ordered.extend(remaining)
                warnings_list.append("Keras layer graph has unresolved dependencies; some edges may be missing.")
                break
        order = ordered

    # Build input boundary nodes.
    nodes: list[dict[str, Any]] = []
    producer_node: dict[int, dict[str, Any]] = {}
    input_specs: list[dict[str, Any]] = []
    input_node_ids: dict[int, str] = {}
    synthetic_input_shape: list[int] | None = None
    if not explicit_input_layers:
        candidate = inner.get("build_input_shape")
        if isinstance(candidate, (list, tuple)) and candidate:
            synthetic_input_shape = to_shape(candidate)
    for index in explicit_input_layers:
        entry = entries_by_index[index]
        layer_config = _layer_config(entry)
        shape = _input_shape(layer_config) or [-1]
        dtype = str(layer_config.get("dtype") or "float32")
        if dtype.startswith("<dtype:"):
            dtype = dtype.split(":", 1)[1].strip("'>")
        node = make_boundary_node(name=names[index], op="Input", kind="input", out_shapes=[shape])
        node["id"] = f"n{len(nodes)}"
        input_node_ids[index] = node["id"]
        producer_node[index] = node
        nodes.append(node)
        input_specs.append({"name": names[index], "shape": shape, "dtype": dtype})
    synthetic_input_id: str | None = None
    if not input_specs:
        if synthetic_input_shape is None:
            synthetic_input_shape = [-1]
            warnings_list.append("Keras config does not record an input shape; input shape uses -1.")
        node = make_boundary_node(name="input", op="Input", kind="input", out_shapes=[synthetic_input_shape])
        node["id"] = f"n{len(nodes)}"
        nodes.append(node)
        input_specs.append({"name": "input", "shape": synthetic_input_shape, "dtype": "float32"})
        synthetic_input_id = node["id"]
    first_regular = first_regular_layer(order, layer_entries)

    edges: list[dict[str, Any]] = []
    name_counts: dict[str, int] = {}
    output_layer_indices: list[int] = []
    output_refs = inner.get("output_layers")
    if isinstance(output_refs, list) and output_refs:
        for ref in output_refs:
            if isinstance(ref, (list, tuple)) and ref:
                producer = name_to_index.get(str(ref[0]))
                if producer is not None:
                    output_layer_indices.append(producer)
    if not output_layer_indices:
        for index in reversed(order):
            if not _is_input_layer(entries_by_index[index]):
                output_layer_indices.append(index)
                break

    total_params = 0
    trainable_params = 0
    node_by_layer: dict[int, dict[str, Any]] = {}
    for index in order:
        entry = entries_by_index[index]
        class_name = _layer_class(entry) or "Custom"
        layer_config = _layer_config(entry)
        if _is_input_layer(entry):
            continue
        in_shapes: list[list[int]] = []
        source_indices = inbound.get(index, [])
        for source in source_indices:
            source_node = producer_node.get(source)
            if source_node is not None and source_node["outShapes"]:
                in_shapes.append(list(source_node["outShapes"][0]))
        if not in_shapes and index == first_regular:
            in_shapes = [list(synthetic_input_shape or [-1])]
        if not in_shapes and layer_config.get("build_config") is None:
            build = entry.get("build_config")
            if isinstance(build, dict):
                shape = build.get("input_shape")
                if isinstance(shape, (list, tuple)) and shape:
                    in_shapes = [to_shape(shape)]

        notes: list[str] = []
        try:
            result = _infer_layer(class_name, layer_config, in_shapes, notes)
        except Exception as exc:  # noqa: BLE001 - a broken layer must not abort the model
            LOGGER.debug("Keras layer inference failed for %s", class_name, exc_info=True)
            warnings_list.append(f"could not infer layer {names[index]!r} ({type(exc).__name__}: {exc}); shape uses -1")
            result = _LayerResult(kind="custom", op=class_name, out_shapes=[[-1]])
            notes = []

        for note in notes:
            warnings_list.append(f"{names[index]}: {note}")

        kind = result.kind
        op_name = result.op
        base_name = names[index]
        count = name_counts.get(base_name, 0) + 1
        name_counts[base_name] = count
        display_name = base_name if count == 1 else f"{base_name}#{count}"
        node = make_node(
            name=display_name,
            op=op_name,
            kind=kind,
            module_path="",
            in_shapes=in_shapes,
            out_shapes=result.out_shapes,
            params=result.params,
            num_params=result.num_params,
            flops=result.flops,
            meta={"keras_layer": class_name},
        )
        node["id"] = f"n{len(nodes)}"
        nodes.append(node)
        producer_node[index] = node
        node_by_layer[index] = node
        total_params += result.num_params
        if bool(layer_config.get("trainable", True)):
            trainable_params += result.num_params

        sources = source_indices
        if not sources and synthetic_input_id is not None and index == first_regular:
            sources = [-1]
        seen: set[str] = set()
        for source in sources:
            if source == -1:
                source_node = {"id": synthetic_input_id, "outShapes": [synthetic_input_shape or [-1]]}
            else:
                source_node = producer_node.get(source)
            if source_node is None or source_node.get("id") in seen:
                continue
            seen.add(source_node["id"])
            edge_kind = "residual" if (kind == "add" and len(seen) > 1) else "data"
            edges.append(
                make_edge(
                    source=source_node["id"],
                    target=node["id"],
                    shapes=source_node.get("outShapes", [])[:1],
                    kind=edge_kind,
                )
            )

    # Output boundary nodes.
    output_specs: list[dict[str, Any]] = []
    output_nodes: list[dict[str, Any]] = []
    for position, layer_index in enumerate(dict.fromkeys(output_layer_indices)):
        producer = producer_node.get(layer_index)
        shape = list(producer["outShapes"][0]) if producer and producer.get("outShapes") else [-1]
        spec_name = "output" if position == 0 else f"output_{position + 1}"
        output_specs.append({"name": spec_name, "shape": shape, "dtype": "float32"})
        output_node = make_boundary_node(name=spec_name, op="Output", kind="output", in_shapes=[shape], out_shapes=[shape])
        output_node["id"] = f"n{len(nodes) + len(output_nodes)}"
        output_nodes.append(output_node)
        if producer is not None:
            edges.append(make_edge(source=producer["id"], target=output_node["id"], shapes=producer["outShapes"][:1]))
    nodes.extend(output_nodes)
    if not output_specs:
        output_specs = [{"name": "output", "shape": [-1], "dtype": "float32"}]
        warnings_list.append("could not determine the Keras output layers; output shape uses -1.")

    keras_version = config.get("keras_version") or metadata.get("keras_version")
    backend = config.get("backend") or metadata.get("backend")
    meta = {
        "format": "keras",
        "kerasVersion": str(keras_version) if keras_version is not None else None,
        "backend": str(backend) if backend is not None else None,
        "modelClass": model_class,
        "layerCount": len(layer_entries),
    }
    model_name = str(inner.get("name") or model_class)
    graph = assemble_graph(
        path=file_path,
        framework="keras",
        mode="keras",
        model_name=model_name,
        class_name=model_class,
        nodes=nodes,
        edges=edges,
        input_specs=input_specs or [{"name": "input", "shape": [-1], "dtype": "float32"}],
        output_specs=output_specs,
        warnings=warnings_list,
        total_params=total_params,
        trainable_params=trainable_params,
        total_flops=sum(int(node.get("flops", 0)) for node in nodes),
        flops_approx=True,
        meta=meta,
    )
    return [graph]


def first_regular_layer(order: Sequence[int], layer_entries: Sequence[tuple[int, dict[str, Any]]]) -> int | None:
    """Index of the first non-InputLayer entry in config order."""
    mapping = dict(layer_entries)
    for index in order:
        if not _is_input_layer(mapping[index]):
            return index
    return None


__all__ = ["analyze_keras"]
