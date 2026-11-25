"""Operator classification, parameter extraction and shape / FLOPs math.

This module is deliberately **torch-free at import time**: the static parser must
work on machines without PyTorch installed.  Live ``nn.Module`` objects are only
inspected through plain attribute access (duck typing), and every helper that
needs torch imports it lazily.

FLOPs convention (SPEC 4.5): ``1 MAC == 2 FLOPs``; bias additions are counted.
"""

from __future__ import annotations

import logging
import math
import operator
from typing import Any, Iterable, Sequence

LOGGER = logging.getLogger("netprism.core.ops")

Shape = list[int]

# ---------------------------------------------------------------------------
# kind tables
# ---------------------------------------------------------------------------

#: Names that are *exactly* one of the interesting kinds.
_ACTIVATIONS = (
    "ReLU",
    "ReLU6",
    "GELU",
    "SiLU",
    "Swish",
    "LeakyReLU",
    "Sigmoid",
    "Tanh",
    "Hardswish",
    "Hardtanh",
    "Hardsigmoid",
    "Mish",
    "ELU",
    "SELU",
    "CELU",
    "PReLU",
    "GLU",
    "Softplus",
    "Softsign",
    "Threshold",
    "LogSigmoid",
    "RReLU",
)

_POOL_NAMES = (
    "MaxPool1d",
    "MaxPool2d",
    "MaxPool3d",
    "AvgPool1d",
    "AvgPool2d",
    "AvgPool3d",
    "AdaptiveAvgPool1d",
    "AdaptiveAvgPool2d",
    "AdaptiveAvgPool3d",
    "AdaptiveMaxPool1d",
    "AdaptiveMaxPool2d",
    "AdaptiveMaxPool3d",
    "MaxUnpool1d",
    "MaxUnpool2d",
    "MaxUnpool3d",
    "LPPool1d",
    "LPPool2d",
    "FractionalMaxPool2d",
)

_RNN_NAMES = ("LSTM", "GRU", "RNN", "LSTMCell", "GRUCell", "RNNCell")

_NORM_SUFFIXES = (
    "BatchNorm1d",
    "BatchNorm2d",
    "BatchNorm3d",
    "SyncBatchNorm",
    "InstanceNorm1d",
    "InstanceNorm2d",
    "InstanceNorm3d",
    "LayerNorm",
    "GroupNorm",
    "RMSNorm",
    "LocalResponseNorm",
    "CrossMapLRN2d",
)

_EMBEDDING_NAMES = ("Embedding", "EmbeddingBag", "PositionalEncoding", "PositionalEmbedding", "LearnedPositionalEncoding")

_TRANSFORMER_FUSED = (
    "TransformerEncoderLayer",
    "TransformerDecoderLayer",
    "TransformerEncoder",
    "TransformerDecoder",
    "Transformer",
)

_CONTAINER_NAMES = (
    "Sequential",
    "ModuleList",
    "ModuleDict",
    "ParameterList",
    "ParameterDict",
    "Module",
)

#: call_function / call_method names that carry no visual information (they are
#: shape plumbing or constant construction).  They are skipped, but their data
#: dependencies are transparently forwarded to later nodes.
SKIP_FUNCTIONS = frozenset(
    {
        "getattr",
        "size",
        "dim",
        "ndim",
        "numel",
        "item",
        "tolist",
        "to",
        "type",
        "type_as",
        "is_floating_point",
        "is_contiguous",
        "device",
        "dtype",
        "arange",
        "range",
        "zeros",
        "ones",
        "full",
        "empty",
        "randn",
        "rand",
        "rand_like",
        "zeros_like",
        "ones_like",
        "full_like",
        "triu",
        "tril",
        "eye",
        "tensor",
        "scalar_tensor",
        "contiguous",
        "clone",
        "detach",
        "float",
        "long",
        "int",
        "bool",
        "half",
        "bfloat16",
        "getitem",
        "len",
        "isinstance",
        "where",
    }
)

#: call_function / call_method names that produce a graph node.
EFFECTIVE_FUNCTIONS = frozenset(
    {
        "relu",
        "relu6",
        "gelu",
        "silu",
        "swish",
        "leaky_relu",
        "prelu",
        "elu",
        "selu",
        "celu",
        "mish",
        "hardswish",
        "hardsigmoid",
        "hardtanh",
        "sigmoid",
        "tanh",
        "glu",
        "softplus",
        "softmax",
        "log_softmax",
        "flatten",
        "unflatten",
        "view",
        "reshape",
        "squeeze",
        "unsqueeze",
        "permute",
        "transpose",
        "expand",
        "repeat",
        "chunk",
        "split",
        "unfold",
        "add",
        "add_",
        "sub",
        "sub_",
        "mul",
        "mul_",
        "div",
        "truediv",
        "pow",
        "neg",
        "cat",
        "concat",
        "stack",
        "matmul",
        "bmm",
        "einsum",
        "scaled_dot_product_attention",
        "masked_fill",
        "masked_fill_",
        "dropout",
        "mean",
        "sum",
        "amax",
        "amin",
        "max",
        "min",
        "prod",
        "avg_pool1d",
        "avg_pool2d",
        "avg_pool3d",
        "max_pool1d",
        "max_pool2d",
        "max_pool3d",
        "adaptive_avg_pool1d",
        "adaptive_avg_pool2d",
        "adaptive_avg_pool3d",
        "adaptive_max_pool1d",
        "adaptive_max_pool2d",
        "adaptive_max_pool3d",
        "interpolate",
        "upsample",
        "upsample_nearest2d",
        "upsample_bilinear2d",
        "embedding",
        "one_hot",
        "linear",
        "conv1d",
        "conv2d",
        "conv3d",
        "layer_norm",
        "batch_norm",
        "group_norm",
    }
)

_TENSOR_OPS = frozenset(
    {
        "torch.Tensor",
        "Tensor",
    }
)


def _norm_name(value: str) -> str:
    return value.rsplit(".", 1)[-1]


def classify_module(op_or_class: str) -> str:
    """Map a module / operator name to a NetPrism node ``kind``."""
    name = _norm_name(str(op_or_class))
    if not name:
        return "unknown"

    if name in ("Input", "Output"):
        return name.lower()
    if name in _CONTAINER_NAMES:
        return "container"

    # --- weighted layers -------------------------------------------------
    if name.startswith("ConvTranspose") or name.startswith("ConvT"):
        return "deconv"
    if name in ("Conv1d", "Conv2d", "Conv3d") or (name.startswith("Conv") and name[-1:].isdigit()):
        return "conv"
    if name in ("Linear", "Bilinear", "Dense") or name.startswith("Linear"):
        return "linear"
    if name in _EMBEDDING_NAMES or name.startswith("Embedding"):
        return "positional" if "position" in name.lower() or "pos_emb" in name.lower() else "embedding"

    # --- attention / transformer ----------------------------------------
    if name in _TRANSFORMER_FUSED or name.startswith("Transformer"):
        return "block"
    if "Attention" in name or name.endswith("Attn") or name == "Attn" or name.startswith("MHA"):
        return "attention"
    if name in ("MultiheadAttention",):
        return "attention"
    if "FFN" in name or "FeedForward" in name or "Feedforward" in name:
        return "ffn"
    if name in ("MLP", "Mlp") or name.endswith("MLP"):
        return "mlp"

    # --- normalization / regularisation ---------------------------------
    if name in _NORM_SUFFIXES or name.endswith("Norm") or "Normalization" in name:
        return "norm"
    if name.startswith("Dropout") or name in ("DropPath", "StochasticDepth"):
        return "dropout"

    # --- activations ------------------------------------------------------
    if name in _ACTIVATIONS or name.startswith(_ACTIVATIONS):
        return "activation"
    if name == "Softmax" or name == "LogSoftmax":
        return "softmax"
    if name == "Identity":
        return "custom"

    # --- pooling / reshaping ---------------------------------------------
    if name in _POOL_NAMES or name.startswith(_POOL_NAMES):
        return "pool"
    if name == "Flatten":
        return "flatten"
    if name in ("Unflatten",):
        return "reshape"
    if name.startswith("Upsample") or name.startswith("PixelShuffle") or name.startswith("PixelUnshuffle"):
        return "upsample"
    if name in _RNN_NAMES or name.endswith("LSTM") or name.endswith("GRU"):
        return "rnn"

    # --- generic custom names --------------------------------------------
    lowered = name.lower()
    if "encoder" in lowered or "decoder" in lowered:
        return "block"
    if "block" in lowered or "bottleneck" in lowered or "resblock" in lowered:
        return "block"
    if name.startswith("Conv") or "Conv" in name:
        return "conv"
    if name.startswith("Linear") or "Linear" in name:
        return "linear"
    return "custom"


def classify_function(op_name: str) -> str | None:
    """Map a ``call_function`` / ``call_method`` op name to a kind.

    Returns ``None`` when the operation should not become a graph node.
    """
    name = str(op_name)
    short = name.rsplit(".", 1)[-1]
    lowered = short.lower()

    if short in ("getattr", "size", "dim", "numel", "item", "tolist"):
        return None
    if short in ("arange", "zeros", "ones", "full", "empty", "randn", "rand", "triu", "tril", "eye", "tensor"):
        return None
    if lowered in ("contiguous", "clone", "detach", "type", "type_as", "to", "float", "long", "int", "bool"):
        return None

    if lowered in ("add", "add_", "__add__", "sub", "sub_", "__sub__"):
        return "add"
    if lowered in ("cat", "concat", "__cat__", "stack"):
        return "concat"
    if lowered in ("softmax", "log_softmax"):
        return "softmax"
    if lowered in ("matmul", "bmm", "einsum", "scaled_dot_product_attention", "masked_fill", "masked_fill_"):
        return "attention"
    if lowered in ("flatten", "unflatten"):
        return "flatten"
    if lowered in ("view", "reshape", "squeeze", "unsqueeze", "permute", "transpose", "expand", "repeat", "chunk", "split", "unfold"):
        return "reshape"
    if lowered in ("dropout", "dropout_"):
        return "dropout"
    if lowered in ("relu", "relu6", "gelu", "silu", "swish", "leaky_relu", "prelu", "elu", "selu", "celu", "mish", "hardswish", "hardsigmoid", "hardtanh", "sigmoid", "tanh", "glu", "softplus"):
        return "activation"
    if lowered in ("mean", "sum", "amax", "amin", "max", "min", "prod"):
        return "pool"
    if "pool" in lowered:
        return "pool"
    if lowered in ("interpolate", "upsample", "upsample_nearest2d", "upsample_bilinear2d"):
        return "upsample"
    if lowered in ("embedding", "one_hot"):
        return "embedding"
    if lowered in ("linear", "conv1d", "conv2d", "conv3d", "layer_norm", "batch_norm", "group_norm"):
        return classify_module(lowered.upper() if lowered.islower() else lowered)
    if lowered in ("mul", "mul_", "div", "truediv", "pow", "neg", "__mul__", "__truediv__"):
        return "custom"
    if lowered in ("split", "chunk"):
        return "reshape"
    return None


def function_op_name(target: Any) -> str:
    """Human readable op name for a ``call_function`` target."""
    if target is operator.add:
        return "add"
    if target is operator.sub:
        return "sub"
    if target is operator.mul:
        return "mul"
    if target is operator.truediv:
        return "div"
    if target is operator.matmul:
        return "matmul"
    name = getattr(target, "__name__", None)
    if isinstance(name, str):
        module = getattr(target, "__module__", "") or ""
        if module.startswith("torch.nn.functional"):
            return f"F.{name}"
        if module == "operator":
            return name
        if module.startswith("torch"):
            return f"torch.{name}"
        return name
    return str(target)


# ---------------------------------------------------------------------------
# tags
# ---------------------------------------------------------------------------

_TAG_BY_KIND: dict[str, list[str]] = {
    "conv": ["spatial", "weighted"],
    "deconv": ["spatial", "weighted", "upsample"],
    "linear": ["weighted"],
    "mlp": ["weighted", "sequence"],
    "norm": ["normalized"],
    "activation": ["elementwise"],
    "pool": ["downsample"],
    "dropout": ["regularization"],
    "softmax": ["normalized"],
    "attention": ["attention", "weighted"],
    "ffn": ["weighted", "sequence"],
    "embedding": ["token", "weighted"],
    "positional": ["token", "positional"],
    "rnn": ["sequence", "weighted"],
    "upsample": ["upsample"],
    "add": ["merge"],
    "concat": ["merge"],
    "flatten": ["reshape"],
    "reshape": ["reshape"],
    "block": ["composite"],
    "container": ["container"],
    "custom": ["custom"],
    "unknown": ["unknown"],
}


def tags_for(kind: str, op: str | None = None, params: dict[str, Any] | None = None) -> list[str]:
    tags = list(_TAG_BY_KIND.get(kind, []))
    lowered = (op or "").lower()
    if params:
        if params.get("groups", 1) and int(params.get("groups") or 1) > 1:
            if "grouped" not in tags:
                tags.append("grouped")
        if kind == "pool" and "adaptive" in lowered:
            if "global" not in tags:
                tags.append("global")
        if kind == "embedding" and "pos" in lowered:
            tags.append("positional")
        if kind == "linear" and "head" in lowered:
            tags.append("head")
        if kind == "conv" and int(params.get("outChannels") or 0) > int(params.get("inChannels") or 0):
            tags.append("upsample")
    if "residual" in lowered or "skip" in lowered:
        if "residual" not in tags:
            tags.append("residual")
    if "head" in lowered and "head" not in tags:
        tags.append("head")
    return tags


# ---------------------------------------------------------------------------
# params from live modules (duck typed, no torch import)
# ---------------------------------------------------------------------------

def _int(value: Any, default: int | None = None) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _int_list(value: Any, dims: int = 2) -> list[int]:
    if value is None:
        return [1] * dims
    if isinstance(value, bool):
        return [int(value)] * dims
    if isinstance(value, int):
        return [value] * dims
    if isinstance(value, (tuple, list)):
        values = [int(v) for v in value]
        if len(values) == 1 and dims > 1:
            return values * dims
        return values
    return [int(value)] * dims


def _op_dims(class_name: str) -> int:
    lowered = class_name.lower()
    for dims, suffix in ((1, "1d"), (2, "2d"), (3, "3d")):
        if lowered.endswith(suffix):
            return dims
    return 2


def _getattr(module: Any, name: str, default: Any = None) -> Any:
    try:
        return getattr(module, name, default)
    except Exception:  # pragma: no cover - exotic custom modules
        return default


def extract_module_params(module: Any, kind: str | None = None) -> dict[str, Any]:
    """Extract JSON-safe hyper-parameters from a live module (duck typed)."""
    cls_name = type(module).__name__
    kind = kind or classify_module(cls_name)
    params: dict[str, Any] = {}

    if kind in ("conv", "deconv"):
        dims = _op_dims(cls_name)
        params["inChannels"] = _int(_getattr(module, "in_channels"), 0)
        params["outChannels"] = _int(_getattr(module, "out_channels"), 0)
        params["kernelSize"] = _int_list(_getattr(module, "kernel_size"), dims)
        params["stride"] = _int_list(_getattr(module, "stride"), dims)
        params["padding"] = _int_list(_getattr(module, "padding"), dims)
        params["dilation"] = _int_list(_getattr(module, "dilation"), dims)
        params["groups"] = _int(_getattr(module, "groups"), 1) or 1
        bias = _getattr(module, "bias")
        params["bias"] = bias is not None
        if kind == "deconv":
            params["outputPadding"] = _int_list(_getattr(module, "output_padding"), dims)
    elif kind == "linear":
        params["inFeatures"] = _int(_getattr(module, "in_features"), 0)
        params["outFeatures"] = _int(_getattr(module, "out_features"), 0)
        params["bias"] = _getattr(module, "bias") is not None
    elif kind == "norm":
        normalized = _getattr(module, "normalized_shape")
        if normalized is not None:
            try:
                params["normalizedShape"] = [int(v) for v in normalized]
            except TypeError:
                params["normalizedShape"] = [int(normalized)]
        else:
            num_features = _int(_getattr(module, "num_features"))
            if num_features is not None:
                params["normalizedShape"] = [num_features]
        num_groups = _int(_getattr(module, "num_groups"))
        if num_groups is not None:
            params["numGroups"] = num_groups
        eps = _getattr(module, "eps")
        if eps is not None:
            params["eps"] = float(eps)
        params["affine"] = _getattr(module, "weight") is not None
    elif kind == "pool":
        params["type"] = "adaptive" if "adaptive" in cls_name.lower() else ("max" if "max" in cls_name.lower() else "avg")
        output_size = _getattr(module, "output_size")
        if output_size is not None:
            try:
                params["outputSize"] = [int(v) for v in output_size] if isinstance(output_size, (tuple, list)) else [int(output_size)]
            except (TypeError, ValueError):
                pass
        else:
            dims = _op_dims(cls_name)
            params["kernelSize"] = _int_list(_getattr(module, "kernel_size"), dims)
            params["stride"] = _int_list(_getattr(module, "stride"), dims)
            params["padding"] = _int_list(_getattr(module, "padding"), dims)
    elif kind == "dropout":
        p = _getattr(module, "p")
        params["p"] = float(p) if p is not None else 0.5
    elif kind == "embedding":
        params["numEmbeddings"] = _int(_getattr(module, "num_embeddings"), 0)
        params["embeddingDim"] = _int(_getattr(module, "embedding_dim"), 0)
        padding = _int(_getattr(module, "padding_idx"), None)
        if padding is not None:
            params["paddingIdx"] = padding
    elif kind in ("attention",):
        embed_dim = _int(_getattr(module, "embed_dim"))
        num_heads = _int(_getattr(module, "num_heads"))
        if embed_dim is not None:
            params["embedDim"] = embed_dim
        if num_heads is not None:
            params["numHeads"] = num_heads
            if embed_dim:
                params["headDim"] = embed_dim // num_heads
        for key, out_key in (("kdim", "kdim"), ("vdim", "vdim")):
            value = _int(_getattr(module, key))
            if value is not None:
                params[out_key] = value
        batch_first = _getattr(module, "batch_first")
        if batch_first is not None:
            params["batchFirst"] = bool(batch_first)
        dropout = _getattr(module, "dropout")
        if dropout is not None:
            params["dropout"] = float(dropout)
    elif kind == "rnn":
        for key, out_key in (
            ("input_size", "inputFeatures"),
            ("hidden_size", "hiddenFeatures"),
            ("num_layers", "numLayers"),
        ):
            value = _int(_getattr(module, key))
            if value is not None:
                params[out_key] = value
        for key, out_key in (("bidirectional", "bidirectional"), ("batch_first", "batchFirst"), ("dropout", "dropout")):
            value = _getattr(module, key)
            if value is not None:
                params[out_key] = value if not isinstance(value, float) else float(value)
        params["type"] = cls_name
    elif kind in ("flatten", "reshape"):
        params["startDim"] = _int(_getattr(module, "start_dim"), 1)
        params["endDim"] = _int(_getattr(module, "end_dim"), -1)
    elif kind == "upsample":
        scale = _getattr(module, "scale_factor")
        if scale is not None:
            params["scaleFactor"] = _int_list(scale, 2) if isinstance(scale, (tuple, list)) else float(scale)
        size = _getattr(module, "size")
        if size is not None and not params:
            params["size"] = [int(v) for v in size] if isinstance(size, (tuple, list)) else _int(size)
        mode = _getattr(module, "mode")
        if mode is not None:
            params["mode"] = str(mode)
    elif kind in ("activation", "softmax"):
        inplace = _getattr(module, "inplace")
        if inplace is not None:
            params["inplace"] = bool(inplace)
        dim = _int(_getattr(module, "dim"))
        if dim is not None:
            params["dim"] = dim
    elif kind == "block":
        # Fused composite layers keep their identifying hyper-parameters.
        for key, out_key in (
            ("d_model", "embedDim"),
            ("nhead", "numHeads"),
            ("dim_feedforward", "hiddenFeatures"),
            ("num_layers", "numLayers"),
            ("norm_first", "normFirst"),
            ("batch_first", "batchFirst"),
            ("dropout", "dropout"),
        ):
            value = _getattr(module, key)
            if value is not None:
                params[out_key] = value if not isinstance(value, float) else float(value)
    return params


# ---------------------------------------------------------------------------
# shape helpers
# ---------------------------------------------------------------------------

def normalize_shape(shape: Any) -> Shape | None:
    """Return a JSON-safe shape (list[int]) or ``None`` when not a tensor shape."""
    if shape is None:
        return None
    try:
        values = list(shape)
    except TypeError:
        return None
    out: Shape = []
    for dim in values:
        if isinstance(dim, bool):
            return None
        if isinstance(dim, int):
            out.append(dim)
        else:
            try:
                out.append(int(dim))
            except (TypeError, ValueError):
                out.append(-1)
    return out or None


def shape_numel(shape: Sequence[int] | None) -> int | None:
    """Number of elements, or ``None`` when a dimension is dynamic."""
    if not shape:
        return None
    total = 1
    for dim in shape:
        if dim < 0:
            return None
        total *= int(dim)
    return total


def shape_of_value(value: Any, *, _depth: int = 0) -> list[Shape]:
    """Collect the shapes of every tensor found in ``value`` (recursively)."""
    import torch  # lazy: only used by the dynamic tracer

    if _depth > 4:
        return []
    if isinstance(value, torch.Tensor):
        shape = normalize_shape(tuple(value.shape))
        return [shape] if shape else []
    if isinstance(value, (list, tuple)):
        shapes: list[Shape] = []
        for item in value:
            shapes.extend(shape_of_value(item, _depth=_depth + 1))
        return shapes
    if isinstance(value, dict):
        shapes = []
        for item in value.values():
            shapes.extend(shape_of_value(item, _depth=_depth + 1))
        return shapes
    return []


def first_shape(value: Any) -> Shape | None:
    shapes = shape_of_value(value)
    return shapes[0] if shapes else None


def _tuple_dim(value: Any, dims: int) -> list[int]:
    return _int_list(value, dims)


def conv_out_size(size: int, kernel: int, stride: int, padding: int, dilation: int) -> int:
    if size < 0:
        return -1
    numerator = size + 2 * padding - dilation * (kernel - 1) - 1
    return numerator // stride + 1


def conv_transpose_out_size(
    size: int, kernel: int, stride: int, padding: int, dilation: int, output_padding: int
) -> int:
    if size < 0:
        return -1
    return (size - 1) * stride - 2 * padding + dilation * (kernel - 1) + output_padding + 1


def _pool_out_size(size: int, kernel: int, stride: int, padding: int, dilation: int = 1, ceil: bool = False) -> int:
    if size < 0:
        return -1
    numerator = size + 2 * padding - dilation * (kernel - 1) - 1
    if ceil:
        return int(math.ceil(numerator / stride)) + 1
    return numerator // stride + 1


def infer_conv_shape(in_shape: Shape, params: dict[str, Any], *, transpose: bool = False) -> Shape:
    in_shape = list(in_shape)
    spatial = in_shape[2:]
    dims = max(1, len(spatial))
    kernels = _tuple_dim(params.get("kernelSize", [1] * dims), dims)
    strides = _tuple_dim(params.get("stride", [1] * dims), dims)
    paddings = _tuple_dim(params.get("padding", [0] * dims), dims)
    dilations = _tuple_dim(params.get("dilation", [1] * dims), dims)
    out_spatial: list[int] = []
    for i, size in enumerate(spatial):
        k, s, p, d = kernels[i], strides[i], paddings[i], dilations[i]
        if transpose:
            out_padding = _tuple_dim(params.get("outputPadding", [0] * dims), dims)[i]
            out_spatial.append(conv_transpose_out_size(size, k, s, p, d, out_padding))
        else:
            out_spatial.append(conv_out_size(size, k, s, p, d))
    out_channels = int(params.get("outChannels") or 0)
    batch = in_shape[0] if in_shape else 1
    if not out_channels:
        out_channels = int(params.get("inChannels") or 0)
    return [batch, out_channels, *out_spatial]


def infer_pool_shape(in_shape: Shape, params: dict[str, Any], op_name: str = "") -> Shape:
    in_shape = list(in_shape)
    spatial = in_shape[2:]
    dims = max(1, len(spatial))
    out_spatial: list[int] = []
    lowered = (op_name or "").lower()
    output_size = params.get("outputSize")
    if output_size is not None:
        sizes = _tuple_dim(output_size, dims)
        out_spatial = [max(1, sizes[i]) for i in range(len(spatial))]
    else:
        kernels = _tuple_dim(params.get("kernelSize", [1] * dims), dims)
        strides = _tuple_dim(params.get("stride", kernels), dims)
        paddings = _tuple_dim(params.get("padding", [0] * dims), dims)
        ceil = bool(params.get("ceilMode")) or ("ceil" in lowered)
        def _pick(values: list[int], index: int) -> int:
            if not values:
                return 1
            return values[index] if index < len(values) else values[-1]
        for i, size in enumerate(spatial):
            out_spatial.append(
                _pool_out_size(size, _pick(kernels, i), _pick(strides, i), _pick(paddings, i), 1, ceil)
            )
    return [in_shape[0], in_shape[1], *out_spatial]


def infer_output_shapes(
    kind: str, params: dict[str, Any], in_shapes: Sequence[Shape], op_name: str = ""
) -> tuple[list[Shape], list[str]]:
    """Best-effort static shape inference.  Returns ``(shapes, warnings)``."""
    warnings: list[str] = []
    if not in_shapes:
        return [], ["no input shape available"]
    first = list(in_shapes[0])
    dims = len(first)
    lowered = (op_name or "").lower()

    if kind in ("conv", "deconv"):
        return [infer_conv_shape(first, params, transpose=(kind == "deconv"))], warnings
    if kind == "linear":
        out_features = int(params.get("outFeatures") or 0)
        in_features = int(params.get("inFeatures") or 0)
        if out_features <= 0:
            return [[-1]], ["linear: unknown outFeatures (constructor argument unresolved)"]
        if in_features <= 0:
            return [[-1, out_features]], ["linear: unknown inFeatures (constructor argument unresolved)"]
        return [[*first[:-1], out_features]], warnings
    if kind == "norm":
        return [first], warnings
    if kind in ("activation", "dropout", "softmax"):
        return [first], warnings
    if kind == "pool":
        if dims >= 3:
            return [infer_pool_shape(first, params, op_name)], warnings
        # Global reductions over a 1-D/2-D tensor keep the batch dimension.
        dim = _int(params.get("dim"))
        if dim is not None and dim != 0 and dim < dims:
            reduced = first[:dim] + first[dim + 1 :]
            return [reduced or [1]], warnings
        return [first], warnings
    if kind == "flatten":
        start = _int(params.get("startDim"), 1) or 1
        if start < 0:
            start += dims
        start = max(0, min(start, dims))
        prefix = first[:start]
        flat = 1
        for value in first[start:]:
            if value < 0:
                flat = -1
                break
            flat *= value
        return [[*prefix, flat]], warnings
    if kind == "reshape":
        target = params.get("shape")
        if isinstance(target, (list, tuple)) and target:
            target_shape: Shape = []
            for value in target:
                if isinstance(value, int):
                    target_shape.append(value)
                else:
                    return [first], [f"reshape: unsupported target dim {value!r}"]
            return [target_shape], warnings
        if "permute" in lowered or "transpose" in lowered:
            order = params.get("dims")
            if isinstance(order, (list, tuple)) and all(isinstance(v, int) and -dims <= v < dims for v in order):
                resolved = [v + dims if v < 0 else v for v in order]
                if sorted(resolved) == list(range(dims)) and dims > 0:
                    return [[first[i] for i in resolved]], warnings
        return [first], warnings
    if kind == "embedding":
        embedding_dim = int(params.get("embeddingDim") or 0)
        if not embedding_dim:
            return [[-1]], ["embedding: unknown embedding_dim"]
        return [[*first, embedding_dim]], warnings
    if kind == "upsample":
        spatial = first[2:]
        scale = params.get("scaleFactor")
        if isinstance(scale, (int, float)):
            factor = float(scale)
            return [[first[0], first[1], *[int(size * factor) if size >= 0 else -1 for size in spatial]]], warnings
        if isinstance(scale, (tuple, list)):
            factors = _tuple_dim(scale, len(spatial))
            return [[first[0], first[1], *[int(size * f) if size >= 0 else -1 for size, f in zip(spatial, factors)]]], warnings
        size = params.get("size")
        if isinstance(size, (list, tuple)):
            return [[first[0], first[1], *[int(v) for v in size]]], warnings
        return [first], warnings
    if kind == "rnn":
        hidden = int(params.get("hiddenFeatures") or 0)
        directions = 2 if params.get("bidirectional") else 1
        if params.get("batchFirst") and dims == 3:
            return [[first[0], first[1], hidden * directions]], warnings
        if dims == 3:
            return [[first[0], first[1], hidden * directions]], warnings
        return [[hidden * directions]], warnings
    if kind == "attention":
        num_heads = int(params.get("numHeads") or 1)
        embed = int(params.get("embedDim") or (first[-1] if first else 0))
        if len(in_shapes) >= 2 and in_shapes[1] != in_shapes[0]:
            return [[*first[:-1], embed]], warnings
        return [first], warnings
    if kind == "block":
        # Fused Transformer layer / encoder: shape preserving.
        return [first], warnings
    if kind == "concat":
        return [first], warnings
    if kind in ("add", "custom", "multiply"):
        return [first], warnings
    return [first], [f"{kind}: shape preserved as best effort"]


# ---------------------------------------------------------------------------
# FLOPs
# ---------------------------------------------------------------------------

def _numel_or_zero(shape: Sequence[int] | None) -> int:
    value = shape_numel(shape)
    return value or 0


def conv_flops(params: dict[str, Any], out_shape: Shape, *, transpose: bool = False) -> int:
    if len(out_shape) < 3:
        return 0
    batch = out_shape[0]
    spatial_out = out_shape[2:]
    spatial_elems = shape_numel(spatial_out) or 0
    cout = int(params.get("outChannels") or 0)
    cin = int(params.get("inChannels") or 0)
    groups = max(1, int(params.get("groups") or 1))
    kernel = shape_numel(_tuple_dim(params.get("kernelSize", [1, 1]), max(1, len(spatial_out)))) or 1
    if batch <= 0 or cout <= 0 or cin <= 0 or spatial_elems <= 0:
        return 0
    macs = batch * cout * spatial_elems * (cin // groups) * kernel
    flops = 2 * macs
    if params.get("bias"):
        flops += batch * cout * spatial_elems
    return int(flops)


def linear_flops(params: dict[str, Any], out_shape: Shape) -> int:
    out_features = int(params.get("outFeatures") or (out_shape[-1] if out_shape else 0))
    in_features = int(params.get("inFeatures") or 0)
    if out_features <= 0 or in_features <= 0:
        return 0
    batch = shape_numel(out_shape[:-1]) or 1
    if batch <= 0:
        return 0
    flops = 2 * batch * in_features * out_features
    if params.get("bias"):
        flops += batch * out_features
    return int(flops)


def attention_flops(params: dict[str, Any], in_shapes: Sequence[Shape]) -> int:
    embed = int(params.get("embedDim") or 0)
    if not embed or not in_shapes:
        return 0
    query = list(in_shapes[0])
    key = list(in_shapes[1]) if len(in_shapes) > 1 else query
    batch_first = bool(params.get("batchFirst", False))
    if len(query) == 3:
        if batch_first:
            batch, seq_q = query[0], query[1]
            seq_k = key[1] if len(key) == 3 else seq_q
        else:
            seq_q, batch = query[0], query[1]
            seq_k = key[0] if len(key) == 3 else seq_q
    elif len(query) == 2:
        seq_q, batch = query[0], 1
        seq_k = key[0] if len(key) == 2 else seq_q
    else:
        return 0
    if any(value < 0 for value in (batch, seq_q, seq_k, embed)):
        return 0
    projections = 4 * batch * seq_q * embed * embed
    scores = 2 * batch * seq_q * seq_k * embed
    return int(projections + scores)


def matmul_flops(in_shapes: Sequence[Shape]) -> int:
    if len(in_shapes) < 2:
        return 0
    left, right = list(in_shapes[0]), list(in_shapes[1])
    if len(left) < 2 or len(right) < 2:
        return 0
    m, k = left[-2], left[-1]
    k2, n = right[-2], right[-1]
    if k != k2:
        return 0
    batch = shape_numel(left[:-2]) or 1
    if any(value < 0 for value in (m, k, n, batch)):
        return 0
    return int(2 * batch * m * n * k)


def reduce_flops(in_shape: Shape, out_shape: Shape, per_element: int = 4) -> int:
    return int(_numel_or_zero(in_shape) * per_element)


def module_flops(
    kind: str,
    params: dict[str, Any],
    in_shapes: Sequence[Shape],
    out_shapes: Sequence[Shape],
    op_name: str = "",
) -> int:
    """FLOPs for a leaf module / op following SPEC 4.5."""
    out_shape = list(out_shapes[0]) if out_shapes else []
    in_shape = list(in_shapes[0]) if in_shapes else []
    if kind == "conv":
        return conv_flops(params, out_shape)
    if kind == "deconv":
        return conv_flops(params, out_shape, transpose=True)
    if kind == "linear":
        return linear_flops(params, out_shape)
    if kind == "attention":
        return attention_flops(params, in_shapes)
    if kind == "norm":
        return int(_numel_or_zero(out_shape) * 4)
    if kind in ("activation", "softmax"):
        return int(_numel_or_zero(out_shape) * (1 if kind == "activation" else 4))
    if kind == "pool":
        return int(_numel_or_zero(in_shape) * 1)
    if kind == "add":
        return int(_numel_or_zero(out_shape) * 1)
    if kind == "concat":
        return int(_numel_or_zero(out_shape) * 1)
    if kind == "upsample":
        return int(_numel_or_zero(out_shape) * 1)
    if kind == "matmul":
        return matmul_flops(in_shapes)
    if kind == "rnn":
        return _rnn_flops(params, in_shape)
    if kind in ("flatten", "reshape", "dropout", "embedding", "positional", "container", "input", "output"):
        return 0
    if kind == "ffn":
        return 0
    if kind == "block":
        return 0
    if kind == "custom":
        return 0
    # Unknown ops: cheap element-wise estimate.
    return int(_numel_or_zero(out_shape) * 1)


def _rnn_flops(params: dict[str, Any], in_shape: Shape) -> int:
    hidden = int(params.get("hiddenFeatures") or 0)
    input_features = int(params.get("inputFeatures") or (in_shape[-1] if in_shape else 0))
    layers = max(1, int(params.get("numLayers") or 1))
    directions = 2 if params.get("bidirectional") else 1
    if not hidden or not input_features:
        return 0
    if len(in_shape) == 3 and bool(params.get("batchFirst")):
        batch, seq = in_shape[0], in_shape[1]
    elif len(in_shape) == 3:
        seq, batch = in_shape[0], in_shape[1]
    else:
        return 0
    if any(value < 0 for value in (batch, seq)):
        return 0
    # 4 gates for LSTM/GRU (approximation for vanilla RNN too).
    per_layer = batch * seq * 4 * directions * (input_features * hidden + hidden * hidden) * 2
    return int(per_layer * layers)


def flops_is_approx(kind: str) -> bool:
    """Whether the FLOPs formula for ``kind`` is an estimate (SPEC 4.5)."""
    return kind in ("attention", "rnn", "custom", "unknown", "block", "matmul")


__all__ = [
    "Shape",
    "SKIP_FUNCTIONS",
    "EFFECTIVE_FUNCTIONS",
    "classify_module",
    "classify_function",
    "function_op_name",
    "tags_for",
    "extract_module_params",
    "normalize_shape",
    "shape_numel",
    "shape_of_value",
    "first_shape",
    "infer_output_shapes",
    "conv_out_size",
    "conv_transpose_out_size",
    "module_flops",
    "attention_flops",
    "matmul_flops",
    "flops_is_approx",
]
