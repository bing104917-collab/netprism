"""TorchScript archive adapter (``.pt`` / ``.pth``).

``torch.jit.save`` archives keep the module hierarchy (nested ``CallMethod``
graphs) and the *input* tensor shapes, but PyTorch erases the per-operator shape
types from the serialized graph.  The adapter therefore:

1. recursively walks the module hierarchy (``prim::CallMethod``) to recover the
   exact module paths (``features.0``, ``layer1.0.conv1``, ...) and the operator
   execution order;
2. aligns that call plan with the flattened ``inlined_graph`` aten ops;
3. propagates tensor shapes from the recorded input shape(s) using the operator
   semantics plus the recorded module / initializer shapes and op constants;
4. derives parameters / FLOPs from the recovered module weights.

When anything cannot be recovered the node keeps ``-1`` shapes and the model
carries a warning, as required by ``SPEC-V2.md`` section 3.

``torch`` is imported lazily: importing this module must not require PyTorch.
"""

from __future__ import annotations

import logging
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..core import ops
from ..core.errors import AnalyzeError
from ._common import (
    assemble_graph,
    make_boundary_node,
    make_edge,
    make_node,
    shape_numel,
    to_shape,
)

LOGGER = logging.getLogger("netprism.adapters.torchscript")

_MISSING = object()

#: aten ops that carry visual / analytical meaning and become graph nodes.
_SIGNIFICANT_ATEN: frozenset[str] = frozenset(
    {
        "_convolution",
        "conv1d",
        "conv2d",
        "conv3d",
        "conv_transpose1d",
        "conv_transpose2d",
        "conv_transpose3d",
        "linear",
        "matmul",
        "bmm",
        "einsum",
        "addmm",
        "batch_norm",
        "native_batch_norm",
        "_native_batch_norm_legit",
        "_native_batch_norm_legit_no_training",
        "layer_norm",
        "group_norm",
        "instance_norm",
        "relu",
        "relu_",
        "gelu",
        "leaky_relu",
        "elu",
        "selu",
        "celu",
        "silu",
        "swish",
        "sigmoid",
        "tanh",
        "hardswish",
        "hardsigmoid",
        "hardtanh",
        "softplus",
        "mish",
        "prelu",
        "softmax",
        "_softmax",
        "log_softmax",
        "dropout",
        "dropout_",
        "feature_dropout",
        "max_pool1d",
        "max_pool2d",
        "max_pool3d",
        "avg_pool1d",
        "avg_pool2d",
        "avg_pool3d",
        "adaptive_avg_pool1d",
        "adaptive_avg_pool2d",
        "adaptive_avg_pool3d",
        "adaptive_max_pool1d",
        "adaptive_max_pool2d",
        "adaptive_max_pool3d",
        "flatten",
        "view",
        "reshape",
        "permute",
        "transpose",
        "squeeze",
        "squeeze_",
        "unsqueeze",
        "unsqueeze_",
        "expand",
        "cat",
        "concat",
        "add",
        "add_",
        "sub",
        "sub_",
        "mul",
        "mul_",
        "div",
        "truediv",
        "mean",
        "sum",
        "embedding",
        "scaled_dot_product_attention",
        "_transformer_encoder_layer_fwd",
        "_transformer_decoder_layer_fwd",
        "_transformer_decoder_only_layer_fwd",
        "transformer_encoder_layer",
        "upsample_nearest2d",
        "upsample_bilinear2d",
        "interpolate",
        "pad",
    }
)

#: Shape preserving aten ops that are *not* graph nodes.
_PASSTHROUGH_ATEN: frozenset[str] = frozenset(
    {
        "to",
        "type_as",
        "_to_copy",
        "contiguous",
        "clone",
        "detach",
        "alias",
        "_unsafe_view",
        "copy_",
        "empty_like",
        "zeros_like",
        "ones_like",
        "requires_grad_",
        "detach_",
    }
)

_KIND_BY_ATEN: dict[str, str] = {
    "_convolution": "conv",
    "conv1d": "conv",
    "conv2d": "conv",
    "conv3d": "conv",
    "conv_transpose1d": "deconv",
    "conv_transpose2d": "deconv",
    "conv_transpose3d": "deconv",
    "linear": "linear",
    "addmm": "linear",
    "matmul": "attention",
    "bmm": "attention",
    "einsum": "attention",
    "batch_norm": "norm",
    "native_batch_norm": "norm",
    "_native_batch_norm_legit": "norm",
    "_native_batch_norm_legit_no_training": "norm",
    "layer_norm": "norm",
    "group_norm": "norm",
    "instance_norm": "norm",
    "relu": "activation",
    "relu_": "activation",
    "gelu": "activation",
    "leaky_relu": "activation",
    "elu": "activation",
    "selu": "activation",
    "celu": "activation",
    "silu": "activation",
    "swish": "activation",
    "sigmoid": "activation",
    "tanh": "activation",
    "hardswish": "activation",
    "hardsigmoid": "activation",
    "hardtanh": "activation",
    "softplus": "activation",
    "mish": "activation",
    "prelu": "activation",
    "softmax": "softmax",
    "_softmax": "softmax",
    "log_softmax": "softmax",
    "dropout": "dropout",
    "dropout_": "dropout",
    "feature_dropout": "dropout",
    "max_pool1d": "pool",
    "max_pool2d": "pool",
    "max_pool3d": "pool",
    "avg_pool1d": "pool",
    "avg_pool2d": "pool",
    "avg_pool3d": "pool",
    "adaptive_avg_pool1d": "pool",
    "adaptive_avg_pool2d": "pool",
    "adaptive_avg_pool3d": "pool",
    "adaptive_max_pool1d": "pool",
    "adaptive_max_pool2d": "pool",
    "adaptive_max_pool3d": "pool",
    "flatten": "flatten",
    "view": "reshape",
    "reshape": "reshape",
    "permute": "reshape",
    "transpose": "reshape",
    "squeeze": "reshape",
    "squeeze_": "reshape",
    "unsqueeze": "reshape",
    "unsqueeze_": "reshape",
    "expand": "reshape",
    "cat": "concat",
    "concat": "concat",
    "add": "add",
    "add_": "add",
    "sub": "custom",
    "sub_": "custom",
    "mul": "custom",
    "mul_": "custom",
    "div": "custom",
    "truediv": "custom",
    "mean": "pool",
    "sum": "pool",
    "embedding": "embedding",
    "scaled_dot_product_attention": "attention",
    "_transformer_encoder_layer_fwd": "block",
    "_transformer_decoder_layer_fwd": "block",
    "_transformer_decoder_only_layer_fwd": "block",
    "transformer_encoder_layer": "block",
    "upsample_nearest2d": "upsample",
    "upsample_bilinear2d": "upsample",
    "interpolate": "upsample",
    "pad": "custom",
}

_FRIENDLY_ATEN: dict[str, str] = {
    "_convolution": "Conv2d",
    "conv1d": "Conv1d",
    "conv2d": "Conv2d",
    "conv3d": "Conv3d",
    "conv_transpose2d": "ConvTranspose2d",
    "linear": "Linear",
    "batch_norm": "BatchNorm",
    "layer_norm": "LayerNorm",
    "group_norm": "GroupNorm",
    "instance_norm": "InstanceNorm",
    "relu": "ReLU",
    "relu_": "ReLU",
    "gelu": "GELU",
    "sigmoid": "Sigmoid",
    "tanh": "Tanh",
    "softmax": "Softmax",
    "_softmax": "Softmax",
    "log_softmax": "LogSoftmax",
    "dropout": "Dropout",
    "dropout_": "Dropout",
    "max_pool2d": "MaxPool2d",
    "avg_pool2d": "AvgPool2d",
    "adaptive_avg_pool2d": "AdaptiveAvgPool2d",
    "adaptive_max_pool2d": "AdaptiveMaxPool2d",
    "flatten": "torch.flatten",
    "view": "view",
    "reshape": "reshape",
    "permute": "permute",
    "transpose": "transpose",
    "unsqueeze": "unsqueeze",
    "squeeze": "squeeze",
    "cat": "cat",
    "add": "add",
    "add_": "add",
    "mean": "mean",
    "sum": "sum",
    "embedding": "Embedding",
    "matmul": "matmul",
    "bmm": "bmm",
    "scaled_dot_product_attention": "scaled_dot_product_attention",
    "_transformer_encoder_layer_fwd": "TransformerEncoderLayer",
    "_transformer_decoder_layer_fwd": "TransformerDecoderLayer",
    "_transformer_decoder_only_layer_fwd": "TransformerDecoderLayer",
    "transformer_encoder_layer": "TransformerEncoderLayer",
}

_TORCH_DTYPE_NAMES = {
    "Float": "float32",
    "Double": "float64",
    "Half": "float16",
    "BFloat16": "bfloat16",
    "Long": "int64",
    "Int": "int32",
    "Short": "int16",
    "Char": "int8",
    "Byte": "uint8",
    "Bool": "bool",
    "ComplexFloat": "complex64",
    "ComplexDouble": "complex128",
}

_TORCH_DTYPE_OBJ = {
    "Float": "float32",
    "Double": "float64",
    "Half": "float16",
    "BFloat16": "bfloat16",
    "Long": "int64",
    "Int": "int32",
    "Short": "int16",
    "Char": "int8",
    "Byte": "uint8",
    "Bool": "bool",
}


# ---------------------------------------------------------------------------
# small graph helpers
# ---------------------------------------------------------------------------

def _inputs(node: Any) -> list[Any]:
    try:
        return list(node.inputs())
    except TypeError:  # pragma: no cover - older torch returns a list already
        return node.inputs()


def _outputs(node: Any) -> list[Any]:
    try:
        return list(node.outputs())
    except TypeError:  # pragma: no cover
        return node.outputs()


def _aten_name(node: Any) -> str:
    kind = str(node.kind())
    return kind.split("::", 1)[1] if "::" in kind else kind


def _is_tensor_value(value: Any) -> bool:
    try:
        return str(value.type()).startswith("Tensor")
    except Exception:  # pragma: no cover - defensive
        return False


def _value_shape(value: Any) -> list[int] | None:
    try:
        tensor_type = value.type()
    except Exception:  # pragma: no cover
        return None
    try:
        if not str(tensor_type).startswith("Tensor"):
            return None
    except Exception:  # pragma: no cover
        return None
    try:
        sizes = tensor_type.sizes()
    except Exception:  # pragma: no cover
        sizes = None
    if sizes is None:
        return None
    return to_shape(sizes)


def _tensor_dtype(value: Any) -> str:
    try:
        scalar = value.type().scalarType()
    except Exception:  # pragma: no cover
        scalar = None
    return _TORCH_DTYPE_NAMES.get(str(scalar), "float32")


def _clean_io_name(name: str) -> str:
    return re.sub(r"\.\d+$", "", str(name or "")) or "input"


def _const_value(value: Any) -> Any:
    node = value.node()
    if str(node.kind()) != "prim::Constant":
        return _MISSING
    try:
        return node.output().toIValue()
    except Exception:  # pragma: no cover
        return _MISSING


def _const_list(value: Any) -> list[Any] | None:
    node = value.node()
    kind = str(node.kind())
    if kind == "prim::ListConstruct":
        items: list[Any] = []
        for item in _inputs(node):
            item_value = _const_value(item)
            if item_value is _MISSING:
                return None
            items.append(item_value)
        return items
    if kind == "prim::Constant":
        item = _const_value(value)
        if isinstance(item, (list, tuple)):
            return list(item)
        if isinstance(item, (int, float, bool)):
            return [item]
    return None


def _const_int(value: Any, default: int | None = None) -> int | None:
    item = _const_value(value)
    if isinstance(item, bool):
        return int(item)
    if isinstance(item, int):
        return int(item)
    return default


def _const_ints(value: Any, default: Sequence[int] | None = None) -> list[int] | None:
    items = _const_list(value)
    if items is None:
        scalar = _const_value(value)
        if isinstance(scalar, int) and not isinstance(scalar, bool):
            return [int(scalar)]
        return list(default) if default is not None else None
    out: list[int] = []
    for item in items:
        if isinstance(item, bool):
            out.append(int(item))
        elif isinstance(item, int):
            out.append(int(item))
        else:
            return list(default) if default is not None else None
    return out


class _GraphValueResolver:
    """Evaluate ``prim::GetAttr`` chains against the loaded module object."""

    def __init__(self, root: Any) -> None:
        self.root = root
        self.cache: dict[str, Any] = {}

    def resolve(self, value: Any) -> Any:
        debug_name = getattr(value, "debugName", lambda: "")()
        if debug_name and debug_name in self.cache:
            return self.cache[debug_name]
        result = self._resolve(value)
        if debug_name:
            self.cache[debug_name] = result
        return result

    def _resolve(self, value: Any) -> Any:
        node = value.node()
        kind = str(node.kind())
        if kind == "prim::Param":
            return self.root
        if kind == "prim::GetAttr":
            parents = _inputs(node)
            if not parents:
                return None
            parent = self.resolve(parents[0])
            if parent is None:
                return None
            try:
                attribute = node.s("name")
            except Exception:  # pragma: no cover
                return None
            try:
                return getattr(parent, attribute)
            except Exception:
                modules = getattr(parent, "_modules", None)
                if isinstance(modules, dict) and attribute in modules:
                    return modules[attribute]
                try:
                    parameters = dict(parent.named_parameters(recurse=False))
                except Exception:
                    parameters = {}
                if attribute in parameters:
                    return parameters[attribute]
                try:
                    buffers = dict(parent.named_buffers(recurse=False))
                except Exception:
                    buffers = {}
                if attribute in buffers:
                    return buffers[attribute]
                return None
        if kind == "prim::Constant":
            item = _const_value(value)
            return None if item is _MISSING else item
        return None


def _module_class_name(module: Any) -> str:
    """Best-effort plain class name of a (scripted) module."""
    name = getattr(module, "original_name", None)
    if isinstance(name, str) and name:
        return name
    try:
        type_name = module._c._type().name()
    except Exception:  # pragma: no cover
        type_name = type(module).__name__
    text = str(type_name)
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    text = re.sub(r"^_+", "", text)
    text = re.sub(r"torch_mangle_\d+", "", text).strip("_")
    return text or "Module"


# ---------------------------------------------------------------------------
# module call plan (recover the exact module hierarchy from the archive)
# ---------------------------------------------------------------------------

@dataclass
class _PlanEntry:
    aten: str = ""
    path: str = ""
    module: Any = None
    module_class: str = ""
    module_path: str = ""
    is_first_in_module: bool = False


def _walk_module_calls(module: Any, path: str, plan: list[_PlanEntry], depth: int = 0) -> None:
    """Append significant aten ops in execution order, annotated with module path."""
    if depth > 64:  # pragma: no cover - defensive against call-graph cycles
        return
    try:
        graph = module.graph
    except Exception:
        return
    try:
        nodes = list(graph.nodes())
    except Exception:  # pragma: no cover
        return
    if not nodes:
        return

    resolver = _GraphValueResolver(module)
    is_leaf = not any(str(node.kind()) == "prim::CallMethod" for node in nodes)
    module_class = _module_class_name(module)
    first_entry = True

    for node in nodes:
        kind = str(node.kind())
        if kind == "prim::CallMethod":
            inputs = _inputs(node)
            if not inputs:
                continue
            callee = resolver.resolve(inputs[0])
            if callee is None:
                continue
            attribute_path = _attribute_path(inputs[0], module, path)
            if attribute_path is None:
                attribute_path = path
            _walk_module_calls(callee, attribute_path, plan, depth + 1)
            continue
        if not kind.startswith("aten::"):
            continue
        attr = _aten_name(node)
        if attr not in _SIGNIFICANT_ATEN:
            continue
        first = is_leaf and first_entry
        plan.append(
            _PlanEntry(
                aten=attr,
                path=path,
                module=module if first else None,
                module_class=module_class if is_leaf else "",
                module_path=path,
                is_first_in_module=first,
            )
        )
        first_entry = False


def _attribute_path(value: Any, module: Any, path: str) -> str | None:
    """Resolve the module path of a ``prim::GetAttr`` chain (e.g. ``layer1.0``)."""
    chain: list[str] = []
    current = value
    for _ in range(32):
        node = current.node()
        kind = str(node.kind())
        if kind == "prim::Param":
            break
        if kind != "prim::GetAttr":
            return None
        try:
            chain.append(node.s("name"))
        except Exception:  # pragma: no cover
            return None
        parents = _inputs(node)
        if not parents:
            return None
        current = parents[0]
    chain.reverse()
    full = ".".join([segment for segment in [path, *chain] if segment])
    return full or None


def _align_plan(
    plan: Sequence[_PlanEntry], significant: Sequence[Any]
) -> tuple[dict[int, _PlanEntry], int]:
    """Map inlined significant node index -> plan entry.

    Returns ``(mapping, unmatched_count)``.
    """
    plan_kinds = [entry.aten for entry in plan]
    graph_kinds = [_aten_name(node) for node in significant]
    if plan_kinds == graph_kinds:
        return {index: plan[index] for index in range(len(plan))}, 0

    from difflib import SequenceMatcher

    mapping: dict[int, _PlanEntry] = {}
    matcher = SequenceMatcher(None, plan_kinds, graph_kinds, autojunk=False)
    matched = 0
    for block in matcher.get_matching_blocks():
        for offset in range(block.size):
            mapping[block.b + offset] = plan[block.a + offset]
            matched += 1
    if not plan_kinds:
        return {}, len(significant)
    ratio = matched / max(1, len(graph_kinds))
    if ratio < 0.75:
        return {}, len(significant)
    return mapping, len(significant) - matched


# ---------------------------------------------------------------------------
# shape inference
# ---------------------------------------------------------------------------

def _conv_out(size: int, kernel: int, stride: int, padding: int, dilation: int) -> int:
    if size < 0:
        return -1
    return (size + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1


def _pool_out(size: int, kernel: int, stride: int, padding: int, dilation: int = 1, ceil_mode: bool = False) -> int:
    if size < 0:
        return -1
    numerator = size + 2 * padding - dilation * (kernel - 1) - 1
    if ceil_mode:
        return -(-numerator // stride) + 1
    return numerator // stride + 1


def _tuple(values: Sequence[int] | None, dims: int, default: int = 1) -> list[int]:
    if not values:
        return [default] * dims
    if len(values) == 1:
        return [int(values[0])] * dims
    return [int(value) for value in values[:dims]]


def _spatial_conv(
    in_shape: Sequence[int],
    kernel: Sequence[int] | None,
    stride: Sequence[int] | None,
    padding: Sequence[int] | None,
    dilation: Sequence[int] | None,
    groups: int,
    out_channels: int,
    *,
    transpose: bool = False,
    output_padding: Sequence[int] | None = None,
) -> list[int]:
    if not in_shape:
        return [-1]
    rank = len(in_shape) - 2
    if rank <= 0:
        return list(in_shape)
    kernel = _tuple(kernel, rank)
    stride = _tuple(stride, rank)
    padding = _tuple(padding, rank, 0)
    dilation = _tuple(dilation, rank)
    out_padding = _tuple(output_padding, rank, 0)
    spatial: list[int] = []
    for index in range(rank):
        size = int(in_shape[2 + index])
        if transpose:
            if size < 0:
                spatial.append(-1)
            else:
                spatial.append(
                    (size - 1) * stride[index]
                    - 2 * padding[index]
                    + dilation[index] * (kernel[index] - 1)
                    + out_padding[index]
                    + 1
                )
        else:
            spatial.append(_conv_out(size, kernel[index], stride[index], padding[index], dilation[index]))
    out_channels = int(out_channels) if out_channels >= 0 else (int(in_shape[1]) if len(in_shape) > 1 else -1)
    return [int(in_shape[0]), out_channels, *spatial]


def _broadcast(left: Sequence[int], right: Sequence[int]) -> list[int]:
    if not left or not right:
        return list(left or right or [-1])
    result: list[int] = []
    for offset in range(1, max(len(left), len(right)) + 1):
        a = left[-offset] if offset <= len(left) else 1
        b = right[-offset] if offset <= len(right) else 1
        if a >= 0 and b >= 0 and a != b and a != 1 and b != 1:
            result.append(max(a, b))
        elif a == 1:
            result.append(b)
        elif b == 1:
            result.append(a)
        elif a < 0 or b < 0:
            result.append(a if b == 1 else b if a == 1 else -1)
        else:
            result.append(a)
    result.reverse()
    return result or [-1]


def _weight_shape(value: Any, resolver: _GraphValueResolver) -> list[int] | None:
    tensor = resolver.resolve(value)
    shape = getattr(tensor, "shape", None)
    if shape is None:
        return None
    try:
        return [int(dim) for dim in shape]
    except TypeError:  # pragma: no cover
        return None


def _weight_numel(value: Any, resolver: _GraphValueResolver) -> int:
    tensor = resolver.resolve(value)
    try:
        return int(tensor.numel())
    except Exception:  # pragma: no cover
        return 0


def _infer_node_shape(
    node: Any,
    in_shapes: Sequence[Sequence[int]],
    resolver: _GraphValueResolver,
) -> list[int] | None:
    """Analytic output shape for one significant aten node."""
    aten = _aten_name(node)
    inputs = _inputs(node)
    first = list(in_shapes[0]) if in_shapes else []

    if aten == "_convolution":
        if len(inputs) < 2:
            return None
        weight_shape = _weight_shape(inputs[1], resolver)
        groups = _const_int(inputs[8], 1) if len(inputs) > 8 else 1
        stride = _const_ints(inputs[3]) if len(inputs) > 3 else None
        padding = _const_ints(inputs[4]) if len(inputs) > 4 else None
        dilation = _const_ints(inputs[5]) if len(inputs) > 5 else None
        transpose = bool(_const_value(inputs[6])) if len(inputs) > 6 else False
        output_padding = _const_ints(inputs[7]) if len(inputs) > 7 else None
        out_channels = weight_shape[0] if weight_shape else -1
        kernel = list(weight_shape[2:]) if weight_shape else None
        return _spatial_conv(
            first,
            kernel,
            stride,
            padding,
            dilation,
            max(1, int(groups or 1)),
            out_channels,
            transpose=transpose is True,
            output_padding=output_padding,
        )
    if aten.startswith("conv"):
        if len(inputs) < 2:
            return None
        weight_shape = _weight_shape(inputs[1], resolver)
        stride = _const_ints(inputs[3]) if len(inputs) > 3 else None
        padding = _const_ints(inputs[4]) if len(inputs) > 4 else None
        dilation = _const_ints(inputs[5]) if len(inputs) > 5 else None
        groups = _const_int(inputs[6], 1) if len(inputs) > 6 else 1
        return _spatial_conv(
            first,
            list(weight_shape[2:]) if weight_shape else None,
            stride,
            padding,
            dilation,
            max(1, int(groups or 1)),
            weight_shape[0] if weight_shape else -1,
            transpose=aten.startswith("conv_transpose"),
        )
    if aten in ("linear", "addmm"):
        if aten == "linear" and len(inputs) >= 2:
            weight_shape = _weight_shape(inputs[1], resolver)
        elif len(inputs) >= 3:
            weight_shape = _weight_shape(inputs[2], resolver)
        else:
            weight_shape = None
        out_features = weight_shape[0] if weight_shape else -1
        if not first:
            return None
        return [*first[:-1], int(out_features)]
    if aten in ("matmul", "bmm"):
        if len(in_shapes) < 2:
            return None
        left, right = list(in_shapes[0]), list(in_shapes[1])
        if len(left) < 2 or len(right) < 2:
            return None
        batch = _broadcast(left[:-2], right[:-2])
        return [*batch, left[-2], right[-1]]
    if aten in ("batch_norm", "native_batch_norm", "_native_batch_norm_legit", "_native_batch_norm_legit_no_training",
                "layer_norm", "group_norm", "instance_norm"):
        return first or None
    if aten in ("relu", "relu_", "gelu", "leaky_relu", "elu", "selu", "celu", "silu", "swish",
                "sigmoid", "tanh", "hardswish", "hardsigmoid", "hardtanh", "softplus", "mish",
                "prelu", "softmax", "_softmax", "log_softmax", "dropout", "dropout_",
                "feature_dropout", "scaled_dot_product_attention"):
        return first or None
    if aten in ("max_pool1d", "max_pool2d", "max_pool3d", "avg_pool1d", "avg_pool2d", "avg_pool3d"):
        if not first:
            return None
        rank = len(first) - 2
        if rank <= 0:
            return first
        kernel = _const_ints(inputs[1]) if len(inputs) > 1 else None
        stride = _const_ints(inputs[2]) if len(inputs) > 2 else kernel
        padding = _const_ints(inputs[3]) if len(inputs) > 3 else None
        dilation = _const_ints(inputs[4]) if (aten.startswith("max") and len(inputs) > 4) else None
        ceil_mode = bool(_const_value(inputs[5])) if (not aten.startswith("max") and len(inputs) > 5) else False
        kernel = _tuple(kernel, rank)
        stride = _tuple(stride or kernel, rank)
        padding = _tuple(padding, rank, 0)
        dilation = _tuple(dilation, rank)
        spatial = [
            _pool_out(first[2 + index], kernel[index], stride[index], padding[index], dilation[index], ceil_mode)
            for index in range(rank)
        ]
        return [first[0], first[1], *spatial]
    if aten.startswith("adaptive_") and "pool" in aten:
        if not first:
            return None
        output_size = _const_ints(inputs[1]) if len(inputs) > 1 else None
        rank = len(first) - 2
        if output_size is None:
            return [first[0], first[1], *([-1] * rank)]
        return [first[0], first[1], *_tuple(output_size, rank)]
    if aten == "flatten":
        if not first:
            return None
        start = _const_int(inputs[1], 1) if len(inputs) > 1 else 1
        end = _const_int(inputs[2], -1) if len(inputs) > 2 else -1
        dims = len(first)
        start = int(start if start is not None else 1)
        end = int(end if end is not None else -1)
        if start < 0:
            start += dims
        if end < 0:
            end += dims
        start = max(0, min(start, dims - 1))
        end = max(start, min(end, dims - 1))
        flat = 1
        for dim in first[start : end + 1]:
            if dim < 0:
                flat = -1
                break
            flat *= dim
        return [*first[:start], flat, *first[end + 1 :]]
    if aten in ("view", "reshape"):
        target = _const_ints(inputs[1]) if len(inputs) > 1 else None
        if target is None:
            return first or None
        total: int | None = 1
        for dim in first:
            if dim < 0:
                total = None
                break
            total *= dim
        known = 1
        dynamic = None
        for index, dim in enumerate(target):
            if dim == -1:
                dynamic = index
            elif dim >= 0:
                known *= dim
        out: list[int] = []
        for dim in target:
            if dim == -1:
                out.append(int(total // known) if total is not None and known else -1)
            else:
                out.append(int(dim))
        return out or first
    if aten == "permute":
        dims = _const_ints(inputs[1]) if len(inputs) > 1 else None
        if dims is None or not first or len(dims) != len(first):
            return first or None
        return [first[index] for index in dims]
    if aten == "transpose":
        if len(inputs) < 3:
            return first or None
        dim0 = _const_int(inputs[1], 0)
        dim1 = _const_int(inputs[2], 1)
        if dim0 is None or dim1 is None or not first:
            return first or None
        out = list(first)
        if not (-len(out) <= dim0 < len(out) and -len(out) <= dim1 < len(out)):
            return out
        dim0 %= len(out)
        dim1 %= len(out)
        out[dim0], out[dim1] = out[dim1], out[dim0]
        return out
    if aten in ("unsqueeze", "unsqueeze_"):
        dim = _const_int(inputs[1], 0) if len(inputs) > 1 else 0
        if dim is None or not first:
            return first or None
        if not (-(len(first) + 1) <= dim <= len(first)):
            return first
        if dim < 0:
            dim += len(first) + 1
        return [*first[:dim], 1, *first[dim:]]
    if aten in ("squeeze", "squeeze_"):
        if len(inputs) > 1:
            dims = _const_ints(inputs[1])
            if dims:
                out = [size for index, size in enumerate(first) if index not in {d % len(first) for d in dims}]
                return out or [1]
        out = [size for size in first if size != 1]
        return out or [1]
    if aten == "expand":
        target = _const_ints(inputs[1]) if len(inputs) > 1 else None
        if target is None:
            return first or None
        if len(target) > len(first):
            prefix = [1] * (len(target) - len(first))
            first = [*prefix, *first]
        return [dim if dim >= 0 else first[index] for index, dim in enumerate(target)]
    if aten in ("cat", "concat"):
        dim = _const_int(inputs[1], 0) if len(inputs) > 1 else 0
        if not in_shapes or dim is None:
            return first or None
        base = list(in_shapes[0])
        if len(base) <= dim:
            return base
        total = 0
        for shape in in_shapes:
            if len(shape) <= dim:
                return base
            if total >= 0 and shape[dim] >= 0:
                total += shape[dim]
            else:
                total = -1
        out = list(base)
        out[dim] = total
        return out
    if aten in ("add", "add_", "sub", "sub_", "mul", "mul_", "div", "truediv"):
        if len(in_shapes) < 2:
            return first or None
        return _broadcast(in_shapes[0], in_shapes[1])
    if aten in ("mean", "sum"):
        if not first:
            return None
        dims: list[int] | None = None
        if len(inputs) > 1:
            dims = _const_ints(inputs[1])
            if dims is None:
                single = _const_int(inputs[1])
                dims = [single] if single is not None else None
        keepdim = False
        if len(inputs) > 2:
            try:
                keepdim = bool(_const_value(inputs[2]))
            except Exception:  # pragma: no cover
                keepdim = False
        if not dims:
            return [1]
        resolved = [dim % len(first) for dim in dims]
        if keepdim:
            return [1 if index in resolved else size for index, size in enumerate(first)]
        out = [size for index, size in enumerate(first) if index not in resolved]
        return out or [1]
    if aten == "embedding":
        if len(inputs) < 2:
            return None
        weight_shape = _weight_shape(inputs[0], resolver)
        if not weight_shape or not in_shapes:
            return None
        return [*in_shapes[0], int(weight_shape[-1])]
    if aten in ("upsample_nearest2d", "upsample_bilinear2d", "interpolate"):
        if not first:
            return None
        rank = len(first) - 2
        output_size = _const_ints(inputs[2]) if len(inputs) > 2 else None
        if output_size and len(output_size) >= rank:
            return [first[0], first[1], *[int(size) for size in output_size[:rank]]]
        scale = None
        for index in (2, 3):
            if len(inputs) > index:
                raw = _const_value(inputs[index])
                if isinstance(raw, (list, tuple)) and raw:
                    scale = [float(item) for item in raw]
                    break
                if isinstance(raw, float):
                    scale = [raw] * rank
                    break
        if scale:
            return [first[0], first[1], *[int(first[2 + index] * scale[index]) if first[2 + index] >= 0 else -1 for index in range(rank)]]
        return first
    if aten == "pad":
        return first or None
    return None


# ---------------------------------------------------------------------------
# module / op parameter extraction
# ---------------------------------------------------------------------------

def _named_parameters(module: Any, *, recurse: bool = False) -> dict[str, Any]:
    try:
        return dict(module.named_parameters(recurse=recurse))
    except Exception:  # pragma: no cover
        return {}


def _named_buffers(module: Any, *, recurse: bool = False) -> dict[str, Any]:
    try:
        return dict(module.named_buffers(recurse=recurse))
    except Exception:  # pragma: no cover
        return {}


def _tensor_numel(value: Any) -> int:
    try:
        return int(value.numel())
    except Exception:  # pragma: no cover
        return 0


def _module_params_payload(
    module: Any | None,
    *,
    module_class: str,
    in_shapes: Sequence[Sequence[int]],
    aten: str,
    node: Any,
    resolver: _GraphValueResolver,
) -> tuple[dict[str, Any], int, int]:
    """Return ``(params, numParams, kind_hint)`` for one significant node."""
    kind = ops.classify_module(module_class) if module_class else _KIND_BY_ATEN.get(aten, "custom")
    params: dict[str, Any] = {}
    num_params = 0
    parameters = _named_parameters(module) if module is not None else {}
    buffers = _named_buffers(module) if module is not None else {}
    weight = parameters.get("weight")
    bias = parameters.get("bias")
    inputs = _inputs(node)
    first = list(in_shapes[0]) if in_shapes else []

    if kind in ("conv", "deconv"):
        if weight is not None:
            weight_shape = [int(dim) for dim in weight.shape]
            kernel = list(weight_shape[2:])
            groups = _const_int(inputs[8], 1) if (aten == "_convolution" and len(inputs) > 8) else (
                _const_int(inputs[6], 1) if len(inputs) > 6 else 1
            )
            groups = max(1, int(groups or 1))
            in_channels = int(first[1]) if len(first) > 1 and first[1] >= 0 else int(weight_shape[1]) * groups
            params = {
                "inChannels": int(in_channels),
                "outChannels": int(weight_shape[0]),
                "kernelSize": kernel,
                "stride": _const_ints(inputs[3]) if len(inputs) > 3 else [1] * len(kernel),
                "padding": _const_ints(inputs[4]) if len(inputs) > 4 else [0] * len(kernel),
                "dilation": _const_ints(inputs[5]) if len(inputs) > 5 else [1] * len(kernel),
                "groups": groups,
                "bias": bias is not None,
            }
            if kind == "deconv":
                params["outputPadding"] = _const_ints(inputs[7]) if len(inputs) > 7 else [0] * len(kernel)
            num_params = _tensor_numel(weight) + (_tensor_numel(bias) if bias is not None else 0)
        else:
            weight_shape = _weight_shape(inputs[1], resolver) if len(inputs) > 1 else None
            if weight_shape:
                params = {
                    "inChannels": int(first[1]) if len(first) > 1 and first[1] >= 0 else int(weight_shape[1]),
                    "outChannels": int(weight_shape[0]),
                    "kernelSize": list(weight_shape[2:]),
                    "stride": _const_ints(inputs[3]) if len(inputs) > 3 else [1] * max(1, len(weight_shape) - 2),
                    "padding": _const_ints(inputs[4]) if len(inputs) > 4 else [0] * max(1, len(weight_shape) - 2),
                    "dilation": _const_ints(inputs[5]) if len(inputs) > 5 else [1] * max(1, len(weight_shape) - 2),
                    "groups": 1,
                    "bias": len(inputs) > 2,
                }
                num_params = shape_numel(weight_shape) or 0
                if len(inputs) > 2:
                    num_params += _weight_numel(inputs[2], resolver)
        return params, int(num_params), kind

    if kind in ("linear", "mlp", "ffn"):
        if weight is not None:
            weight_shape = [int(dim) for dim in weight.shape]
            params = {
                "inFeatures": int(weight_shape[1]),
                "outFeatures": int(weight_shape[0]),
                "bias": bias is not None,
            }
            num_params = _tensor_numel(weight) + (_tensor_numel(bias) if bias is not None else 0)
        else:
            weight_shape = _weight_shape(inputs[1], resolver) if len(inputs) > 1 else None
            if weight_shape:
                params = {
                    "inFeatures": int(weight_shape[1]),
                    "outFeatures": int(weight_shape[0]),
                    "bias": len(inputs) > 2,
                }
                num_params = shape_numel(weight_shape) or 0
                if len(inputs) > 2:
                    num_params += _weight_numel(inputs[2], resolver)
        return params, int(num_params), kind

    if kind == "norm":
        channels = -1
        for value in (weight, bias, buffers.get("running_mean"), buffers.get("running_var")):
            if value is not None:
                try:
                    channels = int(value.shape[0]) if len(value.shape) else -1
                except Exception:  # pragma: no cover
                    channels = -1
                if channels > 0:
                    break
        if channels <= 0 and len(first) > 1 and first[1] >= 0:
            channels = int(first[1])
        params = {
            "normalizedShape": [channels] if channels > 0 else [-1],
            "eps": 1e-5,
            "affine": weight is not None or bias is not None,
        }
        if "layer_norm" in aten and len(inputs) > 1:
            normalized = _const_ints(inputs[1])
            if normalized:
                params["normalizedShape"] = normalized
        num_params = sum(_tensor_numel(value) for value in parameters.values())
        return params, int(num_params), kind

    if kind == "pool":
        low = aten.lower()
        rank = max(1, len(first) - 2)
        pool_params: dict[str, Any] = {
            "type": "max" if "max" in low else ("adaptive" if "adaptive" in low else "avg"),
        }
        if "adaptive" in low:
            output = _const_ints(inputs[1]) if len(inputs) > 1 else None
            pool_params["outputSize"] = output or [-1] * rank
        else:
            pool_params["kernelSize"] = _const_ints(inputs[1]) if len(inputs) > 1 else [1] * rank
            pool_params["stride"] = _const_ints(inputs[2]) if len(inputs) > 2 else pool_params["kernelSize"]
            pool_params["padding"] = _const_ints(inputs[3]) if len(inputs) > 3 else [0] * rank
        return pool_params, 0, kind

    if kind == "flatten":
        params = {
            "startDim": _const_int(inputs[1], 1) if len(inputs) > 1 else 1,
            "endDim": _const_int(inputs[2], -1) if len(inputs) > 2 else -1,
        }
        return params, 0, kind

    if kind == "reshape":
        if aten == "view" or aten == "reshape":
            target = _const_ints(inputs[1]) if len(inputs) > 1 else None
            if target is not None:
                params["shape"] = target
        elif aten == "permute":
            dims = _const_ints(inputs[1]) if len(inputs) > 1 else None
            if dims is not None:
                params["dims"] = dims
        elif aten == "unsqueeze":
            dim = _const_int(inputs[1]) if len(inputs) > 1 else None
            if dim is not None:
                params["dim"] = dim
        return params, 0, kind

    if kind == "dropout":
        p = _const_value(inputs[1]) if len(inputs) > 1 else None
        if isinstance(p, float):
            params["p"] = float(p)
        return params, 0, kind

    if kind == "embedding":
        if weight is not None:
            weight_shape = [int(dim) for dim in weight.shape]
            params = {"numEmbeddings": int(weight_shape[0]), "embeddingDim": int(weight_shape[1]) if len(weight_shape) > 1 else -1}
            num_params = _tensor_numel(weight)
        else:
            weight_shape = _weight_shape(inputs[0], resolver) if len(inputs) > 0 else None
            if weight_shape:
                params = {"numEmbeddings": int(weight_shape[0]), "embeddingDim": int(weight_shape[-1])}
                num_params = shape_numel(weight_shape) or 0
        return params, int(num_params), kind

    if kind in ("block", "attention", "rnn", "custom", "unknown"):
        # Fused transformer / recurrent operators hide their individual
        # projectors; still attribute the module parameters to the node.
        num_params = sum(_tensor_numel(value) for value in parameters.values())
        return params, int(num_params), kind

    return params, 0, kind


# ---------------------------------------------------------------------------
# main analysis
# ---------------------------------------------------------------------------

def _import_torch(path: Path) -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise AnalyzeError(
            "PyTorch is required to read TorchScript archives; install it with "
            "`pip install torch` (see README for the optional dependencies).",
            file=path,
        ) from exc
    return torch


def _load_archive(torch: Any, path: Path) -> Any:
    load = torch.jit.load
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            try:
                module = load(str(path), map_location="cpu", _restore_shapes=True)
            except TypeError:  # pragma: no cover - older torch without the flag
                module = load(str(path), map_location="cpu")
    except Exception as exc:  # noqa: BLE001 - any failure means "not a TorchScript archive"
        raise AnalyzeError(_archive_error_message(torch, path, exc), file=path) from exc
    return module


def _archive_error_message(torch: Any, path: Path, exc: Exception) -> str:
    payload: Any = None
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            payload = torch.load(str(path), map_location="cpu", weights_only=True)
    except Exception:
        payload = None
    if isinstance(payload, dict) and payload:
        try:
            tensors = all(torch.is_tensor(value) for value in payload.values())
        except Exception:  # pragma: no cover
            tensors = False
        if tensors:
            return (
                "this file is a plain state_dict, not a TorchScript archive. "
                "Re-save the model with torch.jit.trace(model, example_input) or "
                "torch.jit.script(model) followed by torch.jit.save(...), or export it to ONNX."
            )
    return f"not a valid TorchScript archive ({type(exc).__name__}: {exc})"


def _dummy_inputs(torch: Any, module: Any) -> tuple[list[Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Return ``(dummies, inputSpec, graph_inputs)`` from the archive metadata."""
    dummies: list[Any] = []
    specs: list[dict[str, Any]] = []
    tensor_inputs: list[Any] = []
    try:
        graph_inputs = list(module.inlined_graph.inputs())
    except Exception:  # pragma: no cover - fall back to the outer graph
        try:
            graph_inputs = list(module.graph.inputs())
        except Exception:
            graph_inputs = []
    dtype_map = {
        "Float": torch.float32,
        "Double": torch.float64,
        "Half": torch.float16,
        "BFloat16": torch.bfloat16,
        "Long": torch.int64,
        "Int": torch.int32,
        "Short": torch.int16,
        "Char": torch.int8,
        "Byte": torch.uint8,
        "Bool": torch.bool,
    }
    for value in graph_inputs:
        if not _is_tensor_value(value):
            continue
        shape = _value_shape(value)
        dtype_name = _tensor_dtype(value)
        specs.append({"name": _clean_io_name(value.debugName()), "shape": shape or [-1], "dtype": dtype_name})
        tensor_inputs.append(value)
        if shape is None or any(dim < 0 for dim in shape):
            dummies.append(None)
            continue
        dtype = dtype_map.get(str(value.type().scalarType()) if hasattr(value.type(), "scalarType") else "", torch.float32)
        try:
            dummies.append(torch.zeros(*shape, dtype=dtype))
        except Exception:  # pragma: no cover
            dummies.append(None)
    return dummies, specs, tensor_inputs


def _activation_input_values(node: Any) -> list[Any]:
    """Tensor values consumed by ``node`` (lists expanded, weights excluded)."""
    collected: list[Any] = []
    for value in _inputs(node):
        producer = value.node()
        kind = str(producer.kind())
        if kind == "prim::GetAttr" or kind == "prim::Constant":
            continue
        if kind == "prim::ListConstruct":
            for item in _inputs(producer):
                item_producer = item.node()
                if str(item_producer.kind()) in ("prim::GetAttr", "prim::Constant"):
                    continue
                if _is_tensor_value(item):
                    collected.append(item)
            continue
        if _is_tensor_value(value):
            collected.append(value)
    return collected


def _producer_output_shape(value: Any, env: dict[str, list[int]]) -> list[int] | None:
    name = value.debugName() if hasattr(value, "debugName") else ""
    if name and name in env:
        return env[name]
    node = value.node()
    kind = str(node.kind())
    if kind == "prim::GetAttr" or kind == "prim::Constant":
        return None
    if kind.startswith("aten::") and _aten_name(node) in _PASSTHROUGH_ATEN:
        for item in _inputs(node):
            if _is_tensor_value(item):
                shape = _producer_output_shape(item, env)
                if shape is not None:
                    return shape
        return _value_shape(value)
    return _value_shape(value)


def _run_outputs(torch: Any, module: Any, dummies: Sequence[Any]) -> list[list[int]] | None:
    if not dummies or any(item is None for item in dummies):
        return None
    try:
        with torch.no_grad(), warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            outputs = module(*dummies)
    except Exception as exc:  # noqa: BLE001 - execution is best effort
        LOGGER.debug("TorchScript replay failed: %s", exc)
        return None
    shapes: list[list[int]] = []

    def collect(value: Any, depth: int = 0) -> None:
        if depth > 4:
            return
        shape = getattr(value, "shape", None)
        if shape is not None:
            shapes.append(to_shape(shape))
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                collect(item, depth + 1)
        elif isinstance(value, dict):
            for item in value.values():
                collect(item, depth + 1)

    collect(outputs)
    return shapes or None


def analyze_torchscript(path: str | Path, *, max_nodes: int = 20000) -> list[dict[str, Any]]:
    """Analyze one TorchScript archive and return a single graph document."""
    file_path = Path(path).expanduser()
    torch = _import_torch(file_path)
    module = _load_archive(torch, file_path)
    warnings_list: list[str] = []

    try:
        module.eval()
    except Exception:  # pragma: no cover
        pass

    try:
        inlined = module.inlined_graph
    except Exception as exc:  # noqa: BLE001
        raise AnalyzeError(f"could not read the TorchScript graph ({exc})", file=file_path) from exc

    try:
        significant = [
            node for node in inlined.nodes() if str(node.kind()).startswith("aten::") and _aten_name(node) in _SIGNIFICANT_ATEN
        ]
    except Exception as exc:  # pragma: no cover
        raise AnalyzeError(f"could not iterate the TorchScript graph ({exc})", file=file_path) from exc

    if not significant:
        raise AnalyzeError("the TorchScript archive contains no analyzable operators", file=file_path)
    if len(significant) + 2 > max_nodes:
        raise AnalyzeError(
            f"model has more than {max_nodes - 2} operators, above the max_nodes limit ({max_nodes})",
            file=file_path,
        )

    plan: list[_PlanEntry] = []
    try:
        _walk_module_calls(module, "", plan)
    except Exception as exc:  # noqa: BLE001 - hierarchy recovery is best effort
        LOGGER.debug("TorchScript module hierarchy walk failed: %s", exc)
        warnings_list.append(f"could not recover the module hierarchy ({type(exc).__name__}: {exc})")
        plan = []
    mapping, unmatched = _align_plan(plan, significant)
    if unmatched:
        warnings_list.append(
            f"could not map {unmatched} TorchScript operation(s) to the module hierarchy; "
            "those nodes are attached to the model root group."
        )

    resolver = _GraphValueResolver(module)
    dummies, input_specs, _tensor_inputs = _dummy_inputs(torch, module)
    if any(dummy is None for dummy in dummies):
        warnings_list.append("the archive does not record every input shape; affected shapes use -1.")
    elif not input_specs:
        warnings_list.append("no tensor inputs were found in the archive.")

    for spec in input_specs:
        if any(dim < 0 for dim in spec["shape"]):
            warnings_list.append(f"input {spec['name']!r} has a dynamic shape; affected shapes use -1.")

    env: dict[str, list[int]] = {}
    input_nodes: list[dict[str, Any]] = []
    producer_by_value: dict[str, int] = {}
    for spec, value in zip(input_specs, _tensor_inputs):
        name = spec["name"]
        shape = spec["shape"]
        node = make_boundary_node(name=name, op="Input", kind="input", out_shapes=[shape])
        node["id"] = f"n{len(input_nodes)}"
        input_nodes.append(node)
        env[value.debugName()] = shape
        producer_by_value[value.debugName()] = len(input_nodes) - 1

    nodes: list[dict[str, Any]] = list(input_nodes)
    edges: list[dict[str, Any]] = []
    name_counts: dict[str, int] = {}
    produced_node_index: dict[str, int] = {}
    unknown_shape_count = 0

    for graph_position, graph_node in enumerate(significant):
        plan_entry = mapping.get(graph_position)
        aten = _aten_name(graph_node)
        module_obj = plan_entry.module if plan_entry is not None else None
        module_class = plan_entry.module_class if plan_entry is not None else ""
        module_path = ""
        if plan_entry is not None:
            if module_obj is not None:
                module_path = plan_entry.module_path
            elif module_class and not plan_entry.is_first_in_module:
                module_path = plan_entry.module_path
            else:
                base = plan_entry.module_path
                suffix = _FRIENDLY_ATEN.get(aten, aten).replace("torch.", "").replace(".", "_")
                module_path = f"{base}.{suffix}" if base else suffix
        op_name = module_class if (module_obj is not None and module_class) else _FRIENDLY_ATEN.get(aten, aten)

        activation_values = _activation_input_values(graph_node)
        in_shapes: list[list[int]] = []
        for value in activation_values:
            shape = _producer_output_shape(value, env)
            if shape is None:
                shape = [-1]
            in_shapes.append(shape)

        params, num_params, kind = _module_params_payload(
            module_obj,
            module_class=module_class if module_obj is not None else "",
            in_shapes=in_shapes,
            aten=aten,
            node=graph_node,
            resolver=resolver,
        )
        out_shape = _infer_node_shape(graph_node, in_shapes, resolver)
        if out_shape is None:
            out_shape = list(in_shapes[0]) if in_shapes else [-1]
            unknown_shape_count += 1
        out_shape = to_shape(out_shape)
        out_shapes = [out_shape]
        for output_value in _outputs(graph_node):
            if _is_tensor_value(output_value):
                env[output_value.debugName()] = out_shape

        base_name = module_path or op_name or aten
        count = name_counts.get(base_name, 0) + 1
        name_counts[base_name] = count
        display_name = base_name if count == 1 else f"{base_name}#{count}"

        flops = ops.module_flops(kind, params, in_shapes, [out_shape], op_name)
        node = make_node(
            name=display_name,
            op=op_name,
            kind=kind,
            module_path=module_path,
            in_shapes=in_shapes,
            out_shapes=out_shapes,
            params=params,
            num_params=num_params,
            flops=flops,
            tags=ops.tags_for(kind, op_name, params),
            meta={"aten": f"aten::{aten}"},
        )
        node["id"] = f"n{len(nodes)}"
        node_index = len(nodes)
        nodes.append(node)
        for output_value in _outputs(graph_node):
            if _is_tensor_value(output_value):
                produced_node_index[output_value.debugName()] = node_index

        # Data edges from activation tensor producers.
        seen_sources: set[int] = set()
        for value in activation_values:
            source_index = _resolve_producer(value, produced_node_index, producer_by_value)
            if source_index is None or source_index == node_index:
                continue
            if source_index in seen_sources:
                continue
            seen_sources.add(source_index)
            edge_kind = "residual" if (kind == "add" and len(seen_sources) > 1) else "data"
            edges.append(
                make_edge(
                    source=nodes[source_index]["id"],
                    target=node["id"],
                    shapes=nodes[source_index]["outShapes"][:1],
                    kind=edge_kind,
                )
            )

    # Output boundary nodes.
    output_specs: list[dict[str, Any]] = []
    try:
        graph_outputs = list(inlined.outputs())
    except Exception:  # pragma: no cover
        graph_outputs = []
    output_shapes = _run_outputs(torch, module, dummies)
    if output_shapes is None:
        warnings_list.append("could not execute the archive to verify output shapes; output shapes use graph propagation.")
    output_shape_list: list[list[int]] = []
    for index, value in enumerate(graph_outputs):
        if not _is_tensor_value(value):
            continue
        shape = None
        if output_shapes and index < len(output_shapes):
            shape = output_shapes[index]
        if shape is None:
            shape = _producer_output_shape(value, env)
        if shape is None:
            shape = _value_shape(value)
        shape = list(shape or [-1])
        output_shape_list.append(shape)
        output_specs.append({"name": f"output_{index + 1}" if index else "output", "shape": shape, "dtype": _tensor_dtype(value)})
    if not output_specs:
        output_specs = [{"name": "output", "shape": [-1], "dtype": "float32"}]
        output_shape_list = [[-1]]
        warnings_list.append("the archive does not expose tensor outputs; output shape uses -1.")

    output_nodes: list[dict[str, Any]] = []
    for spec, shape in zip(output_specs, output_shape_list):
        output_node = make_boundary_node(
            name=spec["name"], op="Output", kind="output", in_shapes=[shape], out_shapes=[shape]
        )
        output_node["id"] = f"n{len(nodes) + len(output_nodes)}"
        output_nodes.append(output_node)
    output_start = len(nodes)
    nodes.extend(output_nodes)
    # Connect graph outputs to their producers.
    for index, value in enumerate([item for item in graph_outputs if _is_tensor_value(item)][: len(output_nodes)]):
        source_index = _resolve_producer(value, produced_node_index, producer_by_value)
        if source_index is None:
            continue
        edges.append(
            make_edge(
                source=nodes[source_index]["id"],
                target=nodes[output_start + index]["id"],
                shapes=nodes[source_index]["outShapes"][:1],
                kind="data",
            )
        )

    if unknown_shape_count:
        warnings_list.append(
            f"shapes for {unknown_shape_count} TorchScript node(s) could not be propagated; filled with -1."
        )
    warnings_list.append(
        "TorchScript archives do not retain per-operator tensor shapes; in/out shapes were "
        "propagated from the recorded input shape and the exported operator constants."
    )

    total_params = 0
    trainable_params = 0
    try:
        for parameter in module.parameters():
            count = _tensor_numel(parameter)
            total_params += count
            if bool(getattr(parameter, "requires_grad", False)):
                trainable_params += count
    except Exception:  # pragma: no cover
        total_params = sum(int(node.get("numParams", 0)) for node in nodes)
        trainable_params = total_params

    class_name = getattr(module, "original_name", None) or _module_class_name(module)
    model_name = str(class_name) if class_name else file_path.stem

    meta = {
        "format": "torchscript",
        "torchVersion": str(getattr(torch, "__version__", "")),
        "archiveClass": class_name,
        "shapeSource": "propagated",
        "operators": len(significant),
        "moduleMapped": len(mapping),
    }
    graph = assemble_graph(
        path=file_path,
        framework="torchscript",
        mode="torchscript",
        model_name=model_name,
        class_name=str(class_name) if class_name else None,
        nodes=nodes,
        edges=edges,
        input_specs=input_specs,
        output_specs=output_specs,
        warnings=warnings_list,
        total_params=total_params,
        trainable_params=trainable_params,
        flops_approx=True,
        meta=meta,
    )
    return [graph]


def _resolve_producer(
    value: Any,
    produced_node_index: dict[str, int],
    producer_by_value: dict[str, int],
) -> int | None:
    """Resolve the significant node that produced ``value`` (through passthrough ops)."""
    name = value.debugName() if hasattr(value, "debugName") else ""
    if name and name in produced_node_index:
        return produced_node_index[name]
    if name and name in producer_by_value:
        return producer_by_value[name]
    node = value.node()
    kind = str(node.kind())
    if kind == "prim::GetAttr" or kind == "prim::Constant":
        return None
    if kind.startswith("aten::") and _aten_name(node) in _PASSTHROUGH_ATEN:
        for item in _inputs(node):
            if _is_tensor_value(item):
                resolved = _resolve_producer(item, produced_node_index, producer_by_value)
                if resolved is not None:
                    return resolved
        return None
    return None


__all__ = ["analyze_torchscript"]
