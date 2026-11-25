"""ONNX adapter (``.onnx``).

Uses ``onnx.load`` + ``onnx.shape_inference.infer_shapes`` for tensor shapes,
initializer shapes for parameters and the recorded op attributes for FLOPs.
Grouping follows the node-name scopes exported by PyTorch / ONNX tooling
(``/features/0/Conv`` -> group ``features.0``); nodes without a scoped name stay
in the model root group.

``onnx`` is imported lazily: importing this module must not require it.
"""

from __future__ import annotations

import logging
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

LOGGER = logging.getLogger("netprism.adapters.onnx")

_DTYPE_NAMES = {
    "FLOAT": "float32",
    "DOUBLE": "float64",
    "FLOAT16": "float16",
    "BFLOAT16": "bfloat16",
    "INT64": "int64",
    "INT32": "int32",
    "INT16": "int16",
    "INT8": "int8",
    "UINT8": "uint8",
    "UINT16": "uint16",
    "UINT32": "uint32",
    "UINT64": "uint64",
    "BOOL": "bool",
    "STRING": "string",
    "COMPLEX64": "complex64",
    "COMPLEX128": "complex128",
}

#: ``op_type`` -> NetPrism node kind.
_KIND_BY_OP: dict[str, str] = {
    "Conv": "conv",
    "ConvTranspose": "deconv",
    "Gemm": "linear",
    "MatMul": "linear",
    "BatchNormalization": "norm",
    "LayerNormalization": "norm",
    "InstanceNormalization": "norm",
    "GroupNormalization": "norm",
    "Relu": "activation",
    "Gelu": "activation",
    "Sigmoid": "activation",
    "Tanh": "activation",
    "Elu": "activation",
    "LeakyRelu": "activation",
    "Selu": "activation",
    "Celu": "activation",
    "Silu": "activation",
    "Swish": "activation",
    "HardSigmoid": "activation",
    "HardSwish": "activation",
    "Softplus": "activation",
    "Mish": "activation",
    "Erf": "activation",
    "PRelu": "activation",
    "Softmax": "softmax",
    "LogSoftmax": "softmax",
    "Dropout": "dropout",
    "MaxPool": "pool",
    "AveragePool": "pool",
    "GlobalAveragePool": "pool",
    "GlobalMaxPool": "pool",
    "LpPool": "pool",
    "MaxUnpool": "pool",
    "Flatten": "flatten",
    "Reshape": "reshape",
    "Squeeze": "reshape",
    "Unsqueeze": "reshape",
    "Transpose": "reshape",
    "Expand": "reshape",
    "Concat": "concat",
    "Add": "add",
    "Sum": "add",
    "Sub": "custom",
    "Mul": "custom",
    "Div": "custom",
    "Pow": "custom",
    "Gather": "embedding",
    "Embedding": "embedding",
    "Attention": "attention",
    "MultiHeadAttention": "attention",
    "LSTM": "rnn",
    "GRU": "rnn",
    "RNN": "rnn",
    "Upsample": "upsample",
    "Resize": "upsample",
    "Pad": "custom",
    "Slice": "custom",
    "Split": "custom",
    "ReduceMean": "pool",
    "ReduceSum": "pool",
    "ReduceMax": "pool",
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _import_onnx(path: Path) -> Any:
    try:
        import onnx
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise AnalyzeError(
            "ONNX is required to read .onnx models; install it with "
            "`pip install onnx` (see README for the optional dependencies).",
            file=path,
        ) from exc
    return onnx


def _decode(value: Any) -> Any:
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:  # pragma: no cover
            return value.hex()
    return value


def _attrs(node: Any, onnx: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for attribute in getattr(node, "attribute", []) or []:
        try:
            payload[attribute.name] = _decode(onnx.helper.get_attribute_value(attribute))
        except Exception:  # pragma: no cover - malformed attribute
            continue
    return payload


def _tensor_type(onnx: Any, value_info: Any) -> tuple[list[int] | None, str | None]:
    """Return ``(shape, dtype)`` for one ``ValueInfoProto``."""
    try:
        tensor_type = value_info.type.tensor_type
    except Exception:  # pragma: no cover - sequence/map types
        return None, None
    dtype_code = int(getattr(tensor_type, "elem_type", 0) or 0)
    dtype = _DTYPE_NAMES.get(onnx.TensorProto.DataType.Name(dtype_code)) if dtype_code else None
    shape: list[int] = []
    try:
        dims = tensor_type.shape.dim
    except Exception:  # pragma: no cover
        return None, dtype
    for dim in dims:
        if dim.HasField("dim_value"):
            shape.append(int(dim.dim_value))
        else:
            shape.append(-1)
    return (shape or None), dtype


def _initializer_info(onnx: Any, initializer: Any) -> dict[str, Any]:
    dims = [int(dim) for dim in getattr(initializer, "dims", []) or []]
    dtype_code = int(getattr(initializer, "data_type", 0) or 0)
    dtype = _DTYPE_NAMES.get(onnx.TensorProto.DataType.Name(dtype_code)) if dtype_code else None
    return {"shape": dims, "numel": shape_numel(dims) or 0, "dtype": dtype}


def _shape_of(value_info_map: dict[str, dict[str, Any]], name: str) -> list[int] | None:
    info = value_info_map.get(name)
    if not info:
        return None
    return info.get("shape")


def _numel(info: dict[str, Any] | None) -> int:
    if not info:
        return 0
    return int(info.get("numel") or 0)


def _initializer_of(init_info: dict[str, dict[str, Any]], name: str) -> dict[str, Any] | None:
    return init_info.get(name)


def _paths_from_name(name: str) -> list[str]:
    cleaned = str(name or "").strip().strip("/")
    parts = [part for part in cleaned.replace("\\", "/").replace(".", "/").split("/") if part]
    # PyTorch's legacy ONNX exporter repeats the top-level module name
    # (``/features/features/0/Conv``); collapse consecutive duplicates.
    collapsed: list[str] = []
    for part in parts:
        if collapsed and collapsed[-1] == part:
            continue
        collapsed.append(part)
    return collapsed


# ---------------------------------------------------------------------------
# per-op payload
# ---------------------------------------------------------------------------

def _op_payload(
    onnx: Any,
    node: Any,
    attrs: dict[str, Any],
    value_info_map: dict[str, dict[str, Any]],
    init_info: dict[str, dict[str, Any]],
    input_names: Sequence[str],
    in_shapes: Sequence[list[int]],
    out_shapes: Sequence[list[int]],
) -> tuple[str, dict[str, Any], int, int, str]:
    """Return ``(kind, params, numParams, flops, displayOp)`` for one ONNX node."""
    op_type = str(node.op_type)
    kind = _KIND_BY_OP.get(op_type, "custom")
    params: dict[str, Any] = {}
    num_params = 0
    flops = 0
    weights = [_initializer_of(init_info, name) for name in input_names]
    out_shape = out_shapes[0] if out_shapes else None
    in_shape = in_shapes[0] if in_shapes else None
    out_elems = shape_numel(out_shape) or 0
    in_elems = shape_numel(in_shape) or 0

    if op_type in ("Conv", "ConvTranspose"):
        weight = weights[1] if len(weights) > 1 else None
        bias = weights[2] if len(weights) > 2 else None
        weight_shape = list(weight["shape"]) if weight else []
        if op_type == "Conv":
            out_channels = int(weight_shape[0]) if weight_shape else (out_shape[1] if out_shape and len(out_shape) > 1 else 0)
            kernel = list(weight_shape[2:]) or list(attrs.get("kernel_shape") or [])
        else:
            groups = max(1, int(attrs.get("group", 1) or 1))
            out_channels = int(weight_shape[1]) * groups if len(weight_shape) > 1 else (
                out_shape[1] if out_shape and len(out_shape) > 1 else 0
            )
            kernel = list(weight_shape[2:]) or list(attrs.get("kernel_shape") or [])
        rank = len(kernel) or max(1, len(in_shape or []) - 2)
        groups = max(1, int(attrs.get("group", 1) or 1))
        in_channels = int(in_shape[1]) if in_shape and len(in_shape) > 1 and in_shape[1] >= 0 else (
            int(weight_shape[1]) * groups if len(weight_shape) > 1 else 0
        )
        pads = list(attrs.get("pads") or [])
        pads = pads[:rank] if pads else [0] * rank
        params = {
            "inChannels": in_channels,
            "outChannels": out_channels,
            "kernelSize": [int(value) for value in kernel],
            "stride": [int(value) for value in (attrs.get("strides") or [1] * rank)],
            "padding": [int(value) for value in pads],
            "dilation": [int(value) for value in (attrs.get("dilations") or [1] * rank)],
            "groups": groups,
            "bias": bias is not None,
        }
        num_params = _numel(weight) + _numel(bias)
        if out_shape is not None:
            flops = ops.conv_flops(params, list(out_shape))
    elif op_type == "Gemm":
        weight = weights[1] if len(weights) > 1 else None
        bias = weights[2] if len(weights) > 2 else None
        weight_shape = list(weight["shape"]) if weight else []
        trans_b = int(attrs.get("transB", 0) or 0)
        if len(weight_shape) == 2:
            in_features, out_features = (weight_shape[1], weight_shape[0]) if trans_b else (weight_shape[:2])
        else:
            in_features = out_features = 0
        if out_shape and len(out_shape) > 0 and out_shape[-1] >= 0:
            out_features = int(out_shape[-1])
        params = {"inFeatures": int(in_features), "outFeatures": int(out_features), "bias": bias is not None}
        num_params = _numel(weight) + _numel(bias)
        flops = ops.linear_flops(params, list(out_shape or []))
    elif op_type == "MatMul":
        weight = weights[0] if len(weights) > 0 and input_names and input_names[0] in init_info else (
            weights[1] if len(weights) > 1 and len(input_names) > 1 and input_names[1] in init_info else None
        )
        params = {}
        if len(in_shapes) >= 2:
            left, right = in_shapes[0], in_shapes[1]
            if len(left) >= 2 and len(right) >= 2:
                params = {"inFeatures": int(left[-1]), "outFeatures": int(right[-1]), "bias": False}
                flops = ops.matmul_flops([left, right])
        num_params = _numel(weight)
    elif op_type in ("BatchNormalization",):
        scale = weights[1] if len(weights) > 1 else None
        bias = weights[2] if len(weights) > 2 else None
        channels = 0
        if scale and scale["shape"]:
            channels = int(scale["shape"][0])
        elif in_shape and len(in_shape) > 1 and in_shape[1] >= 0:
            channels = int(in_shape[1])
        params = {"normalizedShape": [channels] if channels else [-1], "eps": float(attrs.get("epsilon", 1e-5)), "affine": bias is not None}
        # ONNX keeps running_mean / running_var as initializers, but NetPrism
        # follows the PyTorch convention: only scale + offset are parameters.
        num_params = _numel(scale) + _numel(bias)
        flops = int(out_elems * 4)
    elif op_type == "LayerNormalization":
        scale = weights[1] if len(weights) > 1 else None
        bias = weights[2] if len(weights) > 2 else None
        axis = int(attrs.get("axis", -1) or -1)
        normalized: list[int] = []
        if scale and scale["shape"]:
            normalized = list(scale["shape"])
        elif in_shape:
            normalized = list(in_shape[axis:])
        params = {"normalizedShape": normalized or [-1], "eps": float(attrs.get("epsilon", 1e-5)), "affine": bias is not None}
        num_params = _numel(scale) + _numel(bias)
        flops = int(out_elems * 4)
    elif kind == "norm":
        scale = weights[1] if len(weights) > 1 else None
        bias = weights[2] if len(weights) > 2 else None
        channels = int(in_shape[1]) if in_shape and len(in_shape) > 1 and in_shape[1] >= 0 else 0
        params = {"normalizedShape": [channels] if channels else [-1], "eps": float(attrs.get("epsilon", 1e-5)), "affine": True}
        num_params = _numel(scale) + _numel(bias)
        flops = int(out_elems * 4)
    elif kind == "pool":
        lowered = op_type.lower()
        params = {"type": "max" if "max" in lowered else ("adaptive" if "global" in lowered else "avg")}
        if "global" in lowered:
            params["outputSize"] = [1] * max(1, len(in_shape or []) - 2)
        else:
            rank = max(1, len(in_shape or []) - 2)
            kernel = list(attrs.get("kernel_shape") or [1] * rank)
            params["kernelSize"] = [int(value) for value in kernel]
            params["stride"] = [int(value) for value in (attrs.get("strides") or kernel)]
            pads = list(attrs.get("pads") or [])
            params["padding"] = [int(value) for value in (pads[: len(kernel)] if pads else [0] * len(kernel))]
        flops = int(in_elems * 1)
    elif op_type == "Flatten":
        params = {"startDim": int(attrs.get("axis", 1) or 1)}
    elif op_type in ("Reshape", "Expand"):
        if len(input_names) > 1:
            shape_info = init_info.get(input_names[1])
            if shape_info is not None:
                # The shape values themselves are not decoded (avoids numpy import);
                # the output shape from shape inference is exact.
                params["shape"] = list(out_shape or [])
        flops = 0
    elif op_type == "Concat":
        params = {"dim": int(attrs.get("axis", 0) or 0)}
        flops = int(out_elems)
    elif op_type in ("Squeeze", "Unsqueeze", "Transpose"):
        axes = attrs.get("axes") or attrs.get("perm")
        if axes is not None:
            params["dims"] = [int(value) for value in axes]
    elif op_type in ("Add", "Sum"):
        flops = int(out_elems)
    elif kind == "attention":
        flops = 0
    elif kind == "rnn":
        num_params = sum(_numel(weight) for weight in weights)
        flops = 0
    elif kind in ("activation", "softmax", "dropout"):
        flops = 0 if kind == "dropout" else int(out_elems * (4 if kind == "softmax" else 1))
    elif kind in ("flatten", "reshape", "embedding", "upsample", "concat"):
        if kind == "concat":
            flops = int(out_elems)
        if kind == "upsample":
            flops = int(out_elems)
        if kind == "embedding":
            num_params = _numel(weights[0]) if weights else 0
    elif kind == "add":
        flops = int(out_elems)

    return kind, params, int(num_params), int(flops), op_type


# ---------------------------------------------------------------------------
# main analysis
# ---------------------------------------------------------------------------

def _infer_shapes(onnx: Any, model: Any) -> tuple[Any, str | None]:
    try:
        inferred = onnx.shape_inference.infer_shapes(model)
    except Exception as first_error:  # noqa: BLE001 - best effort
        try:
            inferred = onnx.shape_inference.infer_shapes(model, strict_mode=False)
        except Exception:  # noqa: BLE001
            return model, f"ONNX shape inference failed ({type(first_error).__name__}: {first_error}); shapes use -1 where unknown."
    return inferred, None


def analyze_onnx(path: str | Path, *, max_nodes: int = 20000) -> list[dict[str, Any]]:
    """Analyze one ONNX model file and return a single graph document."""
    file_path = Path(path).expanduser()
    onnx = _import_onnx(file_path)
    try:
        with open(file_path, "rb") as handle:
            model = onnx.load(handle)
    except Exception as exc:  # noqa: BLE001 - any parse failure is an AnalyzeError
        raise AnalyzeError(f"not a valid ONNX model ({type(exc).__name__}: {exc})", file=file_path) from exc

    warnings_list: list[str] = []
    inferred, warning = _infer_shapes(onnx, model)
    if warning:
        warnings_list.append(warning)
    graph = getattr(inferred, "graph", None) or model.graph

    value_info_map: dict[str, dict[str, Any]] = {}

    def register(value_info: Any) -> None:
        try:
            name = value_info.name
        except Exception:  # pragma: no cover
            return
        shape, dtype = _tensor_type(onnx, value_info)
        if name not in value_info_map:
            value_info_map[name] = {"shape": shape, "dtype": dtype}

    for value_info in list(graph.input) + list(graph.value_info) + list(graph.output):
        register(value_info)

    init_info: dict[str, dict[str, Any]] = {}
    for initializer in graph.initializer:
        init_info[initializer.name] = _initializer_info(onnx, initializer)
        # Initializers used as weights are not data inputs; make their shape
        # available for op parameter derivation as well.
        if initializer.name not in value_info_map:
            value_info_map[initializer.name] = {
                "shape": list(initializer.dims),
                "dtype": init_info[initializer.name]["dtype"],
            }

    # ``do_constant_folding=False`` exports rename weight tensors through
    # ``Identity`` nodes (a BatchNorm's scale becomes ``Identity(conv.weight)``).
    # Resolve those aliases so parameters keep their real shape and skip the
    # plumbing nodes from the graph.
    alias_info: dict[str, dict[str, Any]] = {}
    skipped_nodes: set[int] = set()
    constant_outputs: set[str] = set()
    for index, onnx_node in enumerate(graph.node):
        if str(onnx_node.op_type) == "Identity" and len(onnx_node.input) == 1:
            source = str(onnx_node.input[0])
            info = init_info.get(source) or alias_info.get(source)
            if info is not None and len(onnx_node.output) == 1:
                alias_info[str(onnx_node.output[0])] = info
                skipped_nodes.add(index)
        elif str(onnx_node.op_type) == "Constant":
            skipped_nodes.add(index)
            constant_outputs.update(str(name) for name in onnx_node.output)
    effective_init = dict(init_info)
    effective_init.update(alias_info)

    # Graph inputs (weights excluded).
    input_specs: list[dict[str, Any]] = []
    input_names: list[str] = []
    for value_info in graph.input:
        name = value_info.name
        if name in init_info:
            continue
        shape, dtype = _tensor_type(onnx, value_info)
        input_specs.append({"name": name, "shape": to_shape(shape), "dtype": dtype or "float32"})
        input_names.append(name)

    if len(graph.node) + len(input_names) + len(graph.output) > max_nodes:
        raise AnalyzeError(
            f"model has {len(graph.node)} ONNX nodes, above the max_nodes limit ({max_nodes})",
            file=file_path,
        )

    nodes: list[dict[str, Any]] = []
    producer: dict[str, int] = {}
    name_counts: dict[str, int] = {}
    edges: list[dict[str, Any]] = []
    unknown_shape_count = 0

    for name in input_names:
        shape = _shape_of(value_info_map, name) or [-1]
        node = make_boundary_node(name=name, op="Input", kind="input", out_shapes=[shape])
        node["id"] = f"n{len(nodes)}"
        producer[name] = len(nodes)
        nodes.append(node)

    for index, onnx_node in enumerate(graph.node):
        if index in skipped_nodes:
            continue
        op_type = str(onnx_node.op_type)
        attrs = _attrs(onnx_node, onnx)
        raw_name = str(onnx_node.name or "").strip()
        display_name = raw_name or f"{op_type}_{index}"
        scoped = _paths_from_name(raw_name)
        module_path = ".".join(scoped[:-1]) if len(scoped) >= 2 else ""
        display_name = ".".join(scoped) if scoped else display_name
        base_name = module_path or display_name
        count = name_counts.get(base_name, 0) + 1
        name_counts[base_name] = count
        if count > 1:
            display_name = f"{display_name}#{count}"

        node_inputs = [str(name) for name in onnx_node.input if name]
        node_outputs = [str(name) for name in onnx_node.output if name]
        activation_inputs = [
            name for name in node_inputs if name not in effective_init and name not in constant_outputs
        ]
        in_shapes: list[list[int]] = []
        for name in activation_inputs:
            shape = _shape_of(value_info_map, name)
            if shape is None:
                shape = [-1]
                unknown_shape_count += 1
            in_shapes.append(shape)
        out_shapes: list[list[int]] = []
        for name in node_outputs:
            shape = _shape_of(value_info_map, name)
            if shape is None:
                shape = [-1]
                unknown_shape_count += 1
            out_shapes.append(shape)

        kind, params, num_params, flops, display_op = _op_payload(
            onnx,
            onnx_node,
            attrs,
            value_info_map,
            effective_init,
            node_inputs,
            in_shapes,
            out_shapes,
        )
        tags = ops.tags_for(kind, display_op, params)
        meta = {
            "op_type": op_type,
            "domain": str(onnx_node.domain or "ai.onnx") or "ai.onnx",
        }
        node = make_node(
            name=display_name,
            op=display_op,
            kind=kind,
            module_path=module_path,
            in_shapes=in_shapes,
            out_shapes=out_shapes,
            params=params,
            num_params=num_params,
            flops=flops,
            tags=tags,
            meta=meta,
        )
        node["id"] = f"n{len(nodes)}"
        node_index = len(nodes)
        nodes.append(node)

        seen_sources: set[int] = set()
        for input_name in node_inputs:
            source_index = producer.get(input_name)
            if source_index is None:
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
        for output_name in node_outputs:
            producer[output_name] = node_index

    # Graph outputs.
    output_specs: list[dict[str, Any]] = []
    output_nodes: list[dict[str, Any]] = []
    for index, value_info in enumerate(graph.output):
        name = value_info.name
        shape, dtype = _tensor_type(onnx, value_info)
        shape = to_shape(shape)
        spec_name = "output" if index == 0 else f"output_{index + 1}"
        output_specs.append({"name": spec_name, "shape": shape, "dtype": dtype or "float32"})
        output_node = make_boundary_node(name=spec_name, op="Output", kind="output", in_shapes=[shape], out_shapes=[shape])
        output_node["id"] = f"n{len(nodes) + len(output_nodes)}"
        output_nodes.append(output_node)
        source_index = producer.get(name)
        if source_index is not None:
            edges.append(
                make_edge(
                    source=nodes[source_index]["id"],
                    target=output_node["id"],
                    shapes=nodes[source_index]["outShapes"][:1],
                    kind="data",
                )
            )
    nodes.extend(output_nodes)

    total_params = sum(int(node.get("numParams", 0)) for node in nodes)
    total_flops = sum(int(node.get("flops", 0)) for node in nodes)
    if unknown_shape_count:
        warnings_list.append(
            f"shape inference did not provide shapes for {unknown_shape_count} tensor(s); filled with -1."
        )
    if any(node.get("kind") in ("attention", "rnn", "custom", "unknown") for node in nodes):
        warnings_list.append("FLOPs for attention / recurrent / custom ONNX operators are estimates.")

    opset = []
    for item in getattr(model, "opset_import", []) or []:
        opset.append({"domain": str(item.domain or "ai.onnx") or "ai.onnx", "version": int(item.version)})
    model_name = file_path.stem
    graph_name = str(getattr(graph, "name", "") or "")
    meta = {
        "format": "onnx",
        "graphName": graph_name,
        "opset": opset,
        "irVersion": int(getattr(model, "ir_version", 0) or 0),
        "producer": str(getattr(model, "producer_name", "") or ""),
        "producerVersion": str(getattr(model, "producer_version", "") or ""),
        "shapeInference": warning is None,
    }
    flops_approx = bool(unknown_shape_count) or any(
        node.get("kind") in ("attention", "rnn", "custom", "unknown", "block") for node in nodes
    )
    graph_doc = assemble_graph(
        path=file_path,
        framework="onnx",
        mode="onnx",
        model_name=model_name,
        class_name=None,
        nodes=nodes,
        edges=edges,
        input_specs=input_specs,
        output_specs=output_specs or [{"name": "output", "shape": [-1], "dtype": "float32"}],
        warnings=warnings_list,
        total_params=total_params,
        trainable_params=total_params,
        total_flops=total_flops,
        flops_approx=flops_approx,
        meta=meta,
    )
    return [graph_doc]


__all__ = ["analyze_onnx"]
