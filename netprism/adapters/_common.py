"""Shared helpers for the optional model-format adapters.

This module is **dependency free at import time**: it never imports ``torch``,
``onnx`` or ``keras``.  The format specific adapters do their own lazy imports
and raise :class:`~netprism.core.errors.AnalyzeError` with an install hint when
the optional package is missing.

Everything that ends up in a graph document is plain JSON: no numpy / torch
scalars, only ``str`` / ``int`` / ``float`` / ``bool`` / ``None`` / lists /
dicts (exactly what :func:`netprism.schema.validate_graph` accepts).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .. import schema
from ..core.errors import AnalyzeError
from ..core.groups import humanize, split_trailing_number

Shape = list[int]

#: Human readable suffix labels (used by ``netprism.adapters``).
FORMAT_LABELS: dict[str, str] = {
    ".onnx": "ONNX",
    ".pt": "TorchScript",
    ".pth": "TorchScript",
    ".keras": "Keras v3",
}

#: Required install hint for every optional backend.
INSTALL_HINTS: dict[str, str] = {
    "onnx": "onnx",
    "torch": "torch",
    "keras": "keras",
}


# ---------------------------------------------------------------------------
# paths / json helpers
# ---------------------------------------------------------------------------

def display_path(path: Path) -> str:
    """Repo-relative POSIX path when possible, absolute POSIX path otherwise."""
    resolved = Path(path).expanduser().resolve()
    try:
        return resolved.relative_to(schema.REPO_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def relative_path(path: Path) -> str:
    """Short relative label stored in ``model.source.relPath``."""
    return Path(path).name


def to_shape(value: Any, *, fallback: Sequence[int] = (-1,)) -> Shape:
    """Convert ``torch.Size`` / tuple / list / ``None`` to a JSON-safe shape.

    Unknown or dynamic dimensions become ``-1``; an empty shape becomes the
    ``fallback`` (schema shapes must be non-empty lists).
    """
    if value is None:
        return [int(dim) for dim in fallback]
    try:
        dims = list(value)
    except TypeError:
        return [int(dim) for dim in fallback]
    out: Shape = []
    for dim in dims:
        if isinstance(dim, bool):
            out.append(-1)
            continue
        try:
            number = int(dim)
        except (TypeError, ValueError):
            number = -1
        out.append(number if number >= 0 else -1)
    return out or [int(dim) for dim in fallback]


def shape_numel(shape: Sequence[int] | None) -> int | None:
    """Number of elements or ``None`` when the shape is dynamic/empty."""
    if not shape:
        return None
    total = 1
    for dim in shape:
        if int(dim) < 0:
            return None
        total *= int(dim)
    return total


def clean_scalar(value: Any) -> Any:
    """Best effort JSON scalar conversion (numpy/torch scalars included)."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [clean_scalar(item) for item in value]
    if isinstance(value, dict):
        return {str(key): clean_scalar(item) for key, item in value.items()}
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return int(number) if number.is_integer() else number


# ---------------------------------------------------------------------------
# node / edge payloads
# ---------------------------------------------------------------------------

def make_node(
    *,
    name: str,
    op: str,
    kind: str,
    module_path: str = "",
    depth: int | None = None,
    in_shapes: Iterable[Sequence[int]] = (),
    out_shapes: Iterable[Sequence[int]] = (),
    params: Mapping[str, Any] | None = None,
    num_params: int = 0,
    flops: int = 0,
    tags: Iterable[str] = (),
    group_id: str | None = None,
    source: Mapping[str, Any] | None = None,
    notes: Iterable[str] = (),
    meta: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one node object satisfying the schema contract."""
    path = str(module_path or "")
    if depth is None:
        segments = [seg for seg in re.split(r"[./]", path) if seg]
        depth = max(0, len(segments) - 1)
    payload: dict[str, Any] = {
        "name": str(name),
        "op": str(op),
        "kind": str(kind),
        "modulePath": path,
        "depth": int(depth),
        "inShapes": [to_shape(shape) for shape in in_shapes],
        "outShapes": [to_shape(shape) for shape in out_shapes],
        "params": {str(key): clean_scalar(value) for key, value in (params or {}).items()},
        "numParams": int(num_params),
        "flops": int(flops),
        "tags": list(dict.fromkeys(str(tag) for tag in tags)),
        "groupId": group_id,
        "source": dict(source) if source else None,
        "notes": [str(note) for note in notes],
    }
    if meta:
        payload["meta"] = {str(key): clean_scalar(value) for key, value in meta.items()}
    return payload


def make_boundary_node(
    *,
    name: str,
    op: str,
    kind: str,
    in_shapes: Iterable[Sequence[int]] = (),
    out_shapes: Iterable[Sequence[int]] = (),
) -> dict[str, Any]:
    """Input / output boundary node (``depth=0``, no group, ``source=null``)."""
    return make_node(
        name=name,
        op=op,
        kind=kind,
        module_path="",
        depth=0,
        in_shapes=in_shapes,
        out_shapes=out_shapes,
        tags=("boundary",),
    )


def make_edge(
    *,
    source: str,
    target: str,
    shapes: Iterable[Sequence[int]] = (),
    kind: str = "data",
    label: str = "",
) -> dict[str, Any]:
    return {
        "source": str(source),
        "target": str(target),
        "kind": kind,
        "shapes": [to_shape(shape) for shape in shapes],
        "label": str(label),
    }


# ---------------------------------------------------------------------------
# group construction
# ---------------------------------------------------------------------------

_GROUP_SEGMENT_RE = re.compile(r"[./]")

_GENERIC_SEGMENTS = frozenset(
    {
        "",
        "block",
        "blocks",
        "layer",
        "layers",
        "stage",
        "stages",
        "unit",
        "units",
        "module",
        "modules",
        "net",
        "network",
        "seq",
        "sequential",
    }
)

_KIND_NOUN = {
    "conv_block": "Conv Block",
    "residual_block": "Residual Block",
    "transformer_block": "Transformer Block",
    "attention_block": "Attention Block",
    "ffn_block": "Feed Forward",
    "mlp": "MLP",
    "unet_block": "UNet Block",
    "encoder": "Encoder",
    "decoder": "Decoder",
}

_STRUCTURAL_KINDS = frozenset(_KIND_NOUN)


def _segments(path: str) -> list[str]:
    return [seg for seg in _GROUP_SEGMENT_RE.split(str(path or "")) if seg]


def _join_segments(segments: Sequence[str]) -> str:
    return ".".join(segments)


def _group_label(path: str, kind: str, *, model_name: str) -> str:
    """Readable, format independent group label."""
    if not path:
        return humanize(model_name) or "Model"
    segments = path.split(".")
    last = segments[-1]
    base, number = split_trailing_number(last)
    if not base and number is not None:
        parent = segments[-2] if len(segments) >= 2 else ""
        parent_base, _parent_number = split_trailing_number(parent)
        noun = _KIND_NOUN.get(kind)
        if noun:
            return f"{noun} {number + 1}"
        if parent_base and parent_base.lower() not in _GENERIC_SEGMENTS:
            return f"{humanize(parent_base)} {number + 1}"
        return f"Block {number + 1}"
    label = humanize(base if number is not None else last)
    if not label:
        label = "Block"
    if number is not None:
        return f"{label} {number}"
    return label


def _classify_group(kinds: Sequence[str]) -> str:
    counts: dict[str, int] = {}
    for kind in kinds:
        counts[kind] = counts.get(kind, 0) + 1
    if counts.get("attention"):
        if counts.get("linear", 0) >= 2 and counts.get("norm", 0) >= 1:
            return "transformer_block"
        return "attention_block"
    if counts.get("add"):
        return "residual_block"
    if counts.get("conv", 0) >= 2 and (counts.get("norm", 0) + counts.get("activation", 0)) >= 1:
        return "conv_block"
    if counts.get("conv", 0) >= 2:
        return "conv_block"
    if counts.get("linear", 0) >= 2 or counts.get("ffn", 0):
        return "mlp"
    if counts.get("conv") or counts.get("deconv"):
        return "module"
    if counts.get("linear"):
        return "module"
    return "module"


def build_groups(
    module_paths: Sequence[str | None],
    nodes: Sequence[dict[str, Any]],
    *,
    model_name: str,
) -> list[dict[str, Any]]:
    """Derive the schema group tree from node ``modulePath`` scopes.

    The rule matches the PyTorch core: every module scope prefix becomes a
    group, a node belongs to the *last* prefix (its direct container), group
    memberships are non-overlapping and boundary nodes stay ungrouped.
    """
    root_id = "g0"
    root: dict[str, Any] = {
        "id": root_id,
        "name": str(model_name) or "Model",
        "label": humanize(model_name) or "Model",
        "kind": "model",
        "parentId": None,
        "depth": 0,
        "nodeIds": [],
        "childGroupIds": [],
        "numParams": 0,
        "flops": 0,
        "source": None,
    }

    direct_group: list[str | None] = [None] * len(nodes)
    group_paths: set[str] = set()
    for index, raw_path in enumerate(module_paths):
        node = nodes[index]
        if node.get("kind") in ("input", "output") or not raw_path:
            continue
        segments = _segments(str(raw_path))
        ancestors = [_join_segments(segments[:depth]) for depth in range(1, len(segments))]
        if not ancestors:
            continue
        group_paths.update(ancestors)
        direct_group[index] = ancestors[-1]

    if not group_paths:
        root["nodeIds"] = [
            node["id"] for index, node in enumerate(nodes) if node.get("kind") not in ("input", "output")
        ]
        for node in nodes:
            if node.get("kind") not in ("input", "output"):
                node["groupId"] = root_id
        root["numParams"] = sum(int(node.get("numParams", 0)) for node in nodes)
        root["flops"] = sum(int(node.get("flops", 0)) for node in nodes)
        return [root]

    ordered_paths = sorted(group_paths, key=lambda item: (item.count("."), item))
    id_for_path = {path: f"g{index + 1}" for index, path in enumerate(ordered_paths)}

    # Direct members + full subtree membership (params / flops totals).
    direct_indices: dict[str, list[int]] = {path: [] for path in ordered_paths}
    for index, path in enumerate(direct_group):
        if path is not None:
            direct_indices[path].append(index)

    parent_of_path: dict[str, str] = {}
    for path in ordered_paths:
        parent = path.rsplit(".", 1)[0] if "." in path else ""
        while parent and parent not in group_paths:
            parent = parent.rsplit(".", 1)[0] if "." in parent else ""
        parent_of_path[path] = parent

    groups: list[dict[str, Any]] = [root]
    group_by_path: dict[str, dict[str, Any]] = {}
    for path in ordered_paths:
        direct = direct_indices[path]
        # Subtree membership: direct nodes of this group and of every child.
        subtree: list[int] = []
        for index, node_path in enumerate(direct_group):
            if node_path and (node_path == path or node_path.startswith(path + ".")):
                subtree.append(index)
        parent_path = parent_of_path.get(path, "")
        parent_group = group_by_path.get(parent_path)
        kinds = [nodes[index]["kind"] for index in (direct or subtree)]
        group_kind = _classify_group(kinds)
        group = {
            "id": id_for_path[path],
            "name": path,
            "label": _group_label(path, group_kind, model_name=model_name),
            "kind": group_kind,
            "parentId": parent_group["id"] if parent_group is not None else root_id,
            "depth": (parent_group["depth"] + 1) if parent_group is not None else 1,
            "nodeIds": [nodes[index]["id"] for index in direct],
            "childGroupIds": [
                id_for_path[child]
                for child in ordered_paths
                if parent_of_path.get(child) == path
            ],
            "numParams": sum(int(nodes[index].get("numParams", 0)) for index in subtree),
            "flops": sum(int(nodes[index].get("flops", 0)) for index in subtree),
            "source": None,
        }
        groups.append(group)
        group_by_path[path] = group

    for index, path in enumerate(direct_group):
        if path is None:
            continue
        group_id = id_for_path[path]
        nodes[index]["groupId"] = group_id

    # Top-level nodes without a natural container live in the model group.
    root_node_ids: list[str] = list(root["nodeIds"])
    for index, node in enumerate(nodes):
        if node.get("kind") in ("input", "output"):
            continue
        if direct_group[index] is not None:
            continue
        node["groupId"] = root_id
        root_node_ids.append(node["id"])
    root["nodeIds"] = root_node_ids
    root["childGroupIds"] = [
        id_for_path[path] for path in ordered_paths if parent_of_path.get(path) == ""
    ]
    root["numParams"] = sum(int(node.get("numParams", 0)) for node in nodes)
    root["flops"] = sum(int(node.get("flops", 0)) for node in nodes)

    _disambiguate_labels(groups)
    return groups


def _disambiguate_labels(groups: Sequence[dict[str, Any]]) -> None:
    """Prefix duplicated group labels with their parent label."""
    root = groups[0]
    by_id = {group["id"]: group for group in groups}
    counts: dict[str, int] = {}
    for group in groups[1:]:
        counts[group["label"]] = counts.get(group["label"], 0) + 1
    used: set[str] = {root["label"]}
    for group in sorted(groups[1:], key=lambda item: (item["depth"], item["name"])):
        label = group["label"]
        if counts.get(label, 0) > 1:
            parent = by_id.get(group.get("parentId") or "")
            if parent is not None and parent["label"] != label:
                candidate = f"{parent['label']} · {label}"
                if candidate not in used:
                    label = candidate
        if label in used:
            suffix = 2
            while f"{label} ({suffix})" in used:
                suffix += 1
            label = f"{label} ({suffix})"
        group["label"] = label
        used.add(label)


# ---------------------------------------------------------------------------
# graph assembly + validation
# ---------------------------------------------------------------------------

def assemble_graph(
    *,
    path: Path,
    framework: str,
    mode: str,
    model_name: str,
    class_name: str | None,
    nodes: Sequence[dict[str, Any]],
    edges: Sequence[dict[str, Any]],
    input_specs: Sequence[Mapping[str, Any]],
    output_specs: Sequence[Mapping[str, Any]],
    warnings: Sequence[str] = (),
    total_params: int | None = None,
    trainable_params: int | None = None,
    total_flops: int | None = None,
    flops_approx: bool = False,
    meta: Mapping[str, Any] | None = None,
    module_paths: Sequence[str | None] | None = None,
) -> dict[str, Any]:
    """Assemble and validate one graph document for a non-PyTorch format."""
    node_list = [dict(node) for node in nodes]
    for index, node in enumerate(node_list):
        node.setdefault("id", f"n{index}")

    edge_list = []
    for index, edge in enumerate(edges):
        payload = dict(edge)
        payload["id"] = f"e{index}"
        edge_list.append(payload)

    scopes = list(module_paths) if module_paths is not None else [
        node.get("modulePath") for node in node_list
    ]
    groups = build_groups(scopes, node_list, model_name=model_name)

    model_params = (
        int(total_params)
        if total_params is not None
        else sum(int(node.get("numParams", 0)) for node in node_list)
    )
    model_trainable = int(trainable_params) if trainable_params is not None else model_params
    model_flops = (
        int(total_flops)
        if total_flops is not None
        else sum(int(node.get("flops", 0)) for node in node_list)
    )
    num_layers = sum(1 for node in node_list if node.get("kind") not in ("input", "output"))

    model: dict[str, Any] = {
        "name": str(model_name),
        "framework": str(framework),
        "className": class_name,
        "mode": str(mode),
        "source": {"file": display_path(path), "relPath": relative_path(path)},
        "inputSpec": [
            {
                "name": str(spec.get("name") or "input"),
                "shape": to_shape(spec.get("shape")),
                "dtype": str(spec.get("dtype") or "float32"),
            }
            for spec in input_specs
        ],
        "outputSpec": [
            {
                "name": str(spec.get("name") or "output"),
                "shape": to_shape(spec.get("shape")),
                "dtype": str(spec.get("dtype") or "float32"),
            }
            for spec in output_specs
        ],
        "totalParams": model_params,
        "trainableParams": model_trainable,
        "totalFlops": model_flops,
        "flopsApprox": bool(flops_approx),
        "numLayers": int(num_layers),
        "warnings": list(dict.fromkeys(str(warning) for warning in warnings)),
    }
    if meta:
        model["meta"] = {str(key): clean_scalar(value) for key, value in meta.items()}

    graph = {
        "schemaVersion": schema.SCHEMA_VERSION,
        "generator": {"name": schema.GENERATOR_NAME, "version": schema.GENERATOR_VERSION},
        "model": model,
        "nodes": node_list,
        "edges": edge_list,
        "groups": groups,
    }
    validate_graph_or_raise(graph, file=path)
    return graph


def validate_graph_or_raise(graph: Mapping[str, Any], *, file: Path | str) -> None:
    """Raise :class:`AnalyzeError` with a readable preview on schema errors."""
    errors = schema.validate_graph(graph)
    if not errors:
        return
    preview = "\n  - ".join(errors[:10])
    more = "" if len(errors) <= 10 else f"\n  ... and {len(errors) - 10} more"
    raise AnalyzeError(
        f"internal error: generated graph failed schema validation:\n  - {preview}{more}",
        file=file,
    )


__all__ = [
    "FORMAT_LABELS",
    "INSTALL_HINTS",
    "Shape",
    "assemble_graph",
    "build_groups",
    "clean_scalar",
    "display_path",
    "make_boundary_node",
    "make_edge",
    "make_node",
    "relative_path",
    "shape_numel",
    "to_shape",
    "validate_graph_or_raise",
]
