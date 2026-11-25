"""NetPrism data contract: shared constants and dependency-free validation.

The canonical machine-readable definition lives in ``schema/graph.schema.json``.
This module duplicates the important rules in pure Python so the parser, CLI and
tests can validate graphs without requiring ``jsonschema`` (which is only a dev
dependency).  Keep both files in sync.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.0"
GENERATOR_NAME = "netprism"
GENERATOR_VERSION = "0.1.0"

KINDS: tuple[str, ...] = (
    "input",
    "output",
    "conv",
    "deconv",
    "linear",
    "mlp",
    "norm",
    "activation",
    "pool",
    "dropout",
    "flatten",
    "reshape",
    "attention",
    "ffn",
    "embedding",
    "positional",
    "add",
    "concat",
    "upsample",
    "rnn",
    "softmax",
    "block",
    "container",
    "custom",
    "unknown",
)

GROUP_KINDS: tuple[str, ...] = (
    "model",
    "sequential",
    "module",
    "conv_block",
    "residual_block",
    "attention_block",
    "transformer_block",
    "ffn_block",
    "mlp",
    "encoder",
    "decoder",
    "unet_block",
    "custom",
)

EDGE_KINDS: tuple[str, ...] = ("data", "skip", "residual", "attention", "control")

FRAMEWORKS: tuple[str, ...] = ("pytorch", "onnx", "keras", "torchscript", "unknown")
MODES: tuple[str, ...] = ("trace", "hybrid", "static", "onnx", "keras", "torchscript")

NODE_ID_RE = re.compile(r"^n\d+$")
EDGE_ID_RE = re.compile(r"^e\d+$")
GROUP_ID_RE = re.compile(r"^g\d+$")

_HERE = Path(__file__).resolve().parent
REPO_ROOT = _HERE.parent
SCHEMA_PATH = REPO_ROOT / "schema" / "graph.schema.json"
DEMO_DIR = REPO_ROOT / "examples" / "demo"
WEB_DIR = REPO_ROOT / "web"


class SchemaError(ValueError):
    """Raised when a graph document does not satisfy the contract."""


def load_json_schema() -> dict[str, Any]:
    """Return the canonical JSON Schema document."""
    with SCHEMA_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _err(errors: list[str], path: str, message: str) -> None:
    errors.append(f"{path}: {message}")


def _check_shape(errors: list[str], path: str, shape: Any) -> None:
    if not isinstance(shape, list) or not shape:
        _err(errors, path, "shape must be a non-empty list of integers")
        return
    for i, dim in enumerate(shape):
        if not isinstance(dim, int) or isinstance(dim, bool) or dim < -1:
            _err(errors, f"{path}[{i}]", f"invalid dimension {dim!r} (integer >= -1 required)")


def _check_shapes(errors: list[str], path: str, shapes: Any) -> None:
    if not isinstance(shapes, list):
        _err(errors, path, "must be a list of shapes")
        return
    for i, shape in enumerate(shapes):
        _check_shape(errors, f"{path}[{i}]", shape)


def _check_source(errors: list[str], path: str, source: Any) -> None:
    if source is None:
        return
    if not isinstance(source, dict):
        _err(errors, path, "must be an object or null")
        return
    for key in ("file", "relPath", "code"):
        if key in source and not isinstance(source[key], str):
            _err(errors, f"{path}.{key}", "must be a string")
    if "line" in source:
        line = source["line"]
        if not isinstance(line, int) or isinstance(line, bool) or line < 1:
            _err(errors, f"{path}.line", "must be a positive integer")


def _check_tensor_specs(errors: list[str], path: str, specs: Any) -> None:
    if not isinstance(specs, list):
        _err(errors, path, "must be a list")
        return
    for i, spec in enumerate(specs):
        loc = f"{path}[{i}]"
        if not isinstance(spec, dict):
            _err(errors, loc, "must be an object")
            continue
        for key in ("name", "dtype"):
            if not isinstance(spec.get(key), str):
                _err(errors, f"{loc}.{key}", "must be a string")
        _check_shape(errors, f"{loc}.shape", spec.get("shape"))


def validate_graph(graph: Any) -> list[str]:
    """Validate a graph document and return a list of human-readable errors.

    An empty list means the document satisfies the contract.
    """
    errors: list[str] = []
    if not isinstance(graph, dict):
        return ["graph: must be an object"]

    if graph.get("schemaVersion") != SCHEMA_VERSION:
        _err(errors, "schemaVersion", f"expected {SCHEMA_VERSION!r}, got {graph.get('schemaVersion')!r}")

    generator = graph.get("generator")
    if not isinstance(generator, dict) or not isinstance(generator.get("name"), str) or not isinstance(
        generator.get("version"), str
    ):
        _err(errors, "generator", "must be an object with string 'name' and 'version'")

    model = graph.get("model")
    if not isinstance(model, dict):
        _err(errors, "model", "must be an object")
    else:
        for key in ("name", "framework", "className", "mode", "source"):
            if key not in model:
                _err(errors, "model", f"missing required key {key!r}")
        if not isinstance(model.get("name"), str):
            _err(errors, "model.name", "must be a string")
        if model.get("framework") not in FRAMEWORKS:
            _err(errors, "model.framework", f"must be one of {FRAMEWORKS}")
        if model.get("mode") not in MODES:
            _err(errors, "model.mode", f"must be one of {MODES}")
        if model.get("className") is not None and not isinstance(model.get("className"), str):
            _err(errors, "model.className", "must be a string or null")
        src = model.get("source")
        if not isinstance(src, dict) or not isinstance(src.get("file"), str) or not isinstance(
            src.get("relPath"), str
        ):
            _err(errors, "model.source", "must be an object with string 'file' and 'relPath'")
        _check_tensor_specs(errors, "model.inputSpec", model.get("inputSpec"))
        _check_tensor_specs(errors, "model.outputSpec", model.get("outputSpec"))
        for key in ("totalParams", "trainableParams", "totalFlops", "numLayers"):
            value = model.get(key)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                _err(errors, f"model.{key}", "must be a non-negative integer")
        if not isinstance(model.get("flopsApprox"), bool):
            _err(errors, "model.flopsApprox", "must be a boolean")
        warnings = model.get("warnings")
        if not isinstance(warnings, list) or not all(isinstance(w, str) for w in warnings):
            _err(errors, "model.warnings", "must be a list of strings")
        for key in ("meta", "metrics"):
            if key in model and not isinstance(model[key], dict):
                _err(errors, f"model.{key}", "must be an object")

    nodes = graph.get("nodes")
    if not isinstance(nodes, list):
        _err(errors, "nodes", "must be a list")
        nodes = []
    node_ids: set[str] = set()
    for i, node in enumerate(nodes):
        loc = f"nodes[{i}]"
        if not isinstance(node, dict):
            _err(errors, loc, "must be an object")
            continue
        nid = node.get("id")
        if not isinstance(nid, str) or not NODE_ID_RE.match(nid):
            _err(errors, f"{loc}.id", f"invalid id {nid!r}")
        elif nid in node_ids:
            _err(errors, f"{loc}.id", f"duplicate node id {nid!r}")
        else:
            node_ids.add(nid)
        for key in ("name", "op", "modulePath"):
            if not isinstance(node.get(key), str):
                _err(errors, f"{loc}.{key}", "must be a string")
        if node.get("kind") not in KINDS:
            _err(errors, f"{loc}.kind", f"must be one of {KINDS}")
        depth = node.get("depth")
        if not isinstance(depth, int) or isinstance(depth, bool) or depth < 0:
            _err(errors, f"{loc}.depth", "must be a non-negative integer")
        _check_shapes(errors, f"{loc}.inShapes", node.get("inShapes"))
        _check_shapes(errors, f"{loc}.outShapes", node.get("outShapes"))
        if not isinstance(node.get("params"), dict):
            _err(errors, f"{loc}.params", "must be an object")
        for key in ("numParams", "flops"):
            value = node.get(key)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                _err(errors, f"{loc}.{key}", "must be a non-negative integer")
        tags = node.get("tags")
        if not isinstance(tags, list) or not all(isinstance(t, str) for t in tags):
            _err(errors, f"{loc}.tags", "must be a list of strings")
        group_id = node.get("groupId")
        if group_id is not None and not isinstance(group_id, str):
            _err(errors, f"{loc}.groupId", "must be a string or null")
        _check_source(errors, f"{loc}.source", node.get("source"))
        notes = node.get("notes")
        if not isinstance(notes, list) or not all(isinstance(n, str) for n in notes):
            _err(errors, f"{loc}.notes", "must be a list of strings")
        for key in ("meta", "metrics"):
            if key in node and not isinstance(node[key], dict):
                _err(errors, f"{loc}.{key}", "must be an object")

    edges = graph.get("edges")
    if not isinstance(edges, list):
        _err(errors, "edges", "must be a list")
        edges = []
    edge_ids: set[str] = set()
    for i, edge in enumerate(edges):
        loc = f"edges[{i}]"
        if not isinstance(edge, dict):
            _err(errors, loc, "must be an object")
            continue
        eid = edge.get("id")
        if not isinstance(eid, str) or not EDGE_ID_RE.match(eid):
            _err(errors, f"{loc}.id", f"invalid id {eid!r}")
        elif eid in edge_ids:
            _err(errors, f"{loc}.id", f"duplicate edge id {eid!r}")
        else:
            edge_ids.add(eid)
        for key in ("source", "target"):
            ref = edge.get(key)
            if not isinstance(ref, str):
                _err(errors, f"{loc}.{key}", "must be a string")
            elif ref not in node_ids:
                _err(errors, f"{loc}.{key}", f"unknown node {ref!r}")
        if edge.get("kind") not in EDGE_KINDS:
            _err(errors, f"{loc}.kind", f"must be one of {EDGE_KINDS}")
        _check_shapes(errors, f"{loc}.shapes", edge.get("shapes"))
        if not isinstance(edge.get("label"), str):
            _err(errors, f"{loc}.label", "must be a string")

    groups = graph.get("groups")
    if not isinstance(groups, list):
        _err(errors, "groups", "must be a list")
        groups = []
    group_map: dict[str, dict[str, Any]] = {}
    for i, group in enumerate(groups):
        loc = f"groups[{i}]"
        if not isinstance(group, dict):
            _err(errors, loc, "must be an object")
            continue
        gid = group.get("id")
        if not isinstance(gid, str) or not GROUP_ID_RE.match(gid):
            _err(errors, f"{loc}.id", f"invalid id {gid!r}")
        elif gid in group_map:
            _err(errors, f"{loc}.id", f"duplicate group id {gid!r}")
        else:
            group_map[gid] = group
        for key in ("name", "label"):
            if not isinstance(group.get(key), str):
                _err(errors, f"{loc}.{key}", "must be a string")
        if group.get("kind") not in GROUP_KINDS:
            _err(errors, f"{loc}.kind", f"must be one of {GROUP_KINDS}")
        parent = group.get("parentId")
        if parent is not None and not isinstance(parent, str):
            _err(errors, f"{loc}.parentId", "must be a string or null")
        depth = group.get("depth")
        if not isinstance(depth, int) or isinstance(depth, bool) or depth < 0:
            _err(errors, f"{loc}.depth", "must be a non-negative integer")
        for key in ("nodeIds", "childGroupIds"):
            values = group.get(key)
            if not isinstance(values, list) or not all(isinstance(v, str) for v in values):
                _err(errors, f"{loc}.{key}", "must be a list of strings")
        for key in ("numParams", "flops"):
            value = group.get(key)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                _err(errors, f"{loc}.{key}", "must be a non-negative integer")
        _check_source(errors, f"{loc}.source", group.get("source"))
        if "meta" in group and not isinstance(group["meta"], dict):
            _err(errors, f"{loc}.meta", "must be an object")

    # Cross-reference checks.
    parent_of: dict[str, str | None] = {}
    for gid, group in group_map.items():
        parent = group.get("parentId")
        parent_of[gid] = parent
        if parent is not None and parent not in group_map:
            _err(errors, f"group {gid}.parentId", f"unknown group {parent!r}")
        for child in group.get("childGroupIds") or []:
            if child not in group_map:
                _err(errors, f"group {gid}.childGroupIds", f"unknown group {child!r}")
            elif group_map[child].get("parentId") != gid:
                _err(errors, f"group {gid}.childGroupIds", f"{child!r} does not point back via parentId")
        for nid in group.get("nodeIds") or []:
            if nid not in node_ids:
                _err(errors, f"group {gid}.nodeIds", f"unknown node {nid!r}")
            else:
                node = next(n for n in nodes if n.get("id") == nid)
                if node.get("groupId") != gid:
                    _err(
                        errors,
                        f"group {gid}.nodeIds",
                        f"node {nid!r} has groupId {node.get('groupId')!r}",
                    )
    for node in nodes:
        gid = node.get("groupId")
        if gid is not None and gid not in group_map:
            _err(errors, f"node {node.get('id')}.groupId", f"unknown group {gid!r}")

    # Group hierarchy cycle check.
    for start in group_map:
        seen: set[str] = set()
        cur: str | None = start
        while cur is not None:
            if cur in seen:
                _err(errors, f"group {start}.parentId", "cycle in group hierarchy")
                break
            seen.add(cur)
            cur = parent_of.get(cur, None)

    return errors


def validate_graph_or_raise(graph: Any, *, where: str = "graph") -> None:
    errors = validate_graph(graph)
    if errors:
        preview = "\n  - ".join(errors[:12])
        more = "" if len(errors) <= 12 else f"\n  ... and {len(errors) - 12} more"
        raise SchemaError(f"{where} does not satisfy NetPrism schema v{SCHEMA_VERSION}:\n  - {preview}{more}")


def validate_index(index: Any) -> list[str]:
    """Validate a ``scan`` index document."""
    errors: list[str] = []
    if not isinstance(index, dict):
        return ["index: must be an object"]
    if index.get("schemaVersion") != SCHEMA_VERSION:
        _err(errors, "schemaVersion", f"expected {SCHEMA_VERSION!r}")
    if not isinstance(index.get("generatedAt"), str):
        _err(errors, "generatedAt", "must be an ISO-8601 string")
    if not isinstance(index.get("root"), str):
        _err(errors, "root", "must be a string")
    models = index.get("models")
    if not isinstance(models, list):
        _err(errors, "models", "must be a list")
        models = []
    seen: set[str] = set()
    for i, model in enumerate(models):
        loc = f"models[{i}]"
        if not isinstance(model, dict):
            _err(errors, loc, "must be an object")
            continue
        mid = model.get("id")
        if not isinstance(mid, str) or not mid:
            _err(errors, f"{loc}.id", "must be a non-empty string")
        elif mid in seen:
            _err(errors, f"{loc}.id", f"duplicate id {mid!r}")
        else:
            seen.add(mid)
        for key in ("name", "file", "source", "mode"):
            if not isinstance(model.get(key), str):
                _err(errors, f"{loc}.{key}", "must be a string")
        if model.get("mode") not in MODES:
            _err(errors, f"{loc}.mode", f"must be one of {MODES}")
        for key in ("numLayers", "totalParams", "totalFlops", "warnings"):
            value = model.get(key)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                _err(errors, f"{loc}.{key}", "must be a non-negative integer")
    errors_raw = index.get("errors")
    if not isinstance(errors_raw, list):
        _err(errors, "errors", "must be a list")
    else:
        for i, item in enumerate(errors_raw):
            if not isinstance(item, dict) or not isinstance(item.get("file"), str) or not isinstance(
                item.get("message"), str
            ):
                _err(errors, f"errors[{i}]", "must be an object with string 'file' and 'message'")
    return errors
