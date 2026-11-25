"""Estimation metrics for NetPrism graphs (standard library only).

Everything here is derived from the graph document itself: output shapes for
activation memory and operator hyper-parameters for the receptive field.  No
torch/onnx import happens, so the module works on every parsed format and in
the frozen EXE build.

All numbers are *estimates*: activation memory assumes float32 (4 bytes per
element), and the receptive field is a conservative scalar estimate for the
spatial axes of a CNN/pool chain (branches keep the largest field).
"""

from __future__ import annotations

import heapq
from typing import Any

FLOAT32_BYTES = 4

_SPATIAL_KINDS = frozenset({"conv", "deconv", "pool"})
_NODE_METRIC_KEYS = ("activationBytes", "receptiveField")


# --------------------------------------------------------------------------- #
# small value helpers
# --------------------------------------------------------------------------- #
def _as_int(value: Any, default: int = 0) -> int:
    """Return ``value`` when it is a non-negative int, else ``default``."""
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return default


def _as_shapes(value: Any) -> list[list[int]]:
    """Normalize a graph shapes field to a list of integer dimension lists."""
    if not isinstance(value, (list, tuple)) or not value:
        return []
    if all(isinstance(dim, int) and not isinstance(dim, bool) for dim in value):
        return [[int(dim) for dim in value]]
    shapes: list[list[int]] = []
    for shape in value:
        if isinstance(shape, (list, tuple)) and shape:
            if all(isinstance(dim, int) and not isinstance(dim, bool) for dim in shape):
                shapes.append([int(dim) for dim in shape])
    return shapes


def _first_out_shape(node: dict[str, Any]) -> list[int] | None:
    shapes = _as_shapes(node.get("outShapes"))
    return shapes[0] if shapes else None


def _element_count(shape: list[int] | None) -> int | None:
    """Product of the dimensions; unknown (-1) dims count as 1."""
    if not shape:
        return None
    count = 1
    for dim in shape:
        count *= dim if dim >= 0 else 1
    return count


def _node_activation_bytes(node: dict[str, Any]) -> int:
    count = _element_count(_first_out_shape(node))
    return count * FLOAT32_BYTES if count is not None else 0


def _positive_int(value: Any, default: int = 1) -> int:
    """Largest positive dimension of an int/list kernel-like hyper-parameter."""
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)) and value >= 1:
        return int(value)
    if isinstance(value, (list, tuple)):
        values = [
            int(item)
            for item in value
            if isinstance(item, (int, float)) and not isinstance(item, bool) and item >= 1
        ]
        if values:
            return max(values)
    return default


def _param(params: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in params and params[name] is not None:
            return params[name]
    return None


def _is_spatial(node: dict[str, Any]) -> bool:
    kind = str(node.get("kind") or "").lower()
    op = str(node.get("op") or "").lower()
    return kind in _SPATIAL_KINDS or "conv" in op or "pool" in op


def _node_sort_key(node_id: str) -> tuple[int, str]:
    if node_id.startswith("n") and node_id[1:].isdigit():
        return (int(node_id[1:]), node_id)
    return (10**9, node_id)


# --------------------------------------------------------------------------- #
# graph traversal
# --------------------------------------------------------------------------- #
def _collect_nodes(graph: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw_nodes = graph.get("nodes")
    nodes: dict[str, dict[str, Any]] = {}
    if not isinstance(raw_nodes, list):
        return nodes
    for index, node in enumerate(raw_nodes):
        if not isinstance(node, dict):
            continue
        node_id = node.get("id")
        if not isinstance(node_id, str) or not node_id:
            node_id = f"n{index}"
        nodes[node_id] = node
    return nodes


def _topo_order(nodes: dict[str, dict[str, Any]], edges: Any) -> tuple[list[str], dict[str, list[str]]]:
    """Kahn topological order; remaining (cyclic) nodes are appended by id."""
    successors: dict[str, list[str]] = {node_id: [] for node_id in nodes}
    predecessors: dict[str, list[str]] = {node_id: [] for node_id in nodes}
    indegree: dict[str, int] = {node_id: 0 for node_id in nodes}
    seen_pairs: set[tuple[str, str]] = set()
    if isinstance(edges, list):
        for edge in edges:
            if not isinstance(edge, dict):
                continue
            source, target = edge.get("source"), edge.get("target")
            if source not in nodes or target not in nodes or source == target:
                continue
            pair = (source, target)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            successors[source].append(target)
            predecessors[target].append(source)
            indegree[target] += 1

    heap = [(_node_sort_key(node_id), node_id) for node_id, degree in indegree.items() if degree == 0]
    heapq.heapify(heap)
    order: list[str] = []
    while heap:
        _, node_id = heapq.heappop(heap)
        order.append(node_id)
        for target in successors[node_id]:
            indegree[target] -= 1
            if indegree[target] == 0:
                heapq.heappush(heap, (_node_sort_key(target), target))
    placed = set(order)
    for node_id in sorted((n for n in nodes if n not in placed), key=_node_sort_key):
        order.append(node_id)
    return order, predecessors


def _receptive_fields(
    nodes: dict[str, dict[str, Any]], order: list[str], predecessors: dict[str, list[str]]
) -> tuple[dict[str, int], dict[str, int]]:
    """Propagate ``(rf, accumulated_stride)`` along the graph."""
    fields: dict[str, int] = {}
    strides: dict[str, int] = {}
    for node_id in order:
        node = nodes[node_id]
        prior_fields = [fields[src] for src in predecessors[node_id] if src in fields]
        prior_strides = [strides[src] for src in predecessors[node_id] if src in strides]
        base_field = max(prior_fields) if prior_fields else 1
        base_stride = max(prior_strides) if prior_strides else 1

        if str(node.get("kind") or "").lower() == "input":
            fields[node_id] = 1
            strides[node_id] = 1
            continue

        if _is_spatial(node):
            params = node.get("params") if isinstance(node.get("params"), dict) else {}
            kernel = _positive_int(_param(params, "kernelSize", "kernel_size", "kernel"))
            stride = _positive_int(_param(params, "stride", "strides"))
            dilation = _positive_int(_param(params, "dilation", "dilations"))
            grown = base_field + (kernel - 1) * dilation * base_stride
            fields[node_id] = max(1, grown)
            strides[node_id] = max(1, base_stride * stride)
        elif str(node.get("kind") or "").lower() == "unknown":
            # An explicitly unknown layer gives no propagation guarantee.
            fields[node_id] = 1
            strides[node_id] = 1
        else:
            # Elementwise/reshape/linear ops keep the current field.
            fields[node_id] = max(1, base_field)
            strides[node_id] = max(1, base_stride)
    return fields, strides


def _peak_activation_bytes(
    nodes: dict[str, dict[str, Any]],
    order: list[str],
    edges: Any,
    activation_by_id: dict[str, int],
) -> int:
    """Estimate the peak live activation set with a last-use liveness pass."""
    position = {node_id: index for index, node_id in enumerate(order)}
    last_use: dict[str, int] = {}
    seen_pairs: set[tuple[str, str]] = set()
    if isinstance(edges, list):
        for edge in edges:
            if not isinstance(edge, dict):
                continue
            source, target = edge.get("source"), edge.get("target")
            if source not in nodes or target not in nodes or source == target:
                continue
            pair = (source, target)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            target_index = position.get(target)
            if target_index is None:
                continue
            last_use[source] = max(last_use.get(source, -1), target_index)

    free_at: dict[int, list[str]] = {}
    for source, index in last_use.items():
        free_at.setdefault(index, []).append(source)

    live = 0
    peak = 0
    added_at: dict[str, int] = {}
    freed: set[str] = set()
    for index, node_id in enumerate(order):
        size = activation_by_id.get(node_id, 0)
        if size:
            live += size
            added_at[node_id] = index
        peak = max(peak, live)
        for source in free_at.get(index, []):
            if source in freed or source not in added_at or added_at[source] >= index:
                continue
            freed.add(source)
            live -= activation_by_id.get(source, 0)
            live = max(0, live)
    return max(peak, max(activation_by_id.values(), default=0))


# --------------------------------------------------------------------------- #
# analysis
# --------------------------------------------------------------------------- #
def _analyze(graph: dict[str, Any]) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Compute model and per-node metrics without writing to the graph."""
    model = graph.get("model") if isinstance(graph, dict) else None
    model = model if isinstance(model, dict) else {}
    nodes = _collect_nodes(graph)
    edges = graph.get("edges") if isinstance(graph, dict) else None
    order, predecessors = _topo_order(nodes, edges)

    activation_by_id = {node_id: _node_activation_bytes(node) for node_id, node in nodes.items()}
    fields, _ = _receptive_fields(nodes, order, predecessors)
    total_activation = sum(activation_by_id.values())
    peak = _peak_activation_bytes(nodes, order, edges, activation_by_id)
    max_receptive_field = max(fields.values(), default=1)

    model_metrics = {
        "paramBytes": _as_int(model.get("totalParams")) * FLOAT32_BYTES,
        "activationBytes": total_activation,
        "peakActivationBytes": peak,
        "maxReceptiveField": max_receptive_field,
        "macs": _as_int(model.get("totalFlops")) // 2,
    }
    node_metrics: dict[str, dict[str, Any]] = {}
    for node_id, node in nodes.items():
        node_metrics[node_id] = {
            "activationBytes": activation_by_id.get(node_id, 0),
            "receptiveField": fields.get(node_id, 1),
            "macs": _as_int(node.get("flops")) // 2,
        }
    return model_metrics, node_metrics


def _is_non_empty_mapping(value: Any) -> bool:
    return isinstance(value, dict) and bool(value)


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def enrich_graph(graph: dict[str, Any], *, force: bool = False) -> dict[str, Any]:
    """Write estimated ``model.metrics`` / ``node.metrics`` and return ``graph``.

    Existing non-empty metrics are kept unless ``force=True`` so exact values
    produced by a future parser adapter are never overwritten by estimates.
    """
    if not isinstance(graph, dict):
        return graph
    model_metrics, node_metrics = _analyze(graph)

    model = graph.get("model")
    if isinstance(model, dict) and (force or not _is_non_empty_mapping(model.get("metrics"))):
        model["metrics"] = model_metrics

    raw_nodes = graph.get("nodes")
    if isinstance(raw_nodes, list):
        for index, node in enumerate(raw_nodes):
            if not isinstance(node, dict):
                continue
            node_id = node.get("id")
            if not isinstance(node_id, str) or not node_id:
                node_id = f"n{index}"
            computed = node_metrics.get(node_id)
            if computed is None:
                continue
            if force or not _is_non_empty_mapping(node.get("metrics")):
                node["metrics"] = dict(computed)
    return graph


def layer_rows(graph: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the flat per-layer table shared by reports and CSV export."""
    if not isinstance(graph, dict):
        return []
    raw_nodes = graph.get("nodes") if isinstance(graph.get("nodes"), list) else []
    raw_groups = graph.get("groups") if isinstance(graph.get("groups"), list) else []
    group_labels: dict[str, str] = {}
    for group in raw_groups:
        if not isinstance(group, dict) or not isinstance(group.get("id"), str):
            continue
        label = group.get("label") or group.get("name") or ""
        group_labels[group["id"]] = str(label)

    need_fallback = any(
        not isinstance(node, dict)
        or not isinstance(node.get("metrics"), dict)
        or any(key not in node["metrics"] for key in _NODE_METRIC_KEYS)
        for node in raw_nodes
        if isinstance(node, dict)
    )
    computed = _analyze(graph)[1] if need_fallback and raw_nodes else {}

    rows: list[dict[str, Any]] = []
    for node in raw_nodes:
        if not isinstance(node, dict):
            continue
        node_id = str(node.get("id") or "")
        stored = node.get("metrics") if isinstance(node.get("metrics"), dict) else {}
        fallback = computed.get(node_id, {})
        group_id = node.get("groupId") if isinstance(node.get("groupId"), str) else None
        rows.append(
            {
                "id": node_id,
                "name": str(node.get("name") or ""),
                "op": str(node.get("op") or ""),
                "kind": str(node.get("kind") or ""),
                "modulePath": str(node.get("modulePath") or ""),
                "inShapes": _as_shapes(node.get("inShapes")),
                "outShapes": _as_shapes(node.get("outShapes")),
                "numParams": _as_int(node.get("numParams")),
                "flops": _as_int(node.get("flops")),
                "activationBytes": _as_int(stored.get("activationBytes"), _as_int(fallback.get("activationBytes"), 0)),
                "receptiveField": _as_int(stored.get("receptiveField"), _as_int(fallback.get("receptiveField"), 1)),
                "groupId": group_id,
                "groupLabel": group_labels.get(group_id, ""),
                "source": node.get("source"),
            }
        )
    return rows


__all__ = ["FLOAT32_BYTES", "enrich_graph", "layer_rows"]
