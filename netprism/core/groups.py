"""High level component grouping.

Groups are derived from the module hierarchy (``features.0`` -> group
``features``) and then classified from their structure: two convolutions with a
norm/activation become a ``conv_block``, a block with a residual add becomes a
``residual_block``, an attention + FFN stack becomes a ``transformer_block`` and
so on.  Memberships are non-overlapping: every node belongs to exactly one
*direct* group and groups form a tree through ``parentId`` / ``childGroupIds``.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from .ir import EdgeSpec, ModuleMeta, NodeSpec
from .sources import SourceLoc

LOGGER = logging.getLogger("netprism.core.groups")

_GENERIC_BASES = {
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

_BASE_LABELS = {
    "features": "Features",
    "feature": "Features",
    "stem": "Stem",
    "classifier": "Classifier",
    "cls": "Classifier",
    "fc": "Classifier",
    "head": "Classifier Head",
    "pool": "Pool",
    "avgpool": "Pool",
    "gap": "Global Pool",
    "downsample": "Downsample",
    "shortcut": "Shortcut",
    "proj": "Projection",
    "projection": "Projection",
    "self_attn": "Self Attention",
    "selfattention": "Self Attention",
    "attn": "Attention",
    "attention": "Attention",
    "cross_attn": "Cross Attention",
    "qkv": "QKV",
    "ffn": "Feed Forward",
    "feedforward": "Feed Forward",
    "mlp": "MLP",
    "blocks": "Blocks",
    "layers": "Layers",
    "network": "Network",
    "encoder": "Encoder",
    "decoder": "Decoder",
}

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

_STRUCTURAL_KINDS = frozenset(
    {"conv_block", "residual_block", "transformer_block", "attention_block", "ffn_block", "mlp", "unet_block"}
)

_ENCODER_WORDS = ("encoder", "enc", "down", "downsample", "contract")
_DECODER_WORDS = ("decoder", "dec", "up", "upsample", "expand")

_TRAILING_NUMBER_RE = re.compile(r"^(?P<base>.*?)(?P<number>\d+)$")
_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


# ---------------------------------------------------------------------------
# naming helpers
# ---------------------------------------------------------------------------

def humanize(name: str) -> str:
    """``self_attn`` -> ``Self Attn``; camel-case boundaries get a space."""
    text = _CAMEL_RE.sub(" ", name)
    text = text.replace("_", " ").replace("-", " ").strip()
    parts = [part for part in text.split() if part]
    return " ".join(part if part.isupper() and len(part) <= 5 else part.capitalize() for part in parts)


def split_trailing_number(name: str) -> tuple[str, int | None]:
    """``layer12`` -> ``('layer', 12)``; ``layer`` -> ``('layer', None)``."""
    match = _TRAILING_NUMBER_RE.match(name)
    if not match:
        return name, None
    base = match.group("base")
    try:
        return base, int(match.group("number"))
    except ValueError:  # pragma: no cover - defensive
        return name, None


def _parent_path(path: str) -> str:
    return path.rsplit(".", 1)[0] if "." in path else ""


def _group_label(path: str, kind: str, *, model_name: str = "") -> str:
    if not path:
        return humanize(model_name) or "Model"
    segments = path.split(".")
    last = segments[-1]
    base, number = split_trailing_number(last)
    numeric_only = base == "" and number is not None
    if numeric_only:
        number = (number or 0) + 1
        parent_last = segments[-2] if len(segments) >= 2 else ""
        base, parent_number = split_trailing_number(parent_last)
        if not base:
            base = "block"
    base_key = base.lower()

    if not numeric_only:
        if base_key in _BASE_LABELS:
            label = _BASE_LABELS[base_key]
            return f"{label} {number}" if number is not None else label
        if base_key in ("enc", "encoder"):
            return f"Encoder {number}" if number is not None else "Encoder"
        if base_key in ("dec", "decoder"):
            return f"Decoder {number}" if number is not None else "Decoder"
        if base_key == "down":
            return f"Down Block {number}" if number is not None else "Down Block"
        if base_key == "up":
            return f"Up Block {number}" if number is not None else "Up Block"

    if base_key in _GENERIC_BASES or not base_key:
        noun = _KIND_NOUN.get(kind)
        if noun:
            return f"{noun} {number}" if number is not None else noun
        if kind in ("encoder", "decoder"):
            return "Encoder" if kind == "encoder" else "Decoder"
        label = humanize(base)
        if number is not None:
            label = f"{label} {number}" if label else f"Block {number}"
        return label or "Block"

    label = humanize(base)
    if number is not None:
        label = f"{label} {number}"
    return label or _KIND_NOUN.get(kind, "Block")


# ---------------------------------------------------------------------------
# structural classification
# ---------------------------------------------------------------------------

@dataclass
class _Counts:
    conv: int = 0
    deconv: int = 0
    linear: int = 0
    norm: int = 0
    activation: int = 0
    attention: int = 0
    pool: int = 0
    add: int = 0
    concat: int = 0
    block: int = 0
    dropout: int = 0


def _count_kinds(nodes: list[NodeSpec], member_indices: Sequence[int]) -> _Counts:
    counts = _Counts()
    for index in member_indices:
        kind = nodes[index].kind
        if kind == "conv":
            counts.conv += 1
        elif kind == "deconv":
            counts.deconv += 1
        elif kind in ("linear", "mlp", "ffn"):
            counts.linear += 1
        elif kind == "norm":
            counts.norm += 1
        elif kind == "activation":
            counts.activation += 1
        elif kind == "attention":
            counts.attention += 1
        elif kind == "pool":
            counts.pool += 1
        elif kind == "add":
            counts.add += 1
        elif kind == "concat":
            counts.concat += 1
        elif kind == "block":
            counts.block += 1
        elif kind == "dropout":
            counts.dropout += 1
    return counts


def _structure_kind(
    nodes: list[NodeSpec],
    edges: list[EdgeSpec],
    member_indices: Sequence[int],
    *,
    name: str,
    model_has_skip: bool,
) -> str:
    """Structural kind for a group that owns at least one direct node."""
    member_set = set(member_indices)
    counts = _count_kinds(nodes, member_indices)
    has_residual = False
    has_cross_cat = False
    for edge in edges:
        if edge.target not in member_set:
            continue
        if edge.kind == "residual":
            has_residual = True
        if nodes[edge.target].kind == "concat" and edge.source not in member_set:
            has_cross_cat = True
    if not has_residual:
        for index in member_indices:
            node = nodes[index]
            if node.kind != "add":
                continue
            if "residual" in node.tags or sum(1 for edge in edges if edge.target == index) >= 2:
                has_residual = True
                break

    lowered = name.lower()
    if counts.attention > 0:
        if counts.linear >= 2 and counts.norm >= 1:
            return "transformer_block"
        return "attention_block"
    if counts.conv == 0 and counts.deconv == 0 and counts.linear == 0 and counts.add == 0 and counts.concat == 0:
        return ""
    if has_residual:
        return "residual_block"
    if counts.conv >= 2 and (counts.norm + counts.activation) >= 1:
        return "conv_block"
    if counts.linear >= 2:
        if "ffn" in lowered or "feedforward" in lowered:
            return "ffn_block"
        return "mlp"
    if any(word in lowered for word in _ENCODER_WORDS) and (model_has_skip or has_cross_cat) and (
        counts.conv + counts.deconv >= 1
    ):
        return "unet_block"
    return ""


def _container_kind(
    path: str,
    meta: ModuleMeta | None,
    *,
    model_has_skip: bool,
    nodes: list[NodeSpec],
    member_indices: Sequence[int],
) -> str:
    last = path.rsplit(".", 1)[-1].lower()
    counts = _count_kinds(nodes, member_indices)
    if "ffn" in last or "feedforward" in last:
        return "ffn_block" if counts.linear >= 2 else "module"
    if last in ("attn", "attention", "self_attn", "cross_attn"):
        return "attention_block" if counts.attention > 0 else "module"
    if "encoder" in last:
        return "encoder"
    if "decoder" in last:
        return "decoder"
    if model_has_skip and (last.startswith("enc") or last.startswith("down")):
        return "unet_block"
    if model_has_skip and (last.startswith("dec") or last.startswith("up")):
        return "unet_block"
    if meta is not None and meta.is_sequential:
        return "sequential"
    return "module"


# ---------------------------------------------------------------------------
# grouping
# ---------------------------------------------------------------------------

def build_groups(
    nodes: list[NodeSpec],
    node_ids: list[str],
    edges: list[EdgeSpec],
    module_meta: dict[str, ModuleMeta],
    *,
    model_name: str,
    class_name: str | None,
    root_source: SourceLoc | None = None,
) -> tuple[list[dict], list[str | None], list[str]]:
    """Build the schema group list; returns ``(groups, node_group_ids, warnings)``."""
    warnings: list[str] = []
    group_paths: set[str] = set()
    direct_group_path: list[str | None] = []

    for node in nodes:
        if node.kind in ("input", "output") or not node.module_path:
            direct_group_path.append(None)
            continue
        segments = node.module_path.split(".")
        ancestors = [".".join(segments[:index]) for index in range(1, len(segments))]
        group_paths.update(ancestors)
        direct_group_path.append(ancestors[-1] if ancestors else None)

    if not group_paths:
        root = _root_group(nodes, model_name, class_name, root_source)
        node_group_ids = [None] * len(nodes)
        for index, node in enumerate(nodes):
            if node.kind in ("input", "output"):
                continue
            root["nodeIds"].append(node_ids[index])
            node_group_ids[index] = root["id"]
            node.group_id = root["id"]
        return [root], node_group_ids, warnings

    parent_of_group: dict[str, str] = {}
    for path in group_paths:
        parent = _parent_path(path)
        while parent and parent not in group_paths:
            parent = _parent_path(parent)
        parent_of_group[path] = parent
    ordered_paths = sorted(group_paths, key=lambda item: (item.count("."), item))

    member_indices_all: dict[str, list[int]] = {path: [] for path in ordered_paths}
    direct_indices: dict[str, list[int]] = {path: [] for path in ordered_paths}
    for index, direct_path in enumerate(direct_group_path):
        if direct_path is None:
            continue
        direct_indices[direct_path].append(index)
        current: str | None = direct_path
        while current:
            member_indices_all[current].append(index)
            current = parent_of_group.get(current, "")

    model_has_skip = any(edge.kind == "skip" for edge in edges)

    id_for_path: dict[str, str] = {}
    next_id = 1
    for path in ordered_paths:
        id_for_path[path] = f"g{next_id}"
        next_id += 1

    groups: list[dict] = []
    group_by_path: dict[str, dict] = {}
    node_group_ids: list[str | None] = [None] * len(nodes)

    root = _root_group(nodes, model_name, class_name, root_source)
    groups.append(root)

    for path in ordered_paths:
        meta = module_meta.get(path)
        member_indices = member_indices_all[path]
        direct = direct_indices[path]
        if direct:
            kind = _structure_kind(
                nodes,
                edges,
                member_indices,
                name=path.rsplit(".", 1)[-1],
                model_has_skip=model_has_skip,
            )
            if not kind:
                if meta is not None and meta.is_sequential:
                    kind = "sequential"
                elif meta is not None and not meta.is_container:
                    kind = "custom"
                else:
                    kind = "module"
        else:
            kind = _container_kind(
                path,
                meta,
                model_has_skip=model_has_skip,
                nodes=nodes,
                member_indices=member_indices,
            )

        child_paths = sorted(child for child, parent in parent_of_group.items() if parent == path)
        parent_path = parent_of_group.get(path, "")
        parent_group = group_by_path.get(parent_path)
        depth = (parent_group["depth"] + 1) if parent_group is not None else 1
        group = {
            "id": id_for_path[path],
            "name": path,
            "label": _group_label(path, kind, model_name=model_name),
            "kind": kind,
            "parentId": parent_group["id"] if parent_group is not None else root["id"],
            "depth": depth,
            "nodeIds": [node_ids[index] for index in direct],
            "childGroupIds": [id_for_path[child] for child in child_paths],
            "numParams": sum(nodes[index].num_params for index in member_indices),
            "flops": sum(nodes[index].flops for index in member_indices),
            "source": _source_dict(meta.source if meta is not None else None),
        }
        groups.append(group)
        group_by_path[path] = group
        for index in direct:
            node_group_ids[index] = id_for_path[path]

    for index, group_id in enumerate(node_group_ids):
        if group_id is not None:
            nodes[index].group_id = group_id

    # Top-level nodes without a natural container belong directly to the model
    # group, so the group tree always covers every non-boundary node.
    root_node_ids: list[str] = list(root["nodeIds"])
    for index, node in enumerate(nodes):
        if node_group_ids[index] is not None or node.kind in ("input", "output"):
            continue
        node_group_ids[index] = root["id"]
        nodes[index].group_id = root["id"]
        root_node_ids.append(node_ids[index])
    root["nodeIds"] = root_node_ids

    root_children = sorted(child for child, parent in parent_of_group.items() if parent == "")
    root["childGroupIds"] = [id_for_path[child] for child in root_children]
    root["numParams"] = sum(node.num_params for node in nodes)
    root["flops"] = sum(node.flops for node in nodes)

    _promote_trailing_merges(
        nodes, node_ids, edges, groups, node_group_ids, parent_of_group, id_for_path
    )
    _recompute_group_totals(nodes, node_ids, groups)

    _relabel_aggregator_groups(groups)
    _disambiguate_labels(groups)

    # Non-overlap sanity: every node may appear in at most one nodeIds list.
    seen_nodes: dict[str, str] = {}
    for group in groups:
        for node_id in group["nodeIds"]:
            if node_id in seen_nodes:
                warnings.append(
                    f"group overlap: node {node_id} listed in {seen_nodes[node_id]} and {group['id']}"
                )
            seen_nodes[node_id] = group["id"]
    return groups, node_group_ids, warnings


def _promote_trailing_merges(
    nodes: list[NodeSpec],
    node_ids: list[str],
    edges: list[EdgeSpec],
    groups: list[dict],
    node_group_ids: list[str | None],
    parent_of_group: dict[str, str],
    id_for_path: dict[str, str],
) -> None:
    """Move a group's trailing concat merge out into its parent group.

    Decoder modules often end with ``torch.cat(block(x), skip)``.  That concat is
    the module's output, so it only lines up *after* the module's layers; keeping
    it inside the group box made skip arcs look like they ended before the
    decoder block, and the layer tree listed the target before the block.  A
    trailing concat that consumes a skip/residual from another group is promoted
    to the parent group so it sits right after the group.
    """
    if not groups:
        return
    index_by_id = {node_id: index for index, node_id in enumerate(node_ids)}
    group_by_id = {group["id"]: group for group in groups}
    path_by_id = {group["id"]: group["name"] for group in groups}
    root_id = groups[0]["id"]
    incoming: dict[str, list[EdgeSpec]] = {}
    for edge in edges:
        incoming.setdefault(node_ids[edge.target], []).append(edge)
    for index, node in enumerate(nodes):
        if node.kind != "concat":
            continue
        group_id = node_group_ids[index]
        if not group_id or group_id == root_id:
            continue
        group = group_by_id.get(group_id)
        path = path_by_id.get(group_id)
        if group is None or not path:
            continue
        members = [
            index_by_id[node_id]
            for node_id in group.get("nodeIds") or []
            if node_id in index_by_id
        ]
        if not members or max(members) != index:
            continue
        external_merge = any(
            edge.kind in ("skip", "residual")
            and node_group_ids[edge.source] not in (None, group_id)
            for edge in incoming.get(node_ids[index], [])
        )
        if not external_merge:
            continue
        parent_path = parent_of_group.get(path, "")
        parent_id = id_for_path.get(parent_path, root_id) if parent_path else root_id
        if not parent_id or parent_id == group_id:
            continue
        group["nodeIds"] = [
            node_id for node_id in group.get("nodeIds") or [] if node_id != node_ids[index]
        ]
        parent_group = group_by_id.get(parent_id)
        if parent_group is not None:
            parent_nodes = parent_group.setdefault("nodeIds", [])
            if node_ids[index] not in parent_nodes:
                parent_nodes.append(node_ids[index])
                parent_nodes.sort(key=lambda node_id: index_by_id.get(node_id, 0))
        node_group_ids[index] = parent_id
        node.group_id = parent_id


def _recompute_group_totals(
    nodes: list[NodeSpec],
    node_ids: list[str],
    groups: list[dict],
) -> None:
    """Recompute per-group params/FLOPs after memberships were adjusted."""
    node_by_id = {node_id: node for node_id, node in zip(node_ids, nodes)}
    group_by_id = {group["id"]: group for group in groups}
    cache: dict[str, tuple[int, int]] = {}

    def totals(group_id: str) -> tuple[int, int]:
        if group_id in cache:
            return cache[group_id]
        group = group_by_id.get(group_id)
        if group is None:
            return (0, 0)
        params = sum(
            node_by_id[node_id].num_params
            for node_id in group.get("nodeIds") or []
            if node_id in node_by_id
        )
        flops = sum(
            node_by_id[node_id].flops
            for node_id in group.get("nodeIds") or []
            if node_id in node_by_id
        )
        for child_id in group.get("childGroupIds") or []:
            child_params, child_flops = totals(child_id)
            params += child_params
            flops += child_flops
        cache[group_id] = (params, flops)
        return cache[group_id]

    for group in groups:
        params, flops = totals(group["id"])
        group["numParams"] = params
        group["flops"] = flops


_PLURAL_LABELS = {
    "transformer_block": "Transformer Blocks",
    "attention_block": "Attention Blocks",
    "residual_block": "Residual Blocks",
    "conv_block": "Conv Blocks",
    "unet_block": "UNet Blocks",
    "ffn_block": "Feed Forward",
    "mlp": "MLPs",
}

#: Only top-level generic containers get a plural label.  Deeper ones keep
#: their own name (``encoder.layers`` -> "Layers"), so the 3D scene does not
#: show the same generic phrase at several nesting levels.
_AGGREGATOR_NAMES = frozenset({"blocks", "layers", "stages", "units", "modules", "network"})

_BLOCK_LABEL_RE = re.compile(r"^(?:Transformer|Residual|Conv|Attention|UNet)\s+Block\s+(\d+)$")


def _relabel_aggregator_groups(groups: list[dict]) -> None:
    """Give top-level aggregator containers a descriptive plural label.

    ``blocks`` (a ``ModuleList`` of transformer blocks) should read
    "Transformer Blocks" instead of the generic "Blocks"; deeper containers
    such as ``encoder.layers`` keep their own name ("Layers").
    """
    by_id = {group["id"]: group for group in groups}
    for group in groups:
        if group.get("nodeIds"):
            continue
        if group.get("kind") not in ("module", "sequential"):
            continue
        if group.get("depth") != 1:
            continue
        last_segment = group.get("name", "").rsplit(".", 1)[-1].lower()
        if last_segment not in _AGGREGATOR_NAMES:
            continue
        child_kinds = {
            by_id[child]["kind"]
            for child in group.get("childGroupIds", [])
            if child in by_id and by_id[child]["kind"] in _PLURAL_LABELS
        }
        if len(child_kinds) == 1:
            group["label"] = _PLURAL_LABELS[child_kinds.pop()]


def _short_context(label: str) -> str:
    """Shorten a parent label so child labels stay readable in 3D."""
    if " · " in label:
        return label.split(" · ", 1)[0]
    match = _BLOCK_LABEL_RE.match(label)
    if match:
        return f"Block {match.group(1)}"
    return label


def _disambiguate_labels(groups: list[dict]) -> None:
    """Make every group label unique, prefixing duplicated ones with context.

    Nested generic blocks (``enc1.block`` / ``dec2.block`` -> "Conv Block") get
    their parent context: "Encoder 1 · Conv Block".  When even that is not
    enough, the full parent label is used, and finally a numeric suffix.
    """
    if not groups:
        return
    root = next((group for group in groups if group.get("parentId") is None), groups[0])
    by_id = {group["id"]: group for group in groups}
    counts: dict[str, int] = {}
    for group in groups:
        if group is root:
            continue
        counts[group["label"]] = counts.get(group["label"], 0) + 1

    used: set[str] = {root["label"]}
    ordered = [group for group in groups if group is not root]
    ordered.sort(key=lambda group: (group.get("depth", 0), group.get("name", "")))
    for group in ordered:
        base = group["label"]
        final = base
        if counts.get(base, 0) > 1:
            parent = by_id.get(group.get("parentId"))
            if parent is not None:
                parent_label = parent["label"]
                context = parent_label if " · " in parent_label else _short_context(parent_label)
                if context and context != base:
                    candidate = f"{context} · {base}"
                    if candidate not in used:
                        final = candidate
                if final == base:
                    candidate = f"{parent_label} · {base}"
                    if candidate not in used:
                        final = candidate
        if final in used:
            suffix = 2
            while f"{final} ({suffix})" in used:
                suffix += 1
            final = f"{final} ({suffix})"
        group["label"] = final
        used.add(final)


def _root_group(
    nodes: list[NodeSpec],
    model_name: str,
    class_name: str | None,
    root_source: SourceLoc | None,
) -> dict:
    return {
        "id": "g0",
        "name": class_name or model_name,
        "label": humanize(model_name) or class_name or "Model",
        "kind": "model",
        "parentId": None,
        "depth": 0,
        "nodeIds": [],
        "childGroupIds": [],
        "numParams": sum(node.num_params for node in nodes),
        "flops": sum(node.flops for node in nodes),
        "source": _source_dict(root_source),
    }


def enrich_attention_heads(
    groups: list[dict],
    node_ids: list[str],
    nodes: list[NodeSpec],
) -> None:
    """Infer ``numHeads`` / ``headDim`` for hand-written attention blocks.

    ``nn.MultiheadAttention`` exposes these attributes directly, but a custom
    attention module only reveals them through reshape shapes such as
    ``(batch, seq, heads, head_dim)``.  We recover them and attach them to the
    attention-kind nodes so the 3D front-end can draw the head slices.
    """
    by_id = {group["id"]: group for group in groups}
    node_by_id = {node_id: node for node_id, node in zip(node_ids, nodes)}

    def subtree_ids(group: dict) -> list[str]:
        collected: list[str] = list(group.get("nodeIds") or [])
        for child in group.get("childGroupIds") or []:
            if child in by_id:
                collected.extend(subtree_ids(by_id[child]))
        return collected

    for group in groups:
        if group.get("kind") not in ("attention_block", "transformer_block"):
            continue
        member_ids = subtree_ids(group)
        candidates: list[tuple[int, int]] = []
        for node_id in member_ids:
            node = node_by_id.get(node_id)
            if node is None:
                continue
            shape = node.params.get("shape") if isinstance(node.params, dict) else None
            if (
                isinstance(shape, list)
                and len(shape) == 4
                and all(isinstance(dim, int) and dim > 0 for dim in shape)
            ):
                candidates.append((int(shape[2]), int(shape[3])))
        if not candidates:
            continue
        heads, head_dim = candidates[0]
        if any(value != candidates[0] for value in candidates[1:]):
            LOGGER.debug("inconsistent attention reshape shapes in %s", group["name"])
        embed_dim = heads * head_dim
        for node_id in member_ids:
            node = node_by_id.get(node_id)
            if node is None or node.kind != "attention":
                continue
            if "numHeads" in node.params:
                continue
            node.params.setdefault("numHeads", heads)
            node.params.setdefault("headDim", head_dim)
            node.params.setdefault("embedDim", embed_dim)


def _source_dict(source: SourceLoc | None) -> dict | None:
    if source is None:
        return None
    payload: dict = {"line": int(source.line)}
    if source.code:
        payload["code"] = source.code
    return payload


__all__ = ["build_groups", "enrich_attention_heads", "humanize", "split_trailing_number"]
