"""Presentation helpers shared by the CLI and the HTTP server.

Real-world repositories frequently expose many architectural variants of the
same class (``ResNet18``/``ResNet34``/``ResNet50``/...).  A one-click folder
scan should show a manageable list by default, while still allowing explicit
access to every variant.  This module folds *variants of the same class in the
same source file* down to the first one; it never changes the underlying graph
or the public ``netprism.core`` API.
"""

from __future__ import annotations

from typing import Any


def _variant_key(model: dict[str, Any]) -> tuple[str, str]:
    class_name = model.get("className") or model.get("name") or ""
    source = model.get("source") or ""
    return str(class_name), str(source)


def fold_variants(
    index: dict[str, Any],
    graphs: dict[str, dict[str, Any]],
    *,
    enabled: bool = True,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Return ``(index, graphs, dropped)`` with one model per class/file.

    ``dropped`` contains the index entries that were folded away.  When
    ``enabled`` is false the input is returned unchanged.
    """
    models = list(index.get("models", []))
    if not enabled or not models:
        return index, graphs, []

    seen: dict[tuple[str, str], str] = {}
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for model in models:
        key = _variant_key(model)
        if key in seen:
            dropped.append(model)
            continue
        seen[key] = str(model.get("id"))
        kept.append(model)

    if not dropped:
        return index, graphs, []

    kept_ids = {str(model.get("id")) for model in kept}
    new_index = dict(index)
    new_index["models"] = kept
    new_graphs = {mid: graph for mid, graph in graphs.items() if mid in kept_ids}
    return new_index, new_graphs, dropped


def format_folded_note(dropped: list[dict[str, Any]]) -> str:
    if not dropped:
        return ""
    examples = ", ".join(str(item.get("name")) for item in dropped[:6])
    suffix = "" if len(dropped) <= 6 else "…"
    return f"已折叠 {len(dropped)} 个同类变体（{examples}{suffix}），用 --all-variants 查看全部"
