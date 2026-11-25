"""Loader for the NetPrism algorithm knowledge base.

The knowledge base is a plain JSON document (``netprism/data/algorithm_kb.json``)
describing architecture *families* (ResNet, Swin, Mamba, ...) and structural
*components* (residual, SE, SSM, ...).  A generic classifier can score a parsed
graph against it using three independent signals:

* ``aliases`` - class names / paper names / abbreviations found in the graph;
* ``required_patterns`` / ``optional_patterns`` - component ids that pattern
  detection can confirm in the graph;
* ``signature_ops`` - the graph ``kind`` multiset, i.e. the structural fingerprint.

This module is deliberately dependency free (standard library only, no torch /
onnx / tensorflow imports) so it also works inside the PyInstaller bundle, where
the JSON lives under ``sys._MEIPASS``.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from threading import Lock
from typing import Any

__all__ = [
    "KB_FILENAME",
    "DOMAINS",
    "SIGNATURE_OPS",
    "FAMILY_FIELDS",
    "COMPONENT_FIELDS",
    "clear_cache",
    "component_index",
    "components",
    "families",
    "family_index",
    "knowledge_candidates",
    "knowledge_path",
    "load_knowledge",
    "validate_knowledge",
]

KB_FILENAME = "algorithm_kb.json"

#: Allowed values of a family ``domain``.
DOMAINS: tuple[str, ...] = (
    "vision",
    "sequence",
    "generative",
    "graph",
    "3d",
    "multimodal",
    "other",
)

#: Graph node ``kind`` vocabulary accepted in ``signature_ops`` / ``required_ops``.
#: A subset of :data:`netprism.schema.KINDS` (tests keep the two in sync).
SIGNATURE_OPS: tuple[str, ...] = (
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

#: Required keys of a family entry.
FAMILY_FIELDS: tuple[str, ...] = (
    "id",
    "label",
    "domain",
    "aliases",
    "summary",
    "components",
    "required_patterns",
    "optional_patterns",
    "signature_ops",
    "formula",
    "references",
)

#: Required keys of a component entry.
COMPONENT_FIELDS: tuple[str, ...] = (
    "id",
    "label",
    "formula",
    "description",
    "required_ops",
    "tags",
)

_ID_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_MAX_SUMMARY_CHARS = 50

_CACHE: dict[str, Any] | None = None
_LOCK = Lock()


def _meipass() -> Path | None:
    """Return the PyInstaller extraction directory when frozen, else ``None``."""
    raw = getattr(sys, "_MEIPASS", None)
    if not raw:
        return None
    try:
        return Path(raw)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return None


def knowledge_candidates() -> tuple[Path, ...]:
    """Ordered locations where the knowledge base may live.

    Covers the source checkout (``netprism/data`` next to this module) plus the
    two common PyInstaller data layouts (``_MEIPASS/data`` and
    ``_MEIPASS/netprism/data``).
    """
    here = Path(__file__).resolve().parent
    roots: list[Path] = []
    meipass = _meipass()
    if meipass is not None:
        roots.append(meipass)
    roots.append(here)
    roots.append(here.parent)

    candidates: list[Path] = []
    for root in roots:
        candidates.append(root / "data" / KB_FILENAME)
        candidates.append(root / "netprism" / "data" / KB_FILENAME)
        candidates.append(root / KB_FILENAME)
    # De-duplicate while preserving order.
    return tuple(dict.fromkeys(candidates))


def knowledge_path() -> Path:
    """Return the first existing knowledge base path.

    Raises:
        FileNotFoundError: when no candidate location contains the JSON file.
    """
    for candidate in knowledge_candidates():
        if candidate.is_file():
            return candidate
    searched = "\n  - ".join(str(candidate) for candidate in knowledge_candidates())
    raise FileNotFoundError(
        f"algorithm knowledge base {KB_FILENAME!r} not found; searched:\n  - {searched}"
    )


def load_knowledge(*, refresh: bool = False) -> dict[str, Any]:
    """Load and cache the knowledge base document.

    Repeated calls return the *same* object; pass ``refresh=True`` (or call
    :func:`clear_cache`) to re-read the file after an update.
    """
    global _CACHE
    if _CACHE is not None and not refresh:
        return _CACHE
    with _LOCK:
        if _CACHE is not None and not refresh:
            return _CACHE
        path = knowledge_path()
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid algorithm knowledge base JSON at {path}: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError(f"algorithm knowledge base at {path} must be a JSON object")
        _CACHE = data
        return _CACHE


def clear_cache() -> None:
    """Drop the cached knowledge base document."""
    global _CACHE
    with _LOCK:
        _CACHE = None


def families() -> list[dict[str, Any]]:
    """Return the architecture family entries (empty list when malformed)."""
    doc = load_knowledge()
    entries = doc.get("families")
    if not isinstance(entries, list):
        return []
    return [entry for entry in entries if isinstance(entry, dict)]


def components() -> list[dict[str, Any]]:
    """Return the structural component entries (empty list when malformed)."""
    doc = load_knowledge()
    entries = doc.get("components")
    if not isinstance(entries, list):
        return []
    return [entry for entry in entries if isinstance(entry, dict)]


def family_index() -> dict[str, dict[str, Any]]:
    """Map ``family id -> family entry``."""
    return {str(entry.get("id")): entry for entry in families() if entry.get("id")}


def component_index() -> dict[str, dict[str, Any]]:
    """Map ``component id -> component entry``."""
    return {str(entry.get("id")): entry for entry in components() if entry.get("id")}


def _is_str_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) and item.strip() for item in value)


def _is_http_url(value: Any) -> bool:
    return isinstance(value, str) and (value.startswith("https://") or value.startswith("http://"))


def validate_knowledge(knowledge: Any | None = None) -> list[str]:
    """Validate a knowledge base document and return human-readable errors.

    An empty list means the document satisfies the contract.  When
    ``knowledge`` is ``None`` the cached document is validated.
    """
    errors: list[str] = []
    doc = load_knowledge() if knowledge is None else knowledge
    if not isinstance(doc, dict):
        return ["knowledge: must be an object"]

    version = doc.get("version")
    if not isinstance(version, str) or not version.strip():
        errors.append("version: must be a non-empty string")

    raw_families = doc.get("families")
    if not isinstance(raw_families, list):
        errors.append("families: must be a list")
        raw_families = []
    raw_components = doc.get("components")
    if not isinstance(raw_components, list):
        errors.append("components: must be a list")
        raw_components = []

    component_ids: set[str] = set()
    for index, component in enumerate(raw_components):
        loc = f"components[{index}]"
        if not isinstance(component, dict):
            errors.append(f"{loc}: must be an object")
            continue
        for field in COMPONENT_FIELDS:
            if field not in component:
                errors.append(f"{loc}: missing field {field!r}")
        cid = component.get("id")
        if not isinstance(cid, str) or not _ID_RE.match(cid):
            errors.append(f"{loc}.id: invalid id {cid!r}")
        elif cid in component_ids:
            errors.append(f"{loc}.id: duplicate component id {cid!r}")
        else:
            component_ids.add(cid)
        for field in ("label", "formula", "description"):
            value = component.get(field)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"{loc}.{field}: must be a non-empty string")
        ops = component.get("required_ops")
        if not _is_str_list(ops):
            errors.append(f"{loc}.required_ops: must be a non-empty list of strings")
        else:
            for op in ops:
                if op not in SIGNATURE_OPS:
                    errors.append(f"{loc}.required_ops: unknown graph kind {op!r}")
        tags = component.get("tags")
        if not _is_str_list(tags):
            errors.append(f"{loc}.tags: must be a non-empty list of strings")

    family_ids: set[str] = set()
    referenced_components: set[str] = set()
    for index, family in enumerate(raw_families):
        loc = f"families[{index}]"
        if not isinstance(family, dict):
            errors.append(f"{loc}: must be an object")
            continue
        for field in FAMILY_FIELDS:
            if field not in family:
                errors.append(f"{loc}: missing field {field!r}")
        fid = family.get("id")
        if not isinstance(fid, str) or not _ID_RE.match(fid):
            errors.append(f"{loc}.id: invalid id {fid!r}")
        elif fid in family_ids:
            errors.append(f"{loc}.id: duplicate family id {fid!r}")
        else:
            family_ids.add(fid)

        label = family.get("label")
        if not isinstance(label, str) or not label.strip():
            errors.append(f"{loc}.label: must be a non-empty string")
        if family.get("domain") not in DOMAINS:
            errors.append(f"{loc}.domain: must be one of {DOMAINS}")

        aliases = family.get("aliases")
        if not _is_str_list(aliases):
            errors.append(f"{loc}.aliases: must be a non-empty list of strings")

        summary = family.get("summary")
        if not isinstance(summary, str) or not summary.strip():
            errors.append(f"{loc}.summary: must be a non-empty string")
        elif len(summary) > _MAX_SUMMARY_CHARS:
            errors.append(
                f"{loc}.summary: must be <= {_MAX_SUMMARY_CHARS} characters (got {len(summary)})"
            )

        formula = family.get("formula")
        if not isinstance(formula, str) or not formula.strip():
            errors.append(f"{loc}.formula: must be a non-empty string")

        references = family.get("references")
        if not isinstance(references, list) or not references:
            errors.append(f"{loc}.references: must be a non-empty list")
        elif not all(_is_http_url(url) for url in references):
            errors.append(f"{loc}.references: entries must be http(s) URLs")

        signature_ops = family.get("signature_ops")
        if not _is_str_list(signature_ops):
            errors.append(f"{loc}.signature_ops: must be a non-empty list of strings")
        else:
            for op in signature_ops:
                if op not in SIGNATURE_OPS:
                    errors.append(f"{loc}.signature_ops: unknown graph kind {op!r}")

        for field in ("components", "required_patterns", "optional_patterns"):
            value = family.get(field)
            if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
                errors.append(f"{loc}.{field}: must be a list of component ids")
                value = []
            for cid in value:
                if cid not in component_ids:
                    errors.append(f"{loc}.{field}: unknown component id {cid!r}")
                else:
                    referenced_components.add(cid)

        all_ids = family.get("components")
        required = family.get("required_patterns")
        optional = family.get("optional_patterns")
        if isinstance(all_ids, list) and isinstance(required, list) and isinstance(optional, list):
            all_set, required_set, optional_set = set(all_ids), set(required), set(optional)
            if not required_set:
                errors.append(f"{loc}.required_patterns: must not be empty")
            overlap = required_set & optional_set
            if overlap:
                errors.append(f"{loc}: components listed as both required and optional: {sorted(overlap)}")
            if required_set | optional_set != all_set:
                errors.append(
                    f"{loc}.components: must equal required_patterns | optional_patterns "
                    f"(diff: {sorted((required_set | optional_set) ^ all_set)})"
                )

    if isinstance(raw_components, list):
        for index, component in enumerate(raw_components):
            if not isinstance(component, dict):
                continue
            cid = component.get("id")
            if isinstance(cid, str) and cid and cid not in referenced_components:
                errors.append(f"components[{index}] ({cid}): not referenced by any family")
    return errors
