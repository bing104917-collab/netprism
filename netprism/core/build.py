"""Assemble the public graph documents and scan folders.

The analyzer works on the mode independent :class:`~netprism.core.ir.Analysis`
produced by the dynamic tracer or the static AST parser; :func:`make_graph`
turns that into the frozen JSON contract (validated with
:func:`netprism.schema.validate_graph`).
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from .. import schema
from . import astscan, discover, groups, tracer
from .errors import AnalyzeError
from .ir import Analysis
from .sources import SourceIndex

LOGGER = logging.getLogger("netprism.core.build")

_SLUG_RE = re.compile(r"[^a-z0-9]+")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def slugify(text: str) -> str:
    slug = _SLUG_RE.sub("_", text.strip().lower()).strip("_")
    return slug or "model"


def model_id(path: Path, class_name: str | None) -> str:
    return f"{slugify(path.stem)}__{slugify(class_name or path.stem)}"


def display_path(path: Path) -> str:
    """Repo-relative POSIX path when possible, absolute POSIX path otherwise."""
    try:
        return path.resolve().relative_to(schema.REPO_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def folder_relative(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


_NORMALIZE_RE = re.compile(r"[^a-z0-9]+")


def _canonical_score(model_id_value: str, display_name: str) -> tuple[int, int, str]:
    """Rank duplicate same-named models; the file matching the name wins."""
    file_stem = model_id_value.split("__", 1)[0]
    name_normalized = _NORMALIZE_RE.sub("", display_name.lower())
    stem_normalized = _NORMALIZE_RE.sub("", file_stem.lower())
    exact = 0 if name_normalized and name_normalized == stem_normalized else 1
    return (exact, len(stem_normalized), model_id_value)


def _source_payload(source: Any, file_label: str) -> dict | None:
    if source is None:
        return None
    payload: dict[str, Any] = {"file": file_label, "line": int(source.line)}
    if getattr(source, "code", None):
        payload["code"] = str(source.code)
    return payload


def _shape_list(shapes: Any) -> list[list[int]]:
    """JSON-safe shape lists; inner shapes are never empty (schema requirement)."""
    payload: list[list[int]] = []
    for shape in shapes or []:
        if not isinstance(shape, (list, tuple)) or not shape:
            payload.append([-1])
            continue
        payload.append([int(dim) for dim in shape])
    return payload


def _clean_float(value: Any) -> Any:
    """JSON-safe scalar conversion (numpy/pytorch scalars -> python)."""
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, str)):
        return value
    if isinstance(value, float):
        return value
    if isinstance(value, (list, tuple)):
        return [_clean_float(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _clean_float(item) for key, item in value.items()}
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return int(number) if number.is_integer() else number


# ---------------------------------------------------------------------------
# graph assembly
# ---------------------------------------------------------------------------

def make_graph(
    analysis: Analysis,
    *,
    file_path: Path,
    scan_root: Path,
    class_name: str | None = None,
) -> dict[str, Any]:
    """Turn an :class:`Analysis` into a validated schema-1.0 graph dict."""
    node_ids = [f"n{index}" for index in range(len(analysis.nodes))]
    file_label = display_path(file_path)
    rel_path = folder_relative(scan_root, file_path)

    group_list, _node_group_ids, group_warnings = groups.build_groups(
        analysis.nodes,
        node_ids,
        analysis.edges,
        analysis.module_meta,
        model_name=analysis.model_name,
        class_name=class_name or analysis.class_name,
        root_source=analysis.root_source,
    )
    for group in group_list:
        source = group.get("source")
        if isinstance(source, dict):
            source.setdefault("file", file_label)
            source.pop("relPath", None)
    groups.enrich_attention_heads(group_list, node_ids, analysis.nodes)

    nodes_payload: list[dict[str, Any]] = []
    for index, node in enumerate(analysis.nodes):
        group_id = getattr(node, "group_id", None)
        nodes_payload.append(
            {
                "id": node_ids[index],
                "name": str(node.name),
                "op": str(node.op),
                "kind": node.kind,
                "modulePath": str(node.module_path),
                "depth": int(getattr(node, "depth", 1)),
                "inShapes": _shape_list(node.in_shapes),
                "outShapes": _shape_list(node.out_shapes),
                "params": _clean_float(node.params),
                "numParams": int(node.num_params),
                "flops": int(node.flops),
                "tags": list(dict.fromkeys(str(tag) for tag in node.tags)),
                "groupId": group_id,
                "source": _source_payload(node.source, file_label),
                "notes": [str(note) for note in node.notes],
            }
        )

    edges_payload: list[dict[str, Any]] = []
    for index, edge in enumerate(analysis.edges):
        if edge.source < 0 or edge.source >= len(node_ids):
            continue
        if edge.target < 0 or edge.target >= len(node_ids):
            continue
        edges_payload.append(
            {
                "id": f"e{index}",
                "source": node_ids[edge.source],
                "target": node_ids[edge.target],
                "kind": edge.kind,
                "shapes": _shape_list(edge.shapes),
                "label": str(edge.label),
            }
        )

    warnings = list(dict.fromkeys([*analysis.warnings, *group_warnings]))
    num_layers = sum(1 for node in analysis.nodes if node.kind not in ("input", "output"))
    total_flops = sum(int(node.flops) for node in analysis.nodes)

    model_payload = {
        "name": analysis.model_name or analysis.class_name or file_path.stem,
        "framework": "pytorch",
        "className": analysis.class_name,
        "mode": analysis.mode,
        "source": {"file": file_label, "relPath": rel_path},
        "inputSpec": [
            {
                "name": str(spec.get("name", "input")),
                "shape": _shape_list([spec.get("shape")])[0],
                "dtype": str(spec.get("dtype", "float32")),
            }
            for spec in analysis.input_specs
        ],
        "outputSpec": [
            {
                "name": str(spec.get("name", "output")),
                "shape": _shape_list([spec.get("shape")])[0],
                "dtype": str(spec.get("dtype", "float32")),
            }
            for spec in analysis.output_specs
        ],
        "totalParams": int(analysis.total_params),
        "trainableParams": int(analysis.trainable_params),
        "totalFlops": int(total_flops),
        "flopsApprox": bool(analysis.flops_approx),
        "numLayers": int(num_layers),
        "warnings": warnings,
    }

    graph = {
        "schemaVersion": schema.SCHEMA_VERSION,
        "generator": {"name": schema.GENERATOR_NAME, "version": schema.GENERATOR_VERSION},
        "model": model_payload,
        "nodes": nodes_payload,
        "edges": edges_payload,
        "groups": group_list,
    }
    errors = schema.validate_graph(graph)
    if errors:
        preview = "; ".join(errors[:6])
        LOGGER.error("generated invalid graph for %s: %s", file_path, preview)
        raise AnalyzeError(
            f"internal error: generated graph failed schema validation: {preview}",
            file=file_path,
        )
    return graph


# ---------------------------------------------------------------------------
# single model / file analysis
# ---------------------------------------------------------------------------

def _analyze_candidate(
    info: discover.ModuleFile,
    candidate: discover.ModelCandidate,
    source_index: SourceIndex,
    options: Any,
    file_path: Path,
    loaded_module: Any,
) -> Analysis:
    warnings: list[str] = []
    module_class = candidate.module_class

    if not options.execute:
        analysis = astscan.analyze_static(info, module_class, source_index, options, warnings=warnings)
        analysis.model_name = candidate.display_name
        analysis.class_name = candidate.class_name
        return analysis

    if not tracer.torch_available():
        warnings.append("PyTorch is not installed; fell back to static AST analysis (mode=static).")
        analysis = astscan.analyze_static(info, module_class, source_index, options, warnings=warnings)
        analysis.model_name = candidate.display_name
        analysis.class_name = candidate.class_name
        return analysis

    module_obj = loaded_module
    if module_obj is None:
        try:
            module_obj = tracer.load_file_module(file_path, info)
        except AnalyzeError as exc:
            LOGGER.debug("could not import module (%s); used static AST analysis", exc.message)
            analysis = astscan.analyze_static(info, module_class, source_index, options, warnings=warnings)
            analysis.model_name = candidate.display_name
            analysis.class_name = candidate.class_name
            return analysis

    synthesis = None
    try:
        model, how, synthesis = tracer.instantiate_with_synthesis(module_obj, candidate, info)
    except AnalyzeError as exc:
        warnings.append(
            f"could not instantiate {candidate.display_name} ({exc.message}); used static AST analysis."
        )
        analysis = astscan.analyze_static(info, module_class, source_index, options, warnings=warnings)
        analysis.model_name = candidate.display_name
        analysis.class_name = candidate.class_name
        return analysis

    if synthesis is not None:
        warnings.extend(synthesis.warnings)

    try:
        analysis = tracer.analyze_model_dynamic(model, info, candidate, options, source_index)
    except AnalyzeError as exc:
        warnings.append(f"dynamic tracing failed ({exc.message}); used static AST analysis.")
    except Exception as exc:  # noqa: BLE001 - tracing user code is best effort
        LOGGER.debug("dynamic tracing raised", exc_info=True)
        warnings.append(
            f"dynamic tracing failed ({type(exc).__name__}: {exc}); used static AST analysis."
        )
    else:
        if synthesis is not None:
            analysis.flops_approx = True
            warnings.append(
                "constructor arguments were synthesized; shapes/FLOPs reflect a synthesized configuration"
            )
        analysis.warnings = list(dict.fromkeys([*warnings, *analysis.warnings]))
        LOGGER.info("traced %s via %s (%d nodes)", candidate.display_name, how, len(analysis.nodes))
        return analysis

    analysis = astscan.analyze_static(info, module_class, source_index, options, warnings=warnings)
    analysis.model_name = candidate.display_name
    analysis.class_name = candidate.class_name
    return analysis


def analyze_file_records(
    path: str | Path,
    options: Any,
    *,
    scan_root: Path | None = None,
    external_components: set[str] | None = None,
) -> list[tuple[str, dict[str, Any]]]:
    """Analyze one file, returning ``(model_id, graph)`` pairs."""
    file_path = Path(path).expanduser()
    if not file_path.exists():
        raise AnalyzeError("file does not exist", file=file_path)
    if not file_path.is_file():
        raise AnalyzeError("not a file", file=file_path)
    if file_path.suffix.lower() != ".py":
        raise AnalyzeError("not a Python file", file=file_path)
    root = scan_root or file_path.parent

    try:
        info = discover.parse_source(file_path)
    except AnalyzeError:
        raise
    candidates = discover.find_top_models(
        info,
        options.class_name,
        external_components=external_components,
    )
    if not candidates:
        LOGGER.debug("no nn.Module models found in %s", file_path)
        return []

    source_index = SourceIndex.build(info.tree, info.source, file_path)
    loaded_module: Any = None
    records: list[tuple[str, dict[str, Any]]] = []
    failures: list[AnalyzeError] = []

    for candidate in candidates:
        try:
            analysis = _analyze_candidate(
                info, candidate, source_index, options, file_path, loaded_module
            )
            graph = make_graph(
                analysis,
                file_path=file_path,
                scan_root=root,
                class_name=candidate.class_name,
            )
        except AnalyzeError as exc:
            failures.append(exc)
            LOGGER.warning("model %s in %s failed: %s", candidate.display_name, file_path.name, exc)
            continue
        except Exception as exc:  # noqa: BLE001 - protect folder scans
            LOGGER.debug("model %s in %s crashed", candidate.display_name, file_path, exc_info=True)
            failures.append(
                AnalyzeError(
                    f"unexpected analysis failure: {type(exc).__name__}: {exc}",
                    file=file_path,
                    line=candidate.line,
                )
            )
            continue
        records.append((model_id(file_path, candidate.display_name), graph))

    if not records:
        if failures:
            raise failures[0]
        raise AnalyzeError("no model graph could be produced", file=file_path)
    return records


# ---------------------------------------------------------------------------
# folder scan
# ---------------------------------------------------------------------------

def _timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def analyze_folder_impl(folder: str | Path, options: Any) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Scan ``folder``; per-file failures are recorded in ``index.errors``."""
    root = Path(folder).expanduser()
    files = discover.iter_python_files(
        root,
        recursive=bool(options.recursive),
        max_files=int(options.max_files),
    )
    LOGGER.info("scanning %d file(s) under %s", len(files), root)

    # Folder-level pre-pass: classes instantiated as sub-modules by another
    # file's model class are components, not standalone top-level models.
    external_components = discover.scan_folder_components(files, root)

    models: list[dict[str, Any]] = []
    graphs: dict[str, dict[str, Any]] = {}
    collected: list[tuple[str, dict[str, Any], Path, str]] = []
    errors: list[dict[str, str]] = []

    for file_path in files:
        try:
            records = analyze_file_records(
                file_path,
                options,
                scan_root=root,
                external_components=external_components.get(file_path.resolve(), set()),
            )
        except AnalyzeError as exc:
            errors.append(
                {
                    "file": folder_relative(root, file_path),
                    "message": exc.describe() if exc.file else exc.message,
                }
            )
            LOGGER.warning("skipping %s: %s", folder_relative(root, file_path), exc)
            continue
        except Exception as exc:  # noqa: BLE001 - never abort a folder scan
            LOGGER.exception("unexpected failure while analyzing %s", file_path)
            errors.append(
                {
                    "file": folder_relative(root, file_path),
                    "message": f"unexpected error: {type(exc).__name__}: {exc}",
                }
            )
            continue

        for wanted_id, graph in records:
            collected.append((wanted_id, graph, file_path, folder_relative(root, file_path)))

    # Same-named architectures (e.g. fourteen ``ViT`` variants across a package)
    # are deduplicated: keep the canonical file for that name (``vit.py`` for
    # ``ViT``, ``simple_vit.py`` for ``SimpleViT``, ...).
    chosen: dict[str, tuple[str, dict[str, Any], Path, str]] = {}
    order: list[str] = []
    skipped_duplicates = 0
    for payload in collected:
        display_name = str(payload[1].get("model", {}).get("name") or "")
        if display_name not in chosen:
            chosen[display_name] = payload
            order.append(display_name)
            continue
        skipped_duplicates += 1
        current = chosen[display_name]
        if _canonical_score(payload[0], display_name) < _canonical_score(current[0], display_name):
            chosen[display_name] = payload
    if skipped_duplicates:
        LOGGER.info("deduplicated %d same-named model(s)", skipped_duplicates)

    taken: set[str] = set()
    for display_name in order:
        wanted_id, graph, _file_path, _source = chosen[display_name]
        unique_id = wanted_id
        suffix = 2
        while unique_id in taken:
            unique_id = f"{wanted_id}-{suffix}"
            suffix += 1
        taken.add(unique_id)
        graphs[unique_id] = graph
        model = graph["model"]
        models.append(
            {
                "id": unique_id,
                "name": model["name"],
                "className": model.get("className"),
                "file": f"{unique_id}.json",
                "source": model["source"]["relPath"],
                "mode": model["mode"],
                "numLayers": model["numLayers"],
                "totalParams": model["totalParams"],
                "totalFlops": model["totalFlops"],
                "warnings": len(model["warnings"]),
            }
        )

    index = {
        "schemaVersion": schema.SCHEMA_VERSION,
        "generatedAt": _timestamp(),
        "root": display_path(root),
        "models": models,
        "errors": errors,
    }
    index_errors = schema.validate_index(index)
    if index_errors:
        raise AnalyzeError(
            "internal error: generated index failed schema validation: "
            + "; ".join(index_errors[:6]),
            file=root,
        )
    return index, graphs


__all__ = [
    "analyze_file_records",
    "analyze_folder_impl",
    "make_graph",
    "model_id",
    "slugify",
]
