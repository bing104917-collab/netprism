"""Optional multi-format model adapters: ONNX, TorchScript and Keras v3.

Entry point used by the CLI / folder scanner::

    from netprism.adapters import FORMAT_SUFFIXES, is_format_file, analyze_format_file

    if is_format_file(path):
        graphs = analyze_format_file(path)

Design rules (see ``SPEC-V2.md`` sections 1 and 3):

* importing this package never imports ``torch`` / ``onnx`` / ``keras``;
* each adapter lazily imports its optional backend and raises
  :class:`~netprism.core.errors.AnalyzeError` with an install hint when it is
  missing;
* every returned graph satisfies :func:`netprism.schema.validate_graph`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from ..core.errors import AnalyzeError

#: Model-file suffix -> format name.
FORMAT_SUFFIXES: dict[str, str] = {
    ".onnx": "onnx",
    ".pt": "torchscript",
    ".pth": "torchscript",
    ".keras": "keras",
}

#: Human readable format names for CLI / UI labels.
FORMAT_LABELS: dict[str, str] = {
    "onnx": "ONNX",
    "torchscript": "TorchScript",
    "keras": "Keras v3",
}

#: Safety limit shared by all adapters (``analyze_format_file`` enforces it).
MAX_NODES_DEFAULT = 20000

_AdapterFn = Callable[..., list[dict[str, Any]]]


def format_for_path(path: str | Path) -> str | None:
    """Return the format name for ``path`` or ``None`` for unknown suffixes."""
    return FORMAT_SUFFIXES.get(Path(path).suffix.lower())


def is_format_file(path: str | Path) -> bool:
    """``True`` when ``path`` has a suffix supported by a format adapter."""
    return format_for_path(path) is not None


def _adapter(format_name: str) -> _AdapterFn:
    """Import the adapter lazily so optional backends stay optional."""
    if format_name == "onnx":
        from .onnx_adapter import analyze_onnx as fn
    elif format_name == "torchscript":
        from .torchscript_adapter import analyze_torchscript as fn
    elif format_name == "keras":
        from .keras_adapter import analyze_keras as fn
    else:  # pragma: no cover - defensive, format names come from FORMAT_SUFFIXES
        raise AnalyzeError(f"unsupported model format {format_name!r}")
    return fn


def analyze_format_file(
    path: str | Path,
    *,
    max_nodes: int = MAX_NODES_DEFAULT,
) -> list[dict[str, Any]]:
    """Analyze one ``.onnx`` / ``.pt`` / ``.pth`` / ``.keras`` model file.

    Returns one graph per model stored in the file (formats with a single model
    return exactly one graph).  Raises :class:`AnalyzeError` when the file is
    missing, the suffix is unsupported, the optional backend is missing, the
    file cannot be parsed, or the configured ``max_nodes`` limit is exceeded.
    """
    file_path = Path(path).expanduser()
    if not file_path.exists():
        raise AnalyzeError("file does not exist", file=file_path)
    if not file_path.is_file():
        raise AnalyzeError("not a file", file=file_path)
    format_name = format_for_path(file_path)
    if format_name is None:
        supported = ", ".join(sorted(FORMAT_SUFFIXES))
        raise AnalyzeError(
            f"unsupported model format {file_path.suffix or '(no suffix)'!r}; "
            f"supported suffixes: {supported}",
            file=file_path,
        )
    if max_nodes <= 0:
        raise AnalyzeError("max_nodes must be a positive integer", file=file_path)

    analyze = _adapter(format_name)
    graphs = analyze(file_path, max_nodes=max_nodes)
    if not graphs:
        raise AnalyzeError("no model graph could be produced", file=file_path)

    from .. import schema

    for graph in graphs:
        node_count = len(graph.get("nodes") or [])
        if node_count > max_nodes:
            raise AnalyzeError(
                f"model has {node_count} nodes, above the max_nodes limit ({max_nodes})",
                file=file_path,
            )
        errors = schema.validate_graph(graph)
        if errors:
            preview = "; ".join(errors[:6])
            raise AnalyzeError(
                f"internal error: generated graph failed schema validation: {preview}",
                file=file_path,
            )
    return graphs


__all__ = [
    "FORMAT_LABELS",
    "FORMAT_SUFFIXES",
    "MAX_NODES_DEFAULT",
    "analyze_format_file",
    "format_for_path",
    "is_format_file",
]
