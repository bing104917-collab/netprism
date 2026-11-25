"""NetPrism parser core: discover, trace and describe PyTorch models.

Public API (frozen by ``SPEC.md`` section 4.1)::

    from netprism.core import AnalyzeOptions, analyze_file, analyze_folder

    graphs = analyze_file("samples/resnet_mini.py")
    result = analyze_folder("samples", AnalyzeOptions(execute=False))

The module top level deliberately does **not** import ``torch``: static analysis
must work in environments without PyTorch.  Torch is imported lazily by
:mod:`netprism.core.tracer`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import AnalyzeError

LOGGER = logging.getLogger("netprism.core")
LOGGER.addHandler(logging.NullHandler())


@dataclass
class AnalyzeOptions:
    """Options shared by :func:`analyze_file` and :func:`analyze_folder`."""

    execute: bool = True
    """Allow importing and executing model code (dynamic trace mode)."""

    class_name: str | None = None
    """Only analyze this top-level model class."""

    input_shape: tuple[int, ...] | None = None
    """Explicit dummy input shape; inferred automatically when ``None``."""

    unknown_input: bool = False
    """Static mode: leave spatial/sequence input dimensions at ``-1`` instead of
    guessing concrete sizes.  Ignored when an explicit ``input_shape`` is set."""

    max_files: int = 200
    """Upper bound on the number of ``.py`` files visited per folder scan."""

    recursive: bool = True
    """Recurse into sub-directories when scanning a folder."""

    verbose: bool = False
    """Enable debug logging for the parser core."""


@dataclass
class FolderResult:
    """Result of a folder scan: ``index`` document + ``id -> graph`` mapping."""

    index: dict[str, Any]
    graphs: dict[str, dict[str, Any]]


def analyze_file(
    path: str | Path,
    options: AnalyzeOptions | None = None,
) -> list[dict[str, Any]]:
    """Return every model graph discovered in ``path``.

    Raises :class:`AnalyzeError` when the file cannot be read, parsed, or when no
    model could be analyzed.  Files without any ``nn.Module`` model return ``[]``.
    """
    from . import build  # local import keeps torch out of module import time

    opts = options or AnalyzeOptions()
    if opts.verbose:
        LOGGER.setLevel(logging.DEBUG)
    records = build.analyze_file_records(path, opts)
    return [graph for _model_id, graph in records]


def analyze_folder(
    folder: str | Path,
    options: AnalyzeOptions | None = None,
) -> FolderResult:
    """Scan ``folder`` for PyTorch models; a broken file never aborts the scan."""
    from . import build

    opts = options or AnalyzeOptions()
    if opts.verbose:
        LOGGER.setLevel(logging.DEBUG)
    index, graphs = build.analyze_folder_impl(folder, opts)
    return FolderResult(index=index, graphs=graphs)


__all__ = [
    "AnalyzeError",
    "AnalyzeOptions",
    "FolderResult",
    "analyze_file",
    "analyze_folder",
]
