"""Graph loading and background parsing for the native GUI.

Parsing reuses the frozen library entry points from :mod:`netprism.cli`
(``_scan`` / ``_write_single_model``) and runs inside a ``QThread`` so the
window never freezes.  No HTTP server, browser or web engine is involved.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

from PySide6.QtCore import QThread, Signal

from ..schema import SCHEMA_VERSION


class ParseError(RuntimeError):
    """User-facing parse failure."""


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _slug(value: str) -> str:
    text = re.sub(r"[^0-9A-Za-z]+", "_", str(value or "model")).strip("_").lower()
    return text or "model"


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def demo_dir() -> Path | None:
    """Locate the bundled 8-model demo (source tree or frozen executable)."""
    candidates: list[Path] = []
    env = os.environ.get("NETPRISM_DEMO_DIR")
    if env:
        candidates.append(Path(env).expanduser())
    if getattr(sys, "frozen", False):
        base = Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
        candidates.append(base / "examples" / "demo")
        candidates.append(Path(sys.executable).resolve().parent / "examples" / "demo")
    candidates.append(project_root() / "examples" / "demo")
    candidates.append(project_root() / "demo")
    for candidate in candidates:
        if candidate.is_dir() and (candidate / "index.json").is_file():
            return candidate
    return None


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ParseError(f"无法读取 {path.name}: {exc}") from exc
    if not isinstance(data, dict):
        raise ParseError(f"{path.name} 不是有效的 graph JSON")
    return data


def _model_entry(graph: dict[str, Any], *, source: str, model_id: str) -> dict[str, Any]:
    model = graph.get("model") if isinstance(graph.get("model"), dict) else {}
    warnings = model.get("warnings") if isinstance(model.get("warnings"), list) else []
    return {
        "id": model_id,
        "name": model.get("name") or model.get("className") or model_id,
        "className": model.get("className"),
        "file": f"{model_id}.json",
        "source": source,
        "mode": model.get("mode") or model.get("framework") or "?",
        "numLayers": int(model.get("numLayers") or 0),
        "totalParams": int(model.get("totalParams") or 0),
        "totalFlops": int(model.get("totalFlops") or 0),
        "warnings": len(warnings),
    }


def _enrich_graph(graph: dict[str, Any]) -> None:
    """Best-effort metrics + algorithm annotations; never block display."""
    try:
        from ..metrics import enrich_graph

        enrich_graph(graph)
    except Exception:
        pass
    try:
        from ..algorithms import enrich_algorithms

        enrich_algorithms(graph)
    except Exception:
        pass


def _graphs_from_index(index: dict[str, Any], root: Path) -> tuple[dict[str, dict[str, Any]], list[dict[str, str]]]:
    graphs: dict[str, dict[str, Any]] = {}
    errors = [error for error in (index.get("errors") or []) if isinstance(error, dict)]
    for entry in index.get("models") or []:
        if not isinstance(entry, dict):
            continue
        model_id = entry.get("id")
        filename = entry.get("file")
        if not isinstance(model_id, str) or not isinstance(filename, str):
            continue
        path = root / filename
        try:
            graph = _read_json(path)
        except ParseError as exc:
            errors.append({"file": filename, "message": str(exc)})
            continue
        graphs[model_id] = graph
    return graphs, errors


def load_scan_dir(directory: str | Path) -> dict[str, Any]:
    """Load an already-written scan directory (``index.json`` + graphs)."""
    root = Path(directory).expanduser().resolve()
    index_path = root / "index.json"
    if not index_path.is_file():
        raise ParseError(f"{root} 不是 NetPrism 扫描目录（缺少 index.json）")
    index = _read_json(index_path)
    graphs, errors = _graphs_from_index(index, root)
    if not graphs:
        raise ParseError("扫描目录里没有可显示的模型")
    if errors:
        index = dict(index)
        index["errors"] = list(index.get("errors") or []) + errors
    return {
        "index": index,
        "graphs": graphs,
        "models": index.get("models") or [],
        "source_path": str(root),
        "source_kind": "scan_dir",
    }


def load_demo_payload() -> dict[str, Any]:
    folder = demo_dir()
    if folder is None:
        raise ParseError("找不到内置 examples/demo 数据")
    payload = load_scan_dir(folder)
    payload["source_path"] = str(folder)
    payload["source_kind"] = "demo"
    return payload


def load_graph_json(path: str | Path) -> dict[str, Any]:
    file_path = Path(path).expanduser().resolve()
    graph = _read_json(file_path)
    model = graph.get("model") if isinstance(graph.get("model"), dict) else {}
    framework = _slug(str(model.get("framework") or "unknown"))
    model_id = _slug(file_path.stem) or "model"
    if not model_id.endswith(f"__{framework}"):
        model_id = f"{model_id}__{framework}"
    entry = _model_entry(graph, source=file_path.name, model_id=model_id)
    index = {
        "schemaVersion": SCHEMA_VERSION,
        "root": str(file_path.parent),
        "models": [entry],
        "errors": [],
    }
    return {
        "index": index,
        "graphs": {model_id: graph},
        "models": [entry],
        "source_path": str(file_path),
        "source_kind": "file",
    }


def _payload_from_graphs(
    graphs: list[dict[str, Any]],
    *,
    source_path: Path,
    source_name: str,
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    graph_map: dict[str, dict[str, Any]] = {}
    used: set[str] = set()
    for graph in graphs:
        model = graph.get("model") if isinstance(graph.get("model"), dict) else {}
        framework = _slug(str(model.get("framework") or "unknown"))
        base = _slug(str(model.get("name") or source_path.stem))
        model_id = f"{base}__{framework}"
        suffix = 2
        while model_id in used:
            model_id = f"{base}_{suffix}__{framework}"
            suffix += 1
        used.add(model_id)
        _enrich_graph(graph)
        entries.append(_model_entry(graph, source=source_name, model_id=model_id))
        graph_map[model_id] = graph
    index = {
        "schemaVersion": SCHEMA_VERSION,
        "root": str(source_path.parent if source_path.is_file() else source_path),
        "models": entries,
        "errors": [],
    }
    return {
        "index": index,
        "graphs": graph_map,
        "models": entries,
        "source_path": str(source_path),
        "source_kind": "file",
    }


def _parse_folder(folder: Path, *, static: bool) -> dict[str, Any]:
    from netprism import cli as cli_module

    with tempfile.TemporaryDirectory(prefix="netprism_gui_scan_") as tmp:
        out = Path(tmp)
        sink = io.StringIO()
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            cli_module._scan(
                str(folder),
                out=str(out),
                execute=not static,
                class_name=None,
                input_shape=None,
                max_files=200,
                quiet=True,
                all_variants=False,
                include_formats=True,
            )
        index = _read_json(out / "index.json")
        graphs, errors = _graphs_from_index(index, out)
        if not graphs:
            message = "文件夹中没有找到可解析的模型。"
            if errors:
                message += f"\n{errors[0].get('file')}: {errors[0].get('message')}"
            raise ParseError(message)
        if errors:
            index = dict(index)
            index["errors"] = list(index.get("errors") or []) + errors
        for graph in graphs.values():
            _enrich_graph(graph)
        return {
            "index": index,
            "graphs": graphs,
            "models": index.get("models") or [],
            "source_path": str(folder),
            "source_kind": "folder",
        }


def _parse_python_static(file_path: Path) -> dict[str, Any]:
    from netprism.core import AnalyzeOptions, analyze_file

    try:
        graphs = analyze_file(file_path, AnalyzeOptions(execute=False))
    except Exception as exc:  # AnalyzeError and any parser failure
        raise ParseError(f"解析失败：{exc}") from exc
    if not graphs:
        raise ParseError("没有找到可解析的模型。")
    return _payload_from_graphs(graphs, source_path=file_path, source_name=file_path.name)


def _parse_single(file_path: Path, *, static: bool) -> dict[str, Any]:
    suffix = file_path.suffix.lower()
    if suffix == ".json":
        return load_graph_json(file_path)
    if suffix == ".py" and static:
        return _parse_python_static(file_path)
    from netprism import cli as cli_module

    with tempfile.TemporaryDirectory(prefix="netprism_gui_file_") as tmp:
        out = Path(tmp)
        sink = io.StringIO()
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            cli_module._write_single_model(file_path, out)
        index = _read_json(out / "index.json")
        graphs, errors = _graphs_from_index(index, out)
        if not graphs:
            message = f"没有找到可解析的模型：{file_path.name}"
            if errors:
                message += f"\n{errors[0].get('file')}: {errors[0].get('message')}"
            raise ParseError(message)
        for graph in graphs.values():
            _enrich_graph(graph)
        return {
            "index": index,
            "graphs": graphs,
            "models": index.get("models") or [],
            "source_path": str(file_path),
            "source_kind": "file",
        }


def parse_path_sync(path: str | Path, *, static: bool = False) -> dict[str, Any]:
    """Parse a folder or model file synchronously.

    ``static=True`` maps to ``AnalyzeOptions(execute=False)`` for Python
    sources.  A graph JSON is loaded directly and never annotated behind the
    user's back (so intentionally meta-less graphs keep their empty state).
    """
    source = Path(path).expanduser().resolve()
    if not source.exists():
        raise ParseError(f"路径不存在：{source}")
    if source.is_dir():
        # A folder with a pre-built index is already a scan output; load it
        # directly instead of re-parsing source files (the bundled demo and
        # previous CLI outputs both fall into this category).
        if (source / "index.json").is_file():
            payload = load_scan_dir(source)
            payload["source_kind"] = "folder"
            return payload
        return _parse_folder(source, static=static)
    if not source.is_file():
        raise ParseError(f"不支持的路径：{source}")
    try:
        return _parse_single(source, static=static)
    except ParseError:
        raise
    except SystemExit as exc:
        message = str(exc) or "解析进程提前退出"
        raise ParseError(message) from exc
    except Exception as exc:
        raise ParseError(f"解析失败：{exc}") from exc


class ParseWorker(QThread):
    """Background parse worker.  Emits the payload or a user-facing message."""

    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(self, path: str | Path, *, static: bool = False, parent=None) -> None:
        super().__init__(parent)
        self.path = str(path)
        self.static = bool(static)

    def run(self) -> None:  # pragma: no cover - timing is exercised by tests
        try:
            payload = parse_path_sync(self.path, static=self.static)
        except ParseError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:  # noqa: BLE001 - background border
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        else:
            self.succeeded.emit(payload)


__all__ = [
    "ParseError",
    "ParseWorker",
    "demo_dir",
    "load_demo_payload",
    "load_graph_json",
    "load_scan_dir",
    "parse_path_sync",
    "project_root",
]
