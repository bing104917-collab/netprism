"""Command line interface for NetPrism.

Commands::

    netprism scan <folder> --out ./netprism_out
    netprism view <folder> --open
    netprism serve ./netprism_out --port 8765
    netprism inspect ./model.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from . import __version__
from .schema import DEMO_DIR, SCHEMA_VERSION, validate_graph, validate_index

_DEFAULT_OUT = "netprism_out"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _human(number: int | float) -> str:
    value = float(number)
    for unit in ("", "K", "M", "G", "T"):
        if abs(value) < 1000 or unit == "T":
            if unit == "":
                return f"{int(value)}"
            return f"{value:.1f}{unit}"
        value /= 1000.0
    return str(number)


def _parse_input_shape(value: str | None) -> tuple[int, ...] | None:
    if not value:
        return None
    parts = [p for p in re.split(r"[,\sxX*]+", value.strip()) if p]
    try:
        dims = tuple(int(p) for p in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"无法解析输入形状: {value!r}") from exc
    if not dims or any(d <= 0 for d in dims):
        raise argparse.ArgumentTypeError(f"输入形状必须全为正整数: {value!r}")
    return dims


def _shape_text(shapes: list[list[int]] | None) -> str:
    if not shapes:
        return "-"
    return " + ".join("×".join(str(d) for d in shape) for shape in shapes)


def _source_text(source: dict[str, Any] | None) -> str:
    if not source:
        return "-"
    code = source.get("code")
    if isinstance(code, str) and code:
        return code.strip()
    line = source.get("line")
    return f"line {line}" if line else "-"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _load_core():
    try:
        from .core import AnalyzeError, AnalyzeOptions, analyze_file, analyze_folder
    except ImportError as exc:  # pragma: no cover - early development / broken install
        print(
            "错误：解析引擎不可用（netprism.core 导入失败）。\n"
            f"  {exc}\n"
            "  请确认在项目根目录运行，或安装 netprism 包。",
            file=sys.stderr,
        )
        raise SystemExit(2) from exc
    return AnalyzeError, AnalyzeOptions, analyze_file, analyze_folder


def _slug(text: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", str(text).strip().lower()).strip("_")
    return value or "model"


_SKIP_DIRS = {
    ".git", ".hg", ".svn", "__pycache__", ".pytest_cache", ".mypy_cache",
    "node_modules", ".venv", "venv", "env", "site-packages", "build", "dist",
    "netprism_out", ".netprism_out", "scratch",
}


def _scan_format_files(folder: Path, *, max_files: int) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], list[dict[str, str]]]:
    """Parse non-Python model files (ONNX / TorchScript / Keras) in ``folder``."""
    try:
        from .adapters import FORMAT_SUFFIXES, analyze_format_file
    except ImportError:
        return [], {}, []
    files: list[Path] = []
    for path in sorted(folder.rglob("*")):
        if len(files) >= max_files:
            break
        if not path.is_file():
            continue
        if any(part in _SKIP_DIRS for part in path.parts):
            continue
        if path.suffix.lower() not in FORMAT_SUFFIXES:
            continue
        files.append(path)

    models: list[dict[str, Any]] = []
    graphs: dict[str, dict[str, Any]] = {}
    errors: list[dict[str, str]] = []
    for path in files:
        rel = path.relative_to(folder).as_posix()
        try:
            parsed = analyze_format_file(path)
        except Exception as exc:  # noqa: BLE001 - one bad file must not abort the scan
            errors.append({"file": rel, "message": f"{type(exc).__name__}: {exc}"})
            continue
        for graph in parsed:
            if not isinstance(graph, dict):
                continue
            model = graph.get("model") or {}
            framework = str(model.get("framework") or "unknown")
            base = f"{_slug(path.stem)}__{_slug(framework)}"
            model_id = base
            suffix = 2
            while model_id in graphs:
                model_id = f"{base}_{suffix}"
                suffix += 1
            graphs[model_id] = graph
            models.append(
                {
                    "id": model_id,
                    "name": model.get("name") or path.stem,
                    "className": model.get("className"),
                    "file": f"{model_id}.json",
                    "source": rel,
                    "mode": model.get("mode", framework),
                    "numLayers": int(model.get("numLayers", 0)),
                    "totalParams": int(model.get("totalParams", 0)),
                    "totalFlops": int(model.get("totalFlops", 0)),
                    "warnings": len(model.get("warnings", [])),
                }
            )
    return models, graphs, errors


def _enrich_metrics(graphs: dict[str, dict[str, Any]]) -> None:
    """Add optional metrics + algorithm semantics when the modules are available."""
    try:
        from .metrics import enrich_graph
    except ImportError:
        enrich_graph = None  # type: ignore[assignment]
    try:
        from .algorithms import enrich_algorithms
    except ImportError:
        enrich_algorithms = None  # type: ignore[assignment]
    for graph in graphs.values():
        if enrich_graph is not None:
            try:
                enrich_graph(graph)
            except Exception:  # noqa: BLE001 - metrics must not block parsing
                pass
        if enrich_algorithms is not None:
            try:
                enrich_algorithms(graph)
            except Exception:  # noqa: BLE001 - algorithm hints must not block parsing
                pass


def _load_graph_source(source: str, *, model_id: str | None = None) -> dict[str, Any]:
    """Load a graph from a graph JSON, model file, .py file, or scan directory."""
    path = Path(source).expanduser().resolve()
    if not path.exists():
        raise SystemExit(f"错误：路径不存在: {path}")

    def _pick_entry(index: dict[str, Any]) -> dict[str, Any]:
        models = index.get("models") or []
        if not models:
            raise SystemExit(f"错误：{path} 中没有模型")
        if model_id:
            for entry in models:
                if entry.get("id") == model_id or entry.get("name") == model_id:
                    return entry
            known = ", ".join(str(entry.get("id")) for entry in models[:10])
            raise SystemExit(f"错误：找不到模型 {model_id!r}；可选：{known}")
        return models[0]

    if path.is_dir():
        index_path = path / "index.json"
        if not index_path.is_file():
            raise SystemExit(f"错误：{path} 中没有 index.json，请先运行 netprism scan")
        index = json.loads(index_path.read_text(encoding="utf-8"))
        entry = _pick_entry(index)
        graph_path = (path / str(entry.get("file"))).resolve()
        try:
            graph_path.relative_to(path)
        except ValueError as exc:
            raise SystemExit(f"错误：index.json 指向非法路径 {graph_path}") from exc
        return json.loads(graph_path.read_text(encoding="utf-8"))

    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if path.suffix.lower() == ".py":
        AnalyzeError, AnalyzeOptions, analyze_file, _ = _load_core()
        try:
            graphs = analyze_file(path, AnalyzeOptions(execute=True))
        except AnalyzeError as exc:
            raise SystemExit(f"错误：{exc}") from exc
        if not graphs:
            raise SystemExit(f"错误：{path} 中没有模型")
        return graphs[0]

    try:
        from .adapters import analyze_format_file
    except ImportError as exc:
        raise SystemExit("错误：格式适配器不可用（netprism.adapters）") from exc
    graphs = analyze_format_file(path)
    if not graphs:
        raise SystemExit(f"错误：{path} 中没有模型")
    return graphs[0]


def _emit_text(text: str, out: str | None) -> None:
    if out:
        target = Path(out).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        # newline="" keeps the \r\n already produced by csv.writer intact.
        with tmp.open("w", encoding="utf-8", newline="") as handle:
            handle.write(text)
        tmp.replace(target)
        print(f"已写入 {target}")
    else:
        # Binary stdout avoids Windows adding a second \r to CSV line endings.
        sys.stdout.flush()
        data = text.encode("utf-8")
        sys.stdout.buffer.write(data)
        if not data.endswith(b"\n"):
            sys.stdout.buffer.write(b"\n")
        sys.stdout.buffer.flush()


def _scan(folder: str, *, out: str, execute: bool, class_name: str | None,
          input_shape: tuple[int, ...] | None, max_files: int, quiet: bool,
          all_variants: bool = False, include_formats: bool = True) -> tuple[dict[str, Any], dict[str, Any]]:
    AnalyzeError, AnalyzeOptions, _analyze_file, analyze_folder = _load_core()
    source = Path(folder).expanduser().resolve()
    if not source.exists():
        print(f"错误：路径不存在: {source}", file=sys.stderr)
        raise SystemExit(2)
    options = AnalyzeOptions(
        execute=execute,
        class_name=class_name,
        input_shape=input_shape,
        max_files=max_files,
    )
    started = time.perf_counter()
    try:
        result = analyze_folder(source, options)
    except AnalyzeError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    elapsed = time.perf_counter() - started

    index = dict(result.index)
    graphs = dict(result.graphs)
    if include_formats and source.is_dir() and not class_name:
        format_models, format_graphs, format_errors = _scan_format_files(source, max_files=max_files)
        index["models"] = list(index.get("models", [])) + format_models
        index["errors"] = list(index.get("errors", [])) + format_errors
        graphs.update(format_graphs)

    from .selection import fold_variants, format_folded_note

    index, graphs, dropped = fold_variants(index, graphs, enabled=not all_variants)
    if class_name:
        marker = f"class '{class_name}' not found"
        index["errors"] = [
            error for error in index.get("errors", [])
            if marker not in str(error.get("message", ""))
        ]
    _enrich_metrics(graphs)
    index.setdefault("schemaVersion", SCHEMA_VERSION)
    index.setdefault("generatedAt", time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    index.setdefault("root", str(source))
    index.setdefault("errors", [])
    out_dir = Path(out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    invalid: list[str] = []
    written = 0
    for entry in index.get("models", []):
        model_id = entry.get("id")
        graph = graphs.get(model_id)
        if not isinstance(graph, dict):
            continue
        errors = validate_graph(graph)
        if errors:
            invalid.append(f"{model_id}: {errors[0]}")
            if not quiet:
                for err in errors[:3]:
                    print(f"  ! schema: {err}", file=sys.stderr)
        filename = Path(str(entry.get("file") or f"{model_id}.json")).name
        entry["file"] = filename
        _write_json(out_dir / filename, graph)
        written += 1
    index_errors = validate_index(index)
    if index_errors and not quiet:
        for err in index_errors[:5]:
            print(f"  ! index: {err}", file=sys.stderr)
    _write_json(out_dir / "index.json", index)

    if not quiet:
        _print_scan_summary(source, index, elapsed, out_dir, written, invalid)
        note = format_folded_note(dropped)
        if note:
            print("\n" + note)
    return index, {"out_dir": str(out_dir), "invalid": invalid, "folded": len(dropped)}


def _print_scan_summary(source: Path, index: dict[str, Any], elapsed: float,
                        out_dir: Path, written: int, invalid: list[str]) -> None:
    models = index.get("models", [])
    print(f"\nNetPrism 解析完成：{len(models)} 个模型，用时 {elapsed:.2f}s")
    print(f"  源目录：{source}")
    print(f"  输出  ：{out_dir}")
    if models:
        header = f"{'MODE':<12} {'LAYERS':>6} {'PARAMS':>9} {'FLOPS':>9}  {'MODEL':<32} SOURCE"
        print("\n" + header)
        print("-" * len(header))
        shown = models if len(models) <= 40 else models[:40]
        for entry in shown:
            warnings = entry.get("warnings") or 0
            flag = f"  !{warnings}" if warnings else ""
            print(
                f"{str(entry.get('mode', '?')):<12} {int(entry.get('numLayers', 0)):>6} "
                f"{_human(int(entry.get('totalParams', 0))):>9} {_human(int(entry.get('totalFlops', 0))):>9}  "
                f"{str(entry.get('name', '?'))[:32]:<32} {entry.get('source', '')}{flag}"
            )
        if len(models) > len(shown):
            print(f"... 还有 {len(models) - len(shown)} 个模型，完整列表见 index.json")
        print(f"\n已写入 {written} 个 graph 文件 + index.json")
    errors = index.get("errors", [])
    if errors:
        print(f"\n跳过的文件（{len(errors)}）：")
        for err in errors[:10]:
            print(f"  - {err.get('file')}: {err.get('message')}")
        if len(errors) > 10:
            print(f"  ... 还有 {len(errors) - 10} 条")
    if invalid:
        print(f"\n警告：{len(invalid)} 个 graph 未通过 schema 校验（仍已写出，供调试）。", file=sys.stderr)


def _print_tree(graph: dict[str, Any]) -> None:
    model = graph.get("model", {})
    print(
        f"\n{model.get('name', '?')}  [{model.get('mode', '?')}]  "
        f"params={model.get('totalParams', 0)}  flops={_human(int(model.get('totalFlops', 0)))}  "
        f"source={model.get('source', {}).get('relPath', '-')}"
    )
    ins = model.get("inputSpec") or []
    outs = model.get("outputSpec") or []

    def spec_text(spec: dict[str, Any]) -> str:
        return f"{spec.get('name')} {_shape_text([spec.get('shape', [])])}"

    if ins:
        print("  input : " + ", ".join(spec_text(spec) for spec in ins))
    if outs:
        print("  output: " + ", ".join(spec_text(spec) for spec in outs))

    nodes = graph.get("nodes", [])
    node_by_id = {n.get("id"): n for n in nodes}
    groups = graph.get("groups", [])
    group_by_id = {g.get("id"): g for g in groups}
    children: dict[str | None, list[dict[str, Any]]] = {}
    for group in groups:
        children.setdefault(group.get("parentId"), []).append(group)
    for bucket in children.values():
        bucket.sort(key=lambda g: g.get("depth", 0) * 100000 + _first_node_index(g, nodes))
    order = {id(n): i for i, n in enumerate(nodes)}

    def node_line(node: dict[str, Any], prefix: str, last: bool) -> None:
        branch = "└─ " if last else "├─ "
        name = str(node.get("name", "?"))
        op = str(node.get("op", "?"))
        in_s = _shape_text(node.get("inShapes"))
        out_s = _shape_text(node.get("outShapes"))
        params = int(node.get("numParams", 0))
        extra = f"  {_source_text(node.get('source'))}" if node.get("source") else ""
        print(
            f"{prefix}{branch}{name:<32} {op:<20} {in_s} -> {out_s}  "
            f"p={_human(params):<7} f={_human(int(node.get('flops', 0))):<7}{extra}"
        )

    def print_group(group: dict[str, Any], prefix: str, last: bool) -> None:
        branch = "└─ " if last else "├─ "
        label = group.get("label") or group.get("name") or "?"
        print(
            f"{prefix}{branch}[{label}]  {group.get('kind', '')}  "
            f"p={_human(int(group.get('numParams', 0)))}  f={_human(int(group.get('flops', 0)))}"
        )
        child_prefix = prefix + ("   " if last else "│  ")
        member_ids = group.get("nodeIds", [])
        child_groups = sorted(
            [group_by_id[c] for c in group.get("childGroupIds", []) if c in group_by_id],
            key=lambda g: _first_node_index(g, nodes),
        )
        entries: list[tuple[int, str, Any]] = []
        for nid in member_ids:
            node = node_by_id.get(nid)
            if node:
                entries.append((-1 if node.get("kind") == "input" else order.get(id(node), 0), "node", node))
        for child in child_groups:
            entries.append((_first_node_index(child, nodes), "group", child))
        entries.sort(key=lambda item: item[0])
        for i, (_idx, typ, obj) in enumerate(entries):
            is_last = i == len(entries) - 1
            if typ == "node":
                node_line(obj, child_prefix, is_last)
            else:
                print_group(obj, child_prefix, is_last)

    roots = children.get(None, [])
    for i, group in enumerate(roots):
        print_group(group, "  ", i == len(roots) - 1)

    grouped = {nid for g in groups for nid in g.get("nodeIds", [])}
    loose = [n for n in nodes if n.get("id") not in grouped]
    if loose:
        print("\n  Ungrouped layers:")
        for i, node in enumerate(loose):
            node_line(node, "  ", i == len(loose) - 1)

    warnings = model.get("warnings") or []
    if warnings:
        print(f"\n  Warnings ({len(warnings)}):")
        for warning in warnings:
            print(f"   - {warning}")


def _first_node_index(group: dict[str, Any], nodes: list[dict[str, Any]]) -> int:
    order = {n.get("id"): i for i, n in enumerate(nodes)}
    indexes = [order[nid] for nid in group.get("nodeIds", []) if nid in order]
    return min(indexes) if indexes else 10**9


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #
def _write_single_model(path: Path, out: Path) -> dict[str, Any]:
    """Analyze one model file (py/onnx/pt/keras) and write a one-model scan dir."""
    graph = _load_graph_source(str(path))
    try:
        from .metrics import enrich_graph
    except ImportError:
        enrich_graph = None  # type: ignore[assignment]
    if enrich_graph:
        enrich_graph(graph)
    model = graph.get("model") or {}
    framework = str(model.get("framework") or "unknown")
    model_id = f"{_slug(path.stem)}__{_slug(framework)}"
    filename = f"{model_id}.json"
    out.mkdir(parents=True, exist_ok=True)
    _write_json(out / filename, graph)
    entry = {
        "id": model_id,
        "name": model.get("name") or path.stem,
        "className": model.get("className"),
        "file": filename,
        "source": path.name,
        "mode": model.get("mode", framework),
        "numLayers": int(model.get("numLayers", 0)),
        "totalParams": int(model.get("totalParams", 0)),
        "totalFlops": int(model.get("totalFlops", 0)),
        "warnings": len(model.get("warnings", [])),
    }
    index = {
        "schemaVersion": SCHEMA_VERSION,
        "generatedAt": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "root": str(path.parent),
        "models": [entry],
        "errors": [],
    }
    _write_json(out / "index.json", index)
    print(f"NetPrism 解析完成：{entry['name']}  [{entry['mode']}]  {entry['numLayers']} 层")
    return index


def cmd_scan(args: argparse.Namespace) -> int:
    result, _ = _scan(
        args.folder,
        out=args.out,
        execute=args.execute,
        class_name=args.class_name,
        input_shape=args.input_shape,
        max_files=args.max_files,
        quiet=args.quiet,
        all_variants=args.all_variants,
        include_formats=not args.no_formats,
    )
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def cmd_explain(args: argparse.Namespace) -> int:
    graph = _load_graph_source(args.source, model_id=args.model)
    try:
        from .algorithms import enrich_algorithms, explain_graph, render_explain_md
    except ImportError as exc:
        print(f"错误：算法理解模块不可用（{exc}）", file=sys.stderr)
        return 2
    enrich_algorithms(graph)
    if args.json:
        print(json.dumps(explain_graph(graph), ensure_ascii=False, indent=2))
    else:
        print(render_explain_md(graph))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    graph = _load_graph_source(args.source, model_id=args.model)
    try:
        from .metrics import enrich_graph
        from .report import render_report
    except ImportError as exc:
        print(f"错误：分析/报告模块不可用（{exc}）", file=sys.stderr)
        return 2
    enrich_graph(graph)
    try:
        from .algorithms import enrich_algorithms
        enrich_algorithms(graph)
    except ImportError:
        pass
    text = render_report(graph, fmt=args.format)
    _emit_text(text, args.out)
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    graph_a = _load_graph_source(args.source_a, model_id=args.model_a)
    graph_b = _load_graph_source(args.source_b, model_id=args.model_b)
    try:
        from .compare import compare_graphs, render_compare
        from .metrics import enrich_graph
    except ImportError as exc:
        print(f"错误：分析/对比模块不可用（{exc}）", file=sys.stderr)
        return 2
    enrich_graph(graph_a)
    enrich_graph(graph_b)
    result = compare_graphs(graph_a, graph_b)
    text = render_compare(result, fmt=args.format)
    _emit_text(text, args.out)
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    path = Path(args.path).expanduser()
    if not path.is_file():
        print(f"错误：文件不存在: {path}", file=sys.stderr)
        return 2
    if path.suffix.lower() != ".py":
        try:
            from .adapters import analyze_format_file
        except ImportError:
            print("错误：格式适配器不可用（netprism.adapters）", file=sys.stderr)
            return 2
        try:
            graphs = analyze_format_file(path)
        except Exception as exc:  # noqa: BLE001 - present a readable error
            print(f"错误：{type(exc).__name__}: {exc}", file=sys.stderr)
            return 2
    else:
        AnalyzeError, AnalyzeOptions, analyze_file, _analyze_folder = _load_core()
        try:
            graphs = analyze_file(
                path,
                AnalyzeOptions(execute=args.execute, class_name=args.class_name, input_shape=args.input_shape),
            )
        except AnalyzeError as exc:
            print(f"错误：{exc}", file=sys.stderr)
            return 2
    if not graphs:
        print("没有找到可解析的模型。", file=sys.stderr)
        return 1
    if args.json:
        payload: Any = graphs[0] if len(graphs) == 1 else graphs
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    try:
        from .metrics import enrich_graph
    except ImportError:
        enrich_graph = None  # type: ignore[assignment]
    for graph in graphs:
        if enrich_graph:
            enrich_graph(graph)
        _print_tree(graph)
    return 0


def _serve_dir(directory: str | None, *, host: str, port: int, open_browser: bool,
               web_dir: str | None, allow_execute: bool) -> int:
    from .server import create_app, serve

    data_dir = Path(directory).expanduser() if directory else DEMO_DIR
    if not data_dir.exists():
        print(f"错误：数据目录不存在: {data_dir}", file=sys.stderr)
        return 2
    app = create_app(data_dir, web_dir, allow_execute=allow_execute)
    if app.build_index() is None:
        print(f"提示：{data_dir} 中没有 index.json，UI 会显示空状态。", file=sys.stderr)
    try:
        serve(app, host=host, port=port, open_browser=open_browser)
    except OSError as exc:
        print(f"错误：无法监听 {host}:{port} — {exc}", file=sys.stderr)
        return 2
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    return _serve_dir(
        args.directory,
        host=args.host,
        port=args.port,
        open_browser=args.open,
        web_dir=args.web_dir,
        allow_execute=args.allow_execute,
    )


def cmd_view(args: argparse.Namespace) -> int:
    source = Path(args.folder).expanduser()
    if source.is_file():
        _write_single_model(source, Path(args.out).expanduser().resolve())
    else:
        _scan(
            args.folder,
            out=args.out,
            execute=args.execute,
            class_name=args.class_name,
            input_shape=args.input_shape,
            max_files=args.max_files,
            quiet=args.quiet,
            all_variants=args.all_variants,
            include_formats=not args.no_formats,
        )
    return _serve_dir(
        args.out,
        host=args.host,
        port=args.port,
        open_browser=args.open,
        web_dir=args.web_dir,
        allow_execute=args.allow_execute,
    )


# --------------------------------------------------------------------------- #
# parser
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="netprism",
        description="NetPrism — 一键解析深度神经网络，并在浏览器中 3D 可视化。",
        epilog="示例：netprism view .\\models --open",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__} (schema {SCHEMA_VERSION})")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_analysis_options(p: argparse.ArgumentParser, *, execute_default: bool = True,
                             variants: bool = False, formats: bool = False) -> None:
        p.add_argument("--class", dest="class_name", default=None, help="只解析指定的模型类名")
        p.add_argument("--input-shape", type=_parse_input_shape, default=None, metavar="1,3,32,32",
                       help="trace 模式使用的 dummy 输入形状，如 1,3,32,32")
        p.add_argument("--max-files", type=int, default=200, help="最多扫描的 .py 文件数（默认 200）")
        p.add_argument("--no-execute", "--static", dest="execute", action="store_false", default=execute_default,
                       help="不执行模型代码，只用静态 AST 解析（安全模式）")
        if variants:
            p.add_argument("--all-variants", action="store_true",
                           help="保留同一类在同一文件里的全部工厂变体（默认只保留第一个）")
        if formats:
            p.add_argument("--no-formats", action="store_true",
                           help="只扫描 .py，不解析 .onnx/.pt/.pth/.keras")

    scan = sub.add_parser("scan", help="解析文件夹并写出 JSON 结果")
    scan.add_argument("folder", help="模型代码文件夹")
    scan.add_argument("--out", default=_DEFAULT_OUT, help=f"输出目录（默认 {_DEFAULT_OUT}）")
    scan.add_argument("--json", action="store_true", help="额外输出 index JSON 到 stdout")
    scan.add_argument("--quiet", action="store_true", help="不打印摘要")
    add_analysis_options(scan, variants=True, formats=True)

    inspect = sub.add_parser("inspect", help="在终端中查看单个文件的结构")
    inspect.add_argument("path", help="Python / ONNX / TorchScript / Keras 模型文件")
    inspect.add_argument("--json", action="store_true", help="输出 graph JSON")
    add_analysis_options(inspect)

    report = sub.add_parser("report", help="生成论文/评审用的结构报告（md/csv/html）")
    report.add_argument("source", help="graph.json、scan 输出目录或模型文件")
    report.add_argument("--model", default=None, help="scan 目录里的模型 id/name")
    report.add_argument("--format", choices=("md", "csv", "html"), default="md")
    report.add_argument("--out", default=None, help="写入文件；默认打印到 stdout")

    compare = sub.add_parser("compare", help="对比两个模型的层/参数/FLOPs 差异")
    compare.add_argument("source_a", help="模型 A：graph.json、scan 目录或模型文件")
    compare.add_argument("source_b", help="模型 B：graph.json、scan 目录或模型文件")
    compare.add_argument("--model-a", default=None)
    compare.add_argument("--model-b", default=None)
    compare.add_argument("--format", choices=("md", "json", "csv"), default="md")
    compare.add_argument("--out", default=None, help="写入文件；默认打印到 stdout")

    explain = sub.add_parser("explain", help="输出算法理解：家族/模式公式/forward 伪代码")
    explain.add_argument("source", help="graph.json、scan 输出目录或模型文件")
    explain.add_argument("--model", default=None, help="scan 目录里的模型 id/name")
    explain.add_argument("--json", action="store_true", help="输出机器可读 JSON")

    serve = sub.add_parser("serve", help="启动 3D Web 服务")
    serve.add_argument("directory", nargs="?", default=None, help="scan 输出目录（默认 examples/demo）")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--open", action="store_true", help="自动打开浏览器")
    serve.add_argument("--allow-execute", action="store_true", help="允许 Web 端上传后执行模型代码（危险）")
    serve.add_argument("--web-dir", default=None, help="覆盖 web/ 目录")

    view = sub.add_parser("view", help="一键解析 + 启动 3D Web 服务")
    view.add_argument("folder", help="模型代码文件夹")
    view.add_argument("--out", default=".netprism_out", help="扫描输出目录（默认 .netprism_out）")
    view.add_argument("--host", default="127.0.0.1")
    view.add_argument("--port", type=int, default=8765)
    view.add_argument("--open", action="store_true", help="自动打开浏览器")
    view.add_argument("--allow-execute", action="store_true", help="允许 Web 端上传后执行模型代码（危险）")
    view.add_argument("--web-dir", default=None, help="覆盖 web/ 目录")
    view.add_argument("--quiet", action="store_true")
    add_analysis_options(view, variants=True, formats=True)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - non-standard streams
        pass
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "scan":
            return cmd_scan(args)
        if args.command == "inspect":
            return cmd_inspect(args)
        if args.command == "report":
            return cmd_report(args)
        if args.command == "compare":
            return cmd_compare(args)
        if args.command == "explain":
            return cmd_explain(args)
        if args.command == "serve":
            return cmd_serve(args)
        if args.command == "view":
            return cmd_view(args)
    except KeyboardInterrupt:
        print("\n已取消。", file=sys.stderr)
        return 130
    parser.error(f"未知命令: {args.command}")
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
