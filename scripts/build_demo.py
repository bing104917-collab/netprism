"""Generate ``examples/demo`` from ``samples/`` (NetPrism demo data).

Usage (from the project root)::

    & $py scripts/build_demo.py                 # trace demo (default)
    & $py scripts/build_demo.py --static        # static-only demo
    & $py scripts/build_demo.py --samples DIR --out DIR

Writes ``index.json`` plus one ``<id>.json`` per model and validates every
document with ``netprism.schema`` (and ``jsonschema`` when available).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from netprism import schema  # noqa: E402
from netprism.core import AnalyzeOptions, analyze_folder  # noqa: E402


def _dump(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def build_demo(samples: Path, out: Path, *, execute: bool) -> int:
    if not samples.is_dir():
        print(f"error: samples folder not found: {samples}", file=sys.stderr)
        return 2
    result = analyze_folder(samples, AnalyzeOptions(execute=execute))
    index = result.index
    graphs = result.graphs
    try:
        from netprism.metrics import enrich_graph
    except ImportError:
        enrich_graph = None  # type: ignore[assignment]
    try:
        from netprism.algorithms import enrich_algorithms
    except ImportError:
        enrich_algorithms = None  # type: ignore[assignment]
    if enrich_graph is not None:
        for graph in graphs.values():
            enrich_graph(graph)
    if enrich_algorithms is not None:
        for graph in graphs.values():
            enrich_algorithms(graph)

    index_errors = schema.validate_index(index)
    if index_errors:
        print("error: generated index failed schema validation:", file=sys.stderr)
        for message in index_errors[:10]:
            print(f"  - {message}", file=sys.stderr)
        return 1

    out.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    jsonschema_module = None
    try:  # optional dev dependency
        import jsonschema as jsonschema_module  # type: ignore[no-redef]
    except ImportError:  # pragma: no cover
        jsonschema_module = None

    json_schema = schema.load_json_schema() if jsonschema_module is not None else None
    for entry in index["models"]:
        model_id = entry["id"]
        graph = graphs.get(model_id)
        if graph is None:
            failures.append(f"{model_id}: missing graph")
            continue
        errors = schema.validate_graph(graph)
        if errors:
            failures.append(f"{model_id}: {'; '.join(errors[:4])}")
            continue
        if jsonschema_module is not None and json_schema is not None:
            try:
                jsonschema_module.validate(instance=graph, schema=json_schema)
            except jsonschema_module.ValidationError as exc:  # pragma: no cover
                failures.append(f"{model_id}: JSON Schema: {exc.message}")
                continue
        _dump(out / entry["file"], graph)

    _dump(out / "index.json", index)
    if failures:
        print("error: demo graph validation failed:", file=sys.stderr)
        for message in failures:
            print(f"  - {message}", file=sys.stderr)
        return 1

    print(f"demo written: {out}  (root={index['root']})")
    print(f"{'id':<52}{'mode':>7}{'layers':>8}{'params':>12}{'flops':>14}  warn")
    for entry in index["models"]:
        print(
            f"{entry['id']:<52}{entry['mode']:>7}{entry['numLayers']:>8}"
            f"{entry['totalParams']:>12}{entry['totalFlops']:>14}  {entry['warnings']}"
        )
    if index["errors"]:
        print("scan errors:")
        for error in index["errors"]:
            print(f"  - {error['file']}: {error['message']}")
    print(f"{len(index['models'])} model(s), {len(index['errors'])} error(s)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate NetPrism demo data from samples/")
    parser.add_argument(
        "--samples",
        default=str(PROJECT_ROOT / "samples"),
        help="folder containing sample models (default: <project>/samples)",
    )
    parser.add_argument(
        "--out",
        default=str(PROJECT_ROOT / "examples" / "demo"),
        help="output folder (default: <project>/examples/demo)",
    )
    parser.add_argument(
        "--static",
        action="store_true",
        help="do not execute model code; produce static-mode graphs",
    )
    args = parser.parse_args(argv)
    return build_demo(Path(args.samples), Path(args.out), execute=not args.static)


if __name__ == "__main__":
    raise SystemExit(main())
