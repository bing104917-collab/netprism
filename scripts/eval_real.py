"""Evaluate NetPrism on real-world repositories (robustness benchmark).

Usage (from the project root)::

    & $py scripts/eval_real.py
    & $py scripts/eval_real.py --repos pytorch-cifar Pytorch-UNet --json out.json
    & $py scripts/eval_real.py --root D:\\my_repos

The evaluation root defaults to ``%TEMP%\\netprism_eval`` and must contain one
sub-folder per repository.  The script only reads the repositories; it never
writes to them.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from netprism.core import AnalyzeOptions, analyze_folder  # noqa: E402
from netprism.schema import validate_graph, validate_index  # noqa: E402

DEFAULT_REPOS = ("pytorch-cifar", "Pytorch-UNet", "nanoGPT", "vit-pytorch")

#: Models the task cares about (used for the summary markers).
INTERESTING = {
    "pytorch-cifar": ("ResNet18", "ResNet34", "ResNet50", "DenseNet121", "VGG"),
    "Pytorch-UNet": ("UNet",),
    "nanoGPT": ("GPT",),
    "vit-pytorch": ("ViT", "SimpleViT", "CaiT", "CrossViT"),
}


def evaluate_repo(root: Path, repo: str, options: AnalyzeOptions) -> dict:
    folder = root / repo
    if not folder.is_dir():
        return {"repo": repo, "missing": True, "error": f"not found: {folder}"}
    started = time.perf_counter()
    payload: dict = {"repo": repo}
    try:
        result = analyze_folder(folder, options)
    except Exception as exc:  # noqa: BLE001 - report, never crash the benchmark
        payload.update({"error": f"{type(exc).__name__}: {exc}", "elapsed": round(time.perf_counter() - started, 2)})
        return payload
    elapsed = time.perf_counter() - started
    index = result.index
    graphs = result.graphs
    models = []
    trace_count = 0
    static_count = 0
    invalid = 0
    for entry in index["models"]:
        graph = graphs.get(entry["id"])
        if graph is None:
            continue
        errors = validate_graph(graph)
        if errors:
            invalid += 1
        if entry["mode"] == "trace":
            trace_count += 1
        else:
            static_count += 1
        models.append(
            {
                "id": entry["id"],
                "name": entry["name"],
                "className": entry.get("className"),
                "mode": entry["mode"],
                "numLayers": entry["numLayers"],
                "totalParams": entry["totalParams"],
                "totalFlops": entry["totalFlops"],
                "warnings": entry["warnings"],
                "source": entry["source"],
            }
        )
    names = {model["name"] for model in models}
    interesting = {
        name: {
            "present": name in names,
            "trace": any(model["name"] == name and model["mode"] == "trace" for model in models),
        }
        for name in INTERESTING.get(repo, ())
    }
    payload.update(
        {
            "elapsed": round(elapsed, 2),
            "models": len(models),
            "trace": trace_count,
            "static": static_count,
            "errors": len(index["errors"]),
            "invalid_graphs": invalid,
            "index_errors": validate_index(index)[:5],
            "model_list": models,
            "interesting": interesting,
            "scan_errors": index["errors"][:10],
        }
    )
    return payload


def print_summary(payload: dict) -> None:
    repo = payload["repo"]
    if payload.get("missing"):
        print(f"{repo:<16} MISSING ({payload['error']})")
        return
    if "error" in payload:
        print(f"{repo:<16} CRASHED: {payload['error']}")
        return
    print(
        f"{repo:<16} models={payload['models']:<3} trace={payload['trace']:<3} "
        f"static={payload['static']:<3} errors={payload['errors']:<2} "
        f"invalid={payload['invalid_graphs']} time={payload['elapsed']}s"
    )
    markers = payload.get("interesting") or {}
    if markers:
        rendered = ", ".join(
            f"{name}:{'trace' if info['trace'] else ('present' if info['present'] else 'missing')}"
            for name, info in markers.items()
        )
        print(f"{'':<16} keys: {rendered}")
    noisy = [model["name"] for model in payload.get("model_list", [])]
    if noisy:
        shown = ", ".join(noisy[:24])
        suffix = f" … (+{len(noisy) - 24} more)" if len(noisy) > 24 else ""
        print(f"{'':<16} names: {shown}{suffix}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark NetPrism on real-world model repos")
    parser.add_argument(
        "--root",
        default=os.environ.get("NETPRISM_EVAL_ROOT")
        or str(Path(os.environ.get("TEMP", "/tmp")) / "netprism_eval"),
        help="folder containing the cloned repositories",
    )
    parser.add_argument("--repos", nargs="*", default=list(DEFAULT_REPOS), help="repository folder names")
    parser.add_argument("--json", default=None, help="optional path to write the full JSON report")
    parser.add_argument("--max-files", type=int, default=400, help="max .py files per repository")
    parser.add_argument("--static", action="store_true", help="force static analysis (no execution)")
    args = parser.parse_args(argv)

    root = Path(args.root)
    if not root.is_dir():
        print(f"error: evaluation root not found: {root}", file=sys.stderr)
        return 2
    options = AnalyzeOptions(execute=not args.static, max_files=args.max_files)
    reports = []
    print(f"NetPrism real-world evaluation  root={root}  execute={not args.static}\n")
    for repo in args.repos:
        payload = evaluate_repo(root, repo, options)
        reports.append(payload)
        print_summary(payload)
        print()
    if args.json:
        output = Path(args.json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(reports, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"report written: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
