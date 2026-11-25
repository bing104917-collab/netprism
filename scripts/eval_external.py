"""Evaluate NetPrism on externally downloaded model repositories.

The corpus lives in ``benchmarks/external/<repo>``.  Analysis is always
static (``--no-execute``): downloaded code is never run.  Each result records
model/error/warning counts and the first warning messages so regressions are
easy to spot.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "benchmarks" / "external"


def run(repo: Path) -> dict:
    with tempfile.TemporaryDirectory(prefix="netprism_ext_") as tmp:
        out = Path(tmp) / "out"
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "netprism",
                "scan",
                str(repo),
                "--out",
                str(out),
                "--no-execute",
                "--quiet",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=300,
        )
        result = {
            "repo": repo.name,
            "returncode": proc.returncode,
            "models": 0,
            "trace": 0,
            "static": 0,
            "errors": 0,
            "warnings": 0,
            "layers": 0,
            "params": 0,
            "first_warnings": [],
        }
        index_path = out / "index.json"
        if index_path.is_file():
            index = json.loads(index_path.read_text(encoding="utf-8"))
            result["models"] = len(index.get("models", []))
            result["errors"] = len(index.get("errors", []))
            result["warnings"] = sum(int(m.get("warnings") or 0) for m in index.get("models", []))
            result["trace"] = sum(1 for m in index.get("models", []) if m.get("mode") == "trace")
            result["static"] = sum(1 for m in index.get("models", []) if m.get("mode") == "static")
            result["layers"] = sum(int(m.get("numLayers") or 0) for m in index.get("models", []))
            result["params"] = sum(int(m.get("totalParams") or 0) for m in index.get("models", []))
            for entry in index.get("models", []):
                graph_path = out / str(entry.get("file"))
                if not graph_path.is_file():
                    continue
                graph = json.loads(graph_path.read_text(encoding="utf-8"))
                for warning in graph.get("model", {}).get("warnings", [])[:2]:
                    if len(result["first_warnings"]) < 6:
                        result["first_warnings"].append(f"{entry.get('name')}: {warning}")
            for error in index.get("errors", [])[:2]:
                if len(result["first_warnings"]) < 6:
                    result["first_warnings"].append("ERROR: " + str(error.get("message", ""))[:160])
        else:
            result["first_warnings"].append((proc.stderr or proc.stdout).strip()[-300:])
        return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", default=None, help="write full results to this path")
    args = parser.parse_args()
    if not CORPUS.is_dir():
        print(f"missing corpus: {CORPUS}", file=sys.stderr)
        return 2
    results = []
    for repo in sorted(p for p in CORPUS.iterdir() if p.is_dir()):
        result = run(repo)
        results.append(result)
        print(
            f"{result['repo']:<42} models={result['models']:<3} "
            f"trace={result['trace']:<3} static={result['static']:<3} "
            f"warn={result['warnings']:<3} errors={result['errors']:<3} "
            f"layers={result['layers']:<5} params={result['params']:,}"
        )
        for warning in result["first_warnings"]:
            print(f"    - {warning}")
    if args.json:
        Path(args.json).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if all(r["returncode"] == 0 for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
