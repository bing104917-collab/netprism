#!/usr/bin/env python3
"""End-to-end smoke check for NetPrism.

Runs the real parser over a folder, validates every graph against the frozen
contract, starts the HTTP server and fetches the index + model JSON.

Usage::

    python scripts/verify.py [folder] [--static]
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import threading
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from netprism.core import AnalyzeOptions, analyze_folder  # noqa: E402
from netprism.schema import validate_graph, validate_index  # noqa: E402
from netprism.server import create_app, create_server  # noqa: E402


def get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as response:
        assert response.status == 200, response.status
        return json.loads(response.read().decode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", nargs="?", default=str(ROOT / "samples"))
    parser.add_argument("--static", action="store_true", help="do not execute model code")
    args = parser.parse_args()

    folder = Path(args.folder).resolve()
    print(f"analyzing {folder} ...")
    result = analyze_folder(folder, AnalyzeOptions(execute=not args.static, max_files=50))
    graph_errors: list[str] = []
    for model_id, graph in result.graphs.items():
        errors = validate_graph(graph)
        if errors:
            graph_errors.append(f"{model_id}: {errors[:3]}")
    index_errors = validate_index(result.index)
    if graph_errors or index_errors:
        print("SCHEMA FAILURES")
        for item in graph_errors:
            print(" -", item)
        for item in index_errors:
            print(" -", item)
        return 1

    print(f"models: {len(result.graphs)}  errors: {len(result.index.get('errors', []))}")
    for entry in result.index["models"]:
        graph = result.graphs[entry["id"]]
        modes = {node["kind"] for node in graph["nodes"]}
        print(
            f"  {entry['id']:<42} {entry['mode']:<6} layers={graph['model']['numLayers']:>3} "
            f"params={graph['model']['totalParams']:>10} flops={graph['model']['totalFlops']:>12} "
            f"kinds={len(modes)}"
        )

    with tempfile.TemporaryDirectory(prefix="netprism_verify_") as tmp:
        out = Path(tmp)
        for entry in result.index["models"]:
            (out / entry["file"]).write_text(
                json.dumps(result.graphs[entry["id"]], ensure_ascii=False, indent=2), encoding="utf-8"
            )
        (out / "index.json").write_text(json.dumps(result.index, ensure_ascii=False, indent=2), encoding="utf-8")

        app = create_app(out)
        httpd = create_server(app, "127.0.0.1", 0)
        port = httpd.server_address[1]
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            base = f"http://127.0.0.1:{port}"
            health = get_json(base + "/api/health")
            index = get_json(base + "/api/index")
            assert health["ok"] is True
            assert len(index["models"]) == len(result.index["models"])
            first = index["models"][0]["id"]
            graph = get_json(base + f"/api/model/{first}")
            assert graph["schemaVersion"] == "1.0"
            print(f"server OK on :{port}  first model: {first}")
        finally:
            httpd.shutdown()
            httpd.server_close()
            thread.join(timeout=5)

    if result.index.get("errors"):
        print("\nskipped files:")
        for item in result.index["errors"]:
            print(f"  - {item['file']}: {item['message']}")
    print("VERIFY OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
