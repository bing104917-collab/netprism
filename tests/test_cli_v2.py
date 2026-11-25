"""CLI v2 tests: report / compare / metric enrichment."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_cli(*args: str, cwd: Path = ROOT) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "netprism", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=180,
    )


def test_report_markdown_from_scan_dir():
    result = run_cli("report", "examples/demo", "--model", "cnn_vgg__vggmini", "--format", "md")
    assert result.returncode == 0, result.stderr
    assert "VGGMini" in result.stdout
    assert "Activation" in result.stdout
    assert "Receptive" in result.stdout
    assert "逐层" in result.stdout


def test_report_csv_and_html_to_files(tmp_path: Path):
    csv_path = tmp_path / "report.csv"
    result = run_cli("report", "examples/demo", "--model", "cnn_vgg__vggmini", "--format", "csv", "--out", str(csv_path))
    assert result.returncode == 0, result.stderr
    raw = csv_path.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")
    assert b"VGGMini" in raw

    html_path = tmp_path / "report.html"
    result = run_cli("report", "examples/demo", "--model", "cnn_vgg__vggmini", "--format", "html", "--out", str(html_path))
    assert result.returncode == 0, result.stderr
    html = html_path.read_text(encoding="utf-8")
    assert "<html" in html and "VGGMini" in html


def test_compare_json(tmp_path: Path):
    out = tmp_path / "compare.json"
    result = run_cli(
        "compare", "examples/demo", "examples/demo",
        "--model-a", "cnn_vgg__vggmini",
        "--model-b", "resnet_mini__resnetmini",
        "--format", "json", "--out", str(out),
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert "summary" in payload
    assert set(payload) & {"added", "removed", "changed"}


def test_explain_json_from_scan_dir():
    result = run_cli("explain", "examples/demo", "--model", "resnet_mini__resnetmini", "--json")
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["family"] == "resnet"
    assert payload["patterns"]
    assert payload["pseudocode"]
    assert payload.get("families")


def test_scan_enriches_metrics(tmp_path: Path):
    out = tmp_path / "scan"
    result = run_cli("scan", "samples", "--out", str(out), "--max-files", "20", "--quiet")
    assert result.returncode == 0, result.stderr
    index = json.loads((out / "index.json").read_text(encoding="utf-8"))
    graph = json.loads((out / index["models"][0]["file"]).read_text(encoding="utf-8"))
    metrics = graph["model"].get("metrics") or {}
    assert metrics.get("activationBytes", 0) > 0
    assert metrics.get("maxReceptiveField", 0) >= 1
    assert any((node.get("metrics") or {}).get("activationBytes", 0) > 0 for node in graph["nodes"])
