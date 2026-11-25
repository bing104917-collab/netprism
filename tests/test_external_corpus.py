"""Regression checks for the downloaded external model corpus.

The corpus is a set of real model source files fetched from public GitHub
repositories (see ``benchmarks/README.md``).  Static scans must never execute
the downloaded code and must never crash a whole repository.

The smaller trace test is limited to four trusted, dependency-light repos; it
checks that high-fidelity analysis still works on real CNN / UNet / GPT / ViT
code after the static-parser changes.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "benchmarks" / "external"

STATIC_REPOS = {
    "BlinkDL__RWKV-LM",
    "facebookresearch__detr",
    "facebookresearch__segment-anything",
    "karpathy__minGPT",
    "karpathy__nanoGPT",
    "KindXiaoming__pykan",
    "kuangliu__pytorch-cifar",
    "lucidrains__denoising-diffusion-pytorch",
    "lucidrains__vit-pytorch",
    "microsoft__Swin-Transformer",
    "milesial__Pytorch-UNet",
    "openai__CLIP",
    "state-spaces__mamba",
    "tkipf__pygcn",
    "ultralytics__yolov5",
    "yanx27__Pointnet_Pointnet2_pytorch",
    "yenchenlin__nerf-pytorch",
}


def _scan(repo: Path, *, execute: bool) -> dict:
    with tempfile.TemporaryDirectory(prefix="netprism_corpus_") as tmp:
        out = Path(tmp) / "out"
        command = [
            sys.executable,
            "-m",
            "netprism",
            "scan",
            str(repo),
            "--out",
            str(out),
            "--quiet",
        ]
        if not execute:
            command.append("--no-execute")
        proc = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=300)
        assert proc.returncode == 0, proc.stderr[-1000:]
        index = json.loads((out / "index.json").read_text(encoding="utf-8"))
        graphs = {}
        for entry in index.get("models", []):
            path = out / str(entry.get("file"))
            if path.is_file():
                graphs[entry.get("name")] = json.loads(path.read_text(encoding="utf-8"))
        return {"index": index, "graphs": graphs}


@pytest.fixture(scope="module")
def static_scan() -> dict:
    if not CORPUS.is_dir():
        pytest.skip(f"external corpus not present: {CORPUS}")
    return _scan(CORPUS, execute=False)


def test_static_corpus_scans_without_errors(static_scan: dict) -> None:
    index = static_scan["index"]
    assert index.get("errors") == []
    names = {entry.get("name") for entry in index.get("models", [])}
    assert len(names) >= 40, f"unexpectedly small corpus result set: {sorted(names)}"
    for expected in ("ResNet18", "UNet", "GPT", "ViT", "SwinTransformer", "Mamba", "GCN"):
        assert expected in names, f"missing expected model {expected}"


def test_static_corpus_graphs_validate(static_scan: dict) -> None:
    from netprism.schema import validate_graph

    for name, graph in static_scan["graphs"].items():
        errors = validate_graph(graph)
        assert errors == [], f"{name}: {errors[:5]}"


def test_static_corpus_has_no_identity_fallback(static_scan: dict) -> None:
    for name, graph in static_scan["graphs"].items():
        text = " ".join(graph.get("model", {}).get("warnings", []))
        assert "treated as identity" not in text, name


def test_trusted_trace_corpus() -> None:
    if not CORPUS.is_dir():
        pytest.skip(f"external corpus not present: {CORPUS}")
    expectations = {
        "kuangliu__pytorch-cifar": (10, 53_331_372),
        "milesial__Pytorch-UNet": (1, 31_038_218),
        "karpathy__nanoGPT": (1, 124_475_904),
    }
    for repo, (count, params) in expectations.items():
        result = _scan(CORPUS / repo, execute=True)
        models = result["index"].get("models", [])
        assert len(models) == count, (repo, [(m.get("name"), m.get("mode")) for m in models])
        assert all(m.get("mode") == "trace" for m in models), repo
        assert sum(int(m.get("totalParams") or 0) for m in models) == params, repo

    vit = _scan(CORPUS / "lucidrains__vit-pytorch", execute=True)
    models = vit["index"].get("models", [])
    assert len(models) == 5
    assert all(m.get("mode") == "trace" for m in models)
    assert {"ViT", "SimpleViT", "CaiT", "CrossFormer", "NaViT"} <= {m.get("name") for m in models}
