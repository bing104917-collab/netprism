"""Independent QA for the external model corpus (task-16).

Everything is verified end-to-end through the real CLI:

* per-repository ``scan --no-execute`` of all 17 downloaded repos;
* spot checks on DETR / SAM / YOLOv5 / PointNet2 / NeRF static graphs;
* default (trace) scans of the four trusted dependency-light repositories;
* ``scripts/eval_external.py`` determinism and ``benchmarks/EXTERNAL.md``
  consistency with the checked-in JSON.

The corpus lives in ``benchmarks/external`` and is treated as read-only.  Tests
skip with an explicit message when the corpus is absent.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from netprism.schema import validate_graph

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "benchmarks" / "external"
EXTERNAL_MD = ROOT / "benchmarks" / "EXTERNAL.md"
RESULTS_JSON = ROOT / "benchmarks" / "external_results.json"
EVAL_SCRIPT = ROOT / "scripts" / "eval_external.py"
PY = sys.executable

#: The 17 repositories promised by the report.
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

#: Task-16 trusted trace expectations: (models, params, layers).
TRACE_EXPECTATIONS = {
    "kuangliu__pytorch-cifar": (10, 53_331_372, 1_551),
    "milesial__Pytorch-UNet": (1, 31_038_218, 67),
    "karpathy__nanoGPT": (1, 124_475_904, 113),
    "lucidrains__vit-pytorch": (5, 29_199_538, 727),
}

#: Representative static graphs to spot check (repo -> documented model count/names).
SPOT_CHECKS = {
    "facebookresearch__detr": {"models": 3, "names": {"Transformer", "SetCriterion", "PostProcess"}},
    "facebookresearch__segment-anything": {
        "models": 4,
        "names": {"Sam", "ImageEncoderViT", "PromptEncoder", "TwoWayTransformer"},
    },
    "ultralytics__yolov5": {"models": 8, "names": {"BottleneckCSP", "Focus", "Classify"}},
    "yanx27__Pointnet_Pointnet2_pytorch": {
        "models": 3,
        "names": {"get_model", "PointNetSetAbstractionMsg", "PointNetFeaturePropagation"},
    },
    "yenchenlin__nerf-pytorch": {"models": 1, "names": {"NeRF"}},
}

TRACE_MODEL_NAMES = {
    "kuangliu__pytorch-cifar": {"ResNet18", "LeNet", "GoogLeNet", "DenseNet121"},
    "milesial__Pytorch-UNet": {"UNet"},
    "karpathy__nanoGPT": {"GPT"},
    "lucidrains__vit-pytorch": {"ViT", "SimpleViT", "CaiT", "CrossFormer", "NaViT"},
}

TRACE_TABLE_REPOS = {
    "kuangliu/pytorch-cifar": "kuangliu__pytorch-cifar",
    "milesial/Pytorch-UNet": "milesial__Pytorch-UNet",
    "karpathy/nanoGPT": "karpathy__nanoGPT",
    "lucidrains/vit-pytorch": "lucidrains__vit-pytorch",
}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def run_scan(repo: Path, *, execute: bool, timeout: int = 900) -> dict:
    """Scan one repository and return ``{"index": ..., "graphs": [...]}``."""
    with tempfile.TemporaryDirectory(prefix="netprism_external_qa_") as tmp:
        out = Path(tmp) / "out"
        command = [PY, "-m", "netprism", "scan", str(repo), "--out", str(out), "--quiet"]
        if not execute:
            command.append("--no-execute")
        proc = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        assert proc.returncode == 0, f"{repo.name}: scan failed\n{proc.stderr[-1500:]}"
        index = json.loads((out / "index.json").read_text(encoding="utf-8"))
        graphs = []
        for entry in index.get("models", []):
            path = out / str(entry.get("file"))
            assert path.is_file(), f"{repo.name}: missing graph file {entry.get('file')}"
            graphs.append((entry, json.loads(path.read_text(encoding="utf-8"))))
        return {"index": index, "graphs": graphs}


def repo_warnings(index: dict) -> int:
    return sum(int(entry.get("warnings") or 0) for entry in index["models"])


def repo_layers(index: dict) -> int:
    return sum(int(entry.get("numLayers") or 0) for entry in index["models"])


def repo_params(index: dict) -> int:
    return sum(int(entry.get("totalParams") or 0) for entry in index["models"])


def assert_no_identity_fallback(graph: dict, label: str) -> None:
    text = " ".join(graph.get("model", {}).get("warnings") or [])
    assert "treated as identity" not in text, f"{label}: identity fallback warning present"
    for node in graph.get("nodes", []):
        notes = " ".join(node.get("notes") or [])
        assert "treated as identity" not in notes, f"{label}/{node.get('id')}: identity fallback note"


def assert_repo_matches_report(index: dict, documented: dict, repo: str) -> None:
    """Report rows are the floor; newly discovered models are allowed (task-16)."""
    live_models = len(index["models"])
    assert live_models >= documented["models"], f"{repo}: lost models vs report"
    if live_models == documented["models"]:
        assert repo_warnings(index) == documented["warnings"], f"{repo}: warning count drifted"
        assert repo_layers(index) == documented["layers"], f"{repo}: layer count drifted"
        assert repo_params(index) == documented["params"], f"{repo}: params drifted"
    else:
        # Extra models may only add to the documented totals.
        assert repo_warnings(index) >= documented["warnings"], f"{repo}: warning count drifted"
        assert repo_layers(index) >= documented["layers"], f"{repo}: layer count drifted"
        assert repo_params(index) >= documented["params"], f"{repo}: params drifted"


def _int_cell(value: str) -> int:
    return int(value.replace(",", "").strip())


def parse_external_md() -> dict:
    """Parse the two result tables and the totals line from ``EXTERNAL.md``."""
    assert EXTERNAL_MD.is_file(), f"missing report: {EXTERNAL_MD}"
    lines = EXTERNAL_MD.read_text(encoding="utf-8").splitlines()
    static: dict[str, dict] = {}
    trace: dict[str, dict] = {}
    section: str | None = None
    for line in lines:
        if line.startswith("## High-fidelity trace"):
            section = "trace"
            continue
        if line.startswith("## Static scan"):
            section = "static"
            continue
        if line.startswith("## "):
            section = None
            continue
        if not section or not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if not cells or cells[0] == "Repo" or set(cells[0]) <= {"-", ":"}:
            continue
        if section == "trace":
            repo, models, trace_count, layers, params = cells
            trace[TRACE_TABLE_REPOS.get(repo, repo)] = {
                "models": _int_cell(models),
                "trace": _int_cell(trace_count),
                "layers": _int_cell(layers),
                "params": _int_cell(params),
            }
        else:
            repo, models, warnings, layers, params = cells
            static[repo] = {
                "models": _int_cell(models),
                "warnings": _int_cell(warnings),
                "layers": _int_cell(layers),
                "params": _int_cell(params),
            }
    match = re.search(
        r"Totals:\s*\*\*(\d+) models,\s*(\d+) file-level errors,\s*(\d+) static warnings\*\*",
        "\n".join(lines),
    )
    assert match, "EXTERNAL.md totals line is missing"
    totals = {
        "models": int(match.group(1)),
        "errors": int(match.group(2)),
        "warnings": int(match.group(3)),
    }
    return {"trace": trace, "static": static, "totals": totals}


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def report() -> dict:
    if not EXTERNAL_MD.is_file():
        pytest.skip(f"external report not present: {EXTERNAL_MD}")
    parsed = parse_external_md()
    assert set(parsed["static"]) == STATIC_REPOS, (
        f"EXTERNAL.md static table repos differ: {sorted(set(parsed['static']) ^ STATIC_REPOS)}"
    )
    assert set(parsed["trace"]) == set(TRACE_EXPECTATIONS), (
        f"EXTERNAL.md trace table repos differ: {sorted(set(parsed['trace']) ^ set(TRACE_EXPECTATIONS))}"
    )
    return parsed


@pytest.fixture(scope="module")
def committed_results() -> list[dict]:
    if not RESULTS_JSON.is_file():
        pytest.skip(f"external results JSON not present: {RESULTS_JSON}")
    data = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
    assert isinstance(data, list)
    return data


@pytest.fixture(scope="module")
def static_scans() -> dict[str, dict]:
    if not CORPUS.is_dir():
        pytest.skip(f"external corpus not present: {CORPUS}")
    return {repo: run_scan(CORPUS / repo, execute=False) for repo in sorted(STATIC_REPOS)}


@pytest.fixture(scope="module")
def trace_scans() -> dict[str, dict]:
    if not CORPUS.is_dir():
        pytest.skip(f"external corpus not present: {CORPUS}")
    return {repo: run_scan(CORPUS / repo, execute=True) for repo in sorted(TRACE_EXPECTATIONS)}


@pytest.fixture(scope="module")
def eval_runs(tmp_path_factory: pytest.TempPathFactory) -> list[list[dict]]:
    """Run scripts/eval_external.py twice, capturing both JSON result documents."""
    if not EVAL_SCRIPT.is_file():
        pytest.skip(f"missing {EVAL_SCRIPT}")
    if not CORPUS.is_dir():
        pytest.skip(f"external corpus not present: {CORPUS}")
    runs: list[list[dict]] = []
    for index in range(2):
        target = tmp_path_factory.mktemp(f"eval_external_{index}") / "results.json"
        proc = subprocess.run(
            [PY, str(EVAL_SCRIPT), "--json", str(target)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=1200,
        )
        assert proc.returncode == 0, f"eval_external.py failed:\n{proc.stderr[-2000:]}"
        assert target.is_file()
        runs.append(json.loads(target.read_text(encoding="utf-8")))
    return runs


# --------------------------------------------------------------------------- #
# 1. per-repository static scan of all 17 repos
# --------------------------------------------------------------------------- #
def test_static_scan_all_17_repos_matches_report(static_scans: dict, report: dict) -> None:
    assert set(static_scans) == STATIC_REPOS
    totals = {"models": 0, "warnings": 0, "layers": 0, "params": 0}
    for repo in sorted(STATIC_REPOS):
        scan = static_scans[repo]
        index = scan["index"]
        documented = report["static"][repo]
        assert index["errors"] == [], f"{repo}: scan file errors {index['errors']}"
        assert all(entry["mode"] == "static" for entry in index["models"]), f"{repo}: non-static model"
        assert_repo_matches_report(index, documented, repo)
        assert index["models"], f"{repo}: no models discovered"
        totals["models"] += len(index["models"])
        totals["warnings"] += repo_warnings(index)
        totals["layers"] += repo_layers(index)
        totals["params"] += repo_params(index)
    assert totals["models"] >= report["totals"]["models"] == 47
    assert totals["warnings"] >= report["totals"]["warnings"]
    assert totals["layers"] >= sum(row["layers"] for row in report["static"].values())
    assert totals["params"] >= sum(row["params"] for row in report["static"].values())
    if totals["models"] == 47:
        assert totals["warnings"] == report["totals"]["warnings"] == 191
        assert totals["layers"] == sum(row["layers"] for row in report["static"].values())
        assert totals["params"] == sum(row["params"] for row in report["static"].values())


def test_static_scan_graphs_are_schema_valid_and_identity_free(static_scans: dict) -> None:
    checked = 0
    for repo, scan in static_scans.items():
        index = scan["index"]
        for entry, graph in scan["graphs"]:
            errors = validate_graph(graph)
            assert errors == [], f"{repo}/{entry['id']}: schema errors {errors[:5]}"
            model = graph["model"]
            assert model["mode"] == "static"
            assert model["name"]
            assert graph["nodes"], f"{repo}/{entry['id']}: empty nodes"
            for node in graph["nodes"]:
                assert isinstance(node["inShapes"], list)
                assert isinstance(node["outShapes"], list)
                for shape in node["inShapes"] + node["outShapes"]:
                    assert isinstance(shape, list) and shape
                    assert all(isinstance(dim, int) and dim >= -1 for dim in shape)
            assert_no_identity_fallback(graph, f"{repo}/{entry['id']}")
            lowered = [warning.lower() for warning in model["warnings"]]
            assert not any("could not import module" in warning for warning in lowered), repo
            checked += 1
    assert checked >= 47, f"expected at least 47 static graphs, validated {checked}"


def test_spot_check_representative_static_graphs(static_scans: dict, report: dict) -> None:
    for repo, expected in SPOT_CHECKS.items():
        scan = static_scans[repo]
        index = scan["index"]
        documented = report["static"][repo]
        assert index["errors"] == [], f"{repo}: errors {index['errors']}"
        assert_repo_matches_report(index, documented, repo)
        assert len(index["models"]) >= expected["models"]
        names = {entry["name"] for entry in index["models"]}
        assert expected["names"] <= names, f"{repo}: missing models {expected['names'] - names}"

        node_total = 0
        for entry, graph in scan["graphs"]:
            assert validate_graph(graph) == [], f"{repo}/{entry['name']}: invalid graph"
            assert graph["nodes"], f"{repo}/{entry['name']}: no nodes"
            node_total += len(graph["nodes"])
            graph_negative_shapes = 0
            for node in graph["nodes"]:
                assert node["outShapes"], f"{repo}/{entry['name']}/{node['id']}: no output shapes"
                for shape in node["inShapes"] + node["outShapes"]:
                    if -1 in shape:
                        graph_negative_shapes += 1
            if graph_negative_shapes and not graph["model"]["warnings"]:
                # Unknown (-1) dimensions must be explained by a warning, never a silent crash.
                assert False, f"{repo}/{entry['name']}: -1 shapes without any warning"
            assert_no_identity_fallback(graph, f"{repo}/{entry['name']}")

        assert node_total >= documented["layers"], (
            f"{repo}: graph nodes ({node_total}) below documented layers ({documented['layers']})"
        )


# --------------------------------------------------------------------------- #
# 2. trusted trace repositories (default CLI scan)
# --------------------------------------------------------------------------- #
def test_trusted_trace_repositories(trace_scans: dict, report: dict) -> None:
    trace_totals = {"models": 0, "params": 0, "layers": 0}
    for repo, (expected_models, expected_params, expected_layers) in TRACE_EXPECTATIONS.items():
        scan = trace_scans[repo]
        index = scan["index"]
        documented = report["trace"][repo]
        assert index["errors"] == [], f"{repo}: trace errors {index['errors']}"
        assert len(index["models"]) == expected_models, repo
        assert all(entry["mode"] == "trace" for entry in index["models"]), f"{repo}: non-trace model"
        assert repo_params(index) == expected_params, f"{repo}: trace params drifted"
        assert repo_layers(index) == expected_layers, f"{repo}: trace layers drifted"
        assert documented == {
            "models": expected_models,
            "trace": expected_models,
            "layers": expected_layers,
            "params": expected_params,
        }, f"{repo}: EXTERNAL.md trace row drifted"
        names = {entry["name"] for entry in index["models"]}
        assert TRACE_MODEL_NAMES[repo] <= names, f"{repo}: missing models {TRACE_MODEL_NAMES[repo] - names}"
        for entry, graph in scan["graphs"]:
            assert validate_graph(graph) == [], f"{repo}/{entry['name']}: invalid trace graph"
            assert graph["nodes"], f"{repo}/{entry['name']}: empty trace graph"
            assert_no_identity_fallback(graph, f"{repo}/{entry['name']}")
        trace_totals["models"] += len(index["models"])
        trace_totals["params"] += repo_params(index)
        trace_totals["layers"] += repo_layers(index)
    assert trace_totals["models"] == 17
    assert trace_totals["params"] == sum(value[1] for value in TRACE_EXPECTATIONS.values())
    assert trace_totals["layers"] == sum(value[2] for value in TRACE_EXPECTATIONS.values())


# --------------------------------------------------------------------------- #
# 3. eval_external.py repeatability + report/JSON consistency
# --------------------------------------------------------------------------- #
def test_external_markdown_matches_results_json(report: dict, committed_results: list[dict]) -> None:
    by_repo = {entry["repo"]: entry for entry in committed_results}
    assert set(by_repo) == STATIC_REPOS, f"JSON repo set differs: {sorted(set(by_repo) ^ STATIC_REPOS)}"
    for repo, row in report["static"].items():
        entry = by_repo[repo]
        assert entry["returncode"] == 0, repo
        assert entry["errors"] == 0, repo
        assert entry["models"] == row["models"], repo
        assert entry["warnings"] == row["warnings"], repo
        assert entry["layers"] == row["layers"], repo
        assert entry["params"] == row["params"], repo
    totals = report["totals"]
    assert totals["models"] == sum(row["models"] for row in report["static"].values()) == 47
    assert totals["warnings"] == sum(row["warnings"] for row in report["static"].values()) == 191
    assert totals["errors"] == sum(entry["errors"] for entry in committed_results) == 0
    assert all(entry["returncode"] == 0 for entry in committed_results)


def test_eval_external_script_is_repeatable(eval_runs: list[list[dict]], report: dict) -> None:
    first, second = eval_runs
    assert first == second, "scripts/eval_external.py output is not deterministic between runs"
    assert len(first) == len(STATIC_REPOS) == 17
    for entry in first:
        assert entry["returncode"] == 0
        assert entry["errors"] == 0
        assert entry["models"] > 0
        assert entry["trace"] == 0 and entry["static"] == entry["models"]
        documented = report["static"][entry["repo"]]
        assert entry["models"] == documented["models"]
        assert entry["warnings"] == documented["warnings"]
        assert entry["layers"] == documented["layers"]
        assert entry["params"] == documented["params"]


def test_eval_script_matches_checked_in_results_and_cli(
    eval_runs: list[list[dict]], committed_results: list[dict], static_scans: dict
) -> None:
    live = eval_runs[0]
    assert live == committed_results, (
        "benchmarks/external_results.json is stale: regenerate it with "
        "`python scripts/eval_external.py --json benchmarks/external_results.json`"
    )
    live_by_repo = {entry["repo"]: entry for entry in live}
    for repo, scan in static_scans.items():
        index = scan["index"]
        entry = live_by_repo[repo]
        assert (entry["models"], entry["warnings"], entry["layers"], entry["params"]) == (
            len(index["models"]),
            repo_warnings(index),
            repo_layers(index),
            repo_params(index),
        ), f"{repo}: eval script and direct CLI scan disagree"


# --------------------------------------------------------------------------- #
# 4. whole-corpus scan safety (canonical numbers are per-repository)
# --------------------------------------------------------------------------- #
def test_whole_corpus_single_scan_is_safe(tmp_path: Path) -> None:
    """One scan over the parent folder must still be crash-free and schema-valid.

    Per-repository scans are the canonical numbers for EXTERNAL.md: scanning the
    parent folder changes folder-level component inference across unrelated
    projects, so model/warning totals are not compared to the report here.
    """
    if not CORPUS.is_dir():
        pytest.skip(f"external corpus not present: {CORPUS}")
    out = tmp_path / "whole_corpus"
    proc = subprocess.run(
        [PY, "-m", "netprism", "scan", str(CORPUS), "--out", str(out), "--quiet", "--no-execute"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=900,
    )
    assert proc.returncode == 0, proc.stderr[-1500:]
    index = json.loads((out / "index.json").read_text(encoding="utf-8"))
    assert index["errors"] == []
    assert len(index["models"]) >= 40, "whole-corpus scan collapsed unexpectedly"
    expected_names = {
        "ResNet18",
        "UNet",
        "GPT",
        "ViT",
        "SwinTransformer",
        "Mamba",
        "GCN",
        "Sam",
        "Transformer",
        "PointNetSetAbstractionMsg",
        "NeRF",
    }
    names = {entry["name"] for entry in index["models"]}
    assert expected_names <= names, f"whole-corpus scan missing {sorted(expected_names - names)}"
    for entry in index["models"]:
        graph = json.loads((out / entry["file"]).read_text(encoding="utf-8"))
        assert validate_graph(graph) == [], f"{entry['id']}: invalid graph"
        assert_no_identity_fallback(graph, entry["id"])
