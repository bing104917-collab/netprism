"""Independent QA for the v2.1 algorithm-understanding layer (SPEC-ALGO).

The CLI (``explain`` / ``report`` / ``scan``) is tested as a black box through
``python -m netprism``; expected values are derived from the knowledge base and
the graph documents (never from the classifier's own output).  The UI part of
the contract lives in ``tests/test_v2_ui.py`` (appended algorithm-panel cases).

Coverage map for task-13:

1. ``explain`` on the five required samples: family, top-3 candidates,
   confidence, patterns (formula / evidence / node ids) and pseudocode shapes.
2. ``report`` md/html contains the algorithm section; a graph without meta does
   not get a fabricated section from the report renderer.
3. ``scan`` writes ``model.meta`` / ``node.meta`` / ``group.meta``.
4. Knowledge base: 102 families / 56 components, validation, key families.
5. Unknown / low-confidence graphs do not crash and explain themselves.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from netprism import algorithm_kb
from netprism.report import render_report
from netprism.schema import KINDS, validate_graph

ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = ROOT / "examples" / "demo"
SAMPLES_DIR = ROOT / "samples"
FORMAT_FIXTURES = ROOT / "tests" / "fixtures" / "formats"
PY = sys.executable

#: Task-13 requirement: the five samples that must be explained.
REQUIRED_SAMPLES = {
    "cnn_vgg__vggmini": {
        "family": "vgg",
        "patterns": {"conv_bn_act", "global_pool", "classifier_head"},
    },
    "resnet_mini__resnetmini": {
        "family": "resnet",
        "patterns": {"residual_block", "conv_bn_act"},
    },
    "unet_mini__unetmini": {
        "family": "unet",
        "patterns": {"unet_skip", "conv_bn_act"},
    },
    "custom_attention__tinygpt": {
        "family": "transformer_decoder",
        "patterns": {"transformer_block", "self_attention", "ffn"},
    },
    "transformer_classifier__tinytransformerclassifier": {
        "family": "transformer_encoder",
        "patterns": {"transformer_block", "self_attention", "ffn"},
    },
}

#: Key knowledge-base families from task-13, with a formula token that proves the
#: entry really describes that architecture (not a placeholder).
KEY_FAMILIES = {
    "resnet": {"domain": "vision", "alias": "resnet", "formula_token": "F(x) + x"},
    "swin": {"domain": "vision", "alias": "swin", "formula_token": "MxM"},
    "mamba": {"domain": "sequence", "alias": "mamba", "formula_token": "Delta"},
    "llama": {"domain": "sequence", "alias": "llama", "formula_token": "RMSNorm"},
    "vit": {"domain": "vision", "alias": "vit", "formula_token": "PatchEmbed"},
    "unet": {"domain": "vision", "alias": "unet", "formula_token": "Up(x)"},
    "pointnet": {"domain": "3d", "alias": "pointnet", "formula_token": "max_i"},
    "nerf": {"domain": "3d", "alias": "nerf", "formula_token": "gamma"},
    "kan": {"domain": "other", "alias": "kan", "formula_token": "phi"},
}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def run_cli(*args: str, cwd: Path = ROOT, timeout: int = 240) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [PY, "-m", "netprism", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def demo_graph(model_id: str) -> dict:
    index = load_json(DEMO_DIR / "index.json")
    entry = next(entry for entry in index["models"] if entry["id"] == model_id)
    return load_json(DEMO_DIR / entry["file"])


def strip_algorithm_meta(graph: dict) -> dict:
    graph = json.loads(json.dumps(graph))
    graph["model"].pop("meta", None)
    for node in graph["nodes"]:
        node.pop("meta", None)
    for group in graph["groups"]:
        group.pop("meta", None)
    return graph


def make_mystery_graph() -> dict:
    """A schema-valid graph using only unknown custom ops: no family evidence."""

    def node(index: int, name: str, op: str, kind: str) -> dict:
        return {
            "id": f"n{index}",
            "name": name,
            "op": op,
            "kind": kind,
            "modulePath": "",
            "depth": 0,
            "inShapes": [[1, 8]],
            "outShapes": [[1, 8]],
            "params": {},
            "numParams": 0,
            "flops": 0,
            "tags": [],
            "groupId": None,
            "source": None,
            "notes": [],
        }

    nodes = [
        node(0, "x", "Input", "input"),
        node(1, "mystery", "MysteryOp", "custom"),
        node(2, "weird", "WeirdOp", "custom"),
        node(3, "y", "Output", "output"),
    ]
    edges = [
        {"id": f"e{i}", "source": f"n{i}", "target": f"n{i + 1}", "kind": "data", "shapes": [], "label": ""}
        for i in range(3)
    ]
    return {
        "schemaVersion": "1.0",
        "generator": {"name": "netprism", "version": "0.1.0"},
        "model": {
            "name": "MysteryNet",
            "framework": "unknown",
            "className": None,
            "mode": "static",
            "source": {"file": "mystery.py", "relPath": "mystery.py"},
            "inputSpec": [{"name": "x", "shape": [1, 8], "dtype": "float32"}],
            "outputSpec": [{"name": "y", "shape": [1, 8], "dtype": "float32"}],
            "totalParams": 0,
            "trainableParams": 0,
            "totalFlops": 0,
            "flopsApprox": False,
            "numLayers": 2,
            "warnings": [],
        },
        "nodes": nodes,
        "edges": edges,
        "groups": [],
    }


def explain_json(*args: str) -> dict:
    proc = run_cli("explain", *args, "--json")
    assert proc.returncode == 0, proc.stderr
    assert "\ufffd" not in proc.stdout, "explain JSON is not valid UTF-8"
    return json.loads(proc.stdout)


# --------------------------------------------------------------------------- #
# 1. CLI `explain`: five required samples
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("model_id", sorted(REQUIRED_SAMPLES))
def test_explain_cli_family_patterns_and_pseudocode(model_id: str) -> None:
    expected = REQUIRED_SAMPLES[model_id]
    graph = demo_graph(model_id)
    graph_node_ids = {node["id"] for node in graph["nodes"]}
    payload = explain_json("examples/demo", "--model", model_id)

    # family / confidence contract
    assert payload["family"] == expected["family"], f"{model_id}: wrong family {payload['family']!r}"
    assert isinstance(payload["familyLabel"], str) and payload["familyLabel"].strip()
    confidence = payload["confidence"]
    assert isinstance(confidence, (int, float)) and not isinstance(confidence, bool)
    assert 0.0 < float(confidence) <= 1.0
    assert payload["familyConfidence"] == pytest.approx(round(float(confidence), 2))

    families = payload["families"]
    assert 1 <= len(families) <= 3, f"{model_id}: top families must be 1-3"
    assert families[0]["id"] == payload["family"]
    family_confidences = [float(entry["confidence"]) for entry in families]
    assert family_confidences == sorted(family_confidences, reverse=True)
    for entry in families:
        assert entry["id"] and entry["label"]
        assert 0.0 <= float(entry["confidence"]) <= 1.0
        assert entry["evidence"], f"{model_id}/{entry['id']}: family evidence missing"

    # patterns: kind / formula / confidence / node ids / evidence
    patterns = payload["patterns"]
    assert patterns, f"{model_id}: no patterns identified"
    pattern_kinds = {pattern["kind"] for pattern in patterns}
    missing = expected["patterns"] - pattern_kinds
    assert not missing, f"{model_id}: missing expected pattern kinds {sorted(missing)}"
    for pattern in patterns:
        assert pattern["kind"] and pattern["label"]
        assert str(pattern["formula"]).strip(), f"{model_id}/{pattern['kind']}: empty formula"
        assert 0.0 <= float(pattern["confidence"]) <= 1.0
        assert pattern["nodeIds"], f"{model_id}/{pattern['kind']}: no node ids"
        assert set(pattern["nodeIds"]) <= graph_node_ids, f"{model_id}/{pattern['kind']}: unknown node ids"
        assert pattern["evidence"], f"{model_id}/{pattern['kind']}: no evidence"
    assert payload["tags"], f"{model_id}: algorithm tags missing"

    # pseudocode: ordered, non-empty, carries shape annotations
    pseudocode = payload["pseudocode"]
    assert len(pseudocode) >= 3, f"{model_id}: too little pseudocode"
    assert all(isinstance(line, str) and line.strip() for line in pseudocode)
    assert "Input(" in pseudocode[0]
    shape_lines = [line for line in pseudocode if "# -> (" in line]
    assert len(shape_lines) >= 3, f"{model_id}: pseudocode lacks shape annotations"
    for line in shape_lines:
        assert re.search(r"# -> \(-?\d", line), f"{model_id}: malformed shape comment {line!r}"

    # groups are exposed for report/UI consumers
    groups = payload["groups"]
    assert len(groups) == len(graph["groups"])
    assert {group["id"] for group in groups} == {group["id"] for group in graph["groups"]}
    for group in groups:
        assert isinstance(group["nodeCount"], int) and group["nodeCount"] >= 0
        assert "pattern" in group


def test_explain_cli_default_text_output() -> None:
    proc = run_cli("explain", "examples/demo", "--model", "cnn_vgg__vggmini")
    assert proc.returncode == 0, proc.stderr
    assert "\ufffd" not in proc.stdout
    text = proc.stdout
    assert "### 算法理解" in text
    assert "VGG" in text
    assert "Conv-BN-Act" in text
    assert "y = Act(BN(Conv(x)))" in text
    assert "**Forward 伪代码**" in text
    assert "x = Input(1, 3, 64, 64)" in text
    assert text.count("# -> (") >= 3


def test_explain_cli_on_graph_json_and_format_file(tmp_path: Path) -> None:
    # A meta-less graph JSON: explain must compute the analysis on the fly.
    graph = strip_algorithm_meta(demo_graph("cnn_vgg__vggmini"))
    path = tmp_path / "old_vgg.json"
    path.write_text(json.dumps(graph, ensure_ascii=False), encoding="utf-8")
    payload = explain_json(str(path))
    assert payload["family"] == "vgg"
    assert payload["patterns"] and payload["pseudocode"]
    assert payload["unknown_reason"] is None

    # Model files are accepted sources too (SPEC-ALGO section 6).
    fixture = FORMAT_FIXTURES / "mlp.onnx"
    if fixture.is_file():
        payload = explain_json(str(fixture))
        assert payload["family"]
        assert 1 <= len(payload["families"]) <= 3
        assert payload["pseudocode"]
        assert 0.0 <= float(payload["confidence"]) <= 1.0


# --------------------------------------------------------------------------- #
# 2. report md/html algorithm section (+ no-meta behaviour)
# --------------------------------------------------------------------------- #
def test_report_md_and_html_include_algorithm_section(tmp_path: Path) -> None:
    md_path = tmp_path / "report.md"
    proc = run_cli(
        "report", "examples/demo", "--model", "cnn_vgg__vggmini", "--format", "md", "--out", str(md_path)
    )
    assert proc.returncode == 0, proc.stderr
    md = md_path.read_text(encoding="utf-8")
    assert "\ufffd" not in md
    assert "## 算法理解 (Algorithm)" in md
    assert "**家族 (Family)** VGG" in md
    assert "Conv-BN-Act" in md
    assert "y = Act(BN(Conv(x)))" in md
    assert "**Forward 伪代码**" in md
    assert "x = Input(1, 3, 64, 64)" in md
    assert md.count("# -> (") >= 3

    html_path = tmp_path / "report.html"
    proc = run_cli(
        "report", "examples/demo", "--model", "cnn_vgg__vggmini", "--format", "html", "--out", str(html_path)
    )
    assert proc.returncode == 0, proc.stderr
    html = html_path.read_text(encoding="utf-8")
    assert "\ufffd" not in html
    assert "算法理解 (Algorithm)" in html
    assert 'id="algo-view"' in html
    assert 'id="algo-family"' in html and "VGG" in html
    assert 'id="algo-patterns"' in html
    assert "y = Act(BN(Conv(x)))" in html
    assert 'id="algo-pseudocode"' in html
    assert "x = Input(1, 3, 64, 64)" in html
    assert "<script" not in html


def test_report_renderer_does_not_fabricate_section_without_meta() -> None:
    """``render_report`` is read-only: a graph without algo meta gets no section.

    (The CLI ``report`` command intentionally calls ``enrich_algorithms`` first;
    see the companion test below.)
    """
    graph = strip_algorithm_meta(demo_graph("cnn_vgg__vggmini"))
    md = render_report(json.loads(json.dumps(graph)), "md")
    html = render_report(json.loads(json.dumps(graph)), "html")
    assert "算法理解" not in md
    assert 'id="algo-view"' not in html
    assert 'id="algo-family"' not in html
    assert "y = Act(BN(Conv(x)))" not in md


def test_cli_report_recomputes_algorithm_section_for_meta_less_graph(tmp_path: Path) -> None:
    """CLI ``report`` enriches algorithms by design, so old graphs gain the section."""
    graph = strip_algorithm_meta(demo_graph("cnn_vgg__vggmini"))
    path = tmp_path / "old_vgg.json"
    path.write_text(json.dumps(graph, ensure_ascii=False), encoding="utf-8")
    for fmt in ("md", "html"):
        proc = run_cli("report", str(path), "--format", fmt)
        assert proc.returncode == 0, proc.stderr
        assert "算法理解" in proc.stdout
    assert "VGG" in run_cli("report", str(path), "--format", "md").stdout


# --------------------------------------------------------------------------- #
# 3. scan writes model / node / group meta
# --------------------------------------------------------------------------- #
def test_scan_samples_writes_algorithm_meta(tmp_path: Path) -> None:
    out = tmp_path / "scan_algo"
    proc = run_cli("scan", "samples", "--out", str(out), "--quiet", "--max-files", "12")
    assert proc.returncode == 0, proc.stderr
    index = load_json(out / "index.json")
    assert index["models"], "scan produced no models"

    group_patterns_total = 0
    models_with_family = 0
    for entry in index["models"]:
        graph = load_json(out / entry["file"])
        assert validate_graph(graph) == [], f"{entry['id']}: invalid graph after scan"
        model = graph["model"]
        meta = model.get("meta")
        assert isinstance(meta, dict) and meta, f"{entry['id']}: model.meta missing"
        assert isinstance(meta.get("family"), str) and meta["family"].strip()
        assert isinstance(meta.get("familyLabel"), str) and meta["familyLabel"].strip()
        confidence = meta.get("familyConfidence")
        assert isinstance(confidence, (int, float)) and not isinstance(confidence, bool)
        assert 0.0 <= float(confidence) <= 1.0
        assert isinstance(meta.get("patterns"), list)
        assert isinstance(meta.get("pseudocode"), list)
        assert meta["pseudocode"], f"{entry['id']}: scan did not write pseudocode"
        assert isinstance(meta.get("algorithmTags"), list)
        assert 1 <= len(meta.get("families") or []) <= 3
        models_with_family += 1

        for node in graph["nodes"]:
            node_meta = node.get("meta")
            assert isinstance(node_meta, dict), f"{entry['id']}/{node['id']}: node.meta missing"
            assert isinstance(node_meta.get("formula"), str) and node_meta["formula"].strip()
            assert isinstance(node_meta.get("semantic"), str) and node_meta["semantic"].strip()
            assert isinstance(node_meta.get("tags"), list) and node_meta["tags"]

        for group in graph["groups"]:
            group_meta = group.get("meta")
            if isinstance(group_meta, dict) and isinstance(group_meta.get("pattern"), dict):
                pattern = group_meta["pattern"]
                assert pattern.get("kind") and pattern.get("label")
                assert str(pattern.get("formula") or "").strip()
                assert 0.0 <= float(pattern.get("confidence") or 0) <= 1.0
                group_patterns_total += 1

    assert models_with_family == len(index["models"])
    assert group_patterns_total >= 1, "no group.meta.pattern was written by scan"

    # The scan directory is a valid report source and carries the algorithm section.
    first = index["models"][0]
    proc = run_cli("report", str(out), "--model", first["id"], "--format", "md")
    assert proc.returncode == 0, proc.stderr
    assert "算法理解" in proc.stdout

    # explain on the scan output reuses the written meta (same family).
    graph = load_json(out / first["file"])
    payload = explain_json(str(out), "--model", first["id"])
    assert payload["family"] == graph["model"]["meta"]["family"]
    assert len(payload["patterns"]) == len(graph["model"]["meta"]["patterns"])


# --------------------------------------------------------------------------- #
# 4. knowledge base end-to-end checks
# --------------------------------------------------------------------------- #
def test_knowledge_base_counts_validation_and_loader() -> None:
    path = algorithm_kb.knowledge_path()
    assert path.is_file() and path.name == algorithm_kb.KB_FILENAME
    assert algorithm_kb.load_knowledge() is algorithm_kb.load_knowledge(), "KB cache is not reused"

    families = algorithm_kb.families()
    components = algorithm_kb.components()
    assert len(families) == 102, f"expected 102 families, got {len(families)}"
    assert len(components) == 56, f"expected 56 components, got {len(components)}"
    assert algorithm_kb.validate_knowledge() == []

    family_ids = [entry["id"] for entry in families]
    component_ids = [entry["id"] for entry in components]
    assert len(set(family_ids)) == len(family_ids)
    assert len(set(component_ids)) == len(component_ids)
    for entry in families + components:
        assert re.fullmatch(r"[a-z][a-z0-9_]*", entry["id"]), f"bad id {entry['id']!r}"

    # Every component must be referenced by a family (validate covers it; assert anyway).
    family_index = algorithm_kb.family_index()
    component_index = algorithm_kb.component_index()
    referenced = set()
    for family in families:
        referenced.update(family["required_patterns"])
        referenced.update(family["optional_patterns"])
    assert referenced == set(component_index), "unreferenced / unknown component ids"

    assert set(algorithm_kb.SIGNATURE_OPS) <= set(KINDS)
    for family in families:
        assert set(family["signature_ops"]) <= set(KINDS)
        for cid in family["required_patterns"] + family["optional_patterns"]:
            assert cid in component_index
            assert str(component_index[cid]["formula"]).strip()
            assert component_index[cid]["description"].strip()


def test_knowledge_base_key_family_signatures() -> None:
    family_index = algorithm_kb.family_index()
    component_index = algorithm_kb.component_index()
    for family_id, expected in KEY_FAMILIES.items():
        family = family_index.get(family_id)
        assert family is not None, f"KB is missing required family {family_id!r}"
        assert family["domain"] == expected["domain"], f"{family_id}: wrong domain"
        assert family["label"].strip()
        assert any(expected["alias"] in alias.lower() for alias in family["aliases"]), (
            f"{family_id}: canonical alias {expected['alias']!r} missing"
        )
        assert expected["formula_token"] in family["formula"], (
            f"{family_id}: formula does not describe the architecture: {family['formula']!r}"
        )
        assert family["required_patterns"], f"{family_id}: required_patterns must not be empty"
        for pattern_id in family["required_patterns"]:
            assert pattern_id in component_index, f"{family_id}: unknown component {pattern_id!r}"
            assert str(component_index[pattern_id]["formula"]).strip()
        assert len(family["signature_ops"]) >= 3, f"{family_id}: signature too short"
        assert set(family["signature_ops"]) <= set(KINDS)
        assert family["references"], f"{family_id}: references missing"
        for url in family["references"]:
            assert url.startswith("http://") or url.startswith("https://"), (
                f"{family_id}: reference is not an http(s) URL: {url!r}"
            )
        assert 0 < len(family["summary"]) <= 50, f"{family_id}: summary length contract"


# --------------------------------------------------------------------------- #
# 5. unknown / low-confidence graphs
# --------------------------------------------------------------------------- #
def test_unknown_graph_is_classified_unknown_without_crashing(tmp_path: Path) -> None:
    graph = make_mystery_graph()
    assert validate_graph(graph) == []
    path = tmp_path / "mystery.json"
    path.write_text(json.dumps(graph, ensure_ascii=False), encoding="utf-8")

    payload = explain_json(str(path))
    assert payload["family"] == "unknown"
    assert isinstance(payload["unknown_reason"], str) and payload["unknown_reason"].strip()
    assert isinstance(payload["families"], list)
    assert payload["pseudocode"], "even unknown graphs should get basic pseudocode"
    assert any("MysteryOp" in line or "mystery" in line for line in payload["pseudocode"])

    # An empty document must not crash either.
    empty = tmp_path / "empty.json"
    empty.write_text("{}", encoding="utf-8")
    payload = explain_json(str(empty))
    assert payload["family"] == "unknown"
    assert payload["unknown_reason"]


def test_low_confidence_graph_does_not_crash() -> None:
    payload = explain_json("examples/demo", "--model", "needs_args__needsargs")
    assert payload["family"] and payload["family"] != "unknown"
    assert 0.0 <= float(payload["confidence"]) <= 0.6, (
        "NeedsArgs demo is a weak match; confidence should stay low"
    )
    assert isinstance(payload["families"], list)
    assert payload["pseudocode"]
