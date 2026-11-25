"""Tests for the algorithm knowledge base (``netprism/data/algorithm_kb.json``).

Covers the data contract itself (fields, ids, cross references, coverage) and the
standard-library loader (caching, PyInstaller ``_MEIPASS`` path handling, error
handling, no heavy ML imports).
"""

from __future__ import annotations

import ast
import copy
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from netprism import algorithm_kb, schema

REPO_ROOT = Path(__file__).resolve().parents[1]
KB_PATH = REPO_ROOT / "netprism" / "data" / "algorithm_kb.json"
ID_RE = re.compile(r"^[a-z][a-z0-9_]*$")

#: Families the knowledge base must cover (SPEC-ALGO / task-12 coverage list).
REQUIRED_FAMILY_IDS = {
    "lenet", "alexnet", "vgg", "resnet", "resnext", "wide_resnet", "densenet",
    "mobilenet", "mobilenet_v2", "mobilenet_v3", "shufflenet_v1", "shufflenet_v2",
    "efficientnet", "regnet", "convnext", "inception", "xception", "senet", "cbam",
    "sknet", "fpn", "panet", "bifpn", "unet", "unet_plus_plus", "fcn", "deeplabv3",
    "pspnet", "hrnet", "segformer", "vit", "deit", "swin", "beit", "mae", "clip",
    "mlp_mixer", "convmixer", "gmlp", "poolformer", "yolo", "detr", "faster_rcnn",
    "gan", "dcgan", "stylegan", "vae", "diffusion_unet", "rnn", "lstm", "gru",
    "seq2seq", "transformer_encoder", "transformer_decoder", "bert", "gpt", "t5",
    "llama", "moe", "mamba", "rwkv", "retnet", "performer", "gcn", "gat", "graphsage",
    "gin", "pointnet", "pointnet2", "unet3d", "nerf", "whisper", "blip", "kan",
    "capsule", "neural_ode",
}

#: Components the knowledge base must provide (task-12 component list).
REQUIRED_COMPONENT_IDS = {
    "residual", "bottleneck", "depthwise_separable", "inverted_residual", "se", "cbam",
    "sk", "inception", "dense_connection", "fpn", "pafpn", "unet_skip",
    "transformer_block", "self_attention", "cross_attention", "ffn", "gated_mlp",
    "swiglu", "rmsnorm", "rope", "alibi", "moe", "ssm", "gnn_message_passing",
    "point_mlp", "nerf_mlp", "vae_reparam", "normalization", "positional_encoding",
    "classification_head", "segmentation_head", "detection_head", "projection_head",
    "pre_norm", "post_norm", "global_pool", "timestep_embedding",
}

FORBIDDEN_IMPORTS = {"torch", "onnx", "tensorflow", "keras", "numpy", "scipy", "pandas"}


@pytest.fixture(scope="module")
def raw() -> dict:
    """Raw knowledge base document as decoded from disk."""
    return json.loads(KB_PATH.read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def _fresh_cache():
    """Never let a cached document leak between tests."""
    algorithm_kb.clear_cache()
    yield
    algorithm_kb.clear_cache()


# ---------------------------------------------------------------------------
# JSON document contract
# ---------------------------------------------------------------------------


def test_kb_file_is_utf8_json_without_bom():
    data = KB_PATH.read_bytes()
    assert not data.startswith(b"\xef\xbb\xbf"), "knowledge base must not carry a BOM"
    text = data.decode("utf-8")  # raises UnicodeDecodeError when not UTF-8
    document = json.loads(text)
    assert isinstance(document, dict)
    # Chinese summaries are expected; guard against accidental mojibake files.
    assert "�" not in text


def test_top_level_keys(raw):
    assert set(raw) <= {"version", "families", "components"}
    assert isinstance(raw["version"], str) and raw["version"].strip()
    assert isinstance(raw["families"], list)
    assert isinstance(raw["components"], list)


def test_minimum_coverage(raw):
    assert len(raw["families"]) >= 60, f"families: {len(raw['families'])} < 60"
    assert len(raw["components"]) >= 30, f"components: {len(raw['components'])} < 30"


def test_family_ids_are_unique_snake_case(raw):
    ids = [family["id"] for family in raw["families"]]
    assert len(ids) == len(set(ids)), "duplicate family ids"
    for family_id in ids:
        assert ID_RE.match(family_id), f"family id not snake_case: {family_id!r}"


def test_component_ids_are_unique_snake_case(raw):
    ids = [component["id"] for component in raw["components"]]
    assert len(ids) == len(set(ids)), "duplicate component ids"
    for component_id in ids:
        assert ID_RE.match(component_id), f"component id not snake_case: {component_id!r}"


def test_family_required_fields(raw):
    for index, family in enumerate(raw["families"]):
        loc = f"families[{index}] ({family.get('id')})"
        for field in algorithm_kb.FAMILY_FIELDS:
            assert field in family, f"{loc}: missing {field!r}"
        assert isinstance(family["label"], str) and family["label"].strip()
        assert family["domain"] in algorithm_kb.DOMAINS, f"{loc}: bad domain"
        assert isinstance(family["summary"], str) and family["summary"].strip()
        assert isinstance(family["formula"], str) and family["formula"].strip(), (
            f"{loc}: empty formula"
        )


def test_family_summary_length(raw):
    for family in raw["families"]:
        assert len(family["summary"]) <= 50, (
            f"{family['id']}: summary {len(family['summary'])} > 50 chars"
        )


def test_family_aliases_are_non_empty_lowercase(raw):
    for family in raw["families"]:
        aliases = family["aliases"]
        assert isinstance(aliases, list) and aliases, f"{family['id']}: aliases required"
        for alias in aliases:
            assert isinstance(alias, str) and alias.strip(), f"{family['id']}: empty alias"
            assert alias == alias.lower(), f"{family['id']}: alias not lowercase: {alias!r}"
        assert family["id"] in aliases or family["id"].replace("_", " ") in [
            alias for alias in aliases
        ], f"{family['id']}: id (or spaced id) should appear in aliases"


def test_family_signature_ops_are_graph_kinds(raw):
    allowed = set(schema.KINDS)
    for family in raw["families"]:
        ops = family["signature_ops"]
        assert isinstance(ops, list) and ops, f"{family['id']}: signature_ops required"
        unknown = sorted(set(ops) - allowed)
        assert not unknown, f"{family['id']}: unknown signature kinds {unknown}"


def test_family_references_are_urls(raw):
    for family in raw["families"]:
        references = family["references"]
        assert isinstance(references, list) and references, f"{family['id']}: refs required"
        for url in references:
            assert isinstance(url, str) and url.startswith(("https://", "http://")), (
                f"{family['id']}: bad reference {url!r}"
            )


def test_family_patterns_reference_known_components(raw):
    known = {component["id"] for component in raw["components"]}
    for family in raw["families"]:
        for field in ("components", "required_patterns", "optional_patterns"):
            values = family[field]
            assert isinstance(values, list), f"{family['id']}.{field}: must be a list"
            unknown = sorted(set(values) - known)
            assert not unknown, f"{family['id']}.{field}: unknown components {unknown}"


def test_family_required_optional_partition(raw):
    for family in raw["families"]:
        components = set(family["components"])
        required = set(family["required_patterns"])
        optional = set(family["optional_patterns"])
        assert required, f"{family['id']}: at least one required pattern"
        assert not (required & optional), f"{family['id']}: required/optional overlap"
        assert required | optional == components, (
            f"{family['id']}: components must equal required | optional"
        )
        assert len(components) == len(family["components"]), f"{family['id']}: duplicate refs"


def test_component_contract(raw):
    for index, component in enumerate(raw["components"]):
        loc = f"components[{index}] ({component.get('id')})"
        for field in algorithm_kb.COMPONENT_FIELDS:
            assert field in component, f"{loc}: missing {field!r}"
        assert component["formula"].strip(), f"{loc}: empty formula"
        assert component["description"].strip(), f"{loc}: empty description"
        assert isinstance(component["tags"], list) and component["tags"], f"{loc}: tags required"
        assert component["required_ops"], f"{loc}: required_ops required"


def test_component_ops_are_graph_kinds(raw):
    allowed = set(schema.KINDS)
    for component in raw["components"]:
        unknown = sorted(set(component["required_ops"]) - allowed)
        assert not unknown, f"{component['id']}: unknown required_ops {unknown}"


def test_every_component_is_used_by_a_family(raw):
    used = {cid for family in raw["families"] for cid in family["components"]}
    unused = sorted({component["id"] for component in raw["components"]} - used)
    assert not unused, f"components never referenced by a family: {unused}"


def test_required_coverage_lists(raw):
    family_ids = {family["id"] for family in raw["families"]}
    component_ids = {component["id"] for component in raw["components"]}
    assert REQUIRED_FAMILY_IDS <= family_ids, (
        f"missing families: {sorted(REQUIRED_FAMILY_IDS - family_ids)}"
    )
    assert REQUIRED_COMPONENT_IDS <= component_ids, (
        f"missing components: {sorted(REQUIRED_COMPONENT_IDS - component_ids)}"
    )


def test_domains_are_spread_across_the_taxonomy(raw):
    domains = {family["domain"] for family in raw["families"]}
    assert domains == set(algorithm_kb.DOMAINS), f"unused domains: {set(algorithm_kb.DOMAINS) - domains}"


# ---------------------------------------------------------------------------
# Loader behaviour
# ---------------------------------------------------------------------------


def test_loader_matches_raw_document(raw):
    document = algorithm_kb.load_knowledge()
    assert document == raw
    assert algorithm_kb.families() == raw["families"]
    assert algorithm_kb.components() == raw["components"]


def test_loader_caches_document():
    first = algorithm_kb.load_knowledge()
    second = algorithm_kb.load_knowledge()
    assert first is second
    algorithm_kb.clear_cache()
    refreshed = algorithm_kb.load_knowledge()
    assert refreshed is not first
    assert refreshed == first


def test_loader_refresh_forces_reload():
    first = algorithm_kb.load_knowledge()
    reloaded = algorithm_kb.load_knowledge(refresh=True)
    assert reloaded is not first
    assert reloaded == first


def test_family_and_component_indexes(raw):
    families_by_id = algorithm_kb.family_index()
    components_by_id = algorithm_kb.component_index()
    assert len(families_by_id) == len(raw["families"])
    assert len(components_by_id) == len(raw["components"])
    assert set(families_by_id) == {family["id"] for family in raw["families"]}
    assert "resnet" in families_by_id and "swin" in families_by_id and "mamba" in families_by_id
    assert "residual" in components_by_id and "ssm" in components_by_id
    assert families_by_id["resnet"]["domain"] == "vision"


def test_validate_knowledge_accepts_the_real_document(raw):
    assert algorithm_kb.validate_knowledge() == []
    assert algorithm_kb.validate_knowledge(raw) == []


def test_validate_knowledge_rejects_non_objects():
    assert algorithm_kb.validate_knowledge([]) == ["knowledge: must be an object"]


def test_validate_knowledge_reports_contract_breaks(raw):
    broken = copy.deepcopy(raw)
    broken["families"][0]["domain"] = "banana"
    assert any("domain" in error for error in algorithm_kb.validate_knowledge(broken))

    broken = copy.deepcopy(raw)
    broken["families"][0]["summary"] = "x" * 80
    assert any("summary" in error for error in algorithm_kb.validate_knowledge(broken))

    broken = copy.deepcopy(raw)
    broken["families"][0]["required_patterns"] = ["does_not_exist"]
    assert any("unknown component id" in error for error in algorithm_kb.validate_knowledge(broken))

    broken = copy.deepcopy(raw)
    broken["families"].append(copy.deepcopy(broken["families"][0]))
    assert any("duplicate family id" in error for error in algorithm_kb.validate_knowledge(broken))

    broken = copy.deepcopy(raw)
    del broken["families"][0]["formula"]
    assert any("missing field" in error for error in algorithm_kb.validate_knowledge(broken))

    broken = copy.deepcopy(raw)
    broken["components"][0]["required_ops"] = ["not_a_kind"]
    assert any("unknown graph kind" in error for error in algorithm_kb.validate_knowledge(broken))

    broken = copy.deepcopy(raw)
    broken["families"][0]["optional_patterns"] = list(broken["families"][0]["required_patterns"])
    assert any("both required and optional" in error for error in algorithm_kb.validate_knowledge(broken))


# ---------------------------------------------------------------------------
# Path handling (source checkout + PyInstaller)
# ---------------------------------------------------------------------------


def test_source_layout_candidates_include_data_file():
    candidates = [path.resolve() for path in algorithm_kb.knowledge_candidates()]
    assert KB_PATH.resolve() in candidates
    assert algorithm_kb.knowledge_path().resolve() == KB_PATH.resolve()


def test_pyinstaller_meipass_data_layout(tmp_path, monkeypatch):
    frozen_data = tmp_path / "data"
    frozen_data.mkdir()
    payload = {"version": "9.9-frozen", "families": [], "components": []}
    (frozen_data / algorithm_kb.KB_FILENAME).write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    algorithm_kb.clear_cache()
    assert algorithm_kb.knowledge_path() == frozen_data / algorithm_kb.KB_FILENAME
    assert algorithm_kb.load_knowledge()["version"] == "9.9-frozen"


def test_pyinstaller_nested_netprism_layout(tmp_path, monkeypatch):
    frozen_data = tmp_path / "netprism" / "data"
    frozen_data.mkdir(parents=True)
    payload = {"version": "9.9-nested", "families": [], "components": []}
    (frozen_data / algorithm_kb.KB_FILENAME).write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    algorithm_kb.clear_cache()
    assert algorithm_kb.knowledge_path() == frozen_data / algorithm_kb.KB_FILENAME
    assert algorithm_kb.load_knowledge()["version"] == "9.9-nested"


def test_pyinstaller_root_layout_is_supported(tmp_path, monkeypatch):
    payload = {"version": "9.9-root", "families": [], "components": []}
    (tmp_path / algorithm_kb.KB_FILENAME).write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    algorithm_kb.clear_cache()
    assert algorithm_kb.knowledge_path() == tmp_path / algorithm_kb.KB_FILENAME


def test_missing_knowledge_file_raises(monkeypatch):
    monkeypatch.setattr(algorithm_kb, "knowledge_candidates", lambda: ())
    algorithm_kb.clear_cache()
    with pytest.raises(FileNotFoundError) as excinfo:
        algorithm_kb.load_knowledge()
    assert algorithm_kb.KB_FILENAME in str(excinfo.value)


def test_invalid_json_raises_value_error(tmp_path, monkeypatch):
    frozen_data = tmp_path / "data"
    frozen_data.mkdir()
    (frozen_data / algorithm_kb.KB_FILENAME).write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    algorithm_kb.clear_cache()
    with pytest.raises(ValueError) as excinfo:
        algorithm_kb.load_knowledge()
    assert "invalid" in str(excinfo.value).lower()


# ---------------------------------------------------------------------------
# Dependency hygiene
# ---------------------------------------------------------------------------


def test_loader_source_has_no_heavy_ml_imports():
    source = Path(algorithm_kb.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imported.add(node.module.split(".")[0])
    assert not (imported & FORBIDDEN_IMPORTS), (
        f"algorithm_kb imports heavy deps: {sorted(imported & FORBIDDEN_IMPORTS)}"
    )


def test_loader_imports_cleanly_in_subprocess():
    code = (
        "import sys\n"
        "import netprism.algorithm_kb as kb\n"
        "kb.load_knowledge()\n"
        "heavy = [m for m in ('torch','onnx','tensorflow','keras','numpy','scipy') if m in sys.modules]\n"
        "print(','.join(heavy))\n"
        "raise SystemExit(1 if heavy else 0)\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert completed.returncode == 0, (
        f"clean import failed: {completed.stdout}\n{completed.stderr}"
    )
