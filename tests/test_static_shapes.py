from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from netprism.core import AnalyzeOptions, analyze_file
from netprism.schema import validate_graph

FIXTURES = Path(__file__).parent / "fixtures"
BENCHMARK = FIXTURES / "benchmark" / "benchmark"


def _graph(path: Path, **options):
    graphs = analyze_file(path, AnalyzeOptions(execute=False, **options))
    assert graphs, f"no model graph produced for {path}"
    graph = graphs[0]
    assert validate_graph(graph) == []
    return graph


def _warnings(graph) -> list[str]:
    return list(graph["model"]["warnings"])


def _inputs(graph) -> list[tuple[str, list[int]]]:
    return [(spec["name"], list(spec["shape"])) for spec in graph["model"]["inputSpec"]]


def _write_model(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "model.py"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


def test_gat_has_three_named_inputs_and_gather_chain() -> None:
    graph = _graph(FIXTURES / "gat_model.py")
    assert _inputs(graph) == [
        ("q_pos", [1, 1, 3]),
        ("s_pos", [1, 1, 3]),
        ("s_val", [1, 1, 3]),
    ]
    warnings = _warnings(graph)
    assert warnings == []
    assert not any("treated as identity" in warning for warning in warnings)
    assert not any("no input shape available" in warning for warning in warnings)
    ops = [node["op"] for node in graph["nodes"]]
    assert any("topk" in op for op in ops)
    assert sum(1 for op in ops if "gather" in op) >= 2
    assert len(graph["nodes"]) >= 40
    assert graph["edges"]
    # A resolved GATLayer must not also be emitted as a fallback block: that
    # would count gat.q/k/v twice and leave its internal chain disconnected.
    names = [node["name"] for node in graph["nodes"]]
    assert "gat" not in names
    assert "gat.out" in names
    assert sum(node.get("numParams") or 0 for node in graph["nodes"]) == graph["model"]["totalParams"]
    incoming = {edge["target"] for edge in graph["edges"]}
    out_node = next(node for node in graph["nodes"] if node["name"] == "gat.out")
    assert out_node["id"] in incoming


def test_implicit_none_return_reuses_internal_tensor_instead_of_fallback(tmp_path: Path) -> None:
    """A custom block without ``return`` must not duplicate its params."""
    path = _write_model(
        tmp_path,
        """
        import torch.nn as nn

        class ImplicitNone(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc = nn.Linear(4, 4)

            def forward(self, x):
                self.fc(x)
        """,
    )
    graph = _graph(path, input_shape=[1, 4])
    model = graph["model"]
    assert model["mode"] == "static"
    assert model["warnings"] == []
    assert model["outputSpec"][0]["shape"] == [1, 4]
    assert sum(node.get("numParams") or 0 for node in graph["nodes"]) == model["totalParams"]
    assert not any(
        node["notes"] and "static analysis fallback" in node["notes"]
        for node in graph["nodes"]
    )


def test_unknown_tensor_function_never_falls_back_to_identity(tmp_path: Path) -> None:
    path = _write_model(
        tmp_path,
        """
        import torch
        import torch.nn as nn

        class Mystery(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc = nn.Linear(6, 6)

            def forward(self, x):
                x = self.fc(x)
                return torch.mystery_shape_op(x)
        """,
    )
    graph = _graph(path, input_shape=[1, 6])
    model = graph["model"]
    assert model["mode"] == "static"
    assert any("shape unknown for 'torch.mystery_shape_op'" in warning for warning in model["warnings"])
    assert not any("treated as identity" in warning for warning in model["warnings"])
    mystery = [node for node in graph["nodes"] if "mystery_shape_op" in node["op"]]
    assert mystery
    assert mystery[0]["outShapes"] == [[-1, -1]]
    assert model["outputSpec"][0]["shape"] == [-1, -1]


def test_unknown_binary_broadcasts_when_shapes_are_compatible(tmp_path: Path) -> None:
    path = _write_model(
        tmp_path,
        """
        import torch
        import torch.nn as nn

        class TwoInputMystery(nn.Module):
            def forward(self, a, b):
                return torch.mystery_binary(a, b)
        """,
    )
    graph = _graph(path)
    assert _inputs(graph) == [("a", [1, 1, 3]), ("b", [1, 1, 3])]
    assert any(
        "shape unknown for 'torch.mystery_binary'" in warning
        for warning in graph["model"]["warnings"]
    )
    mystery = [node for node in graph["nodes"] if "mystery_binary" in node["op"]][0]
    assert mystery["outShapes"] == [[1, 1, 3]]


def test_scalar_min_topk_unpack_and_output_shapes(tmp_path: Path) -> None:
    path = _write_model(
        tmp_path,
        """
        import torch
        import torch.nn as nn

        class TopKNet(nn.Module):
            def forward(self, x):
                k = min(4, 8)
                values, indices = x.topk(k, dim=-1)
                return values + indices
        """,
    )
    graph = _graph(path, input_shape=[1, 8])
    assert graph["model"]["warnings"] == []
    topk = [node for node in graph["nodes"] if "topk" in node["op"]]
    assert len(topk) == 1
    assert len(topk[0]["outShapes"]) == 2
    assert topk[0]["outShapes"][0] == [1, 4]
    assert graph["model"]["outputSpec"][0]["shape"] == [1, 4]


def test_multi_input_specs_are_rank_three_by_default(tmp_path: Path) -> None:
    path = _write_model(
        tmp_path,
        """
        import torch
        import torch.nn as nn

        class ThreeInputs(nn.Module):
            def forward(self, q_pos, s_pos, s_val):
                return q_pos + s_pos + s_val
        """,
    )
    graph = _graph(path)
    assert _inputs(graph) == [
        ("q_pos", [1, 1, 3]),
        ("s_pos", [1, 1, 3]),
        ("s_val", [1, 1, 3]),
    ]
    assert graph["model"]["outputSpec"][0]["shape"] == [1, 1, 3]


def test_import_failure_is_not_a_parse_warning() -> None:
    graph = _graph(FIXTURES / "missing_dep_model.py")
    assert graph["model"]["mode"] == "static"
    assert not any("could not import module" in warning for warning in _warnings(graph))
    assert not any("treated as identity" in warning for warning in _warnings(graph))


@pytest.mark.skipif(not BENCHMARK.is_dir(), reason="benchmark archive not present")
def test_benchmark_models_parse_to_zero_warnings() -> None:
    expected_inputs = {
        "fno_sensor": 1,
        "gat": 3,
        "masked_unet": 1,
        "sensor_transformer": 3,
        "voronoi_cnn": 1,
    }
    seen: dict[str, int] = {}
    differences: dict[str, int] = {}
    for folder, input_count in expected_inputs.items():
        path = BENCHMARK / folder / "model.py"
        if not path.exists():
            continue
        graph = _graph(path)
        model = graph["model"]
        seen[folder] = len(model["inputSpec"])
        assert len(model["inputSpec"]) == input_count, folder
        assert model["mode"] == "static"
        assert model["warnings"] == [], f"{folder}: {model['warnings']}"
        assert not any(
            node["notes"] and "static analysis fallback" in node["notes"]
            for node in graph["nodes"]
        ), folder
        assert not any(
            -1 in shape
            for node in graph["nodes"]
            for shape in (*node["inShapes"], *node["outShapes"])
        ), folder
        assert not any("treated as identity" in warning for warning in model["warnings"])

        node_params = sum(node.get("numParams") or 0 for node in graph["nodes"])
        differences[folder] = model["totalParams"] - node_params
        assert node_params <= model["totalParams"], folder
        if folder == "gat":
            # Fully resolved: every trainable parameter has a node and no
            # GATLayer fallback block remains.
            assert differences[folder] == 0
            assert not any(node["op"] == "GATLayer" for node in graph["nodes"])
        if folder == "fno_sensor":
            # SpectralBlock3d keeps ``weight`` as nn.Parameter; it is emitted as
            # an explicit parameter node, so the node sum matches the model.
            parameter_nodes = [node for node in graph["nodes"] if node["op"] == "Parameter"]
            assert parameter_nodes
            assert sum(node["numParams"] for node in parameter_nodes) == 4 * 262144
            assert differences[folder] == 0
    assert seen, "benchmark archive contained no expected model.py files"
    assert all(difference >= 0 for difference in differences.values())


@pytest.mark.skipif(not BENCHMARK.is_dir(), reason="benchmark archive not present")
def test_unet_loop_keeps_list_state_and_connects_pools() -> None:
    """Regression: ``skips.append(...)`` must advance the static loop state."""
    graph = _graph(BENCHMARK / "masked_unet" / "model.py")
    pools = [node for node in graph["nodes"] if node["op"] == "MaxPool3d"]
    assert len(pools) == 3, [node["name"] for node in pools]
    input_channels = [node["inShapes"][0][1] for node in pools]
    assert input_channels == [16, 32, 64], input_channels
    # Each pool must consume the previous encoder block, not the first block.
    producers = {edge["target"]: edge["source"] for edge in graph["edges"]}
    for index, pool in enumerate(pools[1:], start=1):
        source = graph["nodes"][int(producers[pool["id"]][1:])]
        assert source["name"].startswith(f"downs.{index - 1}.5"), source["name"]
    cats = [node for node in graph["nodes"] if node["op"] == "torch.cat"]
    assert cats, "UNet has no skip concatenations"
    assert all(len(node["inShapes"]) >= 2 for node in cats), [n["inShapes"] for n in cats]
