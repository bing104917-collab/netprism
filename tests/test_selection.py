"""Presentation-layer tests: folding architectural variants for the UI."""

from __future__ import annotations

from netprism.selection import fold_variants, format_folded_note


def _model(model_id: str, name: str, class_name: str, source: str) -> dict:
    return {
        "id": model_id,
        "name": name,
        "className": class_name,
        "source": source,
        "file": f"{model_id}.json",
        "mode": "trace",
        "numLayers": 10,
        "totalParams": 100,
        "totalFlops": 1000,
        "warnings": 0,
    }


def test_fold_variants_keeps_first_per_class_and_file():
    models = [
        _model("resnet__resnet18", "ResNet18", "ResNet", "resnet.py"),
        _model("resnet__resnet34", "ResNet34", "ResNet", "resnet.py"),
        _model("vgg__vgg", "VGG", "VGG", "vgg.py"),
        _model("other__resnet", "ResNet", "ResNet", "other.py"),
    ]
    graphs = {model["id"]: {"model": {"name": model["name"]}} for model in models}
    index = {"schemaVersion": "1.0", "models": models, "errors": []}

    folded, kept_graphs, dropped = fold_variants(index, graphs)

    assert [model["name"] for model in folded["models"]] == ["ResNet18", "VGG", "ResNet"]
    assert set(kept_graphs) == {"resnet__resnet18", "vgg__vgg", "other__resnet"}
    assert [model["name"] for model in dropped] == ["ResNet34"]
    assert index["models"] is models  # input untouched
    note = format_folded_note(dropped)
    assert "1" in note and "ResNet34" in note and "--all-variants" in note


def test_fold_variants_can_be_disabled():
    models = [
        _model("resnet__resnet18", "ResNet18", "ResNet", "resnet.py"),
        _model("resnet__resnet34", "ResNet34", "ResNet", "resnet.py"),
    ]
    index = {"schemaVersion": "1.0", "models": models, "errors": []}
    graphs = {model["id"]: {} for model in models}
    folded, kept_graphs, dropped = fold_variants(index, graphs, enabled=False)
    assert folded is index
    assert kept_graphs is graphs
    assert dropped == []


def test_fold_variants_without_duplicates_is_noop():
    models = [
        _model("a__a", "A", "A", "a.py"),
        _model("b__b", "B", "B", "b.py"),
    ]
    index = {"schemaVersion": "1.0", "models": models, "errors": []}
    graphs = {model["id"]: {} for model in models}
    folded, kept_graphs, dropped = fold_variants(index, graphs)
    assert folded is index
    assert kept_graphs is graphs
    assert dropped == []
