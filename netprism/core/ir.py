"""Internal representation shared by the dynamic tracer and the static parser."""

from __future__ import annotations

from dataclasses import dataclass, field

from .sources import SourceLoc

Shape = list[int]


@dataclass
class NodeSpec:
    """One graph node before ids are assigned."""

    name: str
    op: str
    kind: str
    module_path: str
    in_shapes: list[Shape] = field(default_factory=list)
    out_shapes: list[Shape] = field(default_factory=list)
    params: dict = field(default_factory=dict)
    num_params: int = 0
    flops: int = 0
    tags: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    source: SourceLoc | None = None
    depth: int = 1
    group_id: str | None = None


@dataclass
class EdgeSpec:
    """One graph edge, referencing node indices (not ids)."""

    source: int
    target: int
    kind: str = "data"
    shapes: list[Shape] = field(default_factory=list)
    label: str = ""


@dataclass
class ModuleMeta:
    """Structural information about a (container) module, used for grouping."""

    path: str
    class_name: str
    kind_hint: str = "module"
    is_container: bool = False
    is_sequential: bool = False
    source: SourceLoc | None = None


@dataclass
class Analysis:
    """Mode independent analysis result, ready to be turned into a graph dict."""

    mode: str
    nodes: list[NodeSpec]
    edges: list[EdgeSpec]
    input_specs: list[dict] = field(default_factory=list)
    output_specs: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    total_params: int = 0
    trainable_params: int = 0
    flops_approx: bool = False
    module_meta: dict[str, ModuleMeta] = field(default_factory=dict)
    attention_paths: set[str] = field(default_factory=set)
    class_name: str | None = None
    model_name: str = "model"
    root_source: SourceLoc | None = None


__all__ = ["Shape", "NodeSpec", "EdgeSpec", "ModuleMeta", "Analysis"]
