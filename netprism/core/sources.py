"""Recover ``line`` / ``code`` source locations for traced modules and ops.

Tracing works on live ``nn.Module`` objects, so the mapping from a qualified
module path (``layer1.0.conv1``) back to the source line is rebuilt from the AST:

* ``class_paths[ClassName]['conv1']`` -> the ``self.conv1 = nn.Conv2d(...)`` line
  for assignments inside that class' ``__init__`` (containers are expanded, so
  ``Sequential`` children get ``features.0`` style paths too);
* ``class_op_sources[ClassName]['add']`` -> ordered source locations of the ops
  found in that class' ``forward`` (used for functional nodes such as residual
  adds and ``torch.cat``).
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger("netprism.core.sources")

_CONTAINERS = ("Sequential", "ModuleList", "ModuleDict", "ParameterList", "ParameterDict")


@dataclass(frozen=True)
class SourceLoc:
    """A source line + the (shortened) code snippet on it."""

    line: int
    code: str | None = None


def _short_code(source: str, node: ast.AST, *, limit: int = 200) -> str:
    segment = ast.get_source_segment(source, node) or ""
    collapsed = " ".join(segment.split())
    if len(collapsed) > limit:
        collapsed = collapsed[: limit - 1] + "…"
    return collapsed


def _attr_chain(node: ast.AST) -> list[str] | None:
    parts: list[str] = []
    current: ast.AST | None = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
        return list(reversed(parts))
    return None


def _self_path(target: ast.AST) -> str | None:
    chain = _attr_chain(target)
    if not chain or chain[0] != "self" or len(chain) < 2:
        return None
    return ".".join(chain[1:])


def _container_name(call: ast.Call) -> str | None:
    func = call.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


class _InitCollector:
    def __init__(self, source: str) -> None:
        self.source = source
        self.exact: dict[str, SourceLoc] = {}
        self.prefix: dict[str, SourceLoc] = {}

    def add_call(self, path: str, call: ast.Call) -> None:
        self.exact[path] = SourceLoc(call.lineno, _short_code(self.source, call))

    def walk_value(self, path: str, value: ast.AST) -> None:
        if isinstance(value, ast.Call):
            ctor = _container_name(value)
            if ctor in _CONTAINERS:
                self.walk_container(path, ctor, value)
            else:
                self.add_call(path, value)
            return
        if isinstance(value, (ast.List, ast.Tuple)):
            for index, element in enumerate(value.elts):
                self.walk_value(f"{path}.{index}", element)
            return
        if isinstance(value, ast.Dict):
            for key_node, value_node in zip(value.keys, value.values):
                if isinstance(key_node, ast.Constant):
                    key = key_node.value
                else:
                    key = f"key{len(self.prefix)}"
                self.walk_value(f"{path}.{key}", value_node)
            return
        if isinstance(value, ast.ListComp):
            # ``nn.ModuleList([Block(...) for _ in range(n)])``: every index maps
            # back to the same element constructor.
            for child in ast.walk(value.elt):
                if isinstance(child, ast.Call):
                    self.prefix[path] = SourceLoc(child.lineno, _short_code(self.source, child))
                    break
            return

    def walk_container(self, path: str, ctor: str, call: ast.Call) -> None:
        if ctor == "OrderedDict":
            if call.args and isinstance(call.args[0], (ast.List, ast.Tuple)):
                for element in call.args[0].elts:
                    if isinstance(element, ast.Tuple) and len(element.elts) == 2:
                        key_node, value_node = element.elts
                        key = key_node.value if isinstance(key_node, ast.Constant) else "item"
                        self.walk_value(f"{path}.{key}", value_node)
            return
        args = list(call.args)
        if len(args) == 1 and isinstance(args[0], (ast.List, ast.Tuple, ast.Dict)):
            self.walk_value(path, args[0])
            return
        for index, arg in enumerate(args):
            self.walk_value(f"{path}.{index}", arg)


class _OpCollector:
    """Collect the ordered source locations of interesting ops in a method."""

    _INTERESTING = {
        "add",
        "sub",
        "mul",
        "div",
        "matmul",
        "cat",
        "concat",
        "stack",
        "softmax",
        "log_softmax",
        "flatten",
        "view",
        "reshape",
        "permute",
        "transpose",
        "mean",
        "max",
        "min",
        "sum",
        "masked_fill",
        "dropout",
        "relu",
        "gelu",
        "silu",
        "sigmoid",
        "tanh",
        "embedding",
        "interpolate",
    }

    def __init__(self, source: str) -> None:
        self.source = source
        self.ops: dict[str, list[SourceLoc]] = {}

    def scan(self, node: ast.AST) -> None:
        if isinstance(node, ast.BinOp):
            if isinstance(node.op, ast.Add):
                self._record("add", node)
            elif isinstance(node.op, ast.MatMult):
                self._record("matmul", node)
        if isinstance(node, ast.Call):
            name = _container_name(node)
            if name:
                key = name.lower()
                if key == "concat":
                    key = "concat"
                if key in self._INTERESTING:
                    self._record(key, node)
        for child in ast.iter_child_nodes(node):
            self.scan(child)

    def _record(self, key: str, node: ast.AST) -> None:
        line = getattr(node, "lineno", None)
        if line is None:
            return
        self.ops.setdefault(key, []).append(SourceLoc(line, _short_code(self.source, node)))


@dataclass
class SourceIndex:
    """Per-file source lookup tables."""

    path: Path
    class_paths: dict[str, dict[str, SourceLoc]] = field(default_factory=dict)
    class_sources: dict[str, SourceLoc] = field(default_factory=dict)
    method_sources: dict[str, dict[str, SourceLoc]] = field(default_factory=dict)
    op_sources: dict[str, dict[str, list[SourceLoc]]] = field(default_factory=dict)

    # -- building ---------------------------------------------------------
    @classmethod
    def build(cls, tree: ast.Module, source: str, path: Path) -> "SourceIndex":
        index = cls(path=path)
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            name = node.name
            try:
                class_segment = ast.get_source_segment(source, node) or f"class {name}"
            except Exception:
                class_segment = f"class {name}"
            index.class_sources[name] = SourceLoc(
                node.lineno, " ".join(class_segment.split())[:200]
            )
            paths: dict[str, SourceLoc] = {}
            methods: dict[str, SourceLoc] = {}
            ops: dict[str, list[SourceLoc]] = {}
            for item in node.body:
                if not isinstance(item, ast.FunctionDef):
                    continue
                methods[item.name] = SourceLoc(
                    item.lineno, f"def {item.name}{_signature(item)}:"
                )
                if item.name == "__init__":
                    collector = _InitCollector(source)
                    for statement in item.body:
                        for assign in _iter_assignments(statement):
                            path_name = _self_path(assign[0])
                            if path_name is None:
                                continue
                            collector.walk_value(path_name, assign[1])
                    paths.update(collector.exact)
                    # Prefix fallbacks are only used when the exact path is absent.
                    for key, loc in collector.prefix.items():
                        paths.setdefault(key, loc)
                op_collector = _OpCollector(source)
                op_collector.scan(item)
                for key, locations in op_collector.ops.items():
                    ops.setdefault(key, []).extend(locations)
            index.class_paths[name] = paths
            index.method_sources[name] = methods
            index.op_sources[name] = {
                key: sorted(locations, key=lambda loc: loc.line) for key, locations in ops.items()
            }
        return index

    # -- lookup -----------------------------------------------------------
    def module_source(self, module_path: str, model: Any | None = None) -> SourceLoc | None:
        """Best-effort source for a qualified module path (live model required)."""
        if not module_path:
            return None
        if model is not None:
            prefix = module_path
            while True:
                module = None
                try:
                    module = model.get_submodule(prefix) if prefix else model
                except Exception:
                    module = None
                if module is not None:
                    class_name = type(module).__name__
                    relative = module_path[len(prefix) :].lstrip(".") if prefix else module_path
                    mapping = self.class_paths.get(class_name)
                    if relative and mapping and relative in mapping:
                        return mapping[relative]
                    if not relative:
                        if class_name in self.class_sources:
                            return self.class_sources[class_name]
                if not prefix:
                    break
                prefix = prefix.rsplit(".", 1)[0] if "." in prefix else ""
        return self.class_sources.get(type(model).__name__) if model is not None else None

    def function_source(
        self,
        scope_path: str,
        model: Any | None,
        op_key: str,
        occurrence: int = 0,
    ) -> SourceLoc | None:
        """Source for a functional node inside ``scope_path`` (0-based occurrence)."""
        class_name: str | None = None
        if model is not None:
            try:
                scope_module = model.get_submodule(scope_path) if scope_path else model
                class_name = type(scope_module).__name__
            except Exception:
                class_name = None
        if class_name is None:
            return None
        locations = (self.op_sources.get(class_name) or {}).get(op_key) or []
        if locations:
            return locations[min(occurrence, len(locations) - 1)]
        methods = self.method_sources.get(class_name) or {}
        if "forward" in methods:
            return methods["forward"]
        return self.class_sources.get(class_name)

    def scope_class_name(self, scope_path: str, model: Any | None) -> str | None:
        if model is None:
            return None
        try:
            return type(model.get_submodule(scope_path) if scope_path else model).__name__
        except Exception:
            return None


def _signature(function: ast.FunctionDef) -> str:
    args: list[str] = []
    for arg in function.args.args:
        args.append(arg.arg)
    if function.args.vararg:
        args.append("*" + function.args.vararg.arg)
    if function.args.kwarg:
        args.append("**" + function.args.kwarg.arg)
    return f"({', '.join(args)})"


def _iter_assignments(statement: ast.stmt):
    """Yield ``(target, value)`` for simple assignments (recursing into if/for)."""
    if isinstance(statement, ast.Assign):
        for target in statement.targets:
            yield target, statement.value
        return
    if isinstance(statement, ast.AnnAssign) and statement.value is not None:
        yield statement.target, statement.value
        return
    if isinstance(statement, (ast.If, ast.For, ast.While, ast.With)):
        for child in statement.body:
            yield from _iter_assignments(child)
        for child in getattr(statement, "orelse", []):
            yield from _iter_assignments(child)


__all__ = ["SourceLoc", "SourceIndex"]
