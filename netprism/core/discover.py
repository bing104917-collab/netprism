"""File walking and AST level model discovery.

Discovery never imports or executes user code: it parses ``*.py`` files and
builds enough structure for the tracer:

* ``nn.Module`` subclasses with a ``forward`` method;
* module level zero-argument factory functions (``def ResNet18(): return
  ResNet(BasicBlock, [2, 2, 2, 2])``), resolved transitively through helper
  functions (``def cct_2(): return _cct(...)``);
* ``dataclass`` configuration classes (``GPTConfig``) used to synthesize
  constructor arguments;
* a reference graph so building blocks (``BasicBlock``, ``Attention``,
  ``Transition``...) are not reported as standalone top-level models.
"""

from __future__ import annotations

import ast
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

from .errors import AnalyzeError

LOGGER = logging.getLogger("netprism.core.discover")

SKIP_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "env",
        "node_modules",
        "__pycache__",
        "site-packages",
        "build",
        "dist",
        ".netprism_out",
        ".mypy_cache",
        ".pytest_cache",
        ".idea",
        ".vscode",
        "netprism_out",
    }
)

MAX_FILE_BYTES = 2 * 1024 * 1024

_MODULE_BASES = frozenset(
    {
        "torch.nn.Module",
        "torch.nn.modules.module.Module",
        "torch.nn.modules.Module",
        "nn.Module",
        "Module",
    }
)

FACTORY_FUNCTIONS = ("build_model", "get_model", "create_model", "build", "make_model")

#: Parameter names that strongly suggest a training/utility wrapper rather than
#: a standalone network (a wrapper needs another model passed in).
WRAPPER_PARAM_NAMES = frozenset(
    {
        "model",
        "models",
        "net",
        "network",
        "backbone",
        "teacher",
        "student",
        "encoder",
        "decoder",
        "vit",
        "model_ema",
        "student_model",
        "teacher_model",
        "student_net",
        "teacher_net",
        "transformer",
    }
)

WRAPPER_NAME_RE = re.compile(r"(?:Trainer|Wrapper|Distiller|Mixin|Loss|Recorder|Extractor|Adapter)$", re.I)

#: Dead block-like helpers left in a file that already exposes factories.
BLOCK_LIKE_RE = re.compile(r"(?:Block|Bottleneck|Cell|Transition|Down|Up|Conv|Attention|MLP|FFN|Layer|Head|Norm|SE)$", re.I)

#: Component-looking helpers that can be dropped when another model exists in
#: the same file (they are never referenced anywhere in that file).
COMPONENT_NAME_RE = re.compile(
    r"(?:Head|Loss|Block|Cell|Attention|MLP|FFN|Layer|Norm|Transformer|Encoder|Decoder|Tokenizer|Adapter|Mixin)$",
    re.I,
)


@dataclass
class InitParam:
    name: str
    has_default: bool
    annotation: str | None = None


@dataclass
class DataclassInfo:
    name: str
    node: ast.ClassDef
    line: int
    fields: list[InitParam] = field(default_factory=list)

    @property
    def all_default(self) -> bool:
        return all(item.has_default for item in self.fields)


@dataclass
class FactoryInfo:
    """A module level function that returns an ``nn.Module`` instance."""

    name: str
    node: ast.FunctionDef
    line: int
    required_args: list[str] = field(default_factory=list)
    returns_class: str | None = None
    references: set[str] = field(default_factory=set)

    @property
    def usable(self) -> bool:
        return not self.required_args and bool(self.returns_class)


@dataclass
class ModuleClass:
    """A parsed ``nn.Module`` subclass."""

    name: str
    node: ast.ClassDef
    file: Path
    source: str
    bases: list[str] = field(default_factory=list)
    is_module: bool = False
    has_forward: bool = False
    init_node: ast.FunctionDef | None = None
    methods: dict[str, ast.FunctionDef] = field(default_factory=dict)
    line: int = 0
    required_init_args: list[str] = field(default_factory=list)
    instantiates: set[str] = field(default_factory=set)
    init_params: list[InitParam] = field(default_factory=list)

    @property
    def code(self) -> str:
        segment = ast.get_source_segment(self.source, self.node)
        if segment:
            return segment.splitlines()[0].strip()
        return f"class {self.name}"

    @property
    def default_constructible(self) -> bool:
        return not self.required_init_args

    @property
    def annotations(self) -> dict[str, str]:
        return {item.name: item.annotation for item in self.init_params if item.annotation}


@dataclass
class ModelCandidate:
    """One model to analyze: either a class or a factory function."""

    display_name: str
    class_name: str
    module_class: ModuleClass
    factory: FactoryInfo | None = None

    @property
    def name(self) -> str:
        return self.display_name

    @property
    def file(self) -> Path:
        return self.module_class.file

    @property
    def line(self) -> int:
        return self.module_class.line

    @property
    def required_init_args(self) -> list[str]:
        return self.module_class.required_init_args

    @property
    def default_constructible(self) -> bool:
        return self.factory is None and self.module_class.default_constructible

    @property
    def kind(self) -> str:
        return "factory" if self.factory is not None else "class"


@dataclass
class ModuleFile:
    """Result of parsing one Python source file."""

    path: Path
    source: str
    tree: ast.Module
    aliases: dict[str, str] = field(default_factory=dict)
    classes: dict[str, ModuleClass] = field(default_factory=dict)
    dataclasses: dict[str, DataclassInfo] = field(default_factory=dict)
    factories: list[FactoryInfo] = field(default_factory=list)
    module_level_models: list[str] = field(default_factory=list)
    factory_returns: dict[str, list[str]] = field(default_factory=dict)
    class_references: dict[str, set[str]] = field(default_factory=dict)
    class_instantiations: dict[str, set[str]] = field(default_factory=dict)
    ctor_string_literals: dict[str, list[str]] = field(default_factory=dict)
    class_ctor_args: dict[str, dict[str, list[object]]] = field(default_factory=dict)

    def model_classes(self) -> list[ModuleClass]:
        return [cls for cls in self.classes.values() if cls.is_module and cls.has_forward]


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------

def _dotted(node: ast.AST) -> str | None:
    parts: list[str] = []
    current: ast.AST | None = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
        return ".".join(reversed(parts))
    return None


def _annotation(node: ast.AST | None) -> str | None:
    if node is None:
        return None
    try:
        return ast.unparse(node)
    except Exception:  # pragma: no cover - very old ASTs
        return _dotted(node)


def _resolve_alias(dotted: str, aliases: dict[str, str]) -> str:
    if not dotted:
        return ""
    head, _, tail = dotted.partition(".")
    target = aliases.get(head)
    if target is None:
        return dotted
    return f"{target}.{tail}" if tail else target


def _build_aliases(tree: ast.Module) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                aliases[alias.asname or alias.name] = alias.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                if alias.name == "*":
                    continue
                aliases[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    return aliases


def _class_bases(node: ast.ClassDef, aliases: dict[str, str]) -> list[str]:
    bases: list[str] = []
    for base in node.bases:
        dotted = _dotted(base)
        if dotted:
            bases.append(_resolve_alias(dotted, aliases))
    return bases


def _called_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _required_args(function: ast.FunctionDef | None) -> list[str]:
    if function is None:
        return []
    args = function.args
    positional = list(getattr(args, "posonlyargs", [])) + list(args.args)
    required: list[str] = []
    defaults = list(args.defaults)
    first_default = len(positional) - len(defaults)
    for index, arg in enumerate(positional):
        if arg.arg in ("self", "cls"):
            continue
        if index < first_default:
            required.append(arg.arg)
    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        if default is None:
            required.append(arg.arg)
    return required


def _init_params(function: ast.FunctionDef | None) -> list[InitParam]:
    if function is None:
        return []
    args = function.args
    positional = list(getattr(args, "posonlyargs", [])) + list(args.args)
    defaults = list(args.defaults)
    first_default = len(positional) - len(defaults)
    params: list[InitParam] = []
    for index, arg in enumerate(positional):
        if arg.arg in ("self", "cls"):
            continue
        params.append(
            InitParam(
                name=arg.arg,
                has_default=index >= first_default,
                annotation=_annotation(arg.annotation),
            )
        )
    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        params.append(InitParam(name=arg.arg, has_default=default is not None, annotation=_annotation(arg.annotation)))
    return params


def _iter_function_calls(node: ast.AST, *, skip_nested: bool = True) -> Iterator[ast.Call]:
    """Yield calls in a function/class body, skipping nested def/class bodies."""

    def walk(current: ast.AST) -> Iterator[ast.Call]:
        for child in ast.iter_child_nodes(current):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                if skip_nested:
                    continue
            if isinstance(child, ast.Call):
                yield child
            yield from walk(child)

    return walk(node)


def _iter_returns(node: ast.AST) -> Iterator[ast.Return]:
    def walk(current: ast.AST) -> Iterator[ast.Return]:
        for child in ast.iter_child_nodes(current):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                continue
            if isinstance(child, ast.Return):
                yield child
            yield from walk(child)

    return walk(node)


def _is_dataclass(node: ast.ClassDef) -> bool:
    for decorator in node.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        name = _dotted(target) or ""
        if "dataclass" in name.lower():
            return True
    return False


def _dataclass_fields(node: ast.ClassDef) -> list[InitParam]:
    fields: list[InitParam] = []
    for item in node.body:
        if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
            fields.append(
                InitParam(
                    name=item.target.id,
                    has_default=item.value is not None,
                    annotation=_annotation(item.annotation),
                )
            )
        elif isinstance(item, ast.Assign):
            for target in item.targets:
                if isinstance(target, ast.Name):
                    fields.append(InitParam(name=target.id, has_default=True, annotation=None))
    return fields


def _collect_ctor_literals(tree: ast.Module, class_names: set[str]) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        target = _called_name(node)
        if target not in class_names:
            continue
        for arg in list(node.args) + [keyword.value for keyword in node.keywords]:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                bucket = found.setdefault(target, [])
                if arg.value not in bucket:
                    bucket.append(arg.value)
    return found


def _executable_refs(node: ast.AST, names: set[str]) -> set[str]:
    """Names from ``names`` referenced in executable positions (annotations skipped)."""
    found: set[str] = set()

    def walk(current: ast.AST) -> None:
        if isinstance(current, ast.arg):
            return
        if isinstance(current, ast.arguments):
            # Default values are executable (e.g. ``DLA(block=BasicBlock)``);
            # annotations are not.
            for default in list(current.defaults) + [item for item in current.kw_defaults if item is not None]:
                walk(default)
            return
        if isinstance(current, ast.AnnAssign):
            if current.value is not None:
                walk(current.value)
            return
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            for child in ast.iter_child_nodes(current):
                if isinstance(child, (ast.arg, ast.AnnAssign)):
                    continue
                walk(child)
            return
        if isinstance(current, ast.Name):
            if current.id in names:
                found.add(current.id)
            return
        if isinstance(current, ast.Attribute):
            if current.attr in names:
                found.add(current.attr)
            return
        for child in ast.iter_child_nodes(current):
            walk(child)

    walk(node)
    return found


def _literal_value(node: ast.AST) -> object:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float, bool, str)):
        return node.value
    return _MISSING


_MISSING = object()


def _collect_ctor_arguments(tree: ast.Module, classes: dict[str, "ModuleClass"]) -> dict[str, dict[str, list[object]]]:
    """Collect literal constructor arguments from call sites in this file.

    ``VGG('VGG11')`` -> ``{'VGG': {'vgg_name': ['VGG11']}}``;
    ``ShuffleNetV2(net_size=0.5)`` -> ``{'ShuffleNetV2': {'net_size': [0.5]}}``.
    """
    found: dict[str, dict[str, list[object]]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        target = _called_name(node)
        module_class = classes.get(target or "")
        if module_class is None or not module_class.is_module:
            continue
        params = [item.name for item in module_class.init_params]
        bucket = found.setdefault(module_class.name, {})
        for index, arg in enumerate(node.args):
            value = _literal_value(arg)
            if value is _MISSING or index >= len(params):
                continue
            values = bucket.setdefault(params[index], [])
            if value not in values:
                values.append(value)
        for keyword in node.keywords:
            if keyword.arg is None:
                continue
            value = _literal_value(keyword.value)
            if value is _MISSING:
                continue
            values = bucket.setdefault(keyword.arg, [])
            if value not in values:
                values.append(value)
    return found


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------

def parse_source(path: Path, source: str | None = None) -> ModuleFile:
    """Parse ``path`` and collect module classes / factories; raises :class:`AnalyzeError`."""
    if source is None:
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise AnalyzeError(f"cannot read file: {exc}", file=path) from exc
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        raise AnalyzeError(f"syntax error: {exc.msg}", file=path, line=exc.lineno or None) from exc

    aliases = _build_aliases(tree)
    info = ModuleFile(path=path, source=source, tree=tree, aliases=aliases)

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            module_class = ModuleClass(
                name=node.name,
                node=node,
                file=path,
                source=source,
                bases=_class_bases(node, aliases),
                line=node.lineno,
            )
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    module_class.methods[item.name] = item
                    if item.name == "forward":
                        module_class.has_forward = True
                    elif item.name == "__init__":
                        module_class.init_node = item
            module_class.required_init_args = _required_args(module_class.init_node)
            module_class.init_params = _init_params(module_class.init_node)
            info.classes[node.name] = module_class
            if _is_dataclass(node):
                info.dataclasses[node.name] = DataclassInfo(
                    name=node.name,
                    node=node,
                    line=node.lineno,
                    fields=_dataclass_fields(node),
                )

    module_functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    local_class_names = set(info.classes)
    model_class_names = set()

    def is_module_name(name: str, seen: set[str] | None = None) -> bool:
        module_class = info.classes.get(name)
        if module_class is None:
            return False
        seen = seen or set()
        if name in seen:
            return False
        seen.add(name)
        for base in module_class.bases:
            if base in _MODULE_BASES or base.endswith(".Module"):
                return True
            head = base.split(".", 1)[0]
            if head in info.classes and is_module_name(head, seen):
                return True
        return False

    for name, module_class in info.classes.items():
        module_class.is_module = is_module_name(name)
        if module_class.is_module:
            model_class_names.add(name)

    # Instantiations and executable references per class (all methods).
    for name, module_class in info.classes.items():
        if not module_class.is_module:
            continue
        instantiations: set[str] = set()
        references: set[str] = set()
        for method in module_class.methods.values():
            for call in _iter_function_calls(method):
                target = _called_name(call)
                if target and target in info.classes and target != name:
                    if info.classes[target].is_module:
                        instantiations.add(target)
            references.update(_executable_refs(method, model_class_names))
        for base in module_class.bases:
            head = base.split(".", 1)[0]
            if head in info.classes and head != name:
                references.add(head)
        references.discard(name)
        module_class.instantiates = instantiations
        info.class_instantiations[name] = instantiations
        info.class_references[name] = references

    # Module level helper functions: class references + call graph.
    function_class_refs: dict[str, set[str]] = {}
    function_calls: dict[str, set[str]] = {}
    for name, function in module_functions.items():
        function_class_refs[name] = _executable_refs(function, model_class_names)
        function_calls[name] = {
            target
            for call in _iter_function_calls(function)
            for target in [_called_name(call)]
            if target and target in module_functions and target != name
        }

    def expand_function_refs(function_name: str, seen: set[str] | None = None) -> set[str]:
        seen = seen or set()
        if function_name in seen:
            return set()
        seen.add(function_name)
        refs = set(function_class_refs.get(function_name, set()))
        for callee in function_calls.get(function_name, set()):
            refs |= expand_function_refs(callee, seen)
        return refs

    resolver_cache: dict[str, str | None] = {}

    def _resolve_model_expr(
        expression: ast.AST | None,
        assigned: dict[str, ast.AST],
        seen: set[str],
    ) -> str | None:
        if expression is None:
            return None
        if isinstance(expression, ast.Name):
            if expression.id in assigned:
                return _resolve_model_expr(assigned[expression.id], assigned, seen)
            return None
        if isinstance(expression, ast.IfExp):
            return _resolve_model_expr(expression.body, assigned, seen) or _resolve_model_expr(
                expression.orelse, assigned, seen
            )
        if isinstance(expression, ast.Call):
            target = _called_name(expression)
            if not target:
                return None
            if target in model_class_names:
                return target
            if target in module_functions:
                return resolve_factory_class(target, set(seen))
        return None

    def resolve_factory_class(function_name: str, seen: set[str] | None = None) -> str | None:
        if function_name in resolver_cache:
            return resolver_cache[function_name]
        seen = seen or set()
        if function_name in seen:
            return None
        seen.add(function_name)
        function = module_functions.get(function_name)
        if function is None:
            return None
        assigned: dict[str, ast.AST] = {}
        for node in ast.walk(function):
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                if isinstance(node.value, ast.Call):
                    assigned[node.targets[0].id] = node.value
        resolved: str | None = None
        for return_node in _iter_returns(function):
            resolved = _resolve_model_expr(return_node.value, assigned, seen)
            if resolved:
                break
        resolver_cache[function_name] = resolved
        return resolved

    for name, function in module_functions.items():
        if name.startswith("__"):
            continue
        returns_class = resolve_factory_class(name)
        if returns_class is None:
            continue
        references = expand_function_refs(name)
        references.discard(returns_class)
        info.factories.append(
            FactoryInfo(
                name=name,
                node=function,
                line=function.lineno,
                required_args=_required_args(function),
                returns_class=returns_class,
                references=references,
            )
        )

    # Module level ``model = X(...)`` instances.
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    for call in ast.walk(node.value):
                        if isinstance(call, ast.Call):
                            called = _called_name(call)
                            if called and called in info.classes and info.classes[called].is_module:
                                info.module_level_models.append(called)
        elif isinstance(node, ast.FunctionDef):
            for call in ast.walk(node):
                if isinstance(call, ast.Call):
                    called = _called_name(call)
                    if called and called in info.classes and info.classes[called].is_module:
                        info.factory_returns.setdefault(node.name, []).append(called)

    info.ctor_string_literals = _collect_ctor_literals(tree, local_class_names)
    info.class_ctor_args = _collect_ctor_arguments(tree, info.classes)
    return info


# ---------------------------------------------------------------------------
# top-level model selection
# ---------------------------------------------------------------------------

def _wrapper_like(module_class: ModuleClass) -> bool:
    if WRAPPER_NAME_RE.search(module_class.name):
        return True
    required = [item.name.lower() for item in module_class.init_params if not item.has_default]
    return any(name in WRAPPER_PARAM_NAMES for name in required)


def _expanded_class_references(info: ModuleFile) -> set[str]:
    """Classes referenced by another model class, directly or via helper functions."""
    model_class_names = {name for name, cls in info.classes.items() if cls.is_module}
    module_functions = {node.name: node for node in info.tree.body if isinstance(node, ast.FunctionDef)}

    function_class_refs: dict[str, set[str]] = {}
    function_calls: dict[str, set[str]] = {}
    for name, function in module_functions.items():
        function_class_refs[name] = _executable_refs(function, model_class_names)
        function_calls[name] = {
            target
            for call in _iter_function_calls(function)
            for target in [_called_name(call)]
            if target and target in module_functions and target != name
        }

    def expand(function_name: str, seen: set[str]) -> set[str]:
        if function_name in seen:
            return set()
        seen.add(function_name)
        refs = set(function_class_refs.get(function_name, set()))
        for callee in function_calls.get(function_name, set()):
            refs |= expand(callee, seen)
        return refs

    consumed: set[str] = set()
    for class_name, module_class in info.classes.items():
        if not module_class.is_module:
            continue
        direct = set(info.class_references.get(class_name, set()))
        direct |= set(info.class_instantiations.get(class_name, set()))
        for method in module_class.methods.values():
            for call in _iter_function_calls(method):
                target = _called_name(call)
                if target and target in module_functions:
                    direct |= expand(target, set())
        direct.discard(class_name)
        consumed |= direct
    return consumed


def _local_top_candidates(info: ModuleFile) -> list[ModelCandidate]:
    """Top-level models of a single file, without any folder-level context."""
    models = info.model_classes()
    if not models:
        return []

    usable_factories = [factory for factory in info.factories if factory.usable and not factory.name.startswith("_")]
    factory_returned = {factory.returns_class for factory in usable_factories if factory.returns_class}
    factory_referenced: set[str] = set()
    for factory in usable_factories:
        factory_referenced |= set(factory.references)

    consumed = _expanded_class_references(info)
    consumed |= factory_referenced
    consumed |= factory_returned
    referenced_anywhere = set(factory_referenced) | set(factory_returned)
    for references in info.class_references.values():
        referenced_anywhere |= set(references)
    for instantiations in info.class_instantiations.values():
        referenced_anywhere |= set(instantiations)

    candidates: list[ModelCandidate] = []
    seen_names: set[str] = set()
    for module_class in sorted(models, key=lambda item: item.line):
        if module_class.name in consumed:
            continue
        if _wrapper_like(module_class):
            LOGGER.debug("skipping wrapper-like class %s in %s", module_class.name, info.path.name)
            continue
        if (
            usable_factories
            and module_class.name not in referenced_anywhere
            and BLOCK_LIKE_RE.search(module_class.name)
        ):
            LOGGER.debug("skipping unused block-like helper %s in %s", module_class.name, info.path.name)
            continue
        candidate = ModelCandidate(
            display_name=module_class.name,
            class_name=module_class.name,
            module_class=module_class,
        )
        if candidate.display_name in seen_names:
            continue
        seen_names.add(candidate.display_name)
        candidates.append(candidate)

    for factory in sorted(usable_factories, key=lambda item: item.line):
        module_class = info.classes.get(factory.returns_class or "")
        if module_class is None or not module_class.is_module or not module_class.has_forward:
            continue
        candidate = ModelCandidate(
            display_name=factory.name,
            class_name=module_class.name,
            module_class=module_class,
            factory=factory,
        )
        if candidate.display_name in seen_names:
            continue
        seen_names.add(candidate.display_name)
        candidates.append(candidate)
    candidates.sort(key=lambda item: (item.module_class.line, item.factory.line if item.factory else item.module_class.line))
    # A file that exposes several models may still contain dead component-like
    # helpers that nothing references (e.g. ``AutoencodingHead`` next to ``WWT``).
    if len(candidates) > 1:
        dead_components = [
            candidate
            for candidate in candidates
            if candidate.factory is None
            and candidate.class_name not in referenced_anywhere
            and COMPONENT_NAME_RE.search(candidate.class_name)
        ]
        if dead_components and len(dead_components) < len(candidates):
            candidates = [candidate for candidate in candidates if candidate not in dead_components]
            LOGGER.debug(
                "skipping unreferenced component-like classes in %s: %s",
                info.path.name,
                ", ".join(candidate.class_name for candidate in dead_components),
            )
    return candidates


def find_top_models(
    info: ModuleFile,
    class_name: str | None = None,
    external_components: set[str] | None = None,
) -> list[ModelCandidate]:
    """Return the models that should be analyzed for this file.

    ``external_components`` contains class names that other files of the scanned
    folder instantiate as sub-modules; those classes are components, not
    top-level models (folder scans only, see :func:`scan_folder_components`).
    """
    if class_name:
        requested = info.classes.get(class_name)
        if requested is not None and requested.is_module and requested.has_forward:
            return [ModelCandidate(display_name=requested.name, class_name=requested.name, module_class=requested)]
        factories = [
            factory for factory in info.factories if factory.name == class_name and factory.usable
        ]
        if factories:
            factory = factories[0]
            module_class = info.classes.get(factory.returns_class or "")
            if module_class is not None:
                return [
                    ModelCandidate(
                        display_name=factory.name,
                        class_name=module_class.name,
                        module_class=module_class,
                        factory=factory,
                    )
                ]
        raise AnalyzeError(f"class {class_name!r} not found", file=info.path)
    candidates = _local_top_candidates(info)
    if external_components:
        # Factories are explicit entry points: keep them even if the underlying
        # class is reused as a component somewhere else.
        candidates = [
            candidate
            for candidate in candidates
            if candidate.factory is not None or candidate.class_name not in external_components
        ]
    return candidates


# ---------------------------------------------------------------------------
# folder-level cross-file component detection
# ---------------------------------------------------------------------------

def _try_module_path(base: Path, module: str) -> Path | None:
    parts = [part for part in module.split(".") if part]
    if not parts:
        return None
    candidate = base.joinpath(*parts).with_suffix(".py")
    if candidate.is_file():
        return candidate.resolve()
    package = base.joinpath(*parts) / "__init__.py"
    if package.is_file():
        return package.resolve()
    return None


def _build_module_index(infos: dict[Path, ModuleFile], root: Path) -> dict[str, Path]:
    """Map dotted module names (and unique bare names) to files below ``root``."""
    raw: dict[str, list[Path]] = {}
    for path in infos:
        try:
            relative = path.relative_to(root)
        except ValueError:
            continue
        parts = list(relative.with_suffix("").parts)
        if parts and parts[-1] == "__init__":
            parts = parts[:-1]
        if not parts:
            continue
        dotted = ".".join(parts)
        raw.setdefault(dotted, []).append(path)
        raw.setdefault(parts[-1], []).append(path)
    return {key: values[0] for key, values in raw.items() if len(values) == 1}


def _resolve_module_file(
    root: Path,
    importer: Path,
    module: str | None,
    level: int,
    module_index: dict[str, Path],
) -> Path | None:
    if not module:
        return None
    if level and level > 0:
        base = importer.parent
        for _ in range(level - 1):
            base = base.parent
        return _try_module_path(base, module)
    for base in (importer.parent, root):
        resolved = _try_module_path(base, module)
        if resolved is not None:
            return resolved
    return module_index.get(module)


class _Imports:
    """Local import bindings of one file (classes and module aliases)."""

    def __init__(self) -> None:
        self.classes: dict[str, tuple[Path, str]] = {}
        self.modules: dict[str, Path] = {}


def _collect_imports(
    info: ModuleFile,
    path: Path,
    root: Path,
    module_index: dict[str, Path],
    infos: dict[Path, ModuleFile],
) -> _Imports:
    imports = _Imports()
    for node in ast.walk(info.tree):
        if isinstance(node, ast.ImportFrom):
            target_path = _resolve_module_file(root, path, node.module, node.level or 0, module_index)
            if target_path is None:
                continue
            target_info = infos.get(target_path)
            if target_info is None:
                continue
            if node.module is None:
                # ``from . import submodule``
                base = path.parent
                for _ in range((node.level or 1) - 1):
                    base = base.parent
                for alias in node.names:
                    sub = _try_module_path(base, alias.name)
                    if sub is not None:
                        imports.modules[alias.asname or alias.name] = sub
                continue
            for alias in node.names:
                if alias.name == "*":
                    for class_name, module_class in target_info.classes.items():
                        if module_class.is_module:
                            imports.classes[class_name] = (target_path, class_name)
                    continue
                local_name = alias.asname or alias.name
                target_class = target_info.classes.get(alias.name)
                if target_class is not None and target_class.is_module:
                    imports.classes[local_name] = (target_path, target_class.name)
                    continue
                if target_path.suffix == ".py":
                    sibling = _try_module_path(target_path.parent, alias.name)
                    if sibling is not None:
                        imports.modules[local_name] = sibling
        elif isinstance(node, ast.Import):
            for alias in node.names:
                resolved = _resolve_module_file(root, path, alias.name, 0, module_index)
                if resolved is None:
                    continue
                if alias.asname:
                    imports.modules[alias.asname] = resolved
                else:
                    first = alias.name.split(".")[0]
                    first_path = _resolve_module_file(root, path, first, 0, module_index)
                    if first_path is not None:
                        imports.modules[first] = first_path
    return imports


def _resolve_imported_call(
    call: ast.Call,
    imports: _Imports,
    infos: dict[Path, ModuleFile],
) -> tuple[Path, str] | None:
    func = call.func
    if isinstance(func, ast.Name):
        return imports.classes.get(func.id)
    dotted = _dotted(func)
    chain = dotted.split(".") if dotted else []
    if len(chain) == 2:
        module_path = imports.modules.get(chain[0])
        if module_path is not None:
            target_info = infos.get(module_path)
            if target_info is not None:
                module_class = target_info.classes.get(chain[1])
                if module_class is not None and module_class.is_module:
                    return (module_path, module_class.name)
    return None


def scan_folder_components(files: Iterable[Path], root: str | Path) -> dict[Path, set[str]]:
    """Find classes that other files instantiate as sub-modules.

    Returns ``{defining_file: {class_name, ...}}``.  Only instantiations inside a
    *module class method* of another file count; references from training
    scripts, module-level code or factory functions do not consume a model.
    """
    root = Path(root).resolve()
    infos: dict[Path, ModuleFile] = {}
    for path in files:
        try:
            infos[path.resolve()] = parse_source(path)
        except AnalyzeError:
            LOGGER.debug("component scan skipped unparsable file %s", path)
    if not infos:
        return {}
    module_index = _build_module_index(infos, root)
    consumed: dict[Path, set[str]] = {}
    for path, info in infos.items():
        imports = _collect_imports(info, path, root, module_index, infos)
        if not imports.classes and not imports.modules:
            continue
        for candidate in _local_top_candidates(info):
            if candidate.factory is not None:
                continue
            module_class = candidate.module_class
            for method in module_class.methods.values():
                for call in _iter_function_calls(method):
                    target = _resolve_imported_call(call, imports, infos)
                    if target is None:
                        continue
                    target_path, target_class = target
                    consumed.setdefault(target_path.resolve(), set()).add(target_class)
    if consumed:
        LOGGER.info(
            "cross-file components suppressed: %s",
            ", ".join(f"{path.name}:{sorted(names)}" for path, names in list(consumed.items())[:6]),
        )
    return consumed


def iter_python_files(
    folder: str | Path,
    *,
    recursive: bool = True,
    max_files: int = 200,
    skip_dirs: Iterable[str] = SKIP_DIRS,
) -> list[Path]:
    """Deterministically list candidate source files below ``folder``."""
    root = Path(folder).expanduser()
    if not root.exists():
        raise AnalyzeError("folder does not exist", file=root)
    if not root.is_dir():
        raise AnalyzeError("not a folder", file=root)
    skip = set(skip_dirs)
    files: list[Path] = []
    iterator = root.rglob("*.py") if recursive else root.glob("*.py")
    for path in iterator:
        try:
            if not path.is_file():
                continue
        except OSError:
            continue
        try:
            relative_parts = path.relative_to(root).parts
        except ValueError:
            relative_parts = path.parts
        if any(part.startswith(".") or part in skip for part in relative_parts[:-1]):
            continue
        try:
            if path.stat().st_size > MAX_FILE_BYTES:
                LOGGER.debug("skipping oversized file %s", path)
                continue
        except OSError:
            continue
        files.append(path)
    files.sort(key=lambda item: str(item).lower())
    return files[: max(0, max_files)]


def detect_framework(info: ModuleFile) -> str:
    for alias in info.aliases.values():
        if alias == "torch" or alias.startswith("torch."):
            return "pytorch"
    return "unknown"


def iter_calls_in(node: ast.AST) -> Iterator[ast.Call]:
    """Public helper used by the static analyzer."""
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            yield child


def has_relative_imports(tree: ast.Module) -> bool:
    """Whether the file uses ``from .pkg import x`` style imports."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.level or 0) > 0:
            return True
    return False


def collect_sibling_string_literals(
    directory: Path, class_name: str, *, exclude: Path | None = None, limit: int = 40
) -> list[str]:
    """Collect string literals passed to ``class_name(...)`` in sibling files.

    Used to synthesize constructor strings such as ``VGG('VGG11')`` from the
    training script that lives next to the model definition.
    """
    values: list[str] = []
    if not directory.is_dir():
        return values
    count = 0
    for path in sorted(directory.glob("*.py")):
        if exclude is not None and path.resolve() == exclude.resolve():
            continue
        count += 1
        if count > limit:
            break
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(source, filename=str(path))
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or _called_name(node) != class_name:
                continue
            for arg in list(node.args) + [keyword.value for keyword in node.keywords]:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value not in values:
                    values.append(arg.value)
    return values


def collect_sibling_ctor_arguments(
    directory: Path,
    class_name: str,
    param_names: list[str],
    *,
    exclude: Path | None = None,
    limit: int = 40,
) -> dict[str, list[object]]:
    """Collect literal constructor arguments from sibling files.

    ``train.py`` calling ``VGG('VGG11')`` or ``ShuffleNetV2(net_size=0.5)`` gives
    the synthesizer real values instead of guesses.
    """
    found: dict[str, list[object]] = {}
    if not directory.is_dir():
        return found
    count = 0
    for path in sorted(directory.glob("*.py")):
        if exclude is not None and path.resolve() == exclude.resolve():
            continue
        count += 1
        if count > limit:
            break
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(source, filename=str(path))
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or _called_name(node) != class_name:
                continue
            for index, arg in enumerate(node.args):
                value = _literal_value(arg)
                if value is _MISSING or index >= len(param_names):
                    continue
                values = found.setdefault(param_names[index], [])
                if value not in values:
                    values.append(value)
            for keyword in node.keywords:
                if keyword.arg is None or keyword.arg not in param_names:
                    continue
                value = _literal_value(keyword.value)
                if value is _MISSING:
                    continue
                values = found.setdefault(keyword.arg, [])
                if value not in values:
                    values.append(value)
    return found


__all__ = [
    "DataclassInfo",
    "FactoryInfo",
    "InitParam",
    "ModelCandidate",
    "ModuleClass",
    "ModuleFile",
    "SKIP_DIRS",
    "FACTORY_FUNCTIONS",
    "collect_sibling_ctor_arguments",
    "collect_sibling_string_literals",
    "detect_framework",
    "find_top_models",
    "has_relative_imports",
    "iter_calls_in",
    "iter_python_files",
    "parse_source",
]
