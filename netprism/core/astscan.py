"""Static AST analysis (``mode="static"``).

Nothing is imported or executed: the parser reads ``__init__`` to build a module
tree from constructor calls, then symbolically executes ``forward`` (including
helper methods such as ``_forward_impl`` and ``for`` loops over ``ModuleList``)
to discover data flow and infer tensor shapes, parameters and FLOPs.

Common PyTorch patterns are supported: ``nn.Sequential`` / ``ModuleList`` /
``ModuleDict``, custom sub-modules (inlined when parseable, otherwise emitted as
a single block node), residual ``+``, ``torch.cat`` skip connections,
``flatten`` / ``view`` / ``permute`` / ``transpose``, `F.relu(...)`` chains,
``nn.MultiheadAttention`` and ``nn.TransformerEncoder`` wrappers.  Anything that
cannot be resolved becomes a node with a warning instead of an exception.
"""

from __future__ import annotations

import ast
import logging
import math
import sys
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Sequence

from . import ops
from .discover import ModuleClass, ModuleFile, iter_calls_in
from .errors import AnalyzeError
from .ir import Analysis, EdgeSpec, ModuleMeta, NodeSpec, Shape
from .sources import SourceIndex, SourceLoc

LOGGER = logging.getLogger("netprism.core.astscan")

CONTAINER_OPS = frozenset({"Sequential", "ModuleList", "ModuleDict", "ParameterList", "ParameterDict"})
TRANSFORMER_LAYER_OPS = ("TransformerEncoderLayer",)
TRANSFORMER_STACK_OPS = ("TransformerEncoder",)
FUSED_OPS = ("TransformerDecoderLayer", "TransformerDecoder", "Transformer")
MAX_DEPTH = 12


class _StaticError(Exception):
    """Internal: a value could not be derived statically."""


# ---------------------------------------------------------------------------
# module declaration tree
# ---------------------------------------------------------------------------

@dataclass
class Decl:
    path: str
    op: str
    kind: str = "custom"
    params: dict = field(default_factory=dict)
    children: dict[str, "Decl"] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)
    is_container: bool = False
    is_sequential: bool = False
    class_info: ModuleClass | None = None
    bind: dict[str, Any] = field(default_factory=dict)
    source: SourceLoc | None = None
    num_params: int = 0
    synthetic: bool = False
    is_transformer_layer: bool = False
    attrs: dict[str, Any] = field(default_factory=dict)
    tensor_shape: Shape | None = None

    @property
    def is_leaf(self) -> bool:
        return not self.children

    def add(self, name: str, child: "Decl") -> "Decl":
        if name in self.children:
            suffix = 2
            while f"{name}#{suffix}" in self.children:
                suffix += 1
            name = f"{name}#{suffix}"
        self.children[name] = child
        self.order.append(name)
        return child

    def walk(self) -> Iterator["Decl"]:
        yield self
        for name in self.order:
            yield from self.children[name].walk()

    def total_params(self) -> int:
        if self.children:
            return sum(child.total_params() for child in self.children.values())
        return self.num_params


def _class_last(value: ast.AST) -> str | None:
    if isinstance(value, ast.Attribute):
        return value.attr
    if isinstance(value, ast.Name):
        return value.id
    return None


def _const(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    return _StaticError.__name__  # sentinel not used as a value


def _tensor_creation_shape(node: ast.AST, env: dict[str, Any]) -> Shape | None:
    """Evaluate ``torch.randn(ch, ch, *modes)`` / zeros / empty argument shapes."""
    if isinstance(node, ast.BinOp):
        left = _tensor_creation_shape(node.left, env)
        return left or _tensor_creation_shape(node.right, env)
    if not isinstance(node, ast.Call):
        return None
    name = _class_last(node.func)
    if name not in ("randn", "zeros", "ones", "empty", "rand", "full", "tensor"):
        return None
    dims: list[int] = []
    for argument in node.args:
        if isinstance(argument, ast.Starred):
            value = _eval_value(argument.value, env)
            values = list(value) if isinstance(value, (list, tuple)) else []
        else:
            value = _eval_value(argument, env)
            values = list(value) if isinstance(value, (list, tuple)) else [value]
        for item in values:
            if isinstance(item, int) and not isinstance(item, bool):
                dims.append(item)
    return dims or None


def _source_loc(source: str, node: ast.AST) -> SourceLoc | None:
    line = getattr(node, "lineno", None)
    if line is None:
        return None
    segment = ast.get_source_segment(source, node)
    code = " ".join(segment.split()) if segment else None
    if code and len(code) > 200:
        code = code[:199] + "…"
    return SourceLoc(int(line), code)


def _product(values: Iterable[int]) -> int:
    total = 1
    for value in values:
        total *= int(value)
    return total


def _safe_eval_int(node: ast.AST, env: dict[str, Any]) -> int | _StaticError:
    """Evaluate a small arithmetic expression used in constructor arguments."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool):
            return int(node.value)
        if isinstance(node.value, int):
            return node.value
        raise _StaticError(f"not an int literal: {node.value!r}")
    if isinstance(node, ast.Name):
        value = env.get(node.id)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        raise _StaticError(f"unknown int name {node.id!r}")
    if isinstance(node, ast.Attribute):
        chain = _attribute_chain(node)
        if chain and chain[0] == "self":
            value = env.get(".".join(chain[1:]))
            if isinstance(value, int) and not isinstance(value, bool):
                return value
        raise _StaticError("unsupported attribute int")
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_safe_eval_int(node.operand, env)
    if isinstance(node, ast.BinOp):
        left = _safe_eval_int(node.left, env)
        right = _safe_eval_int(node.right, env)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, (ast.Div, ast.FloorDiv)):
            if right == 0:
                raise _StaticError("division by zero")
            return int(left / right)
        if isinstance(node.op, ast.Pow):
            return int(left**right)
        if isinstance(node.op, ast.Mod):
            if right == 0:
                raise _StaticError("modulo by zero")
            return left % right
    raise _StaticError(f"cannot evaluate {type(node).__name__}")


def _attribute_chain(node: ast.AST) -> list[str] | None:
    parts: list[str] = []
    current: ast.AST | None = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
        return list(reversed(parts))
    return None


def _eval_iterable(node: ast.AST, env: dict[str, Any]) -> list[Any] | None:
    """Evaluate ``range(...)`` / list / tuple expressions for comprehension use."""
    if isinstance(node, (ast.List, ast.Tuple)):
        values = []
        for item in node.elts:
            if isinstance(item, ast.Starred):
                nested = _eval_value(item.value, env)
                if isinstance(nested, (list, tuple)):
                    values.extend(nested)
                continue
            values.append(_eval_value(item, env))
        return values
    if isinstance(node, ast.Call):
        chain = _attribute_chain(node.func)
        name = chain[-1] if chain else None
        if name == "range":
            values = [_eval_value(arg, env) for arg in node.args]
            ints = [int(value) for value in values if isinstance(value, int) and not isinstance(value, bool)]
            if len(ints) == len(values) and ints:
                try:
                    return list(range(*ints))[:256]
                except ValueError:
                    return None
        if name in ("list", "tuple"):
            if len(node.args) == 1:
                value = _eval_value(node.args[0], env)
                if isinstance(value, (list, tuple)):
                    return list(value) if name == "list" else tuple(value)
        return None
    value = _eval_value(node, env)
    return list(value) if isinstance(value, (list, tuple)) else None


def _eval_comprehension(node: ast.AST, env: dict[str, Any]) -> Any:
    """Evaluate simple list/set/generator comprehensions over range/iterables."""
    generators = getattr(node, "generators", [])
    if not generators:
        return None
    element = node.elt
    results: list[Any] = []

    def run(index: int, scope: dict[str, Any]) -> None:
        if index >= len(generators):
            results.append(_eval_value(element, scope))
            return
        generator = generators[index]
        iterable = _eval_iterable(generator.iter, scope)
        if iterable is None:
            return
        for item in iterable[:256]:
            nested = dict(scope)
            target = generator.target
            if isinstance(target, ast.Name):
                nested[target.id] = item
            elif isinstance(target, (ast.Tuple, ast.List)) and isinstance(item, (list, tuple)):
                for sub_target, sub_item in zip(target.elts, item):
                    if isinstance(sub_target, ast.Name):
                        nested[sub_target.id] = sub_item
            run(index + 1, nested)

    run(0, dict(env))
    if isinstance(node, ast.GeneratorExp):
        return tuple(results)
    if isinstance(node, ast.SetComp):
        return results
    return results


def _eval_value(node: ast.AST, env: dict[str, Any]) -> Any:
    """Evaluate constructor literals / simple arithmetic; ``None`` when unknown."""
    try:
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id in env:
                return env[node.id]
            if node.id in ("True", "False", "None"):
                return {"True": True, "False": False, "None": None}[node.id]
            return None
        if isinstance(node, ast.Attribute):
            chain = _attribute_chain(node)
            if chain and chain[0] == "self":
                return env.get(".".join(chain[1:]))
            return None
        if isinstance(node, (ast.Tuple, ast.List)):
            values = [_eval_value(item, env) for item in node.elts]
            return values
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
            return _eval_comprehension(node, env)
        if isinstance(node, ast.Call):
            chain = _attribute_chain(node.func)
            name = chain[-1] if chain else None
            args = [_eval_value(arg, env) for arg in node.args]
            if name in ("min", "max") and args and all(isinstance(value, int) for value in args):
                return min(args) if name == "min" else max(args)
            if name in ("int", "float", "str", "bool") and len(args) == 1:
                try:
                    return {"int": int, "float": float, "str": str, "bool": bool}[name](args[0])
                except (TypeError, ValueError):
                    return None
            if name in ("tuple", "list") and len(args) == 1 and isinstance(args[0], (list, tuple)):
                return tuple(args[0]) if name == "tuple" else list(args[0])
            if name == "len" and len(args) == 1 and hasattr(args[0], "__len__"):
                return len(args[0])
            if name == "range":
                ints = [value for value in args if isinstance(value, int) and not isinstance(value, bool)]
                if len(ints) == len(args) and ints:
                    try:
                        return list(range(*ints))[:256]
                    except ValueError:
                        return None
            return None
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            value = _eval_value(node.operand, env)
            return -value if isinstance(value, (int, float)) else None
        if isinstance(node, ast.BinOp):
            left = _eval_value(node.left, env)
            right = _eval_value(node.right, env)
            if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
                return None
            if isinstance(left, bool) or isinstance(right, bool):
                return None
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right if right else None
            if isinstance(node.op, ast.FloorDiv):
                return left // right if right else None
            if isinstance(node.op, ast.Pow):
                return left**right
            return None
        if isinstance(node, ast.Subscript):
            value = _eval_value(node.value, env)
            index = _eval_value(node.slice, env)
            if isinstance(value, (list, tuple)):
                if isinstance(index, int):
                    return value[index] if -len(value) <= index < len(value) else None
                if isinstance(index, slice):
                    return list(value[index])
            return None
        return None
    except (_StaticError, RecursionError):
        return None


def _compare_values(operator: ast.cmpop, left: Any, right: Any) -> bool | None:
    """Compare scalars, ``None`` and (nested) list/tuple values statically."""
    if isinstance(operator, (ast.Is, ast.IsNot)):
        same = left is right
        return same if isinstance(operator, ast.Is) else not same
    if isinstance(operator, (ast.Eq, ast.NotEq)):
        def equal(a: Any, b: Any) -> bool:
            if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
                return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
            if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
                return False
            if isinstance(a, bool) or isinstance(b, bool):
                return bool(a) == bool(b)
            if a is None or b is None:
                return a is b
            if isinstance(a, (int, float, str)) and isinstance(b, (int, float, str)):
                return a == b
            return False

        same = equal(left, right)
        return same if isinstance(operator, ast.Eq) else not same
    if isinstance(operator, (ast.Lt, ast.LtE, ast.Gt, ast.GtE)):
        if not isinstance(left, (int, float, str)) or not isinstance(right, (int, float, str)):
            return None
        if isinstance(operator, ast.Lt):
            return left < right
        if isinstance(operator, ast.LtE):
            return left <= right
        if isinstance(operator, ast.Gt):
            return left > right
        return left >= right
    return None


def _kwargs(call: ast.Call, env: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for keyword in call.keywords:
        if keyword.arg is None:
            continue
        result[keyword.arg] = _eval_value(keyword.value, env)
    return result


def _positional(call: ast.Call, env: dict[str, Any]) -> list[Any]:
    values: list[Any] = []
    for arg in call.args:
        if isinstance(arg, ast.Starred):
            value = _eval_value(arg.value, env)
            if isinstance(value, (list, tuple)):
                values.extend(value)
            else:
                values.append(None)
        else:
            values.append(_eval_value(arg, env))
    return values


def _as_int(value: Any, default: int) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, (list, tuple)) and value:
        return _as_int(value[0], default)
    return default


def _as_int_list(value: Any, dims: int) -> list[int]:
    if isinstance(value, int) and not isinstance(value, bool):
        return [value] * dims
    if isinstance(value, (list, tuple)):
        values = [int(item) for item in value if isinstance(item, int)]
        if len(values) == 1:
            return values * dims
        return values
    return [1] * dims


def _conv_dims(op_name: str) -> int:
    for dims, suffix in ((1, "1d"), (2, "2d"), (3, "3d")):
        if op_name.lower().endswith(suffix):
            return dims
    return 2


def _static_params(op_name: str, call: ast.Call, env: dict[str, Any]) -> dict[str, Any]:
    """Constructor parameters for the leaf/container ops we understand."""
    args = _positional(call, env)
    kwargs = _kwargs(call, env)
    name = op_name

    def pick(index: int, key: str, default: Any = None) -> Any:
        if key in kwargs and kwargs[key] is not None:
            return kwargs[key]
        return args[index] if index < len(args) else default

    kind = ops.classify_module(name)
    params: dict[str, Any] = {}
    if kind in ("conv", "deconv"):
        dims = _conv_dims(name)
        params["inChannels"] = _as_int(pick(0, "in_channels", 0), 0)
        params["outChannels"] = _as_int(pick(1, "out_channels", 0), 0)
        params["kernelSize"] = _as_int_list(pick(2, "kernel_size", 1), dims)
        params["stride"] = _as_int_list(pick(3, "stride", 1), dims)
        params["padding"] = _as_int_list(pick(4, "padding", 0), dims)
        params["dilation"] = _as_int_list(pick(5, "dilation", 1), dims)
        params["groups"] = _as_int(pick(6, "groups", 1), 1) or 1
        bias = pick(7, "bias", True)
        params["bias"] = bool(bias)
        if kind == "deconv":
            params["outputPadding"] = _as_int_list(kwargs.get("output_padding", 0), dims)
    elif kind == "linear":
        params["inFeatures"] = _as_int(pick(0, "in_features", 0), 0)
        params["outFeatures"] = _as_int(pick(1, "out_features", 0), 0)
        params["bias"] = bool(pick(2, "bias", True))
    elif kind == "norm":
        normalized = pick(0, "normalized_shape", None)
        if name.startswith(("BatchNorm", "InstanceNorm")):
            normalized = pick(0, "num_features", normalized)
        if isinstance(normalized, (list, tuple)):
            params["normalizedShape"] = [int(v) for v in normalized if isinstance(v, int)]
        elif isinstance(normalized, int):
            params["normalizedShape"] = [normalized]
        params["eps"] = kwargs.get("eps", 1e-5)
        affine = kwargs.get("affine", True)
        params["affine"] = bool(affine) if affine is not None else True
    elif kind == "pool":
        params["type"] = "max" if "max" in name.lower() else ("adaptive" if "adaptive" in name.lower() else "avg")
        output_size = kwargs.get("output_size")
        if name.lower().startswith("adaptive") and output_size is None and args:
            output_size = args[0]
        if output_size is not None:
            dims = _conv_dims(name) if name[-1:].isdigit() else 2
            params["outputSize"] = _as_int_list(output_size, dims)
        else:
            dims = _conv_dims(name) if name[-1:].isdigit() else 2
            params["kernelSize"] = _as_int_list(kwargs.get("kernel_size", args[0] if args else 1), dims)
            params["stride"] = _as_int_list(kwargs.get("stride", kwargs.get("kernel_size", args[0] if args else 1)), dims)
            params["padding"] = _as_int_list(kwargs.get("padding", 0), dims)
    elif kind == "dropout":
        params["p"] = float(kwargs.get("p", args[0] if args else 0.5) or 0.5)
    elif kind == "embedding":
        params["numEmbeddings"] = _as_int(pick(0, "num_embeddings", 0), 0)
        params["embeddingDim"] = _as_int(pick(1, "embedding_dim", 0), 0)
    elif kind == "attention":
        embed = _as_int(pick(0, "embed_dim", 0), 0)
        heads = _as_int(pick(1, "num_heads", 1), 1) or 1
        params.update({"embedDim": embed, "numHeads": heads})
        if embed:
            params["headDim"] = embed // heads
        params["batchFirst"] = bool(pick(5, "batch_first", False))
        params["dropout"] = float(kwargs.get("dropout", 0.0) or 0.0)
        params["kdim"] = _as_int(kwargs.get("kdim", embed), embed)
        params["vdim"] = _as_int(kwargs.get("vdim", embed), embed)
    elif kind == "rnn":
        params["inputFeatures"] = _as_int(pick(0, "input_size", 0), 0)
        params["hiddenFeatures"] = _as_int(pick(1, "hidden_size", 0), 0)
        params["numLayers"] = _as_int(kwargs.get("num_layers", 1), 1) or 1
        params["bidirectional"] = bool(kwargs.get("bidirectional", False))
        params["batchFirst"] = bool(kwargs.get("batch_first", False))
        params["type"] = name
    elif kind == "flatten":
        params["startDim"] = _as_int(pick(0, "start_dim", 1), 1)
        params["endDim"] = _as_int(pick(1, "end_dim", -1), -1)
    elif kind == "upsample":
        scale = kwargs.get("scale_factor", args[0] if args else None)
        size = kwargs.get("size")
        if size is not None:
            params["size"] = _as_int_list(size, 2)
        elif scale is not None:
            params["scaleFactor"] = _as_int_list(scale, 2) if isinstance(scale, (list, tuple)) else float(scale)
        params["mode"] = str(kwargs.get("mode", "nearest"))
    elif kind in ("activation", "softmax"):
        inplace = kwargs.get("inplace")
        if inplace is not None:
            params["inplace"] = bool(inplace)
        dim = kwargs.get("dim")
        if isinstance(dim, int):
            params["dim"] = dim
    return params


def _static_num_params(op_name: str, params: dict[str, Any]) -> int:
    kind = ops.classify_module(op_name)
    if kind in ("conv", "deconv"):
        cin = int(params.get("inChannels") or 0)
        cout = int(params.get("outChannels") or 0)
        groups = max(1, int(params.get("groups") or 1))
        kernel = _product(_as_int_list(params.get("kernelSize", 1), _conv_dims(op_name)))
        if kind == "deconv":
            weight = cin * (cout // groups) * kernel
        else:
            weight = cout * (cin // groups) * kernel
        return int(weight + (cout if params.get("bias") else 0))
    if kind == "linear":
        value = int(params.get("inFeatures") or 0) * int(params.get("outFeatures") or 0)
        return value + (int(params.get("outFeatures") or 0) if params.get("bias") else 0)
    if kind == "norm":
        shape = params.get("normalizedShape") or []
        total = _product(int(v) for v in shape)
        return total * 2 if params.get("affine", True) else 0
    if kind == "embedding":
        return int(params.get("numEmbeddings") or 0) * int(params.get("embeddingDim") or 0)
    if kind == "attention":
        embed = int(params.get("embedDim") or 0)
        kdim = int(params.get("kdim") or embed)
        vdim = int(params.get("vdim") or embed)
        return 3 * embed * embed + 3 * embed + embed * vdim + embed
    if kind == "rnn":
        input_features = int(params.get("inputFeatures") or 0)
        hidden = int(params.get("hiddenFeatures") or 0)
        layers = max(1, int(params.get("numLayers") or 1))
        directions = 2 if params.get("bidirectional") else 1
        per_layer = 4 * (input_features * hidden + hidden * hidden) + 8 * hidden
        return int(per_layer * layers * directions)
    return 0


# ---------------------------------------------------------------------------
# builder: __init__ -> Decl tree
# ---------------------------------------------------------------------------

class _Builder:
    def __init__(self, module_file: ModuleFile, source_index: SourceIndex) -> None:
        self.info = module_file
        self.index = source_index
        self.source = module_file.source
        self.warnings: list[str] = []
        self._depth = 0
        self.module_functions: dict[str, ast.FunctionDef] = {
            node.name: node
            for node in module_file.tree.body
            if isinstance(node, ast.FunctionDef) and not node.name.startswith("__")
        }

    # -- public ----------------------------------------------------------
    def build_model(self, candidate: ModuleClass) -> Decl:
        root = Decl(
            path="",
            op=candidate.name,
            kind=ops.classify_module(candidate.name),
            class_info=candidate,
            source=self.index.class_sources.get(candidate.name),
        )
        self._populate(root, candidate, {})
        root.num_params = root.total_params()
        return root

    def _populate(self, decl: Decl, class_info: ModuleClass, bind: dict[str, Any]) -> None:
        if class_info.init_node is None:
            return
        if self._depth >= MAX_DEPTH:
            self.warnings.append(
                f"{decl.path or class_info.name}: module nesting deeper than {MAX_DEPTH}; internals skipped"
            )
            return
        self._depth += 1
        try:
            env: dict[str, Any] = self._init_defaults(class_info)
            env.update(bind)
            self._walk_init(decl, class_info, class_info.init_node, env)
            decl.attrs = {
                key: value
                for key, value in env.items()
                if not isinstance(value, Decl)
                and not callable(value)
            }
        finally:
            self._depth -= 1

    def _init_defaults(self, class_info: ModuleClass) -> dict[str, Any]:
        env: dict[str, Any] = {}
        init = class_info.init_node
        if init is None:
            return env
        positional = list(getattr(init.args, "posonlyargs", [])) + list(init.args.args)
        defaults = list(init.args.defaults)
        offset = len(positional) - len(defaults)
        for index, arg in enumerate(positional):
            if arg.arg in ("self", "cls"):
                continue
            if index >= offset:
                env[arg.arg] = _eval_value(defaults[index - offset], env)
        for arg, default in zip(init.args.kwonlyargs, init.args.kw_defaults):
            if default is not None:
                env[arg.arg] = _eval_value(default, env)
        return env

    def _walk_init(self, decl: Decl, class_info: ModuleClass, node: ast.AST, env: dict[str, Any]) -> None:
        if isinstance(node, ast.FunctionDef):
            for statement in node.body:
                self._init_statement(decl, class_info, statement, env)
            return
        for statement in getattr(node, "body", []):
            self._init_statement(decl, class_info, statement, env)

    def _init_statement(self, decl: Decl, class_info: ModuleClass, statement: ast.stmt, env: dict[str, Any]) -> None:
        if isinstance(statement, ast.Assign):
            for target in statement.targets:
                self._init_assignment(decl, class_info, target, statement.value, env)
        elif isinstance(statement, ast.AnnAssign) and statement.value is not None:
            self._init_assignment(decl, class_info, statement.target, statement.value, env)
        elif isinstance(statement, ast.For):
            self._init_for(decl, class_info, statement, env)
        elif isinstance(statement, ast.If):
            condition = _eval_value(statement.test, env)
            body = statement.body if condition is not False else statement.orelse
            for child in body:
                self._init_statement(decl, class_info, child, env)
        elif isinstance(statement, ast.FunctionDef):
            env[statement.name] = statement
        elif isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
            self._init_expr_call(decl, class_info, statement.value, env)

    def _init_assignment(
        self, decl: Decl, class_info: ModuleClass, target: ast.AST, value: ast.AST, env: dict[str, Any]
    ) -> None:
        if isinstance(target, (ast.Tuple, ast.List)):
            values = value.elts if isinstance(value, (ast.Tuple, ast.List)) else None
            if values is not None and len(values) == len(target.elts):
                for element, expression in zip(target.elts, values):
                    self._init_assignment(decl, class_info, element, expression, env)
            return
        chain = _attribute_chain(target)
        if chain and chain[0] == "self" and len(chain) > 1:
            attribute_path = ".".join(chain[1:])
            built = self._decl_from_expr(value, f"{decl.path}.{attribute_path}" if decl.path else attribute_path, env, class_info)
            if built is not None:
                parent, name = self._container_parent(decl, attribute_path)
                parent.add(name, built)
                return
            evaluated = _eval_value(value, env)
            if evaluated is not None:
                env[attribute_path] = evaluated
            return
        if isinstance(target, ast.Name):
            if isinstance(value, ast.Call):
                built = self._decl_from_expr(value, f"{decl.path}.{target.id}" if decl.path else target.id, env, class_info)
                if built is not None:
                    env[target.id] = built
                    return
            if isinstance(value, (ast.List, ast.Tuple)):
                env[target.id] = [value]
                return
            evaluated = _eval_value(value, env)
            if evaluated is not None:
                env[target.id] = evaluated

    def _container_parent(self, decl: Decl, attribute_path: str) -> tuple[Decl, str]:
        parts = attribute_path.split(".")
        parent = decl
        for part in parts[:-1]:
            child = parent.children.get(part)
            if child is None:
                break
            parent = child
        return parent, parts[-1]

    def _decl_from_expr(
        self, expression: ast.AST, path: str, env: dict[str, Any], class_info: ModuleClass
    ) -> Decl | None:
        if isinstance(expression, ast.Call):
            return self._decl_from_call(expression, path, env, class_info)
        if isinstance(expression, ast.Name):
            value = env.get(expression.id)
            if isinstance(value, Decl):
                return self._clone_decl(value, path)
            if isinstance(value, list):
                return None
        if isinstance(expression, (ast.List, ast.Tuple)):
            return None
        return None

    def _clone_decl(self, decl: Decl, path: str) -> Decl:
        clone = Decl(
            path=path,
            op=decl.op,
            kind=decl.kind,
            params=dict(decl.params),
            is_container=decl.is_container,
            is_sequential=decl.is_sequential,
            class_info=decl.class_info,
            bind=dict(decl.bind),
            source=decl.source,
            num_params=decl.num_params,
            synthetic=decl.synthetic,
            is_transformer_layer=decl.is_transformer_layer,
            tensor_shape=list(decl.tensor_shape) if decl.tensor_shape else None,
        )
        for name in decl.order:
            clone.add(name, self._clone_decl(decl.children[name], f"{path}.{name}"))
        return clone

    def _decl_from_call(
        self, call: ast.Call, path: str, env: dict[str, Any], class_info: ModuleClass
    ) -> Decl | None:
        op_name = _class_last(call.func)
        if not op_name:
            return None
        helper = self.module_functions.get(op_name)
        if helper is None:
            local_helper = env.get(op_name)
            if isinstance(local_helper, ast.FunctionDef):
                helper = local_helper
        if helper is not None:
            return self._decl_from_module_function(helper, call, path, env, class_info)
        source = _source_loc(self.source, call)
        if op_name in CONTAINER_OPS:
            decl = Decl(
                path=path,
                op=op_name,
                kind="container",
                is_container=True,
                is_sequential=op_name == "Sequential",
                source=source,
            )
            self._build_container(decl, op_name, call, env, class_info)
            decl.num_params = decl.total_params()
            return decl

        local_class = self.info.classes.get(op_name)
        if op_name in TRANSFORMER_STACK_OPS:
            return self._build_transformer_stack(call, path, env, class_info, source)
        if op_name in TRANSFORMER_LAYER_OPS:
            return self._build_transformer_layer(call, path, env, source)

        if op_name in ("Parameter", "Buffer"):
            tensor_shape = _tensor_creation_shape(call.args[0], env) if call.args else None
            decl = Decl(
                path=path,
                op=op_name,
                kind="custom",
                params={},
                source=source,
                tensor_shape=tensor_shape,
            )
            if tensor_shape is not None:
                decl.num_params = _product(dim for dim in tensor_shape if dim > 0)
            return decl

        params = _static_params(op_name, call, env)
        kind = ops.classify_module(op_name)
        decl = Decl(
            path=path,
            op=op_name,
            kind=kind,
            params=params,
            source=source,
            class_info=local_class if local_class is not None and local_class.is_module else None,
        )
        if decl.class_info is not None and decl.class_info.has_forward:
            bind = self._bind_init(decl.class_info, call, env)
            decl.kind = ops.classify_module(op_name)
            self._populate(decl, decl.class_info, bind)
            decl.num_params = decl.total_params()
        else:
            decl.num_params = _static_num_params(op_name, params)
            if kind == "block" and op_name in FUSED_OPS:
                self.warnings.append(
                    f"{path or op_name}: fused {op_name} kept as a single block node (static analysis)"
                )
        return decl

    def _bind_init(self, class_info: ModuleClass, call: ast.Call, env: dict[str, Any]) -> dict[str, Any]:
        init = class_info.init_node
        if init is None:
            return {}
        positional = list(getattr(init.args, "posonlyargs", [])) + list(init.args.args)
        names = [arg.arg for arg in positional if arg.arg not in ("self", "cls")]
        values = _positional(call, env)
        kwargs = _kwargs(call, env)
        bind: dict[str, Any] = {}
        for index, name in enumerate(names):
            if index < len(values):
                bind[name] = values[index]
            elif name in kwargs:
                bind[name] = kwargs[name]
        bind.update(kwargs)
        return bind

    def _bind_function_call(
        self,
        function: ast.FunctionDef,
        call: ast.Call,
        caller_env: dict[str, Any],
    ) -> dict[str, Any]:
        positional = [arg.arg for arg in function.args.args if arg.arg not in ("self", "cls")]
        values = _positional(call, caller_env)
        keywords = _kwargs(call, caller_env)
        bound: dict[str, Any] = {}
        for index, name in enumerate(positional):
            if index < len(values):
                bound[name] = values[index]
            elif name in keywords:
                bound[name] = keywords[name]
        bound.update(keywords)
        return bound

    def _function_defaults(self, function: ast.FunctionDef) -> dict[str, Any]:
        env: dict[str, Any] = {}
        positional = [arg.arg for arg in function.args.args if arg.arg not in ("self", "cls")]
        defaults = list(function.args.defaults)
        offset = len(positional) - len(defaults)
        for index, default in enumerate(defaults):
            name = positional[offset + index]
            env[name] = _eval_value(default, env)
        for arg, default in zip(function.args.kwonlyargs, function.args.kw_defaults):
            if default is not None:
                env[arg.arg] = _eval_value(default, env)
        return env

    def _decl_from_module_function(
        self,
        function: ast.FunctionDef,
        call: ast.Call,
        path: str,
        caller_env: dict[str, Any],
        class_info: ModuleClass,
    ) -> Decl | None:
        if self._depth > MAX_DEPTH:
            return None
        env = self._function_defaults(function)
        env.update(self._bind_function_call(function, call, caller_env))
        self._depth += 1
        try:
            return self._exec_decl_statements(function.body, path, env, class_info)
        finally:
            self._depth -= 1

    def _exec_decl_statements(
        self,
        statements: Sequence[ast.stmt],
        path: str,
        env: dict[str, Any],
        class_info: ModuleClass,
    ) -> Decl | None:
        for statement in statements:
            if isinstance(statement, ast.Return):
                if statement.value is None:
                    return None
                if isinstance(statement.value, ast.Name):
                    value = env.get(statement.value.id)
                    if isinstance(value, Decl):
                        return self._clone_decl(value, path)
                return self._decl_from_expr(statement.value, path, env, class_info)
            if isinstance(statement, (ast.Assign, ast.AnnAssign)):
                value_node = statement.value
                targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
                for target in targets:
                    if isinstance(target, ast.Name) and isinstance(value_node, ast.Call):
                        built = self._decl_from_expr(value_node, path, env, class_info)
                        if built is not None:
                            env[target.id] = built
                            continue
                    if isinstance(target, ast.Name):
                        evaluated = _eval_value(value_node, env)
                        if evaluated is not None:
                            env[target.id] = evaluated
                continue
            if isinstance(statement, ast.FunctionDef):
                env[statement.name] = statement
                continue
            if isinstance(statement, ast.For) and isinstance(statement.target, ast.Name):
                iterable = _eval_value(statement.iter, env)
                items = iterable if isinstance(iterable, (list, tuple)) else []
                if not items:
                    count = _as_int(iterable, 0)
                    items = list(range(max(0, min(count, 64))))
                for item in items[:64]:
                    env[statement.target.id] = item
                    built = self._exec_decl_statements(statement.body, path, env, class_info)
                    if built is not None:
                        return built
                continue
            if isinstance(statement, ast.If):
                condition = _eval_value(statement.test, env)
                body = statement.body if condition is not False else statement.orelse
                built = self._exec_decl_statements(body, path, env, class_info)
                if built is not None:
                    return built
                continue
        return None

    def _build_container(
        self, decl: Decl, ctor: str, call: ast.Call, env: dict[str, Any], class_info: ModuleClass
    ) -> None:
        items: list[ast.AST] = []
        if ctor == "ModuleDict":
            if call.args and isinstance(call.args[0], ast.Dict):
                for index, (key_node, value_node) in enumerate(zip(call.args[0].keys, call.args[0].values)):
                    key = _eval_value(key_node, env) if key_node is not None else f"item{index}"
                    child = self._decl_from_expr(value_node, f"{decl.path}.{key}", env, class_info)
                    if child is not None:
                        decl.add(str(key), child)
            return
        args = list(call.args)
        if len(args) == 1 and isinstance(args[0], (ast.List, ast.Tuple)):
            items = list(args[0].elts)
        elif len(args) == 1 and isinstance(args[0], ast.ListComp):
            for element, element_env in self._expand_listcomp_env(args[0], env):
                child = self._decl_from_expr(
                    element, f"{decl.path}.{len(decl.order)}", element_env, class_info
                )
                if child is not None:
                    decl.add(str(len(decl.order)), child)
            return
        elif len(args) == 1 and isinstance(args[0], ast.Call) and _class_last(args[0].func) == "OrderedDict":
            if args[0].args and isinstance(args[0].args[0], (ast.List, ast.Tuple)):
                for element in args[0].args[0].elts:
                    if isinstance(element, ast.Tuple) and len(element.elts) == 2:
                        key = _eval_value(element.elts[0], env)
                        child = self._decl_from_expr(element.elts[1], f"{decl.path}.{key}", env, class_info)
                        if child is not None:
                            decl.add(str(key), child)
            return
        else:
            items = args
        for index, item in enumerate(items):
            if isinstance(item, ast.Starred):
                value = env.get(item.value.id) if isinstance(item.value, ast.Name) else None
                if isinstance(value, list):
                    for sub in value:
                        if isinstance(sub, ast.AST):
                            child = self._decl_from_expr(sub, f"{decl.path}.{len(decl.order)}", env, class_info)
                            if child is not None:
                                decl.add(str(len(decl.order)), child)
                continue
            built = self._decl_from_expr(item, f"{decl.path}.{len(decl.order)}", env, class_info)
            if built is not None:
                decl.add(str(len(decl.order)), built)

    def _expand_listcomp_env(
        self, node: ast.ListComp, env: dict[str, Any]
    ) -> list[tuple[ast.AST, dict[str, Any]]]:
        results: list[tuple[ast.AST, dict[str, Any]]] = []

        def run(index: int, scope: dict[str, Any]) -> None:
            if index >= len(node.generators):
                results.append((node.elt, dict(scope)))
                return
            generator = node.generators[index]
            iterable = _eval_value(generator.iter, scope)
            if not isinstance(iterable, (list, tuple)):
                fallback = _as_int(iterable, 0)
                iterable = list(range(max(0, min(fallback, 64))))
            for item in list(iterable)[:64]:
                nested = dict(scope)
                target = generator.target
                if isinstance(target, ast.Name):
                    nested[target.id] = item
                elif isinstance(target, (ast.Tuple, ast.List)) and isinstance(item, (list, tuple)):
                    for sub_target, sub_item in zip(target.elts, item):
                        if isinstance(sub_target, ast.Name):
                            nested[sub_target.id] = sub_item
                run(index + 1, nested)

        run(0, dict(env))
        return results

    def _build_transformer_stack(
        self,
        call: ast.Call,
        path: str,
        env: dict[str, Any],
        class_info: ModuleClass,
        source: SourceLoc | None,
    ) -> Decl:
        decl = Decl(path=path, op="TransformerEncoder", kind="block", source=source, is_container=True)
        if call.args:
            layer_expr = call.args[0]
            layer = self._decl_from_expr(layer_expr, f"{path}.layers.0", env, class_info)
            if layer is None:
                layer = self._build_transformer_layer(layer_expr, f"{path}.layers.0", env, source) if isinstance(layer_expr, ast.Call) else None
        else:
            layer = None
        kwargs = _kwargs(call, env)
        if "num_layers" in kwargs:
            num_layers = _as_int(kwargs["num_layers"], 1)
        elif len(call.args) > 1:
            num_layers = _as_int(_eval_value(call.args[1], env), 1)
        else:
            num_layers = _as_int(env.get("num_layers"), 1)
        num_layers = max(0, min(num_layers, 64))
        if layer is not None:
            for index in range(num_layers):
                clone = self._clone_decl(layer, f"{path}.layers.{index}")
                decl.children[f"layers.{index}"] = clone
                decl.order.append(f"layers.{index}")
        # Optional final norm argument (nn.LayerNorm instance is the 3rd arg).
        if len(call.args) > 2 and isinstance(call.args[2], ast.Call):
            norm = self._decl_from_expr(call.args[2], f"{path}.norm", env, class_info)
            if norm is not None:
                decl.add("norm", norm)
        elif "norm" in call.keywords:
            for keyword in call.keywords:
                if keyword.arg == "norm" and isinstance(keyword.value, ast.Call):
                    norm = self._decl_from_expr(keyword.value, f"{path}.norm", env, class_info)
                    if norm is not None:
                        decl.add("norm", norm)
        decl.is_transformer_stack = True  # type: ignore[attr-defined]
        decl.num_params = decl.total_params()
        return decl

    def _rebase_decl(self, decl: Decl, new_path: str) -> Decl:
        rebuilt = self._clone_decl(decl, new_path)
        return rebuilt

    def _build_transformer_layer(
        self, call: ast.Call, path: str, env: dict[str, Any], source: SourceLoc | None
    ) -> Decl:
        params = _static_params("TransformerEncoderLayer", call, env)
        kwargs = _kwargs(call, env)
        args = _positional(call, env)
        d_model = _as_int(kwargs.get("d_model", args[0] if args else 0), 0)
        nhead = _as_int(kwargs.get("nhead", args[1] if len(args) > 1 else 1), 1) or 1
        dim_ff = _as_int(kwargs.get("dim_feedforward", args[2] if len(args) > 2 else 2048), 2048)
        dropout = float(kwargs.get("dropout", 0.1) or 0.1)
        batch_first = bool(kwargs.get("batch_first", False))
        norm_first = bool(kwargs.get("norm_first", False))
        activation = str(kwargs.get("activation", "relu") or "relu")
        decl = Decl(
            path=path,
            op="TransformerEncoderLayer",
            kind="block",
            params={
                "embedDim": d_model,
                "numHeads": nhead,
                "headDim": d_model // nhead if nhead else 0,
                "hiddenFeatures": dim_ff,
                "dropout": dropout,
                "batchFirst": batch_first,
                "normFirst": norm_first,
            },
            source=source,
            is_transformer_layer=True,
        )
        decl.transformer_args = {  # type: ignore[attr-defined]
            "d_model": d_model,
            "nhead": nhead,
            "dim_feedforward": dim_ff,
            "dropout": dropout,
            "batch_first": batch_first,
            "norm_first": norm_first,
            "activation": activation,
        }
        mha = Decl(
            path=f"{path}.self_attn",
            op="MultiheadAttention",
            kind="attention",
            params={
                "embedDim": d_model,
                "numHeads": nhead,
                "headDim": d_model // nhead if nhead else 0,
                "kdim": d_model,
                "vdim": d_model,
                "batchFirst": batch_first,
                "dropout": dropout,
            },
            synthetic=True,
        )
        mha.num_params = _static_num_params("MultiheadAttention", mha.params)
        decl.add("self_attn", mha)
        for name, in_features, out_features in (
            ("linear1", d_model, dim_ff),
            ("linear2", dim_ff, d_model),
        ):
            linear = Decl(
                path=f"{path}.{name}",
                op="Linear",
                kind="linear",
                params={"inFeatures": in_features, "outFeatures": out_features, "bias": True},
                synthetic=True,
            )
            linear.num_params = _static_num_params("Linear", linear.params)
            decl.add(name, linear)
        for name in ("norm1", "norm2"):
            norm = Decl(
                path=f"{path}.{name}",
                op="LayerNorm",
                kind="norm",
                params={"normalizedShape": [d_model], "eps": 1e-5, "affine": True},
                synthetic=True,
            )
            norm.num_params = _static_num_params("LayerNorm", norm.params)
            decl.add(name, norm)
        for name in ("dropout1", "dropout2", "dropout"):
            drop = Decl(
                path=f"{path}.{name}",
                op="Dropout",
                kind="dropout",
                params={"p": dropout},
                synthetic=True,
            )
            decl.add(name, drop)
        act_op = "GELU" if "gelu" in activation.lower() else "ReLU"
        act = Decl(path=f"{path}.activation", op=act_op, kind="activation", params={}, synthetic=True)
        decl.add("activation", act)
        decl.num_params = decl.total_params()
        return decl

    def _init_for(self, decl: Decl, class_info: ModuleClass, statement: ast.For, env: dict[str, Any]) -> None:
        iterable = statement.iter
        if isinstance(iterable, ast.Call) and _class_last(iterable.func) == "range":
            values = _positional(iterable, env)
            count = _as_int(values[0] if values else 0, 0)
        else:
            count = 0
        count = max(0, min(count, 64))
        if count and isinstance(statement.target, ast.Name):
            for index in range(count):
                env[statement.target.id] = index
                for child in statement.body:
                    self._init_statement(decl, class_info, child, env)
            return
        for child in statement.body:
            self._init_statement(decl, class_info, child, env)

    def _init_expr_call(self, decl: Decl, class_info: ModuleClass, call: ast.Call, env: dict[str, Any]) -> None:
        """Handle ``self.blocks.append(Block(...))`` inside ``__init__`` loops."""
        func = call.func
        if not isinstance(func, ast.Attribute) or func.attr != "append":
            return
        chain = _attribute_chain(func.value)
        if not chain or chain[0] != "self" or len(chain) < 2:
            return
        attribute_path = ".".join(chain[1:])
        parent, name = self._container_parent(decl, attribute_path)
        target = parent.children.get(name)
        if target is None or not target.is_container or not call.args:
            return
        child = self._decl_from_expr(call.args[0], f"{target.path}.{len(target.order)}", env, class_info)
        if child is not None:
            target.add(str(len(target.order)), child)
            target.num_params = target.total_params()


# ---------------------------------------------------------------------------
# symbolic forward execution
# ---------------------------------------------------------------------------

@dataclass
class _Tensor:
    shape: Shape
    producers: list[int] = field(default_factory=list)

    def copy(self, producers: list[int] | None = None) -> "_Tensor":
        return _Tensor(list(self.shape), list(self.producers) if producers is None else list(producers))


@dataclass
class _Multi:
    items: list[Any]


@dataclass
class _ReturnSignal:
    value: Any


@dataclass
class _ModuleRef:
    decl: Decl


@dataclass
class _FuncRef:
    name: str


@dataclass
class _MethodRef:
    name: str
    base: Any = None


@dataclass
class _ParamRef:
    name: str


def _first_tensor(value: Any) -> _Tensor | None:
    if isinstance(value, _Tensor):
        return value
    if isinstance(value, _Multi):
        for item in value.items:
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
        return None
    if isinstance(value, (list, tuple)):
        for item in value:
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    return None


def _tensor_inputs(args: Sequence[Any], kwargs: dict[str, Any]) -> list[_Tensor]:
    tensors: list[_Tensor] = []

    def collect(value: Any) -> None:
        if isinstance(value, _Tensor):
            tensors.append(value)
        elif isinstance(value, _Multi):
            for item in value.items:
                collect(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                collect(item)
        elif isinstance(value, dict):
            for item in value.values():
                collect(item)

    for arg in args:
        collect(arg)
    for value in kwargs.values():
        collect(value)
    return tensors


def _broadcast_shape(left: Shape, right: Shape) -> Shape:
    if not left:
        return list(right)
    if not right:
        return list(left)
    if left == right:
        return list(left)
    if len(left) == len(right):
        out: Shape = []
        for a, b in zip(left, right):
            if a == b:
                out.append(a)
            elif a == 1:
                out.append(b)
            elif b == 1:
                out.append(a)
            elif a < 0:
                out.append(b)
            elif b < 0:
                out.append(a)
            else:
                out.append(max(a, b))
        return out
    longer, shorter = (left, right) if len(left) > len(right) else (right, left)
    if longer[-len(shorter) :] == shorter:
        return list(longer)
    out = [1] * (len(longer) - len(shorter)) + list(shorter)
    return _broadcast_shape(list(longer), out)


def _reduce_shape(shape: Shape, dim: int | Sequence[int] | None, keepdim: bool = False) -> Shape:
    if dim is None:
        return [1] if keepdim else [1]
    dims = [dim] if isinstance(dim, int) else list(dim)
    rank = len(shape)
    resolved = sorted({value + rank if value < 0 else value for value in dims})
    if keepdim:
        return [1 if index in resolved else value for index, value in enumerate(shape)]
    reduced = [value for index, value in enumerate(shape) if index not in resolved]
    return reduced or [1]


def _matmul_shape(left: Shape, right: Shape) -> Shape | None:
    if len(left) < 2 or len(right) < 2:
        return None
    if left[-1] != right[-2]:
        return None
    batch = _broadcast_shape(left[:-2], right[:-2])
    return [*batch, left[-2], right[-1]]


def _flatten_shape(shape: Shape, start: int = 1, end: int = -1) -> Shape:
    rank = len(shape)
    if start < 0:
        start += rank
    if end < 0:
        end += rank
    start = max(0, min(start, rank - 1))
    end = max(start, min(end, rank - 1))
    flat = 1
    dynamic = False
    for value in shape[start : end + 1]:
        if value < 0:
            dynamic = True
            flat = -1
            break
        flat *= value
    return [*shape[:start], flat, *shape[end + 1 :]]


def _normalize_dim(dim: int, rank: int) -> int:
    return dim + rank if dim < 0 else dim


def _broadcast_many(shapes: Sequence[Shape]) -> Shape | None:
    """Broadcast several shapes; ``None`` when they are incompatible."""
    if not shapes:
        return None
    result = list(shapes[0])
    for shape in shapes[1:]:
        try:
            result = _broadcast_shape(result, list(shape))
        except Exception:  # pragma: no cover - defensive
            return None
    return result


def _expand_shape(shape: Shape, dims: Sequence[Any]) -> Shape:
    """PyTorch ``Tensor.expand`` shape rules (``-1`` keeps the input size)."""
    rank = len(shape)
    target_rank = max(rank, len(dims))
    offset = target_rank - rank
    out: Shape = []
    for index in range(target_rank):
        source = shape[index - offset] if index - offset >= 0 else 1
        value = dims[index] if index < len(dims) else source
        if isinstance(value, int) and value >= 0:
            if value == 1 and source not in (1, -1):
                out.append(source)
            else:
                out.append(value)
        else:
            out.append(source if source >= 0 else -1)
    return out


def _repeat_shape(shape: Shape, repeats: Sequence[int]) -> Shape:
    rank = len(shape)
    if not repeats:
        return list(shape)
    target_rank = max(rank, len(repeats))
    padded = [1] * (target_rank - rank) + list(shape)
    out: Shape = []
    for index in range(target_rank):
        repeat = repeats[index] if index < len(repeats) else 1
        source = padded[index]
        out.append(source * repeat if source >= 0 and repeat >= 0 else -1)
    return out


def _topk_shape(shape: Shape, dim: Any, k: Any) -> Shape:
    out = list(shape)
    if isinstance(dim, int) and shape:
        resolved = _normalize_dim(dim, len(shape))
        if 0 <= resolved < len(out):
            out[resolved] = int(k) if isinstance(k, int) and k >= 0 else -1
    return out


def _index_select_shape(shape: Shape, dim: Any, index: _Tensor | None) -> Shape:
    out = list(shape)
    if index is None or not index.shape:
        if isinstance(dim, int) and shape:
            resolved = _normalize_dim(dim, len(shape))
            if 0 <= resolved < len(out):
                out[resolved] = -1
        return out
    if isinstance(dim, int) and shape:
        resolved = _normalize_dim(dim, len(shape))
        if 0 <= resolved < len(out):
            out[resolved] = index.shape[0] if index.shape else -1
    return out


def _unknown_shape_for(in_shapes: Sequence[Shape]) -> Shape:
    rank = max((len(shape) for shape in in_shapes if shape), default=0)
    if rank <= 0:
        return [-1]
    return [-1] * rank


#: Functions whose output keeps the input shape (element-wise unary ops).
_UNARY_SHAPE_PRESERVING = frozenset(
    {
        "abs",
        "acos",
        "asin",
        "atan",
        "ceil",
        "clamp",
        "clamp_min",
        "clamp_max",
        "cos",
        "cosh",
        "erf",
        "erfc",
        "exp",
        "exp2",
        "expm1",
        "floor",
        "frac",
        "isinf",
        "isnan",
        "log",
        "log10",
        "log1p",
        "log2",
        "logical_not",
        "neg",
        "negative",
        "reciprocal",
        "round",
        "rsqrt",
        "sigmoid",
        "sign",
        "sin",
        "sinh",
        "sqrt",
        "square",
        "tan",
        "tanh",
        "trunc",
        "zero_",
        "abs_",
        "ceil_",
        "floor_",
        "clamp_",
        "nan_to_num",
        "nanmean_",
    }
)

#: Functions that combine several tensors with broadcasting.
_BINARY_FUNCS = frozenset(
    {
        "add",
        "sub",
        "subtract",
        "mul",
        "multiply",
        "div",
        "divide",
        "true_divide",
        "floor_divide",
        "remainder",
        "fmod",
        "pow",
        "maximum",
        "minimum",
        "atan2",
        "logaddexp",
        "logaddexp2",
        "copysign",
        "nextafter",
        "hypot",
        "xlogy",
        "outer",
        "add_",
        "sub_",
        "mul_",
        "div_",
    }
)

_REDUCE_FUNCTIONS = frozenset({"mean", "sum", "amax", "amin", "prod", "norm", "nansum", "nanmean", "median"})


def _view_dims_from_args(args: Sequence[Any], kwargs: dict[str, Any]) -> list[int]:
    """Flatten view/reshape arguments (ints, tuples/lists, unknown -> -1)."""
    dims: list[int] = []
    values: list[Any] = []
    if "shape" in kwargs:
        values.append(kwargs["shape"])
    values.extend(args)
    for value in values:
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            dims.append(value)
        elif isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, bool):
                    continue
                dims.append(int(item) if isinstance(item, int) else -1)
        elif value is None:
            dims.append(-1)
    return dims


def _first_consuming_decl_node(root: Decl) -> Decl | None:
    """First atomic conv/linear/embedding/attention declaration in a module tree."""
    for decl in root.walk():
        if decl.kind in ("conv", "deconv", "linear", "embedding", "attention", "rnn") and not decl.children:
            return decl
    return None


def _fft_shape(shape: Shape, op: str, s: Any, dim: Any) -> Shape:
    out = list(shape)
    rank = len(shape)
    if op in ("fftshift", "ifftshift"):
        return out
    if op in ("fft", "ifft"):
        return out
    if op in ("rfft", "irfft"):
        dims = [dim] if isinstance(dim, int) else [rank - 1]
    else:
        if isinstance(dim, (list, tuple)) and dim:
            dims = [int(value) for value in dim]
        else:
            dims = [rank - 2, rank - 1] if rank >= 2 else [rank - 1]
    dims = [_normalize_dim(value, rank) for value in dims]
    sizes = list(s) if isinstance(s, (list, tuple)) else None
    for position, axis in enumerate(dims):
        if not (0 <= axis < rank):
            continue
        target = int(sizes[position]) if sizes is not None and position < len(sizes) and isinstance(sizes[position], int) else None
        if op in ("rfftn", "rfft", "rfft2"):
            length = target if target is not None else out[axis]
            out[axis] = length // 2 + 1 if isinstance(length, int) and length >= 0 else -1
        elif op in ("irfftn", "irfft", "irfft2"):
            if target is not None:
                out[axis] = target
            else:
                length = out[axis]
                out[axis] = (length - 1) * 2 if isinstance(length, int) and length > 0 else -1
    return out


def _einsum_shape(equation: str, shapes: Sequence[Shape]) -> Shape:
    if "->" not in equation:
        return _unknown_shape_for(shapes)
    left, right = equation.split("->", 1)
    terms = [term.strip() for term in left.split(",") if term.strip()]
    output = right.strip()
    if not output or "..." in left or "..." in right:
        return _unknown_shape_for(shapes)
    dims: dict[str, int] = {}
    for index, term in enumerate(terms):
        shape = list(shapes[index]) if index < len(shapes) else []
        if len(term) != len(shape):
            continue
        for axis, letter in enumerate(term):
            if letter not in dims:
                dims[letter] = int(shape[axis])
    result: Shape = []
    for letter in output:
        result.append(dims.get(letter, -1))
    return result or [-1]


def _try_broadcast(shapes: Sequence[Shape]) -> Shape | None:
    """Broadcast shapes when compatible; ``None`` when they clearly are not."""
    if not shapes:
        return None
    result = list(shapes[0])
    for shape in shapes[1:]:
        other = list(shape)
        if len(other) > len(result):
            result = [1] * (len(other) - len(result)) + result
        else:
            other = [1] * (len(result) - len(other)) + other
        out: Shape = []
        for left, right in zip(result, other):
            if left == right:
                out.append(left)
            elif left == 1:
                out.append(right)
            elif right == 1:
                out.append(left)
            elif left < 0 or right < 0:
                out.append(-1)
            else:
                return None
        result = out
    return result or [-1]


def _shape_expr_param(node: ast.AST) -> tuple[str, str] | None:
    """Recognize ``param.shape`` / ``param.size()``; returns (param, kind)."""
    if isinstance(node, ast.Attribute) and node.attr in ("shape", "size"):
        if isinstance(node.value, ast.Name):
            return node.value.id, node.attr
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "size":
        if isinstance(node.func.value, ast.Name) and not node.args:
            return node.func.value.id, "size"
    return None


def _forward_input_hints(
    forward: ast.FunctionDef | None,
    root: Decl,
) -> dict[str, dict[str, Any]]:
    """Infer rank / last-dim / dtype hints for forward parameters from the AST."""
    hints: dict[str, dict[str, Any]] = {}
    if forward is None:
        return hints

    def hint(name: str) -> dict[str, Any]:
        return hints.setdefault(name, {"rank": None, "index_rank": None, "last": None, "dtype": "float32"})

    def scan(node: ast.AST) -> None:
        if isinstance(node, ast.Assign):
            shape_info = _shape_expr_param(node.value)
            if shape_info is not None:
                name, _kind = shape_info
                for target in node.targets:
                    if isinstance(target, (ast.Tuple, ast.List)):
                        info = hint(name)
                        rank = len(target.elts)
                        info["rank"] = max(info["rank"] or 0, rank)
        if isinstance(node, ast.Subscript):
            shape_info = _shape_expr_param(node.value)
            if shape_info is not None:
                name, _kind = shape_info
                index = node.slice
                if isinstance(index, ast.Constant) and isinstance(index.value, int) and index.value >= 0:
                    info = hint(name)
                    info["index_rank"] = max(info["index_rank"] or 0, index.value + 1)
        if isinstance(node, ast.Call):
            func_chain = _attribute_chain(node.func)
            if func_chain and func_chain[0] == "self" and len(func_chain) >= 2:
                module_decl = _resolve_decl_chain(root, func_chain[1:])
                first_arg = node.args[0] if node.args else None
                if module_decl is not None and isinstance(first_arg, ast.Name):
                    target = _first_consuming_decl_node(module_decl)
                    if target is not None:
                        info = hint(first_arg.id)
                        if target.kind == "linear" and target.params.get("inFeatures"):
                            info["last"] = int(target.params["inFeatures"])
                        elif target.kind == "embedding":
                            info["dtype"] = "int64"
                            info["last"] = int(target.params.get("embeddingDim") or 0)
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                continue
            scan(child)

    for statement in forward.body:
        scan(statement)
    return hints


def _resolve_decl_chain(root: Decl, parts: Sequence[str]) -> Decl | None:
    node = root
    for part in parts:
        child = node.children.get(part)
        if child is None:
            return None
        node = child
    return node


class _Runner:
    def __init__(
        self,
        root: Decl,
        builder: _Builder,
        options: Any,
        warnings: list[str],
    ) -> None:
        self.root = root
        self.builder = builder
        self.source = builder.source
        self.options = options
        self.warnings = warnings
        self.nodes: list[NodeSpec] = []
        self.edges: list[EdgeSpec] = []
        self._name_counts: dict[str, int] = {}
        self._scope_stack: list[str] = []
        self._call_stack: list[str] = []
        self._steps = 0
        self.attention_paths: set[str] = set()
        for decl in root.walk():
            if decl.kind == "attention" or "attention" in decl.op.lower():
                self.attention_paths.add(decl.path)
        self._func_cache: dict[str, Any] = {}
        self.module_functions: dict[str, ast.FunctionDef] = {
            node.name: node
            for node in builder.info.tree.body
            if isinstance(node, ast.FunctionDef) and not node.name.startswith("__")
        }
        self._helper_stack: list[str] = []

    # -- helpers ---------------------------------------------------------
    @property
    def scope(self) -> str:
        return self._scope_stack[-1] if self._scope_stack else ""

    def _allocate(self, base: str) -> str:
        count = self._name_counts.get(base, 0) + 1
        self._name_counts[base] = count
        return base if count == 1 else f"{base}#{count}"

    def _step(self) -> None:
        self._steps += 1
        if self._steps > 20000:
            raise _StaticError("symbolic execution step budget exhausted")

    def run(self, input_specs: list[dict[str, Any]]) -> tuple[list[NodeSpec], list[_Tensor]]:
        input_tensors: list[_Tensor] = []
        for index, spec in enumerate(input_specs):
            shape = [int(value) for value in (spec.get("shape") or [-1])]
            if not shape:
                shape = [-1]
            name = str(spec.get("name") or f"input{index}")
            self.nodes.append(
                NodeSpec(
                    name=name,
                    op="Input",
                    kind="input",
                    module_path="",
                    in_shapes=[],
                    out_shapes=[shape],
                    tags=["boundary"],
                    depth=0,
                )
            )
            input_tensors.append(_Tensor(shape, [index]))
        try:
            result = self._run_forward_decl(self.root, input_tensors)
        except _StaticError as exc:
            self.warnings.append(f"static forward inference stopped: {exc}")
            result = [_Tensor([-1], [0])]
        outputs = [_first_tensor(item) for item in result if _first_tensor(item) is not None]
        if not outputs:
            last_index = max(0, len(self.nodes) - 1)
            outputs = [_Tensor([-1], [last_index])]
        producers = sorted({producer for tensor in outputs for producer in tensor.producers})
        output_index = len(self.nodes)
        # Keep the main chain connected: any node whose symbolic dependency could
        # not be recovered gets a fallback edge from the previous model node.
        incoming = {edge.target for edge in self.edges}
        for index, node in enumerate(self.nodes):
            if node.kind in ("input", "output") or index in incoming:
                continue
            previous = next(
                (candidate for candidate in range(index - 1, -1, -1) if self.nodes[candidate].kind not in ("input", "output")),
                None,
            )
            if previous is None:
                continue
            self.edges.append(
                EdgeSpec(
                    source=previous,
                    target=index,
                    kind="data",
                    shapes=[list(self.nodes[previous].out_shapes[0])] if self.nodes[previous].out_shapes else [],
                )
            )
        for param_decl in self.root.walk():
            if param_decl.op not in ("Parameter", "Buffer") or not param_decl.tensor_shape:
                continue
            self.nodes.append(
                NodeSpec(
                    name=param_decl.path or param_decl.op,
                    op=param_decl.op,
                    kind="block",
                    module_path=param_decl.path.rsplit(".", 1)[0] if "." in param_decl.path else "",
                    in_shapes=[],
                    out_shapes=[list(param_decl.tensor_shape)],
                    params={},
                    num_params=param_decl.total_params(),
                    tags=["parameter"],
                    notes=["parameter tensor (static shape)"],
                    depth=1,
                )
            )
        self.nodes.append(
            NodeSpec(
                name="output",
                op="Output",
                kind="output",
                module_path="",
                in_shapes=[list(tensor.shape) for tensor in outputs],
                out_shapes=[list(tensor.shape) for tensor in outputs],
                tags=["boundary"],
                depth=0,
            )
        )
        for source in producers:
            if 0 <= source < output_index:
                self.edges.append(
                    EdgeSpec(
                        source=source,
                        target=output_index,
                        kind="data",
                        shapes=[list(self.nodes[source].out_shapes[0])] if self.nodes[source].out_shapes else [],
                    )
                )
        return self.nodes, outputs

    # -- module execution -------------------------------------------------
    def _run_forward_decl(self, decl: Decl, args: list[Any]) -> list[_Tensor]:
        self._call_stack.append(decl.path)
        try:
            if decl.is_transformer_layer:
                return self._run_transformer_layer(decl, args)
            if getattr(decl, "is_transformer_stack", False):
                return self._run_transformer_stack(decl, args)
            if decl.class_info is not None and decl.class_info.has_forward and not decl.is_container:
                return self._run_custom_forward(decl, decl.class_info, args)
            if decl.is_container:
                value: Any = args[0] if args else None
                for name in decl.order:
                    value = self._call_module(decl.children[name], [value])
                tensor = _first_tensor(value)
                return [tensor] if tensor is not None else []
            return self._emit_module(decl, args)
        finally:
            self._call_stack.pop()

    def _run_transformer_stack(self, decl: Decl, args: list[Any]) -> list[_Tensor]:
        value: Any = args[0] if args else _Tensor([-1])
        layer_names = [name for name in decl.order if name.startswith("layers.")]
        for name in layer_names:
            value = self._call_module(decl.children[name], [value])
        if "norm" in decl.children:
            value = self._call_module(decl.children["norm"], [value])
        tensor = _first_tensor(value)
        return [tensor] if tensor is not None else []

    def _run_transformer_layer(self, decl: Decl, args: list[Any]) -> list[_Tensor]:
        params = decl.params
        batch_first = bool(params.get("batchFirst", False))
        norm_first = bool(params.get("normFirst", False))
        x = _first_tensor(args[0]) if args else None
        if x is None:
            x = _Tensor([-1])
        batch = _first_tensor(args[1]) if len(args) > 1 else None
        current = x.copy()
        if norm_first:
            pre = self._call_module(decl.children["norm1"], [current])
            sa = self._call_module(decl.children["self_attn"], [pre, pre, pre])
            sa_tensor = _first_tensor(sa) or current.copy()
            drop1 = self._call_module(decl.children["dropout1"], [sa_tensor])
            current = self._emit_add(current, _first_tensor(drop1) or sa_tensor)
            ff = self._run_ffn(decl, current)
            current = self._emit_add(current, ff)
            current = _first_tensor(self._call_module(decl.children["norm2"], [current])) or current
        else:
            sa = self._call_module(decl.children["self_attn"], [current, current, current])
            sa_tensor = _first_tensor(sa) or current.copy()
            drop1 = self._call_module(decl.children["dropout1"], [sa_tensor])
            merged = self._emit_add(current, _first_tensor(drop1) or sa_tensor)
            current = _first_tensor(self._call_module(decl.children["norm1"], [merged])) or merged
            ff = self._run_ffn(decl, current)
            merged2 = self._emit_add(current, ff)
            current = _first_tensor(self._call_module(decl.children["norm2"], [merged2])) or merged2
        _ = batch
        return [current]

    def _run_ffn(self, decl: Decl, value: _Tensor) -> _Tensor:
        linear1 = _first_tensor(self._call_module(decl.children["linear1"], [value])) or value
        activation = _first_tensor(self._call_module(decl.children["activation"], [linear1])) or linear1
        drop = _first_tensor(self._call_module(decl.children["dropout"], [activation])) or activation
        linear2 = _first_tensor(self._call_module(decl.children["linear2"], [drop])) or drop
        drop2 = _first_tensor(self._call_module(decl.children["dropout2"], [linear2])) or linear2
        return drop2

    def _call_module(self, decl: Decl, args: list[Any]) -> list[_Tensor]:
        self._step()
        if decl.path in self._call_stack:
            self.warnings.append(f"recursive module call {decl.path!r} truncated to a block node")
            return self._emit_fallback_node(decl, args)
        if decl.path and len(self._call_stack) > MAX_DEPTH:
            self.warnings.append(f"module nesting too deep at {decl.path!r}; emitted as block node")
            return self._emit_fallback_node(decl, args)
        return self._run_forward_decl(decl, args)

    def _emit_fallback_node(self, decl: Decl, args: list[Any]) -> list[_Tensor]:
        tensors = _tensor_inputs(args, {})
        shape = tensors[0].shape if tensors else [-1]
        self._emit_node(
            name_base=decl.path or decl.op,
            op=decl.op,
            kind=decl.kind if decl.kind != "custom" else "block",
            module_path=decl.path,
            in_tensors=tensors,
            out_shapes=[list(shape)],
            params=dict(decl.params),
            num_params=decl.total_params(),
            source=decl.source,
            notes=["static fallback: internals not resolved"],
        )
        return [_Tensor(list(shape), [len(self.nodes) - 1])]

    # -- custom forward ---------------------------------------------------
    def _run_custom_forward(self, decl: Decl, class_info: ModuleClass, args: list[Any]) -> list[_Tensor]:
        forward = class_info.methods.get("forward")
        if forward is None:
            return self._emit_fallback_node(decl, args)
        env: dict[str, Any] = {}
        positional = [arg.arg for arg in forward.args.args if arg.arg != "self"]
        for index, name in enumerate(positional):
            env[name] = args[index] if index < len(args) else None
        defaults = list(forward.args.defaults)
        offset = len(positional) - len(defaults)
        for index, default in enumerate(defaults):
            name = positional[offset + index]
            if env.get(name) is None:
                env[name] = _eval_value(default, {})
        node_start = len(self.nodes)
        edge_start = len(self.edges)
        self._scope_stack.append(decl.path)
        signal: _ReturnSignal | None = None
        failed = False
        try:
            signal = self._exec_block(forward.body, env, decl)
        except _StaticError as exc:
            self.warnings.append(f"{decl.path or decl.op}: static forward fallback ({exc})")
            failed = True
        finally:
            self._scope_stack.pop()
        if failed:
            # Roll back partially emitted internals before emitting the single
            # fallback block, otherwise parameters are counted twice.
            del self.nodes[node_start:]
            del self.edges[edge_start:]
            return self._emit_fallback_node(decl, args)
        if signal is not None:
            tensors = self._result_tensors(signal.value)
            if tensors:
                return tensors
        if len(self.nodes) > node_start:
            # The body emitted graph nodes but did not yield a usable tensor
            # (for example a missing ``return``).  Reuse the last internal
            # tensor as this module's output instead of duplicating its params
            # in a fallback block.
            last_index = len(self.nodes) - 1
            last_node = self.nodes[last_index]
            if last_node.out_shapes:
                return [_Tensor(list(last_node.out_shapes[-1]), [last_index])]
        return self._emit_fallback_node(decl, args)

    def _exec_block(
        self, statements: Sequence[ast.stmt], env: dict[str, Any], decl: Decl
    ) -> _ReturnSignal | None:
        for statement in statements:
            result = self._exec_statement(statement, env, decl)
            if result is not None:
                return result
        return None

    def _exec_statement(self, statement: ast.stmt, env: dict[str, Any], decl: Decl) -> list[_Tensor] | None:
        self._step()
        if isinstance(statement, ast.Assign):
            value = self._eval(statement.value, env, decl)
            for target in statement.targets:
                self._bind(target, value, env)
            return None
        if isinstance(statement, ast.AnnAssign):
            if statement.value is not None:
                self._bind(statement.target, self._eval(statement.value, env, decl), env)
            return None
        if isinstance(statement, ast.AugAssign):
            current = self._eval(statement.target, env, decl)
            right = self._eval(statement.value, env, decl)
            combined = self._binary(statement.op, current, right, decl, statement)
            self._bind(statement.target, combined, env)
            return None
        if isinstance(statement, ast.Return):
            value = self._eval(statement.value, env, decl) if statement.value is not None else None
            return _ReturnSignal(value)
        if isinstance(statement, ast.Expr):
            self._eval(statement.value, env, decl)
            return None
        if isinstance(statement, ast.If):
            condition = self._truth(statement.test, env, decl)
            if condition is True:
                return self._exec_block(statement.body, env, decl)
            if condition is False:
                return self._exec_block(statement.orelse, env, decl)
            self.warnings.append("static forward: unresolved branch, analyzing the body only")
            return self._exec_block(statement.body, env, decl)
        if isinstance(statement, ast.For):
            return self._exec_for(statement, env, decl)
        if isinstance(statement, ast.With):
            return self._exec_block(statement.body, env, decl)
        if isinstance(statement, ast.Try):
            return self._exec_block(statement.body, env, decl)
        return None

    def _exec_for(self, statement: ast.For, env: dict[str, Any], decl: Decl) -> list[_Tensor] | None:
        iterable = self._eval(statement.iter, env, decl)
        items: list[Any] = []
        if isinstance(iterable, _ModuleRef):
            module_decl = iterable.decl
            items = [_ModuleRef(module_decl.children[name]) for name in module_decl.order]
        elif isinstance(iterable, _Multi):
            items = list(iterable.items)
        elif isinstance(iterable, (list, tuple)):
            items = list(iterable)
        elif isinstance(iterable, int) and not isinstance(iterable, bool):
            items = list(range(min(iterable, 64)))
        else:
            self.warnings.append("static forward: unknown for-loop iterable, loop body analyzed once")
            items = [None]
        for item in items[:64]:
            self._bind(statement.target, item, env)
            signal = self._exec_block(statement.body, env, decl)
            if signal is not None:
                return signal
        return None

    def _bind(self, target: ast.AST, value: Any, env: dict[str, Any]) -> None:
        if isinstance(target, ast.Name):
            env[target.id] = value
        elif isinstance(target, (ast.Tuple, ast.List)):
            items = self._unpack(value, len(target.elts))
            for element, item in zip(target.elts, items):
                self._bind(element, item, env)
        elif isinstance(target, ast.Attribute):
            chain = _attribute_chain(target)
            if chain and chain[0] == "self":
                env["self." + ".".join(chain[1:])] = value

    def _unpack(self, value: Any, count: int) -> list[Any]:
        if isinstance(value, _Multi):
            items = list(value.items)
        elif isinstance(value, (list, tuple)):
            items = list(value)
        elif isinstance(value, _Tensor):
            items = [value]
        else:
            items = [value]
        while len(items) < count:
            items.append(None)
        return items[:count]

    # -- expression evaluation --------------------------------------------
    def _eval(self, expression: ast.AST | None, env: dict[str, Any], decl: Decl) -> Any:
        if expression is None:
            return None
        if isinstance(expression, ast.Constant):
            return expression.value
        if isinstance(expression, ast.Name):
            if expression.id in env:
                return env[expression.id]
            if expression.id == "True":
                return True
            if expression.id == "False":
                return False
            if expression.id == "None":
                return None
            return _FuncRef(expression.id)
        if isinstance(expression, ast.Attribute):
            return self._eval_attribute(expression, env, decl)
        if isinstance(expression, ast.Call):
            return self._eval_call(expression, env, decl)
        if isinstance(expression, ast.BinOp):
            left = self._eval(expression.left, env, decl)
            right = self._eval(expression.right, env, decl)
            return self._binary(expression.op, left, right, decl, expression)
        if isinstance(expression, ast.UnaryOp):
            operand = self._eval(expression.operand, env, decl)
            if isinstance(expression.op, ast.USub) and isinstance(operand, (int, float)):
                return -operand
            if isinstance(expression.op, ast.Not):
                return not bool(operand)
            if isinstance(operand, _Tensor) and isinstance(expression.op, ast.USub):
                return self._emit_function("neg", "custom", [operand], operand.shape, {}, expression)
            return operand
        if isinstance(expression, (ast.Tuple, ast.List)):
            return [self._eval(item, env, decl) for item in expression.elts]
        if isinstance(expression, ast.Subscript):
            value = self._eval(expression.value, env, decl)
            index = self._eval(expression.slice, env, decl)
            return self._subscript(value, index)
        if isinstance(expression, ast.Starred):
            return self._eval(expression.value, env, decl)
        if isinstance(expression, ast.Slice):
            lower = self._eval(expression.lower, env, decl) if expression.lower is not None else None
            upper = self._eval(expression.upper, env, decl) if expression.upper is not None else None
            step = self._eval(expression.step, env, decl) if expression.step is not None else None
            if all(value is None or isinstance(value, int) for value in (lower, upper, step)):
                return slice(lower, upper, step)
            return None
        if isinstance(expression, ast.Compare):
            left = self._eval(expression.left, env, decl)
            rights = [self._eval(item, env, decl) for item in expression.comparators]
            result: bool | None = True
            previous = left
            for operator_node, right in zip(expression.ops, rights):
                compared = _compare_values(operator_node, previous, right)
                if compared is None:
                    return None
                result = bool(result) and compared
                previous = right
            return result
        if isinstance(expression, ast.BoolOp):
            values = [self._eval(item, env, decl) for item in expression.values]
            if isinstance(expression.op, ast.And):
                return all(bool(value) for value in values)
            return any(bool(value) for value in values)
        if isinstance(expression, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
            return _eval_comprehension(expression, env)
        if isinstance(expression, ast.IfExp):
            condition = self._truth(expression.test, env, decl)
            if condition is True:
                return self._eval(expression.body, env, decl)
            if condition is False:
                return self._eval(expression.orelse, env, decl)
            return self._eval(expression.body, env, decl)
        if isinstance(expression, ast.JoinedStr):
            return None
        return None

    def _truth(self, expression: ast.AST, env: dict[str, Any], decl: Decl | None = None) -> bool | None:
        value = self._eval(expression, env, decl or _DUMMY_DECL)
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        return None

    def _eval_attribute(self, expression: ast.Attribute, env: dict[str, Any], decl: Decl) -> Any:
        chain = _attribute_chain(expression)
        if chain and chain[0] == "self":
            return self._resolve_self(chain[1:], env, decl)
        base = self._eval(expression.value, env, decl)
        if isinstance(base, _ModuleRef):
            child = base.decl.children.get(expression.attr)
            if child is not None:
                return _ModuleRef(child)
            return None
        if isinstance(base, _Tensor):
            if expression.attr == "shape":
                return list(base.shape)
            if expression.attr == "T":
                order = list(reversed(range(len(base.shape))))
                out_shape = [base.shape[i] for i in order]
                return self._emit_function("transpose", "reshape", [base], out_shape, {"dims": order}, expression)
            return _MethodRef(expression.attr, base)
        if isinstance(base, _FuncRef):
            return _FuncRef(f"{base.name}.{expression.attr}")
        if isinstance(base, _ParamRef):
            if expression.attr == "shape":
                return [-1]
            return _ParamRef(f"{base.name}.{expression.attr}")
        if isinstance(base, (list, tuple)):
            return _MethodRef(expression.attr, base)
        if isinstance(base, dict):
            if expression.attr in {"get", "items", "keys", "values", "pop", "update"}:
                return _MethodRef(expression.attr, base)
            return base.get(expression.attr)
        return None

    def _resolve_self(self, chain: list[str], env: dict[str, Any], decl: Decl) -> Any:
        key = ".".join(chain)
        current = decl
        traversed: list[str] = []
        for part in chain:
            child = current.children.get(part)
            if child is not None:
                if child.tensor_shape is not None:
                    return _Tensor(list(child.tensor_shape))
                current = child
                traversed.append(part)
                continue
            if not traversed:
                attrs = getattr(decl, "attrs", {}) or {}
                if part in attrs:
                    return attrs[part]
                if decl.class_info is not None and part in decl.class_info.methods:
                    return _MethodRef(f"self.{part}")
                runtime_value = env.get("self." + key)
                if runtime_value is not None:
                    return runtime_value
            return _ParamRef(".".join([*traversed, part]) if traversed else part)
        return _ModuleRef(current)

    def _eval_call(self, call: ast.Call, env: dict[str, Any], decl: Decl) -> Any:
        func = self._eval(call.func, env, decl)
        args = [self._eval(arg, env, decl) for arg in call.args if not isinstance(arg, ast.Starred)]
        for arg in call.args:
            if isinstance(arg, ast.Starred):
                value = self._eval(arg.value, env, decl)
                if isinstance(value, (list, tuple)):
                    args.extend(value)
        kwargs: dict[str, Any] = {}
        for keyword in call.keywords:
            if keyword.arg is None:
                continue
            kwargs[keyword.arg] = self._eval(keyword.value, env, decl)

        if isinstance(func, _ModuleRef):
            result = self._call_module(func.decl, args)
            return self._unwrap_module_result(result)
        if isinstance(func, _MethodRef):
            return self._call_method(func, args, kwargs, call)
        if isinstance(func, _FuncRef):
            return self._call_function(func.name, args, kwargs, call)
        return None

    @staticmethod
    def _unwrap_module_result(result: list[Any]) -> Any:
        if len(result) == 1:
            return result[0]
        if len(result) > 1:
            return _Multi(list(result))
        return None

    def _call_method(self, func: _MethodRef, args: list[Any], kwargs: dict[str, Any], call: ast.Call) -> Any:
        name = func.name.rsplit(".", 1)[-1]
        if func.name.startswith("self."):
            return self._call_helper_method(name, args, kwargs)
        base = func.base
        if isinstance(base, _Tensor):
            return self._tensor_method(name, base, args, kwargs, call)
        if isinstance(base, list):
            if name == "append" and args:
                base.append(args[0])
                return None
            if name == "extend" and args:
                value = args[0]
                if isinstance(value, (list, tuple)):
                    base.extend(value)
                return None
            if name == "insert" and len(args) >= 2 and isinstance(args[0], int):
                base.insert(args[0], args[1])
                return None
            if name == "pop":
                if not base:
                    return None
                index = args[0] if args and isinstance(args[0], int) else -1
                try:
                    return base.pop(index)
                except IndexError:
                    return None
            if name == "clear":
                base.clear()
                return None
            if name == "copy":
                return list(base)
            if name == "index" and args:
                try:
                    return base.index(args[0])
                except ValueError:
                    return None
            if name == "count" and args:
                return base.count(args[0])
        if isinstance(base, dict):
            if name == "get" and args:
                return base.get(args[0], args[1] if len(args) > 1 else None)
            if name == "items":
                return list(base.items())
            if name == "keys":
                return list(base.keys())
            if name == "values":
                return list(base.values())
            if name == "pop" and args:
                return base.pop(args[0], None)
            if name == "update" and args and isinstance(args[0], dict):
                base.update(args[0])
                return None
        return None

    def _call_module_function(
        self,
        function: ast.FunctionDef,
        args: list[Any],
        kwargs: dict[str, Any],
    ) -> Any:
        """Symbolically execute a module-level helper function."""
        name = function.name
        if name in self._helper_stack or len(self._helper_stack) > 8:
            self._warn(f"recursive helper function {name!r} skipped")
            return None
        positional = [arg.arg for arg in function.args.args if arg.arg not in ("self", "cls")]
        defaults = list(function.args.defaults)
        offset = len(positional) - len(defaults)
        env: dict[str, Any] = {}
        for index, parameter in enumerate(positional):
            if index < len(args):
                env[parameter] = args[index]
            elif parameter in kwargs:
                env[parameter] = kwargs[parameter]
        for index, default in enumerate(defaults):
            parameter = positional[offset + index]
            if env.get(parameter) is None:
                env[parameter] = _eval_value(default, env)
        self._helper_stack.append(name)
        try:
            scope_decl = self._scope_decl() or self.root
            signal = self._exec_block(function.body, env, scope_decl)
        except _StaticError as exc:
            self._warn(f"helper function {name!r}: {exc}")
            signal = None
        finally:
            self._helper_stack.pop()
        if signal is None:
            return None
        return signal.value

    def _call_helper_method(self, name: str, args: list[Any], kwargs: dict[str, Any]) -> Any:
        decl = self._scope_decl()
        if decl is None or decl.class_info is None:
            return None
        method = decl.class_info.methods.get(name)
        if method is None:
            return None
        forward = decl.class_info.methods.get("forward")
        positional = [arg.arg for arg in method.args.args if arg.arg != "self"]
        env: dict[str, Any] = {}
        for index, param in enumerate(positional):
            env[param] = args[index] if index < len(args) else kwargs.get(param)
        signal = self._exec_block(method.body, env, decl)
        if signal is None:
            return None
        return signal.value

    def _scope_decl(self) -> Decl | None:
        path = self.scope
        if not path:
            return self.root
        node = self.root
        for part in path.split("."):
            if part not in node.children:
                return None
            node = node.children[part]
        return node

    def _warn(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)

    def _unknown_tensor_result(
        self,
        label: str,
        tensors: Sequence[_Tensor],
        call: ast.Call,
    ) -> _Tensor:
        """Conservative fallback for an unknown tensor op: never identity."""
        in_shapes = [list(tensor.shape) for tensor in tensors if tensor.shape]
        if len(tensors) >= 2:
            broadcast = _try_broadcast(in_shapes)
            shape = broadcast if broadcast is not None else _unknown_shape_for(in_shapes)
        else:
            shape = _unknown_shape_for(in_shapes)
        self._warn(f"shape unknown for {label!r}")
        short = label.rsplit(".", 1)[-1]
        return self._emit_function(short, "custom", tensors, shape, {}, call)

    def _emit_multi_function(
        self,
        op_short: str,
        op: str,
        kind: str,
        in_tensors: Sequence[_Tensor],
        out_shapes: list[Shape],
        params: dict[str, Any],
        call: ast.Call,
    ) -> list[_Tensor]:
        name_base = self._function_base(op_short)
        node_index = self._emit_node(
            name_base=name_base,
            op=op,
            kind=kind,
            module_path=name_base,
            in_tensors=list(in_tensors),
            out_shapes=[list(shape) for shape in out_shapes],
            params=params,
            source=_source_loc(self.source, call),
        )
        return [_Tensor(list(shape), [node_index]) for shape in out_shapes]

    def _tensor_method(
        self, name: str, tensor: _Tensor, args: list[Any], kwargs: dict[str, Any], call: ast.Call
    ) -> Any:
        shape = list(tensor.shape)
        rank = len(shape)
        if name in ("size", "dim", "numel", "ndimension"):
            if name == "size":
                if args and isinstance(args[0], int):
                    dim = _normalize_dim(int(args[0]), rank)
                    return shape[dim] if 0 <= dim < rank else -1
                return list(shape)
            if name in ("dim", "ndimension"):
                return rank
            return _product(value for value in shape if value >= 0) if all(v >= 0 for v in shape) else -1
        if name in ("item", "tolist", "numpy", "detach_"):
            return None
        if name in (
            "contiguous",
            "clone",
            "detach",
            "to",
            "float",
            "double",
            "half",
            "long",
            "int",
            "short",
            "bool",
            "byte",
            "type",
            "type_as",
            "cpu",
            "cuda",
            "requires_grad_",
        ):
            return tensor
        if name in ("flatten", "unflatten"):
            start = args[0] if args else kwargs.get("start_dim", 1)
            end = args[1] if len(args) > 1 else kwargs.get("end_dim", -1)
            start_value = int(start) if isinstance(start, int) else 1
            end_value = int(end) if isinstance(end, int) else -1
            out = _flatten_shape(shape, start_value, end_value)
            return self._emit_function(name, "flatten", [tensor], out, {"startDim": start_value}, call)
        if name in ("view", "reshape"):
            dims = _view_dims_from_args(args, kwargs)
            out = _resolve_view_shape(shape, dims) if dims else list(shape)
            params = {"shape": dims} if dims else {}
            return self._emit_function(name, "reshape", [tensor], out, params, call)
        if name in ("permute",):
            dims = [int(value) for value in args if isinstance(value, int)]
            if len(dims) == rank:
                out = [shape[_normalize_dim(dim, rank)] for dim in dims]
                return self._emit_function("permute", "reshape", [tensor], out, {"dims": dims}, call)
            return self._unknown_tensor_result("Tensor.permute", [tensor], call)
        if name in ("transpose", "swapdims", "swapaxes"):
            if len(args) >= 2 and isinstance(args[0], int) and isinstance(args[1], int):
                first = _normalize_dim(int(args[0]), rank)
                second = _normalize_dim(int(args[1]), rank)
                out = list(shape)
                if 0 <= first < rank and 0 <= second < rank:
                    out[first], out[second] = out[second], out[first]
                return self._emit_function("transpose", "reshape", [tensor], out, {"dims": [first, second]}, call)
            return self._unknown_tensor_result("Tensor.transpose", [tensor], call)
        if name in ("unsqueeze", "squeeze"):
            if name == "unsqueeze":
                dim = _normalize_dim(int(args[0]) if args and isinstance(args[0], int) else 0, rank + 1)
                out = shape[:dim] + [1] + shape[dim:]
            else:
                dim = args[0] if args and isinstance(args[0], int) else None
                if dim is None:
                    out = [value for value in shape if value != 1]
                else:
                    resolved = _normalize_dim(int(dim), rank)
                    out = shape[:resolved] + shape[resolved + 1 :] if 0 <= resolved < rank else shape
            return self._emit_function(name, "reshape", [tensor], out, {}, call)
        if name in ("mean", "sum", "amax", "amin", "prod", "norm", "median", "nansum", "nanmean"):
            dim = kwargs.get("dim", args[0] if args else None)
            keepdim = bool(kwargs.get("keepdim", args[1] if len(args) > 1 and isinstance(args[1], bool) else False))
            out = _reduce_shape(shape, dim if isinstance(dim, (int, list, tuple)) else None, keepdim)
            params = {"dim": dim} if isinstance(dim, (int, list, tuple)) else {}
            return self._emit_function(name, "pool", [tensor], out, params, call)
        if name in ("max", "min"):
            dim = kwargs.get("dim", args[0] if args else None)
            keepdim = bool(kwargs.get("keepdim", args[1] if len(args) > 1 and isinstance(args[1], bool) else False))
            if isinstance(dim, int):
                out = _reduce_shape(shape, dim, keepdim)
                return self._emit_multi_function(
                    name, f"Tensor.{name}", "pool", [tensor], [out, out], {"dim": dim}, call
                )
            out = _reduce_shape(shape, None)
            return self._emit_function(name, "pool", [tensor], out, {}, call)
        if name in ("chunk", "split"):
            dim = kwargs.get("dim", args[1] if len(args) > 1 and isinstance(args[1], int) else 0)
            resolved = _normalize_dim(int(dim), rank) if isinstance(dim, int) else 0
            if not (0 <= resolved < rank):
                return self._unknown_tensor_result(f"Tensor.{name}", [tensor], call)
            size = shape[resolved]
            requested = args[0] if args else kwargs.get("split_size_or_sections")
            if isinstance(requested, (list, tuple)):
                sizes = [int(value) if isinstance(value, int) else -1 for value in requested]
            else:
                chunks = requested if isinstance(requested, int) and requested > 0 else 2
                chunks = min(chunks, max(1, size)) if size >= 0 else min(chunks, 8)
                if size >= 0:
                    chunk_size = max(1, math.ceil(size / chunks))
                    sizes = []
                    remaining = size
                    while remaining > 0:
                        sizes.append(min(chunk_size, remaining))
                        remaining -= chunk_size
                else:
                    sizes = [-1] * chunks
            out_shapes: list[Shape] = []
            for chunk_size in sizes:
                part = list(shape)
                part[resolved] = chunk_size
                out_shapes.append(part)
            result = self._emit_multi_function(
                name,
                f"Tensor.{name}",
                "reshape",
                [tensor],
                out_shapes,
                {"chunks": len(out_shapes), "dim": resolved},
                call,
            )
            return _Multi(result)
        if name in ("topk",):
            k = args[0] if args else kwargs.get("k")
            dim = kwargs.get("dim", args[1] if len(args) > 1 and isinstance(args[1], int) else -1)
            out = _topk_shape(shape, dim, k)
            return self._emit_multi_function(
                "topk", "Tensor.topk", "custom", [tensor], [out, out], {"k": k, "dim": dim}, call
            )
        if name in ("sort",):
            dim = kwargs.get("dim", args[0] if args and isinstance(args[0], int) else -1)
            return self._emit_multi_function(
                "sort", "Tensor.sort", "custom", [tensor], [list(shape), list(shape)], {"dim": dim}, call
            )
        if name in ("argsort",):
            dim = kwargs.get("dim", args[0] if args and isinstance(args[0], int) else -1)
            return self._emit_function("argsort", "custom", [tensor], shape, {"dim": dim}, call)
        if name in ("gather", "index_select", "take_along_dim", "scatter", "index_copy"):
            index = None
            dim = None
            if name in ("gather", "index_select", "scatter") and len(args) >= 2:
                dim = args[0]
                index = _first_tensor([args[1]])
            elif name == "take_along_dim" and len(args) >= 2:
                dim = kwargs.get("dim", args[0] if isinstance(args[0], int) else None)
                index = _first_tensor([args[1]])
            else:
                index = _first_tensor(args[1:]) or _first_tensor(list(kwargs.values()))
                dim = kwargs.get("dim")
            if name == "index_select":
                out = _index_select_shape(shape, dim, index)
            elif name in ("scatter", "index_copy"):
                out = list(shape)
            elif index is not None and index.shape:
                out = list(index.shape)
            else:
                out = _unknown_shape_for([shape])
            in_tensors = [tensor, *([index] if index is not None else [])]
            return self._emit_function(name, "custom", in_tensors, out, {"dim": dim} if isinstance(dim, int) else {}, call)
        if name in ("expand", "expand_as"):
            if name == "expand_as":
                target = _first_tensor(args)
                out = list(target.shape) if target is not None else _unknown_shape_for([shape])
            else:
                dims = [value if isinstance(value, int) else -1 for value in args]
                out = _expand_shape(shape, dims)
            return self._emit_function(name if name == "expand" else "expand_as", "reshape", [tensor], out, {}, call)
        if name == "repeat":
            repeats = args[0] if args and isinstance(args[0], (list, tuple)) else args
            repeat_values = [int(value) if isinstance(value, int) else -1 for value in repeats]
            out = _repeat_shape(shape, repeat_values)
            return self._emit_function("repeat", "reshape", [tensor], out, {"repeats": repeat_values}, call)
        if name in ("repeat_interleave",):
            dim = kwargs.get("dim", args[1] if len(args) > 1 and isinstance(args[1], int) else None)
            out = list(shape)
            if isinstance(dim, int) and 0 <= _normalize_dim(dim, rank) < rank:
                out[_normalize_dim(dim, rank)] = -1
            return self._emit_function(name, "reshape", [tensor], out, {}, call)
        if name in ("masked_fill", "masked_fill_"):
            masks = _tensor_inputs(args, kwargs)
            return self._emit_function("masked_fill", "attention", [tensor, *masks], shape, {}, call)
        if name in ("masked_select",):
            return self._emit_function("masked_select", "custom", [tensor, *_tensor_inputs(args, kwargs)], [-1], {}, call)
        if name in ("softmax", "log_softmax"):
            dim = kwargs.get("dim", args[0] if args else None)
            return self._emit_function(name, "softmax", [tensor], shape, {"dim": dim} if isinstance(dim, int) else {}, call)
        if name in ("relu", "relu6", "gelu", "silu", "swish", "sigmoid", "tanh", "elu", "selu", "mish", "hardswish", "hardsigmoid", "leaky_relu"):
            return self._emit_function(name, "activation", [tensor], shape, {}, call)
        if name == "dropout":
            return self._emit_function("dropout", "dropout", [tensor], shape, {}, call)
        if name in _UNARY_SHAPE_PRESERVING:
            return self._emit_function(name, "custom", [tensor], shape, {}, call)
        return self._unknown_tensor_result(f"Tensor.{name}", [tensor], call)

    def _call_function(self, name: str, args: list[Any], kwargs: dict[str, Any], call: ast.Call) -> Any:
        short = name.rsplit(".", 1)[-1]
        tensors = _tensor_inputs(args, kwargs)

        if name in self.module_functions:
            return self._call_module_function(self.module_functions[name], args, kwargs)

        # --- scalar helpers (no tensors) ---------------------------------
        if short in ("tuple", "list") and not tensors:
            if len(args) == 1 and isinstance(args[0], (list, tuple)):
                return tuple(args[0]) if short == "tuple" else list(args[0])
            return tuple(args) if short == "tuple" else list(args)
        if short in ("min", "max") and not tensors:
            numbers = [value for value in args if isinstance(value, (int, float)) and not isinstance(value, bool)]
            if not numbers and len(args) == 1 and isinstance(args[0], (list, tuple)) and args[0]:
                numbers = [value for value in args[0] if isinstance(value, (int, float))]
            if numbers:
                return min(numbers) if short == "min" else max(numbers)
            return None
        if short in ("range",) and not tensors:
            values = [value for value in args if isinstance(value, int)]
            return list(range(*values)) if values else []
        if short in ("abs", "round", "floor", "ceil") and not tensors and args and isinstance(args[0], (int, float)):
            mapping = {"abs": abs, "round": round, "floor": math.floor, "ceil": math.ceil}
            return mapping[short](args[0])
        if short in ("float", "int", "bool", "str") and not tensors:
            if not args:
                return None
            try:
                return {"float": float, "int": int, "bool": bool, "str": str}[short](args[0])
            except (TypeError, ValueError):
                return None
        if short == "len" and not tensors:
            return len(args[0]) if args and hasattr(args[0], "__len__") else None
        if short in ("getattr", "size", "isinstance", "print", "super", "setattr", "hasattr"):
            return None

        # --- multi-output shape rules ------------------------------------
        if short in ("topk",):
            source = _first_tensor(args[0]) if args else None
            if source is None:
                return None
            k = args[1] if len(args) > 1 else kwargs.get("k")
            dim = kwargs.get("dim", args[2] if len(args) > 2 and isinstance(args[2], int) else -1)
            out = _topk_shape(source.shape, dim, k)
            return self._emit_multi_function(
                "topk", "topk", "custom", [source], [out, out], {"k": k, "dim": dim}, call
            )
        if short == "sort":
            source = _first_tensor(args[0]) if args else None
            if source is None:
                return None
            return self._emit_multi_function(
                "sort", "sort", "custom", [source], [list(source.shape), list(source.shape)], {}, call
            )
        if short == "argsort":
            source = _first_tensor(args[0]) if args else None
            if source is None:
                return None
            return self._emit_function("argsort", "custom", [source], list(source.shape), {}, call)

        # --- indexing / gather -------------------------------------------
        if short in ("chunk", "split"):
            source = _first_tensor(args[0]) if args else None
            if source is None:
                return None
            return self._tensor_method(short, source, list(args[1:]), kwargs, call)
        if short in ("gather", "index_select", "take_along_dim"):
            source = _first_tensor(args[0]) if args else None
            if source is None:
                return None
            if short == "index_select":
                dim = args[1] if len(args) > 1 else None
                index = _first_tensor(args[2:]) if len(args) > 2 else _first_tensor(kwargs.values())
                out = _index_select_shape(source.shape, dim, index)
            elif short == "gather":
                dim = args[1] if len(args) > 1 else None
                index = _first_tensor(args[2:]) if len(args) > 2 else _first_tensor(kwargs.values())
                out = list(index.shape) if index is not None and index.shape else _unknown_shape_for([source.shape])
            else:
                index = _first_tensor(args[1:]) if len(args) > 1 else _first_tensor(kwargs.values())
                out = list(index.shape) if index is not None and index.shape else _unknown_shape_for([source.shape])
            in_tensors = [source, *([index] if index is not None else [])]
            return self._emit_function(short, "custom", in_tensors, out, {}, call)

        # --- reshape / expand / layout -----------------------------------
        if short in ("reshape", "view"):
            source = _first_tensor(args[0]) if args else None
            if source is None:
                return None
            dims = _view_dims_from_args(args[1:], kwargs)
            out = _resolve_view_shape(list(source.shape), dims) if dims else list(source.shape)
            return self._emit_function(short, "reshape", [source], out, {"shape": dims} if dims else {}, call)
        if short in ("flatten",):
            source = _first_tensor(args[0]) if args else None
            if source is None:
                return None
            start = args[1] if len(args) > 1 and isinstance(args[1], int) else 0
            end = args[2] if len(args) > 2 and isinstance(args[2], int) else -1
            out = _flatten_shape(list(source.shape), int(start), int(end))
            return self._emit_function("flatten", "flatten", [source], out, {"startDim": int(start)}, call)
        if short in ("squeeze", "unsqueeze"):
            source = _first_tensor(args[0]) if args else None
            if source is None:
                return None
            return self._tensor_method(short, source, args[1:], kwargs, call)
        if short == "permute":
            source = _first_tensor(args[0]) if args else None
            if source is None:
                return None
            dims = args[1] if len(args) > 1 else None
            return self._tensor_method("permute", source, list(dims) if isinstance(dims, (list, tuple)) else list(args[1:]), kwargs, call)
        if short == "transpose":
            source = _first_tensor(args[0]) if args else None
            if source is None:
                return None
            return self._tensor_method("transpose", source, list(args[1:]), kwargs, call)
        if short in ("expand", "broadcast_to"):
            source = _first_tensor(args[0]) if args else None
            if source is None:
                return None
            dims = args[1] if len(args) > 1 and isinstance(args[1], (list, tuple)) else args[1:]
            dims = [value if isinstance(value, int) else -1 for value in dims]
            out = _expand_shape(list(source.shape), dims)
            return self._emit_function(short, "reshape", [source], out, {}, call)
        if short in ("zeros_like", "ones_like", "rand_like", "full_like", "empty_like"):
            source = _first_tensor(args[0]) if args else None
            return _Tensor(list(source.shape), list(source.producers)) if source is not None else None
        if short in ("triu", "tril"):
            source = _first_tensor(args[0]) if args else None
            if source is None:
                return None
            return self._emit_function(short, "custom", [source], list(source.shape), {}, call)

        # --- reductions ---------------------------------------------------
        if short in _REDUCE_FUNCTIONS:
            source = _first_tensor(args[0]) if args else None
            if source is None:
                return None
            dim = kwargs.get("dim", args[1] if len(args) > 1 else None)
            keepdim = bool(kwargs.get("keepdim", args[2] if len(args) > 2 and isinstance(args[2], bool) else False))
            out = _reduce_shape(list(source.shape), dim if isinstance(dim, (int, list, tuple)) else None, keepdim)
            return self._emit_function(short, "pool", [source], out, {"dim": dim} if isinstance(dim, (int, list, tuple)) else {}, call)
        if short in ("max", "min") and tensors:
            if len(tensors) == 1:
                dim = kwargs.get("dim", args[1] if len(args) > 1 and isinstance(args[1], int) else None)
                if isinstance(dim, int):
                    out = _reduce_shape(list(tensors[0].shape), dim)
                    return self._emit_multi_function(short, short, "pool", [tensors[0]], [out, out], {"dim": dim}, call)
                out = _reduce_shape(list(tensors[0].shape), None)
                return self._emit_function(short, "pool", [tensors[0]], out, {}, call)
            out = _try_broadcast([list(t.shape) for t in tensors]) or _unknown_shape_for([list(t.shape) for t in tensors])
            return self._emit_function(short, "custom", tensors, out, {}, call)

        # --- merge / elementwise -----------------------------------------
        if short in ("cat", "concat"):
            sequences = [arg for arg in args if isinstance(arg, (list, tuple))]
            items: list[_Tensor] = []
            for sequence in sequences:
                for item in sequence:
                    tensor = _first_tensor(item)
                    if tensor is not None:
                        items.append(tensor)
            if items:
                dim = kwargs.get("dim", args[1] if len(args) > 1 and isinstance(args[1], int) else 0)
                out = _concat_shape([item.shape for item in items], int(dim))
                return self._emit_function("cat", "concat", items, out, {"dim": int(dim)}, call)
            return None
        if short == "stack":
            items = [item for item in (_first_tensor(entry) for entry in (args[0] if args and isinstance(args[0], (list, tuple)) else args)) if item is not None]
            if not items:
                return None
            dim = kwargs.get("dim", 0)
            resolved = _normalize_dim(int(dim), len(items[0].shape) + 1)
            base = list(items[0].shape)
            out = base[:resolved] + [len(items)] + base[resolved:]
            return self._emit_function("stack", "concat", items, out, {"dim": resolved}, call)
        if short in ("where",):
            if len(tensors) >= 3:
                out = _try_broadcast([list(t.shape) for t in tensors]) or _unknown_shape_for([list(t.shape) for t in tensors])
                return self._emit_function("where", "custom", tensors, out, {}, call)
            return None
        if short in ("masked_fill", "masked_fill_", "masked_select"):
            source = _first_tensor(args[0]) if args else None
            if source is None:
                return None
            if short == "masked_select":
                return self._emit_function("masked_select", "custom", [source, *tensors[1:]], [-1], {}, call)
            return self._emit_function("masked_fill", "attention", [source, *tensors[1:]], list(source.shape), {}, call)
        if short in _BINARY_FUNCS:
            if not tensors:
                left = args[0] if args else None
                right = args[1] if len(args) > 1 else None
                return _math_scalar(short, left, right) if isinstance(left, (int, float)) and isinstance(right, (int, float)) else None
            out = _try_broadcast([list(t.shape) for t in tensors]) or _unknown_shape_for([list(t.shape) for t in tensors])
            kind = "add" if short in ("add", "sub", "subtract", "add_", "sub_") else "custom"
            return self._emit_function(short, kind, tensors, out, {}, call)
        if short in _UNARY_SHAPE_PRESERVING:
            source = _first_tensor(args[0]) if args else None
            if source is None:
                return None
            return self._emit_function(short, "custom", [source], list(source.shape), {}, call)

        # --- module functions --------------------------------------------
        if short in ("relu", "relu6", "gelu", "silu", "swish", "leaky_relu", "sigmoid", "tanh", "elu", "selu", "mish", "hardswish", "hardsigmoid", "softplus"):
            source = _first_tensor(args[0]) if args else None
            if source is None:
                return None
            return self._emit_function(short, "activation", [source], list(source.shape), {}, call)
        if short in ("softmax", "log_softmax"):
            source = _first_tensor(args[0]) if args else None
            if source is None:
                return None
            dim = kwargs.get("dim", args[1] if len(args) > 1 and isinstance(args[1], int) else None)
            return self._emit_function(short, "softmax", [source], list(source.shape), {"dim": dim} if isinstance(dim, int) else {}, call)
        if short in ("dropout", "dropout_"):
            source = _first_tensor(args[0]) if args else None
            if source is None:
                return None
            return self._emit_function("dropout", "dropout", [source], list(source.shape), {}, call)
        if short in ("fft", "ifft", "fftn", "ifftn", "rfft", "irfft", "rfftn", "irfftn", "rfft2", "irfft2", "fftshift", "ifftshift"):
            source = _first_tensor(args[0]) if args else None
            if source is None:
                return None
            s_value = kwargs.get("s", args[1] if len(args) > 1 and isinstance(args[1], (list, tuple)) else None)
            dim_value = kwargs.get("dim", args[2] if len(args) > 2 else (args[1] if len(args) > 1 and isinstance(args[1], int) else None))
            out = _fft_shape(list(source.shape), short, s_value, dim_value)
            return self._emit_function(short, "custom", [source], out, {}, call)
        if short == "einsum":
            equation = next((value for value in args if isinstance(value, str)), kwargs.get("equation"))
            if not isinstance(equation, str):
                return self._unknown_tensor_result("einsum", tensors, call) if tensors else None
            out = _einsum_shape(equation, [list(t.shape) for t in tensors])
            return self._emit_function("einsum", "custom", tensors, out, {"equation": equation}, call)
        if short in ("matmul", "bmm"):
            left = _first_tensor(args[0]) if args else None
            right = _first_tensor(args[1]) if len(args) > 1 else None
            if left is not None and right is not None:
                out = _matmul_shape(list(left.shape), list(right.shape)) or _unknown_shape_for([list(left.shape), list(right.shape)])
                return self._emit_function("matmul", "attention", [left, right], out, {}, call)
            return None
        if short == "scaled_dot_product_attention":
            query = _first_tensor(args[0]) if args else None
            key = _first_tensor(args[1]) if len(args) > 1 else None
            value = _first_tensor(args[2]) if len(args) > 2 else None
            if query is None:
                return None
            return self._emit_function(
                "scaled_dot_product_attention",
                "attention",
                [item for item in (query, key, value) if item is not None],
                list(query.shape),
                {},
                call,
            )
        if short in ("linear", "conv1d", "conv2d", "conv3d", "layer_norm", "batch_norm", "group_norm"):
            source = _first_tensor(args[0]) if args else None
            if source is None:
                return None
            kind = ops.classify_module(short)
            params = {"inFeatures": source.shape[-1] if source.shape else 0}
            return self._emit_function(short, kind, [source], list(source.shape), params, call)
        if short in ("adaptive_avg_pool2d", "avg_pool2d", "max_pool2d"):
            source = _first_tensor(args[0]) if args else None
            if source is None:
                return None
            target = args[1] if len(args) > 1 else kwargs.get("output_size") if short.startswith("adaptive") else None
            params = {"outputSize": target} if short.startswith("adaptive") else {"kernelSize": target}
            out, _warnings = ops.infer_output_shapes("pool", params, [list(source.shape)], short)
            return self._emit_function(short, "pool", [source], out[0] if out else list(source.shape), params, call)
        if short == "interpolate":
            source = _first_tensor(args[0]) if args else None
            if source is None:
                return None
            size = kwargs.get("size", args[1] if len(args) > 1 else None)
            out = None
            if isinstance(size, (list, tuple)) and size:
                dims = len(source.shape) - 2
                values = [int(value) if isinstance(value, int) else -1 for value in size]
                if len(values) == dims:
                    out = [*source.shape[:2], *values]
                elif len(values) == 2 and dims == 2:
                    out = [*source.shape[:2], *values]
                else:
                    out = [*source.shape[: len(source.shape) - len(values)], *values]
            if out is None:
                scale = kwargs.get("scale_factor", args[2] if len(args) > 2 else None)
                if isinstance(scale, (int, float)) and scale > 0 and len(source.shape) >= 3:
                    out = [*source.shape[:-2], *(int(dim * scale) if dim >= 0 else -1 for dim in source.shape[-2:])]
                else:
                    out = _unknown_shape_for([list(source.shape)])
            return self._emit_function("interpolate", "upsample", [source], out, {}, call)
        if short == "pad":
            source = _first_tensor(args[0]) if args else None
            if source is None:
                return None
            pad = args[1] if len(args) > 1 else kwargs.get("pad")
            if isinstance(pad, (list, tuple)):
                out = list(source.shape)
                for index in range(0, len(pad), 2):
                    left = pad[index] if index < len(pad) else 0
                    right = pad[index + 1] if index + 1 < len(pad) else 0
                    dim = len(out) - 1 - index // 2
                    if dim >= 0 and isinstance(left, int) and isinstance(right, int) and out[dim] >= 0:
                        out[dim] += left + right
                return self._emit_function("pad", "custom", [source], out, {}, call)
            return self._unknown_tensor_result("pad", [source], call)
        if short in ("embedding",):
            weight = _first_tensor(args[0]) if args else None
            indices = _first_tensor(args[1]) if len(args) > 1 else None
            if weight is not None and indices is not None and len(weight.shape) == 2:
                return self._emit_function("embedding", "embedding", [indices], [*indices.shape, weight.shape[1]], {}, call)
            return None
        if short == "one_hot":
            source = _first_tensor(args[0]) if args else None
            classes = args[1] if len(args) > 1 and isinstance(args[1], int) else 0
            if source is None:
                return None
            return self._emit_function("one_hot", "embedding", [source], [*source.shape, classes], {}, call)

        # --- constants / constructors ------------------------------------
        if short in ("arange",):
            numbers = [value for value in args[:3] if isinstance(value, (int, float)) and not isinstance(value, bool)]
            if len(numbers) >= 3 and numbers[2] != 0:
                length = int(math.ceil((numbers[1] - numbers[0]) / numbers[2]))
                return _Tensor([max(0, length)])
            if len(numbers) == 2 and numbers[1] != 0:
                return _Tensor([max(0, int(math.ceil((numbers[1] - numbers[0]) / 1)))])
            if len(args) == 1 and isinstance(args[0], int):
                return _Tensor([args[0]])
            return _Tensor([-1])
        if short in ("linspace",):
            steps = args[2] if len(args) > 2 else kwargs.get("steps")
            return _Tensor([int(steps)] if isinstance(steps, int) and steps > 0 else [-1])
        if short in ("zeros", "ones", "randn", "rand", "empty", "full", "tensor"):
            size = args[0] if args else kwargs.get("size")
            if isinstance(size, (list, tuple)):
                return _Tensor([int(value) for value in size if isinstance(value, int)] or [-1])
            int_args = [value for value in args if isinstance(value, int) and not isinstance(value, bool)]
            return _Tensor(int_args if int_args else [-1])
        if short in ("meshgrid",):
            if not tensors:
                return None
            shape_out = _try_broadcast([list(t.shape) for t in tensors]) or [-1]
            return _Multi(self._emit_multi_function("meshgrid", "meshgrid", "custom", tensors, [list(shape_out) for _ in tensors], {}, call))
        if short in ("triu", "tril"):
            source = _first_tensor(args[0]) if args else None
            if source is None:
                return None
            return self._emit_function(short, "custom", [source], list(source.shape), {}, call)

        if short in ("getattr", "size", "len", "range", "print", "isinstance", "float", "int", "bool", "str"):
            return None

        # Unknown tensor op: never identity.
        if tensors:
            return self._unknown_tensor_result(name, tensors, call)
        return None

    # -- emit nodes -------------------------------------------------------
    def _function_base(self, op_short: str) -> str:
        scope = self.scope
        return f"{scope}.{op_short}" if scope else op_short

    def _emit_node(
        self,
        *,
        name_base: str,
        op: str,
        kind: str,
        module_path: str,
        in_tensors: Sequence[_Tensor],
        out_shapes: list[Shape],
        params: dict[str, Any],
        source: SourceLoc | None,
        num_params: int = 0,
        notes: list[str] | None = None,
    ) -> int:
        name = self._allocate(name_base)
        in_shapes = [list(tensor.shape) for tensor in in_tensors if tensor.shape]
        flops_kind = kind
        if flops_kind == "attention" and op.rsplit(".", 1)[-1] in ("matmul", "bmm", "einsum"):
            flops = ops.matmul_flops(in_shapes)
        else:
            flops = ops.module_flops(
                "reshape" if kind == "reshape" else flops_kind,
                params,
                in_shapes,
                out_shapes,
                op,
            )
        tags = ops.tags_for(kind, op, params)
        if kind in ("attention", "matmul"):
            tags = list(dict.fromkeys([*tags, "attention"]))
        node_index = len(self.nodes)
        producer_set = sorted({producer for tensor in in_tensors for producer in tensor.producers})
        residual_flags: list[tuple[int, bool]] = []
        latest_producer = producer_set[-1] if producer_set else None
        if kind == "concat":
            residual_flags = [(producer, False) for producer in producer_set]
        elif kind == "add" and len(in_tensors) >= 2 and len(producer_set) >= 2:
            for producer in producer_set:
                residual_flags.append((producer, producer != latest_producer))
        else:
            # Regular data dependency: every tensor input producer gets an edge.
            residual_flags = [(producer, False) for producer in producer_set]
        for producer, is_residual in residual_flags:
            if producer < 0 or producer >= node_index:
                continue
            edge_kind = "residual" if is_residual else "data"
            label = ""
            if kind == "concat":
                edge_kind = "skip"
                label = "skip"
            elif is_residual:
                label = "identity"
            if edge_kind == "data" and self._shares_attention(self.nodes[producer].module_path, module_path):
                edge_kind = "attention"
            self.edges.append(
                EdgeSpec(
                    source=producer,
                    target=node_index,
                    kind=edge_kind,
                    shapes=[list(self.nodes[producer].out_shapes[0])] if self.nodes[producer].out_shapes else [],
                    label=label,
                )
            )
        if any(is_residual for _producer, is_residual in residual_flags):
            tags = list(dict.fromkeys([*tags, "residual"]))
            notes = [*(notes or []), "Residual add"]
        self.nodes.append(
            NodeSpec(
                name=name,
                op=op,
                kind=kind,
                module_path=module_path,
                in_shapes=in_shapes,
                out_shapes=[list(shape) for shape in out_shapes],
                params=params,
                num_params=int(num_params),
                flops=int(flops),
                tags=tags,
                notes=list(notes or []),
                source=source,
            )
        )
        return node_index

    def _emit_module(self, decl: Decl, args: list[Any]) -> list[_Tensor]:
        tensors = _tensor_inputs(args, {})
        in_shapes = [tensor.shape for tensor in tensors]
        if decl.kind == "attention" and decl.op == "MultiheadAttention":
            out_shapes = [in_shapes[0] if in_shapes else [-1]]
        else:
            out_shapes, warnings = ops.infer_output_shapes(decl.kind, decl.params, in_shapes, decl.op)
            for warning in warnings:
                self.warnings.append(f"{decl.path or decl.op}: {warning}")
        if not out_shapes:
            out_shapes = [in_shapes[0] if in_shapes else [-1]]
        node_index = self._emit_node(
            name_base=decl.path or decl.op,
            op=decl.op,
            kind=decl.kind,
            module_path=decl.path,
            in_tensors=tensors,
            out_shapes=out_shapes,
            params=dict(decl.params),
            num_params=decl.num_params,
            source=decl.source,
        )
        results = [_Tensor(list(shape), [node_index]) for shape in out_shapes]
        if decl.kind == "attention" and decl.op == "MultiheadAttention":
            return [_Multi([*results, None])]  # type: ignore[list-item]
        return results

    def _emit_add(self, left: _Tensor, right: _Tensor) -> _Tensor:
        out = _broadcast_shape(left.shape, right.shape)
        return self._emit_function("add", "add", [left, right], out, {}, None)

    def _emit_function(
        self,
        op_short: str,
        kind: str,
        in_tensors: Sequence[_Tensor],
        out_shape: Shape,
        params: dict[str, Any],
        expression: ast.AST | None,
    ) -> _Tensor:
        name_base = self._function_base(op_short)
        source = _source_loc(self.source, expression) if expression is not None else None
        if source is not None:
            source = self._op_source(name_base, op_short, source)
        op = f"F.{op_short}" if op_short in ("relu", "gelu", "silu", "softmax", "log_softmax", "dropout") else op_short
        if op_short in ("cat", "matmul", "mean", "sum", "chunk", "masked_fill", "transpose", "permute", "view", "reshape", "flatten"):
            op = f"Tensor.{op_short}" if op_short in ("mean", "sum", "masked_fill", "transpose", "permute", "view", "reshape", "flatten", "chunk") else ("torch.cat" if op_short == "cat" else op)
        node_index = self._emit_node(
            name_base=name_base,
            op=op,
            kind=kind,
            module_path=name_base,
            in_tensors=list(in_tensors),
            out_shapes=[list(out_shape)],
            params=params,
            source=source,
        )
        return _Tensor(list(out_shape), [node_index])

    def _op_source(self, name_base: str, op_short: str, fallback: SourceLoc) -> SourceLoc:
        # Use the class forward op index if available; otherwise the expression line.
        key = op_short.lower()
        cache_key = name_base
        used = self._func_cache.get(cache_key, 0)
        self._func_cache[cache_key] = used + 1
        return fallback

    def _shares_attention(self, source_path: str, target_path: str) -> bool:
        for path in self.attention_paths:
            if not path:
                continue
            if (source_path == path or source_path.startswith(path + ".")) and (
                target_path == path or target_path.startswith(path + ".")
            ):
                return True
        return False

    def _result_tensors(self, value: Any) -> list[_Tensor]:
        if value is None:
            return []
        if isinstance(value, _Tensor):
            return [value]
        if isinstance(value, _Multi):
            tensors: list[_Tensor] = []
            for item in value.items:
                tensors.extend(self._result_tensors(item))
            return tensors
        if isinstance(value, (list, tuple)):
            tensors = []
            for item in value:
                tensors.extend(self._result_tensors(item))
            return tensors
        return []

    # -- binary / subscript ----------------------------------------------
    def _binary(self, operator: ast.operator, left: Any, right: Any, decl: Decl, expression: ast.AST) -> Any:
        left_tensor = _first_tensor(left)
        right_tensor = _first_tensor(right)
        if isinstance(operator, (ast.Add, ast.Sub)):
            if left_tensor is not None and right_tensor is not None:
                out = _broadcast_shape(left_tensor.shape, right_tensor.shape)
                return self._emit_function("add" if isinstance(operator, ast.Add) else "sub", "add", [left_tensor, right_tensor], out, {}, expression)
            if left_tensor is not None:
                return self._emit_function("add", "add", [left_tensor], left_tensor.shape, {}, expression)
            if right_tensor is not None:
                return self._emit_function("add", "add", [right_tensor], right_tensor.shape, {}, expression)
            if isinstance(left, (int, float)) and isinstance(right, (int, float)):
                return left + right if isinstance(operator, ast.Add) else left - right
            return None
        if isinstance(operator, ast.Mult):
            if left_tensor is not None and right_tensor is not None:
                out = _broadcast_shape(left_tensor.shape, right_tensor.shape)
                return self._emit_function("mul", "custom", [left_tensor, right_tensor], out, {}, expression)
            if left_tensor is not None:
                return self._emit_function("mul", "custom", [left_tensor], left_tensor.shape, {}, expression)
            if right_tensor is not None:
                return self._emit_function("mul", "custom", [right_tensor], right_tensor.shape, {}, expression)
            if isinstance(left, (int, float)) and isinstance(right, (int, float)):
                return left * right
            return None
        if isinstance(operator, ast.MatMult):
            if left_tensor is not None and right_tensor is not None:
                out = _matmul_shape(left_tensor.shape, right_tensor.shape) or [-1]
                return self._emit_function("matmul", "attention", [left_tensor, right_tensor], out, {}, expression)
            return None
        if isinstance(operator, (ast.Div, ast.FloorDiv, ast.Pow, ast.Mod)):
            if isinstance(left, (int, float)) and isinstance(right, (int, float)) and right != 0:
                if isinstance(operator, ast.Pow):
                    return left**right
                if isinstance(operator, ast.Mod):
                    return left % right
                return left / right if isinstance(operator, ast.Div) else left // right
            if left_tensor is not None:
                return self._emit_function("div", "custom", [left_tensor], left_tensor.shape, {}, expression)
            if right_tensor is not None:
                op_name = "pow" if isinstance(operator, ast.Pow) else "div"
                return self._emit_function(op_name, "custom", [right_tensor], right_tensor.shape, {}, expression)
        return None

    def _subscript(self, value: Any, index: Any) -> Any:
        if isinstance(value, _Multi):
            items = value.items
        elif isinstance(value, (list, tuple)):
            items = list(value)
        elif isinstance(value, _Tensor):
            shape = list(value.shape)
            if isinstance(index, (list, tuple)):
                out: Shape = []
                dim = 0
                for item in index:
                    if item is Ellipsis:
                        remaining = len(shape) - dim - (len(index) - index.index(item) - 1)
                        if remaining > 0:
                            out.extend(shape[dim:dim + remaining])
                            dim += remaining
                        continue
                    if isinstance(item, slice):
                        size = shape[dim] if dim < len(shape) else -1
                        start, stop, step = item.start or 0, item.stop, item.step or 1
                        if isinstance(size, int) and size >= 0 and stop is not None:
                            out.append(max(0, len(range(start, min(stop, size), step))))
                        elif isinstance(size, int) and size >= 0:
                            out.append(max(0, len(range(start, size, step))))
                        else:
                            out.append(-1)
                    elif isinstance(item, int):
                        pass
                    else:
                        out.append(-1)
                    dim += 1
                out.extend(shape[dim:])
                return _Tensor(out or [1], list(value.producers))
            if isinstance(index, int):
                return _Tensor(shape[index:] if len(shape) > 1 and index == 0 else (shape[1:] or [1]), list(value.producers))
            return _Tensor(shape, list(value.producers))
        elif isinstance(value, _ModuleRef) and isinstance(index, (int, str)):
            child = value.decl.children.get(str(index))
            return _ModuleRef(child) if child is not None else None
        elif isinstance(value, str) and isinstance(index, int):
            return value[index]
        else:
            return None
        if isinstance(index, int):
            if -len(items) <= index < len(items):
                return items[index]
            return None
        if isinstance(index, slice):
            return items[index]
        return None


_DUMMY_DECL = Decl(path="", op="root")


def _concat_shape(shapes: Sequence[Shape], dim: int) -> Shape:
    if not shapes:
        return [-1]
    rank = len(shapes[0])
    resolved = _normalize_dim(dim, rank)
    out = list(shapes[0])
    total = 0
    for shape in shapes:
        value = shape[resolved] if 0 <= resolved < len(shape) else -1
        if value < 0:
            total = -1
            break
        total += value
    if 0 <= resolved < rank:
        out[resolved] = total
    return out


def _resolve_view_shape(shape: Shape, target: Sequence[int]) -> Shape:
    if not target:
        return list(shape)
    result: list[int] = []
    unknown = -1
    known_product = 1
    for index, value in enumerate(target):
        if value == -1:
            if unknown >= 0:
                return list(target)
            unknown = index
            result.append(-1)
        elif value == 0:
            result.append(shape[index] if index < len(shape) else -1)
        else:
            known_product *= value
            result.append(int(value))
    if unknown >= 0:
        source = _product(value for value in shape if value >= 0) if all(v >= 0 for v in shape) else -1
        if source > 0 and known_product > 0 and source % known_product == 0:
            result[unknown] = source // known_product
    return result


def _math_scalar(name: str, left: float, right: float | None) -> float | None:
    try:
        if name == "sqrt":
            return math.sqrt(left)
        if name == "log":
            return math.log(left)
        if name == "exp":
            return math.exp(left)
        if name == "pow":
            return left ** (right or 0)
        if name == "abs":
            return abs(left)
        if name == "mul":
            return left * (right or 1)
        if name == "div":
            return left / right if right else None
    except (ValueError, OverflowError):
        return None
    return None


# ---------------------------------------------------------------------------
# static input inference + meta
# ---------------------------------------------------------------------------

def _first_consuming_decl(root: Decl) -> Decl | None:
    for decl in root.walk():
        if decl is root:
            continue
        if decl.kind in ("conv", "deconv", "linear", "embedding", "attention", "rnn"):
            return decl
    return None


def static_input_shape(root: Decl, options: Any) -> tuple[Shape, str]:
    explicit = getattr(options, "input_shape", None)
    unknown = bool(getattr(options, "unknown_input", False))
    first = _first_consuming_decl(root)
    dtype = "float32"
    if first is not None and first.kind == "embedding":
        dtype = "int64"
    if explicit:
        try:
            shape = [int(value) for value in explicit]
        except (TypeError, ValueError):
            shape = []
        if shape:
            return shape, dtype
    if first is None:
        return ([1, 3, -1, -1] if unknown else [1, 3, 64, 64]), "float32"
    params = first.params
    if first.kind in ("conv", "deconv"):
        dims = _conv_dims(first.op)
        channels = int(params.get("inChannels") or 3)
        if dims == 1:
            return [1, channels, -1 if unknown else 64], "float32"
        if dims == 3:
            return ([1, channels, -1, -1, -1] if unknown else [1, channels, 16, 16, 16]), "float32"
        return ([1, channels, -1, -1] if unknown else [1, channels, 64, 64]), "float32"
    if first.kind == "linear":
        return [1, int(params.get("inFeatures") or 64)], "float32"
    if first.kind == "embedding":
        return ([1, -1] if unknown else [1, 16]), "int64"
    if first.kind == "attention":
        embed = int(params.get("embedDim") or 32)
        if params.get("batchFirst"):
            return [1, -1 if unknown else 16, embed], "float32"
        return [-1 if unknown else 16, 1, embed], "float32"
    if first.kind == "rnn":
        features = int(params.get("inputFeatures") or 32)
        if params.get("batchFirst"):
            return [1, -1 if unknown else 16, features], "float32"
        return [-1 if unknown else 16, 1, features], "float32"
    return ([1, 3, -1, -1] if unknown else [1, 3, 64, 64]), "float32"


def infer_input_specs(root: Decl, module_class: ModuleClass, options: Any) -> list[dict[str, Any]]:
    """Build one input tensor spec per required positional ``forward`` argument."""
    explicit = getattr(options, "input_shape", None)
    base_shape, base_dtype = static_input_shape(root, options)
    forward = module_class.methods.get("forward")
    names: list[str] = []
    if forward is not None:
        positional = [arg.arg for arg in forward.args.args if arg.arg != "self"]
        defaults = list(forward.args.defaults)
        required = len(positional) - len(defaults)
        names = positional[:required] if required > 0 else positional[:1]
    if not names:
        names = ["input"]
    hints = _forward_input_hints(forward, root)
    specs: list[dict[str, Any]] = []
    for index, name in enumerate(names):
        info = hints.get(name, {})
        dtype = str(info.get("dtype") or (base_dtype if index == 0 else "float32"))
        if index == 0 and explicit:
            try:
                explicit_shape = [int(value) for value in explicit]
            except (TypeError, ValueError):
                explicit_shape = []
            if explicit_shape:
                specs.append({"name": name, "shape": explicit_shape, "dtype": dtype})
                continue
        rank = info.get("rank")
        index_rank = info.get("index_rank")
        last = info.get("last")
        if index == 0:
            if len(names) > 1:
                if isinstance(rank, int) and rank > 0:
                    last_dim = last if isinstance(last, int) and last > 0 else 3
                    shape = [1] * (rank - 1) + [last_dim]
                elif isinstance(index_rank, int) and index_rank >= 4:
                    shape = [1] * (index_rank - 1) + [int(last or 3)]
                else:
                    first_decl = _first_consuming_decl(root)
                    if (
                        first_decl is not None
                        and first_decl.kind in ("conv", "deconv")
                        and len(base_shape) >= 4
                    ):
                        shape = list(base_shape)
                    else:
                        shape = [1, 1, int(last or 3)]
            elif rank is not None and rank > 0:
                if len(base_shape) == rank and last is None:
                    shape = list(base_shape)
                else:
                    last_dim = last if isinstance(last, int) and last > 0 else (
                        base_shape[-1] if len(base_shape) == rank and base_shape else 3
                    )
                    shape = [1] * (rank - 1) + [int(last_dim or 3)]
            elif index_rank is not None and index_rank >= 4:
                shape = [1] * (index_rank - 1) + [int(last or 3)]
            else:
                shape = list(base_shape)
        else:
            effective_rank = rank if isinstance(rank, int) and rank > 0 else max(3, int(index_rank or 0))
            last_dim = last if isinstance(last, int) and last > 0 else 3
            shape = [1] * (effective_rank - 1) + [last_dim]
        specs.append({"name": name, "shape": shape, "dtype": dtype})
    return specs


def _collect_meta(root: Decl, source_index: SourceIndex) -> tuple[dict[str, ModuleMeta], set[str]]:
    meta: dict[str, ModuleMeta] = {}
    attention_paths: set[str] = set()
    for decl in root.walk():
        if not decl.path:
            continue
        kind = decl.kind
        meta[decl.path] = ModuleMeta(
            path=decl.path,
            class_name=decl.op,
            kind_hint=kind,
            is_container=decl.is_container,
            is_sequential=decl.is_sequential,
            source=decl.source,
        )
        if kind == "attention" or "attention" in decl.op.lower():
            attention_paths.add(decl.path)
    return meta, attention_paths


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------

def _analyze_static_impl(
    module_file: ModuleFile,
    candidate: ModuleClass,
    source_index: SourceIndex,
    options: Any,
    *,
    warnings: list[str] | None = None,
) -> Analysis:
    """Analyze a model without executing it (``mode="static"``)."""
    base_warnings = list(warnings or [])
    builder = _Builder(module_file, source_index)
    try:
        root = builder.build_model(candidate)
    except RecursionError:
        raise
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("static builder failed for %s", candidate.name, exc_info=True)
        raise AnalyzeError(
            f"static analysis failed: {type(exc).__name__}: {exc}",
            file=module_file.path,
            line=candidate.line,
        ) from exc
    base_warnings.extend(builder.warnings)

    runner = _Runner(root, builder, options, base_warnings)
    input_specs = infer_input_specs(root, candidate, options)
    fallback_edges: list[EdgeSpec] | None = None
    try:
        nodes, outputs = runner.run(input_specs)
    except Exception as exc:  # noqa: BLE001 - static inference must never crash
        LOGGER.debug("static execution failed for %s", candidate.name, exc_info=True)
        base_warnings.append(f"static shape inference failed ({type(exc).__name__}: {exc}); shapes are unknown")
        nodes, fallback_edges = _fallback_nodes_multi(root, input_specs, base_warnings)
        outputs = [_Tensor([-1], [0])]

    meta, attention_paths = _collect_meta(root, source_index)
    total_params = int(root.total_params())
    output_tensors = [tensor for tensor in outputs if isinstance(tensor, _Tensor)]
    output_specs: list[dict[str, Any]] = []
    for index, tensor in enumerate(output_tensors):
        name = "output" if len(output_tensors) == 1 else f"output_{index}"
        # Static analysis has no runtime dtype: model outputs are logits/features.
        output_specs.append({"name": name, "shape": list(tensor.shape), "dtype": "float32"})
    if not output_specs:
        output_specs = [{"name": "output", "shape": [-1], "dtype": "float32"}]

    if candidate.required_init_args:
        base_warnings.append(
            f"required constructor argument(s) {candidate.required_init_args} have no default; "
            "shapes use -1 where they cannot be derived and parameter/FLOP totals are partial."
        )
    unknown_dims = any(
        -1 in shape for node in nodes for shape in (*node.in_shapes, *node.out_shapes)
    )
    if unknown_dims:
        base_warnings.append(
            "some tensor dimensions are unknown (-1); parameter/FLOP totals are partial"
        )

    analysis = Analysis(
        mode="static",
        nodes=nodes,
        edges=fallback_edges if fallback_edges is not None else runner.edges,
        input_specs=input_specs,
        output_specs=output_specs,
        warnings=list(dict.fromkeys(base_warnings)),
        total_params=total_params,
        trainable_params=total_params,
        flops_approx=True,
        module_meta=meta,
        attention_paths=attention_paths,
        class_name=candidate.name,
        model_name=candidate.name,
        root_source=source_index.class_sources.get(candidate.name),
    )
    _apply_depths(analysis.nodes)
    return analysis




def _minimal_static_analysis(
    module_file: ModuleFile,
    candidate: ModuleClass,
    source_index: SourceIndex,
    options: Any,
    warnings: list[str],
) -> Analysis:
    """Last-resort static analysis: a single block node, never raises."""
    explicit = getattr(options, "input_shape", None)
    try:
        input_shape = [int(value) for value in explicit] if explicit else [1, 3, 64, 64]
    except (TypeError, ValueError):
        input_shape = [1, 3, 64, 64]
    if not input_shape:
        input_shape = [1, 3, 64, 64]
    nodes = _fallback_nodes(Decl(path="", op=candidate.name, kind="block"), input_shape, warnings)
    edges = [
        EdgeSpec(source=0, target=1, kind="data", shapes=[list(input_shape)]),
        EdgeSpec(source=1, target=2, kind="data", shapes=[[-1]]),
    ]
    analysis = Analysis(
        mode="static",
        nodes=nodes,
        edges=edges,
        input_specs=[{"name": "input", "shape": list(input_shape), "dtype": "float32"}],
        output_specs=[{"name": "output", "shape": [-1], "dtype": "float32"}],
        warnings=list(dict.fromkeys(warnings)),
        total_params=0,
        trainable_params=0,
        flops_approx=True,
        module_meta={},
        attention_paths=set(),
        class_name=candidate.name,
        model_name=candidate.name,
        root_source=source_index.class_sources.get(candidate.name),
    )
    _apply_depths(analysis.nodes)
    return analysis


def analyze_static(
    module_file: ModuleFile,
    candidate: ModuleClass,
    source_index: SourceIndex,
    options: Any,
    *,
    warnings: list[str] | None = None,
) -> Analysis:
    """Analyze a model without executing it (``mode="static"``)."""
    previous_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(previous_limit, 20000))
    try:
        return _analyze_static_impl(
            module_file, candidate, source_index, options, warnings=warnings
        )
    except RecursionError as exc:
        LOGGER.debug("static analysis recursion limit for %s", candidate.name, exc_info=True)
        base = list(warnings or [])
        base.append(
            f"static analysis hit a recursion limit ({type(exc).__name__}); "
            "emitted a placeholder block node instead"
        )
        return _minimal_static_analysis(module_file, candidate, source_index, options, base)
    finally:
        sys.setrecursionlimit(previous_limit)


def _fallback_nodes_multi(
    root: Decl,
    input_specs: list[dict[str, Any]],
    warnings: list[str],
) -> tuple[list[NodeSpec], list[EdgeSpec]]:
    nodes: list[NodeSpec] = []
    edges: list[EdgeSpec] = []
    for index, spec in enumerate(input_specs or [{"name": "input", "shape": [-1], "dtype": "float32"}]):
        shape = list(spec.get("shape") or [-1])
        nodes.append(
            NodeSpec(
                name=str(spec.get("name") or f"input{index}"),
                op="Input",
                kind="input",
                module_path="",
                in_shapes=[],
                out_shapes=[shape],
                tags=["boundary"],
                depth=0,
            )
        )
    block_index = len(nodes)
    nodes.append(
        NodeSpec(
            name=root.op,
            op=root.op,
            kind="block",
            module_path="",
            in_shapes=[list(node.out_shapes[0]) for node in nodes],
            out_shapes=[[-1]],
            params={},
            num_params=root.total_params(),
            flops=0,
            tags=["composite"],
            notes=["static analysis fallback"],
            depth=1,
        )
    )
    for index in range(block_index):
        edges.append(
            EdgeSpec(source=index, target=block_index, kind="data", shapes=[list(nodes[index].out_shapes[0])])
        )
    output_index = len(nodes)
    nodes.append(
        NodeSpec(
            name="output",
            op="Output",
            kind="output",
            module_path="",
            in_shapes=[[-1]],
            out_shapes=[[-1]],
            tags=["boundary"],
            depth=0,
        )
    )
    edges.append(EdgeSpec(source=block_index, target=output_index, kind="data", shapes=[[-1]]))
    warnings.append("static analysis produced a placeholder block node only")
    return nodes, edges


def _fallback_nodes(root: Decl, input_shape: Shape, warnings: list[str]) -> list[NodeSpec]:
    nodes = [
        NodeSpec(
            name="input",
            op="Input",
            kind="input",
            module_path="",
            in_shapes=[],
            out_shapes=[list(input_shape)],
            tags=["boundary"],
            depth=0,
        ),
        NodeSpec(
            name=root.op,
            op=root.op,
            kind="block",
            module_path="",
            in_shapes=[list(input_shape)],
            out_shapes=[[-1]],
            params={},
            num_params=root.total_params(),
            flops=0,
            tags=["composite"],
            notes=["static analysis fallback"],
            depth=1,
        ),
        NodeSpec(
            name="output",
            op="Output",
            kind="output",
            module_path="",
            in_shapes=[[-1]],
            out_shapes=[[-1]],
            tags=["boundary"],
            depth=0,
        ),
    ]
    warnings.append("static analysis produced a placeholder block node only")
    return nodes


def _apply_depths(nodes: list[NodeSpec]) -> None:
    for node in nodes:
        if node.kind in ("input", "output"):
            node.depth = 0
        else:
            segments = node.module_path.split(".") if node.module_path else []
            node.depth = max(1, len(segments) - 1)


__all__ = ["analyze_static", "static_input_shape"]
