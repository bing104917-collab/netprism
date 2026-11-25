"""Dynamic tracing: selective ``torch.fx`` tracing + eager hook fallback.

The preferred path is a :class:`torch.fx.Tracer` subclass that keeps atomic
``torch.nn`` layers as leaves (``call_module`` nodes) while expanding containers,
Transformer wrappers and user-defined ``nn.Module`` classes so their internals
become visible.  A small ``torch.fx.Interpreter`` subclass records the concrete
input/output shapes of every node while executing the graph once on a dummy
tensor, which gives us real shapes, real parameters and real data dependencies
(residual / U-Net skip edges included).

If symbolic tracing fails (data-dependent control flow is the typical reason),
we fall back to eager execution with forward hooks on atomic modules: shapes are
still real, but the dependency graph degrades to execution order.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import logging
import re
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, Iterator, Sequence

from . import ops
from .discover import (
    FACTORY_FUNCTIONS,
    ModelCandidate,
    ModuleFile,
    has_relative_imports,
)
from .errors import AnalyzeError
from .ir import Analysis, EdgeSpec, ModuleMeta, NodeSpec, Shape
from .sources import SourceIndex, SourceLoc
from .synth import SynthResult, synthesize_candidate

LOGGER = logging.getLogger("netprism.core.tracer")

#: A single eager forward slower than this is not worth tracing symbolically:
#: the FX interpreter would run the same computation again (CPU/minutes for big
#: token counts).  Such models fall back to static analysis with a warning.
MAX_FORWARD_SECONDS = 4.0

#: Classes whose forward exceeded the budget once; other factories returning the
#: same class are analyzed statically straight away (e.g. cct_2 ... cct_16).
_SLOW_CLASSES: set[tuple[str, str]] = set()


@contextmanager
def _suppress_user_output() -> Iterator[None]:
    """Hide ``print`` noise from user model code during analysis."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        yield


CONTAINER_CLASS_NAMES = ("Sequential", "ModuleList", "ModuleDict", "ParameterList", "ParameterDict")
TRANSFORMER_COMPOSITES = (
    "TransformerEncoderLayer",
    "TransformerEncoder",
    "TransformerDecoderLayer",
    "TransformerDecoder",
)


# ---------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------

def _torch():
    try:
        import torch  # noqa: F401  (lazy import is intentional)
    except Exception:  # pragma: no cover - torch is a declared optional dep
        return None
    return sys.modules["torch"]


def torch_available() -> bool:
    return _torch() is not None


def _package_import_target(path: Path) -> tuple[str | None, Path | None]:
    """Return ``(dotted_module_name, sys_path_entry)`` for package-relative files."""
    chain = [path.stem]
    parent = path.parent
    while (parent / "__init__.py").is_file():
        chain.insert(0, parent.name)
        parent = parent.parent
    if len(chain) <= 1:
        return None, None
    return ".".join(chain), parent


def _import_dotted_module(dotted: str, path: Path, package_root: Path) -> ModuleType:
    """Import ``dotted`` (relative imports need the real package context)."""
    inserted: list[str] = []
    for entry in (str(package_root.resolve()), str(path.parent.resolve())):
        if entry and entry not in sys.path:
            sys.path.insert(0, entry)
            inserted.append(entry)
    top = dotted.split(".", 1)[0]

    def import_once() -> ModuleType:
        return importlib.import_module(dotted)

    try:
        try:
            module = import_once()
        except Exception:
            # A different repository may have imported a package with the same
            # name earlier in this process: drop it and retry outside the cache.
            for name in list(sys.modules):
                if name == top or name.startswith(top + "."):
                    sys.modules.pop(name, None)
            module = import_once()
    finally:
        for entry in inserted:
            try:
                sys.path.remove(entry)
            except ValueError:
                pass
    module_file = getattr(module, "__file__", None)
    if not module_file or Path(module_file).resolve() != path.resolve():
        raise AnalyzeError(
            f"imported module {dotted!r} does not point at {path}",
            file=path,
        )
    return module


def load_file_module(path: Path, info: ModuleFile | None = None) -> ModuleType:
    """Import a user source file, respecting its package context if needed."""
    torch = _torch()
    if torch is None:
        raise AnalyzeError("PyTorch is not installed; dynamic tracing unavailable", file=path)
    uses_relative = info is not None and has_relative_imports(info.tree)
    dotted, package_root = _package_import_target(path)
    if uses_relative and dotted and package_root is not None:
        try:
            return _import_dotted_module(dotted, path, package_root)
        except AnalyzeError:
            raise
        except Exception as exc:  # noqa: BLE001 - fall back to direct import below
            LOGGER.debug("package import of %s failed: %s", dotted, exc)
    module_name = f"_netprism_dyn_{abs(hash(str(path.resolve()))) & 0xFFFFFFFF:08x}_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise AnalyzeError("cannot create import spec", file=path)
    module = importlib.util.module_from_spec(spec)
    parent = str(path.parent.resolve())
    grandparent = str(path.parent.resolve().parent)
    if dotted and package_root is not None:
        grandparent = str(package_root.resolve())
    inserted: list[str] = []
    for entry in (parent, grandparent):
        if entry and entry not in sys.path:
            sys.path.insert(0, entry)
            inserted.append(entry)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001 - relative imports without package context
        sys.modules.pop(module_name, None)
        # Last resort for package-relative files: dotted import.
        if dotted and package_root is not None:
            try:
                return _import_dotted_module(dotted, path, package_root)
            except Exception as retry_exc:  # noqa: BLE001
                raise AnalyzeError(f"import failed: {retry_exc}", file=path) from retry_exc
        raise AnalyzeError(f"import failed: {exc}", file=path) from exc
    finally:
        for entry in inserted:
            try:
                sys.path.remove(entry)
            except ValueError:
                pass
    return module


def instantiate_candidate(
    module_obj: ModuleType, candidate: ModelCandidate
) -> tuple[Any, str]:
    """Instantiate ``candidate`` (factory function, default ctor or legacy factory)."""
    torch = _torch()
    assert torch is not None
    failures: list[str] = []

    if candidate.factory is not None:
        factory = getattr(module_obj, candidate.factory.name, None)
        if not callable(factory):
            raise AnalyzeError(
                f"factory function {candidate.factory.name!r} not found", file=candidate.file
            )
        try:
            model = factory()
        except Exception as exc:  # noqa: BLE001
            raise AnalyzeError(
                f"factory {candidate.factory.name}() failed: {type(exc).__name__}: {exc}",
                file=candidate.file,
                line=candidate.line,
            ) from exc
        if not isinstance(model, torch.nn.Module):
            raise AnalyzeError(
                f"factory {candidate.factory.name}() did not return an nn.Module",
                file=candidate.file,
                line=candidate.line,
            )
        return model, f"{candidate.factory.name}() [factory for {candidate.class_name}]"

    cls = getattr(module_obj, candidate.class_name, None)
    if candidate.default_constructible and isinstance(cls, type):
        try:
            model = cls()
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{candidate.class_name}(): {type(exc).__name__}: {exc}")
        else:
            if isinstance(model, torch.nn.Module):
                return model, f"{candidate.class_name}()"
            failures.append(f"{candidate.class_name}() did not return an nn.Module")

    for factory_name in FACTORY_FUNCTIONS:
        factory = getattr(module_obj, factory_name, None)
        if not callable(factory):
            continue
        try:
            model = factory()
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{factory_name}(): {type(exc).__name__}: {exc}")
            continue
        if isinstance(model, torch.nn.Module):
            return model, f"{factory_name}()"

    module_model = getattr(module_obj, "model", None)
    if isinstance(module_model, torch.nn.Module):
        return module_model, "module-level 'model'"

    reason = "; ".join(failures[-3:]) if failures else "no default constructor and no factory"
    raise AnalyzeError(
        f"cannot instantiate model {candidate.display_name!r}: {reason}",
        file=candidate.file,
        line=candidate.line,
    )


def instantiate_with_synthesis(
    module_obj: ModuleType,
    candidate: ModelCandidate,
    info: ModuleFile,
) -> tuple[Any, str, SynthResult | None]:
    """Try the plain constructor first, then synthesize required arguments."""
    with _suppress_user_output():
        try:
            model, how = instantiate_candidate(module_obj, candidate)
        except AnalyzeError:
            result = synthesize_candidate(module_obj, candidate, info)
            return result.model, result.description, result
    return model, how, None


def _is_container_module(module: Any) -> bool:
    torch = _torch()
    assert torch is not None
    nn = torch.nn
    return isinstance(module, (nn.Sequential, nn.ModuleList, nn.ModuleDict, nn.ParameterList, nn.ParameterDict))


def is_atomic_module(module: Any) -> bool:
    """Whether ``module`` should stay a leaf ``call_module`` node."""
    torch = _torch()
    assert torch is not None
    nn = torch.nn
    if isinstance(module, (nn.Sequential, nn.ModuleList, nn.ModuleDict, nn.ParameterList, nn.ParameterDict)):
        return False
    classes = tuple(getattr(nn, name) for name in TRANSFORMER_COMPOSITES if hasattr(nn, name))
    if classes and isinstance(module, classes):
        return False
    module_namespace = type(module).__module__ or ""
    if module_namespace.startswith(("torch.nn", "torch.ao.nn")):
        return True
    return False


def _module_paths(model: Any) -> dict[int, str]:
    return {id(module): name for name, module in model.named_modules()}


def _dtype_name(tensor: Any) -> str:
    name = str(getattr(tensor, "dtype", "float32"))
    return name.replace("torch.", "")


def _first_consuming_module(model: Any) -> tuple[str, Any, str | None]:
    best_name, best_module, best_kind = "", model, None
    for name, module in model.named_modules():
        if not name or _is_container_module(module):
            continue
        # Prefer an actual atomic layer: a composite custom module (e.g. UNet's
        # DoubleConv) does not expose in_channels / kernel_size itself.
        try:
            has_children = next(module.children(), None) is not None
        except Exception:
            has_children = False
        if has_children:
            continue
        kind = ops.classify_module(type(module).__name__)
        if kind in ("conv", "deconv", "linear", "embedding", "attention", "rnn"):
            return name, module, kind
    return best_name, best_module, best_kind


def _pair_value(value: Any) -> tuple[int, int] | None:
    """Normalize an int / tuple image-side attribute to ``(h, w)``."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return (value, value) if value > 0 else None
    if isinstance(value, (tuple, list)) and value:
        try:
            first = int(value[0])
            second = int(value[1]) if len(value) > 1 else first
        except (TypeError, ValueError):
            return None
        return (first, second) if first > 0 and second > 0 else None
    return None


def _patch_from_rearrange(model: Any) -> tuple[int, int] | None:
    """Read ``p1``/``p2`` from an ``einops.layers.Rearrange`` patch embedding."""
    for module in model.modules():
        if type(module).__name__ != "Rearrange":
            continue
        axes = getattr(module, "axes_lengths", None) or getattr(module, "_axes_lengths", None)
        values: dict[str, int] = {}
        if isinstance(axes, dict):
            for key, value in axes.items():
                if isinstance(value, int):
                    values[str(key)] = value
        if not values:
            pattern = str(getattr(module, "pattern", "") or "")
            for name in ("p1", "p2"):
                match = re.search(rf"\bp{name[-1]}\s*=\s*(\d+)", pattern)
                if match:
                    values[name] = int(match.group(1))
        if "p1" in values and "p2" in values:
            return values["p1"], values["p2"]
        if "p" in values:
            return values["p"], values["p"]
    return None


def _attribute_image_candidates(model: Any, torch: Any) -> list[tuple[int, int]]:
    """Infer image input size from model attributes / positional-embedding grids.

    ViT-style models often start with an ``einops.Rearrange`` + ``Linear`` patch
    embedding instead of a Conv2d, so the class of the first layer does not tell
    us the spatial size.  They do, however, keep ``patch_size`` (or a Rearrange
    ``p1``/``p2``) and a ``pos_embedding`` tensor whose row count is the square
    of the patch grid.
    """
    patch = None
    for name in ("patch_size", "patch", "patch_dim", "p"):
        patch = _pair_value(getattr(model, name, None))
        if patch is not None:
            break
    if patch is None:
        patch = _patch_from_rearrange(model)
    channels = None
    for name in ("channels", "in_chans", "in_channels", "n_channels"):
        value = getattr(model, name, None)
        if isinstance(value, int) and value > 0:
            channels = value
            break
    if channels is None:
        for module in model.modules():
            module_channels = getattr(module, "in_channels", None)
            if isinstance(module_channels, int) and module_channels > 0:
                channels = module_channels
                break
    channels = channels or 3

    spatial: list[int] = []
    for name in ("image_size", "img_size", "image_height", "image_width", "input_size"):
        pair = _pair_value(getattr(model, name, None))
        if pair is not None:
            spatial.extend([pair[0], pair[1]])

    tensors: list[Any] = []
    try:
        for _name, buffer in model.named_buffers():
            tensors.append(buffer)
        for _name, parameter in model.named_parameters():
            tensors.append(parameter)
    except Exception:  # pragma: no cover - defensive
        pass
    for value in list(vars(model).values()):
        if isinstance(value, torch.Tensor):
            tensors.append(value)
    for tensor in tensors:
        try:
            shape = tuple(tensor.shape)
        except Exception:
            continue
        if len(shape) == 2:
            token_counts = [shape[0]]
        elif len(shape) == 3 and shape[0] == 1:
            token_counts = [shape[1], shape[1] - 1]
        else:
            continue
        if patch is None:
            continue
        for count in token_counts:
            if count <= 1:
                continue
            grid = int(round(count**0.5))
            if grid * grid != count:
                continue
            spatial.append(grid * patch[0])

    unique: list[int] = []
    for size in spatial:
        if 8 <= size <= 1024 and size not in unique:
            unique.append(size)
    return [(size, channels) for size in unique[:3]]


def dummy_input_candidates(model: Any, options: Any) -> list[tuple[tuple[Any, ...], str]]:
    """Ordered dummy input guesses (SPEC 4.3.2), as ``(args, label)`` pairs."""
    torch = _torch()
    assert torch is not None
    candidates: list[tuple[tuple[Any, ...], str]] = []
    seen: set[tuple[Any, ...]] = set()

    def add(shape: tuple[int, ...], dtype: str, label: str) -> None:
        key = (tuple(int(v) for v in shape), dtype)
        if key in seen:
            return
        seen.add(key)
        tensor_dtype = torch.int64 if dtype == "int64" else torch.float32
        tensor = torch.zeros(*shape, dtype=tensor_dtype)
        candidates.append(((tensor,), label))

    explicit = getattr(options, "input_shape", None)
    _, first_module, first_kind = _first_consuming_module(model)
    explicit_dtype = "int64" if first_kind == "embedding" else "float32"

    if explicit:
        try:
            shape = tuple(int(v) for v in explicit)
        except (TypeError, ValueError):
            shape = ()
        if shape:
            add(shape, explicit_dtype, f"options.input_shape={shape}")

    # Prefer image sizes recorded on the model (ViT-style patch embeddings).
    for size, channels in _attribute_image_candidates(model, torch):
        add((1, channels, size, size), "float32", f"model attribute dummy ({size}x{size})")

    if first_kind in ("conv", "deconv"):
        channels = int(getattr(first_module, "in_channels", 3) or 3)
        kernel = getattr(first_module, "kernel_size", (3, 3))
        spatial_dims = len(kernel) if isinstance(kernel, (tuple, list)) else 1
        if spatial_dims == 1:
            for length in (64, 32, 128):
                add((1, channels, length), "float32", f"Conv1d dummy ({length})")
        elif spatial_dims == 3:
            add((1, channels, 16, 16, 16), "float32", "Conv3d dummy (16^3)")
        else:
            for size in (64, 32, 128, 224):
                add((1, channels, size, size), "float32", f"Conv2d dummy ({size}x{size})")
    elif first_kind == "linear":
        in_features = int(getattr(first_module, "in_features", 0) or 0)
        if in_features:
            add((1, in_features), "float32", f"Linear dummy ({in_features})")
    elif first_kind == "embedding":
        for seq_len in (16, 8, 32):
            add((1, seq_len), "int64", f"Embedding dummy (seq={seq_len})")
    elif first_kind == "attention":
        embed_dim = int(getattr(first_module, "embed_dim", 0) or 0)
        batch_first = bool(getattr(first_module, "batch_first", False))
        if embed_dim:
            for seq_len in (16, 8, 32):
                if batch_first:
                    add((1, seq_len, embed_dim), "float32", f"MHA batch_first dummy (seq={seq_len})")
                else:
                    add((seq_len, 1, embed_dim), "float32", f"MHA dummy (seq={seq_len})")
    elif first_kind == "rnn":
        input_size = int(getattr(first_module, "input_size", 0) or 0)
        batch_first = bool(getattr(first_module, "batch_first", False))
        if input_size:
            for seq_len in (16, 8):
                shape = (1, seq_len, input_size) if batch_first else (seq_len, 1, input_size)
                add(shape, "float32", f"RNN dummy (seq={seq_len})")

    add((1, 3, 64, 64), "float32", "generic dummy (1,3,64,64)")
    add((1, 64), "float32", "generic dummy (1,64)")
    return candidates


# ---------------------------------------------------------------------------
# FX tracing
# ---------------------------------------------------------------------------

@contextmanager
def transformer_trace_mode() -> Iterator[None]:
    """Disable the MHA fast path while tracing Transformer wrapper modules.

    ``nn.TransformerEncoderLayer.forward`` branches on ``src.dim()`` which cannot
    be evaluated symbolically; disabling the fast path short-circuits that check
    and lets us expand the layer into MHA + FFN + LayerNorm call nodes.
    """
    torch = _torch()
    restores: list[Any] = []
    assert torch is not None
    try:
        mha_backend = getattr(getattr(torch, "backends", None), "mha", None)
        if mha_backend is not None and hasattr(mha_backend, "set_fastpath_enabled") and hasattr(
            mha_backend, "get_fastpath_enabled"
        ):
            previous = mha_backend.get_fastpath_enabled()
            mha_backend.set_fastpath_enabled(False)
            restores.append(lambda: mha_backend.set_fastpath_enabled(previous))
        import torch.nn.modules.transformer as transformer_module

        previous_seq_len = getattr(transformer_module, "_get_seq_len", None)
        if previous_seq_len is not None:
            transformer_module._get_seq_len = lambda src, batch_first: None  # type: ignore[assignment]
            restores.append(
                lambda: setattr(transformer_module, "_get_seq_len", previous_seq_len)
            )
    except Exception:  # pragma: no cover - best effort only
        LOGGER.debug("could not patch transformer tracing helpers", exc_info=True)
    try:
        yield
    finally:
        for restore in reversed(restores):
            try:
                restore()
            except Exception:  # pragma: no cover
                pass


class PrismTracer:
    """Factory wrapper so torch is imported lazily inside :func:`trace_fx`."""

    @staticmethod
    def build() -> Any:
        torch = _torch()
        assert torch is not None
        from torch.fx import Tracer

        class _Tracer(Tracer):
            def is_leaf_module(self, module: Any, module_qualified_name: str) -> bool:
                return is_atomic_module(module)

        return _Tracer()


@dataclass
class _NodeRecord:
    in_shapes: list[Shape] = field(default_factory=list)
    out_shapes: list[Shape] = field(default_factory=list)
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)
    result: Any = None


class ShapeInterpreter:
    """Factory for the ``torch.fx.Interpreter`` subclass that records shapes."""

    @staticmethod
    def build() -> Any:
        torch = _torch()
        assert torch is not None
        from torch.fx import Interpreter

        class _Interpreter(Interpreter):
            def __init__(self, module: Any) -> None:
                super().__init__(module)
                self.records: dict[str, _NodeRecord] = {}

            def run_node(self, node: Any) -> Any:  # noqa: D401 - torch.fx API
                try:
                    args, kwargs = self.fetch_args_kwargs_from_env(node)
                except Exception:  # pragma: no cover - defensive
                    args, kwargs = (), {}
                in_shapes = ops.shape_of_value(args) + ops.shape_of_value(kwargs)
                result = super().run_node(node)
                out_shapes = ops.shape_of_value(result)
                self.records[node.name] = _NodeRecord(
                    in_shapes=in_shapes,
                    out_shapes=out_shapes,
                    args=args,
                    kwargs=kwargs,
                    result=result,
                )
                return result

        return _Interpreter


def _collect_refs(torch: Any, value: Any, out: list[Any]) -> None:
    if isinstance(value, torch.fx.Node):
        out.append(value)
        return
    if isinstance(value, dict):
        for item in value.values():
            _collect_refs(torch, item, out)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _collect_refs(torch, item, out)
        return
    slice_type = getattr(torch, "Tensor", None)
    if slice_type is not None and isinstance(value, slice):
        _collect_refs(torch, value.start, out)
        _collect_refs(torch, value.stop, out)
        _collect_refs(torch, value.step, out)


def _function_params(op_short: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    try:
        if op_short in ("view", "reshape"):
            dims: list[int] = []
            for arg in args[1:]:
                if isinstance(arg, int) and not isinstance(arg, bool):
                    dims.append(arg)
                elif isinstance(arg, (tuple, list)) and all(
                    isinstance(value, int) and not isinstance(value, bool) for value in arg
                ):
                    dims.extend(int(value) for value in arg)
            if dims:
                params["shape"] = dims
        elif op_short in ("flatten", "unflatten"):
            if len(args) > 1 and isinstance(args[1], int):
                params["startDim"] = int(args[1])
            if len(args) > 2 and isinstance(args[2], int):
                params["endDim"] = int(args[2])
        elif op_short in ("permute",):
            dims = args[1:] if len(args) > 1 else ()
            if dims and all(isinstance(value, int) for value in dims):
                params["dims"] = [int(value) for value in dims]
        elif op_short in ("transpose", "swapdims", "swapaxes"):
            if len(args) >= 3 and isinstance(args[1], int) and isinstance(args[2], int):
                params["dims"] = [int(args[1]), int(args[2])]
        elif op_short in ("cat", "concat", "stack"):
            dim = kwargs.get("dim", args[1] if len(args) > 1 else None)
            if isinstance(dim, int) and not isinstance(dim, bool):
                params["dim"] = int(dim)
        elif op_short in ("mean", "sum", "amax", "amin", "max", "min", "prod"):
            dim = kwargs.get("dim", args[1] if len(args) > 1 else None)
            if isinstance(dim, int) and not isinstance(dim, bool):
                params["dim"] = int(dim)
            elif isinstance(dim, (tuple, list)):
                params["dim"] = [int(value) for value in dim]
            if kwargs.get("keepdim"):
                params["keepdim"] = True
        elif op_short in ("softmax", "log_softmax"):
            dim = kwargs.get("dim", args[1] if len(args) > 1 else None)
            if isinstance(dim, int) and not isinstance(dim, bool):
                params["dim"] = int(dim)
        elif op_short == "chunk":
            if len(args) > 1 and isinstance(args[1], int):
                params["chunks"] = int(args[1])
            dim = kwargs.get("dim", args[2] if len(args) > 2 else None)
            if isinstance(dim, int):
                params["dim"] = int(dim)
        elif op_short == "split":
            if len(args) > 1:
                split = args[1]
                if isinstance(split, int):
                    params["splitSize"] = int(split)
                elif isinstance(split, (tuple, list)) and all(isinstance(v, int) for v in split):
                    params["splitSizes"] = [int(v) for v in split]
            dim = kwargs.get("dim", args[2] if len(args) > 2 else None)
            if isinstance(dim, int):
                params["dim"] = int(dim)
        elif op_short in ("scaled_dot_product_attention",):
            if len(args) > 3 and args[3] is not None:
                params["attnMask"] = True
            if kwargs.get("is_causal"):
                params["isCausal"] = True
    except Exception:  # pragma: no cover - hyper-parameters are best effort
        return params
    return params


def _function_flops(
    op_short: str, kind: str, in_shapes: Sequence[Shape], out_shapes: Sequence[Shape]
) -> int:
    out_numel = ops.shape_numel(out_shapes[0]) if out_shapes else None
    in_numel = ops.shape_numel(in_shapes[0]) if in_shapes else None
    if op_short in ("matmul", "bmm", "einsum"):
        return ops.matmul_flops(in_shapes)
    if op_short == "scaled_dot_product_attention":
        return 0
    if kind == "activation":
        return int(out_numel or 0)
    if kind == "softmax":
        return int(4 * (out_numel or 0))
    if kind == "add":
        return int(out_numel or 0)
    if kind == "concat":
        return 0
    if kind in ("reshape", "flatten", "dropout"):
        return 0
    if kind == "pool":
        return int(in_numel or 0)
    if kind == "attention":
        return int(out_numel or 0)
    if kind == "custom":
        return int(out_numel or 0)
    return 0


def _node_scope_hint(fx_node: Any) -> str | None:
    """Innermost module scope recorded by FX in ``nn_module_stack`` metadata.

    ``torch.fx`` annotates every node created while tracing inside a module with
    the module stack, e.g. ``blocks.0.attn`` for a matmul inside a custom
    attention block.  This is far more precise than inferring the scope from the
    graph neighbourhood, so it is preferred whenever available.
    """
    meta = getattr(fx_node, "meta", None) or {}
    stack = meta.get("nn_module_stack")
    if not stack:
        return None
    try:
        keys = list(stack.keys())
    except (AttributeError, TypeError):
        return None
    if not keys:
        return None
    key = keys[-1]
    return key if isinstance(key, str) else None


def _name_allocator() -> Any:
    counts: dict[str, int] = {}

    def allocate(base: str) -> str:
        count = counts.get(base, 0) + 1
        counts[base] = count
        return base if count == 1 else f"{base}#{count}"

    return allocate


def _regex_parent(path: str) -> str:
    return path.rsplit(".", 1)[0] if "." in path else ""


def _longest_common_prefix(paths: Iterable[str]) -> str:
    segments = [path.split(".") for path in paths if path]
    if not segments:
        return ""
    prefix = segments[0]
    for candidate in segments[1:]:
        limit = min(len(prefix), len(candidate))
        index = 0
        while index < limit and prefix[index] == candidate[index]:
            index += 1
        prefix = prefix[:index]
        if not prefix:
            return ""
    return ".".join(prefix)


def _infer_scope(candidates: Iterable[str]) -> str:
    """Pick the most plausible module scope from neighbouring module parents."""
    values = [value for value in candidates if value is not None]
    if not values:
        return ""
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    if "" in counts:
        others = [value for value in counts if value]
        if len(others) <= 1:
            return ""
        ancestors = [
            value
            for value in others
            if all(other == value or other.startswith(value + ".") for other in others)
        ]
        if ancestors:
            return max(ancestors, key=lambda value: value.count("."))
        return ""
    best = max(counts.values())
    winners = [value for value, count in counts.items() if count == best]
    if len(winners) == 1:
        return winners[0]
    return _longest_common_prefix(winners)


def _nearest_module_paths(
    start: int,
    neighbours: dict[int, list[int]],
    kept: list["_KeptNode"],
) -> list[str]:
    found: list[str] = []
    seen = {start}
    frontier = [start]
    while frontier:
        nxt: list[int] = []
        for current in frontier:
            for neighbour in neighbours.get(current, []):
                if neighbour in seen:
                    continue
                seen.add(neighbour)
                node = kept[neighbour]
                if node.is_module:
                    if node.spec.module_path:
                        found.append(node.spec.module_path)
                else:
                    nxt.append(neighbour)
        frontier = nxt
    return found


@dataclass
class _KeptNode:
    fx_name: str
    spec: NodeSpec
    is_module: bool = False
    is_boundary: bool = False
    ancestors: list[int] = field(default_factory=list)
    scope_hint: str | None = None


def _build_graph_analysis(
    graph: Any,
    records: dict[str, _NodeRecord],
    model: Any,
    final_result: Any,
    dummy_args: tuple[Any, ...],
    module_file: ModuleFile,
    candidate: ModelCandidate,
    source_index: SourceIndex,
    model_meta: dict[str, ModuleMeta],
    attention_paths: set[str],
    torch: Any,
) -> Analysis:
    warnings: list[str] = []
    kept: list[_KeptNode] = []
    kept_by_fx: dict[str, int] = {}
    allocate = _name_allocator()

    # --- pass 1: keep the interesting nodes ------------------------------
    for fx_node in graph.nodes:
        record = records.get(fx_node.name, _NodeRecord())
        op = str(fx_node.op)

        if op == "placeholder":
            raw_name = fx_node.target if isinstance(fx_node.target, str) else "input"
            spec = NodeSpec(
                name=raw_name or "input",
                op="Input",
                kind="input",
                module_path="",
                in_shapes=[],
                out_shapes=record.out_shapes or [[-1]],
                params={},
                tags=["boundary"],
            )
            kept.append(_KeptNode(fx_name=fx_node.name, spec=spec, is_boundary=True))
            kept_by_fx[fx_node.name] = len(kept) - 1
            continue

        if op == "output":
            spec = NodeSpec(
                name="output",
                op="Output",
                kind="output",
                module_path="",
                in_shapes=record.in_shapes,
                out_shapes=record.out_shapes or [[-1]],
                params={},
                tags=["boundary"],
            )
            kept.append(_KeptNode(fx_name=fx_node.name, spec=spec, is_boundary=True))
            kept_by_fx[fx_node.name] = len(kept) - 1
            continue

        if op == "get_attr":
            continue

        if op == "call_module":
            target = fx_node.target
            if not isinstance(target, str):
                continue
            try:
                module = model.get_submodule(target)
            except Exception:
                module = None
            cls_name = type(module).__name__ if module is not None else str(target)
            kind = ops.classify_module(cls_name)
            params = ops.extract_module_params(module, kind) if module is not None else {}
            in_shapes = record.in_shapes
            out_shapes = record.out_shapes
            if not out_shapes:
                # Modules returning None (in-place ops) are not visualizable.
                continue
            num_params = sum(int(p.numel()) for p in module.parameters()) if module is not None else 0
            flops = ops.module_flops(kind, params, in_shapes, out_shapes, cls_name)
            tags = ops.tags_for(kind, cls_name, params)
            notes: list[str] = []
            if ops.flops_is_approx(kind):
                notes.append("FLOPs estimated")
            source = source_index.module_source(target, model)
            spec = NodeSpec(
                name=allocate(target),
                op=cls_name,
                kind=kind,
                module_path=target,
                in_shapes=in_shapes,
                out_shapes=out_shapes,
                params=params,
                num_params=num_params,
                flops=flops,
                tags=tags,
                notes=notes,
                source=source,
            )
            kept.append(_KeptNode(fx_name=fx_node.name, spec=spec, is_module=True))
            kept_by_fx[fx_node.name] = len(kept) - 1
            continue

        if op not in ("call_function", "call_method"):
            continue

        if op == "call_function":
            op_name = ops.function_op_name(fx_node.target)
        else:
            op_name = str(fx_node.target)
        op_short = op_name.rsplit(".", 1)[-1]
        kind = ops.classify_function(op_name)
        if kind is None:
            continue
        in_shapes = record.in_shapes
        out_shapes = record.out_shapes
        if not out_shapes:
            continue
        params = _function_params(op_short, record.args, record.kwargs)
        flops = _function_flops(op_short, kind, in_shapes, out_shapes)
        tags = ops.tags_for(kind, op_name, params)
        spec = NodeSpec(
            name=op_short,  # scope is inferred later
            op=op_name,
            kind=kind,
            module_path="",
            in_shapes=in_shapes,
            out_shapes=out_shapes,
            params=params,
            num_params=0,
            flops=flops,
            tags=tags,
        )
        scope_hint = _node_scope_hint(fx_node)
        kept.append(_KeptNode(fx_name=fx_node.name, spec=spec, scope_hint=scope_hint))
        kept_by_fx[fx_node.name] = len(kept) - 1

    if not kept:
        raise AnalyzeError("FX graph contained no visualizable nodes", file=module_file.path)

    # --- pass 2: data dependencies (resolve skipped nodes) ---------------
    memo: dict[str, list[int]] = {}
    fx_nodes_by_name = {node.name: node for node in graph.nodes}

    def resolve(fx_node: Any) -> list[int]:
        if fx_node.name in kept_by_fx:
            return [kept_by_fx[fx_node.name]]
        if fx_node.name in memo:
            return memo[fx_node.name]
        refs: list[Any] = []
        _collect_refs(torch, fx_node.args, refs)
        _collect_refs(torch, fx_node.kwargs, refs)
        resolved: list[int] = []
        for ref in refs:
            for index in resolve(ref):
                if index not in resolved:
                    resolved.append(index)
        memo[fx_node.name] = resolved
        return resolved

    for position, entry in enumerate(kept):
        if entry.is_boundary and entry.spec.kind == "input":
            continue
        fx_node = fx_nodes_by_name.get(entry.fx_name)
        if fx_node is None:
            continue
        refs: list[Any] = []
        _collect_refs(torch, fx_node.args, refs)
        _collect_refs(torch, fx_node.kwargs, refs)
        resolved: list[int] = []
        for ref in refs:
            for index in resolve(ref):
                if index != position and index not in resolved:
                    resolved.append(index)
        entry.ancestors = resolved

    upstream: dict[int, list[int]] = {index: [] for index in range(len(kept))}
    downstream: dict[int, list[int]] = {index: [] for index in range(len(kept))}
    for index, entry in enumerate(kept):
        for ancestor in entry.ancestors:
            upstream[index].append(ancestor)
            downstream[ancestor].append(index)

    # --- pass 3: infer the module scope of functional nodes --------------
    allocate_scope = _name_allocator()
    for index, entry in enumerate(kept):
        if entry.is_module or entry.is_boundary:
            continue
        if entry.scope_hint is not None:
            scope = entry.scope_hint
        else:
            candidates: list[str] = []
            for path in _nearest_module_paths(index, upstream, kept):
                candidates.append(_regex_parent(path))
            for path in _nearest_module_paths(index, downstream, kept):
                candidates.append(_regex_parent(path))
            scope = _infer_scope(candidates)
        base = f"{scope}.{entry.spec.name}" if scope else entry.spec.name
        entry.spec.module_path = base
        entry.spec.name = allocate_scope(base)
        if entry.spec.module_path == "":
            entry.spec.module_path = entry.spec.name

    # --- pass 4: function sources with per-scope occurrence counters -----
    occurrences: dict[tuple[str, str], int] = {}
    for entry in kept:
        if entry.is_module or entry.is_boundary or entry.spec.source is not None:
            continue
        scope = _regex_parent(entry.spec.module_path) if "." in entry.spec.module_path else ""
        op_key = entry.spec.op.rsplit(".", 1)[-1]
        key = (scope, op_key)
        occurrence = occurrences.get(key, 0)
        occurrences[key] = occurrence + 1
        source = source_index.function_source(scope, model, op_key, occurrence)
        if source is not None:
            entry.spec.source = source

    # --- pass 5: edges -----------------------------------------------------
    edges: list[EdgeSpec] = []
    for index, entry in enumerate(kept):
        if entry.is_boundary and entry.spec.kind == "input":
            continue
        entry.spec.tags = list(dict.fromkeys(entry.spec.tags))
        sources = sorted(entry.ancestors)
        if not sources and index > 0:
            sources = [index - 1]
            warnings.append(f"node {entry.spec.name!r} has no tensor input; linked sequentially")
        if not sources:
            continue
        latest = max(sources)
        has_residual = False
        for source_index_value in sources:
            source_spec = kept[source_index_value].spec
            target_spec = entry.spec
            edge_kind = "data"
            label = ""
            if target_spec.kind == "concat":
                # The latest producer is the main decoder path (data); only a
                # shape-compatible earlier branch is a real skip.  Marking every
                # concat input as skip made the upsampling path look like a skip
                # connection (same rule as the static parser).
                if source_index_value != latest and _concat_skip_compatible(
                    source_spec, target_spec
                ):
                    edge_kind = "skip"
                    label = "skip"
            elif target_spec.kind == "add" and len(sources) >= 2:
                if source_index_value != latest:
                    edge_kind = "residual"
                    label = "identity"
                    has_residual = True
            if edge_kind == "data" and _shares_attention_path(
                source_spec.module_path, target_spec.module_path, attention_paths
            ):
                edge_kind = "attention"
            edges.append(
                EdgeSpec(
                    source=source_index_value,
                    target=index,
                    kind=edge_kind,
                    shapes=[list(shape) for shape in source_spec.out_shapes],
                    label=label,
                )
            )
        if has_residual and "residual" not in entry.spec.tags:
            entry.spec.tags.append("residual")
            if "Residual add" not in entry.spec.notes:
                entry.spec.notes.append("Residual add")

    # --- pass 6: model level metadata ------------------------------------
    inputs: list[dict] = []
    input_tensors = [item for item in dummy_args if hasattr(item, "shape") or hasattr(item, "dtype")]
    input_index = 0
    for entry in kept:
        if entry.spec.kind != "input":
            continue
        shape = entry.spec.out_shapes[0] if entry.spec.out_shapes else [-1]
        dtype = "float32"
        if input_index < len(input_tensors):
            dtype = _dtype_name(input_tensors[input_index])
        input_index += 1
        inputs.append({"name": entry.spec.name, "shape": shape, "dtype": dtype})
    if not inputs:
        inputs.append({"name": "input", "shape": [-1], "dtype": "float32"})

    output_shapes = ops.shape_of_value(final_result)
    if not output_shapes:
        for entry in reversed(kept):
            if entry.spec.kind == "output" and entry.spec.out_shapes:
                output_shapes = entry.spec.out_shapes
                break
    if not output_shapes:
        output_shapes = [[-1]]
    output_tensors = _tensors_of(final_result, torch)
    outputs = []
    for position, shape in enumerate(output_shapes):
        dtype = _dtype_name(output_tensors[position]) if position < len(output_tensors) else "float32"
        name = "output" if len(output_shapes) == 1 else f"output_{position}"
        outputs.append({"name": name, "shape": list(shape), "dtype": dtype})

    total_params = sum(int(parameter.numel()) for parameter in model.parameters())
    trainable_params = sum(
        int(parameter.numel()) for parameter in model.parameters() if parameter.requires_grad
    )
    flops_approx = any(
        entry.spec.kind in ("attention", "rnn", "block", "custom", "unknown")
        or entry.spec.op.rsplit(".", 1)[-1] in ("matmul", "bmm", "einsum", "scaled_dot_product_attention")
        for entry in kept
    )

    nodes = [entry.spec for entry in kept]
    _apply_depths(nodes)
    return Analysis(
        mode="trace",
        nodes=nodes,
        edges=edges,
        input_specs=inputs,
        output_specs=outputs,
        warnings=_dedupe(warnings),
        total_params=total_params,
        trainable_params=trainable_params,
        flops_approx=flops_approx,
        module_meta=model_meta,
        attention_paths=attention_paths,
        class_name=candidate.class_name,
        model_name=candidate.display_name,
        root_source=source_index.class_sources.get(candidate.class_name),
    )


def _apply_depths(nodes: list[NodeSpec]) -> None:
    """Node depth follows the module-path hierarchy (boundaries are depth 0)."""
    for node in nodes:
        if node.kind in ("input", "output"):
            node.depth = 0
        else:
            segments = node.module_path.split(".") if node.module_path else []
            node.depth = max(1, len(segments) - 1)


def _tensors_of(value: Any, torch: Any, depth: int = 0) -> list[Any]:
    if depth > 4:
        return []
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (list, tuple)):
        result: list[Any] = []
        for item in value:
            result.extend(_tensors_of(item, torch, depth + 1))
        return result
    if isinstance(value, dict):
        result = []
        for item in value.values():
            result.extend(_tensors_of(item, torch, depth + 1))
        return result
    return []


def _concat_skip_compatible(source: NodeSpec, target: NodeSpec) -> bool:
    """A concat branch is a real skip only when its non-concat dims match.

    Mirrors the static parser's rule (:meth:`StaticWalker._concat_skip_compatible`):
    the branch coming from the most recent node is the main decoder path, and the
    other branch only counts as a skip when it lines up with the concat output.
    """
    source_shapes = source.out_shapes or []
    target_shapes = target.out_shapes or []
    if not source_shapes or not target_shapes:
        return False
    dim = target.params.get("dim", 1)
    if not isinstance(dim, int) or isinstance(dim, bool):
        return False
    source_shape = source_shapes[0]
    output = target_shapes[0]
    if len(source_shape) != len(output):
        return False
    normalized = dim if dim >= 0 else len(output) + dim
    if normalized < 0 or normalized >= len(output):
        return False
    for position, (left, right) in enumerate(zip(source_shape, output)):
        if position == normalized:
            continue
        if left != right and left != -1 and right != -1:
            return False
    return True


def _shares_attention_path(source: str, target: str, attention_paths: set[str]) -> bool:
    for path in attention_paths:
        if not path:
            continue
        if (source == path or source.startswith(path + ".")) and (
            target == path or target.startswith(path + ".")
        ):
            return True
    return False


def _dedupe(items: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(item for item in items if item))


def collect_module_meta(
    model: Any, source_index: SourceIndex, torch: Any
) -> tuple[dict[str, ModuleMeta], set[str]]:
    nn = torch.nn
    meta: dict[str, ModuleMeta] = {}
    for name, module in model.named_modules():
        if not name:
            continue
        class_name = type(module).__name__
        kind = ops.classify_module(class_name)
        is_container = isinstance(
            module, (nn.Sequential, nn.ModuleList, nn.ModuleDict, nn.ParameterList, nn.ParameterDict)
        )
        meta[name] = ModuleMeta(
            path=name,
            class_name=class_name,
            kind_hint=kind,
            is_container=is_container,
            is_sequential=isinstance(module, nn.Sequential),
            source=source_index.module_source(name, model),
        )
    attention_paths = {
        path
        for path, item in meta.items()
        if item.kind_hint == "attention"
        or "attention" in item.class_name.lower()
        or item.class_name.lower().endswith("attn")
    }
    return meta, attention_paths


def _trace_fx(
    model: Any,
    module_file: ModuleFile,
    candidate: ModelCandidate,
    dummy_args: tuple[Any, ...],
    label: str,
    source_index: SourceIndex,
    torch: Any,
) -> Analysis:
    tracer = PrismTracer.build()
    with transformer_trace_mode():
        graph = tracer.trace(model)
    graph_module = torch.fx.GraphModule(model, graph)
    interpreter_cls = ShapeInterpreter.build()
    interpreter = interpreter_cls(graph_module)
    with torch.no_grad():
        final_result = interpreter.run(*dummy_args)
    model_meta, attention_paths = collect_module_meta(model, source_index, torch)
    analysis = _build_graph_analysis(
        graph,
        interpreter.records,
        model,
        final_result,
        dummy_args,
        module_file,
        candidate,
        source_index,
        model_meta,
        attention_paths,
        torch,
    )
    if label:
        analysis.warnings = _dedupe([*analysis.warnings])
    return analysis


# ---------------------------------------------------------------------------
# eager hook fallback
# ---------------------------------------------------------------------------

def _hook_target(module: Any) -> bool:
    if is_atomic_module(module):
        return True
    torch = _torch()
    assert torch is not None
    try:
        has_children = next(module.children(), None) is not None
    except Exception:
        has_children = True
    return not has_children and not isinstance(module, torch.nn.ModuleList)


def _trace_hooks(
    model: Any,
    module_file: ModuleFile,
    candidate: ModelCandidate,
    dummy_args: tuple[Any, ...],
    dummy_label: str,
    source_index: SourceIndex,
    fx_error: BaseException | None,
    torch: Any,
) -> Analysis:
    calls: list[dict[str, Any]] = []
    hooks: list[Any] = []

    def make_hook(name: str) -> Any:
        def hook(module: Any, args: tuple[Any, ...], output: Any) -> None:
            calls.append(
                {
                    "module": module,
                    "path": name,
                    "in_shapes": ops.shape_of_value(args),
                    "out_shapes": ops.shape_of_value(output),
                }
            )

        return hook

    try:
        for name, module in model.named_modules():
            if not name or not _hook_target(module):
                continue
            hooks.append(module.register_forward_hook(make_hook(name)))
        with torch.no_grad():
            final_result = model(*dummy_args)
    finally:
        for handle in hooks:
            handle.remove()

    warnings: list[str] = []
    reason = f"{type(fx_error).__name__}: {fx_error}" if fx_error is not None else "unknown error"
    warnings.append(
        "FX symbolic tracing failed ("
        + reason.splitlines()[0][:200]
        + "); fell back to eager hook tracing: nodes follow execution order, "
        "residual/skip links may be missing."
    )
    warnings.append("Functional ops (flatten/cat/relu in forward) are not captured by the hook fallback.")

    allocate = _name_allocator()
    nodes: list[NodeSpec] = []
    edges: list[EdgeSpec] = []

    input_shapes: list[Shape] = []
    for item in dummy_args:
        if hasattr(item, "shape"):
            shape = ops.normalize_shape(tuple(item.shape))
            if shape:
                input_shapes.append(shape)
    if not input_shapes:
        input_shapes = [[-1]]
    input_name = "input"
    inputs = [
        {
            "name": input_name,
            "shape": input_shapes[0],
            "dtype": _dtype_name(dummy_args[0]) if dummy_args else "float32",
        }
    ]
    nodes.append(
        NodeSpec(
            name=input_name,
            op="Input",
            kind="input",
            module_path="",
            in_shapes=[],
            out_shapes=[list(input_shapes[0])],
            tags=["boundary"],
            depth=0,
        )
    )

    for call in calls:
        module = call["module"]
        cls_name = type(module).__name__
        kind = ops.classify_module(cls_name)
        params = ops.extract_module_params(module, kind)
        out_shapes = call["out_shapes"]
        if not out_shapes:
            continue
        path = call["path"]
        name = allocate(path)
        num_params = sum(int(p.numel()) for p in module.parameters())
        flops = ops.module_flops(kind, params, call["in_shapes"], out_shapes, cls_name)
        tags = ops.tags_for(kind, cls_name, params)
        notes = ["FLOPs estimated"] if ops.flops_is_approx(kind) else []
        nodes.append(
            NodeSpec(
                name=name,
                op=cls_name,
                kind=kind,
                module_path=path,
                in_shapes=call["in_shapes"],
                out_shapes=out_shapes,
                params=params,
                num_params=num_params,
                flops=flops,
                tags=tags,
                notes=notes,
                source=source_index.module_source(path, model),
            )
        )

    # Eager execution only gives us an ordering: chain consecutive nodes.
    for index in range(1, len(nodes)):
        edges.append(
            EdgeSpec(
                source=index - 1,
                target=index,
                kind="data",
                shapes=[list(shape) for shape in nodes[index - 1].out_shapes],
            )
        )
    output_shapes = ops.shape_of_value(final_result) or [[-1]]
    output_index = len(nodes)
    nodes.append(
        NodeSpec(
            name="output",
            op="Output",
            kind="output",
            module_path="",
            in_shapes=[list(shape) for shape in nodes[-1].out_shapes] if nodes else [],
            out_shapes=[list(shape) for shape in output_shapes],
            tags=["boundary"],
            depth=0,
        )
    )
    if output_index > 0:
        edges.append(
            EdgeSpec(
                source=output_index - 1,
                target=output_index,
                kind="data",
                shapes=[list(shape) for shape in nodes[output_index - 1].out_shapes],
            )
        )
    _apply_depths(nodes)

    output_tensors = _tensors_of(final_result, torch)
    outputs = []
    for position, shape in enumerate(output_shapes):
        dtype = _dtype_name(output_tensors[position]) if position < len(output_tensors) else "float32"
        outputs.append(
            {
                "name": "output" if len(output_shapes) == 1 else f"output_{position}",
                "shape": list(shape),
                "dtype": dtype,
            }
        )
    meta, attention_paths = collect_module_meta(model, source_index, torch)
    total_params = sum(int(parameter.numel()) for parameter in model.parameters())
    trainable_params = sum(
        int(parameter.numel()) for parameter in model.parameters() if parameter.requires_grad
    )
    flops_approx = any(
        node.kind in ("attention", "rnn", "block", "custom", "unknown") for node in nodes
    )
    return Analysis(
        mode="trace",
        nodes=nodes,
        edges=edges,
        input_specs=inputs,
        output_specs=outputs,
        warnings=warnings,
        total_params=total_params,
        trainable_params=trainable_params,
        flops_approx=flops_approx,
        module_meta=meta,
        attention_paths=attention_paths,
        class_name=candidate.class_name,
        model_name=candidate.display_name,
        root_source=source_index.class_sources.get(candidate.class_name),
    )


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------

def analyze_model_dynamic(
    model: Any,
    module_file: ModuleFile,
    candidate: ModelCandidate,
    options: Any,
    source_index: SourceIndex,
) -> Analysis:
    """Trace ``model`` dynamically, preferring FX and falling back to hooks."""
    torch = _torch()
    if torch is None:
        raise AnalyzeError("PyTorch is not installed", file=module_file.path)
    slow_key = (str(module_file.path.resolve()), candidate.class_name)
    if slow_key in _SLOW_CLASSES:
        raise AnalyzeError(
            f"dynamic tracing skipped: another instance of {candidate.class_name} "
            f"exceeded the {MAX_FORWARD_SECONDS:.0f}s forward budget",
            file=module_file.path,
            line=candidate.line,
        )
    model = model.eval()
    candidates = dummy_input_candidates(model, options)
    if not candidates:
        raise AnalyzeError("no dummy input candidates", file=module_file.path)

    eager_ok: tuple[tuple[Any, ...], str, Any] | None = None
    fx_error: BaseException | None = None
    last_error: BaseException | None = None
    slow_error: AnalyzeError | None = None

    for dummy_args, label in candidates:
        started = time.perf_counter()
        try:
            with _suppress_user_output(), torch.no_grad():
                eager_result = model(*dummy_args)
        except Exception as exc:  # noqa: BLE001 - try the next shape guess
            last_error = exc
            LOGGER.debug("dummy input %s failed eager forward: %s", label, exc)
            continue
        elapsed = time.perf_counter() - started
        if elapsed > MAX_FORWARD_SECONDS:
            _SLOW_CLASSES.add(slow_key)
            slow_error = AnalyzeError(
                f"dynamic tracing skipped: eager forward took {elapsed:.1f}s "
                f"(budget {MAX_FORWARD_SECONDS:.0f}s); static analysis used",
                file=module_file.path,
                line=candidate.line,
            )
            LOGGER.info("skipping dynamic trace of %s: forward took %.1fs", candidate.display_name, elapsed)
            continue
        if eager_ok is None:
            eager_ok = (dummy_args, label, eager_result)
        try:
            with _suppress_user_output():
                return _trace_fx(
                    model, module_file, candidate, dummy_args, label, source_index, torch
                )
        except Exception as exc:  # noqa: BLE001 - symbolic tracing is best effort
            fx_error = exc
            LOGGER.debug("FX trace failed for %s: %s", label, exc, exc_info=True)

    if eager_ok is None:
        if slow_error is not None:
            raise slow_error
        detail = f"{type(last_error).__name__}: {last_error}" if last_error is not None else "unknown error"
        raise AnalyzeError(
            f"forward pass failed for every dummy input guess ({detail})",
            file=module_file.path,
            line=candidate.line,
        )
    dummy_args, label, _ = eager_ok
    with _suppress_user_output():
        return _trace_hooks(
            model,
            module_file,
            candidate,
            dummy_args,
            label,
            source_index,
            fx_error,
            torch,
        )


__all__ = [
    "analyze_model_dynamic",
    "load_file_module",
    "instantiate_candidate",
    "torch_available",
    "dummy_input_candidates",
    "collect_module_meta",
]
