"""Constructor argument synthesis for models that need required parameters.

Real-world repositories often only expose models through a configuration object
(``GPT(config)``) or require a handful of dimensions (``UNet(n_channels,
n_classes)``, ``ViT(image_size, patch_size, ...)``).  NetPrism synthesizes those
arguments so the model can still be traced:

* ``dataclass`` configs with declared defaults are instantiated directly;
* common parameter names get sensible values (``num_classes=10``,
  ``image_size=32``, ``vocab_size=128``, ``dim=32``, ...);
* string arguments such as ``vgg_name`` are recovered from call sites in the
  same folder (``VGG('VGG11')``) or from a small list of known variants.

Every synthesized value is reported through warnings; the caller marks such
graphs as approximate.
"""

from __future__ import annotations

import dataclasses
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .discover import (
    InitParam,
    ModelCandidate,
    ModuleFile,
    collect_sibling_ctor_arguments,
    collect_sibling_string_literals,
)

LOGGER = logging.getLogger("netprism.core.synth")

#: Exact-name heuristic values (checked before substring rules).
_EXACT_VALUES: dict[str, Any] = {
    "n_channels": 3,
    "in_channels": 3,
    "in_chans": 3,
    "channels": 3,
    "c": 3,
    "n_classes": 10,
    "num_classes": 10,
    "classes": 10,
    "n_cls": 10,
    "image_size": 32,
    "img_size": 32,
    "image_height": 32,
    "image_width": 32,
    "vocab_size": 128,
    "vocab": 128,
    "embed_dim": 32,
    "embedding_dim": 32,
    "dim": 32,
    "d_model": 32,
    "n_embd": 32,
    "hidden_dim": 64,
    "mlp_dim": 64,
    "dim_head": 8,
    "head_dim": 8,
    "num_heads": 4,
    "n_head": 4,
    "n_heads": 4,
    "nhead": 4,
    "heads": 4,
    "depth": 2,
    "num_layers": 2,
    "n_layer": 2,
    "n_layers": 2,
    "layers": 2,
    "seq_len": 16,
    "max_len": 16,
    "block_size": 16,
    "context_length": 16,
    "num_frames": 4,
    "frames": 4,
    "tubelet_size": 2,
    "frame_patch_size": 2,
    "patch_size": 4,
    "kernel_size": 3,
    "stride": 1,
    "padding": 1,
    "pooling_kernel_size": 2,
    "pooling_stride": 2,
    "pooling_padding": 0,
    "mlp_ratio": 1,
    "num_tokens": 8,
    "num_cls_tokens": 1,
    "dim_head": 8,
    "dropout": 0.0,
    "drop_rate": 0.0,
    "emb_dropout": 0.0,
    "attention_dropout": 0.0,
    "stochastic_depth_rate": 0.0,
    "bias": False,
    "bilinear": False,
    "pretrained": False,
    "flash": True,
    "scale": 1.0,
}

_SUBSTRING_VALUES: tuple[tuple[str, Any], ...] = (
    ("patch_size", 4),
    ("num_frames", 4),
    ("tubelet", 2),
    ("dim_head", 8),
    ("head_dim", 8),
    ("num_heads", 4),
    ("n_head", 4),
    ("nhead", 4),
    ("mlp", 64),
    ("hidden", 64),
    ("num_classes", 10),
    ("num_class", 10),
    ("n_classes", 10),
    ("vocab", 128),
    ("embedding", 32),
    ("embed", 32),
    ("dim", 32),
    ("depth", 2),
    ("layer", 2),
    ("image_size", 32),
    ("img_size", 32),
    ("image", 32),
    ("resolution", 32),
    ("kernel", 3),
    ("channel", 3),
    ("seq", 16),
    ("len", 16),
    ("stride", 1),
    ("padding", 1),
    ("pool", 2),
    ("rate", 0.0),
    ("drop", 0.0),
    ("ratio", 1),
)

_STRING_CANDIDATES: dict[str, list[str]] = {
    "pool": ["cls", "mean"],
    "pooling": ["cls", "mean"],
    "activation": ["gelu", "relu"],
    "act": ["gelu", "relu"],
    "nonlinearity": ["gelu", "relu"],
    "mode": ["bilinear", "nearest"],
    "name": ["VGG11", "ResNet18"],
    "variant": ["VGG11"],
    "arch": ["VGG11"],
}

_MAX_CONFIGS = 8


@dataclass
class SynthResult:
    model: Any
    description: str
    warnings: list[str] = field(default_factory=list)
    guessed: bool = False


def _last_annotation(annotation: str | None) -> str:
    if not annotation:
        return ""
    cleaned = re.sub(r"[^A-Za-z0-9_.]", " ", annotation)
    parts = [part for part in cleaned.split() if part]
    return parts[-1].split(".")[-1] if parts else ""


def _primitive_annotation(annotation: str | None) -> str | None:
    if not annotation:
        return None
    lowered = annotation.lower()
    if lowered in ("int", "float", "bool", "str"):
        return lowered
    if "optional[" in lowered or "|" in lowered:
        for primitive in ("int", "float", "bool", "str"):
            if primitive in lowered:
                return primitive
    return None


def _guess_int(name: str) -> tuple[int, bool]:
    lowered = name.lower()
    if lowered in _EXACT_VALUES and isinstance(_EXACT_VALUES[lowered], int) and not isinstance(
        _EXACT_VALUES[lowered], bool
    ):
        return int(_EXACT_VALUES[lowered]), True
    for token, value in _SUBSTRING_VALUES:
        if token in lowered and isinstance(value, int) and not isinstance(value, bool):
            return int(value), token != "ratio"
    return 1, False


def _guess_float(name: str) -> tuple[float, bool]:
    lowered = name.lower()
    value = _EXACT_VALUES.get(lowered)
    if isinstance(value, float):
        return value, True
    if any(token in lowered for token in ("dropout", "rate", "eps", "temperature", "scale")):
        return 0.0, True
    return 0.0, False


def _guess_bool(name: str) -> tuple[bool, bool]:
    lowered = name.lower()
    value = _EXACT_VALUES.get(lowered)
    if isinstance(value, bool):
        return value, True
    if lowered in ("bias", "pre_norm", "norm_first", "channel_last", "flash"):
        return False, True
    return False, False


def _guess_string(name: str, class_name: str, literals: list[str]) -> tuple[list[str], bool]:
    lowered = name.lower()
    if literals:
        return list(dict.fromkeys(literals)), True
    for token, values in _STRING_CANDIDATES.items():
        if token in lowered:
            return list(values), True
    return [], False


def _find_config_class(module_obj: Any, info: ModuleFile, param: InitParam) -> type | None:
    """Resolve a ``config`` constructor argument to a dataclass class."""
    annotation_name = _last_annotation(param.annotation)
    candidates: list[Any] = []
    if annotation_name:
        local = info.dataclasses.get(annotation_name)
        if local is not None:
            candidates.append(getattr(module_obj, local.name, None))
        candidates.append(getattr(module_obj, annotation_name, None))
    if not candidates and param.name.lower() in ("config", "cfg", "conf", "args", "options", "hparams"):
        dataclass_classes = [
            value for value in vars(module_obj).values() if isinstance(value, type) and dataclasses.is_dataclass(value)
        ]
        if len(dataclass_classes) == 1:
            candidates.append(dataclass_classes[0])
        else:
            named = [
                value
                for value in dataclass_classes
                if "config" in value.__name__.lower() or "cfg" in value.__name__.lower()
            ]
            if len(named) == 1:
                candidates.append(named[0])
    for candidate in candidates:
        if isinstance(candidate, type) and dataclasses.is_dataclass(candidate):
            return candidate
    return None


def _instantiate_config(
    module_obj: Any,
    config_cls: type,
    warnings: list[str],
) -> Any | None:
    try:
        fields = list(dataclasses.fields(config_cls))
    except TypeError:
        return None
    required = [
        item
        for item in fields
        if item.default is dataclasses.MISSING and item.default_factory is dataclasses.MISSING
    ]
    kwargs: dict[str, Any] = {}
    guessed = False
    for item in required:
        param = InitParam(name=item.name, has_default=False, annotation=str(item.type))
        value, confident = _guess_for_param(param, config_cls.__name__, [])
        if not confident:
            warnings.append(
                f"could not synthesize required dataclass field {item.name!r} of {config_cls.__name__}"
            )
            return None
        kwargs[item.name] = value
        guessed = True
    try:
        config = config_cls(**kwargs)
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"could not instantiate dataclass {config_cls.__name__}: {exc}")
        return None
    if kwargs:
        warnings.append(
            f"synthesized dataclass {config_cls.__name__}("
            + ", ".join(f"{key}={value!r}" for key, value in kwargs.items())
            + ") with heuristic field values"
        )
    else:
        warnings.append(
            f"synthesized constructor argument from dataclass defaults: {config_cls.__name__}()"
        )
    _ = guessed
    return config


def _guess_for_param(param: InitParam, class_name: str, literals: list[str]) -> tuple[Any, bool]:
    annotation = _primitive_annotation(param.annotation)
    name = param.name
    if annotation == "bool":
        return _guess_bool(name)
    if annotation == "float":
        return _guess_float(name)
    if annotation == "str":
        candidates, confident = _guess_string(name, class_name, literals)
        return (candidates[0] if candidates else "default"), confident
    if annotation == "int":
        return _guess_int(name)
    # No primitive annotation: infer from the parameter name.
    lowered = name.lower()
    if any(token in lowered for token in ("dropout", "rate", "eps", "temperature")):
        return _guess_float(name)
    if any(token in lowered for token in ("bias", "bilinear", "pretrained", "flash", "norm_first")):
        return _guess_bool(name)
    if any(token in lowered for token in ("name", "variant", "arch", "mode", "activation", "pool")):
        candidates, confident = _guess_string(name, class_name, literals)
        return (candidates[0] if candidates else "default"), confident
    return _guess_int(name)


def _string_literals(module_obj: Any, info: ModuleFile, candidate: ModelCandidate, class_name: str, param: str) -> list[str]:
    literals = list(info.ctor_string_literals.get(class_name, []))
    if not literals:
        try:
            literals = collect_sibling_string_literals(candidate.file.parent, class_name, exclude=candidate.file)
        except Exception:  # noqa: BLE001 - best effort only
            literals = []
    # Keep only plausible values for this parameter.
    lowered = param.lower()
    if "name" not in lowered and "variant" not in lowered and "arch" not in lowered:
        if not any(token in lowered for token in ("activation", "pool", "mode")):
            return []
    return literals


def synthesize_candidate(
    module_obj: Any,
    candidate: ModelCandidate,
    info: ModuleFile,
) -> SynthResult:
    """Try to build the model with synthesized constructor arguments."""
    from .errors import AnalyzeError

    torch = None
    try:
        import torch as torch_module  # noqa: F401
    except Exception:  # pragma: no cover
        raise AnalyzeError("PyTorch is not installed", file=candidate.file)
    torch = torch_module
    cls = getattr(module_obj, candidate.class_name, None)
    if not isinstance(cls, type):
        raise AnalyzeError(f"class {candidate.class_name!r} not found", file=candidate.file)

    module_class = candidate.module_class
    required = [item for item in module_class.init_params if not item.has_default]
    if not required:
        try:
            model = cls()
        except Exception as exc:  # noqa: BLE001
            raise AnalyzeError(
                f"{candidate.display_name}() failed: {type(exc).__name__}: {exc}",
                file=candidate.file,
                line=candidate.line,
            ) from exc
        return SynthResult(model=model, description=f"{candidate.display_name}()", warnings=[])

    warnings: list[str] = []
    base_kwargs: dict[str, Any] = {}
    guessed = False
    string_variants: dict[str, list[str]] = {}

    callsite_args: dict[str, list[object]] = dict(info.class_ctor_args.get(candidate.class_name, {}))
    missing_params = [item.name for item in required if item.name not in callsite_args]
    if missing_params:
        try:
            sibling = collect_sibling_ctor_arguments(
                candidate.file.parent,
                candidate.class_name,
                [item.name for item in module_class.init_params],
                exclude=candidate.file,
            )
        except Exception:  # noqa: BLE001 - best effort only
            sibling = {}
        for key, values in sibling.items():
            callsite_args.setdefault(key, values)

    for param in required:
        callsite_values = [value for value in callsite_args.get(param.name, []) if value is not None]
        if callsite_values:
            first = callsite_values[0]
            if isinstance(first, str):
                variants = [str(value) for value in callsite_values if isinstance(value, str)]
                string_variants[param.name] = list(dict.fromkeys([*variants, *_UNKNOWN_INT_FALLBACKS_STR]))[:4]
                base_kwargs[param.name] = string_variants[param.name][0]
            else:
                base_kwargs[param.name] = first
            warnings.append(
                f"{candidate.display_name}: used call-site literal for constructor argument "
                f"{param.name}={base_kwargs[param.name]!r}"
            )
            continue
        config_cls = _find_config_class(module_obj, info, param)
        if config_cls is not None:
            config = _instantiate_config(module_obj, config_cls, warnings)
            if config is not None:
                base_kwargs[param.name] = config
                continue
        literals = _string_literals(module_obj, info, candidate, candidate.class_name, param.name)
        annotation = _primitive_annotation(param.annotation)
        lowered = param.name.lower()
        is_string = annotation == "str" or (
            annotation is None
            and any(token in lowered for token in ("name", "variant", "arch", "activation", "pool", "mode"))
        )
        if is_string:
            candidates, confident = _guess_string(param.name, candidate.class_name, literals)
            if not candidates or not confident:
                raise AnalyzeError(
                    f"cannot confidently synthesize required string argument {param.name!r} "
                    f"for {candidate.class_name}; static analysis used",
                    file=candidate.file,
                    line=candidate.line,
                )
            string_variants[param.name] = list(dict.fromkeys([*candidates, *_UNKNOWN_INT_FALLBACKS_STR]))[:4]
            base_kwargs[param.name] = string_variants[param.name][0]
            continue
        value, confident = _guess_for_param(param, candidate.class_name, literals)
        if not confident:
            raise AnalyzeError(
                f"cannot confidently synthesize required constructor argument {param.name!r} "
                f"for {candidate.class_name}; static analysis used",
                file=candidate.file,
                line=candidate.line,
            )
        base_kwargs[param.name] = value

    descriptions = [f"{key}={value!r}" for key, value in base_kwargs.items()]
    warnings.append(
        f"{candidate.display_name}: synthesized constructor arguments ({', '.join(descriptions)}); "
        "shapes and FLOPs reflect this synthesized configuration"
    )

    variants: list[dict[str, Any]] = [dict(base_kwargs)]
    for param, values in string_variants.items():
        expanded: list[dict[str, Any]] = []
        for config in variants:
            for value in values:
                candidate_config = dict(config)
                candidate_config[param] = value
                expanded.append(candidate_config)
        variants = expanded[:_MAX_CONFIGS]
        if values:
            guessed = True

    last_error: Exception | None = None
    for config in variants:
        try:
            model = cls(**config)
        except TypeError:
            # Positional-only constructors: pass values in signature order.
            positional = [config[item.name] for item in required if item.name in config]
            try:
                model = cls(*positional)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                continue
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            continue
        if isinstance(model, torch.nn.Module):
            description = f"{candidate.display_name}({', '.join(f'{k}={v!r}' for k, v in config.items())})"
            # Values from helper defaults/call sites are still synthesized arguments:
            # never report such a graph as exact (the caller forces flopsApprox).
            return SynthResult(model=model, description=description, warnings=warnings, guessed=True)
        last_error = TypeError(f"{candidate.display_name}() did not return an nn.Module")

    detail = f"{type(last_error).__name__}: {last_error}" if last_error is not None else "unknown error"
    raise AnalyzeError(
        f"constructor synthesis failed for {candidate.display_name}: {detail}",
        file=candidate.file,
        line=candidate.line,
    )


#: Extra string fallbacks tried after the name based candidates (kept small).
_UNKNOWN_INT_FALLBACKS_STR = ("default", "vgg11", "VGG11", "bidirectional")

__all__ = ["SynthResult", "synthesize_candidate"]
