"""Structural model comparison for NetPrism graphs.

Layers are matched by ``modulePath`` (falling back to ``name``); duplicate
paths get an occurrence suffix so repeated functional calls still pair up.  A
layer is *changed* when its op, input shapes, output shapes or parameter count
differ.  The result is a plain JSON-serialisable dict, ready to be consumed by
research scripts.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any

from .metrics import enrich_graph, layer_rows

_CHANGED_FIELDS = ("op", "inShapes", "outShapes", "numParams")

_CSV_COLUMNS: tuple[str, ...] = (
    "status",
    "key",
    "modulePath",
    "name",
    "op_a",
    "op_b",
    "kind_a",
    "kind_b",
    "numParams_a",
    "numParams_b",
    "flops_a",
    "flops_b",
    "activationBytes_a",
    "activationBytes_b",
    "receptiveField_a",
    "receptiveField_b",
    "changedFields",
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _as_int(value: Any, default: int = 0) -> int:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return default


def _metric_value(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def _model(graph: dict[str, Any]) -> dict[str, Any]:
    model = graph.get("model") if isinstance(graph, dict) else None
    return model if isinstance(model, dict) else {}


def _prepare(graph: Any) -> dict[str, Any]:
    if not isinstance(graph, dict):
        return {}
    if isinstance(graph.get("model"), dict):
        enrich_graph(graph)
    return graph


def _model_snapshot(graph: dict[str, Any]) -> dict[str, Any]:
    model = _model(graph)
    metrics = model.get("metrics") if isinstance(model.get("metrics"), dict) else {}
    warnings = model.get("warnings") if isinstance(model.get("warnings"), list) else []
    return {
        "name": str(model.get("name") or ""),
        "framework": str(model.get("framework") or ""),
        "mode": str(model.get("mode") or ""),
        "numLayers": _as_int(model.get("numLayers")),
        "totalParams": _as_int(model.get("totalParams")),
        "totalFlops": _as_int(model.get("totalFlops")),
        "activationBytes": _metric_value(metrics.get("activationBytes")),
        "maxReceptiveField": _metric_value(metrics.get("maxReceptiveField")),
        "warnings": len(warnings),
    }


def _keyed_rows(graph: dict[str, Any]) -> dict[str, dict[str, Any]]:
    keyed: dict[str, dict[str, Any]] = {}
    occurrences: dict[str, int] = {}
    for row in layer_rows(graph):
        base = str(row.get("modulePath") or row.get("name") or row.get("id") or "node")
        occurrences[base] = occurrences.get(base, 0) + 1
        key = base if occurrences[base] == 1 else f"{base}#{occurrences[base]}"
        while key in keyed:  # extremely defensive: never drop a row
            occurrences[base] += 1
            key = f"{base}#{occurrences[base]}"
        keyed[key] = {"key": key, **row}
    return keyed


def _triplet(a_value: int | None, b_value: int | None) -> dict[str, Any]:
    diff = b_value - a_value if a_value is not None and b_value is not None else None
    return {"a": a_value, "b": b_value, "diff": diff}


def _json_value(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _changed_fields(row_a: dict[str, Any], row_b: dict[str, Any]) -> dict[str, dict[str, Any]]:
    fields: dict[str, dict[str, Any]] = {}
    for field in _CHANGED_FIELDS:
        value_a, value_b = row_a.get(field), row_b.get(field)
        if value_a != value_b:
            fields[field] = {"a": _json_value(value_a), "b": _json_value(value_b)}
    return fields


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def compare_graphs(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Compare two graphs and return a JSON-serialisable difference document."""
    graph_a = _prepare(a)
    graph_b = _prepare(b)
    rows_a = _keyed_rows(graph_a)
    rows_b = _keyed_rows(graph_b)

    added = [_json_value(row) for key, row in rows_b.items() if key not in rows_a]
    removed = [_json_value(row) for key, row in rows_a.items() if key not in rows_b]
    changed: list[dict[str, Any]] = []
    unchanged = 0
    for key, row_a in rows_a.items():
        row_b = rows_b.get(key)
        if row_b is None:
            continue
        fields = _changed_fields(row_a, row_b)
        if fields:
            changed.append(
                {
                    "key": key,
                    "modulePath": row_b.get("modulePath") or row_a.get("modulePath") or "",
                    "name": row_b.get("name") or row_a.get("name") or "",
                    "idA": row_a.get("id"),
                    "idB": row_b.get("id"),
                    "fields": fields,
                }
            )
        else:
            unchanged += 1

    snapshot_a = _model_snapshot(graph_a)
    snapshot_b = _model_snapshot(graph_b)
    summary = {
        "params": _triplet(snapshot_a["totalParams"], snapshot_b["totalParams"]),
        "flops": _triplet(snapshot_a["totalFlops"], snapshot_b["totalFlops"]),
        "activationBytes": _triplet(snapshot_a["activationBytes"], snapshot_b["activationBytes"]),
        "maxReceptiveField": _triplet(snapshot_a["maxReceptiveField"], snapshot_b["maxReceptiveField"]),
    }
    return {
        "a": snapshot_a,
        "b": snapshot_b,
        "summary": summary,
        "counts": {
            "added": len(added),
            "removed": len(removed),
            "changed": len(changed),
            "unchanged": unchanged,
        },
        "added": added,
        "removed": removed,
        "changed": changed,
    }


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
def _fmt_int(value: Any) -> str:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return f"{value:,}"
    return "—"


def _fmt_signed(value: Any) -> str:
    if isinstance(value, int) and not isinstance(value, bool):
        return f"{value:+,}"
    return "—"


def _fmt_bytes(value: Any) -> str:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        return "—"
    if value < 1024:
        return f"{value:,} B"
    units = ("KiB", "MiB", "GiB", "TiB", "PiB")
    size = float(value)
    index = -1
    while size >= 1024 and index < len(units) - 1:
        size /= 1024
        index += 1
    return f"{size:.2f} {units[index]} ({value:,} B)"


def _shapes_text(shapes: Any, *, sep: str = "×") -> str:
    if not isinstance(shapes, list) or not shapes:
        return "—"
    parts: list[str] = []
    for shape in shapes:
        if isinstance(shape, (list, tuple)) and shape:
            parts.append(sep.join(str(dim) for dim in shape))
    return " / ".join(parts) if parts else "—"


def _md_cell(value: Any) -> str:
    text = str(value) if value is not None else "—"
    return text.replace("\\", "\\\\").replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def _table(headers: list[str], rows: list[list[Any]], *, align: list[str] | None = None) -> list[str]:
    if align is None:
        align = ["---"] * len(headers)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(align) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(_md_cell(value) for value in row) + " |")
    return lines


def _model_label(snapshot: dict[str, Any]) -> str:
    name = snapshot.get("name") or "—"
    framework = snapshot.get("framework") or "—"
    mode = snapshot.get("mode") or "—"
    return f"{name} ({framework}/{mode})"


def _changed_text(fields: Any) -> str:
    if not isinstance(fields, dict) or not fields:
        return "—"
    parts: list[str] = []
    for field in _CHANGED_FIELDS:
        if field not in fields:
            continue
        value = fields[field]
        if not isinstance(value, dict):
            continue
        before, after = value.get("a"), value.get("b")
        if field in ("inShapes", "outShapes"):
            before, after = _shapes_text(before), _shapes_text(after)
        elif field == "numParams":
            before, after = _fmt_int(before), _fmt_int(after)
        parts.append(f"{field}: {before} → {after}")
    return "; ".join(parts) if parts else "—"


def _render_md(result: dict[str, Any]) -> str:
    result = result if isinstance(result, dict) else {}
    snapshot_a = result.get("a") if isinstance(result.get("a"), dict) else {}
    snapshot_b = result.get("b") if isinstance(result.get("b"), dict) else {}
    summary = result.get("summary") if isinstance(result.get("summary"), dict) else {}
    counts = result.get("counts") if isinstance(result.get("counts"), dict) else {}
    lines = ["# 模型对比报告", ""]
    lines.append(f"**A** {_md_cell(_model_label(snapshot_a))} · **B** {_md_cell(_model_label(snapshot_b))}")
    lines.append("")

    lines += ["## 摘要 (Summary)", ""]
    summary_rows: list[list[Any]] = []
    for key, label, formatter in (
        ("params", "Params", _fmt_int),
        ("flops", "FLOPs", _fmt_int),
        ("activationBytes", "Activation (估算)", _fmt_bytes),
        ("maxReceptiveField", "Receptive field (估算)", _fmt_int),
    ):
        item = summary.get(key) if isinstance(summary.get(key), dict) else {}
        summary_rows.append(
            [
                label,
                formatter(item.get("a")),
                formatter(item.get("b")),
                _fmt_signed(item.get("diff")),
            ]
        )
    lines += _table(["Metric", "A", "B", "Δ (B-A)"], summary_rows, align=["---", "---:", "---:", "---:"])
    lines.append("")

    added = result.get("added") if isinstance(result.get("added"), list) else []
    lines += [f"## 新增节点 (Added: {len(added)})", ""]
    if added:
        lines += _table(
            ["Module", "Name", "Op", "Kind", "Out", "Params", "FLOPs"],
            [
                [
                    row.get("modulePath") or "—",
                    row.get("name") or "—",
                    row.get("op") or "—",
                    row.get("kind") or "—",
                    _shapes_text(row.get("outShapes")),
                    _fmt_int(row.get("numParams")),
                    _fmt_int(row.get("flops")),
                ]
                for row in added
                if isinstance(row, dict)
            ],
        )
    else:
        lines.append("无新增节点。")
    lines.append("")

    removed = result.get("removed") if isinstance(result.get("removed"), list) else []
    lines += [f"## 删除节点 (Removed: {len(removed)})", ""]
    if removed:
        lines += _table(
            ["Module", "Name", "Op", "Kind", "Out", "Params", "FLOPs"],
            [
                [
                    row.get("modulePath") or "—",
                    row.get("name") or "—",
                    row.get("op") or "—",
                    row.get("kind") or "—",
                    _shapes_text(row.get("outShapes")),
                    _fmt_int(row.get("numParams")),
                    _fmt_int(row.get("flops")),
                ]
                for row in removed
                if isinstance(row, dict)
            ],
        )
    else:
        lines.append("无删除节点。")
    lines.append("")

    changed = result.get("changed") if isinstance(result.get("changed"), list) else []
    lines += [f"## 变更节点 (Changed: {len(changed)})", ""]
    if changed:
        lines += _table(
            ["Module", "Name", "变更"],
            [
                [
                    row.get("modulePath") or "—",
                    row.get("name") or "—",
                    _changed_text(row.get("fields")),
                ]
                for row in changed
                if isinstance(row, dict)
            ],
        )
    else:
        lines.append("无变更节点。")
    lines.append("")

    lines.append(
        "> 对比说明：按 modulePath 匹配（为空时回退 name）；"
        f"未变 {_fmt_int(counts.get('unchanged'))} 层。"
    )
    lines.append("")
    return "\n".join(lines)


def _render_csv(result: dict[str, Any]) -> str:
    result = result if isinstance(result, dict) else {}
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer)
    writer.writerow(_CSV_COLUMNS)

    def value(row: Any, name: str) -> Any:
        if not isinstance(row, dict):
            return ""
        item = row.get(name)
        return "" if item is None else item

    def emit(status: str, row_a: Any, row_b: Any, changed_fields: str = "") -> None:
        identity = row_b if isinstance(row_b, dict) else row_a
        writer.writerow(
            [
                status,
                value(identity, "key"),
                value(identity, "modulePath"),
                value(identity, "name"),
                value(row_a, "op"),
                value(row_b, "op"),
                value(row_a, "kind"),
                value(row_b, "kind"),
                value(row_a, "numParams"),
                value(row_b, "numParams"),
                value(row_a, "flops"),
                value(row_b, "flops"),
                value(row_a, "activationBytes"),
                value(row_b, "activationBytes"),
                value(row_a, "receptiveField"),
                value(row_b, "receptiveField"),
                changed_fields,
            ]
        )

    for row in result.get("added") if isinstance(result.get("added"), list) else []:
        if isinstance(row, dict):
            emit("added", None, row)
    for row in result.get("removed") if isinstance(result.get("removed"), list) else []:
        if isinstance(row, dict):
            emit("removed", row, None)
    for entry in result.get("changed") if isinstance(result.get("changed"), list) else []:
        if not isinstance(entry, dict):
            continue
        fields = entry.get("fields") if isinstance(entry.get("fields"), dict) else {}

        def field_value(name: str, side: str) -> Any:
            value = fields.get(name)
            return value.get(side) if isinstance(value, dict) else ""

        writer.writerow(
            [
                "changed",
                entry.get("key") or "",
                entry.get("modulePath") or "",
                entry.get("name") or "",
                field_value("op", "a"),
                field_value("op", "b"),
                field_value("kind", "a"),
                field_value("kind", "b"),
                field_value("numParams", "a"),
                field_value("numParams", "b"),
                field_value("flops", "a"),
                field_value("flops", "b"),
                field_value("activationBytes", "a"),
                field_value("activationBytes", "b"),
                field_value("receptiveField", "a"),
                field_value("receptiveField", "b"),
                "; ".join(field for field in _CHANGED_FIELDS if field in fields),
            ]
        )
    return "\ufeff" + buffer.getvalue()


def _render_json(result: dict[str, Any]) -> str:
    payload = result if isinstance(result, dict) else {}
    return json.dumps(payload, ensure_ascii=False, indent=2, default=_json_value)


def render_compare(result: dict[str, Any], fmt: str = "md") -> str:
    """Render a :func:`compare_graphs` result as ``md``, ``json`` or ``csv``."""
    fmt = str(fmt or "md").lower()
    if fmt == "md":
        return _render_md(result)
    if fmt == "json":
        return _render_json(result)
    if fmt == "csv":
        return _render_csv(result)
    raise ValueError(f"不支持的对比格式 {fmt!r}（可选 md/json/csv）")


def write_compare(result: dict[str, Any], out: str | Path, fmt: str = "md") -> None:
    """Write rendered comparison; CSV uses UTF-8-SIG so Excel opens it cleanly."""
    fmt = str(fmt or "md").lower()
    text = render_compare(result, fmt=fmt)
    path = Path(out)
    if fmt == "csv":
        if text.startswith("\ufeff"):
            text = text[1:]
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            handle.write(text)
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(text)


__all__ = ["compare_graphs", "render_compare", "write_compare"]
