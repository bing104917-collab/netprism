"""Markdown / CSV / HTML reports for NetPrism model graphs.

The report is deliberately self-contained: the HTML variant is one file with
inline CSS, no scripts and no external links, so it can be attached to a paper
submission or a review email.  CSV columns mirror :func:`netprism.metrics.layer_rows`
and files written by :func:`write_report` use UTF-8-SIG so Excel opens Chinese
model/group names correctly.
"""

from __future__ import annotations

import csv
import html
import io
from pathlib import Path
from typing import Any

from .metrics import enrich_graph, layer_rows

_LAYER_COLUMNS: tuple[str, ...] = (
    "id",
    "name",
    "op",
    "kind",
    "modulePath",
    "inShapes",
    "outShapes",
    "numParams",
    "flops",
    "activationBytes",
    "receptiveField",
    "groupId",
    "groupLabel",
    "source",
)


# --------------------------------------------------------------------------- #
# value formatting
# --------------------------------------------------------------------------- #
def _model(graph: dict[str, Any]) -> dict[str, Any]:
    model = graph.get("model") if isinstance(graph, dict) else None
    return model if isinstance(model, dict) else {}


def _groups(graph: dict[str, Any]) -> list[dict[str, Any]]:
    groups = graph.get("groups") if isinstance(graph, dict) else None
    return [group for group in groups if isinstance(group, dict)] if isinstance(groups, list) else []


def _as_int(value: Any, default: int = 0) -> int:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return default


def _fmt_int(value: Any) -> str:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return f"{value:,}"
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


def _source_label(source: Any) -> str:
    if not isinstance(source, dict):
        return "—"
    rel_path = source.get("relPath") or source.get("file")
    line = source.get("line")
    if rel_path:
        label = str(rel_path)
        if isinstance(line, int) and not isinstance(line, bool) and line > 0:
            label += f":{line}"
        return label
    code = source.get("code")
    if isinstance(code, str) and code.strip():
        return code.strip().replace("\n", " ")[:60]
    return "—"


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


def _group_rows(graph: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group in _groups(graph):
        rows.append(
            {
                "id": str(group.get("id") or ""),
                "label": str(group.get("label") or group.get("name") or "—"),
                "kind": str(group.get("kind") or ""),
                "depth": _as_int(group.get("depth")),
                "nodeCount": len(group.get("nodeIds") or []) if isinstance(group.get("nodeIds"), list) else 0,
                "numParams": _as_int(group.get("numParams")),
                "flops": _as_int(group.get("flops")),
            }
        )
    return rows


def _summary_values(graph: dict[str, Any]) -> dict[str, Any]:
    model = _model(graph)
    metrics = model.get("metrics") if isinstance(model.get("metrics"), dict) else {}
    warnings = model.get("warnings") if isinstance(model.get("warnings"), list) else []
    return {
        "name": str(model.get("name") or "(未命名模型)"),
        "framework": str(model.get("framework") or "—"),
        "mode": str(model.get("mode") or "—"),
        "params": _as_int(model.get("totalParams")),
        "flops": _as_int(model.get("totalFlops")),
        "flops_approx": bool(model.get("flopsApprox")),
        "activation": metrics.get("activationBytes"),
        "peak_activation": metrics.get("peakActivationBytes"),
        "receptive_field": metrics.get("maxReceptiveField"),
        "macs": metrics.get("macs"),
        "warnings": len(warnings),
    }


def _prepare(graph: dict[str, Any]) -> dict[str, Any]:
    """Fill missing metrics in place; tolerate empty/malformed documents."""
    if not isinstance(graph, dict):
        return {}
    if isinstance(graph.get("model"), dict):
        enrich_graph(graph)
    return graph


_ESTIMATE_NOTE = (
    "估算说明：激活显存、峰值显存、感受野与 macs 由输出形状和算子超参数推导，"
    "仅用于量级参考，并非实测值。"
)
_FLOPS_APPROX_NOTE = "该模型的 FLOPs 由静态规则推算，属于近似值。"


# --------------------------------------------------------------------------- #
# algorithm-understanding meta (read-only; written by netprism.algorithms)
# --------------------------------------------------------------------------- #
def _algo_meta(graph: dict[str, Any]) -> dict[str, Any]:
    meta = _model(graph).get("meta")
    return meta if isinstance(meta, dict) else {}


def _algo_family(meta: dict[str, Any]) -> str | None:
    family = meta.get("family")
    if not family:
        return None
    return str(meta.get("familyLabel") or family)


def _algo_patterns(meta: dict[str, Any]) -> list[dict[str, Any]]:
    patterns = meta.get("patterns")
    if not isinstance(patterns, list):
        return []
    return [pattern for pattern in patterns if isinstance(pattern, dict)][:10]


def _algo_pseudocode(meta: dict[str, Any]) -> list[str]:
    code = meta.get("pseudocode")
    if isinstance(code, str):
        lines = code.splitlines()
    elif isinstance(code, list):
        lines = [line for line in code if isinstance(line, str)]
    else:
        return []
    return [line for line in lines if line.strip()]


def _algo_payload(graph: dict[str, Any]) -> dict[str, Any] | None:
    """Return the algo section payload, or ``None`` when meta has nothing useful."""
    meta = _algo_meta(graph)
    family = _algo_family(meta)
    patterns = _algo_patterns(meta)
    pseudocode = _algo_pseudocode(meta)
    if family is None and not patterns and not pseudocode:
        return None
    return {"family": family, "patterns": patterns, "pseudocode": pseudocode}


def _pattern_label(pattern: dict[str, Any]) -> str:
    return str(pattern.get("label") or pattern.get("kind") or pattern.get("id") or "—")


def _pattern_formula(pattern: dict[str, Any]) -> str:
    return str(pattern.get("formula") or "—")


def _pattern_confidence(pattern: dict[str, Any]) -> str:
    value = pattern.get("confidence")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{float(value):.2f}"
    return "—"


def _pattern_evidence(pattern: dict[str, Any]) -> str:
    evidence = pattern.get("evidence")
    if isinstance(evidence, str):
        return evidence
    if isinstance(evidence, list):
        items = [str(item) for item in evidence if item is not None]
        return "; ".join(items) if items else "—"
    return "—"


def _render_algo_md(algo: dict[str, Any]) -> list[str]:
    lines = ["## 算法理解 (Algorithm)", ""]
    if algo["family"]:
        lines += [f"**家族 (Family)** {_md_cell(algo['family'])}", ""]
    patterns = algo["patterns"]
    if patterns:
        lines += _table(
            ["模式", "Kind", "公式", "置信度", "证据"],
            [
                [
                    _pattern_label(pattern),
                    str(pattern.get("kind") or "—"),
                    _pattern_formula(pattern),
                    _pattern_confidence(pattern),
                    _pattern_evidence(pattern),
                ]
                for pattern in patterns
            ],
            align=["---", "---", "---", "---:", "---"],
        )
    else:
        lines.append("未识别到关键模式。")
    lines.append("")
    if algo["pseudocode"]:
        lines += ["**Forward 伪代码**", "", "```text"]
        lines += [line.replace("```", "'''") for line in algo["pseudocode"]]
        lines += ["```", ""]
    return lines


_ALGO_CSS = """
.badge {
  display: inline-flex; align-items: center; padding: 5px 14px; border-radius: 999px;
  background: rgba(103,232,249,.14); border: 1px solid rgba(103,232,249,.42);
  color: #a5f3fc; font-weight: 650; font-size: 14px; letter-spacing: .2px;
}
.algo-family { margin: 0 0 18px; display: flex; align-items: center; gap: 10px; }
.algo-family-label { color: #8ea2c4; font-size: 12px; }
h3.algo-sub { margin: 20px 0 10px; font-size: 15px; font-weight: 650; color: #cfdcf2; }
pre.pseudo {
  margin: 0; padding: 16px 18px; overflow: auto; background: #080d1a;
  border: 1px solid #23304d; border-radius: 10px; color: #cfe3ff;
  font-family: "Cascadia Code", Consolas, "SFMono-Regular", Menlo, monospace;
  font-size: 12.5px; line-height: 1.55; white-space: pre;
}
"""


def _render_algo_html(algo: dict[str, Any]) -> str:
    parts = ['<section id="algo-view">', "<h2>算法理解 (Algorithm)</h2>"]
    if algo["family"]:
        parts.append(
            f'<p class="algo-family" id="algo-family"><span class="badge">{_escape(algo["family"])}</span>'
            '<span class="algo-family-label">模型家族</span></p>'
        )
    patterns = algo["patterns"]
    if patterns:
        parts.append('<h3 class="algo-sub">关键模式 (Patterns)</h3>')
        parts.append(
            '<div class="table-wrap" id="algo-patterns">'
            + _html_table(
                ["模式", "Kind", "公式", "置信度", "证据"],
                [
                    [
                        _pattern_label(pattern),
                        str(pattern.get("kind") or "—"),
                        _pattern_formula(pattern),
                        _pattern_confidence(pattern),
                        _pattern_evidence(pattern),
                    ]
                    for pattern in patterns
                ],
                numeric={3},
            )
            + "</div>"
        )
    else:
        parts.append('<p class="empty">未识别到关键模式。</p>')
    if algo["pseudocode"]:
        parts.append('<h3 class="algo-sub">Forward 伪代码</h3>')
        parts.append(
            '<pre class="pseudo" id="algo-pseudocode">'
            + _escape("\n".join(algo["pseudocode"]))
            + "</pre>"
        )
    parts.append("</section>")
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# markdown
# --------------------------------------------------------------------------- #
def _render_md(graph: dict[str, Any]) -> str:
    summary = _summary_values(graph)
    rows = layer_rows(graph)
    groups = _group_rows(graph)
    lines = [f"# {summary['name']} 分析报告", ""]
    summary_line = " · ".join(
        [
            f"**Framework** {_md_cell(summary['framework'])}",
            f"**Mode** {_md_cell(summary['mode'])}",
            f"**Params** {_fmt_int(summary['params'])}",
            f"**FLOPs** {_fmt_int(summary['flops'])}" + (" (approx)" if summary["flops_approx"] else ""),
            f"**Activation (估算)** {_fmt_bytes(summary['activation'])}",
            f"**Receptive field (估算)** {_fmt_int(summary['receptive_field'])}",
            f"**Warnings** {_fmt_int(summary['warnings'])}",
        ]
    )
    lines += [summary_line, ""]

    algo = _algo_payload(graph)
    if algo is not None:
        lines += _render_algo_md(algo)

    lines += ["## 分组汇总 (Groups)", ""]
    if groups:
        lines += _table(
            ["Group", "Kind", "Depth", "Nodes", "Params", "FLOPs"],
            [
                [group["label"], group["kind"], group["depth"], group["nodeCount"], _fmt_int(group["numParams"]), _fmt_int(group["flops"])]
                for group in groups
            ],
            align=["---", "---", "---:", "---:", "---:", "---:"],
        )
    else:
        lines.append("无分组信息。")
    lines.append("")

    lines += ["## 逐层明细 (Layers)", ""]
    if rows:
        lines += _table(
            ["ID", "Name", "Module", "Op", "Kind", "In", "Out", "Params", "FLOPs", "Activation (估算)", "RF (估算)", "Source"],
            [
                [
                    row["id"],
                    row["name"],
                    row["modulePath"] or "—",
                    row["op"],
                    row["kind"],
                    _shapes_text(row["inShapes"]),
                    _shapes_text(row["outShapes"]),
                    _fmt_int(row["numParams"]),
                    _fmt_int(row["flops"]),
                    _fmt_bytes(row["activationBytes"]),
                    _fmt_int(row["receptiveField"]),
                    _source_label(row["source"]),
                ]
                for row in rows
            ],
            align=["---", "---", "---", "---", "---", "---", "---", "---:", "---:", "---:", "---:", "---"],
        )
    else:
        lines.append("无节点数据。")
    lines.append("")

    lines.append(f"> {_ESTIMATE_NOTE}")
    if summary["flops_approx"]:
        lines.append(">")
        lines.append(f"> {_FLOPS_APPROX_NOTE}")
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# csv
# --------------------------------------------------------------------------- #
def _render_csv(graph: dict[str, Any]) -> str:
    summary = _summary_values(graph)
    rows = layer_rows(graph)
    buffer = io.StringIO(newline="")
    # A single ASCII comment line identifies the model without breaking the
    # layer-row column contract below it.  Excel treats it as a normal row.
    model_name = str(summary["name"]).replace("\r", " ").replace("\n", " ")
    buffer.write(
        f"# model: {model_name} | framework: {summary['framework']} | mode: {summary['mode']} | "
        f"params: {summary['params']} | flops: {summary['flops']} | "
        f"activationBytes: {summary['activation'] if summary['activation'] is not None else ''} | "
        f"maxReceptiveField: {summary['receptive_field'] if summary['receptive_field'] is not None else ''} | "
        f"warnings: {summary['warnings']}\n"
    )
    writer = csv.writer(buffer)
    writer.writerow(_LAYER_COLUMNS)
    for row in rows:
        source = _source_label(row["source"])
        in_shapes = _shapes_text(row["inShapes"], sep="x")
        out_shapes = _shapes_text(row["outShapes"], sep="x")
        writer.writerow(
            [
                row["id"],
                row["name"],
                row["op"],
                row["kind"],
                row["modulePath"],
                "" if in_shapes == "—" else in_shapes,
                "" if out_shapes == "—" else out_shapes,
                row["numParams"],
                row["flops"],
                row["activationBytes"],
                row["receptiveField"],
                row["groupId"] if row["groupId"] is not None else "",
                row["groupLabel"],
                "" if source == "—" else source,
            ]
        )
    # The CLI writes rendered text with plain UTF-8; a leading BOM makes the
    # downloaded CSV open correctly in Excel.
    return "\ufeff" + buffer.getvalue()


# --------------------------------------------------------------------------- #
# html
# --------------------------------------------------------------------------- #
_HTML_CSS = """
:root { color-scheme: light dark; }
* { box-sizing: border-box; }
body {
  margin: 0; padding: 32px 24px 48px;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Noto Sans SC", Arial, sans-serif;
  background: #0b1020; color: #e6edf7; line-height: 1.5;
}
.wrap { max-width: 1440px; margin: 0 auto; }
h1 { margin: 0 0 4px; font-size: 28px; letter-spacing: .2px; }
h2 { margin: 32px 0 12px; font-size: 19px; font-weight: 650; }
.sub { margin: 0 0 24px; color: #93a4c3; font-size: 13px; }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; }
.card {
  background: linear-gradient(160deg, rgba(103,232,249,.08), rgba(103,232,249,.02));
  border: 1px solid rgba(103,232,249,.18); border-radius: 12px; padding: 14px 16px;
}
.card .card-label { display: block; color: #8ea2c4; font-size: 12px; margin-bottom: 6px; }
.card strong { font-size: 17px; font-weight: 650; color: #f2f7ff; word-break: break-word; }
.table-wrap { overflow: auto; max-height: 62vh; border: 1px solid #23304d; border-radius: 12px; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { padding: 8px 12px; border-bottom: 1px solid #1b2740; text-align: left; white-space: nowrap; }
th { position: sticky; top: 0; background: #131c33; color: #b8c8e4; font-weight: 600; z-index: 1; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
tbody tr:nth-child(even) { background: rgba(255,255,255,.02); }
tbody tr:hover { background: rgba(103,232,249,.06); }
.note {
  margin-top: 28px; padding: 14px 18px; border-left: 3px solid #67e8f9;
  background: rgba(103,232,249,.06); border-radius: 6px; color: #b9c9e4; font-size: 13px;
}
.note p { margin: 0; }
.note p + p { margin-top: 6px; }
.empty { color: #8ea2c4; font-size: 13px; }
"""


def _escape(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _html_table(headers: list[str], rows: list[list[Any]], *, numeric: set[int] | None = None) -> str:
    numeric = numeric or set()
    parts = ["<table>", "<thead><tr>"]
    for index, header in enumerate(headers):
        css = ' class="num"' if index in numeric else ""
        parts.append(f"<th{css}>{_escape(header)}</th>")
    parts.append("</tr></thead><tbody>")
    for row in rows:
        parts.append("<tr>")
        for index, value in enumerate(row):
            css = ' class="num"' if index in numeric else ""
            parts.append(f"<td{css}>{_escape(value)}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return "".join(parts)


def _card(label: str, value: Any) -> str:
    return f'<div class="card"><span class="card-label">{_escape(label)}</span><strong>{_escape(value)}</strong></div>'


def _render_html(graph: dict[str, Any]) -> str:
    summary = _summary_values(graph)
    rows = layer_rows(graph)
    groups = _group_rows(graph)
    algo = _algo_payload(graph)
    algo_section = _render_algo_html(algo) if algo is not None else None
    cards = [
        _card("Framework", summary["framework"]),
        _card("Mode", summary["mode"]),
        _card("Params", _fmt_int(summary["params"])),
        _card("FLOPs", _fmt_int(summary["flops"]) + (" (approx)" if summary["flops_approx"] else "")),
        _card("Activation (估算)", _fmt_bytes(summary["activation"])),
        _card("Peak activation (估算)", _fmt_bytes(summary["peak_activation"])),
        _card("Receptive field (估算)", _fmt_int(summary["receptive_field"])),
        _card("Warnings", _fmt_int(summary["warnings"])),
    ]

    group_section = '<p class="empty">无分组信息。</p>'
    if groups:
        group_section = (
            '<div class="table-wrap">'
            + _html_table(
                ["Group", "Kind", "Depth", "Nodes", "Params", "FLOPs"],
                [
                    [group["label"], group["kind"], group["depth"], group["nodeCount"], _fmt_int(group["numParams"]), _fmt_int(group["flops"])]
                    for group in groups
                ],
                numeric={2, 3, 4, 5},
            )
            + "</div>"
        )

    layer_section = '<p class="empty">无节点数据。</p>'
    if rows:
        layer_section = (
            '<div class="table-wrap">'
            + _html_table(
                ["ID", "Name", "Module", "Op", "Kind", "In", "Out", "Params", "FLOPs", "Activation (估算)", "RF (估算)", "Source"],
                [
                    [
                        row["id"],
                        row["name"],
                        row["modulePath"] or "—",
                        row["op"],
                        row["kind"],
                        _shapes_text(row["inShapes"]),
                        _shapes_text(row["outShapes"]),
                        _fmt_int(row["numParams"]),
                        _fmt_int(row["flops"]),
                        _fmt_bytes(row["activationBytes"]),
                        _fmt_int(row["receptiveField"]),
                        _source_label(row["source"]),
                    ]
                    for row in rows
                ],
                numeric={7, 8, 9, 10},
            )
            + "</div>"
        )

    notes = f"<p>{_escape(_ESTIMATE_NOTE)}</p>"
    if summary["flops_approx"]:
        notes += f"<p>{_escape(_FLOPS_APPROX_NOTE)}</p>"

    return "\n".join(
        [
            "<!DOCTYPE html>",
            '<html lang="zh-CN">',
            "<head>",
            '<meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            f"<title>{_escape(summary['name'])} · NetPrism 分析报告</title>",
            f"<style>{_HTML_CSS}{_ALGO_CSS if algo_section else ''}</style>",
            "</head>",
            "<body>",
            '<main class="wrap">',
            "<header>",
            f"<h1>{_escape(summary['name'])}</h1>",
            f'<p class="sub">NetPrism 结构分析报告 · {_escape(summary["framework"])} / {_escape(summary["mode"])} · 所有复杂度指标均为估算</p>',
            "</header>",
            '<section class="cards">',
            *cards,
            "</section>",
            *([algo_section] if algo_section else []),
            "<section>",
            "<h2>分组汇总 (Groups)</h2>",
            group_section,
            "</section>",
            "<section>",
            "<h2>逐层明细 (Layers)</h2>",
            layer_section,
            "</section>",
            f'<footer class="note">{notes}</footer>',
            "</main>",
            "</body>",
            "</html>",
        ]
    ) + "\n"


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def render_report(graph: dict[str, Any], fmt: str = "md") -> str:
    """Render ``graph`` as ``md``, ``csv`` or a self-contained ``html`` page."""
    fmt = str(fmt or "md").lower()
    graph = _prepare(graph)
    if fmt == "md":
        return _render_md(graph)
    if fmt == "csv":
        return _render_csv(graph)
    if fmt == "html":
        return _render_html(graph)
    raise ValueError(f"不支持的报告格式 {fmt!r}（可选 md/csv/html）")


def write_report(graph: dict[str, Any], out: str | Path, fmt: str) -> None:
    """Write :func:`render_report` output; CSV files are UTF-8-SIG for Excel."""
    fmt = str(fmt or "md").lower()
    text = render_report(graph, fmt=fmt)
    path = Path(out)
    if fmt == "csv":
        if text.startswith("\ufeff"):
            text = text[1:]
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            handle.write(text)
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(text)


__all__ = ["render_report", "write_report"]
