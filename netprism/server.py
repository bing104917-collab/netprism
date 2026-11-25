"""Small, dependency-free HTTP server that powers the NetPrism 3D viewer.

It serves the static ``web/`` application and exposes a handful of JSON
endpoints.  All analysis is delegated to :mod:`netprism.core`; the server never
imports torch itself, so it can serve pre-parsed results on machines without
PyTorch installed.
"""

from __future__ import annotations

import json
import logging
import mimetypes
import os
import re
import tempfile
import threading
import time
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import unquote, urlparse

from .schema import DEMO_DIR, GENERATOR_VERSION, SCHEMA_VERSION, WEB_DIR, validate_graph, validate_index

log = logging.getLogger("netprism.server")

MAX_UPLOAD_BYTES = 8 * 1024 * 1024
MAX_UPLOAD_FILE_BYTES = 1_500_000
MAX_UPLOAD_FILES = 300
MODEL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,159}$")

_STATIC_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".ico": "image/x-icon",
    ".woff2": "font/woff2",
    ".map": "application/json; charset=utf-8",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _empty_index(root: str) -> dict[str, Any]:
    return {
        "schemaVersion": SCHEMA_VERSION,
        "generatedAt": _now_iso(),
        "root": root,
        "models": [],
        "errors": [],
    }


class ServerApp:
    """Holds the data directory plus any models uploaded during this session."""

    def __init__(self, data_dir: str | Path, web_dir: str | Path | None = None, *, allow_execute: bool = False):
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.web_dir = Path(web_dir).expanduser().resolve() if web_dir else WEB_DIR
        self.allow_execute = allow_execute
        self.overlay_models: dict[str, dict[str, Any]] = {}
        self.overlay_graphs: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    # -- index ---------------------------------------------------------
    def base_index(self) -> dict[str, Any] | None:
        path = self.data_dir / "index.json"
        if not path.is_file():
            return None
        try:
            with path.open("r", encoding="utf-8") as fh:
                index = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("cannot read %s: %s", path, exc)
            return None
        if not isinstance(index, dict):
            return None
        return index

    def build_index(self) -> dict[str, Any] | None:
        base = self.base_index()
        with self._lock:
            overlay = list(self.overlay_models.values())
        if base is None and not overlay:
            return None
        if base is None:
            index = _empty_index(str(self.data_dir))
        else:
            index = json.loads(json.dumps(base))  # cheap deep copy
        existing = {m.get("id") for m in index.get("models", [])}
        for entry in overlay:
            if entry.get("id") not in existing:
                index.setdefault("models", []).append(dict(entry))
                existing.add(entry.get("id"))
        index["generatedAt"] = index.get("generatedAt") or _now_iso()
        return index

    def model_graph(self, model_id: str) -> dict[str, Any] | None:
        if not MODEL_ID_RE.match(model_id):
            return None
        with self._lock:
            if model_id in self.overlay_graphs:
                return self.overlay_graphs[model_id]
        index = self.base_index()
        if not index:
            return None
        for entry in index.get("models", []):
            if entry.get("id") != model_id:
                continue
            filename = entry.get("file")
            if not isinstance(filename, str) or not filename:
                return None
            path = (self.data_dir / filename).resolve()
            try:
                path.relative_to(self.data_dir)
            except ValueError:
                return None
            if not path.is_file():
                return None
            try:
                with path.open("r", encoding="utf-8") as fh:
                    graph = json.load(fh)
            except (OSError, json.JSONDecodeError):
                return None
            return graph if isinstance(graph, dict) else None
        return None

    # -- analysis ------------------------------------------------------
    def analyze(self, path: str | Path, *, execute: bool, class_name: str | None = None,
                max_files: int = 200) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
        """Analyze a folder and return (index, graphs). Thread-safe."""
        from .core import AnalyzeOptions, analyze_folder  # lazy: torch optional

        if execute and not self.allow_execute:
            raise PermissionError(
                "执行模型代码在当前服务中未启用；请用 --allow-execute 启动，或使用静态解析。"
            )
        result = analyze_folder(
            path,
            AnalyzeOptions(execute=execute, class_name=class_name, max_files=max_files),
        )
        from .selection import fold_variants

        index, graphs, folded = fold_variants(result.index, result.graphs, enabled=True)
        try:
            from .metrics import enrich_graph
        except ImportError:
            enrich_graph = None  # type: ignore[assignment]
        if enrich_graph is not None:
            for graph in graphs.values():
                try:
                    enrich_graph(graph)
                except Exception:
                    log.debug("metrics enrichment failed", exc_info=True)
        try:
            from .algorithms import enrich_algorithms
        except ImportError:
            enrich_algorithms = None  # type: ignore[assignment]
        if enrich_algorithms is not None:
            for graph in graphs.values():
                try:
                    enrich_algorithms(graph)
                except Exception:
                    log.debug("algorithm enrichment failed", exc_info=True)
        if folded:
            log.info("folded %d architectural variants for the UI", len(folded))
        # The API returns everything the UI needs; validate defensively.
        bad = [gid for gid, graph in graphs.items() if validate_graph(graph)]
        if bad:
            log.warning("graphs failed validation: %s", ", ".join(bad))
        if validate_index(index):
            log.warning("index failed validation: %s", validate_index(index)[:3])
        return index, graphs

    def _unique_id(self, wanted: str, taken: set[str]) -> str:
        candidate = wanted
        suffix = 2
        while candidate in taken:
            candidate = f"{wanted}-{suffix}"
            suffix += 1
        return candidate

    def ingest_upload(self, files: list[dict[str, Any]], *, execute: bool = False) -> dict[str, Any]:
        """Write uploaded source files to a temp dir, analyze statically, cache graphs."""
        if execute and not self.allow_execute:
            raise PermissionError("执行模型代码未启用")
        if not isinstance(files, list) or not files:
            raise ValueError("files 不能为空")
        if len(files) > MAX_UPLOAD_FILES:
            raise ValueError(f"文件数量超过上限 {MAX_UPLOAD_FILES}")
        total = 0
        with tempfile.TemporaryDirectory(prefix="netprism_upload_") as tmp:
            root = Path(tmp).resolve()
            written = 0
            for item in files:
                if not isinstance(item, dict):
                    continue
                raw_path = item.get("path") or item.get("name")
                content = item.get("content")
                if not isinstance(raw_path, str) or not isinstance(content, str):
                    continue
                rel = PurePosixPath(raw_path.replace("\\", "/"))
                parts = [p for p in rel.parts if p not in ("", ".", "..")]
                if not parts:
                    continue
                if not parts[-1].endswith(".py"):
                    continue
                encoded = content.encode("utf-8", errors="replace")
                if len(encoded) > MAX_UPLOAD_FILE_BYTES:
                    raise ValueError(f"{raw_path}: 单文件超过 {MAX_UPLOAD_FILE_BYTES // 1024} KB")
                total += len(encoded)
                if total > MAX_UPLOAD_BYTES:
                    raise ValueError(f"上传总大小超过 {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
                target = (root.joinpath(*parts)).resolve()
                try:
                    target.relative_to(root)
                except ValueError:
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")
                written += 1
            if not written:
                raise ValueError("没有收到可解析的 .py 文件")

            index, graphs = self.analyze(root, execute=execute)
            # Persist a stable, relocatable source path in the served graphs.
            for graph in graphs.values():
                model = graph.get("model", {})
                source = model.get("source") or {}
                rel_path = source.get("relPath") or source.get("file") or ""
                if rel_path:
                    model["source"] = {"file": rel_path, "relPath": rel_path}
                for node in graph.get("nodes", []):
                    node_src = node.get("source")
                    if isinstance(node_src, dict):
                        node_src["file"] = rel_path

        by_old_id = {m.get("id"): m for m in index.get("models", [])}
        with self._lock:
            taken = set(self.overlay_models) | {
                m.get("id") for m in (self.base_index() or {}).get("models", [])
            }
            response_graphs: dict[str, dict[str, Any]] = {}
            for old_id, graph in graphs.items():
                new_id = self._unique_id(old_id, taken)
                taken.add(new_id)
                model = graph.get("model", {})
                entry = dict(by_old_id.get(old_id, {}))
                entry["id"] = new_id
                entry.setdefault("file", f"{new_id}.json")
                entry.setdefault("name", model.get("name", new_id))
                entry.setdefault("className", model.get("className"))
                entry.setdefault("mode", model.get("mode", "static"))
                entry.setdefault("numLayers", model.get("numLayers", 0))
                entry.setdefault("totalParams", model.get("totalParams", 0))
                entry.setdefault("totalFlops", model.get("totalFlops", 0))
                entry.setdefault("warnings", len(model.get("warnings", [])))
                entry.setdefault("source", (model.get("source") or {}).get("relPath", ""))
                self.overlay_models[new_id] = entry
                self.overlay_graphs[new_id] = graph
                response_graphs[new_id] = graph
        merged = self.build_index() or _empty_index("uploaded")
        merged["errors"] = list(index.get("errors", []))
        merged["generatedAt"] = _now_iso()
        return {"index": merged, "graphs": response_graphs, "errors": index.get("errors", [])}

    def ingest_scan(self, path: str, *, execute: bool = False, class_name: str | None = None,
                    max_files: int = 200) -> dict[str, Any]:
        folder = Path(path).expanduser()
        if not folder.is_dir():
            raise FileNotFoundError(f"目录不存在: {folder}")
        index, graphs = self.analyze(folder, execute=execute, class_name=class_name, max_files=max_files)
        by_old_id = {m.get("id"): m for m in index.get("models", [])}
        with self._lock:
            taken = set(self.overlay_models) | {
                m.get("id") for m in (self.base_index() or {}).get("models", [])
            }
            response_graphs: dict[str, dict[str, Any]] = {}
            for old_id, graph in graphs.items():
                new_id = self._unique_id(old_id, taken)
                taken.add(new_id)
                model = graph.get("model", {})
                entry = dict(by_old_id.get(old_id, {}))
                entry["id"] = new_id
                entry.setdefault("file", f"{new_id}.json")
                entry.setdefault("name", model.get("name", new_id))
                entry.setdefault("className", model.get("className"))
                entry.setdefault("mode", model.get("mode", "static"))
                entry.setdefault("numLayers", model.get("numLayers", 0))
                entry.setdefault("totalParams", model.get("totalParams", 0))
                entry.setdefault("totalFlops", model.get("totalFlops", 0))
                entry.setdefault("warnings", len(model.get("warnings", [])))
                entry.setdefault("source", (model.get("source") or {}).get("relPath", ""))
                self.overlay_models[new_id] = entry
                self.overlay_graphs[new_id] = graph
                response_graphs[new_id] = graph
        merged = self.build_index() or _empty_index(str(folder))
        merged["errors"] = list(index.get("errors", []))
        merged["generatedAt"] = _now_iso()
        return {"index": merged, "graphs": response_graphs, "errors": index.get("errors", [])}


class NetPrismHandler(BaseHTTPRequestHandler):
    """HTTP handler bound to a :class:`ServerApp` instance."""

    server_version = f"NetPrism/{GENERATOR_VERSION}"
    protocol_version = "HTTP/1.1"

    def __init__(self, *args: Any, app: ServerApp, **kwargs: Any):
        self.app = app
        super().__init__(*args, **kwargs)

    # -- helpers -------------------------------------------------------
    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
        log.debug("%s - %s", self.address_string(), fmt % args)

    def _send_json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: int, message: str) -> None:
        self._send_json({"error": message, "status": status}, status)

    def _read_json(self, max_bytes: int = MAX_UPLOAD_BYTES) -> dict[str, Any]:
        raw_length = self.headers.get("Content-Length")
        if not raw_length:
            raise ValueError("缺少 Content-Length")
        length = int(raw_length)
        if length <= 0 or length > max_bytes:
            raise ValueError(f"请求体过大（上限 {max_bytes // (1024 * 1024)} MB）")
        raw = self.rfile.read(length)
        payload = json.loads(raw.decode("utf-8", errors="replace"))
        if not isinstance(payload, dict):
            raise ValueError("请求体必须是 JSON 对象")
        return payload

    def _is_loopback(self) -> bool:
        host = self.client_address[0]
        return host in ("127.0.0.1", "::1", "localhost")

    # -- routing -------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        try:
            if path == "/api/health":
                self._send_json({"ok": True, "version": GENERATOR_VERSION, "schemaVersion": SCHEMA_VERSION})
                return
            if path == "/api/index":
                index = self.app.build_index()
                if index is None:
                    self._error(404, "还没有解析结果。请先运行 netprism scan，或上传模型文件夹。")
                    return
                self._send_json(index)
                return
            if path.startswith("/api/model/"):
                model_id = path[len("/api/model/"):].strip("/")
                graph = self.app.model_graph(model_id)
                if graph is None:
                    self._error(404, f"模型不存在: {model_id}")
                    return
                self._send_json(graph)
                return
            if path.startswith("/api/"):
                self._error(404, f"未知 API: {path}")
                return
            self._serve_static(path)
        except BrokenPipeError:
            pass
        except Exception as exc:  # pragma: no cover - defensive
            log.exception("GET %s failed", path)
            self._error(500, f"服务器错误: {exc}")

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        try:
            if not self._is_loopback():
                self._error(403, "仅允许本机访问分析接口")
                return
            if path == "/api/upload-parse":
                payload = self._read_json()
                result = self.app.ingest_upload(
                    payload.get("files") or [],
                    execute=bool(payload.get("execute")),
                )
                self._send_json(result)
                return
            if path == "/api/scan":
                payload = self._read_json(max_bytes=64 * 1024)
                result = self.app.ingest_scan(
                    str(payload.get("path") or ""),
                    execute=bool(payload.get("execute")),
                    class_name=payload.get("className") or payload.get("class_name"),
                    max_files=int(payload.get("maxFiles") or 200),
                )
                self._send_json(result)
                return
            self._error(404, f"未知 API: {path}")
        except PermissionError as exc:
            self._error(403, str(exc))
        except (ValueError, FileNotFoundError, json.JSONDecodeError) as exc:
            self._error(400, str(exc))
        except BrokenPipeError:
            pass
        except Exception as exc:  # pragma: no cover - defensive
            log.exception("POST %s failed", path)
            self._error(500, f"服务器错误: {exc}")

    # -- static files --------------------------------------------------
    def _serve_static(self, path: str) -> None:
        rel = path.lstrip("/") or "index.html"
        if rel.endswith("/"):
            rel += "index.html"
        candidate = (self.app.web_dir / rel).resolve()
        try:
            candidate.relative_to(self.app.web_dir)
        except ValueError:
            self._error(403, "非法路径")
            return
        if candidate.is_dir():
            candidate = candidate / "index.html"
        if not candidate.is_file():
            # SPA fallback for hash routes.
            if "." not in PurePosixPath(rel).name:
                candidate = self.app.web_dir / "index.html"
            if not candidate.is_file():
                self._error(404, f"文件不存在: /{rel}")
                return
        content_type = _STATIC_TYPES.get(candidate.suffix.lower())
        if content_type is None:
            content_type = mimetypes.guess_type(str(candidate))[0] or "application/octet-stream"
        data = candidate.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)


class NetPrismHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], app: ServerApp):
        self.app = app
        super().__init__(address, self._make_handler())

    def _make_handler(self):
        app = self.app

        class _Handler(NetPrismHandler):
            def __init__(self, *args: Any, **kwargs: Any):
                super().__init__(*args, app=app, **kwargs)

        return _Handler


def create_app(data_dir: str | Path, web_dir: str | Path | None = None, *, allow_execute: bool = False) -> ServerApp:
    return ServerApp(data_dir, web_dir, allow_execute=allow_execute)


def create_server(app: ServerApp, host: str = "127.0.0.1", port: int = 8765) -> NetPrismHTTPServer:
    return NetPrismHTTPServer((host, port), app)


def serve(app: ServerApp, host: str = "127.0.0.1", port: int = 8765, *, open_browser: bool = False) -> None:
    httpd = create_server(app, host, port)
    actual_host, actual_port = httpd.server_address[:2]
    display_host = "127.0.0.1" if actual_host in ("0.0.0.0", "::") else actual_host
    url = f"http://{display_host}:{actual_port}/"
    print(f"NetPrism ready at {url}")
    print(f"  data: {app.data_dir}")
    print(f"  web : {app.web_dir}")
    if app.allow_execute:
        print("  !! --allow-execute 已启用：服务端会执行模型源码，仅限可信环境。")
    print("Press Ctrl+C to stop.")
    if open_browser:
        threading.Thread(target=_open_later, args=(url,), daemon=True).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping NetPrism server.")
    finally:
        httpd.server_close()


def _open_later(url: str, delay: float = 0.6) -> None:
    time.sleep(delay)
    try:
        import webbrowser

        webbrowser.open(url)
    except Exception:  # pragma: no cover - best effort
        log.debug("could not open browser", exc_info=True)


def default_demo_dir() -> Path:
    return DEMO_DIR
