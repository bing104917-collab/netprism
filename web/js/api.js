/* Data access: live NetPrism API with a static demo fallback. */

const API_HEALTH = "/api/health";
const API_INDEX = "/api/index";
const API_MODEL = "/api/model/";
const API_UPLOAD = "/api/upload-parse";
const DEMO_INDEX = "demo/index.json";

export class ApiError extends Error {
  constructor(message, { status = 0, url = "", cause = null } = {}) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.url = url;
    this.cause = cause;
  }
}

function withTimeout(ms) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), ms);
  return { signal: controller.signal, done: () => clearTimeout(timer) };
}

async function fetchJson(url, { timeout = 15000, ...options } = {}) {
  const { signal, done } = withTimeout(timeout);
  try {
    const response = await fetch(url, {
      cache: "no-store",
      signal,
      headers: { Accept: "application/json" },
      ...options,
    });
    if (!response.ok) {
      throw new ApiError(`HTTP ${response.status} ${response.statusText || ""}`.trim(), {
        status: response.status,
        url,
      });
    }
    const text = await response.text();
    try {
      return JSON.parse(text);
    } catch (error) {
      throw new ApiError("响应不是合法 JSON", { status: response.status, url, cause: error });
    }
  } catch (error) {
    if (error instanceof ApiError) throw error;
    if (error && error.name === "AbortError") {
      throw new ApiError("请求超时", { url, cause: error });
    }
    throw new ApiError("无法连接服务", { url, cause: error });
  } finally {
    done();
  }
}

/** Probe /api/health. Never throws; a down server is a valid state. */
export async function checkHealth(timeout = 1600) {
  try {
    const data = await fetchJson(API_HEALTH, { timeout });
    return {
      online: Boolean(data && data.ok),
      static: Boolean(data && data.static),
      version: data && data.version,
    };
  } catch (error) {
    return { online: false, static: false, error };
  }
}

function normalizeIndex(raw, source) {
  if (!raw || typeof raw !== "object" || !Array.isArray(raw.models)) {
    throw new ApiError("模型索引格式无效：缺少 models 数组", { url: source });
  }
  const models = raw.models
    .filter((model) => model && typeof model === "object" && model.id)
    .map((model) => ({
      ...model,
      id: String(model.id),
      name: model.name || model.className || String(model.id),
      className: model.className ?? null,
      file: model.file || `${model.id}.json`,
      mode: model.mode || "static",
      numLayers: Number(model.numLayers) || 0,
      totalParams: Number(model.totalParams) || 0,
      totalFlops: Number(model.totalFlops) || 0,
      warnings: Number(model.warnings) || 0,
    }));
  return { ...raw, models, source };
}

/**
 * Load the model index.
 * `/api/health` is a cheap probe; if it fails we try `/api/index` once anyway,
 * then fall back to the bundled demo index. The response describes where the
 * data came from so the UI can show a badge.
 */
export async function loadIndex({ forceDemo = false } = {}) {
  let apiError = null;
  if (!forceDemo) {
    const health = await checkHealth();
    if (!health.static) {
      try {
        const raw = await fetchJson(API_INDEX, { timeout: 8000 });
        const index = normalizeIndex(raw, "api");
        return { index, source: "api", health, apiError: null };
      } catch (error) {
        apiError = error;
      }
    } else {
      apiError = new ApiError("静态预览模式（web/api/health 标记）");
    }
  }

  try {
    const raw = await fetchJson(DEMO_INDEX, { timeout: 8000 });
    const index = normalizeIndex(raw, "demo");
    return { index, source: "demo", health: { online: false }, apiError };
  } catch (error) {
    const reason = apiError ? `${apiError.message}；${error.message}` : error.message;
    throw new ApiError(`无法加载模型索引：${reason}`, {
      status: error.status || apiError?.status || 0,
      url: DEMO_INDEX,
      cause: error,
    });
  }
}

/** Fetch one graph for a model entry. */
export async function loadGraph(entry, source) {
  if (!entry || !entry.id) throw new ApiError("缺少模型 id");
  if (source === "api") {
    return fetchJson(`${API_MODEL}${encodeURIComponent(entry.id)}`, { timeout: 20000 });
  }
  if (source === "demo") {
    const file = String(entry.file || `${entry.id}.json`).replace(/^\.?\//, "");
    return fetchJson(`demo/${file}`, { timeout: 20000 });
  }
  if (source === "upload") {
    throw new ApiError("上传模型应使用响应中的 graphs");
  }
  throw new ApiError(`未知数据源：${source}`);
}

/** POST .py sources to the static parser. */
export async function uploadParse(files, { timeout = 60000 } = {}) {
  if (!Array.isArray(files) || files.length === 0) {
    throw new ApiError("没有可解析的 .py 文件");
  }
  const payload = { files, execute: false };
  return fetchJson(API_UPLOAD, {
    method: "POST",
    timeout,
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify(payload),
  });
}

export function isUploadUnsupported(error) {
  if (!(error instanceof ApiError)) return false;
  return [404, 405, 501].includes(error.status) || error.status === 0;
}

/** Loose graph sanity check so bad payloads become friendly errors. */
export function assertGraph(graph, id) {
  if (!graph || typeof graph !== "object") {
    throw new ApiError(`模型 ${id} 数据为空`);
  }
  if (!Array.isArray(graph.nodes) || !Array.isArray(graph.edges) || !Array.isArray(graph.groups)) {
    throw new ApiError(`模型 ${id} 缺少 nodes / edges / groups`);
  }
  if (!graph.model || typeof graph.model !== "object") {
    throw new ApiError(`模型 ${id} 缺少 model 元信息`);
  }
  return graph;
}
