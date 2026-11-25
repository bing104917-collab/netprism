#!/usr/bin/env node
/**
 * NetPrism headless screenshot helper (no dependencies).
 *
 * Usage (from web/):
 *   node scripts/screenshot.mjs --out ../scratch/frontend/shot.png
 *   node scripts/screenshot.mjs --url http://127.0.0.1:8765/ --out shot.png
 *
 * Starts a tiny static server for web/ unless --url is given, launches the
 * locally installed Chrome / Edge headlessly with SwiftShader WebGL, and waits
 * for the PNG to be written.
 */

import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import os from "node:os";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";

const WEB_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

function parseArgs(argv) {
  const out = {
    url: "",
    out: path.join(WEB_ROOT, "screenshot.png"),
    width: 1600,
    height: 1000,
    port: 0,
    budget: 12000,
    chrome: "",
    keepProfile: false,
  };
  for (let i = 2; i < argv.length; i += 1) {
    const arg = argv[i];
    const next = argv[i + 1];
    switch (arg) {
      case "--url":
        out.url = next;
        i += 1;
        break;
      case "--out":
        out.out = path.resolve(next);
        i += 1;
        break;
      case "--width":
        out.width = Number(next) || 1600;
        i += 1;
        break;
      case "--height":
        out.height = Number(next) || 1000;
        i += 1;
        break;
      case "--port":
        out.port = Number(next) || 0;
        i += 1;
        break;
      case "--budget":
        out.budget = Number(next) || 12000;
        i += 1;
        break;
      case "--chrome":
        out.chrome = next;
        i += 1;
        break;
      case "--keep-profile":
        out.keepProfile = true;
        break;
      case "--help":
      case "-h":
        printHelp();
        process.exit(0);
        break;
      default:
        if (!arg.startsWith("-")) out.out = path.resolve(arg);
        break;
    }
  }
  return out;
}

function printHelp() {
  console.log(`NetPrism screenshot helper
  --url <url>         use an already running server (skips built-in static server)
  --out <file>        output PNG (default web/screenshot.png)
  --width <px>        viewport width (default 1600)
  --height <px>       viewport height (default 1000)
  --port <port>       static server port (default: random free port)
  --budget <ms>       Chrome virtual time budget (default 12000)
  --chrome <path>     Chrome/Edge executable
  --keep-profile      keep the temporary Chrome profile
`);
}

function findChrome(explicit) {
  const candidates = [
    explicit,
    process.env.CHROME_PATH,
    "C:\\\\Program Files\\\\Google\\\\Chrome\\\\Application\\\\chrome.exe",
    "C:\\\\Program Files (x86)\\\\Google\\\\Chrome\\\\Application\\\\chrome.exe",
    "C:\\\\Program Files (x86)\\\\Microsoft\\\\Edge\\\\Application\\\\msedge.exe",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  ].filter(Boolean);
  for (const candidate of candidates) {
    if (fs.existsSync(candidate)) return candidate;
  }
  throw new Error("找不到 Chrome/Edge。请用 --chrome 或 CHROME_PATH 指定可执行文件。");
}

const MIME = {
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
  ".txt": "text/plain; charset=utf-8",
  ".md": "text/markdown; charset=utf-8",
};

function startStaticServer(port) {
  const server = http.createServer((request, response) => {
    try {
      const pathname = decodeURIComponent(new URL(request.url, "http://127.0.0.1").pathname);
      let filePath = path.resolve(WEB_ROOT, `.${pathname === "/" ? "/index.html" : pathname}`);
      if (!filePath.startsWith(WEB_ROOT + path.sep) && filePath !== path.join(WEB_ROOT, "index.html")) {
        response.writeHead(403);
        response.end("Forbidden");
        return;
      }
      if (fs.existsSync(filePath) && fs.statSync(filePath).isDirectory()) {
        filePath = path.join(filePath, "index.html");
      }
      if (!fs.existsSync(filePath) || !fs.statSync(filePath).isFile()) {
        response.writeHead(404, { "Content-Type": "text/plain; charset=utf-8" });
        response.end(`Not found: ${pathname}`);
        return;
      }
      response.writeHead(200, {
        "Content-Type": MIME[path.extname(filePath).toLowerCase()] || "application/octet-stream",
        "Cache-Control": "no-store",
      });
      fs.createReadStream(filePath).pipe(response);
    } catch (error) {
      response.writeHead(500, { "Content-Type": "text/plain; charset=utf-8" });
      response.end(String(error));
    }
  });
  return new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(port, "127.0.0.1", () => resolve(server));
  });
}

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function waitForStableFile(filePath, timeoutMs) {
  const deadline = Date.now() + timeoutMs;
  let lastSize = -1;
  let stableSince = 0;
  while (Date.now() < deadline) {
    if (fs.existsSync(filePath)) {
      const size = fs.statSync(filePath).size;
      if (size > 1000) {
        if (size === lastSize) {
          if (Date.now() - stableSince > 350) return true;
        } else {
          lastSize = size;
          stableSince = Date.now();
        }
      }
    }
    await delay(250);
  }
  return false;
}

async function main() {
  const args = parseArgs(process.argv);
  if (!fs.existsSync(path.dirname(args.out))) fs.mkdirSync(path.dirname(args.out), { recursive: true });
  if (fs.existsSync(args.out)) fs.rmSync(args.out, { force: true });

  const chrome = findChrome(args.chrome);
  let server = null;
  let url = args.url;
  if (!url) {
    server = await startStaticServer(args.port);
    url = `http://127.0.0.1:${server.address().port}/`;
  }

  const profile = fs.mkdtempSync(path.join(os.tmpdir(), "netprism-shot-"));
  const chromeArgs = [
    "--headless=new",
    "--disable-gpu",
    "--use-gl=angle",
    "--use-angle=swiftshader",
    "--enable-unsafe-swiftshader",
    "--hide-scrollbars",
    "--no-first-run",
    "--no-default-browser-check",
    "--disable-background-networking",
    "--disable-component-update",
    "--force-device-scale-factor=1",
    `--virtual-time-budget=${args.budget}`,
    `--window-size=${args.width},${args.height}`,
    `--user-data-dir=${profile}`,
    `--screenshot=${args.out}`,
    url,
  ];

  console.log(`[netprism] Chrome: ${chrome}`);
  console.log(`[netprism] URL:    ${url}`);
  console.log(`[netprism] Output: ${args.out}`);
  const child = spawn(chrome, chromeArgs, { stdio: "ignore", windowsHide: true });
  let exitCode = null;
  child.on("exit", (code) => {
    exitCode = code;
  });

  const ok = await waitForStableFile(args.out, 90000);
  if (server) server.close();
  if (!ok) {
    console.error(`[netprism] 截图失败（Chrome exit=${exitCode}）。若使用无头 WebGL，请确认 Chrome 支持 SwiftShader。`);
    process.exitCode = 1;
  } else {
    const size = fs.statSync(args.out).size;
    console.log(`[netprism] OK ${(size / 1024).toFixed(1)} KB`);
  }
  if (!args.keepProfile) {
    try {
      await delay(400);
      fs.rmSync(profile, { recursive: true, force: true });
    } catch {
      /* profile may still be locked on Windows; harmless */
    }
  }
  setTimeout(() => process.exit(process.exitCode || 0), 100);
}

main().catch((error) => {
  console.error("[netprism] screenshot failed:", error && error.message ? error.message : error);
  process.exit(1);
});
