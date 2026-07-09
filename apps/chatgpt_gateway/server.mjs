import { createServer } from "node:http";
import { spawn } from "node:child_process";
import { Readable } from "node:stream";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { createChatGPTHandler } from "@opencoredev/loginwithchatgpt-server";

const port = Number(process.env.PORT || "10000");
const host = process.env.HOST || "0.0.0.0";
const pythonPort = Number(process.env.PYTHON_PORT || "10001");
const pythonBin = process.env.PYTHON_BIN || "python3";
const pythonUrl = (process.env.SHERman_GATEWAY_PYTHON_URL || process.env.SHERMAN_GATEWAY_PYTHON_URL || `http://127.0.0.1:${pythonPort}`).replace(/\/+$/, "");
const startPython = process.env.SHERMAN_GATEWAY_START_PYTHON !== "false" && !process.env.SHERMAN_GATEWAY_PYTHON_URL;
const sessionStorePath = process.env.LWC_SESSION_STORE_PATH || "/tmp/sherman-chatgpt-sessions.json";
const maxRequestBytes = Number(process.env.LWC_MAX_RESPONSES_BYTES || String(40 * 1024 * 1024));
const allowedModels = (process.env.LWC_ALLOWED_MODELS || "gpt-5.5,gpt-5.4,gpt-5.4-mini,gpt-5.3-codex-spark")
  .split(",")
  .map((item) => item.trim())
  .filter(Boolean);

class FileSessionStore {
  constructor(path) {
    this.path = path;
  }

  async get(key) {
    const entries = await this.read();
    const entry = entries[key];
    if (!entry) return undefined;
    if (entry.expiresAt !== undefined && entry.expiresAt <= Date.now()) {
      delete entries[key];
      await this.write(entries);
      return undefined;
    }
    return entry.value;
  }

  async set(key, value, options = {}) {
    const entries = await this.read();
    entries[key] = {
      value,
      expiresAt: options.ttlMs !== undefined ? Date.now() + options.ttlMs : undefined,
    };
    await this.write(entries);
  }

  async delete(key) {
    const entries = await this.read();
    delete entries[key];
    await this.write(entries);
  }

  async read() {
    try {
      return JSON.parse(await readFile(this.path, "utf8"));
    } catch {
      return {};
    }
  }

  async write(entries) {
    await mkdir(dirname(this.path), { recursive: true });
    await writeFile(this.path, `${JSON.stringify(entries)}\n`, { mode: 0o600 });
  }
}

const auth = createChatGPTHandler({
  basePath: "/api/chatgpt",
  secret: process.env.LWC_SECRET || process.env.SECRET_KEY,
  sessionStore: new FileSessionStore(sessionStorePath),
  defaultModel: process.env.SHERMAN_CHAT_MODEL || "gpt-5.5",
  responsesProxy: {
    allowedModels: allowedModels.length > 0 ? allowedModels : undefined,
    maxRequestBytes,
    rateLimit: {
      limit: Number(process.env.LWC_RATE_LIMIT || "30"),
      windowMs: Number(process.env.LWC_RATE_WINDOW_MS || "60000"),
    },
  },
  instructions: "You are ShermanAI, a concise manual-grounded production support assistant.",
  reasoningEffort: process.env.SHERMAN_CHAT_REASONING_EFFORT || "low",
  textVerbosity: "low",
});

let pythonProcess;
if (startPython) {
  const env = {
    ...process.env,
    PORT: String(pythonPort),
    SHERMAN_CHAT_FRONTEND_DIR: process.env.SHERMAN_CHAT_FRONTEND_DIR || "/app/frontend",
  };
  pythonProcess = spawn(
    pythonBin,
    ["-m", "uvicorn", "apps.api.chat_main:app", "--host", "127.0.0.1", "--port", String(pythonPort)],
    { env, stdio: ["ignore", "inherit", "inherit"] },
  );
  pythonProcess.on("error", (error) => {
    console.error(`[sherman-gateway] failed to start Python API with ${pythonBin}`, error);
    process.exit(1);
  });
  pythonProcess.on("exit", (code, signal) => {
    console.error(`[sherman-gateway] Python API exited with code=${code} signal=${signal}`);
    process.exit(code ?? 1);
  });
}

const server = createServer(async (req, res) => {
  try {
    const request = nodeRequestToWeb(req);
    const url = new URL(request.url);
    let response;
    if (url.pathname === "/api/chatgpt/complete") {
      response = await handleJsonCompletion(request);
    } else if (url.pathname.startsWith("/api/chatgpt/")) {
      response = await auth.handler(request);
    } else {
      response = await proxyToPython(request);
    }
    await sendWebResponse(res, response);
  } catch (error) {
    console.error("[sherman-gateway] request failed", error);
    if (!res.headersSent) {
      res.writeHead(502, { "content-type": "application/json", "cache-control": "no-store" });
    }
    res.end(JSON.stringify({ error: "gateway_error", message: "ShermanChat gateway request failed." }));
  }
});

server.listen(port, host, () => {
  console.log(`[sherman-gateway] listening on ${host}:${port}, python upstream ${pythonUrl}`);
});

for (const signal of ["SIGINT", "SIGTERM"]) {
  process.on(signal, () => {
    server.close(() => process.exit(0));
    if (pythonProcess && !pythonProcess.killed) pythonProcess.kill(signal);
    setTimeout(() => process.exit(0), 5000).unref();
  });
}

function nodeRequestToWeb(req) {
  const headers = new Headers();
  for (const [key, value] of Object.entries(req.headers)) {
    if (Array.isArray(value)) {
      for (const item of value) headers.append(key, item);
    } else if (value !== undefined) {
      headers.set(key, value);
    }
  }
  const proto = String(headers.get("x-forwarded-proto") || "http").split(",")[0].trim() || "http";
  const host = headers.get("host") || `127.0.0.1:${port}`;
  const url = `${proto}://${host}${req.url || "/"}`;
  const method = req.method || "GET";
  const init = { method, headers };
  if (method !== "GET" && method !== "HEAD") {
    init.body = Readable.toWeb(req);
    init.duplex = "half";
  }
  return new Request(url, init);
}

async function proxyToPython(request) {
  const target = new URL(request.url);
  const upstream = new URL(`${pythonUrl}${target.pathname}${target.search}`);
  const headers = new Headers(request.headers);
  const init = {
    method: request.method,
    headers,
    redirect: "manual",
  };
  if (request.method !== "GET" && request.method !== "HEAD") {
    init.body = request.body;
    init.duplex = "half";
  }
  return fetch(upstream, init);
}

async function handleJsonCompletion(request) {
  if (request.method !== "POST") {
    return json({ error: "method_not_allowed" }, { status: 405 });
  }

  const body = await request.text();
  const headers = new Headers({
    "content-type": "application/json",
    accept: "text/event-stream",
  });
  const cookie = request.headers.get("cookie");
  if (cookie) headers.set("cookie", cookie);
  const forwardedProto = request.headers.get("x-forwarded-proto");
  if (forwardedProto) headers.set("x-forwarded-proto", forwardedProto);
  const origin = request.headers.get("origin");
  if (origin) headers.set("origin", origin);

  const responsesUrl = new URL("/api/chatgpt/responses", request.url);
  const upstream = await auth.handler(new Request(responsesUrl.toString(), { method: "POST", headers, body }));
  const raw = await upstream.text();
  if (!upstream.ok) {
    return new Response(raw, {
      status: upstream.status,
      headers: {
        "content-type": upstream.headers.get("content-type") || "application/json",
        "cache-control": "no-store",
      },
    });
  }

  const outputText = extractResponseText(raw);
  return json({
    output_text: outputText,
    model: extractRequestedModel(body),
    response_format: "login-with-chatgpt",
  });
}

function extractRequestedModel(body) {
  try {
    const parsed = JSON.parse(body);
    return typeof parsed.model === "string" ? parsed.model : undefined;
  } catch {
    return undefined;
  }
}

function extractResponseText(raw) {
  const jsonText = tryExtractJsonResponseText(raw);
  if (jsonText) return jsonText;

  let deltaText = "";
  let completedText = "";
  for (const event of parseServerSentEvents(raw)) {
    if (!event || typeof event !== "object") continue;
    const type = typeof event.type === "string" ? event.type : "";
    if (typeof event.delta === "string" && type.includes("output_text")) {
      deltaText += event.delta;
      continue;
    }
    if (typeof event.text === "string" && type.endsWith(".delta")) {
      deltaText += event.text;
      continue;
    }
    if (type === "response.completed" || type === "response.done") {
      completedText ||= textFromResponseObject(event.response);
    }
  }
  return (deltaText || completedText).trim();
}

function tryExtractJsonResponseText(raw) {
  try {
    return textFromResponseObject(JSON.parse(raw));
  } catch {
    return "";
  }
}

function parseServerSentEvents(raw) {
  const events = [];
  for (const block of raw.split(/\r?\n\r?\n/)) {
    const dataLines = block
      .split(/\r?\n/)
      .filter((line) => line.startsWith("data:"))
      .map((line) => line.slice(5).trimStart());
    if (dataLines.length === 0) continue;
    const data = dataLines.join("\n").trim();
    if (!data || data === "[DONE]") continue;
    try {
      events.push(JSON.parse(data));
    } catch {
      // Ignore non-JSON SSE events.
    }
  }
  return events;
}

function textFromResponseObject(value) {
  if (!value || typeof value !== "object") return "";
  if (typeof value.output_text === "string") return value.output_text.trim();
  const chunks = [];
  const output = Array.isArray(value.output) ? value.output : [];
  for (const item of output) {
    const content = item && typeof item === "object" && Array.isArray(item.content) ? item.content : [];
    for (const part of content) {
      if (part && typeof part === "object" && typeof part.text === "string") chunks.push(part.text);
    }
  }
  return chunks.join("\n").trim();
}

async function sendWebResponse(res, response) {
  res.statusCode = response.status;
  response.headers.forEach((value, key) => {
    res.setHeader(key, value);
  });
  if (!response.body) {
    res.end();
    return;
  }
  Readable.fromWeb(response.body).pipe(res);
}

function json(data, init = {}) {
  return new Response(JSON.stringify(data), {
    status: init.status || 200,
    headers: { "content-type": "application/json", "cache-control": "no-store" },
  });
}
