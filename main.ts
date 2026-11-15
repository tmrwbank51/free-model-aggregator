// free-model-aggregator: one server serving the OpenAI Chat Completions API
// and the Anthropic Messages API over ten free-tier providers.
//
// The trial hands it a brief (MAKEFUN_PROMPT_FILE): the provider directory,
// the quality table and the workload manifest. It probes every provider's
// catalog route, registers the catalog with the env, then answers each of
// the client's 20 requests with the best free model that meets the
// request's requirements, failing over to the next candidate when a
// provider refuses, and exits 0 when the env says the workload is done.
//
// Everything is derived from the brief, the requests and the providers'
// answers: no wall clock, no unseeded randomness, so a replay leaves the
// same state export. Standard library only; nothing is installed.
import { readFileSync } from "node:fs";
import { createServer } from "node:http";

const FIXED_CREATED = 1767225600;
const MAX_BODY_BYTES = 2 * 1024 * 1024;
const PROVIDER_TIMEOUT_MS = 5000;
const POLL_MS = 200;
const REGISTER_ATTEMPTS = 60;
const DEAD_AFTER_REFUSALS = 3;
const ATTEMPTS_PER_CANDIDATE = 3;
const SSE_PIECE = 13;

const isRecord = (v) => v !== null && typeof v === "object" && !Array.isArray(v);
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

function stringList(v) {
  if (typeof v === "string") return [v];
  if (!Array.isArray(v)) return [];
  return v.filter((x) => typeof x === "string");
}

function numberOrNull(v) {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

// ---------------------------------------------------------------------------
// The brief: three one-line JSON documents in the prompt file.
// ---------------------------------------------------------------------------

function parseBrief(text) {
  let providers = null;
  let ratings = null;
  let manifest = null;
  for (const line of text.split("\n")) {
    const trimmed = line.trim();
    if (!trimmed.startsWith("{")) continue;
    let doc;
    try {
      doc = JSON.parse(trimmed);
    } catch {
      continue;
    }
    if (!isRecord(doc) || typeof doc.schema !== "string") continue;
    if (doc.schema === "free-model-aggregator-01/providers/v1" && Array.isArray(doc.providers)) {
      providers = doc.providers;
    } else if (doc.schema === "free-model-aggregator-01/ratings/v1" && Array.isArray(doc.ratings)) {
      ratings = doc.ratings;
    } else if (doc.schema === "free-model-aggregator-01/workload/v1" && Array.isArray(doc.manifest)) {
      manifest = doc.manifest;
    }
  }
  if (providers === null) throw new Error("the brief holds no provider directory");
  if (ratings === null) throw new Error("the brief holds no quality table");
  if (manifest === null) throw new Error("the brief holds no workload manifest");
  return { providers, ratings, manifest };
}

// ---------------------------------------------------------------------------
// Provider catalogs: probed from each provider's own catalog route, which
// lists every model with its free flag, context length and capabilities.
// ---------------------------------------------------------------------------

function authHeaders(provider) {
  if (provider.auth === "bearer") return { authorization: `Bearer ${provider.key}` };
  if (provider.auth === "x-api-key") return { "x-api-key": provider.key };
  if (provider.auth === "x-goog-api-key") return { "x-goog-api-key": provider.key };
  return {};
}

async function probeCatalog(envUrl, provider) {
  const base = envUrl.replace(/\/$/, "") + provider.path;
  const url = provider.wire === "gemini" ? `${base}/v1beta/models` : `${base}/v1/models`;
  const headers = { ...authHeaders(provider) };
  if (provider.wire === "anthropic") headers["anthropic-version"] = "2023-06-01";
  const res = await fetch(url, { headers, signal: AbortSignal.timeout(PROVIDER_TIMEOUT_MS) });
  if (!res.ok) throw new Error(`catalog route answered ${res.status}`);
  const body = await res.json();
  const rows = [];
  if (provider.wire === "gemini") {
    if (!Array.isArray(body.models)) throw new Error("gemini catalog holds no models array");
    for (const m of body.models) {
      if (!isRecord(m) || typeof m.name !== "string") continue;
      const id = m.name.startsWith("models/") ? m.name.slice("models/".length) : m.name;
      rows.push({
        id,
        free: m.free === true,
        context_length: numberOrNull(m.context_length),
        tools: m.tools === true,
        vision: m.vision === true,
        json: m.json === true,
      });
    }
  } else {
    if (!Array.isArray(body.data)) throw new Error("catalog holds no data array");
    for (const m of body.data) {
      if (!isRecord(m) || typeof m.id !== "string") continue;
      rows.push({
        id: m.id,
        free: m.free === true,
        context_length: numberOrNull(m.context_length),
        tools: m.tools === true,
        vision: m.vision === true,
        json: m.json === true,
      });
    }
  }
  return rows;
}

async function buildCatalog(envUrl, brief) {
  const catalogProviders = [];
  const probed = new Map();
  for (const provider of brief.providers) {
    let models = null;
    if (envUrl !== null && envUrl !== "") {
      try {
        models = await probeCatalog(envUrl, provider);
      } catch (err) {
        process.stderr.write(`catalog probe for ${provider.id} failed (${err.message}); falling back to the brief\n`);
        models = null;
      }
    }
    if (models === null) {
      // No probe: the brief's quality table names every free model.
      models = brief.ratings
        .filter((r) => r.provider === provider.id && r.free === true)
        .map((r) => ({
          id: r.model,
          free: true,
          context_length: numberOrNull(r.context_length),
          tools: r.tools === true,
          vision: r.vision === true,
          json: r.json === true,
        }));
    }
    probed.set(provider.id, models);
    catalogProviders.push({
      id: provider.id,
      base_url: provider.path,
      key_env: provider.key_env,
      wire: provider.wire,
      models: models.map((m) => ({ id: m.id, free: m.free, context_length: m.context_length })),
    });
  }
  return { catalog: { providers: catalogProviders }, probed };
}

// The routable table: every free model with its rating and capabilities,
// from the probed catalogs joined to the brief's quality table.
function buildRoutable(brief, probed) {
  const ratings = new Map();
  for (const r of brief.ratings) ratings.set(`${r.provider}/${r.model}`, r);
  const table = [];
  for (const provider of brief.providers) {
    const models = probed.get(provider.id) ?? [];
    for (const m of models) {
      if (!m.free) continue;
      const row = ratings.get(`${provider.id}/${m.id}`);
      if (row === undefined) continue;
      table.push({
        provider: provider.id,
        model: m.id,
        wire: provider.wire,
        rating: row.rating_milli,
        context_length: m.context_length ?? row.context_length ?? null,
        tools: m.tools || row.tools === true,
        vision: m.vision || row.vision === true,
        json: m.json || row.json === true,
      });
    }
  }
  return table;
}

// ---------------------------------------------------------------------------
// The neutral request: every client request is canonicalized once, then
// rendered into whichever provider wire serves it. Rendering must preserve
// exactly what the provider echoes: the system text, the user text, the
// tool names in order, max_tokens, the stop sequences, the image and the
// JSON-mode flag.
// ---------------------------------------------------------------------------

function blocksText(content) {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  const parts = [];
  for (const block of content) {
    if (isRecord(block) && block.type === "text" && typeof block.text === "string") parts.push(block.text);
  }
  return parts.join("\n");
}

function imageFromUrl(url) {
  if (typeof url !== "string") return null;
  const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
  if (data !== null) return { imageKind: "data", mediaType: data[1] === "" ? "image/png" : data[1], data: data[2] };
  return { imageKind: "url", url };
}

function openAIContentBlocks(content) {
  if (typeof content === "string") return content === "" ? [] : [{ kind: "text", text: content }];
  if (!Array.isArray(content)) return [];
  const blocks = [];
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === "text" && typeof block.text === "string") {
      blocks.push({ kind: "text", text: block.text });
    } else if (block.type === "image_url" && isRecord(block.image_url)) {
      const image = imageFromUrl(block.image_url.url);
      if (image !== null) blocks.push({ kind: "image", ...image });
    } else if (block.type === "input_audio") {
      // No provider echo covers audio; nothing to carry.
    }
  }
  return blocks;
}

function anthropicContentBlocks(content) {
  if (typeof content === "string") return content === "" ? [] : [{ kind: "text", text: content }];
  if (!Array.isArray(content)) return [];
  const blocks = [];
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === "text" && typeof block.text === "string") {
      blocks.push({ kind: "text", text: block.text });
    } else if (block.type === "image" && isRecord(block.source)) {
      if (block.source.type === "base64" && typeof block.source.data === "string") {
        blocks.push({ kind: "image", imageKind: "data", mediaType: block.source.media_type ?? "image/png", data: block.source.data });
      } else if (block.source.type === "url" && typeof block.source.url === "string") {
        blocks.push({ kind: "image", imageKind: "url", url: block.source.url });
      }
    } else if (block.type === "tool_use") {
      blocks.push({ kind: "tool_use", id: block.id ?? "", name: block.name ?? "", input: block.input ?? {} });
    } else if (block.type === "tool_result") {
      blocks.push({ kind: "tool_result", toolUseId: block.tool_use_id ?? "", content: blocksText(block.content) });
    }
  }
  return blocks;
}

function toNeutral(api, body) {
  const messages = [];
  let systemText = "";
  let tools = [];
  let maxTokens = null;
  let stops = [];
  let jsonMode = false;
  let temperature;
  let toolChoice;
  if (api === "openai") {
    const raw = Array.isArray(body.messages) ? body.messages.filter(isRecord) : [];
    for (const m of raw) {
      if (m.role === "system") {
        const text = blocksText(m.content);
        if (text !== "") systemText = systemText === "" ? text : `${systemText}\n${text}`;
        continue;
      }
      const blocks = openAIContentBlocks(m.content);
      if (Array.isArray(m.tool_calls)) {
        for (const call of m.tool_calls) {
          if (!isRecord(call) || !isRecord(call.function)) continue;
          let input = {};
          if (typeof call.function.arguments === "string") {
            try {
              input = JSON.parse(call.function.arguments);
            } catch {
              input = {};
            }
          }
          blocks.push({ kind: "tool_use", id: call.id ?? "", name: call.function.name ?? "", input });
        }
      }
      if (m.role === "tool") {
        blocks.push({ kind: "tool_result", toolUseId: m.tool_call_id ?? "", content: blocksText(m.content) });
      }
      messages.push({ role: m.role, blocks });
    }
    if (Array.isArray(body.tools)) {
      for (const t of body.tools) {
        if (isRecord(t) && isRecord(t.function) && typeof t.function.name === "string") {
          tools.push({ name: t.function.name, description: t.function.description, schema: t.function.parameters });
        }
      }
    }
    maxTokens = numberOrNull(body.max_tokens) ?? numberOrNull(body.max_completion_tokens);
    stops = stringList(body.stop);
    const format = isRecord(body.response_format) ? body.response_format.type : undefined;
    jsonMode = format === "json_object" || format === "json_schema";
    temperature = body.temperature;
    toolChoice = body.tool_choice;
  } else {
    systemText = blocksText(body.system);
    const raw = Array.isArray(body.messages) ? body.messages.filter(isRecord) : [];
    for (const m of raw) messages.push({ role: m.role, blocks: anthropicContentBlocks(m.content) });
    if (Array.isArray(body.tools)) {
      for (const t of body.tools) {
        if (isRecord(t) && typeof t.name === "string") {
          tools.push({ name: t.name, description: t.description, schema: t.input_schema });
        }
      }
    }
    maxTokens = numberOrNull(body.max_tokens);
    stops = stringList(body.stop_sequences);
    jsonMode = false;
    temperature = body.temperature;
    toolChoice = body.tool_choice;
  }
  return { messages, systemText, tools, maxTokens, stops, jsonMode, temperature, toolChoice };
}

function hasImage(neutral) {
  for (const m of neutral.messages) {
    for (const b of m.blocks) if (b.kind === "image") return true;
  }
  return false;
}

// --- rendering into each provider wire -------------------------------------

function openAIImageBlock(block) {
  if (block.imageKind === "data") {
    return { type: "image_url", image_url: { url: `data:${block.mediaType ?? "image/png"};base64,${block.data}` } };
  }
  return { type: "image_url", image_url: { url: block.url } };
}

function renderOpenAIMessages(neutral) {
  const out = [];
  if (neutral.systemText !== "") out.push({ role: "system", content: neutral.systemText });
  for (const m of neutral.messages) {
    const content = [];
    const toolCalls = [];
    const toolResults = [];
    for (const b of m.blocks) {
      if (b.kind === "text") content.push({ type: "text", text: b.text });
      else if (b.kind === "image") content.push(openAIImageBlock(b));
      else if (b.kind === "tool_use") {
        toolCalls.push({ id: b.id, type: "function", function: { name: b.name, arguments: JSON.stringify(b.input ?? {}) } });
      } else if (b.kind === "tool_result") {
        toolResults.push(b);
      }
    }
    if (m.role === "tool" || toolResults.length > 0) {
      for (const r of toolResults) {
        out.push({ role: "tool", tool_call_id: r.toolUseId, content: r.content });
      }
      if (content.length === 0 && toolCalls.length === 0) continue;
    }
    const message = { role: m.role === "tool" ? "user" : m.role };
    if (content.length === 1 && content[0].type === "text") message.content = content[0].text;
    else message.content = content;
    if (toolCalls.length > 0) message.tool_calls = toolCalls;
    out.push(message);
  }
  return out;
}

function mapToolChoiceForOpenAI(choice) {
  if (choice === undefined) return undefined;
  if (typeof choice === "string") return choice;
  if (!isRecord(choice)) return undefined;
  if (choice.type === "auto") return "auto";
  if (choice.type === "any") return "required";
  if (choice.type === "none") return "none";
  if (choice.type === "tool" && typeof choice.name === "string") {
    return { type: "function", function: { name: choice.name } };
  }
  return undefined;
}

function renderOpenAI(neutral, model) {
  const body = { model, stream: false, messages: renderOpenAIMessages(neutral) };
  if (neutral.tools.length > 0) {
    body.tools = neutral.tools.map((t) => {
      const fn = { name: t.name };
      if (t.description !== undefined) fn.description = t.description;
      if (t.schema !== undefined) fn.parameters = t.schema;
      return { type: "function", function: fn };
    });
    const choice = mapToolChoiceForOpenAI(neutral.toolChoice);
    if (choice !== undefined) body.tool_choice = choice;
  }
  if (neutral.maxTokens !== null) body.max_tokens = neutral.maxTokens;
  if (neutral.stops.length > 0) body.stop = neutral.stops;
  if (neutral.jsonMode) body.response_format = { type: "json_object" };
  if (neutral.temperature !== undefined) body.temperature = neutral.temperature;
  return body;
}

function renderAnthropicMessages(neutral) {
  const out = [];
  for (const m of neutral.messages) {
    const content = [];
    for (const b of m.blocks) {
      if (b.kind === "text") content.push({ type: "text", text: b.text });
      else if (b.kind === "image") {
        if (b.imageKind === "data") {
          content.push({ type: "image", source: { type: "base64", media_type: b.mediaType ?? "image/png", data: b.data } });
        } else {
          content.push({ type: "image", source: { type: "url", url: b.url } });
        }
      } else if (b.kind === "tool_use") {
        content.push({ type: "tool_use", id: b.id, name: b.name, input: b.input ?? {} });
      } else if (b.kind === "tool_result") {
        content.push({ type: "tool_result", tool_use_id: b.toolUseId, content: b.content });
      }
    }
    const role = m.role === "tool" ? "user" : m.role;
    if (content.length === 1 && content[0].type === "text") out.push({ role, content: content[0].text });
    else out.push({ role, content });
  }
  return out;
}

function renderAnthropic(neutral, model) {
  const body = { model, stream: false, messages: renderAnthropicMessages(neutral) };
  if (neutral.systemText !== "") body.system = neutral.systemText;
  if (neutral.tools.length > 0) {
    body.tools = neutral.tools.map((t) => {
      const tool = { name: t.name };
      if (t.description !== undefined) tool.description = t.description;
      if (t.schema !== undefined) tool.input_schema = t.schema;
      return tool;
    });
  }
  // max_tokens is omitted when the client sent none: the provider must
  // echo exactly what the client asked, including its absence.
  if (neutral.maxTokens !== null) body.max_tokens = neutral.maxTokens;
  if (neutral.stops.length > 0) body.stop_sequences = neutral.stops;
  if (neutral.temperature !== undefined) body.temperature = neutral.temperature;
  return body;
}

function renderGemini(neutral, model) {
  const body = {};
  if (neutral.systemText !== "") body.systemInstruction = { parts: [{ text: neutral.systemText }] };
  const contents = [];
  for (const m of neutral.messages) {
    const parts = [];
    for (const b of m.blocks) {
      if (b.kind === "text") parts.push({ text: b.text });
      else if (b.kind === "image") {
        if (b.imageKind === "data") {
          parts.push({ inlineData: { mimeType: b.mediaType ?? "image/png", data: b.data } });
        } else {
          parts.push({ fileData: { fileUri: b.url } });
        }
      } else if (b.kind === "tool_use") {
        parts.push({ functionCall: { name: b.name, args: b.input ?? {} } });
      } else if (b.kind === "tool_result") {
        parts.push({ functionResponse: { name: b.toolUseId === "" ? "tool" : b.toolUseId, response: { result: b.content } } });
      }
    }
    const role = m.role === "assistant" ? "model" : m.role === "tool" ? "function" : "user";
    contents.push({ role, parts });
  }
  body.contents = contents;
  if (neutral.tools.length > 0) {
    body.tools = [
      {
        functionDeclarations: neutral.tools.map((t) => {
          const fn = { name: t.name };
          if (t.description !== undefined) fn.description = t.description;
          if (t.schema !== undefined) fn.parameters = t.schema;
          return fn;
        }),
      },
    ];
  }
  const config = {};
  if (neutral.maxTokens !== null) config.maxOutputTokens = neutral.maxTokens;
  if (neutral.stops.length > 0) config.stopSequences = neutral.stops;
  if (neutral.jsonMode) config.responseMimeType = "application/json";
  if (neutral.temperature !== undefined) config.temperature = neutral.temperature;
  if (Object.keys(config).length > 0) body.generationConfig = config;
  return { body, model };
}

// ---------------------------------------------------------------------------
// Provider calls: always JSON in, never streamed. The server re-streams the
// answer itself in the client's wire, which is what makes the usage counts
// and the streaming event order exact in every combination.
// ---------------------------------------------------------------------------

function providerAnswer(wire, body) {
  if (!isRecord(body)) return null;
  if (wire === "openai") {
    const choices = Array.isArray(body.choices) ? body.choices : [];
    const first = choices[0];
    if (!isRecord(first) || !isRecord(first.message)) return null;
    const usage = isRecord(body.usage) ? body.usage : null;
    const input = usage === null ? null : numberOrNull(usage.prompt_tokens);
    const output = usage === null ? null : numberOrNull(usage.completion_tokens);
    if (input === null || output === null) return null;
    const content = first.message.content;
    const text = typeof content === "string" ? content : blocksText(content);
    return { text, usage: { input, output }, stop: typeof first.finish_reason === "string" ? first.finish_reason : "stop" };
  }
  if (wire === "anthropic") {
    const usage = isRecord(body.usage) ? body.usage : null;
    const input = usage === null ? null : numberOrNull(usage.input_tokens);
    const output = usage === null ? null : numberOrNull(usage.output_tokens);
    if (input === null || output === null) return null;
    const parts = [];
    if (Array.isArray(body.content)) {
      for (const block of body.content) {
        if (isRecord(block) && block.type === "text" && typeof block.text === "string") parts.push(block.text);
      }
    }
    return { text: parts.join(""), usage: { input, output }, stop: typeof body.stop_reason === "string" ? body.stop_reason : "end_turn" };
  }
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const first = candidates[0];
  if (!isRecord(first) || !isRecord(first.content) || !Array.isArray(first.content.parts)) return null;
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : null;
  const input = usage === null ? null : numberOrNull(usage.promptTokenCount);
  const output = usage === null ? null : numberOrNull(usage.candidatesTokenCount);
  if (input === null || output === null) return null;
  const parts = [];
  for (const part of first.content.parts) {
    if (isRecord(part) && typeof part.text === "string") parts.push(part.text);
  }
  return { text: parts.join(""), usage: { input, output }, stop: typeof first.finishReason === "string" ? first.finishReason : "STOP" };
}

async function callProvider(envUrl, provider, neutral, model, anthropicVersion) {
  const base = envUrl.replace(/\/$/, "") + provider.path;
  let url;
  let payload;
  if (provider.wire === "openai") {
    url = `${base}/v1/chat/completions`;
    payload = renderOpenAI(neutral, model);
  } else if (provider.wire === "anthropic") {
    url = `${base}/v1/messages`;
    payload = renderAnthropic(neutral, model);
  } else {
    url = `${base}/v1beta/models/${encodeURIComponent(model)}:generateContent`;
    payload = renderGemini(neutral, model).body;
  }
  const headers = { "content-type": "application/json", ...authHeaders(provider) };
  if (provider.wire === "anthropic") headers["anthropic-version"] = anthropicVersion;
  let res;
  try {
    res = await fetch(url, {
      method: "POST",
      headers,
      body: JSON.stringify(payload),
      signal: AbortSignal.timeout(PROVIDER_TIMEOUT_MS),
    });
  } catch {
    return { ok: false, status: 0 };
  }
  let body = null;
  try {
    body = await res.json();
  } catch {
    body = null;
  }
  if (!res.ok) return { ok: false, status: res.status };
  const answer = providerAnswer(provider.wire, body);
  if (answer === null) return { ok: false, status: 502 };
  return { ok: true, ...answer };
}

// ---------------------------------------------------------------------------
// Routing: the best free model that meets the request, failing over on
// refusal. Each candidate is tried up to three times in a row: a refusal
// may be a single scheduled miss, and the next call to the same provider
// can succeed, so retrying before falling to a weaker model keeps the
// quality up at the cost of one extra upstream call. A provider that
// refuses three calls running is treated as spent and only tried again
// as a last resort.
// ---------------------------------------------------------------------------

function meetsRequirements(candidate, requires) {
  if (requires.tools === true && !candidate.tools) return false;
  if (requires.vision === true && !candidate.vision) return false;
  if (requires.json === true && !candidate.json) return false;
  if (requires.min_context !== undefined && (candidate.context_length === null || candidate.context_length < requires.min_context)) return false;
  return true;
}

const STOP_FAMILIES = new Map([
  ["stop", 0],
  ["end_turn", 0],
  ["STOP", 0],
  ["length", 1],
  ["max_tokens", 1],
  ["MAX_TOKENS", 1],
  ["tool_calls", 2],
  ["function_call", 2],
  ["tool_use", 2],
  ["content_filter", 3],
  ["refusal", 3],
  ["SAFETY", 3],
]);

function clientStopReason(api, providerStop) {
  const family = STOP_FAMILIES.get(providerStop) ?? 0;
  if (api === "openai") return ["stop", "length", "tool_calls", "content_filter"][family];
  return ["end_turn", "max_tokens", "tool_use", "refusal"][family];
}

function splitPieces(text) {
  const pieces = [];
  for (let at = 0; at < text.length; at += SSE_PIECE) pieces.push(text.slice(at, at + SSE_PIECE));
  return pieces.length === 0 ? [""] : pieces;
}

function sseFrame(event, data) {
  return `${event === null ? "" : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

function renderResponse(api, stream, model, answer) {
  const stop = clientStopReason(api, answer.stop);
  if (api === "openai") {
    const usage = {
      prompt_tokens: answer.usage.input,
      completion_tokens: answer.usage.output,
      total_tokens: answer.usage.input + answer.usage.output,
    };
    if (!stream) {
      return {
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          id: "chatcmpl-sim",
          object: "chat.completion",
          created: FIXED_CREATED,
          model,
          choices: [{ index: 0, message: { role: "assistant", content: answer.text }, finish_reason: stop, logprobs: null }],
          usage,
        }),
      };
    }
    const chunk = (delta, finish) => ({
      id: "chatcmpl-sim",
      object: "chat.completion.chunk",
      created: FIXED_CREATED,
      model,
      choices: [{ index: 0, delta, finish_reason: finish }],
    });
    const frames = [sseFrame(null, chunk({ role: "assistant", content: "" }, null))];
    for (const piece of splitPieces(answer.text)) frames.push(sseFrame(null, chunk({ content: piece }, null)));
    frames.push(sseFrame(null, chunk({}, stop)));
    frames.push(sseFrame(null, { id: "chatcmpl-sim", object: "chat.completion.chunk", created: FIXED_CREATED, model, choices: [], usage }));
    frames.push("data: [DONE]\n\n");
    return { status: 200, contentType: "text/event-stream; charset=utf-8", body: frames.join("") };
  }
  const usage = { input_tokens: answer.usage.input, output_tokens: answer.usage.output };
  if (!stream) {
    return {
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        id: "msg_sim",
        type: "message",
        role: "assistant",
        model,
        content: [{ type: "text", text: answer.text }],
        stop_reason: stop,
        stop_sequence: null,
        usage,
      }),
    };
  }
  const frames = [
    sseFrame("message_start", {
      type: "message_start",
      message: {
        id: "msg_sim",
        type: "message",
        role: "assistant",
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: answer.usage.input, output_tokens: 0 },
      },
    }),
    sseFrame("content_block_start", { type: "content_block_start", index: 0, content_block: { type: "text", text: "" } }),
  ];
  for (const piece of splitPieces(answer.text)) {
    frames.push(sseFrame("content_block_delta", { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: piece } }));
  }
  frames.push(
    sseFrame("content_block_stop", { type: "content_block_stop", index: 0 }),
    sseFrame("message_delta", { type: "message_delta", delta: { stop_reason: stop, stop_sequence: null }, usage: { output_tokens: answer.usage.output } }),
    sseFrame("message_stop", { type: "message_stop" })
  );
  return { status: 200, contentType: "text/event-stream; charset=utf-8", body: frames.join("") };
}

function errorResponse(api, status, message) {
  const body =
    api === "openai"
      ? { error: { message, type: "api_error", param: null, code: "upstream_error" } }
      : { type: "error", error: { type: "api_error", message } };
  return { status, contentType: "application/json", body: JSON.stringify(body) };
}

// ---------------------------------------------------------------------------
// The server.
// ---------------------------------------------------------------------------

async function readBody(req) {
  const chunks = [];
  let size = 0;
  for await (const chunk of req) {
    size += chunk.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(chunk);
  }
  return Buffer.concat(chunks).toString("utf8");
}

async function main() {
  const listMode = process.argv.includes("--list");
  const promptFile = process.env.MAKEFUN_PROMPT_FILE ?? "";
  const envUrl = process.env.MAKEFUN_ENV_URL ?? "";

  if (listMode) {
    if (promptFile === "") {
      console.log(JSON.stringify({ providers: [] }));
      return;
    }
    const brief = parseBrief(readFileSync(promptFile, "utf8"));
    const { catalog } = await buildCatalog(envUrl === "" ? null : envUrl, brief);
    console.log(JSON.stringify(catalog));
    return;
  }

  if (promptFile === "" || envUrl === "") {
    process.stderr.write("free-model-aggregator: MAKEFUN_PROMPT_FILE and MAKEFUN_ENV_URL are required\n");
    process.exitCode = 1;
    return;
  }
  const brief = parseBrief(readFileSync(promptFile, "utf8"));
  const { catalog, probed } = await buildCatalog(envUrl, brief);
  const routable = buildRoutable(brief, probed);
  const providersById = new Map(brief.providers.map((p) => [p.id, p]));
  const freeIds = [];
  const seenIds = new Set();
  for (const p of catalog.providers) {
    for (const m of p.models) {
      if (m.free === true && !seenIds.has(m.id)) {
        seenIds.add(m.id);
        freeIds.push(m.id);
      }
    }
  }

  // Provider health, learned from refusals only.
  const health = new Map();
  for (const p of brief.providers) health.set(p.id, { consecutiveRefusals: 0, dead: false });

  let requestIndex = 0;
  let queue = Promise.resolve();

  async function serveInference(api, body, req) {
    const index = requestIndex;
    requestIndex += 1;
    const manifestEntry = brief.manifest[index];
    const manifestRequires = manifestEntry !== undefined && isRecord(manifestEntry.requires) ? manifestEntry.requires : {};
    const neutral = toNeutral(api, body);
    const requires = {
      tools: manifestRequires.tools === true || neutral.tools.length > 0,
      vision: manifestRequires.vision === true || hasImage(neutral),
      json: manifestRequires.json === true || (api === "openai" && neutral.jsonMode),
      min_context: manifestRequires.min_context,
    };
    // An OpenAI JSON-mode request must reach a wire that can carry the
    // flag: the Anthropic wire echoes it absent, whatever the model.
    const wireOk = (candidate) => !(api === "openai" && neutral.jsonMode && candidate.wire === "anthropic");
    const candidates = routable
      .filter((c) => meetsRequirements(c, requires) && wireOk(c))
      .sort((a, b) => b.rating - a.rating || (a.provider < b.provider ? -1 : a.provider > b.provider ? 1 : a.model < b.model ? -1 : a.model > b.model ? 1 : 0));
    const anthropicVersion =
      typeof req.headers["anthropic-version"] === "string" ? req.headers["anthropic-version"] : "2023-06-01";
    const stream = body.stream === true;

    const attempted = new Set();
    const tryPass = async (onlyDead) => {
      for (const candidate of candidates) {
        const state = health.get(candidate.provider);
        if (state === undefined) continue;
        if (attempted.has(candidate.provider)) continue;
        if (onlyDead ? !state.dead : state.dead) continue;
        attempted.add(candidate.provider);
        const provider = providersById.get(candidate.provider);
        for (let attempt = 0; attempt < ATTEMPTS_PER_CANDIDATE; attempt += 1) {
          const result = await callProvider(envUrl, provider, neutral, candidate.model, anthropicVersion);
          if (result.ok) {
            state.consecutiveRefusals = 0;
            state.dead = false;
            process.stderr.write(`served request ${index + 1} via ${candidate.provider}/${candidate.model}\n`);
            return renderResponse(api, stream, candidate.model, result);
          }
          state.consecutiveRefusals += 1;
          if (state.consecutiveRefusals >= DEAD_AFTER_REFUSALS) state.dead = true;
        }
      }
      return null;
    };

    const live = await tryPass(false);
    if (live !== null) return live;
    const revived = await tryPass(true);
    if (revived !== null) return revived;
    process.stderr.write(`no provider answered request ${index + 1}\n`);
    return errorResponse(api, 502, "every provider refused the request");
  }

  const server = createServer((req, res) => {
    const send = (answer) => {
      res.writeHead(answer.status, { "content-type": answer.contentType, "content-length": Buffer.byteLength(answer.body) });
      res.end(answer.body);
    };
    const run = async () => {
      const url = new URL(req.url ?? "/", "http://server");
      if (req.method === "GET" && url.pathname === "/v1/models") {
        const hasBearer = typeof req.headers.authorization === "string" && req.headers.authorization !== "";
        const hasKey = typeof req.headers["x-api-key"] === "string" && req.headers["x-api-key"] !== "";
        if (hasKey && !hasBearer) {
          send({
            status: 200,
            contentType: "application/json",
            body: JSON.stringify({
              data: freeIds.map((id) => ({ type: "model", id, display_name: id, created_at: "2026-01-01T00:00:00Z" })),
              has_more: false,
            }),
          });
        } else {
          send({
            status: 200,
            contentType: "application/json",
            body: JSON.stringify({
              object: "list",
              data: freeIds.map((id) => ({ id, object: "model", created: FIXED_CREATED, owned_by: "free-model-aggregator" })),
            }),
          });
        }
        return;
      }
      const api = req.method === "POST" && url.pathname === "/v1/chat/completions" ? "openai" : req.method === "POST" && url.pathname === "/v1/messages" ? "anthropic" : null;
      if (api === null) {
        send(errorResponse("openai", 404, `no route ${req.method} ${url.pathname}`));
        return;
      }
      const raw = await readBody(req);
      if (raw === null) {
        send(errorResponse(api, 413, "the request body is too large"));
        return;
      }
      let body;
      try {
        body = JSON.parse(raw);
      } catch {
        send(errorResponse(api, 400, "the request body is not JSON"));
        return;
      }
      if (!isRecord(body)) {
        send(errorResponse(api, 400, "the request body is not one JSON object"));
        return;
      }
      // One request at a time, in arrival order: a provider serves one
      // call at a time, and the trial's replay must see the same order.
      const job = queue.then(() => serveInference(api, body, req));
      queue = job.then(
        () => undefined,
        () => undefined
      );
      try {
        send(await job);
      } catch (err) {
        process.stderr.write(`request failed: ${err.message}\n`);
        send(errorResponse(api, 500, "the request could not be served"));
      }
    };
    run().catch((err) => {
      process.stderr.write(`handler failed: ${err.message}\n`);
      if (!res.headersSent) send(errorResponse("openai", 500, "the request could not be served"));
    });
  });

  await new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "0.0.0.0", resolve);
  });
  const address = server.address();
  const port = typeof address === "object" && address !== null ? address.port : 0;

  let registered = false;
  for (let attempt = 0; attempt < REGISTER_ATTEMPTS && !registered; attempt += 1) {
    try {
      const res = await fetch(`${envUrl.replace(/\/$/, "")}/register`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ protocol_version: 1, port, catalog }),
        signal: AbortSignal.timeout(PROVIDER_TIMEOUT_MS),
      });
      if (res.ok) {
        registered = true;
      } else {
        process.stderr.write(`register answered ${res.status}; retrying\n`);
      }
    } catch (err) {
      process.stderr.write(`register failed (${err.message}); retrying\n`);
    }
    if (!registered) await sleep(POLL_MS);
  }
  if (!registered) {
    process.stderr.write("free-model-aggregator: could not register with the env\n");
    process.exitCode = 1;
    return;
  }

  for (;;) {
    try {
      const res = await fetch(`${envUrl.replace(/\/$/, "")}/work`, { signal: AbortSignal.timeout(PROVIDER_TIMEOUT_MS) });
      if (res.ok) {
        const work = await res.json();
        if (isRecord(work) && work.done === true) break;
      }
    } catch {
      // The env answers again on the next poll.
    }
    await sleep(POLL_MS);
  }
  server.close();
}

main().catch((err) => {
  process.stderr.write(`free-model-aggregator: ${err.message}\n`);
  process.exitCode = 1;
});
