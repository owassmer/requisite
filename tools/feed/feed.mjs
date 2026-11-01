#!/usr/bin/env node
// tools/feed/feed <run>    Follow a worker or verifier run in Pi's own transcript view, read-only, until it finishes.
//
// <run> is a run name (.runs/<run>.log), a path to its .log or .jsonl, or nothing for the newest run in this checkout.
// scripts/work and scripts/verify tee each run's Pi JSON event stream to .runs/<run>.jsonl (Pi's docs/json.md); this
// replays it from the start and then follows it, through the same components Pi's interactive mode draws a session
// with (messages, thinking, tool cards, diffs) in Pi's fullscreen screen, with Pi's settings, theme and keybindings.
// Nothing can be typed to the run. q or ctrl+c closes the view.
//
// It sets its terminal title to "⧉ <run>". The coordinator's extension (.pi/extensions/factory-runs.ts) opens it in a
// Ghostty split with FEED_CLOSE=1 (the view closes its own Ghostty pane, found by that title, when it exits) and, when
// it follows a run, FEED_AUTOCLOSE=1 (it exits a few seconds after the run finishes; any key keeps it open). When a run
// has finished is tools/feed/runs.mjs's rule. Pi's package is found from FEED_PI_ROOT, else from `pi` on the PATH.
import { execFileSync, spawnSync } from "node:child_process";
import { closeSync, existsSync, openSync, readdirSync, readFileSync, readSync, realpathSync, statSync, watch } from "node:fs";
import { basename, dirname, join, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { runState } from "./runs.mjs";

const LINGER_S = Number(process.env.FEED_LINGER ?? 8);

// ---- where things are -----------------------------------------------------------------------------------------------
function piRoot() {
  const fromEnv = process.env.FEED_PI_ROOT;
  if (fromEnv && existsSync(join(fromEnv, "package.json"))) return fromEnv;
  const which = spawnSync("/bin/sh", ["-lc", "command -v pi"], { encoding: "utf8" }).stdout.trim();
  if (!which) throw new Error("cannot find Pi: set FEED_PI_ROOT to the @earendil-works/pi-coding-agent package folder");
  // Pi's managed launcher (~/.pi/agent/bin/pi) runs the release named in install/current-version.
  const agent = dirname(dirname(realpathSync(which)));
  const current = join(agent, "install", "current-version");
  if (existsSync(current)) {
    const release = join(agent, "install", "releases", readFileSync(current, "utf8").trim(), "node_modules", "@earendil-works", "pi-coding-agent");
    if (existsSync(join(release, "package.json"))) return release;
  }
  let dir = dirname(realpathSync(which));
  while (dir !== "/") {
    const pkg = join(dir, "package.json");
    if (existsSync(pkg) && JSON.parse(readFileSync(pkg, "utf8")).name === "@earendil-works/pi-coding-agent") return dir;
    dir = dirname(dir);
  }
  throw new Error(`cannot find Pi's package from ${which}`);
}

function findRun(arg) {
  if (arg && existsSync(arg)) {
    const p = resolve(arg);
    return { log: p.replace(/\.jsonl$/, ".log"), jsonl: p.replace(/\.log$/, ".jsonl") };
  }
  const root = execFileSync("git", ["rev-parse", "--show-toplevel"], { encoding: "utf8" }).trim();
  const runs = join(root, ".runs");
  let name = arg;
  if (!name) {
    const logs = existsSync(runs) ? readdirSync(runs).filter((f) => f.endsWith(".log")) : [];
    logs.sort((a, b) => statSync(join(runs, b)).mtimeMs - statSync(join(runs, a)).mtimeMs);
    name = logs[0]?.slice(0, -4);
    if (!name) throw new Error("no runs yet; scripts/work and scripts/verify write them to .runs/");
  }
  const log = join(runs, `${name}.log`);
  if (!existsSync(log)) throw new Error(`no run ${name} in ${runs}`);
  return { log, jsonl: join(runs, `${name}.jsonl`) };
}

const { log: LOG, jsonl: JSONL } = findRun(process.argv[2]);
const RUN = basename(LOG, ".log");
const PI = piRoot();
const imp = (rel) => import(pathToFileURL(join(PI, rel)).href);
/** A package Pi depends on: nested under Pi's own node_modules, or beside it where the install is flat. */
const dep = (name, rel) => {
  const nested = join(PI, "node_modules/@earendil-works", name, rel);
  return pathToFileURL(existsSync(nested) ? nested : join(dirname(PI), name, rel)).href;
};

const tuiLib = await import(dep("pi-tui", "dist/index.js")).catch(
  () => import("@earendil-works/pi-tui"),
);
const { Container, Spacer, Text, TruncatedText, ScrollView, VStack, ProcessTerminal, TuiAltScreen, setKeybindings, matchesKey, truncateToWidth, visibleWidth } = tuiLib;
const [components, statusMod, themeMod, themeCtl, renderers, settingsMod, keybindingsMod, configMod, aiJson] = await Promise.all([
  imp("dist/modes/interactive/components/index.js"),
  imp("dist/modes/interactive/components/status-indicator.js"),
  imp("dist/modes/interactive/theme/theme.js"),
  imp("dist/modes/interactive/theme/theme-controller.js"),
  imp("dist/core/tools/renderers/index.js"),
  imp("dist/core/settings-manager.js"),
  imp("dist/core/keybindings.js"),
  imp("dist/config.js"),
  import(dep("pi-ai", "dist/utils/json-parse.js")).catch(() => ({})),
]);
const { AssistantMessageComponent, ToolExecutionComponent, UserMessageComponent, CompactionSummaryMessageComponent } = components;
const { theme, getMarkdownTheme } = themeMod;
const { withBuiltInRenderers } = renderers;
const parseStreamingJson = aiJson.parseStreamingJson ?? ((s) => { try { return JSON.parse(s); } catch { return {}; } });

// ---- the run's plain log: who it is, how it ended -------------------------------------------------------------------
const meta = { role: "run", target: "", model: "", finished: null, notes: [], started: statSync(LOG).birthtimeMs || statSync(LOG).mtimeMs };
function readLog() {
  const text = readFileSync(LOG, "utf8");
  const head = /^=== (worker|verifier) on (\S+) \(([^)]*)\)/m.exec(text);
  if (head) [meta.role, meta.target, meta.model] = [head[1], head[2], head[3]];
  meta.notes = text.split("\n").filter((l) => l.startsWith("=== ") && !/^=== (worker|verifier) on |^=== dispatched by |^=== steerable|^=== waiting for a worker slot|^=== got a worker slot|^=== finished/.test(l)).map((l) => l.slice(4));
  // When the run is waiting, running or finished: the same rule the coordinator's extension uses (runs.mjs).
  const run = runState(text, statSync(LOG).mtimeMs);
  meta.finished = run.state === "finished" ? run.end : null;
  meta.waiting = run.state === "waiting";
  meta.productUp = /"url": "http/.test(text); // drive's reply once the role's product is running
}

// ---- the screen, as Pi builds it ------------------------------------------------------------------------------------
const settings = settingsMod.SettingsManager.create(process.cwd(), configMod.getAgentDir());
const keybindings = keybindingsMod.KeybindingsManager.create();
setKeybindings(keybindings);
const ui = new TuiAltScreen(new ProcessTerminal(), false, configMod.getAgentDir(), {
  scrollToEndIndicator: () => theme.bg("selectedBg", theme.fg("text", " ↓ Jump to latest ")),
  wheelScrollLines: settings.getFullscreenWheelScrollLines?.() ?? "auto",
  copyOnSelect: settings.getFullscreenCopyOnSelect?.() ?? true,
  copySelection: async (text) => {
    try { execFileSync("pbcopy", { input: text }); return true; } catch (e) { return String(e); }
  },
  searchMatchStyle: (t) => theme.underline(theme.bg("searchMatchBg", theme.fg("searchMatchText", t))),
  searchCurrentMatchStyle: (t) => theme.bold(theme.inverse(theme.bg("searchMatchBg", theme.fg("searchMatchText", t)))),
  searchNavigationButtonStyle: (t, hovered) => (hovered ? theme.underline(t) : t),
});
const themes = new themeCtl.InteractiveThemeController(ui, {
  getSettingsManager: () => settings,
  showError: () => {},
  onChanged: () => ui.invalidate(),
});

const hideThinking = settings.getHideThinkingBlock();
const outputPad = settings.getOutputPad();
const md = () => ({ ...getMarkdownTheme(), codeBlockIndent: settings.getCodeBlockIndent() });
const toolOptions = () => ({ showImages: settings.getShowImages(), imageWidthCells: settings.getImageWidthCells() });

const chat = new Container();
const status = new Container();
let cwd = process.cwd();
let thinkingLevel = /:(\w+)$/.exec(meta.model)?.[1] ?? "";

// The editor's place: one border line in the thinking level's colour, carrying the working indicator, as Pi's does.
let indicator; // a pi-tui Loader while the run works
let indicatorKind = "";
let closingIn = null; // seconds left before the pane closes itself, once the run finished
let kept = false;
const dock = {
  invalidate() {},
  render(width) {
    const border = (s) => (theme.getThinkingBorderColor?.(thinkingLevel || "off") ?? ((x) => theme.fg("border", x)))(s);
    let left = "";
    if (meta.finished) {
      const ok = /^exit 0\b/.test(meta.finished);
      left = theme.fg(ok ? "success" : "error", `${ok ? "✓" : "✗"} finished ${meta.finished}`);
    } else if (indicator) left = indicator.renderInBorder(Math.max(10, width - 30));
    let right = theme.fg("dim", "read-only");
    if (closingIn !== null) right = theme.fg("muted", `closing in ${closingIn}s · any key keeps it`);
    else if (meta.finished) right = theme.fg("dim", "q to close");
    const lw = visibleWidth(left), rw = visibleWidth(right);
    const fill = Math.max(1, width - lw - rw - 6);
    const line = `${border("──")}${left ? ` ${left} ` : border("─")}${border("─".repeat(fill))} ${right} ${border("─")}`;
    return [truncateToWidth(line, width, "")];
  },
};

// Pi's footer, from what the stream carries: the run instead of the folder, its usage, its model.
const usage = { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, cost: 0 };
let context = { tokens: 0, window: 0 };
const fmt = (n) => (n < 1000 ? `${n}` : n < 10_000 ? `${(n / 1000).toFixed(1)}k` : n < 1_000_000 ? `${Math.round(n / 1000)}k` : `${(n / 1_000_000).toFixed(1)}M`);
const footer = {
  invalidate() {},
  render(width) {
    const elapsed = Math.max(0, Math.round(((meta.finishedAt ?? Date.now()) - meta.started) / 1000));
    const clock = `${Math.floor(elapsed / 60)}m${String(elapsed % 60).padStart(2, "0")}s`;
    const first = `${meta.role} on ${meta.target || RUN} • ${clock}${meta.notes.length ? ` • ${meta.notes.at(-1)}` : ""}`;
    const parts = [];
    if (usage.input) parts.push(`↑${fmt(usage.input)}`);
    if (usage.output) parts.push(`↓${fmt(usage.output)}`);
    if (usage.cacheRead) parts.push(`R${fmt(usage.cacheRead)}`);
    if (usage.cacheWrite) parts.push(`W${fmt(usage.cacheWrite)}`);
    if (usage.cost) parts.push(`$${usage.cost.toFixed(3)}`);
    if (context.tokens) parts.push(context.window ? `${((context.tokens / context.window) * 100).toFixed(1)}%/${fmt(context.window)}` : `${fmt(context.tokens)} ctx`);
    const left = parts.join(" ");
    const model = meta.model.replace(/:(\w+)$/, (_, l) => ` • ${l}`);
    const pad = width - visibleWidth(left) - visibleWidth(model);
    const second = pad >= 2 ? left + " ".repeat(pad) + model : truncateToWidth(left, width, "");
    return [truncateToWidth(theme.fg("dim", first), width, theme.fg("dim", "...")), theme.fg("dim", second)];
  },
};

const transcript = new ScrollView(chat, {
  follow: "end",
  primary: true,
  overscroll: "chain",
  scrollbar: settings.getFullscreenScrollbar?.() ?? "auto",
  scrollbarTrackStyle: (t) => theme.fg("scrollbarTrack", t),
  scrollbarThumbStyle: (t) => theme.fg("scrollbarThumb", t),
});
const root = new VStack([
  { component: transcript, basis: 0, grow: 1, shrink: 1, minSize: 1 },
  {
    component: new VStack([
      { component: status, shrink: 1, minSize: 0 },
      { component: dock, shrink: 0, minSize: 1 },
      { component: footer, shrink: 1, minSize: 0 },
    ]),
    basis: "auto",
    grow: 0,
    shrink: 1,
    minSize: 1,
  },
]);
for (const c of [chat, status, dock, footer]) ui.addChild(c);
ui.setLayoutRoot(root);

// ---- events, handled as Pi's interactive mode handles them (modes/interactive/interactive-mode.ts handleEvent) -------
let started = false; // Pi's stream has begun
let streaming; // { component, message, json: Map<contentIndex, partial tool-call JSON> }
const pendingTools = new Map();
let catchingUp = true; // replaying what was written before the view opened: render once at the end

function setIndicator(kind, message, color = "accent") {
  if (indicatorKind === kind && indicator) { indicator.setMessage?.(message); return; }
  indicator?.stop?.();
  indicator = new statusMod.StatusIndicator(kind, ui, (s) => theme.fg(color, s), (t) => theme.fg("muted", t), message);
  indicatorKind = kind;
}
function clearIndicator(kind) {
  if (kind && indicatorKind !== kind) return;
  indicator?.stop?.();
  indicator = undefined;
  indicatorKind = "";
}

function toolComponent(name, id, args) {
  let c = pendingTools.get(id);
  if (!c) {
    c = new ToolExecutionComponent(name, id, args, toolOptions(), withBuiltInRenderers(name, undefined), ui, cwd);
    c.setExpanded(false);
    chat.addChild(c);
    pendingTools.set(id, c);
  }
  return c;
}
function syncToolCalls(message) {
  for (const content of message.content) {
    if (content.type !== "toolCall") continue;
    if (pendingTools.has(content.id)) pendingTools.get(content.id).updateArgs(content.arguments);
    else toolComponent(content.name, content.id, content.arguments);
  }
}
function addUsage(u) {
  if (!u) return;
  usage.input += u.input ?? 0;
  usage.output += u.output ?? 0;
  usage.cacheRead += u.cacheRead ?? 0;
  usage.cacheWrite += u.cacheWrite ?? 0;
  usage.cost += u.cost?.total ?? 0;
  context.tokens = (u.input ?? 0) + (u.cacheRead ?? 0) + (u.cacheWrite ?? 0) + (u.output ?? 0);
}

function applyDelta(m, e) {
  const i = e.contentIndex;
  switch (e.type) {
    case "text_start": m.content[i] = { type: "text", text: "" }; break;
    case "text_delta": if (m.content[i]) m.content[i].text += e.delta; break;
    case "text_end": m.content[i] = { ...(m.content[i] ?? { type: "text" }), text: e.content }; break;
    case "thinking_start": m.content[i] = { type: "thinking", thinking: "" }; break;
    case "thinking_delta": if (m.content[i]) m.content[i].thinking += e.delta; break;
    case "thinking_end": m.content[i] = { ...(m.content[i] ?? { type: "thinking" }), thinking: e.content }; break;
    case "toolcall_start": m.content[i] = { type: "toolCall", id: e.id, name: e.toolName, arguments: {} }; streaming.json.set(i, ""); break;
    case "toolcall_delta": {
      const json = (streaming.json.get(i) ?? "") + e.delta;
      streaming.json.set(i, json);
      if (m.content[i]) m.content[i].arguments = parseStreamingJson(json) ?? {};
      break;
    }
    case "toolcall_end": m.content[i] = e.toolCall; break;
  }
}

function handle(event) {
  switch (event.type) {
    case "session":
      if (event.cwd) cwd = event.cwd;
      break;
    case "agent_start":
      pendingTools.clear();
      break;
    case "turn_start":
      setIndicator("working", "Working");
      break;
    case "thinking_level_changed":
      thinkingLevel = event.level ?? thinkingLevel;
      break;
    case "message_start": {
      const m = event.message;
      if (m.role === "user") {
        const text = typeof m.content === "string" ? m.content : (m.content ?? []).filter((c) => c.type === "text").map((c) => c.text).join("\n");
        if (text) {
          if (chat.children.length) chat.addChild(new Spacer(1));
          chat.addChild(new UserMessageComponent(text, md(), outputPad));
        }
      } else if (m.role === "assistant") {
        const component = new AssistantMessageComponent(undefined, hideThinking, md(), undefined, outputPad);
        streaming = { component, message: { ...m, content: [...(m.content ?? [])] }, json: new Map() };
        chat.addChild(component);
        if (!catchingUp) component.updateContent(streaming.message, true);
      }
      break;
    }
    case "message_update":
      if (!streaming) break;
      applyDelta(streaming.message, event.assistantMessageEvent ?? {});
      if (event.usage) streaming.message.usage = event.usage;
      if (!catchingUp) {
        streaming.component.updateContent(streaming.message, true);
        syncToolCalls(streaming.message);
      }
      break;
    case "message_end": {
      const m = event.message;
      if (m.role !== "assistant" || !streaming) break;
      streaming.component.updateContent(m, false);
      syncToolCalls(m);
      addUsage(m.usage);
      if (m.stopReason === "aborted" || m.stopReason === "error") {
        for (const c of pendingTools.values()) c.updateResult({ content: [{ type: "text", text: m.errorMessage || "Error" }], isError: true });
        pendingTools.clear();
      } else for (const c of pendingTools.values()) c.setArgsComplete();
      streaming = undefined;
      break;
    }
    case "tool_execution_start":
      if (event.parentToolCallId) break;
      toolComponent(event.toolName, event.toolCallId, event.args).markExecutionStarted();
      setIndicator("working", `Running ${event.toolName}`);
      break;
    case "tool_execution_update":
      pendingTools.get(event.toolCallId)?.updateResult({ ...event.partialResult, isError: false }, true);
      break;
    case "tool_execution_end": {
      const c = pendingTools.get(event.toolCallId);
      if (c) {
        c.updateResult({ ...event.result, isError: event.isError });
        pendingTools.delete(event.toolCallId);
      }
      setIndicator("working", "Working");
      break;
    }
    case "agent_end":
      if (streaming) { chat.removeChild(streaming.component); streaming = undefined; }
      pendingTools.clear();
      clearIndicator();
      break;
    case "compaction_start":
      setIndicator("compaction", event.reason === "overflow" ? "Context overflow detected, compacting..." : "Compacting context...");
      break;
    case "compaction_end":
      clearIndicator("compaction");
      if (event.result) {
        chat.addChild(new Spacer(1));
        const c = new CompactionSummaryMessageComponent(
          { role: "compactionSummary", summary: event.result.summary, tokensBefore: event.result.tokensBefore, timestamp: new Date().toISOString() },
          md(),
        );
        chat.addChild(c);
      } else if (event.errorMessage) chat.addChild(new Text(theme.fg("error", event.errorMessage), 1, 0));
      break;
    case "auto_retry_start":
      setIndicator("retry", `Retrying (${event.attempt}/${event.maxAttempts}): ${String(event.errorMessage ?? "").slice(0, 120)}`, "warning");
      break;
    case "auto_retry_end":
      clearIndicator("retry");
      if (!event.success) chat.addChild(new Text(theme.fg("error", `Retry failed after ${event.attempt} attempts: ${event.finalError || "Unknown error"}`), 1, 0));
      break;
  }
}

// ---- following the files ---------------------------------------------------------------------------------------------
let pos = 0;
let pending = Buffer.alloc(0);
function readJsonl() {
  if (!existsSync(JSONL)) return false;
  const size = statSync(JSONL).size;
  if (size <= pos) return false;
  const fd = openSync(JSONL, "r");
  const buf = Buffer.alloc(size - pos);
  readSync(fd, buf, 0, buf.length, pos);
  closeSync(fd);
  pos = size;
  pending = Buffer.concat([pending, buf]);
  let nl;
  // Split on LF bytes only (Pi's docs/json.md: Unicode separators inside strings are not record ends).
  while ((nl = pending.indexOf(0x0a)) >= 0) {
    const line = pending.subarray(0, nl).toString("utf8").replace(/\r$/, "");
    pending = pending.subarray(nl + 1);
    if (!line) continue;
    let event;
    try { event = JSON.parse(line); } catch { continue; }
    started = true;
    try { handle(event); } catch (e) { chat.addChild(new Text(theme.fg("error", `feed: ${e?.message ?? e}`), 1, 0)); }
  }
  return true;
}

function finishCard() {
  chat.addChild(new Spacer(1));
  const ok = /^exit 0\b/.test(meta.finished);
  const lines = [theme.bold(theme.fg(ok ? "success" : "error", `${ok ? "✓" : "✗"} ${meta.role} finished ${meta.finished}`))];
  for (const n of meta.notes) lines.push(theme.fg("muted", n));
  chat.addChild(new Text(lines.join("\n"), 1, 0));
}

let finishedSeen = false;
let closeTimer;
function tick() {
  // Its log removed (the run cleaned up): nothing more to follow.
  if (!existsSync(LOG)) return quit();
  readLog();
  const changed = readJsonl();
  // Before Pi starts, the script builds the role's workspace and starts its product; drive's reply then joins the log.
  if (!started && !meta.finished) {
    setIndicator("preparing", meta.waiting ? "Waiting for a worker slot" : meta.productUp ? `Starting the ${meta.role}` : "Preparing the workspace and starting the product");
  } else if (indicatorKind === "preparing") clearIndicator("preparing");
  if (catchingUp) {
    catchingUp = false;
    if (streaming) {
      streaming.component.updateContent(streaming.message, true);
      syncToolCalls(streaming.message);
    }
  }
  if (meta.finished && !finishedSeen) {
    finishedSeen = true;
    meta.finishedAt = statSync(LOG).mtimeMs;
    readJsonl();
    clearIndicator();
    finishCard();
    // Opened by the coordinator's extension to follow a run (FEED_AUTOCLOSE=1): the pane goes away by itself shortly
    // after the run finishes, also when it had already finished by the time the feed opened.
    if (autoClose) {
      closingIn = LINGER_S;
      closeTimer = setInterval(() => {
        if (kept) return;
        closingIn -= 1;
        if (closingIn <= 0) quit();
        ui.requestRender();
      }, 1000);
    }
  }
  if (changed || !meta.finished) ui.requestRender();
}

// ---- input, title, start ---------------------------------------------------------------------------------------------
function closeOwnPane() {
  if (process.env.FEED_CLOSE !== "1") return;
  // The pane is found by the title this view set; the extension's split gave no other handle.
  const title = `⧉ ${RUN}`;
  spawnSync("osascript", ["-e", `tell application "Ghostty" to close (every terminal whose name is "${title}")`], { timeout: 3000 });
}
let quitting = false;
function quit() {
  if (quitting) return;
  quitting = true;
  try { ui.stop(); } catch {}
  process.exit(0);
}
// Whatever ends this view (q, the countdown, an error), its pane goes with it; only Ghostty closing the pane (SIGHUP)
// needs nothing more. A pane left behind would keep the title and look like a live feed.
let hungUp = false;
process.on("exit", () => { if (!hungUp) closeOwnPane(); });
process.on("uncaughtException", (e) => {
  try { ui.stop(); } catch {}
  console.error(`feed: ${e?.stack ?? e}`);
  setTimeout(() => process.exit(1), 5000);
});

ui.addInputListener((data) => {
  // q would be a search query's letter while Pi's transcript search (an overlay) is open.
  if (matchesKey(data, "ctrl+c") || (data === "q" && !ui.hasOverlay())) { quit(); return { consume: true }; }
  if (closingIn !== null && !kept) {
    kept = true;
    closingIn = null;
    clearInterval(closeTimer);
    ui.requestRender();
  }
  return undefined;
});
process.on("SIGHUP", () => { hungUp = true; quit(); });
process.on("SIGTERM", () => quit());

readLog();
const autoClose = process.env.FEED_AUTOCLOSE !== undefined ? process.env.FEED_AUTOCLOSE === "1" : process.env.FEED_CLOSE === "1" && !meta.finished;
ui.start();
ui.terminal.setTitle(`⧉ ${RUN}`);
themes.applyFromSettings();
await themes.waitForTerminalColors();
tick();
try { watch(dirname(LOG), () => tick()); } catch {}
setInterval(tick, 500);
