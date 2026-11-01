/**
 * The coordinator's view of the runs it dispatches: told when each finishes, and shown each live beside the session.
 *
 * scripts/work and scripts/verify write each run to <checkout>/.runs/<run>.log, record the session that started them
 * ("=== dispatched by session <id>", from Pi's PI_SESSION_ID), tee Pi's JSON event stream to .runs/<run>.jsonl and end
 * the log with "=== finished exit <n>" (tools/feed/runs.mjs says when a run is waiting, running or finished). This watches the .runs/ folder of every checkout of the repository.
 *
 * Told: when a run this session dispatched finishes, it sends this session a message (which starts a turn, or follows
 * the current one) with the run, its exit status, its checkout's newest commit and, for a verifier, its WORKS and BROKEN
 * counts. Other sessions' runs, and runs started outside Pi, are not reported.
 *
 * Shown: when this session runs inside Ghostty, each run it dispatches opens as a read-only feed (tools/feed/feed, Pi's
 * own transcript view of the run) in a split to the right of this session's pane; several runs stack in that column.
 * The feed closes its own pane a few seconds after its run finishes. Ctrl+Shift+R hides every feed of this session, or
 * shows again those still running (or, when none is, the newest run this session dispatched). Splitting never takes the
 * keyboard: focus goes back to the pane that had it, without bringing Ghostty to the front.
 *
 * Ghostty's AppleScript names a pane only by id, title and folder, and two Pi sessions in one checkout share the last
 * two. So this finds its own pane when it needs it: it sets the terminal title to a one-off tag, asks Ghostty which pane
 * carries it and puts the title back (about 0.1 s), then keeps that id while the pane exists. Nothing depends on where
 * focus was when the session started, so reload, resume and a pane moved to another tab or window all work. Feed panes
 * are found by their title, "⧉ <run>", which the feed sets, so no state survives in memory between reloads.
 *
 * Only the interactive session watches; workers and verifiers run Pi in JSON mode and skip it.
 */
import { execFile, execFileSync } from "node:child_process";
import { existsSync, readdirSync, readFileSync, realpathSync, statSync, watch, type FSWatcher } from "node:fs";
import { dirname, join } from "node:path";
import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
// When a run is waiting, running or finished: one definition, shared with the feed.
import { runState } from "../../tools/feed/runs.mjs";

const FEED_KEY = "ctrl+shift+r";
const FEED_TITLE = "⧉ ";

const git = (cwd: string, ...a: string[]) => {
  try {
    return execFileSync("git", a, { cwd, encoding: "utf8", maxBuffer: 64 * 1024 * 1024 }).trim();
  } catch {
    return "";
  }
};
type RunState = { state: "waiting" | "running" | "finished"; end?: string };

/** Where a run's log says the session limit ended it: the reset as the limit gave it, and its instant from when the run ended. */
export function limitReset(text: string, ended: Date): { said: string; at: Date | null } | null {
  const m = /session limit \u00b7 resets (\d{1,2})(?::(\d{2}))?\s*(am|pm)\s*\(([^)]+)\)/i.exec(text);
  if (!m) return null;
  return { said: `${m[1]}${m[2] ? `:${m[2]}` : ""}${m[3]} ${m[4]}`, at: resetInstant(Number(m[1]), Number(m[2] ?? 0), m[3]!.toLowerCase() as "am" | "pm", m[4]!, ended) };
}

/**
 * The next instant a session limit's reset falls at: the clock time it gives ("10:40pm") in the time zone it names
 * ("America/New_York"), today there or, where that has passed, tomorrow. Null where the zone is not one Intl knows.
 */
export function resetInstant(hour12: number, minute: number, half: "am" | "pm", zone: string, now = new Date()): Date | null {
  if (hour12 < 1 || hour12 > 12 || minute > 59) return null;
  const hour = (hour12 % 12) + (half === "pm" ? 12 : 0);
  let f: Intl.DateTimeFormat;
  try { f = new Intl.DateTimeFormat("en-US", { timeZone: zone, hourCycle: "h23", year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit" }); } catch { return null; }
  const wall = (d: Date) => { const p = Object.fromEntries(f.formatToParts(d).map((x) => [x.type, x.value])); return Date.UTC(+p.year!, +p.month! - 1, +p.day!, +p.hour!, +p.minute!); };
  // The zone's offset now, as wall time less the instant (to the minute), carries the wall time of the reset back to an instant.
  const offset = wall(now) - Math.floor(now.getTime() / 60_000) * 60_000;
  const p = Object.fromEntries(f.formatToParts(now).map((x) => [x.type, x.value]));
  let at = Date.UTC(+p.year!, +p.month! - 1, +p.day!, hour, minute) - offset;
  if (at <= now.getTime()) at += 24 * 60 * 60_000;
  return new Date(at);
}
// Logs can vanish between listing and reading (a run cleaned up in any checkout); a log that cannot be read is skipped.
const readLog = (file: string): string | null => {
  try {
    return readFileSync(file, "utf8");
  } catch {
    return null;
  }
};
const stateOf = (file: string, text = readLog(file)): RunState | null => {
  if (text === null) return null;
  try {
    return runState(text, statSync(file).mtimeMs) as RunState;
  } catch {
    return null;
  }
};
const runName = (file: string) => file.split("/").pop()!.replace(/\.log$/, "");

/** AppleScript string literal. */
const as = (s: string) => `"${s.replace(/\\/g, "\\\\").replace(/"/g, '\\"')}"`;
/** Shell word, single-quoted. */
const sh = (s: string) => `'${s.replace(/'/g, `'\\''`)}'`;

function osascript(script: string): Promise<string> {
  return new Promise((resolve, reject) =>
    execFile("osascript", ["-e", script], { timeout: 5000 }, (err, out, errOut) =>
      err ? reject(new Error(String(errOut || err.message).trim())) : resolve(String(out).trim()),
    ),
  );
}

/** The folder of the Pi package running this session, so the feed renders with the same Pi. */
function piRoot(): string | undefined {
  try {
    let dir = dirname(realpathSync(process.argv[1] ?? ""));
    while (dir !== "/") {
      const pkg = join(dir, "package.json");
      if (existsSync(pkg) && JSON.parse(readFileSync(pkg, "utf8")).name === "@earendil-works/pi-coding-agent") return dir;
      dir = dirname(dir);
    }
  } catch {}
  return undefined;
}

/**
 * Every outcome checkout carries main's factory files (scripts/sync), so a merged factory pull request reaches each
 * checkout without the coordinator remembering to put it there (Owen, October 8). Run at the start of a session, after
 * each run finishes, and when a turn settles (at most once a minute); what it did shows as a notice when something
 * changed or could not be brought into step.
 */
let lastSync = 0;
export function syncCheckouts(cwd: string, o: { force?: boolean } = {}): string {
  if (!o.force && Date.now() - lastSync < 60_000) return "";
  lastSync = Date.now();
  try {
    return execFileSync(join(cwd, "scripts/sync"), [], { cwd, encoding: "utf8", timeout: 120_000 }).trim();
  } catch (e: any) {
    return String(e?.stdout ?? e?.message ?? e).trim();
  }
}
/**
 * What the sync did that the coordinator needs to hear, said once. Bringing a checkout to main is an event, told each time
 * it happens. A checkout left as it is (uncommitted changes, not pushed, could not be brought) is a standing state that
 * every sync reports again; told each time, it made a notice after nearly every turn (October 13), and a notice between
 * the coordinator's reply and Owen's next message makes the model client rebuild the whole conversation without its
 * cache. So a standing state is told when it first appears or changes, and not again while it stays the same.
 */
const toldState = new Map<string, string>();
const syncNews = (out: string) => out.split("\n").filter((l) => {
  const checkout = l.split(" [")[0] ?? l;
  if (/brought to main/.test(l)) { toldState.delete(checkout); return true; }
  if (!/could not|uncommitted|not pushed/.test(l)) { toldState.delete(checkout); return false; }
  // The commit it is behind changes as main moves; the state is the rest of the line.
  const state = l.replace(/\b[0-9a-f]{7,40}\b/g, "");
  if (toldState.get(checkout) === state) return false;
  toldState.set(checkout, state);
  return true;
}).join("\n");

export default function (pi: ExtensionAPI) {
  const watchers: FSWatcher[] = [];
  let timer: NodeJS.Timeout | null = null;
  let redeliver: () => void = () => {};
  let live: ExtensionContext | null = null; // the current session's context, replaced on every session_start
  // This Pi's Ghostty pane belongs to the process, not to this runtime: kept across reloads, so a reload finds it
  // without probing (a fresh runtime's title writes reach the terminal only when Pi next redraws).
  const PANE = Symbol.for("factory-runs.ghostty-pane");
  const process_ = globalThis as unknown as Record<symbol, string | null | undefined>;
  let selfPane: string | null = process_[PANE] ?? null;
  // What the shortcut needs from the session now active.
  let session: { ctx: ExtensionContext; logs: () => string[]; mine: (text: string) => boolean } | null = null;
  const inGhostty = process.env.TERM_PROGRAM === "ghostty" && !process.env.TMUX && !process.env.SSH_TTY;

  const stop = () => {
    gen++; // Ghostty work queued for the session that is ending is dropped
    for (const w of watchers) w.close();
    watchers.length = 0;
    if (timer) clearInterval(timer);
    timer = null;
    live = null;
    session = null;
  };

  // ---- Ghostty panes -------------------------------------------------------------------------------------------------

  // One Ghostty operation at a time. Finding this pane retitles it, and a split reads which feeds exist before adding
  // one, so two at once (several runs going when a session starts) lose each other's titles and split side by side.
  // After a reload or a session switch the old context is stale and any use of it throws, which would end Pi when it
  // happens in background work; so work queued before then is dropped (`gen`), and UI calls go through `ui`.
  // The queue belongs to the process too: after a reload the new runtime's work waits for the old one's last step.
  const QUEUE = Symbol.for("factory-runs.ghostty-queue");
  const shared = globalThis as unknown as Record<symbol, Promise<unknown> | undefined>;
  let gen = 0;
  const serial = <T>(fn: () => Promise<T>, skipped: T): Promise<T> => {
    const mine = gen;
    const run = () => (mine === gen ? fn() : Promise.resolve(skipped));
    const queue = shared[QUEUE] ?? Promise.resolve();
    const next = queue.then(run, run);
    shared[QUEUE] = next.catch(() => undefined);
    return next;
  };
  const ui = (ctx: ExtensionContext) => {
    try {
      return ctx.hasUI ? ctx.ui : null;
    } catch {
      return null;
    }
  };
  const notify = (ctx: ExtensionContext, text: string, level: "info" | "warning") => {
    try {
      ui(ctx)?.notify(text, level);
    } catch {}
  };
  const setTitle = (ctx: ExtensionContext, title: string) => {
    try {
      ui(ctx)?.setTitle(title);
      return true;
    } catch {
      return false;
    }
  };

  /** This session's Ghostty pane id: the cached one while it exists, else found by a one-off title. */
  async function findSelf(ctx: ExtensionContext): Promise<string | null> {
    // Only a definite "no" drops the known pane; a slow or failed check (Ghostty busy) keeps it.
    if (selfPane && (await osascript(`tell application "Ghostty" to exists (first terminal whose id is ${as(selfPane)})`).catch(() => "unknown")) !== "false")
      return selfPane;
    selfPane = null;
    const folder = process.cwd().split("/").pop() ?? ""; // not ctx.cwd: the context may go stale during the probe
    const mine = gen; // a reload ends this session's runtime: its search then stops rather than fight the new one
    // Every pane's title before the probe, one "id<TAB>title" per line, to put this pane's back afterwards.
    const titles = new Map(
      (
        await osascript(`tell application "Ghostty"
          set out to {}
          repeat with x in terminals
            set end of out to (id of x) & (character id 9) & (name of x)
          end repeat
          set AppleScript's text item delimiters to linefeed
          return out as text
        end tell`).catch(() => "")
      )
        .split("\n")
        .map((l) => [l.slice(0, l.indexOf("\t")), l.slice(l.indexOf("\t") + 1)] as const),
    );
    const tag = `pi-pane-${process.pid}-${Math.random().toString(36).slice(2, 10)}`;
    // The probe title goes straight to the terminal through Pi's own output stream (so it is ordered with Pi's writes)
    // rather than through Pi's title, which reaches the terminal only on Pi's next redraw.
    const title = (t: string) => {
      try {
        if (ctx.mode !== "tui" || !process.stdout.isTTY) return false;
        process.stdout.write(`\x1b]2;${t.replace(/[\x00-\x1f\x7f]/g, "")}\x07`);
        return true;
      } catch {
        return false;
      }
    };
    try {
      // Pi may set its own title meanwhile (it does around a reload), so the tag is set again on each try.
      for (let i = 0; i < 20 && !selfPane; i++) {
        if (mine !== gen || !title(tag)) return null;
        selfPane = (await osascript(`tell application "Ghostty" to get id of (first terminal whose name is ${as(tag)})`).catch(() => "")) || null;
        if (!selfPane) await new Promise((r) => setTimeout(r, 50));
      }
    } finally {
      const before = selfPane ? titles.get(selfPane) : undefined;
      const back = before && before !== tag && !before.startsWith("pi-pane-") ? before : `π - ${folder}`;
      title(back);
      setTitle(ctx, back);
      process_[PANE] = selfPane;
    }
    return selfPane;
  }

  /**
   * The AppleScript prelude that finds this session's tab and the feed panes in it: `ownPane`, `theTab`, `feeds` (their ids,
   * in the tab's order) and `feedNames`.
   */
  const inTab = (self: string) => `
    set ownPane to first terminal whose id is ${as(self)}
    set theTab to missing value
    repeat with w in windows
      repeat with t in tabs of w
        if (id of terminals of t) contains ${as(self)} then set theTab to t
      end repeat
    end repeat
    set feeds to {}
    set feedNames to {}
    repeat with x in terminals of theTab
      if name of x starts with ${as(FEED_TITLE)} then
        set end of feeds to id of x
        set end of feedNames to name of x
      end if
    end repeat`;

  /** Open a feed on one run, to the right of this pane or below the last feed. Returns false when it could not. */
  /**
   * Open a feed on one run (serialised; see `serial`). autoClose: the feed closes itself once its run has finished,
   * even if it already had when the feed opened; without it (the newest run shown on request) it stays until q.
   */
  const openFeed = (ctx: ExtensionContext, log: string, autoClose = true) => serial(() => openFeedNow(ctx, log, autoClose), false);
  async function openFeedNow(ctx: ExtensionContext, log: string, autoClose: boolean): Promise<boolean> {
    if (!inGhostty) return false;
    const mine = gen;
    const self = await findSelf(ctx);
    if (mine !== gen) return false; // the session ended meanwhile
    if (!self) {
      notify(ctx, `Could not find this session's Ghostty pane, so the feed for ${runName(log)} did not open`, "warning");
      return false;
    }
    const root = git(ctx.cwd, "rev-parse", "--show-toplevel");
    const feed = join(root, "tools/feed/feed.mjs");
    if (!existsSync(feed)) {
      notify(ctx, `No ${feed} in this checkout, so the feed for ${runName(log)} did not open`, "warning");
      return false;
    }
    const title = FEED_TITLE + runName(log);
    const env = [`FEED_CLOSE=1`, `FEED_AUTOCLOSE=${autoClose ? 1 : 0}`, ...(piRoot() ? [`FEED_PI_ROOT=${piRoot()}`] : [])];
    const command = `${sh(process.execPath)} ${sh(feed)} ${sh(log)}`;
    const result = await osascript(`
      tell application "Ghostty"
        ${inTab(self)}
        if feedNames contains ${as(title)} then return "open"
        set hadFocus to id of focused terminal of theTab
        set cfg to new surface configuration
        set command of cfg to ${as(command)}
        set initial working directory of cfg to ${as(root)}
        set environment variables of cfg to {${env.map(as).join(", ")}}
        if (count of feeds) is 0 then
          set newPane to split ownPane direction right with configuration cfg
        else
          set newPane to split (first terminal whose id is (item -1 of feeds)) direction down with configuration cfg
          -- Stacked feeds share the column evenly; the split between this pane and the column stays where it was.
          perform action "equalize_splits" on newPane
        end if
        delay 0.15
        -- A split takes the keyboard. Give it back without bringing Ghostty to the front: step left to this pane, and
        -- only when Ghostty is in front anyway, focus whichever other pane had it.
        perform action "goto_split:left" on newPane
        if hadFocus is not ${as(self)} and hadFocus is not (id of newPane) and frontmost then focus (first terminal whose id is hadFocus)
        return id of newPane
      end tell`).catch((e) => `error: ${e.message}`);
    if (result.startsWith("error")) {
      notify(ctx, `Feed for ${runName(log)} did not open (${result}); tools/feed/feed ${runName(log)} follows it`, "warning");
      return false;
    }
    return true;
  }

  /** Close the given feed panes (by run log) in this session's tab; returns how many closed. */
  const closeFeeds = (ctx: ExtensionContext, logs: string[]) => serial(() => closeFeedsNow(ctx, logs), 0);
  async function closeFeedsNow(ctx: ExtensionContext, logs: string[]): Promise<number> {
    const self = await findSelf(ctx);
    if (!self || logs.length === 0) return 0;
    const titles = `{${logs.map((l) => as(FEED_TITLE + runName(l))).join(", ")}}`;
    const n = await osascript(`
      tell application "Ghostty"
        ${inTab(self)}
        set closed to 0
        repeat with i from 1 to count of feeds
          if ${titles} contains (item i of feedNames) then
            close (first terminal whose id is (item i of feeds))
            set closed to closed + 1
          end if
        end repeat
        return closed
      end tell`).catch(() => "0");
    return Number(n) || 0;
  }

  const openFeedNames = (ctx: ExtensionContext) => serial(() => openFeedNamesNow(ctx), [] as string[]);
  async function openFeedNamesNow(ctx: ExtensionContext): Promise<string[]> {
    const self = await findSelf(ctx);
    if (!self) return [];
    const out = await osascript(`tell application "Ghostty"
      ${inTab(self)}
      set AppleScript's text item delimiters to linefeed
      return feedNames as text
    end tell`).catch(() => "");
    return out.split("\n").filter(Boolean);
  }

  // ---- runs ------------------------------------------------------------------------------------------------------------

  pi.on("session_start", async (_event, ctx) => {
    if (ctx.mode !== "tui") return;
    // Every start (startup, reload, new, resume, fork) watches for the session now active.
    stop();
    live = ctx;
    const startNews = syncNews(syncCheckouts(ctx.cwd, { force: true }));
    if (startNews) pi.sendMessage({ customType: "factory-sync", content: `Factory files from main:\n${startNews}`, display: true }, { triggerTurn: false });
    const me = ctx.sessionManager.getSessionId();
    const checkouts = () =>
      git(ctx.cwd, "worktree", "list", "--porcelain")
        .split("\n")
        .filter((l) => l.startsWith("worktree "))
        .map((l) => l.slice("worktree ".length));
    const mine = (text: string) => text.includes(`=== dispatched by session ${me}`);
    const logs = () =>
      checkouts().flatMap((root) => {
        const runs = join(root, ".runs");
        try {
          return readdirSync(runs).filter((f) => f.endsWith(".log")).map((f) => join(runs, f));
        } catch {
          return [];
        }
      });
    const told = new Set<string>();
    const pending = new Map<string, string>();
    const shown = new Set<string>();
    const failed = new Set<string>();
    // A running run's progress: how much of its log is more than failed model calls, and since when it has not grown.
    const progress = new Map<string, { len: number; since: number; warned: boolean }>();

    // Called from a file watcher and a timer, where a throw would end Pi itself: nothing may escape it.
    const check = () => {
      if (live !== ctx) return;
      for (const file of logs()) {
        try {
          checkOne(file);
        } catch (e) {
          if (!failed.has(file)) notify(ctx, `factory-runs could not read ${file}: ${(e as Error)?.message ?? e}`, "warning");
          failed.add(file);
        }
      }
    };
    const checkOne = (file: string) => {
      if (told.has(file)) return;
      const text = readLog(file);
      if (text === null || !mine(text)) return;
      const run = stateOf(file, text);
      if (!run) return;
      // A run queued for a worker slot gets its feed when it starts.
      if (run.state === "waiting") return;
      if (run.state === "running") {
        // A run whose model calls keep failing never fails outright: it retries for as long as the network is down, and
        // 003-29 sat two days that way with nothing said (October 11). An hour with nothing but failed calls is told once,
        // and told again only after it has moved since.
        const failing = (l: string) => /✗ model error|… retrying/.test(l);
        const len = text.split("\n").filter((l) => !failing(l)).join("\n").length;
        const p = progress.get(file);
        if (!p || p.len !== len) progress.set(file, { len, since: Date.now(), warned: false });
        else if (!p.warned && Date.now() - p.since > 60 * 60_000 && failing(text.trimEnd().split("\n").pop() ?? "")) {
          p.warned = true;
          const last = text.trimEnd().split("\n").pop()!.trim();
          pi.sendMessage({
            customType: "factory-run-stalled", display: true,
            content: `Run ${runName(file)} in ${dirname(dirname(file))} has done nothing for over an hour but retry failed model calls (last: ${last}). Find why (the network, the provider), and stop it to resume it once the cause is gone (scripts/steer ${runName(file)} --stop, then scripts/work <task> --resume ${runName(file)}).`,
          }, { triggerTurn: true, deliverAs: "followUp" });
        }
        if (!shown.has(file)) {
          shown.add(file);
          openFeed(ctx, file).catch((e) => notify(ctx, `Feed for ${runName(file)}: ${e?.message ?? e}`, "warning"));
        }
        return;
      }
      told.add(file);
      const root = dirname(dirname(file));
      // The run has ended, so its checkout can take what main has merged meanwhile.
      const synced = syncNews(syncCheckouts(ctx.cwd, { force: true }));
      const end = run.end;
      const lines = [`Run ${runName(file)} in ${root} finished (${end}). Newest commit there: ${git(root, "log", "-1", "--format=%h %s")}.`];
      const target = /^=== verifier on (\S+)/.exec(text)?.[1];
      if (target) {
        const seen = join(root, target === "--map" ? "outcomes/map.seen.md" : target.replace(/\.md$/, ".seen.md"));
        if (existsSync(seen)) {
          const list = readFileSync(seen, "utf8").split("\n");
          const works = list.filter((l) => /^(- )?WORKS/.test(l)).length;
          const broken = list.filter((l) => /^(- )?BROKEN/.test(l)).length;
          lines.push(`The verifier's list ${seen}: ${works} WORKS, ${broken} BROKEN.`);
        }
      }
      // A worker's report and the whole of what was applied, so judging the work needs no fragments: what Requisite's models read
      // first, then everything else the commit changed, tests included (Owen, October 8: the rest was being fetched in
      // pieces, which leaves the reading to habit).
      const report = /^=== report\n([\s\S]*?)^=== end of report/m.exec(text)?.[1]?.trim();
      // Its beyond lines went to the outcome's held list (scripts/work), to be read with the verifier's findings.
      const heldNote = /^=== beyond: (.*)$/m.exec(text)?.[1];
      const shownReport = heldNote && report ? report.replace(/^beyond:\n[\s\S]*?(?=^(?:found):|(?![\s\S]))/m, `beyond: ${heldNote}\n\n`) : report;
      if (/^=== worker on /m.test(text)) lines.push(shownReport ? `The worker's report:\n${shownReport}` : "The worker filed no report.");
      const applied = /^=== applied to \S+ as ([0-9a-f]+)/m.exec(text)?.[1];
      // Whether the branch, with every run applied to it, still passes lint and the type check (scripts/work checks it
      // after applying): runs applied side by side each passed their own gates, not necessarily together.
      const broke = /^=== the branch does not check with this applied:\n((?:(?!===).*\n?)*)/m.exec(text);
      if (broke) lines.push(`The branch does not check with this applied:\n${broke[1]!.trim()}`);
      if (applied) {
        const listed = (f: string) => existsSync(join(root, f))
          ? readFileSync(join(root, f), "utf8").split("\n").map((l) => l.trim()).filter((l) => l && !l.startsWith("#"))
          : [];
        // The definitions first (the property's world and the law's trees): every kind, link, action and reading lives
        // there, so a change to how the world hangs together is read as that, whole, before anything built on it (Owen,
        // October 15: 005's relationships were built outside the definition and reviewed diff by diff).
        const defs = listed("tools/roles/definitions");
        const defDiff = defs.length ? git(root, "show", "--format=", applied, "--", ...defs) : "";
        lines.push(defDiff ? `The definitions changed (${applied}), in full:\n${defDiff}` : `Neither definition changed in ${applied}.`);
        const reads = listed("tools/roles/requisite-reads");
        const diff = reads.length ? git(root, "show", "--format=", applied, "--", ...reads, ...defs.map((r) => `:(exclude)${r}`)) : "";
        lines.push(diff ? `What Requisite's models read changed (${applied}), in full:\n${diff}` : `Nothing else Requisite's models read changed in ${applied}.`);
        const rest = git(root, "show", "--format=", applied, "--", ".", ...[...defs, ...reads].map((r) => `:(exclude)${r}`));
        lines.push(rest ? `Everything else the commit changed (${applied}), in full:\n${rest}` : `Nothing else changed in ${applied}.`);
      }
      if (synced) lines.push(`Factory files from main:\n${synced}`);
      // A run the subscription's session limit ended stopped mid-work, and this session shares that subscription, so it
      // was most likely stopped too: its notice arrives while it cannot act, and nothing else wakes it when the limit
      // resets. 006-1f and 006-1g ended this way at 21:22 on October 18 and the night was lost until 11:25 (October 19).
      // So the reset is a wake of its own: at the time the limit gives, this session is told which runs it ended.
      const limit = limitReset(text, new Date());
      if (limit) {
        lines.push(`The subscription's session limit ended it (it resets at ${limit.said}): its work was applied unfinished. ${limit.at ? `This session is woken when it resets, ${limit.at.toISOString()}.` : "The reset time could not be read as a time in its zone, so no wake is set for it."}`);
        if (limit.at) resumeAt(limit.at, `${runName(file)} in ${root}`);
      }
      lines.push(`Log: ${file}. Read it and carry on with the outcome.`);
      deliver(file, lines.join("\n"));
    };
    // One wake per reset, naming every run that limit ended.
    const resumes = new Map<number, { runs: string[]; timer: ReturnType<typeof setTimeout> }>();
    const resumeAt = (at: Date, run: string) => {
      const key = at.getTime();
      const r = resumes.get(key);
      if (r) { if (!r.runs.includes(run)) r.runs.push(run); return; }
      const runs = [run];
      // Two minutes past the reset, so the first call is not refused at the boundary.
      const timer = setTimeout(() => {
        resumes.delete(key);
        if (live !== ctx) return;
        pi.sendMessage({
          customType: "factory-limit-reset", display: true,
          content: `The subscription's session limit has reset (${at.toISOString()}). It ended these runs mid-work, each with its work applied unfinished: ${runs.join("; ")}. Read each log for where it stopped, judge what landed, and carry on with the outcome.`,
        }, { triggerTurn: true, deliverAs: "followUp" });
      }, Math.max(0, key - Date.now()) + 2 * 60_000);
      resumes.set(key, { runs, timer });
    };
    // A notice sent while the session is working waits in Pi's queue, and an interrupt drops what is queued (Owen saw a
    // finished run go unreported this way). So a notice counts as delivered only once it is in the session; until then
    // it is sent again whenever the session settles.
    const deliver = (file: string, content: string) => {
      pending.set(file, content);
      pi.sendMessage({ customType: "factory-run", content, display: true }, { triggerTurn: true, deliverAs: "followUp" });
    };
    redeliver = () => {
      if (live !== ctx || pending.size === 0) return;
      const said = JSON.stringify(ctx.sessionManager.getBranch().filter((e: any) => e.type === "custom_message" && e.customType === "factory-run").map((e: any) => e.content));
      for (const [file, content] of [...pending]) {
        if (said.includes(`Run ${runName(file)} in `)) pending.delete(file);
        else pi.sendMessage({ customType: "factory-run", content, display: true }, { triggerTurn: true, deliverAs: "followUp" });
      }
    };

    // Runs that had already finished when this session started were reported to the session that started them; runs
    // still going are shown (a feed already open for one is left as it is).
    for (const file of logs()) if (stateOf(file)?.state === "finished") {
      told.add(file);
      // A wake set for a reset is this runtime's; a reload or a new session sets it again from the logs, for any reset
      // still ahead of a run the limit ended.
      const text = readLog(file);
      const limit = text !== null && mine(text) ? limitReset(text, statSync(file).mtime) : null;
      if (limit?.at && limit.at.getTime() > Date.now()) resumeAt(limit.at, `${runName(file)} in ${dirname(dirname(file))}`);
    }
    for (const root of checkouts()) {
      const runs = join(root, ".runs");
      if (existsSync(runs)) watchers.push(watch(runs, () => check()));
    }
    session = { ctx, logs, mine };
    timer = setInterval(check, 5_000);
    check();
    notify(ctx, `Watching for runs this session dispatches${inGhostty ? ` (feeds: ${FEED_KEY})` : ""}`, "info");

  });

  pi.on("agent_settled", (_event, ctx) => {
    try {
      redeliver();
    } catch {}
    if (ctx?.mode !== "tui") return;
    const news = syncNews(syncCheckouts(ctx.cwd));
    if (news) pi.sendMessage({ customType: "factory-sync", content: `Factory files from main:\n${news}`, display: true }, { triggerTurn: false });
  });

  /** Hide the feeds of this session's runs, or show those still running (or the newest run, when none is). */
  async function toggleFeeds(sctx: ExtensionContext) {
    if (!session) return;
    const { logs, mine } = session;
    if (!inGhostty) {
      notify(sctx, "Feeds open as Ghostty splits; here, tools/feed/feed <run> follows a run", "info");
      return;
    }
    const all = logs().filter((f) => mine(readLog(f) ?? ""));
    const open = new Set(await openFeedNames(sctx));
    const visible = all.filter((f) => open.has(FEED_TITLE + runName(f)));
    if (visible.length) {
      const n = await closeFeeds(sctx, visible);
      notify(sctx, n ? `Feeds hidden (${n}); ${FEED_KEY} shows them again` : "Could not hide the feeds", n ? "info" : "warning");
      return;
    }
    const showing = all.filter((f) => stateOf(f)?.state === "running");
    if (showing.length === 0) {
      const newest = all.sort((a, b) => runName(b).slice(-15).localeCompare(runName(a).slice(-15)))[0];
      if (!newest) {
        notify(sctx, "This session has dispatched no runs yet", "info");
        return;
      }
      // A finished run shown on request stays until q.
      await openFeed(sctx, newest, false);
      return;
    }
    // The feeds appearing say it worked (openFeed says when one could not open). Pi rewrites a status line that
    // follows another in place without redrawing, so a "showing" notice would not be seen.
    for (const f of showing) await openFeed(sctx, f);
  }

  pi.registerShortcut(FEED_KEY, {
    description: "Hide the feeds of this session's runs, or show those still running",
    handler: (sctx) => toggleFeeds(sctx),
  });
  pi.registerCommand("feeds", {
    description: `Hide the feeds of this session's runs, or show those still running (also ${FEED_KEY})`,
    handler: async (_args, sctx) => toggleFeeds(sctx),
  });

  pi.on("session_shutdown", async () => {
    stop();
  });
}
