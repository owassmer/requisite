// What state a worker or verifier run is in, from its .runs/<run>.log. One definition, used by the coordinator's
// extension (.pi/extensions/factory-runs.ts: when to show a feed, when to report a finish) and by the feed itself
// (tools/feed/feed.mjs: when to close).
//
//   waiting   queued for a worker slot ("=== waiting for a worker slot", not yet "=== got a worker slot")
//   running   started and not finished
//   finished  a "=== finished ..." line anywhere (lines a coordinator appends later do not undo it), or the script
//             that wrote the log is gone without writing one (its pid is on the "=== steerable (pid N)" line; a log
//             from before that line existed counts as finished once nothing has touched it for two hours)
import { readFileSync, statSync } from "node:fs";

const STALE_MS = 2 * 60 * 60 * 1000;

function alive(pid) {
  try {
    process.kill(pid, 0);
    return true;
  } catch (e) {
    return e.code === "EPERM";
  }
}

/** { state: "waiting" | "running" | "finished", end?: string } for a log's text (and its mtime, for old logs). */
export function runState(text, mtimeMs = Date.now()) {
  const fin = /^=== finished:? ?(.*)$/m.exec(text);
  if (fin) return { state: "finished", end: fin[1].trim() || "finished" };
  const pid = Number(/^=== steerable \(pid (\d+)\)/m.exec(text)?.[1]);
  if (pid > 0 ? !alive(pid) : Date.now() - mtimeMs > STALE_MS)
    return { state: "finished", end: "without a finish line (its script is gone)" };
  if (/^=== waiting for a worker slot/m.test(text) && !/^=== got a worker slot/m.test(text)) return { state: "waiting" };
  return { state: "running" };
}

/** runState for a log file. */
export function runStateOf(log) {
  return runState(readFileSync(log, "utf8"), statSync(log).mtimeMs);
}
