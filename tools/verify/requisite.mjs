#!/usr/bin/env node
// The verifier's `requisite`: runs the product's CLI through the bridge (tools/verify/bridge.py), passing its
// arguments and anything piped to it, and prints what the product printed, exiting as it exited.
//   requisite <args...>          the product's CLI, replaying recorded responses (a clean copy of the committed product)
//   requisite --file var/<path>  a file the product wrote, printed
//   requisite --ls var/<path>    a directory the product wrote, listed
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const port = readFileSync(join(dirname(fileURLToPath(import.meta.url)), "..", "port"), "utf8").trim();
const base = `http://127.0.0.1:${port}`;
const args = process.argv.slice(2);

async function stdin() {
  if (process.stdin.isTTY) return null;
  const chunks = [];
  for await (const c of process.stdin) chunks.push(c);
  return chunks.length ? Buffer.concat(chunks).toString("utf8") : null;
}

if (args[0] === "--file" || args[0] === "--ls") {
  const r = await fetch(`${base}/${args[0].slice(2)}?path=${encodeURIComponent(args[1] ?? "")}`);
  const body = Buffer.from(await r.arrayBuffer());
  if (!r.ok) { process.stderr.write(body.toString() + "\n"); process.exit(1); }
  process.stdout.write(args[0] === "--ls" ? JSON.parse(body.toString()).join("\n") + "\n" : body);
} else {
  const r = await fetch(`${base}/run`, { method: "POST", body: JSON.stringify({ args, stdin: await stdin() }) });
  const out = await r.json();
  process.stdout.write(out.stdout ?? "");
  process.stderr.write(out.stderr ?? "");
  process.exit(out.exit ?? 1);
}
