/**
 * The sandbox a worker or verifier runs in, so a role can only see and change its own workspace.
 *
 * scripts/work and scripts/verify load this with `pi -e`, and set ROLE_WORKSPACE to the role's workspace (also its
 * working directory), ROLE_ALLOW_READ to any other folders it must read (the Node install), and ROLE_ALLOW_WRITE to the
 * parts of the workspace it may change (default: all of it), each list separated by ":".
 *
 * - bash runs under the operating system's sandbox (sandbox-exec on macOS, through @anthropic-ai/sandbox-runtime):
 *   reading anything under /Users or the temporary folders is refused except the workspace and ROLE_ALLOW_READ,
 *   writing is refused outside ROLE_ALLOW_WRITE, and the only network is loopback (the role's own product). Its shell
 *   gets a folder beside the workspace as its home and temporary folder (outside the workspace's git repository: the
 *   product's credential store refuses to save inside one), no global git configuration, and the role's gates on its PATH.
 * - read refuses any path outside the workspace; write and edit refuse any path outside ROLE_ALLOW_WRITE.
 * - If the sandbox cannot start, bash refuses every command: a role never runs unconfined.
 */
import { execFileSync, spawn } from "node:child_process";
import { existsSync, realpathSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, isAbsolute, join, resolve } from "node:path";
import { SandboxManager } from "@anthropic-ai/sandbox-runtime";
import type { ExtensionAPI, BashOperations } from "@earendil-works/pi-coding-agent";
import { createBashTool } from "@earendil-works/pi-coding-agent";

const workspace = process.env.ROLE_WORKSPACE ? realpathSync(process.env.ROLE_WORKSPACE) : null;
const extraReads = (process.env.ROLE_ALLOW_READ ?? "").split(":").filter(Boolean);
/** The role's temporary folder: beside the workspace, so it is not inside the workspace's git repository. */
const scratch = workspace ? `${workspace}.tmp` : null;
const writable = workspace ? (process.env.ROLE_ALLOW_WRITE ?? "").split(":").filter(Boolean).map((p) => resolve(workspace, p)) : [];
if (workspace && writable.length === 0) writable.push(workspace);
/** macOS's /usr/bin/git is a shim that writes a cache in the user's temporary folder, which the sandbox denies; the
 * real git's folder goes first on the role's PATH. */
const realGit = (() => {
  try {
    return dirname(execFileSync("xcrun", ["--find", "git"], { encoding: "utf8" }).trim());
  } catch {
    return null;
  }
})();
// The role's own credentials (the subscription token its model runs on, any API key) are its process's, never its
// shell's: a shell that can print them can hand them to a page through a tool that reaches the network.
const SECRET = /(TOKEN|SECRET|PASSWORD|API_?KEY|PRIVATE_KEY|CREDENTIAL)/i;
const withoutSecrets = Object.fromEntries(Object.entries(process.env).filter(([k]) => !SECRET.test(k)));
const shellEnv = workspace
  ? {
      ...withoutSecrets,
      HOME: scratch!,
      TMPDIR: scratch!,
      GIT_CONFIG_GLOBAL: "/dev/null",
      GIT_CONFIG_NOSYSTEM: "1",
      PATH: [join(workspace, ".role/bin"), realGit, process.env.PATH ?? ""].filter(Boolean).join(":"),
    }
  : process.env;

/** The real path of p, or of its nearest existing parent for a file not yet written. */
function real(p: string): string {
  let at = p;
  const rest: string[] = [];
  while (!existsSync(at) && dirname(at) !== at) {
    rest.unshift(at.slice(dirname(at).length + 1));
    at = dirname(at);
  }
  return join(realpathSync(at), ...rest);
}

function within(p: string, roots: string[]): boolean {
  if (!workspace) return false;
  const r = real(isAbsolute(p) ? p : resolve(workspace, p));
  return roots.some((root) => r === root || r.startsWith(root + "/"));
}

/** The time limit, in seconds, of a command that names none. */
const DEFAULT_TIMEOUT_S = 15 * 60;

function sandboxedOps(): BashOperations {
  return {
    async exec(command, cwd, { onData, signal, timeout }) {
      // The sandbox library sets TMPDIR to its own shared folder; each command uses the role's own instead.
      const wrapped = await SandboxManager.wrapWithSandbox(scratch ? `export TMPDIR=${JSON.stringify(scratch)}; ${command}` : command);
      return new Promise((ok, no) => {
        const child = spawn("bash", ["-c", wrapped], { cwd, detached: true, stdio: ["ignore", "pipe", "pipe"], env: shellEnv });
        let timedOut = false;
        const kill = () => {
          try {
            if (child.pid) process.kill(-child.pid, "SIGKILL");
          } catch {
            child.kill("SIGKILL");
          }
        };
        const timer = timeout && timeout > 0 ? setTimeout(() => ((timedOut = true), kill()), timeout * 1000) : undefined;
        child.stdout?.on("data", onData);
        child.stderr?.on("data", onData);
        signal?.addEventListener("abort", kill, { once: true });
        child.on("error", (e) => (timer && clearTimeout(timer), no(e)));
        child.on("close", (code) => {
          if (timer) clearTimeout(timer);
          signal?.removeEventListener("abort", kill);
          if (signal?.aborted) no(new Error("aborted"));
          else if (timedOut) no(new Error(`timeout:${timeout}`));
          else ok({ exitCode: code });
        });
      });
    },
  };
}

export default function (pi: ExtensionAPI) {
  let ready = false;
  let why = "the sandbox has not started";
  const cwd = workspace ?? process.cwd();
  const bash = createBashTool(cwd, { operations: sandboxedOps() });

  pi.registerTool({
    ...bash,
    label: "bash (sandboxed to the workspace)",
    description: `${bash.description} A command that names no timeout is ended after ${DEFAULT_TIMEOUT_S} seconds; one that needs longer (a long fixture build) names its own.`,
    async execute(id, params, signal, onUpdate, ctx) {
      if (!ready) throw new Error(`bash is unavailable: ${why}`);
      // A command given no time limit gets one, so a call that cannot return (a background loop holding its output open)
      // ends in minutes, with everything it started, rather than holding the run for half an hour unseen (October 13,
      // 005-4). A command that truly needs longer (a long fixture build) names its own timeout.
      const timeout = (params as { timeout?: number }).timeout ?? DEFAULT_TIMEOUT_S;
      try {
        return await bash.execute(id, { ...params, timeout }, signal, onUpdate, ctx);
      } catch (e) {
        const message = String((e as Error)?.message ?? e);
        if (/timed out after/i.test(message)) {
          throw new Error(`${message}\nIt was ended with everything it started. A command holding a background process that never ends cannot return: run it in the foreground. One that truly needs longer than ${DEFAULT_TIMEOUT_S} seconds names its own timeout.`);
        }
        throw e;
      }
    },
  });

  pi.on("tool_call", async (event) => {
    if (!["read", "write", "edit"].includes(event.toolName)) return;
    const path = (event.input as { path?: string; file_path?: string }).path ?? (event.input as { file_path?: string }).file_path;
    if (typeof path !== "string") return;
    if (event.toolName === "read" && !within(path, [workspace!])) return { block: true, reason: `${path} is outside your workspace (${cwd})` };
    if (event.toolName !== "read" && !within(path, writable)) return { block: true, reason: `${path} is not yours to change; you may change ${writable.join(", ")}, and scratch files go in $TMPDIR (${scratch})` };
  });

  pi.on("session_start", async () => {
    if (!workspace) return void (why = "ROLE_WORKSPACE is not set");
    try {
      (await import("node:fs")).mkdirSync(scratch!, { recursive: true });
      await SandboxManager.initialize({
        // LibreOffice, headless, recalculates the pitch model (Owen, October 13): it needs these three system services
        // and its single-instance pipe, a unix socket it makes in /private/var/tmp when /tmp is closed to it. Tested under
        // these rules with srt: a conversion runs with them and exits 1 without them.
        network: {
          allowedDomains: [], deniedDomains: [], allowLocalBinding: true,
          allowMachLookup: ["com.apple.CoreServices.coreservicesd", "com.apple.SystemConfiguration.configd", "com.apple.DiskArbitration.diskarbitrationd"],
          // The tsx CLI talks to its child over a unix socket it makes in $TMPDIR (the role's scratch); refused, every
          // worker fell back to node --import tsx and said so in its report (October 12-14). Tested with srt: listen EPERM
          // without it, runs with it.
          allowUnixSockets: ["/private/var/tmp", scratch!],
        },
        filesystem: {
          denyRead: ["/Users", realpathSync(tmpdir()), "/private/tmp", "/tmp"],
          // The machine's lock folder (lock files only): one full gates run at a time across all roles.
          allowRead: [workspace, scratch!, "/private/tmp/requisite-locks", "/private/var/tmp", ...extraReads],
          allowWrite: [...writable, scratch!, "/private/tmp/requisite-locks", "/private/var/tmp"],
          denyWrite: [],
        },
      });
      ready = true;
    } catch (e) {
      why = `the sandbox could not start (${e instanceof Error ? e.message : String(e)})`;
    }
  });

  pi.on("session_shutdown", async () => {
    if (ready) await SandboxManager.reset().catch(() => {});
  });
}
