/**
 * The coordinator holds the posture the product is built from, whole, in its system prompt, as workers and reviewers do
 * (scripts/work). The principles have one home, design/posture.md, and reach whoever acts on them from there, rather than
 * restated in the coordinator's memory (Owen, October 27: fewer words, each principle in one home). It is main's posture,
 * since the coordinator's own checkout may stand on an older outcome branch; read at the start of each run, so an edit
 * merged to main reaches the next one.
 */
import { execFileSync } from "node:child_process";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

const PATH = "design/posture.md";

export default function grounding(pi: ExtensionAPI) {
  pi.on("before_agent_start", async (event, ctx) => {
    let content: string;
    try {
      content = execFileSync("git", ["show", `origin/main:${PATH}`], { cwd: ctx.cwd, encoding: "utf8", timeout: 10_000, stdio: ["ignore", "pipe", "pipe"] });
    } catch (e) {
      // Without it the coordinator works without its principles, so this is said, not passed over.
      ctx.ui.notify(`The posture (${PATH} on origin/main) could not be read into the system prompt: ${e instanceof Error ? e.message : String(e)}`, "warning");
      return;
    }
    const files = event.systemPromptOptions.contextFiles;
    const at = files.findIndex((f) => f.path === PATH);
    if (at >= 0) files[at] = { path: PATH, content };
    else files.push({ path: PATH, content });
  });
}
