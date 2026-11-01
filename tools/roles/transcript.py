"""A run's own record of its work, from its Pi event stream: each thought and message, each command and what it
returned, in order. Carried into a run that continues it (scripts/work, "Carries:")."""

import json
import sys


def text_of(content):
    out = []
    for c in content if isinstance(content, list) else [content]:
        if isinstance(c, dict):
            if c.get("type") == "text":
                out.append(c.get("text", ""))
            elif c.get("type") == "thinking":
                out.append("[thinking] " + c.get("thinking", ""))
            elif c.get("type") == "toolCall":
                args = c.get("arguments", {})
                shown = args.get("command") or args.get("path") or json.dumps(args)[:400]
                out.append(f"[{c.get('name')}] {shown}")
        elif isinstance(c, str):
            out.append(c)
    return "\n".join(t for t in out if t.strip())


print("# The interrupted run's record: its thoughts, messages, commands and their results, in order\n")
for line in open(sys.argv[1], encoding="utf-8"):
    try:
        e = json.loads(line)
    except ValueError:
        continue
    if e.get("type") != "message_end":
        continue
    m = e.get("message", {})
    role = m.get("role")
    body = text_of(m.get("content", []))
    if not body.strip():
        continue
    if role == "toolResult":
        body = body if len(body) <= 3000 else body[:1500] + "\n[... cut ...]\n" + body[-1500:]
        print(f"### result\n{body}\n")
    elif role in ("assistant", "user"):
        print(f"### {role}\n{body}\n")
