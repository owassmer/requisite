"""Plain HTML for the dossier: escaping, the few elements it uses, the page shell and its stylesheet. No script and no
network asset, so a page reads the same from any disk, offline (spec §13 WP16)."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from html import escape
from typing import Any


def esc(value: object) -> str:
    return escape(str(value), quote=True)


def anchor(prefix: str, name: str) -> str:
    """An element id for a name of any characters, one name to one id: letters, digits, '-', '_' and '.' stand as
    they are, any other character as '~' and its code point in hex and '~'."""
    out = [c if c.isascii() and (c.isalnum() or c in "-_.") else f"~{ord(c):x}~" for c in name]
    return f"{prefix}-{''.join(out)}"


def link(href: str, text: str, cls: str | None = None) -> str:
    """A link whose text is already HTML."""
    c = f' class="{cls}"' if cls else ""
    return f'<a href="{esc(href)}"{c}>{text}</a>'


def code(text: object) -> str:
    return f"<code>{esc(text)}</code>"


def pre(text: object, cls: str | None = None) -> str:
    c = f' class="{cls}"' if cls else ""
    return f"<pre{c}>{esc(text)}</pre>"


def para(html: str) -> str:
    return f"<p>{html}</p>"


def badge(text: str, kind: str) -> str:
    return f'<span class="badge {esc(kind)}">{esc(text)}</span>'


def heading(level: int, html: str, id: str | None = None) -> str:
    i = f' id="{esc(id)}"' if id else ""
    return f"<h{level}{i}>{html}</h{level}>"


def table(head: Sequence[str], rows: Iterable[Sequence[str]], cls: str | None = None) -> str:
    """A table whose cells are already HTML."""
    c = f' class="{cls}"' if cls else ""
    hs = "".join(f"<th>{h}</th>" for h in head)
    body = "".join("<tr>" + "".join(f"<td>{x}</td>" for x in row) + "</tr>" for row in rows)
    return f"<table{c}><thead><tr>{hs}</tr></thead><tbody>{body}</tbody></table>"


def dl(pairs: Iterable[tuple[str, str]]) -> str:
    """A definition list: each term is text, each value already HTML."""
    return "<dl>" + "".join(f"<dt>{esc(k)}</dt><dd>{v}</dd>" for k, v in pairs) + "</dl>"


def ul(items: Iterable[str], cls: str | None = None) -> str:
    c = f' class="{cls}"' if cls else ""
    return f"<ul{c}>" + "".join(f"<li>{x}</li>" for x in items) + "</ul>"


def as_json(value: Any) -> str:
    """A value as the record holds it, in indented JSON with sorted keys."""
    return json.dumps(value, indent=1, sort_keys=True, ensure_ascii=False)


def value(v: Any) -> str:
    """A record's field value as HTML: a scalar as text, None as 'none', anything else as its JSON."""
    if v is None:
        return "<em>none</em>"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str | int | float):
        return esc(v)
    return pre(as_json(v))


def document(*, title: str, header: str, body: str, css: str, case: str, synthetic: bool,
             record: tuple[str, str] | None) -> str:  # fmt: skip
    """A whole page. `record` names the committed record the page shows (kind, key), carried on <main> with its case so
    a reader or a test can tell which record a page is."""
    cls = ' class="synthetic"' if synthetic else ""
    data = f' data-case="{esc(case)}"' + (f' data-kind="{esc(record[0])}" data-key="{esc(record[1])}"' if record else "")
    return (
        "<!DOCTYPE html>\n"
        f'<html lang="en"><head><meta charset="utf-8"><title>{esc(title)}</title>'
        f'<link rel="stylesheet" href="{esc(css)}"></head>\n'
        f"<body{cls}>\n<header>{header}</header>\n<main{data}>\n<h1>{esc(title)}</h1>\n{body}\n</main>\n</body></html>\n"
    )


CSS = """body { font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; margin: 0; color: #1d1d1f;
  background: #fff; line-height: 1.45; font-size: 15px; }
body.synthetic { background: #fffaf0; }
header { background: #f2f2f4; border-bottom: 1px solid #d0d0d6; padding: 0.6em 1.5em; font-size: 0.9em; }
header nav a { margin-right: 1em; }
header .case { font-weight: 600; }
main { padding: 1em 1.5em 3em; max-width: 1200px; }
h1 { font-size: 1.5em; margin-top: 0.3em; }
h2 { font-size: 1.25em; border-bottom: 1px solid #e0e0e6; padding-bottom: 0.2em; margin-top: 1.6em; }
h3 { font-size: 1.08em; margin-top: 1.3em; }
table { border-collapse: collapse; margin: 0.5em 0 1em; font-size: 0.92em; }
th, td { border: 1px solid #d8d8de; padding: 0.25em 0.5em; text-align: left; vertical-align: top; }
th { background: #f6f6f8; }
pre { background: #f7f7f9; border: 1px solid #e4e4ea; padding: 0.5em; white-space: pre-wrap; word-break: break-word;
  font-size: 0.88em; margin: 0.2em 0; }
td pre { margin: 0; }
code { font-size: 0.92em; background: #f3f3f6; padding: 0 0.2em; word-break: break-all; }
dl { display: grid; grid-template-columns: max-content auto; gap: 0.25em 1em; }
dt { font-weight: 600; }
dd { margin: 0; }
.badge { display: inline-block; font-size: 0.78em; font-weight: 700; padding: 0.05em 0.45em; border-radius: 3px;
  margin: 0 0.25em; letter-spacing: 0.03em; text-transform: uppercase; vertical-align: middle; }
.badge.synthetic { background: #b35c00; color: #fff; }
.badge.rejected { background: #b00020; color: #fff; }
.badge.active, .badge.passed, .badge.met { background: #1b6e3a; color: #fff; }
.badge.proposed, .badge.awaiting { background: #8a6d00; color: #fff; }
.badge.failed, .badge.missed { background: #b00020; color: #fff; }
.badge.other { background: #555; color: #fff; }
ul.tree, ul.tree ul { list-style: none; padding-left: 1.3em; border-left: 1px dotted #b8b8c0; }
ul.tree li { margin: 0.25em 0; }
.node-kind { font-weight: 600; }
.stated { font-size: 0.88em; color: #444; }
.note { color: #444; font-size: 0.92em; }
.rule { background: #eef3fb; border-left: 3px solid #4a6fa5; padding: 0.4em 0.7em; margin: 0.6em 0; }
section.prose { border-left: 3px solid #d0d0d6; padding-left: 0.8em; margin: 1em 0; }
"""
