"""Reading a rendered dossier as a reader's browser would: each page's links, element ids, the record it shows, and
whether it is marked synthetic, following relative links from the index."""

from __future__ import annotations

import posixpath
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path


@dataclass
class Parsed:
    hrefs: list[str] = field(default_factory=list)
    ids: set[str] = field(default_factory=set)
    record: tuple[str, str] | None = None
    case: str | None = None
    body_classes: set[str] = field(default_factory=set)
    scripts: int = 0
    sources: list[str] = field(default_factory=list)  # src attributes: assets a page would load


class _Parser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.out = Parsed()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = {k: v or "" for k, v in attrs}
        if "id" in a:
            self.out.ids.add(a["id"])
        if tag == "a" and "href" in a:
            self.out.hrefs.append(a["href"])
        if tag == "link" and "href" in a:
            self.out.sources.append(a["href"])
        if "src" in a:
            self.out.sources.append(a["src"])
        if tag == "script":
            self.out.scripts += 1
        if tag == "body":
            self.out.body_classes = set(a.get("class", "").split())
        if tag == "main":
            self.out.case = a.get("data-case")
            if "data-kind" in a:
                self.out.record = (a["data-kind"], a["data-key"])


def parse(path: Path) -> Parsed:
    p = _Parser()
    p.feed(path.read_text())
    return p.out


def resolve(root: Path, page: Path, href: str) -> tuple[Path, str | None]:
    """The file a relative link from a page names, and its fragment."""
    target, _, fragment = href.partition("#")
    rel = page.relative_to(root).as_posix()
    joined = posixpath.normpath(posixpath.join(posixpath.dirname(rel), target)) if target else rel
    return root / joined, fragment or None


def crawl(root: Path) -> dict[Path, Parsed]:
    """Every page reached from root/index.html by following links, each parsed."""
    seen: dict[Path, Parsed] = {}
    todo = [root / "index.html"]
    while todo:
        page = todo.pop()
        if page in seen:
            continue
        parsed = parse(page)
        seen[page] = parsed
        for href in parsed.hrefs:
            target, _ = resolve(root, page, href)
            if target.suffix == ".html" and target not in seen:
                todo.append(target)
    return seen


def tree(root: Path) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}
