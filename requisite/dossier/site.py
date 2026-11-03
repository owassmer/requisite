"""What a dossier is made of: the committed state of a case read by key (`Case`), the files the dossier writes
(`Files`), where each page of a case's dossier lives (`Site`), and the page being written with its links (`Page`).

The dossier states what a case's records say and nothing else: it reads the store and never a provider, the network or
the clock, so the same store renders the same bytes. Records are read in key order, never commit order (a build's
order is not the case's). A field a build writes from its own clock rather than from what it recorded is left out
(WALL_CLOCK), since the same state built twice differs in it.
"""

from __future__ import annotations

import posixpath
import re
import shutil
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

from requisite import core
from requisite.dossier.html import CSS, as_json, badge, document, esc, link
from requisite.store import Refused, Store, open_case

R = TypeVar("R", bound=core.Record)

# Fields a build fills from its own clock, not from a recording: two builds of one state differ in them, so no page
# shows them (coordinator, task 007-1: they are fixed at their source separately). The store's own commit times and
# proposal event times are never read here at all.
WALL_CLOCK: Mapping[str, tuple[str, ...]] = {
    "agent_run": ("at",),
    "synthetic_artifact": ("created_at",),
    "derivation": ("computed_at",),
}
AS_OF_FIELDS = ("as_of", "evidence_at")  # the instants a record states its evidence as of (spec I3; core.ScenarioPacket)
MARKER = ".requisite-dossier"  # names a directory as a rendered dossier, which a render replaces whole


def slug(key: str) -> str:
    """A file name for a key: readable, and one key to one name on a file system that ignores case."""
    readable = re.sub(r"[^a-z0-9._-]+", "_", key.lower()).strip("_.")[:60] or "record"
    return f"{readable}-{core.sha256(key)[:12]}"


def kind_name(kind: str) -> str:
    return kind if re.fullmatch(r"[a-z0-9_]+", kind) else slug(kind)


def shown(kind: str, body: Any) -> Any:
    """A record's body as the dossier shows it: whole, but for the fields WALL_CLOCK leaves out."""
    drop = WALL_CLOCK.get(kind, ())
    if not drop or not isinstance(body, dict):
        return body
    return {k: v for k, v in body.items() if k not in drop}


# ---------------------------------------------------------------- a case's committed state


class Case:
    """A case's committed state, read by key."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.store: Store = open_case(self.root)
        self.name = self.store.case
        self.synthetic = self.store.synthetic
        self.keys: dict[str, list[str]] = {k: self.store.keys(k) for k in self.store.kinds()}
        self._bodies: dict[tuple[str, str], Any] = {}
        self.proposals: dict[str, core.Proposal] = {}
        self.proposal_keys: dict[str, str] = {}
        self.histories: dict[str, list[tuple[str, list[core.GateResult]]]] = {}
        for key in self.keys.get("proposal", []):
            pid = str(self.body("proposal", key)["id"])
            current = self.store.proposal(pid)
            assert current is not None  # its submitted record is committed
            self.proposals[pid] = current
            self.proposal_keys[pid] = key
            self.histories[pid] = self.store.proposal_history(pid)

    def close(self) -> None:
        self.store.close()

    def has(self, kind: str, key: str) -> bool:
        return self.body(kind, key) is not None

    def body(self, kind: str, key: str) -> Any:
        if (kind, key) not in self._bodies:
            self._bodies[(kind, key)] = self.store.body(kind, key)
        return self._bodies[(kind, key)]

    def get(self, kind: str, key: str, cls: type[R]) -> R:
        body = self.body(kind, key)
        if body is None:
            raise KeyError(f"no {kind} {key} is committed in case {self.name}")
        return cls.model_validate_json(core.canonical_json(body))

    def records(self, kind: str, cls: type[R]) -> list[tuple[str, R]]:
        """Every record of a kind, in key order."""
        return [(k, self.get(kind, k, cls)) for k in self.keys.get(kind, [])]

    def bodies(self) -> Iterator[tuple[str, str, Any]]:
        for kind, keys in self.keys.items():
            for key in keys:
                yield kind, key, self.body(kind, key)

    def content(self, artifact: str) -> bytes:
        return self.store.content(artifact)

    def proposals_of(self, kind: str) -> list[core.Proposal]:
        """This case's proposals of a kind as they now stand, by id."""
        return [p for p in self.all_proposals() if p.kind == kind]

    def all_proposals(self) -> list[core.Proposal]:
        """This case's proposals as they now stand, by id."""
        return [p for _, p in sorted(self.proposals.items())]

    def as_of(self) -> list[tuple[str, list[str]]]:
        """Each as-of instant the case's records state (AS_OF_FIELDS, at any depth of a record, a proposal's payload
        included), with the kinds of record that state it."""
        found: dict[str, set[str]] = {}

        def walk(value: Any, carrier: str) -> None:
            if isinstance(value, dict):
                for k, v in value.items():
                    if k in AS_OF_FIELDS and isinstance(v, str):
                        found.setdefault(v, set()).add(carrier)
                    else:
                        walk(v, carrier)
            elif isinstance(value, list):
                for v in value:
                    walk(v, carrier)

        for kind, _, body in self.bodies():
            carrier = f"{kind} ({body.get('kind')})" if kind == "proposal" and isinstance(body, dict) else kind
            walk(body, carrier)
        return [(at, sorted(kinds)) for at, kinds in sorted(found.items())]


# ---------------------------------------------------------------- the files a dossier writes


class Files:
    """Every file of a render, by its path within the dossier, written at the end in path order."""

    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}

    def add(self, path: str, content: str | bytes) -> None:
        data = content.encode() if isinstance(content, str) else content
        if self.files.get(path, data) != data:
            raise ValueError(f"the dossier would write {path} twice with different content")
        self.files[path] = data

    def write(self, out: Path) -> None:
        """Write the dossier to `out`, replacing an earlier render there whole; a directory there that is not a
        rendered dossier is refused, never replaced."""
        if out.exists() and not (out / MARKER).exists():
            raise Refused(f"{out} exists and is not a rendered dossier; it is left as it is")
        partial = out.with_name(out.name + ".partial")
        if partial.exists():
            shutil.rmtree(partial)
        for path in sorted(self.files):
            target = partial / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(self.files[path])
        if out.exists():
            shutil.rmtree(out)
        partial.rename(out)


# ---------------------------------------------------------------- a case's pages


@dataclass
class Site:
    """The dossier of one case, under `prefix` in the dossier (empty for the case rendered, fuzz/<name>/ for each
    synthetic fuzz case beside it, spec I4), and what links its records to one another and to its neighbours'."""

    case: Case
    prefix: str
    files: Files
    parent: Site | None = None
    children: list[Site] = field(default_factory=list)
    refs: Any = None  # the cross references (requisite.dossier.refs.Refs), built once every case is open

    @property
    def family(self) -> list[Site]:
        root = self.parent or self
        return [root, *root.children]

    def index_path(self) -> str:
        return f"{self.prefix}index.html"

    def page_path(self, name: str) -> str:
        return f"{self.prefix}{name}.html"

    def kind_path(self, kind: str) -> str:
        return f"{self.prefix}kind/{kind_name(kind)}.html"

    def record_path(self, kind: str, key: str) -> str:
        return f"{self.prefix}record/{kind_name(kind)}/{slug(key)}.html"

    def json_path(self, kind: str, key: str) -> str:
        return f"{self.prefix}json/{kind_name(kind)}/{slug(key)}.json"

    def text_path(self, what: str, key: str, suffix: str) -> str:
        return f"{self.prefix}text/{what}/{slug(key)}{suffix}"

    def css_path(self) -> str:
        return "style.css"

    def write(self, path: str, title: str, body: str, record: tuple[str, str] | None = None) -> None:
        page = Page(self, path)
        self.files.add(path, document(title=title, header=header(page), body=body, css=page.href(self.css_path()),
                                      case=self.case.name, synthetic=self.case.synthetic, record=record))  # fmt: skip

    def proposal_path(self, pid: str) -> str:
        return self.record_path("proposal", self.case.proposal_keys[pid])


@dataclass(frozen=True)
class Page:
    """The page being written: its site and its path within the dossier, from which every link is relative."""

    site: Site
    path: str

    def href(self, target: str, anchor: str | None = None) -> str:
        rel = posixpath.relpath(target, posixpath.dirname(self.path) or ".")
        return rel + (f"#{anchor}" if anchor else "")

    def mark(self, site: Site) -> str:
        """The synthetic badge on a link from a real case's page into a synthetic case (spec I4: shown wherever it is)."""
        return badge("synthetic", "synthetic") if site.case.synthetic and not self.site.case.synthetic else ""

    def record(self, kind: str, key: str, text: str | None = None, *, site: Site | None = None,
               anchor: str | None = None) -> str:  # fmt: skip
        """A link to a committed record's page; a key the case does not hold is shown as text, saying so."""
        s = site or self.site
        shown_text = esc(text if text is not None else key)
        if not s.case.has(kind, key):
            return f'{shown_text} <span class="note">(no {esc(kind)} {esc(key)} is committed in case {esc(s.case.name)})</span>'
        return link(self.href(s.record_path(kind, key), anchor), shown_text) + self.mark(s)

    def proposal(self, pid: str, *, site: Site | None = None, anchor: str | None = None, text: str | None = None) -> str:
        """A link to a proposal's page with its status."""
        s = site or self.site
        p = s.case.proposals.get(pid)
        if p is None:
            return f'{esc(text or pid)} <span class="note">(no proposal {esc(pid)} in case {esc(s.case.name)})</span>'
        return (link(self.href(s.proposal_path(pid), anchor), esc(text or pid)) + " " + status_badge(p.status)
                + self.mark(s))  # fmt: skip

    def artifact(self, aid: str) -> str:
        kind = "synthetic_artifact" if aid.startswith("syn_art_") else "artifact"
        return self.record(kind, aid)


def status_badge(status: str) -> str:
    cls = {"rejected": "rejected", "active": "active", "promoted": "active", "validated": "active",
           "proposed": "proposed"}.get(status, "other")  # fmt: skip
    return badge(status, cls)


def header(page: Page) -> str:
    site = page.site
    case = site.case
    nav = [link(page.href(site.index_path()), "Index"), link(page.href(site.page_path("queue")), "Review queue"),
           link(page.href(site.page_path("kinds")), "Records by kind")]  # fmt: skip
    for name, title in FLOW_PAGES:
        if name in site.refs.flows:
            nav.append(link(page.href(site.page_path(name)), esc(title)))
    if site.parent is not None:
        nav.append(link(page.href(site.parent.index_path()), f"The case it fuzzes: {esc(site.parent.case.name)}"))
    synthetic = (badge("synthetic", "synthetic") + " a synthetic case (spec I4): its evidence is generated for fuzzing, "
                 "never observed, and is never evidence in a real case") if case.synthetic else ""  # fmt: skip
    instants = site.refs.as_of
    as_of = ("; ".join(f"{esc(at)} <span class=\"note\">({esc(', '.join(kinds))})</span>" for at, kinds in instants)
             if instants else "no record of this case states an as-of instant")  # fmt: skip
    return (f"<nav>{' '.join(nav)}</nav>"
            f'<div><span class="case">Case {esc(case.name)}</span> {synthetic}</div>'
            f"<div>As of: {as_of}</div>"
            '<div class="note">Analysis, not advice; prices are indicative unless quoted. Every value shown is a '
            "committed record's, beside the record that holds it.</div>")  # fmt: skip


FLOW_PAGES = (("f1", "Filed contracts (F1)"), ("rulebooks", "Rulebooks"), ("fuzz", "Fuzz reports"),
              ("wide", "Wide discovery"))  # fmt: skip

# ---------------------------------------------------------------- renderers, one per record kind

RecordRenderer = Callable[[Page, str, Any], str]
PayloadRenderer = Callable[[Page, core.Proposal], str]
Summary = Callable[[Page, str, Any], str]

RECORDS: dict[str, RecordRenderer] = {}  # a record kind's page body: (page, key, body) -> HTML
PAYLOADS: dict[str, PayloadRenderer] = {}  # a proposal kind's payload: (page, proposal) -> HTML
SUMMARIES: dict[str, Summary] = {}  # a record kind's line in its kind's list


def renders(kind: str) -> Callable[[RecordRenderer], RecordRenderer]:
    def register(fn: RecordRenderer) -> RecordRenderer:
        RECORDS[kind] = fn
        return fn

    return register


def renders_payload(kind: str) -> Callable[[PayloadRenderer], PayloadRenderer]:
    def register(fn: PayloadRenderer) -> PayloadRenderer:
        PAYLOADS[kind] = fn
        return fn

    return register


def summarizes(kind: str) -> Callable[[Summary], Summary]:
    def register(fn: Summary) -> Summary:
        SUMMARIES[kind] = fn
        return fn

    return register


def json_copy(site: Site, kind: str, key: str) -> str:
    """The record as committed, in indented JSON, but for its WALL_CLOCK fields; written beside its page."""
    path = site.json_path(kind, key)
    site.files.add(path, as_json(shown(kind, site.case.body(kind, key))) + "\n")
    return path


def write_css(files: Files) -> None:
    files.add("style.css", CSS)
    files.add(MARKER, "A dossier rendered by `requisite dossier` from a case's committed state; a render replaces it.\n")
