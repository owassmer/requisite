"""Contracts in the dossier: an IR rendered readably, every node with an anchor by its id and the rulebook sections that
state it (spec §8.7: "Every clause links to its IR node, questions and evidence"; spec §13 WP16); F1's filed series, each
reading of one into the IR with its support classification, assumptions and unsupported terms in the filing's words, its
bindings and each market's settled-market check (spec §9.3, §12 F1; addendum A05 §5.5); and each rulebook's sections in
spec §8.5's order, each cited node linked to the IR."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from requisite import core
from requisite.dossier.html import anchor, code, dl, esc, heading, link, para, pre, table, ul, value
from requisite.dossier.refs import ContractKey, refs, reports_of
from requisite.dossier.site import Page, Site, renders, renders_payload, summarizes
from requisite.fuzz.revise import EXAMPLES, examples_text
from requisite.fuzz.run import SECTION_ORDER, SECTION_TITLES


def node_anchor(node_id: str) -> str:
    return anchor("node", node_id)


def observable_anchor(observable_id: str) -> str:
    return anchor("observable", observable_id)


def section_anchor(name: str) -> str:
    return anchor("section", name)


def iso(v: object) -> str:
    return v.isoformat() if isinstance(v, datetime | date) else str(v)


def window(w: core.TimeWindow) -> str:
    start = iso(w.start) if w.start is not None else "no start"
    end = iso(w.end) if w.end is not None else "no end"
    return f"{'[' if w.start_closed else '('}{start}, {end}{']' if w.end_closed else ')'} ({w.tz})"


def cmp_value(v: core.CmpValue) -> str:
    return f"{iso(v.value)} ({v.kind})"


def ir_nodes(ir: core.ContractIR) -> dict[str, Any]:
    roots = [ir.trigger] + ([ir.payout.quantity] if ir.payout.quantity is not None else [])
    return {n.id: n for r in roots for n in core.walk(r)}


def contract_record_key(ref: core.Ref) -> str:
    """The key a contract IR is committed under (requisite/contract/harvest.commit)."""
    return f"{ref.id}|{ref.version}"


# ---------------------------------------------------------------- the IR


class IRView:
    """One IR on one page, with what states each of its nodes and what asks of it."""

    def __init__(self, page: Page, ir: core.ContractIR) -> None:
        self.page, self.ir = page, ir
        fam = refs(page.site).family
        key: ContractKey = (ir.id, ir.version)
        self.stated = fam.stated.get(key, {})
        self.asked = fam.asked.get(key, {})
        self.observables = {o.id for o in ir.observables}

    def obs(self, observable: str) -> str:
        return link(f"#{observable_anchor(observable)}", code(observable))

    def node_ref(self, node: str) -> str:
        return link(f"#{node_anchor(node)}", code(node))

    def stated_by(self, ident: str) -> str:
        found = self.stated.get(ident, [])
        if not found:
            return '<div class="stated">Stated by no rulebook section in this case.</div>'
        links = [self.page.proposal(pid, site=s, anchor=section_anchor(name), text=f"{pid} § {SECTION_TITLES[name]}")
                 for s, pid, name in found]  # fmt: skip
        return f'<div class="stated">Stated by: {"; ".join(links)}</div>'

    def asked_of(self, node: str) -> str:
        found = self.asked.get(node, [])
        if not found:
            return ""
        by_question: dict[str, list[tuple[Site, str, core.Judgment]]] = {}
        for s, k, j in sorted(found, key=lambda x: (x[2].config.provider, x[2].config.paraphrase_id,
                                                    x[2].config.render_id, x[2].state_hash, x[1])):
            by_question.setdefault(j.question.id, []).append((s, k, j))
        parts = []
        for qid, js in sorted(by_question.items()):
            links = ", ".join(self.page.record("judgment", k, f"{j.config.provider} {j.config.paraphrase_id}", site=s)
                              for s, k, j in js)  # fmt: skip
            parts.append(f"{code(qid)}: {links}")
        return f'<div class="stated">Asked of fuzz packets, in the rendering check, as {"; ".join(parts)}</div>'

    def describe(self, n: Any) -> str:
        if isinstance(n, core.Cmp):
            return f"{self.obs(n.observable)} {esc(n.op)} {esc(cmp_value(n.value))}"
        if isinstance(n, core.And):
            return "all of these hold"
        if isinstance(n, core.Or):
            return "any of these holds"
        if isinstance(n, core.Not):
            return "this does not hold"
        if isinstance(n, core.By):
            since = ("" if n.since is None else
                     f" from {esc(iso(n.since))} ({'included' if n.since_closed else 'excluded'})")  # fmt: skip
            return (f"holds at some published observation{since} up to {esc(iso(n.deadline))} "
                    f"({'included' if n.closed else 'excluded'})")  # fmt: skip
        if isinstance(n, core.At):
            return f"holds at {esc(iso(n.at))}"
        if isinstance(n, core.Throughout):
            return f"holds at every published observation in {esc(window(n.window))}"
        if isinstance(n, core.Interpretive):
            return (f"a resolver judges it by the question template {code(n.question_template)}, with "
                    f"{value(dict(n.bindings))}")  # fmt: skip
        if isinstance(n, core.NumCmp):
            return f"the number below {esc(n.op)} {esc(n.value)}"
        if isinstance(n, core.CountTrue):
            return f"the number of calendar days of {esc(window(n.days))} on which this holds"
        if isinstance(n, core.DurationIn):
            return f"the time within {esc(window(n.window))} during which this holds, in {esc(n.unit)}"
        if isinstance(n, core.ReadValue):
            bands = ", ".join(f"[{lo}, {hi}{']' if i == len(n.bands) - 1 else ')'}" for i, (lo, hi) in enumerate(n.bands))
            return (f"a number in {esc(n.unit)} a resolver reads by the question template {code(n.question_template)}, "
                    f"with {value(dict(n.bindings))} answered as one of the bands {esc(bands)}")  # fmt: skip
        if isinstance(n, core.ValueAt):
            return f"{self.obs(n.observable)}'s value at {esc(iso(n.at))}"
        if isinstance(n, core.Computed):
            return f"computed by the registered node {code(f'{n.node.id} v{n.node.version}')} from its inputs below"
        if isinstance(n, core.Exists):
            named = ", ".join(esc(s.name) for s in sorted(n.sources, key=lambda s: s.rank))
            governs = ("the highest-ranked source's record" if n.on_conflict == "highest_ranked" else
                       "the official record of the body that took the action")  # fmt: skip
            return (f"some {esc(n.member)} of {esc(n.domain)}, as recorded by {named}, meets this, each judged for "
                    f"itself; where their records conflict, {governs} governs; none does only once each of them has "
                    "been read through the observation cutoff")  # fmt: skip
        raise TypeError(f"no wording for an IR node of kind {type(n).__name__}")

    def children(self, n: Any) -> list[tuple[str, Any]]:
        if isinstance(n, core.And | core.Or):
            return [("", a) for a in n.args]
        if isinstance(n, core.Not):
            return [("", n.arg)]
        if isinstance(n, core.By | core.At | core.Throughout | core.CountTrue | core.DurationIn | core.NumCmp):
            return [("", n.expr)]
        if isinstance(n, core.Exists):
            return [("for each member: ", n.expr)]
        if isinstance(n, core.Computed):
            return [(f"input {name}: ", sub) for name, sub in n.inputs.items()]
        return []

    def tree(self, n: Any, label: str = "") -> str:
        kids = "".join(self.tree(c, lbl) for lbl, c in self.children(n))
        return (f'<li id="{esc(node_anchor(n.id))}">{esc(label)}{code(n.id)} <span class="node-kind">{esc(n.kind)}</span>: '
                f"{self.describe(n)}{self.stated_by(n.id)}{self.asked_of(n.id)}"
                + (f'<ul class="tree">{kids}</ul>' if kids else "") + "</li>")  # fmt: skip

    def render(self) -> str:
        ir = self.ir
        out = [heading(2, "Observables")]
        for o in ir.observables:
            sources = table(["source", "name", "rank", "role", "url"],
                            [[code(s.id), esc(s.name), esc(s.rank), esc(s.role), code(s.url) if s.url else "<em>none</em>"]
                             for s in o.sources])  # fmt: skip
            pairs = [("description", esc(o.description)), ("value type", esc(o.value_type)), ("unit", value(o.unit)),
                     ("published precision", value(None if o.precision is None else str(o.precision))),
                     ("schedule", value(o.schedule)), ("revision policy", esc(o.revision_policy))]  # fmt: skip
            if o.final_deadline is not None:
                pairs.append(("final by", esc(iso(o.final_deadline))))
            if o.final_fallback is not None:
                pairs.append(("if no final value", esc(o.final_fallback)))
            pairs.append(("sources", sources))
            out.append(f'<div id="{esc(observable_anchor(o.id))}">' + heading(3, code(o.id)) + dl(pairs)
                       + self.stated_by(o.id) + "</div>")  # fmt: skip
        out.append(heading(2, "Trigger") + para("Evaluated in Kleene three-valued logic (spec §6.3): unknown stays "
                                                "unknown (I6).")  # fmt: skip
                   + f'<ul class="tree">{self.tree(ir.trigger)}</ul>')
        p = ir.payout
        pairs = [("kind", esc(p.kind)), ("per unit", esc(f"{p.unit_value} {p.currency}")),
                 ("cap", value(None if p.cap is None else str(p.cap))),
                 ("rounding", esc(f"to {p.rounding_quantum}, {p.rounding_mode}") if p.rounding_mode is not None else
                  "<em>none: the unit value is paid as stated</em>"), ("payment", esc(p.payment))]  # fmt: skip
        if p.periods:
            pairs.append(("periods", ul(esc(window(w)) for w in p.periods)))
        quantity = f'<ul class="tree">{self.tree(p.quantity)}</ul>' if p.quantity is not None else "<em>none</em>"
        pairs.append(("quantity", quantity))
        out.append(heading(2, "Payout") + dl(pairs))
        t = ir.timing
        out.append(heading(2, "Timing") + dl([
            ("open", esc(iso(t.open))), ("close", esc(iso(t.close))), ("expiration", esc(iso(t.expiration))),
            ("observation cutoff", esc(iso(t.observation_cutoff))),
            ("settlement lag", esc(f"{'at most ' if t.settlement_lag == 'at_most' else ''}{t.settlement_lag_days} "
                                   f"{t.lag_days} days")),
            ("payment lag", esc(f"{t.payment_lag_days} {t.lag_days} days")),
            ("holiday calendar", value(t.holiday_calendar)), ("time zone", esc(t.tz)),
        ]))  # fmt: skip
        r = ir.resolution
        early = r.early_expiration if r.early_expiration == "none" else f"{r.early_expiration}, {r.early_expiration_at}"
        if r.early_expiration_rule:
            early += f", by {r.early_expiration_rule}"
        pairs = [("precedence (highest authority first)", esc(", ".join(r.precedence)) or "<em>none</em>"),
                 ("conflict policy", esc(r.conflict_policy.procedure + (f" among {', '.join(r.conflict_policy.sources)}"
                                                                        if r.conflict_policy.sources else ""))),
                 ("on unknown at expiry", esc(r.on_unknown_at_expiry)), ("early expiration", esc(early))]  # fmt: skip
        if r.on_unknown_at_expiry == "extend":
            pairs.append(("extension", esc(f"{r.extension_days} days, then {r.after_extension}")))
        if r.unknown_rule is not None:
            pairs.append(("an unknown at expiry is settled by", esc(r.unknown_rule)))
        v = r.void_transfer
        if v is not None:
            pairs.append(("a void returns", esc(v.returns + (f" {v.amount}" if v.amount is not None else ""))))
        out.append(heading(2, "Resolution procedure") + dl(pairs))
        if ir.exclusions:
            rows = [[esc(x.text), "operative" if x.operative else "a disclosure, not operative",
                     self.node_ref(x.ir_node) if x.ir_node else "<em>none</em>"] for x in ir.exclusions]  # fmt: skip
            out.append(heading(2, "Exclusions") + table(["text", "operative", "carried by"], rows))
        if ir.examples:
            rows = [[esc(x.scenario_id), esc(x.description), esc(x.expected), code(x.evaluation)] for x in ir.examples]
            out.append(heading(2, "Examples") + table(["scenario", "description", "the evaluator's outcome",
                                                       "its evaluation (content hash)"], rows))  # fmt: skip
        return "".join(out)


@renders("contract")
def contract(page: Page, key: str, body: Any) -> str:
    ir = core.ContractIR.model_validate_json(core.canonical_json(body))
    site = page.site
    out = [dl([("contract", code(ir.id)), ("version", esc(ir.version))])]
    families = refs(site).families.get((ir.id, ir.version), [])
    if families:
        out.append(para("Read from its filed series by: " + "; ".join(page.proposal(pid) for pid in families)))
    reports = reports_of(refs(site).family, contract=ir.ref)
    if reports:
        out.append(para("Fuzzed against its prose in: " + "; ".join(
            page.record("fuzz_report", k, k, site=s) for s, k, _ in reports)))  # fmt: skip
    out.append(IRView(page, ir).render())
    return "".join(out)


@summarizes("contract")
def _contract_line(page: Page, key: str, body: Any) -> str:
    return esc(f"{body['id']} v{body['version']}")


# ---------------------------------------------------------------- F1: filed series and their readings


def filings(page: Page, filed: core.FiledSeries) -> str:
    xs = refs(page.site).extractions
    rows = []
    for what, aid in (("contract terms", filed.terms), ("product certification", filed.contract)):
        if aid is None:
            rows.append([esc(what), "<em>the series record carries none</em>", ""])
            continue
        texts = [page.record("extraction", x, "extracted text") for x in xs.get(aid, [])]
        rows.append([esc(what), page.artifact(aid), ", ".join(texts) or "<em>no extraction is committed</em>"])
    return table(["filing", "artifact", "its text"], rows)


@renders("filed_series")
def filed_series(page: Page, key: str, body: Any) -> str:
    filed = core.FiledSeries.model_validate_json(core.canonical_json(body))
    case = page.site.case
    out = [dl([("series", code(filed.series)), ("drawn at", esc(iso(filed.as_of))), ("why it is in F1", esc(filed.why)),
               ("its series record", page.artifact(filed.record))])]  # fmt: skip
    out.append(heading(2, "Filings") + filings(page, filed))
    readings = readings_of(page.site, filed.key)
    out.append(heading(2, "Readings into the IR"))
    if readings:
        rows = [[page.proposal(p.id), esc(version_shown(p)), esc(p.payload["contract"]["support"]),
                 esc(p.payload["contract"]["equivalence"])] for p in readings]  # fmt: skip
        out.append(table(["contract proposal", "version", "support", "equivalence"], rows))
    else:
        out.append(para("No reading of this series is proposed in this case."))
    harvested = [k for k in case.keys.get("harvested_contract", []) if k.startswith(f"{filed.key}|")]
    if harvested:
        out.append(heading(2, "Harvested contracts committed") + ul(page.record("harvested_contract", k) for k in harvested))
    rows = [[code(t), page.artifact(a)] for t, a in sorted(filed.markets.items())]
    out.append(heading(2, "Harvested markets") + table(["market", "its record"], rows))
    return "".join(out)


@summarizes("filed_series")
def _filed_line(page: Page, key: str, body: Any) -> str:
    return esc(f"{body['series']}: {body['why']}")


def readings_of(site: Site, filing: str) -> list[core.Proposal]:
    found = [p for p in site.case.proposals_of("contract") if p.payload.get("contract", {}).get("filing") == filing]
    return sorted(found, key=lambda p: (version_of(p), p.id))


def version_shown(p: core.Proposal) -> str:
    rep = p.payload.get("representative")
    return f"v{rep['version']}" if rep else "no IR (unsupported)"


def version_of(p: core.Proposal) -> int:
    rep = p.payload.get("representative")
    return int(rep["version"]) if rep else 0


def market_results(p: core.Proposal) -> dict[str, str]:
    """Each market's line in the tests gate's detail, as requisite/contract/harvest.py's check states it ("  <ticker>:
    <result>" after "Markets:")."""
    out: dict[str, str] = {}
    for g in p.gates:
        if g.gate != "tests":
            continue
        listed = False
        for line in g.detail.splitlines():
            if line == "Markets:":
                listed = True
            elif listed and line.startswith("  ") and ": " in line:
                ticker, text = line[2:].split(": ", 1)
                out[ticker] = text
    return out


def bound(ir: core.ContractIR, b: core.ListingBinding) -> str:
    """The value a binding filled in a market's IR, as the IR holds it."""
    if b.node is None:
        return iso(getattr(ir.timing, b.slot))
    n = ir_nodes(ir)[b.node]
    if b.slot == "threshold":
        return iso(n.value if isinstance(n, core.NumCmp) else n.value.value)
    if b.slot in ("at", "since", "deadline"):
        return iso(getattr(n, b.slot))
    w = n.days if isinstance(n, core.CountTrue) else n.window
    return iso(w.start if b.slot == "window_start" else w.end)


@renders_payload("contract")
def contract_family(page: Page, p: core.Proposal) -> str:
    family = core.ContractFamily.model_validate_json(core.canonical_json(dict(p.payload)))
    c = family.contract
    site = page.site
    filed = site.case.body("filed_series", c.filing)
    out = [dl([
        ("filing", page.record("filed_series", c.filing)),
        ("filed terms read", ul(page.artifact(a) + "".join(
            " " + page.record("extraction", x, "(its text)") for x in refs(site).extractions.get(a, []))
            for a in c.filed_terms)),
        ("support (addendum A05 §5.5)", esc(c.support)),
        ("equivalence (addendum A09 §10.5: the scope of the claim that the IR means what the filing says)",
         esc(c.equivalence)),
        ("the representative IR", esc(f"{c.ir.id} v{c.ir.version}") if c.ir else "<em>none: unsupported</em>"),
    ])]  # fmt: skip
    out.append(heading(3, "Assumptions, in the filing's words")
               + (ul(esc(a) for a in c.assumptions) if c.assumptions else para("None.")))
    out.append(heading(3, "Unsupported terms, in the filing's words")
               + (ul(esc(u) for u in c.unsupported_terms) if c.unsupported_terms else para("None.")))
    rep = family.representative
    if rep is None:
        return "".join(out)
    rows = [[code(b.field), esc(b.slot), link(f"#{node_anchor(b.node)}", code(b.node)) if b.node else "the timing"]
            for b in family.bindings]  # fmt: skip
    out.append(heading(3, "Bindings: which IR value each market's listing field fills")
               + table(["listing field", "slot", "node"], rows))
    s = family.settlement
    out.append(para("Settlement value: a market record's expiration_value reports "
                    + (f"{link(f'#{observable_anchor(s.observable)}', code(s.observable))}, for the instant in its "
                       f"{code(s.at_field)}" if s else "no observable the bindings name") + "."))  # fmt: skip
    out.append(heading(3, esc(f"The representative IR: {rep.id} v{rep.version}")) + IRView(page, rep).render())
    results = market_results(p)
    records = dict(filed["markets"]) if filed else {}
    rows = []
    for m in family.markets:
        values = ", ".join(dict.fromkeys(f"{b.field}={bound(m, b)}" for b in family.bindings))
        committed = (page.record("contract", contract_record_key(m.ref), "its IR") if p.status == "active"
                     else "<em>not committed: the reading is not active</em>")  # fmt: skip
        record = page.artifact(records[m.id]) if m.id in records else "<em>not in the harvest</em>"
        check = esc(results[m.id]) if m.id in results else (
            "<em>the tests gate did not run</em>" if not any(g.gate == "tests" for g in p.gates)
            else "<em>the tests gate's detail names no result for it</em>")  # fmt: skip
        rows.append([code(m.id), record, esc(values), committed, check])
    out.append(heading(3, "Each market: its bound values and its settled-market check")
               + para('<span class="note">Each market\'s IR is the representative with its bindings filled from the '
                      "market's committed record; its check result is the tests gate's, as that gate states it "
                      "(requisite/contract/harvest.py, the settled-market check; its scope is in the gate's detail "
                      "above).</span>")  # fmt: skip
               + table(["market", "its record", "bound values", "committed", "settled-market check"], rows))
    return "".join(out)


@renders("harvested_contract")
def harvested_contract(page: Page, key: str, body: Any) -> str:
    c = core.HarvestedContract.model_validate_json(core.canonical_json(body))
    return dl([
        ("filing", page.record("filed_series", c.filing)), ("filed terms read", ul(page.artifact(a) for a in c.filed_terms)),
        ("support", esc(c.support)), ("equivalence", esc(c.equivalence)),
        ("the representative IR", page.record("contract", contract_record_key(c.ir), f"{c.ir.id} v{c.ir.version}")
         if c.ir else "<em>none</em>"),
        ("assumptions, in the filing's words", ul(esc(a) for a in c.assumptions) if c.assumptions else "<em>none</em>"),
        ("unsupported terms, in the filing's words",
         ul(esc(u) for u in c.unsupported_terms) if c.unsupported_terms else "<em>none</em>"),
    ])  # fmt: skip


def f1_page(site: Site) -> None:
    path = site.page_path("f1")
    page = Page(site, path)
    out = [para("The filed contracts F1 harvested (spec §9.3, §12 F1): each series with its filings and every reading "
                "of it into the IR, each a contract proposal checked against how its markets settled.")]  # fmt: skip
    for key, filed in site.case.records("filed_series", core.FiledSeries):
        rows = [[page.proposal(p.id), esc(version_shown(p)), esc(p.payload["contract"]["support"]),
                 esc(p.payload["contract"]["equivalence"])] for p in readings_of(site, filed.key)]  # fmt: skip
        out.append(heading(2, page.record("filed_series", key, filed.series))
                   + para(esc(f"Drawn at {iso(filed.as_of)}: {filed.why}."))
                   + filings(page, filed)
                   + (table(["reading", "version", "support", "equivalence"], rows) if rows
                      else para("No reading of it is proposed.")))  # fmt: skip
    site.write(path, "Filed contracts (F1)", "".join(out))


# ---------------------------------------------------------------- rulebooks (spec §8.5)


@renders_payload("clause")
def rulebook(page: Page, p: core.Proposal) -> str:
    rb = core.Rulebook.model_validate_json(core.canonical_json(dict(p.payload)))
    site = page.site
    ckey = contract_record_key(rb.contract)
    ir = site.case.get("contract", ckey, core.ContractIR) if site.case.has("contract", ckey) else None
    nodes = set(ir_nodes(ir)) if ir is not None else set()
    observables = {o.id for o in ir.observables} if ir is not None else set()
    reports = reports_of(refs(site).family, prose=p.id)
    examples = [(k, site.case.get(EXAMPLES, k, core.ScenarioExample)) for k in site.case.keys.get(EXAMPLES, [])
                if k.startswith(f"{p.id}|")]  # fmt: skip
    out = [para(f"The rulebook prose of {page.record('contract', ckey, f'{rb.contract.id} v{rb.contract.version}')}, "
                "in spec §8.5's sections and order. Each section names the IR nodes and observables it states; each "
                "links to its place in the IR, where every node lists the sections that state it.")]  # fmt: skip
    if reports:
        out.append(para("Fuzzed in: " + "; ".join(
            page.record("fuzz_report", k, k, site=s) + esc(f" (equivalence {r.equivalence})") for s, k, r in reports)))
    shown = list(rb.sections)
    if examples and "examples" not in {s.name for s in shown}:  # published examples take the section's place (spec §8.5)
        shown.append(core.RulebookSection(name="examples", text="", ir_nodes=[]))
        shown.sort(key=lambda s: SECTION_ORDER.index(s.name))
    for section in shown:
        cites = []
        for ident in section.ir_nodes:
            if ident in nodes:
                cites.append(page.record("contract", ckey, ident, anchor=node_anchor(ident)))
            if ident in observables:
                cites.append(page.record("contract", ckey, ident, anchor=observable_anchor(ident)) + " (observable)")
            if ident not in nodes and ident not in observables:
                cites.append(code(ident) + ' <span class="note">(neither a node nor an observable of the committed IR)'
                             "</span>")  # fmt: skip
        suspected = [
            page.record("fuzz_report", k, f"scenario {a.scenario}", site=s, anchor=anchor("scenario", a.scenario))
            + esc(f" ({a.cause}, {a.status}; Ochiai {score!r})")
            for s, k, r in reports for a in r.attributions for name, score in a.suspected_clauses if name == section.name
        ]  # fmt: skip
        body = pre(section.text) if section.text.strip() else para(
            "<em>Empty as drafted: the examples are the coverage scenarios that passed fuzzing, rendered by code with "
            "their IR outcomes (spec §8.5), never drafted.</em>" if section.name == "examples" else "<em>Empty.</em>")
        if section.name == "examples" and examples:
            rows = [[page.record(EXAMPLES, k, x.scenario_id), esc(x.description), code(x.expected), code(x.evaluation)]
                    for k, x in examples]  # fmt: skip
            body += para("Published by code from a run of this rulebook that passed: its passing coverage scenarios "
                         "(spec §8.6 step 1), each rendered with the evaluator's outcome (spec §8.5; requisite.fuzz.revise).")
            body += pre(examples_text([x for _, x in examples])) + table(
                ["scenario", "description", "the evaluator's outcome", "its evaluation (content hash)"], rows)
        out.append(f'<section class="prose" id="{esc(section_anchor(section.name))}">'
                   + heading(3, esc(SECTION_TITLES[section.name]))
                   + body
                   + para("States: " + (", ".join(cites) if cites else "<em>no IR node</em>"))
                   + (para("Suspected in fuzz attributions: " + "; ".join(suspected)) if suspected else "")
                   + "</section>")  # fmt: skip
    return "".join(out)


def rulebooks_page(site: Site) -> None:
    path = site.page_path("rulebooks")
    page = Page(site, path)
    rows = []
    for p in site.case.proposals_of("clause"):
        c = p.payload.get("contract", {})
        rows.append([page.proposal(p.id), page.record("contract", f"{c.get('id')}|{c.get('version')}"),
                     esc(", ".join(s["name"] for s in p.payload.get("sections", ())))])  # fmt: skip
    site.write(path, "Rulebooks", para("Each rulebook is a clause proposal: the clause drafter's prose of one contract "
                                       "version's IR (spec §8.5), through its schema and tests gates.")  # fmt: skip
               + table(["rulebook", "its contract", "sections"], rows))

