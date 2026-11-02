# Decisions

What was decided and why. Each line carries its reason and, where it is Owen's, his words or the date he gave it. The
work is judged against this file; its writer is the coordinator (a worker's report proposes changes in `decisions`).

## Scope

- Requisite builds the runtime of the build spec v3 (November 1, 2026): agents propose structure, decision models test
  bounded judgments, code commits state and computes consequences, humans authorize at G1–G4. Two flows: Discovery (deep
  and wide) and Construction. Out of scope: order placement, RFQ submission, account linking, retrospective critique of
  any specific trade, and anything presented as legal or investment advice. Every market adapter is read-only.
- Binding: the invariants I1–I10 (spec §3) and the interfaces (spec §4) as amended. Discretionary: module internals, as
  long as the acceptance tests pass. When unsure, the smaller implementation that preserves an invariant over a larger one
  that adds a feature (spec §0).
- The mathematical addendum is accepted (Owen, November 1: "1. I accept"). Its amendments A01–A12 enter the canonical §4
  module once, in `requisite/core/`; no second schema. Tier-0 behaviour is implemented where an amendment changes a Tier-0
  acceptance test (WP4, WP5, WP6, WP7, WP8, WP9, WP10, WP11b, WP14); C01's reduced costs are reported at Tier 0, since
  the LP's duals come with the solve, and the agent search loop over them is Tier 1.
- Acceptance tests the addendum replaces, so nothing is built to the old ones: WP10's "perfect instrument: residual equals
  premium" and "uncorrelated instruments: buys nothing" become the dominated-payoff fixture with premium tie-breaking and
  the zero-covariance counterexample (A06 §6.5); WP13's "exclusive set summing to 1.10 is flagged" becomes a quote-aware
  payoff-dominance diagnostic (A04 §4.5); WP8's "at least N+1 scenarios" becomes feasible unique-cause witnesses with
  infeasible obligations recorded (A09 §10.2); §11's "accepted_uncovered can never be a defect" becomes "accepted risk
  and component defects coexist" (A10 §11.1).
- Tiers (spec §0): Tier 0 runs end to end from a clean clone and is due Tue Nov 3, 09:00 ET; Tier 1 starts only after
  Tier 0 is green; the demo is Tue Nov 3, 13:30 ET. Never cut: I1–I10, the three-valued evaluator, the rendering check,
  the diff (spec §13).

## Discovery finds what it analyzes

- The discovery flow chooses its own corpus and its own lead. From the objective a human approves at G1, it defines its
  frame from the public record as of a pinned date, screens and samples it by a declared design, retrieves and freezes
  what it reads, and its leads decide which docket and which business the deep case takes. No one picks a docket
  beforehand (Owen, November 1: "the whole point of discovery is that it finds it independently through it's flow we
  build. If you decide beforehand then 1. we will likely hyperfixate on the docket, and 2. the whole repeatable flow that
  is supposed to find these is not faithfully tested"). This replaces spec §12's "proposes three dockets; Owen picks one"
  and §15's Sun 22:00 docket question, and removes the cut rule "Run F2 from a hand-picked docket comment instead of the
  F4 wide run", which is the hand pick this decision removes.
- An objective says what its brief means, never what today's adapters reach; the frame records which corpora it reached,
  and every estimate is a share of the frame's items with that reach stated beside it (Owen, November 1: "is this just
  one of the discovery flows? Why would we restrict to just federal dockets?"; spec §7.3 lists the corpora in priority
  order). F4's objective, drafted by the objective agent from spec §12's F2/F4 wording, was approved at G1 as drafted
  (Owen, November 1: "Approve as drafted").
- The dockets the coordinator surveyed by hand on November 1 enter nothing: no source identifier is written into code,
  prompts, tasks or fixtures before the flow produces it, and no template, question or structure is shaped to a source
  the flow has not yet chosen.
- Once the flow's version and as_of are frozen and it has run, its result stands. A change made after seeing a result is
  a defect fix, recorded with its cause, and the whole flow runs again from the same objective and as_of; it is never
  re-run until a preferred lead appears (addendum A01 §2.5: tests used adaptively to revise are not independent
  validation).
- The wide flow reads its frame by a two-phase design (coordinator's reasoning, November 1). Reading one item from
  regulations.gov is one request, and the key allows 1,000 an hour (measured November 1), so a broad frame cannot be read
  whole. Phase 1 is a stratified simple random sample of the frame without replacement, read and model-labelled. Phase 2
  is the gold sample Owen labels: a stratified simple random subsample of phase 1, whose strata may come from the model's
  phase-1 labels. A frame small enough to read whole is a phase-1 census, and then the estimator is exactly addendum
  A07's finite-frame form. The plain difference estimator (λ = 1) is the primary estimate, since its design-unbiasedness
  is exact; a tuned λ is reported beside it with its measured coverage (A07 §8.3). Two-phase theory is standard survey
  sampling (Särndal, Swensson and Wretman 1992, ch. 9), not the addendum's, and is proved in the tests by exact
  enumeration. The unit is the item, so a wide prevalence is the frame share of items (comments) whose indicator holds,
  for example "in a business's voice and describing exposure to this event family" (spec §7.2: items are comments,
  letters, filings, posts). A share of businesses would need a cluster design over submitters, which the product does not
  draw and so does not claim. Phase-2 strata are nested in phase-1 strata and formed from the model's phase-1 labels;
  each nonempty phase-2 stratum is allocated at least two gold items, or all of its items when it has fewer, so every
  item has a positive inclusion probability (A07 §8.2: pi_i > 0) and every stratum's residual variance can be
  estimated from its own gold (coordinator's reasoning: one gold item gives no within-stratum variance, and the design's
  unbiased variance estimator needs one; the 000-2 task, November 1).
- Wide inference as built (worker on 000-2, November 1, measured by simulation under the exact design, 1,000 draws).
  The interval is a score interval: each bound is the estimate minus or plus z times the SD evaluated at the phase-2
  stratum prevalences restricted to that bound (Wilson's interval for one stratum); a stratum's f is credited with
  reducing the variance only as far as that stratum's own gold shows, never borrowed across strata, since a judge's
  false negatives sit exactly where its f does not separate the labels. Wald under-covered when a heavy stratum's gold
  held no positive (0.918 at λ = 1, 0.657 cross-fitted); MOVER-of-Wilson missed on one side only. The tuned λ is
  cross-fitted leave-one-out, which keeps its point estimate exactly unbiased (proved by enumeration); its variance is
  asymptotic and its coverage measured by requisite/discovery/sampling.py's measure_coverage. The flow draws both
  phases with that module's draw_phase1 and draw_phase2, so the sample is the design the estimator assumes.
- Phase-2 allocation (coordinator, November 1, from 000-2's numbers). Within each phase-1 stratum, 80% of its gold goes
  to the phase-2 stratum the judge scores below the cut (the large stratum where its missed positives live) and 20% to
  the rest, each at least two. Equal allocation over-covered rare families (0.981–0.986, almost no upper misses) because
  their dominant stratum's gold expected under one positive; 80/20 brought the same scenarios to 0.970–0.978. Before the
  gold is drawn, measure_coverage on the actual phase-1 predictions gives each family's expected gold positives, and the
  run reports beside each family's interval its measured coverage and expected positives. Where a family is so rare in
  the frame that its interval over-covers at any allocation the labelling budget allows, the interval is still valid (it
  covers at least nominally) and wide for a stated reason: the gold saw almost none of it.
- F4's sample design, fixed before its frame's sizes were seen (coordinator, November 1, 21:20 ET, while the frame was
  still being taken). Every number below is a rule over the frozen frame, so nobody chooses a stratum, a seed or an item.
  - Phase-1 strata: each kept docket's comment count (core.FrameDocket.comments) in decades, [min_comments, 100),
    [100, 1,000), [1,000, 10,000) and [10,000, ∞); an item takes its docket's band. The count is known from the frame
    alone, so the strata depend on no label. The bands separate where single filers write from where mass campaigns of
    individuals concentrate (our reasoning: the largest dockets are dominated by form letters), so the voice indicator
    should differ across them, which is when stratifying gains precision. The phase-2 constraint fixes how few: each
    phase-1 stratum carries two phase-2 strata of at least two gold items each, out of about 130.
  - Phase-1 size: 1,200 items, read whole (one API request each, plus attachments from the file CDN), split equally over
    the nonempty strata; a stratum with no more items than its share is read whole, and what it leaves is shared equally
    among the rest, repeated until every share fits, the indivisible remainder going one each to the strata in band order.
    Equal shares read the smaller dockets more than their weight, where leads are likelier; the estimator weights strata
    by N_h/N, so the estimate is unaffected.
  - Seeds: phase 1's is the first 16 hex digits of the frame's content hash read as an integer; the stability subsample's
    and phase 2's are the same of the phase-1 draw's content hash with "stability" or "phase2" appended.
  - Phase-1 labels: both objective questions on every phase-1 item, by labels.PHASE1_CONFIG; exposure_to_measure is bound
    to the item's own measure (amended November 2, below: "What each record means, from the external evaluation").
  - Stability (spec §2.1, §7.2 step 3): a simple random subsample of 60 phase-1 items (spec: at least 50; ten more so
    refusals cannot take it under), every configuration of labels.stability_configs, both questions.
  - Phase 2: the stratifier is the judge's probability that the item is in the estimand's numerator, P(in_population) x
    P(exposure in labor, inputs or labor_and_inputs) from the phase-1 labels; the cut is 0.5; within each phase-1 stratum
    130/H gold items (H the nonempty phase-1 strata, rounded down, the remainder one each in band order) split 80/20 by
    the decision above, each phase-2 stratum at least two. Every gold item gets both questions. measure_coverage on the
    actual phase-1 predictions is run and reported before the draw.
- The evaluator's semantics where the definition is silent (worker on 001-1, accepted by the coordinator, November 1).
  A value is "for" an instant; At and ValueAt read exactly that instant; By and Throughout range over the published
  instants within their bounds. By, Throughout, CountTrue and DurationIn stay open until the observation cutoff, and a
  By or Throughout over no published instant is unknown; precedence_first_available is decided at the cutoff. Reason: a
  known truth never reverses as publications arrive, and nothing resolves on a provisional value (property-tested).
  extension_days counts in Timing.lag_days, and an extension moves both the expiration and the observation cutoff;
  final_fallback applies at max(expiration, observation_cutoff). A payout quantity still an interval at settlement leaves
  its transfer unresolved, naming the quantity. Per-period payments are made per period only on YES; binary or capped
  strips, and strips with early expiration, are refused until the definition says what they mean. A threshold over a
  reading, duration or computed number is stepped at the precision it is written to; the IR's constraints over an
  observable's values use an exact region-by-class encoding (A09 §10.2: no impossible history is a witness). A.T06 is
  covered by tests/contract/test_timing_gap.py (the evaluator's dates fed to the optimizer).
- Ten grammar findings from expressing the Western Grazers terms (001-1, November 1) are the coordinator's to settle in
  the definition, each where the construct is general (decisions, F1: never fitted to one contract): an existential over
  instruments ("a Qualifying Action") with interpretations keyed per instrument; a single-value band anywhere in a
  ReadValue; an unknown-at-expiry rule that defers to an exchange rule, and the split between "not documented, so NO"
  and "cannot be determined"; a source hierarchy for the evidence of interpretive conditions ("the official records of
  the body that took the action govern"); a settlement lag stated as a deadline; VoidTransfer optional where no void
  path exists; a date read from text; a CashTransfer with a known amount and an unresolved date; a missing-data rule for
  a payout quantity (A05 §5.3); and what a binary or capped payout pays per period.
- A listed series is read as an instrument family (worker on 001-4, accepted by the coordinator, November 1). The
  contract reader writes one representative market's IR (the first harvested market by ticker) and bindings saying which
  IR value each per-market listing field fills; code gives every market its IR from those, a pure function, and the
  proposal's payload is core.ContractFamily. Bindings and the reader's input are limited to the listing fields
  (kalshi.LISTING_FIELDS): the reader never sees an outcome or a price, so it cannot fit the check. f1.retrieve commits
  each series as a core.FiledSeries with the draw's instant, so later steps read the harvest from the store. A kind's
  registered schema check runs once its record validates (spec §5: the contract layer's validator is the IR type check
  and the evaluator). A replayed agent run is the proposal it already made (agents/trace.made_from).
- The F1 settled-market check (001-4): a settled market's expiration_value enters as an Observation of the observable
  the family's settlement binding names, from its highest-precedence consulted source, for the instant the bound listing
  field gives, taken as published at that instant with the market record as its artifact: the record states the value
  the exchange settled on, not when the source published it, and its own date (settlement_ts) falls after expiration,
  where a faithful cutoff would see nothing. The IR is settled at its own final instant, because exchanges settle early
  by rules outside the filing (Kalshi Rule 7.2 settled KXBRDEP's NO markets months before their latest expiration), so
  the check compares outcomes, not dates, and says so. A market reproduces when its status matches the result and any
  stated payout equals settlement_value_dollars. The gate passes only with at least one market checked and all checked
  markets reproduced; it reports whether any checked market lay at or next to its comparison's boundary (found by the
  evaluator: the comparison made inclusive or strict, and the adjacent market's strike), since only such a market tells
  an off-by-one strike or an inclusive bound apart.
- An eleventh grammar finding (001-4): the IR expires early only when its trigger is true, while Kalshi's terms expire a
  market early once its outcome is decided either way; a faithful IR of those terms stays pending until the latest
  expiration. In the definition batch with the other ten.
- The grammar batch is built tonight, beside the main path, by the coordinator (Owen, November 2, 16:45 ET: "the
  definition changes are not the slowest kind. You own them and can do them yourself"). Its first measure is Castle's
  own filed contract, KXCAGOAT: the constructs it needs are general (an existential over an open class of instruments
  with each instrument's conditions judged for it, a source hierarchy for the evidence of judged conditions, an outcome
  at expiry left to a named authority's rule, a settlement date stated as a deadline, expiry once the outcome is
  decided either way), and KXCAGOAT is read again under them by the same reader and checked against how its markets
  settled.
- F1's readings re-run, all five, after two causes were found in the first live run (coordinator, November 1, 22:50
  ET). The reader was not told that an observable's values are in the form its source publishes them, so it compared
  the football organizer's result with Kalshi's internal team identifiers, though the filed terms specify teams "by
  name" (KXUAEPLGAME: a correct counterexample of that reading). And the check called a market not checkable when its
  record states its outcome but no value of the observable (KXMAMDANIPOLL's "At least 60%"), though the ladder's
  outcomes bound the value: such markets are now checked jointly, one value of the observable having to give every
  market at that instant its outcome, and markets whose outcomes are all alike stay not checkable, since then no value
  is excluded. The prompt and the check are general; every series is re-read under them, the passing ones too, and the
  new readings stand whatever they show. The earlier proposals stay as the record.
- The verifier (scripts/verify; Owen approved its skill on November 1, on the condition that it gives the verifier
  enough of the analyst's context that a finding is never the verifier's own flaw). It checks each outcome from a clean
  copy of the committed code (git archive HEAD, no case stores), replaying recorded responses only, so every
  verification is also the cold start. It reaches the product only through tools/verify/bridge.py: the `requisite` CLI
  and the files the product writes under var/, never the code, tests or cassettes. Its skill tells it what the analyst's
  work is for, to learn each step from the product's own help and messages, and to rule out its own use before it
  calls anything BROKEN.
- Resolvability fuzzing (worker on 001-3, accepted by the coordinator, November 1). The rendering check asks every
  provider in the original wording; an interpretive condition is asked by its own library template, or as
  condition_holds over its template name and bindings where the library has none; a computed comparison is asked
  through its readings and published values, since arithmetic is code's (I7). The prose's outcome is the top of the
  resolvers' pooled distribution (equal weight per provider), a tie at the top read as ambiguous; the prose's transfers
  are the IR's where the outcomes agree, otherwise the evaluator's for the forced outcome at expiration without early
  expiration; scenarios are evaluated at the determination instant. The resolver template follows the procedure:
  resolve_contract, with void, extended and disputed variants where the contract can end so, each with its own frozen
  paraphrase set. Severity is the money at stake; mandatory scenarios are coverage witnesses and boundary scenarios;
  the held-out share is 30%, seeded by the contract version alone so a scenario keeps its set when the prose is
  revised. A failing scenario opens every live hypothesis, each open until checked (A09 §10.1); suspected clauses are
  ranked by the Ochiai coefficient over the scored scenarios.
- F4's estimand is a ratio, estimated as one (002-2, accepted by the coordinator, November 2). two_phase_ratio estimates
  R = theta_num / theta_den from the plain difference estimates; the variance of theta_num_hat - R theta_den_hat is the
  design's own unbiased estimator, one bilinear definition (_design_covariance) shared with the mean, exact for every
  fixed R (proved by enumerating every sample). Fieller's set is reported raw (two rays or the whole line where
  theta_den is not distinguishable from zero) and within [0, 1]. The interval to read is the score set, the variance
  evaluated at category probabilities restricted to the tested R, because Fieller covered 0.913 in F4's shape, every
  miss above, when the heaviest stratum's gold held no numerator item. Measured at 0.945 to 0.983 over four settings;
  it over-covers in F4's shape (0.983) because the stratum holding 65% of the frame expects 0.64 numerator items in its
  gold, which is valid and wide for that reason. Each family's interval is marginal. The gold-draw's coverage report
  measures this ratio (sampling.measure_ratio_coverage), not the numerator's frame share.
- F4 from the labels to the leads (coordinator, November 2, 12:05 ET, before any phase-1 label, gold label or lead
  question exists; every rule below is over data not yet seen).
  - Scope: the estimate is over the frame's items with readable text. A phase-1 item with none (nothing the reader
    could extract, labels.item_state) is outside what the run reads, so for the estimate its indicators are known 0 and
    its prediction 0; it is not drawn for gold (nothing would be labelled), and the estimate states how many there were
    beside the scope. Its labels are not guessed: the claim is about readable items only.
  - A question the phase-1 judge refused has no probability; its prediction is the mean of that question's probability
    over the answered items of the same phase-1 stratum. Any prediction fixed from phase-1 data keeps the difference
    estimator unbiased (the gold corrects it); this one uses only the judge's own answers.
  - A gold answer of "unsure" or not_established is unresolved, never read as 0 (A07 §8.1). The estimate is computed
    under the two completions that bound it: the lowest ratio (each unresolved label set to put the item in the
    denominator and out of the numerator where it can be) and the highest (into the numerator where it can be, and out
    of the denominator otherwise). The interval read is from the lower completion's lower bound to the upper
    completion's upper bound, and the number of unresolved labels is shown beside it.
  - Event families: the clause the objective approved ("Families are defined by the run from what it finds"). An agent
    proposes families over the measures the readable phase-1 items address, each a cost channel (labor, inputs, or both)
    and a mechanism; every such measure is in exactly one family (the proposal's tests gate). An item's family is its
    measure's (amended November 2, below).
  - Leads (spec §7.2 step 5): a lead is a phase-1 item in the estimand's numerator: by its gold labels where it has
    them, otherwise by the phase-1 judge's probability P(in_population) x P(exposed) >= 0.5. Each lead is asked the
    spec's item templates cost_driver, direction and seeks_certainty (phase-1 configuration), and an agent extracts its
    quantified impact as claims (spec §7.2: "extracted by an agent as a Claim proposal"). Ranked in three tiers:
    (1) it asks for relief, protection or predictability and states a quantified cost impact; (2) one of the two;
    (3) neither. Within a tier: gold-confirmed before judged, then the judge's probability (higher first), then the
    stated annual cost impact in one currency (larger first; none stated last), then item id. Reason: a business that
    states what the rule would cost it and asks for certainty is the one for whom protection is both sizable and
    wanted, which is what construction needs; the order is fixed before any lead is seen.
  - A stated impact (amended November 2, 17:50 ET, before any lead question was asked; Owen: "source quote validated
    (not numerical), agent translates unstructured -> a standard structured formatting, that structured formatting is
    validated"). Its quote is the filer's words: code finds it, exactly, in one part of the filing's text, a check of
    text against text. Its amount, currency and period are the impact extractor's reading of those words, in the
    answer's structured form (an exact decimal in units of the currency, an ISO 4217 code, a period from a closed set),
    which the answer type validates; code never matches figures against the words. The runner gives the extractor's
    answer type the texts it read, so a quote not found there goes back to it with the reason (runner.run's context),
    and one still not found is an error, never a skipped impact. An amount of money only: a percentage, hours or a
    headcount is not one unless the filing states the money it comes to (a percentage had been taken as an amount in
    dollars, and any impact moves a lead's tier). Cause: results._impacts matched the amount's digits against the quote,
    which refused every round amount (str() of 120000 normalized is 1.2E+5) and passed 114 for "$11,400"; a better
    pattern would still be code reading words. No `wide leads` had run, so nothing is re-run.
- What each record means, from the external evaluation (coordinator, November 2, 14:30 ET; Owen: "patterns and proxies
  standing in for meaning ... segments standing in for the curve"). The evaluation found records that claimed more than
  the operation that filled them, each verified in the code before this change. F4's are fixed here, at the definition,
  before any phase-1 label, gold label or estimate exists, so nothing is re-rolled: the flow runs again from the same
  objective (G1) and the same as_of, with each fix's cause recorded here.
  - The frame's events are the objective's. The objective puts in class measures "proposed but not final" and "final but
    not yet effective" as of 2026-11-01. The status test dropped every docket whose proposal a final rule followed,
    unread, so its "pending" meant only the first: 183 dockets of the live frame were dropped for that alone, 121 with a
    rule posted in the last year. Now a final rule keeps its docket when it takes effect after as_of, by the effective
    date the Federal Register states (regulations.gov states none: its effectiveDate is null in the listing and in the
    rule's record, checked on two rules), and a rule whose date the Federal Register does not state leaves the status
    unknown, kept and said, never in effect (I6). Every statement read is an artifact the docket names.
  - Each item is asked about the measure it addresses. A docket can hold several rulemakings; the source states which
    document each comment answers (commentOnDocumentId), and the item had discarded it, binding every comment to its
    docket's latest proposal. Now core.Item.addresses keeps it, and core.ItemMeasure, committed when the item is read,
    states that document with its own status as of the frame's as_of; where the source names none, the docket's own
    measure, said so. Event families group measures, not dockets.
  - The exposure question binds the objective's event class and horizon (exposure_to_measure v2), and answers
    not_in_class where the measure as stated is outside it. Version 1 bound only the measure and the loss definition,
    whose text refers to the event class and horizon without stating them, so the judges and the gold answered a
    narrower question than the estimand counts. Its paraphrase set is drafted again for v2 before labelling.
  - What could not be read is a fact about each part, recorded by the reader with whose fact it is
    (core.ItemPart.by): the source's (restricted, not served, empty, a file whose own bytes no reader reads) or ours (a
    format no reader here reads). Only an item with no readable part is outside the readable scope, read from its
    parts; an item holding a part of ours stops the run until it is read; a judgment state that cannot be built for a
    readable item is an error, never "unreadable". Before, every refusal, integrity failures included, was counted
    unreadable and taken out of the gold's scope, and an unreadable attachment made a readable filing unreadable.
  - A claim is an assertion: its id hashes who asserts it with its words and where they are. Identical bytes filed by
    two submitters share an artifact (I2) and are two claims; the part keeps where each filing's copy was retrieved.
  - An approval authorizes one gate. The store's human gate passes only on an approval at the gate the kind requires
    (core.HUMAN_GATE: G1 for an objective, a decision spec, an entity model and a lead's promotion); before, any gate's
    approval of the exact payload passed it.
  - The design the draws name is committed (core.SamplingDesign, kind sampling_design) with the gold draw, when its
    phase-2 cells exist. The coverage simulation before the gold is a statement about the design under a label model:
    where that model puts no item in the denominator it says the ratio is undefined and the gold is drawn regardless.
- F4's reading, after the read and before the gold draw (coordinator, November 2, 17:30 ET; each found in F4's own
  items before any gold label or estimate, so fixed at its source and the run continued from the same objective and
  as_of, with no item chosen by hand):
  - A PDF text layer can carry glyph codes, not characters (TREAS-DO-2026-0232-0055: a font with no Unicode map on page
    1, fonts whose Unicode map is wrong on pages 2-6, all read as punctuation and recorded as read). The PDF reader
    (pdf/2) verifies a suspect page against OCR, an independent reading of the same page, and uses OCR's text where the
    layer carries under half of OCR's words, saying so in the page's method. Suspect: set in a subset font with no
    Unicode map, or more than 15% symbols prose rarely holds (F4's 6,850 text-layer pages: the glyph-coded 0.63-0.65,
    the next 0.46, prose about 0.01). A page that passes the screen is taken from its layer unverified, and a garbling
    into plain letters would pass it; verifying every page by OCR was the alternative, read twice over thousands of pages.
    Superseded by pdf/4 (below): no page is accepted by a screen.
  - An item is the filing frozen when the sample was read: a re-fetch never replaces it, but a newer reader's reading of
    the same retrieved bytes does (items.committed), and `wide read` re-reads each sample item's files that an older
    reader read, keeping every part that reads the same, so its record stays true of what read it.
  - A provider that declines an input over its size limit (openai_decisions: 262,144 serialized bytes; 8 sampled filings
    are larger, up to 6.5 MB of exhibits) has refused to read it: its words are kept and read as a refusal, never an
    error that stops the run or an answer it did not give. The phase-1 rule fixed before the data then applies (a refused
    question takes its stratum's mean as the item's prediction), and the item can still be drawn for the gold sample,
    read whole by its labeller, so the estimate stays unbiased; such an item cannot rank as a lead on judges' answers it
    does not have.
  - pypdf leaves out the text of a form XObject it cannot decode and says so only in a logged warning (F4's read: "Impossible
    to decode XFormObject /Group_93: Reached maximum number of tokens per stream", a form past pypdf's parsing budget),
    and the page was recorded as read whole from its text layer (coordinator, November 2, 17:40 ET). The PDF reader
    (pdf/3) reads such a page by OCR, which reads the rendered page whole, and says why in the page's method; where OCR
    cannot read it either, the layer's text stands and the page is recorded as read in part. Its precondition is held:
    pypdf's warnings silenced would make every page pass, so the reader then refuses to run. pdf/3 reads as pdf/2
    wherever pypdf decodes every part of every page, so pdf/2's recorded reading of the same bytes is taken as pdf/3's
    there and its OCR is not run again (136 minutes of serial OCR); no label had been asked over these readings.
  - Every text layer is checked against OCR (pdf/4; coordinator, November 2, 17:55 ET, before any gold draw; Owen: "Code
    should never attempt to deterministically check something that is inherently unstructured"). pdf/2's screen (symbols
    prose rarely holds, or a font with no Unicode map) decided by a pattern over the text which layers were accepted
    unread by any second reader: 3,544 of F4's text-layer pages. Now every page with a text layer is read by OCR too and
    the two readings are compared as text: the layer stands where it carries at least half the words OCR reads, OCR's
    text is used where it does not, and the page's method states the share. Where OCR reads fewer than 20 words or cannot
    read the page, the layer stands and its method says it was not checked. pdf/3's reuse of pdf/2's recordings goes:
    pdf/4 differs from both on every text-layer page. Labels asked over pdf/3's reading replay wherever pdf/4 reads an
    item's text the same; an item whose text it changes is asked again before the gold draw.
- A frame item is a comment, held once (coordinator, November 2, 15:45 ET). The live F4 frame failed to freeze because
  regulations.gov listed one comment id, CMS-2025-0058-0103, for two records (a withdrawn one posted February 2 and one
  posted February 25, distinct objectIds), and the frame refuses an item twice. The frame now holds each comment id once,
  as its record not withdrawn, or the latest where every one is withdrawn, counts it once, and says in its docket's
  reason which ids were listed twice (frame.one_per_comment). Found and fixed before any sample was drawn; the frame is
  taken again from its recorded responses, from the same objective and as_of.
- What the wording test's records mean, from the external evaluation (coordinator, November 2, 16:00 ET). Each was a
  record claiming more than the operation that filled it; each is fixed where the record is defined, and each has a test
  built from the evaluation's counterexample (tests/fuzz).
  - What the prose pays is read, not assumed. The IR's payment terms (what one contract pays, when it is determined and
    paid, what a void returns) are stated by code from the IR (statements.payment_terms) and asked of the rulebook, by
    every provider, as core.TermReading (states, states otherwise, not stated; the reading is its pooled answers' top,
    ambiguous where tied). Every settled outcome rests on its own term, paying nothing included: YES on what YES pays,
    NO on what NO pays, a void on its refund, and the payment date wherever anything is paid; pending and extended on the
    payment date (core.terms_used). A scenario's prose transfers are the evaluator's, the one settlement function, only
    where every term its outcome rests on was read as stated: the prose states exactly the IR's term and the outcome
    agrees, so the one settlement function gives the prose's transfers. A term read otherwise or not stated leaves the
    scenario's payment unestablished: no equivalence is claimed, and the critical-mismatch gate does not pass where a
    critical scenario's payment is unestablished. It is never a counterexample, since a reading of the whole rulebook
    cannot say one scenario pays differently (the prose may state the term otherwise only where it does not apply:
    an exception, a rate on a zero quantity, another split of the lags). Comparing each scenario's payments themselves
    needs the readers to state what the prose pays on that scenario (A09 §10.1); that per-scenario payment reading is
    the full method, owed after the demo, and this one claims only what it reads. A term not stated, an outcome that determines no payment, or an amount not yet determined leaves the
    payment unestablished: None, never an empty list that says "pays nothing" (I6), and no equivalence is claimed.
    One arithmetic, two independently grounded inputs: before, the IR's transfers stood for the prose's whenever the
    outcomes agreed, and a prose paying 100 times the amount a month late scored fidelity 1.0. Nine rounds of the
    definition review sharpened this. The clause drafter's prompt now asks for the payout and what a void returns.
  - A boundary point is infeasible only with a proof (core.BoundaryPoint.status); a point no search built is
    unconstructed, counted unmet by the coverage gate. The computed-number search moves a second input when one alone
    does not reach the value (z = 2x + 3y reaches 501 only so).
  - The rendering check is per provider: a packet passes only where every provider answered and each recovered the
    intended state; a refusal is a miss, never dropped before pooling.
  - The development/held-out split is over what the readers are shown (fuzz.concrete_input, the renderer request):
    scenarios with one concrete input are one test case and fall in one set.
  - A failed rendering check opens a rendering attribution; it does not confirm one, since the check shows the readers
    did not recover the state, not whether the renderer, the question or the readers caused it.
  Tests derive what they expect from the rule and the plan (the split, each scenario's truths), never from ids one
  split happened to give (Owen, November 2: "Hardcoded values will inherently drift").
- The static dossier (007-1, accepted by the coordinator, November 2): it renders from a case's store alone, in key
  order, never commit order. It leaves out the fields a build fills from its own clock (AgentRun.at,
  SyntheticArtifact.created_at, DerivationReceipt.computed_at) until they are fixed at their source, so identical stores
  render identical bytes. It copies every text it links and shows other artifacts by metadata. Every link stays inside
  the dossier, with source URLs shown as text. Thresholds are shown as the rule they are, with the spec section and the
  product constant that holds them. The review queue's gate per kind is pinned by a test to core.REQUIRED_GATES.
- Public sources are reached through their official APIs and the files those APIs link to, identified honestly (spec
  §14). Requisite's User-Agent is `Mozilla/5.0 (compatible; RequisiteResearch/1.0; +https://github.com/owassmer/requisite)`,
  the conventional self-identifying crawler form: regulations.gov's file CDN refuses agents without the Mozilla/5.0
  prefix (measured November 1: 403 for curl, httpx and a bare product token; 200 for this form), and the form names the
  runtime and where to learn about it, so nothing is impersonated. The regulations.gov key goes in the X-Api-Key header,
  never in a URL, so it never reaches a cassette key or a log.
  Amended November 2 at Owen's question ("can we not get the sources another way than the API?"): the rule's reason is
  authoritative evidence, which a verified mirror keeps, so a public mirror is a source where its copy is checked against
  the official one. Mirrulations (Moravian University; AWS Open Data, s3://mirrulations) stores each regulations.gov
  comment record as the API's detail response (checked: equal in every attribute and its included attachments, lacking
  only the API's "links" block; current through November 1, 2026). The reading step takes a comment's record from it
  where the copy is current against the frame's official listing: same id and docket, the record's modifyDate equal to
  the listing's lastModifiedDate, and every field the two share equal; it reads the API otherwise
  (requisite/evidence/mirrulations.py, items.current). The artifact names the mirror as its source. The key's hourly
  limit is then spent only where the mirror is not current. The frame's listing stays on the API: the mirror keeps one
  file per comment and no listing with posting dates, so listing a docket from it would mean fetching every record.
- An item's text is everything its submitter filed: the comment and the text of each attachment. A PDF page without a
  text layer is read by OCR (tesseract, installed on this machine; CI installs it), each page's method recorded in the
  extraction, so a scanned business letter is read rather than counted as silent (coordinator's reasoning: an item the
  judges cannot read would be labelled not_established for a reason of ours, not the world's). Extractions are committed
  state, so a replay reads them from the frozen case rather than re-running OCR on another machine.
- F1's filed contracts are drawn by a rule over the filed record, fixed here before the draw runs (coordinator,
  November 1). The Western Grazers contract is in by name: spec §9.3 names it as the reference specimen, and it is
  Kalshi's series KXCAGOAT, "California goat herder wage obligations" (its product certification and contract terms
  are filed PDFs; found November 1 through the read-only Kalshi API). The others: among Kalshi series with filed contract
  terms (a non-empty contract_terms_url), ordered by sha256("requisite-F1|" + ticker), a series is eligible when one of
  its markets settled in the 45 days before the pinned instant (a market counts as settled by the instant when its
  recorded settlement_ts is at or before it). Its strike types are those of its latest settled event's markets settled
  by the instant (the event of its market settled last; a tie goes to the smaller event ticker), and it is eligible
  under each. Each strike type's draw is the first eligible series in hash order; the walk stops once four types are
  drawn, so at most four series are drawn. A drawn series' harvested markets are those of the event that drew it; the
  named series' are all its markets. The named series is apart from the draw: it counts toward no type, and the draw
  passes over it to the next in hash order. (Bounded by the worker on 001-2 before any live draw ran, accepted by the
  coordinator, November 1: the first wording needed every settled Kalshi market, millions; all 15,589 listed series
  carry contract_terms_url, and Kalshi offers no count, no strike filter and no archive time filter. Forty-five days is
  longer than a month and shorter than the archive's lag, cutoff 2026-09-02 when read November 1, and keeps the compared
  outcomes recent, so most likely settled under the terms contract_terms_url now links to.) The pinned instant is
  2026-11-02T00:00:00Z, fixed here before the live draw runs: the first midnight UTC after the rule was fixed. Kalshi
  requests are keyed by path, query and the instant the reads serve, in UTC: the record changes over time, so the same
  request for another instant is another request. The hash makes the choice
  independent of anyone's view of which contracts are interesting; spec §12 asks for 2–5 contracts and spec §9.3 for
  every strike type over the harvest as a whole. The IR grammar comes from spec §8.1 and the definition, never fitted to
  a drawn contract: a filed construct the IR cannot express is a finding about the grammar, fixed there when the
  construct is general, and the contract is otherwise unsupported or supported with explicit assumptions (A05 §5.5).

- A deep case opens from a lead approved at G1 (003-2, `requisite deep open`; coordinator, November 2). Its evidence is
  the lead's filing exactly as the wide case committed it (item, artifacts, extractions, its claims and the lead's
  quoted impacts, each checked equal to the wide record) and the ItemMeasure of the document it addresses; nothing else
  is gathered (spec §7.1 names no gathering step). Its event is that measure taking effect within the approved
  objective's horizon. The entity model's horizon and loss definition are the objective's, set by code; the decomposer
  cites each sourced number's claim with the words that state it (provenance, checked as text against the claims it was
  given; the number is its reading) or states the assumption and why. The decision time is the first instant everything
  the case holds is observable, never before the frame's as_of: the filing is read after the frame is taken and a
  fetched text is evidenced only from its retrieval (I3; A11 §11.3), so at the frame's as_of none of the business's
  words would be evidence. Objective, alpha, tolerance (not negative: core.DecisionSpec), floor, actions, information,
  recourse and currency are the human's, as given, never defaulted. The judges read each of the evidence's words once:
  a claim within another held claim of the same extraction (a lead's quote, within its part) stays a claim of the case
  and is not shown to them again (deep._evidence). Only `wide objective`, `market f1-retrieve` and `contract read` begin a case; every other
  command opens the case committed at --case-dir or is refused, so a mistyped path is never a new empty case.

## No escape routes, loud or quiet

- Owen, November 1: "I am also uncompromisingly averse to 'loud failure' escape routes and deferrals/dismissals/shortcuts
  to reasoning." Making a gap visible (an error, an unknown, a flag, "needs review", a BROKEN line, a "not true" in a
  report, an OPEN item) is not the job done; the bar is no failure (Handoff's decision of October 6 is the same: aiming at
  visible failure gives an exit from the bar). Nothing waits on a later tier, a follow-up, a held list or Owen when
  reasoning can settle it now; nothing is called harmless, flaky, pre-existing or cosmetic without being checked. How it
  reads in the product, where unknowns are real findings, is in design/posture.md.

## Numbers

- Two kinds of probability, held apart (addendum A04 §4.1). Scenario and branch probabilities (a ProbabilityEstimate,
  an event tree's branches, the optimizer's scenarios) are exact: Fraction, Decimal or int, summing to exactly one, never
  normalized (addendum §4.2), with a float refused. Expectations under them (CVaR, E[D], a breach probability) are exact
  Fractions, since a third of a dollar has no finite Decimal. A decision model's answer distribution is the provider's
  float output, read at the step the provider reports at (measured November 1: Decisions two decimals, Clef four, Jev
  one). A reported vector is a rounding when some distribution lies within half a step of every reported value: it is
  corrected to the closest such distribution (an equal shift of every value, clipped to its own interval), and the
  Judgment keeps both the reported and the corrected values (addendum A04 §4.2: "Numerical round-off corrections need a
  declared tolerance and log"). Beyond that tolerance it is no rounding, and the judgment has no
  distribution, with the reason; it is never normalized. The stability kernel then computes over distributions that sum
  to one within the oracle's 1e-10 (worker on 000-1 and coordinator, November 1). Money is Decimal, and the optimizer's
  cash accounting is exact Decimal, in which a rounding raises.
- The liquidity optimizer (000-1, addendum A06, C01 §7.2). Receipts arrive per grid date and the kernel cumulates them,
  so its input is the I8 evaluator's dated transfers placed on the grid. The business arrives as B_0, operating cash
  before the modeled loss, and the loss, so the kernel also computes the loss, payoff and terminal residual of the
  scenario table. Money is scaled by the power of ten at the leading digit of max |B_st − r_t|, the business's own cash,
  so a generous cap cannot widen the tolerance. ε_solver is absolute on the scaled primary objective, default 1e-9 (the
  oracle's). The premium-minimizing solve is held to ε/2, and the chosen portfolio's exactly recomputed CVaR must lie
  within ε × scale of the LP optimum, or the solve fails (HiGHS answers miss an exact ≤ ε by about 1e-16, as in the
  oracle's check 6). Deficits are reported exactly. A breach of a solve's portfolios is a deficit beyond ε × scale,
  because the premium tie-break may leave an otherwise fully funded portfolio that close to the floor. That tolerance is
  far below a cent for any business whose cash is under ten billion dollars; the coordinator's reasoning is that a
  sub-cent deficit is no breach of anything money can measure. Dominance with a dated premium means A_stj ≤ 0.
- Coverage (000-3, addendum A09 §10.2). World constraints may mention conditions that are not in the trigger: a state
  of the trigger's conditions is feasible when some assignment of the others satisfies every constraint, and every
  feasible state the coverage kernel shows carries that assignment as its certificate, checkable by evaluating the
  constraints (coordinator, settled in 000-3: refusing such constraints would leave real contracts unanalysable). A
  minimal group of coupled redundant conditions is a coverage obligation of its own, met by two feasible scenarios that
  agree outside the group and differ in outcome: none of its conditions has an independent effect, and without the
  group obligation a fully coupled trigger would be "covered" by a single scenario (A09 §10.2: "requiring a different
  justified coverage criterion"; worker's reasoning on 000-3). Enumeration gives way to python-sat above 10 conditions,
  where enumeration's self-evident proof still costs milliseconds.
- Computation (000-4, spec §5, addendum A11 §11.2). One graph of sources and instances, each carrying its §5 layer. A
  source is a value read from committed records and names at least one of them, so no constant written into code
  enters the graph; an instance is a node (id, version) over named inputs. The value hash is a typed canonical hash
  (values that print alike but differ in type differ; only immutable data hashes); the lineage hash is, for a source,
  its id, value hash and records' content hashes, and for an instance its node id and version and each input's name,
  value hash and lineage: a DerivationReceipt's fields, so every receipt is checkable. A judgment input's question
  version, state hash and configuration enter its consumer's key (spec §5). Receipts commit through
  Store.commit_derivations, one transaction per recompute, keyed by lineage; a committed lineage with another value is
  refused, the symptom of a node changed without a new version or an impure one. A diff lists an approval where it stops
  covering its vertex's lineage; explain() states any approval's standing. Approvals of graph values name the vertex id;
  vertex ids are written "v:<name>", so none collides with a proposal id. I7's static walk applies its purity rules
  (no provider, cassette, key, agent or store module; no network, process, database, clock, randomness, environment,
  file or dynamic import) to every product module a node's module reaches, since the walk sees modules, not calls.
- The exposure register's dated cash impact is monthly (core.DatedImpact; coordinator, November 1): deep discovery's
  materiality screens which exposures matter, and its drivers are stated per period. Intra-month liquidity timing is not
  the register's to judge: construction's optimizer takes the instrument's dated transfers (I8) and the business's cash
  on its own dated grid (A06 §6.1), where a payout late within a month shows as a breach.
- Reduced costs are diagnostics only. HiGHS's duals are not unique (in the addendum's own timing fixture a late payout
  and an early one both price at −110), so wherever the product shows a reduced cost it shows beside it the re-solve's
  measured improvement (C01 §7.2: a re-solve always runs).
- The oracle's recorded values (spec/verified_synthetic_results.json, local) are written into the tests that reproduce
  them, since spec/ is not in git; each transcription is checked against the JSON (the coordinator re-checked 000-1's:
  checks 5–9 match exactly).

## The factory

- The repository is public on GitHub, created by the coordinator, with no branch protection (Owen, November 1: "make it
  public and I don't need to set up branch protection. Create it yourself"). So CODEOWNERS binds nothing, and the
  coordinator merges factory pull requests itself; changes that loosen a check on its own work still go to Owen first,
  by asking, since nothing in GitHub enforces it.
- Since the repository is public, the source documents (`spec/`: the build spec, which carries the demo plan, and the
  addendum) stay out of git, local to each checkout, and are copied into each role's workspace. Secrets are never in git,
  logs, cassettes or fixtures (spec §14).
- Roles, as in Handoff's factory: the coordinator writes the definition (`requisite/core/`) and anything that holds the
  design; workers (Claude Opus 5.5, xhigh) take work packages that stand apart, each in a sandbox with no network beyond
  loopback; a verifier and a reviewer of the other family (GPT-6 Astra) judge outcomes and each staged change to the
  definition. A fresh reader restates each task before it goes.
- Tier-0 work lands on one branch, `outcome/tier0`, in the main checkout: its outcomes (000-007) are one connected demo,
  each built on the last, so separate branches would only add merges. It goes to main at Gate A, Gate B and Tier 0 green,
  each after the verifier has judged the outcomes it carries (coordinator's reasoning, November 1).

## Providers and keys

- Decision models (spec §6.2): OpenAI Decisions, Clef (Cloudflare Workers AI) and Jev (Typesafe, through OpenRouter),
  using the same keys as Slope and Handoff (Owen, November 1: "Use from same key as Slope/Handoff (cloudflare also for
  Clef) Requisite can use these"). The api.data.gov key is in this checkout's `.env` (GOV_API_KEY). Keys are read at run
  time from the environment, `.env`, Slope's `.env`, then the Keychain (`handoff.<NAME>`), and never printed.
- Spend caps are flexible for now (Owen, November 1); each run still records its calls and dollars (spec §6.6), and live
  calls happen only outside the workers' sandbox, by the coordinator, recorded as cassettes.
- One question per Decisions request, the facts as the state: Slope measured on October 29 that a question's situation
  placed in a batched question's instructions moved its answer (0.52 alone, 0.13 batched), while repeat asks alone were
  identical. Spec §6.2's "batch only mutually independent questions" does not prevent that.
- OpenAI's organization limit is 2,000,000 input tokens a minute, shared with Slope through the same key, so Requisite
  paces its Decisions requests at 1,000,000 a minute (estimated as request bytes over four) and retries a 429 with
  backoff (coordinator, November 1, from Slope's measured limit of October 29).
- Requisite's own agents (spec WP15: discovery, decomposer, clause drafter, scenario renderer, template proposer,
  post-mortem) run on Claude Opus 5.5 through the same OAuth subscription route as the coordinator's claude-cli
  extension, at the reasoning level the work warrants (Owen, November 1). Their outputs are Proposals, recorded, so the
  demo replays offline.
