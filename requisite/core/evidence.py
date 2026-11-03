"""The evidence layer: artifacts and the claims quoted from them (spec §5 evidence layer; invariants I2, I3, I4).

Observed and synthetic evidence are different types with different identifiers, so a synthetic claim cannot stand where
an observed one is required (spec I4: "enforce this at the type level").
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, Literal

from pydantic import StringConstraints, model_validator

from requisite.core.base import Instant, Record, sha256

HASH = r"[0-9a-f]{64}"

# Identifiers carry their namespace: observed evidence is art_/clm_, synthetic evidence is syn_art_/syn_clm_ (spec I4).
ArtifactId = Annotated[str, StringConstraints(pattern=rf"^art_{HASH}$")]
ClaimId = Annotated[str, StringConstraints(pattern=rf"^clm_{HASH}$")]
SyntheticArtifactId = Annotated[str, StringConstraints(pattern=rf"^syn_art_{HASH}$")]
SyntheticClaimId = Annotated[str, StringConstraints(pattern=rf"^syn_clm_{HASH}$")]


def artifact_id(content: bytes) -> str:
    """Content addressing (spec I2): an artifact's identifier is the hash of its bytes, so the same id is the same
    content, and a changed source is a new artifact."""
    return f"art_{sha256(content)}"


def synthetic_artifact_id(content: bytes) -> str:
    return f"syn_art_{sha256(content)}"


class Artifact(Record):
    """One retrieved source document, immutable (spec I2). Its bytes live in the blob store under its id.

    locator: where it was retrieved from (a URL or an API path), for retrieval and citation.
    source: the publisher or system of record (e.g. "regulations.gov", "Federal Register").
    published_at: when the source says it was published, with `published_at_basis` saying where that date comes from;
        None where the source states none (spec I3).
    observed_at: when the runtime retrieved it (spec I3: observed claims need observed_at <= the evaluation time).
    availability: how its availability at a past instant is evidenced (addendum A11 §11.3: "A page's old publication date
        does not establish that the currently fetched text was available then"): "retrieved" (only from observed_at) or
        "archived" (an archive snapshot, in `archive_locator`, evidences the text from `archived_at`).
    predecessor: the artifact this one replaces when the same locator returned changed content (spec I2: "A changed
        source is a new version linked to its predecessor").
    """

    id: ArtifactId
    namespace: Literal["observed"] = "observed"
    locator: str
    source: str
    media_type: str
    size: int
    published_at: Instant | None
    published_at_basis: str
    observed_at: Instant
    availability: Literal["retrieved", "archived"]
    archive_locator: str | None = None
    archived_at: Instant | None = None
    predecessor: ArtifactId | None = None

    @model_validator(mode="after")
    def _archive(self) -> Artifact:
        archived = self.archive_locator is not None and self.archived_at is not None
        if (self.availability == "archived") != archived:
            raise ValueError("an archived artifact names its archive snapshot and its time; a retrieved one names none")
        return self

    @property
    def available_from(self) -> Instant:
        """The earliest instant at which this content is evidenced to have existed, the bound I3's as_of check uses
        beside published_at (addendum §11.3; ours: without an archive, only the retrieval itself evidences the text)."""
        if self.availability == "archived" and self.archived_at is not None:
            return self.archived_at
        return self.observed_at


class Claim(Record):
    """A passage quoted from an observed artifact (spec §5: Supports, with a span). Agents propose claims (spec §4
    Proposal kind "claim"); code admits one only if `text` is exactly the artifact's text at `span`.

    span: [start, end) character offsets into the artifact's extracted text, whose extraction is versioned by
        `extraction` (ours: offsets are only meaningful against one extraction of the bytes).
    authority: who asserts the passage (an agency, a commenter, a court).
    """

    id: ClaimId
    namespace: Literal["observed"] = "observed"
    artifact: ArtifactId
    extraction: str
    span: tuple[int, int]
    text: str
    authority: str

    @model_validator(mode="after")
    def _span(self) -> Claim:
        start, end = self.span
        if not 0 <= start < end:
            raise ValueError(f"a span is [start, end) with 0 <= start < end, not {self.span}")
        if end - start != len(self.text):
            raise ValueError("a claim's text is exactly its span of the artifact")
        if self.id != claim_id(self.artifact, self.extraction, self.span, self.text, self.authority):
            raise ValueError("a claim's id is the hash of its artifact, extraction, span, text and authority")
        return self


def claim_id(artifact: str, extraction: str, span: tuple[int, int], text: str, authority: str) -> str:
    """A claim is an assertion: these words, at this span of this content, asserted by this authority. Identical bytes
    filed by two submitters share an artifact (spec I2: content addressing) and are two claims (external evaluation,
    November 2: content identity had stood for assertion identity, and the second filer's claim collided with the
    first's)."""
    return f"clm_{sha256(f'{artifact}|{extraction}|{span[0]}:{span[1]}|{text}|{authority}')}"


class SyntheticArtifact(Record):
    """A generated document: a fuzzing scenario's rendered packet or a simulation (spec I4). It lives in its own
    namespace and records what generated it. published_at is the date a generated document states it was published, a
    fact of its scenario's fiction; None for a generated record that is no document and states no date (a fuzz
    scenario's own definition). Recording it is ours: the rendering check holds each packet document to its scenario's
    evidence instant (core.ScenarioPacket). created_at is when it was generated."""

    id: SyntheticArtifactId
    namespace: Literal["synthetic"] = "synthetic"
    scenario: str
    generator: str
    media_type: str
    size: int
    published_at: Instant | None
    created_at: Instant


class SyntheticClaim(Record):
    """A passage of a synthetic artifact. It may stand as observed only in a synthetic case (spec I4)."""

    id: SyntheticClaimId
    namespace: Literal["synthetic"] = "synthetic"
    artifact: SyntheticArtifactId
    span: tuple[int, int]
    text: str
    authority: str

    @model_validator(mode="after")
    def _span(self) -> SyntheticClaim:
        start, end = self.span
        if not 0 <= start < end or end - start != len(self.text):
            raise ValueError("a synthetic claim's text is exactly its span")
        if self.id != f"syn_clm_{sha256(f'{self.artifact}|{start}:{end}|{self.text}')}":
            raise ValueError("a synthetic claim's id is the hash of its artifact, span and text")
        return self


def observable_at(artifact: Artifact, as_of: Instant, evaluated_at: Instant, mode: str = "prospective") -> bool:
    """Whether a claim from this artifact may stand as observed in a state as of `as_of`, judged at `evaluated_at`.

    Spec I3: "Observed claims come only from artifacts with published_at <= as_of and observed_at <= the evaluation
    time." An artifact whose source states no publication date counts from when its text is evidenced to exist. And a
    publication date does not establish that the text as fetched was available then (addendum A11 §11.3), so the text
    must be evidenced to exist by as_of (`available_from`: its retrieval, or an archive snapshot) unless the state is a
    retrospective reconstruction, which says so of itself and is never a clean prospective or replayed evaluation.
    The one place this rule is written: everything that builds a JudgmentState selects evidence through it."""
    published = artifact.published_at if artifact.published_at is not None else artifact.available_from
    if not (published <= as_of and artifact.observed_at <= evaluated_at):
        return False
    return mode == "retrospective_reconstruction" or artifact.available_from <= as_of


class ItemPart(Record):
    """One part of what a submitter filed, in the order filed: the filing's own record, then each file. Its reading status
    is stated part by part, so what could not be read is a fact about that part, never inferred from text (external
    evaluation, November 2: an unreadable attachment recorded by its label made the whole item read as unreadable).

    locator: where this filing's part was retrieved from, None where nothing was retrieved. Two filings of identical
    bytes share one artifact (spec I2), so where each occurred is recorded here, on the filing. artifact: the content
    retrieved, None where nothing was (a restricted file, one the record names but does not include). extraction: the
    extraction whose text is this part's, None where no text could be read. unreadable: why some or all of it could not
    be read (a part read with some pages unreadable has both), None where it was read whole. by: whose fact that is:
    "source" where the source published nothing readable (restricted, no longer served, empty, a file whose own bytes
    no reader can read), "reader" where the file is published in a format no reader here reads, which is work not done
    rather than a fact about the filing (AGENTS.md; posture: an unknown caused by work we did not do is our defect)."""

    label: str
    locator: str | None
    artifact: ArtifactId | None
    extraction: str | None
    unreadable: str | None
    by: Literal["source", "reader"] | None

    @model_validator(mode="after")
    def _status(self) -> ItemPart:
        if self.extraction is None and self.unreadable is None:
            raise ValueError(f"{self.label}: a part is read, or says why it could not be")
        if (self.unreadable is None) != (self.by is None):
            raise ValueError(f"{self.label}: what could not be read says whose fact that is, and only it does")
        if self.extraction is not None and self.artifact is None:
            raise ValueError(f"{self.label}: text is read from retrieved content")
        if (self.artifact is None) != (self.locator is None):
            raise ValueError(f"{self.label}: retrieved content has the locator it was retrieved from, and only it does")
        return self


class Item(Record):
    """One unit of a wide corpus (spec §7.2: "split it into items: comments, letters, filings, posts"): what one
    submitter filed, read whole (decisions: an item's text is everything filed, scans read by OCR).

    container: the docket or collection it was filed in. addresses: the document the source states the filing comments
    on (regulations.gov's commentOnDocumentId), None where the source states none; the measure an item is asked about is
    the one it addresses, never inferred from its docket (external evaluation, November 2: a docket can hold several
    documents and rulemakings). submitter: the name or organization the submitter gave, as the source publishes it, or
    None where none is given. parts: the filing's record and each file, in the order filed, each with its reading status.
    text: every part, each headed by its label and artifact, read text where there is some and the reason where there is
    not; what could not be read is stated rather than dropped.
    """

    id: str
    source: str
    container: str
    addresses: str | None
    posted_at: Instant
    submitter: str | None
    parts: Sequence[ItemPart]
    text: str

    @model_validator(mode="after")
    def _whole(self) -> Item:
        if not self.parts or self.parts[0].artifact is None:
            raise ValueError("an item is read from at least its own record")
        return self

    @property
    def artifacts(self) -> list[str]:
        """The content retrieved for its parts, in order, each once."""
        return list(dict.fromkeys(p.artifact for p in self.parts if p.artifact is not None))

    @property
    def extractions(self) -> list[str]:
        return [p.extraction for p in self.parts if p.extraction is not None]

    @property
    def unreadable(self) -> list[str]:
        """Each part that could not be read whole, with why."""
        return [f"{p.label}: {p.unreadable}" for p in self.parts if p.unreadable is not None]

    @property
    def unread_by_us(self) -> list[str]:
        """Each part published in a format no reader here reads: to be read, never set aside."""
        return [f"{p.label}: {p.unreadable}" for p in self.parts if p.by == "reader"]

    @property
    def readable(self) -> bool:
        """Whether any part's text was read (an empty text is recorded as unreadable when the item is read)."""
        return any(p.extraction is not None for p in self.parts)
