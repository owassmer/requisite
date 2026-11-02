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
        if self.id != claim_id(self.artifact, self.extraction, self.span, self.text):
            raise ValueError("a claim's id is the hash of its artifact, extraction, span and text")
        return self


def claim_id(artifact: str, extraction: str, span: tuple[int, int], text: str) -> str:
    return f"clm_{sha256(f'{artifact}|{extraction}|{span[0]}:{span[1]}|{text}')}"


class SyntheticArtifact(Record):
    """A generated document: a fuzzing scenario's rendered packet or a simulation (spec I4). It lives in its own
    namespace and records what generated it."""

    id: SyntheticArtifactId
    namespace: Literal["synthetic"] = "synthetic"
    scenario: str
    generator: str
    media_type: str
    size: int
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


class Item(Record):
    """One unit of a wide corpus (spec §7.2: "split it into items: comments, letters, filings, posts"): what one
    submitter filed, read whole (decisions: an item's text is everything filed, scans read by OCR).

    container: the docket or collection it was filed in. submitter: the name or organization the submitter gave, as the
    source publishes it, or None where none is given. artifacts: the filing's record and each attachment, in order.
    text: the comment and each attachment's extracted text, in that order, each part headed by the artifact it came from.
    extractions: the extraction ids that produced the text. unreadable: any attachment whose text could not be read, with
    the reason; the item still stands, and what could not be read is stated rather than dropped.
    """

    id: str
    source: str
    container: str
    posted_at: Instant
    submitter: str | None
    artifacts: Sequence[ArtifactId]
    text: str
    extractions: Sequence[str]
    unreadable: Sequence[str] = ()

    @model_validator(mode="after")
    def _whole(self) -> Item:
        if not self.artifacts:
            raise ValueError("an item is read from at least its own record")
        return self
