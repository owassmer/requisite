"""Retrieved responses into committed artifacts (spec I2, I3; addendum A11 §11.3).

An artifact is the bytes exactly as the source returned them, so its id is the hash of those bytes: the same bytes
retrieved twice are the same artifact, recorded once with its first retrieval, and a locator that returns other bytes
(any field changed, metadata such as lastModifiedDate included, since the source did change) gives a new artifact linked
to the one it replaces (coordinator's decision for task 002-1). What derives from an artifact's text compares by value,
so an unchanged comment changes nothing downstream but its lineage (addendum A11 §11.2).

published_at is what the source states (a comment's postedDate, a market's settlement time), with its basis named;
observed_at is the retrieval, as the cassette recorded it, so a replay commits the same artifact; availability is
"retrieved": only the retrieval evidences the text (addendum A11 §11.3). The source is the one the response came from.
"""

from __future__ import annotations

from datetime import datetime

from requisite.core import Artifact
from requisite.core.evidence import artifact_id
from requisite.evidence.fetched import Fetched
from requisite.store import Store

LISTING_BASIS = "none stated: a listing response has no publication date of its own; its retrieval evidences it"


def describe(fetched: Fetched, *, published_at: datetime | None, basis: str, predecessor: str | None = None) -> Artifact:
    """The artifact these bytes are, as a record (not committed): what `commit` commits, and what a reading that has not
    committed its responses (a draw printed, not kept) judges with core.observable_at."""
    return Artifact(
        id=artifact_id(fetched.content),
        locator=fetched.locator,
        source=fetched.source,
        media_type=fetched.media_type,
        size=len(fetched.content),
        published_at=published_at,
        published_at_basis=basis,
        observed_at=fetched.retrieved_at,
        availability="retrieved",
        predecessor=predecessor,
    )


def commit(store: Store, fetched: Fetched, *, published_at: datetime | None, basis: str) -> Artifact:
    """The artifact these bytes are, committed once: the one already committed where the bytes are known, else a new one
    linked to the latest other version retrieved from the same locator."""
    aid = artifact_id(fetched.content)
    known = store.get("artifact", aid, Artifact)
    if known is not None:
        return known
    before = [a for a in store.artifacts_at(fetched.locator) if a.id != aid]
    artifact = describe(fetched, published_at=published_at, basis=basis, predecessor=before[-1].id if before else None)
    store.commit_artifact(artifact, fetched.content)
    return artifact


def listed(listing_page: Artifact, posted: datetime, basis: str) -> Artifact:
    """One entry of a listing as evidence: the listing page's retrieval, with the entry's own date (a comment's
    postedDate, a market's settlement time) as its publication, so whether the entry was public by an instant is decided
    by core.observable_at like any artifact's."""
    return Artifact.model_validate({**listing_page.model_dump(), "published_at": posted, "published_at_basis": basis})
