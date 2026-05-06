"""Base class defining the adapter contract.

An adapter has one job: read a source dataset's native format and yield
`EditTriplet` instances. Adapters are pure I/O — no filtering, no
computation, no downstream annotation. Filtering decisions belong in
later pipeline stages where they can be logged and reversed.

The `ingest` method is a generator, not a list-returning function.
This is intentional: datasets can be millions of entries, and eager
materialization would blow memory. Callers who want a list can always
call `list(adapter.ingest())`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator

from edit2forensics.data.triplet import EditTriplet


class BaseAdapter(ABC):
    """Abstract base for all dataset adapters."""

    #: Human-readable dataset identifier used in `EditTriplet.source_dataset`.
    #: Subclasses MUST override this as a class attribute.
    source_dataset: str = ""

    @abstractmethod
    def ingest(self) -> Iterator[EditTriplet]:
        """Yield `EditTriplet` records one at a time.

        Implementations should:
        - Skip malformed records (logging a warning) rather than raising.
        - Ensure `triplet_id` is stable across runs (don't use `uuid4`).
        - Resolve all paths to absolute form before yielding.
        - Never load image bytes — only paths.
        """
        raise NotImplementedError

    def __post_init_check__(self) -> None:
        """Optional sanity check subclasses can call after construction."""
        if not self.source_dataset:
            raise ValueError(
                f"{type(self).__name__} must set class attribute `source_dataset`"
            )
