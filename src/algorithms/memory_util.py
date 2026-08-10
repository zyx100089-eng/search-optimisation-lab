"""Shared memory measurement utility using tracemalloc."""

from __future__ import annotations

import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass
class _MemTracker:
    peak_bytes: int = 0


@contextmanager
def track_memory():
    """Context manager that yields a mutable tracker object.

    On exit the tracker's ``peak_bytes`` attribute is populated with the peak
    memory (in bytes) traced during the block.  Read it *after* the ``with``
    body completes::

        with track_memory() as mem:
            do_search()
        # mem.peak_bytes is now populated
        return Result(..., peak_memory_bytes=mem.peak_bytes)
    """
    tracker = _MemTracker()
    tracemalloc.start()
    try:
        yield tracker
    finally:
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        tracker.peak_bytes = peak