"""Shared memory measurement utility using tracemalloc."""

from __future__ import annotations

import tracemalloc
from contextlib import contextmanager


@contextmanager
def track_memory():
    """Context manager that yields a dict; on exit, stores peak memory in bytes."""
    result = {"peak_bytes": 0}
    tracemalloc.start()
    try:
        yield result
    finally:
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        result["peak_bytes"] = peak
