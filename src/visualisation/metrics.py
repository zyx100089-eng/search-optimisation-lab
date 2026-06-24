from __future__ import annotations

import pandas as pd


def result_to_dict(name: str, result) -> dict:
    d = {
        "Algorithm": name,
        "Path Length": result.path_length,
        "Path Cost": round(result.path_cost, 2),
        "Nodes Explored": result.explored_count,
        "Runtime (ms)": round(result.runtime * 1000, 3),
        "Found Path": result.path_length > 0,
    }
    if hasattr(result, "peak_memory_bytes"):
        d["Peak Memory (KB)"] = round(result.peak_memory_bytes / 1024, 1)
    return d


def compare_results(results: dict[str, object]) -> pd.DataFrame:
    rows = [result_to_dict(name, res) for name, res in results.items()]
    return pd.DataFrame(rows)
