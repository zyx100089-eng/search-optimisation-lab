from __future__ import annotations

import pandas as pd


def result_to_dict(name: str, result) -> dict:
    return {
        "Algorithm": name,
        "Path Length": result.path_length,
        "Path Cost": round(result.path_cost, 2),
        "Nodes Explored": result.explored_count,
        "Runtime (ms)": round(result.runtime * 1000, 3),
        "Found Path": result.path_length > 0,
    }


def compare_results(results: dict[str, object]) -> pd.DataFrame:
    rows = [result_to_dict(name, res) for name, res in results.items()]
    return pd.DataFrame(rows)
