"""Helpers for loading a warm-start schedule saved by run_grid_search.py's
--save-schedule-dir feature.

JSON forces all dict keys to strings, so day numbers and job ids must be
cast back to int when loading. This module centralizes that conversion so
callers (run_grid_search.py, notebooks, ad-hoc scripts) don't each
reimplement it slightly differently.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


def load_warm_start_schedule(path: str | Path) -> Dict[int, List[int]]:
    """Load a sorted_schedule dict from a schedule JSON file.

    Parameters
    ----------
    path :
        Path to a schedule_n{n}_k{k}.json file produced by run_grid_search.py
        when called with --save-schedule-dir.

    Returns
    -------
    dict mapping day (int) -> list of job ids (int), suitable for passing
    directly as `warm_start_schedule` to solve_gurobi_baseline.

    Raises
    ------
    FileNotFoundError if the path does not exist.
    KeyError if the file does not contain a 'sorted_schedule' field (e.g. it
    is an older-format file saved before this feature existed).
    """
    path = Path(path)
    with open(path, 'r') as f:
        payload = json.load(f)

    if 'sorted_schedule' not in payload:
        raise KeyError(
            f"{path} does not contain a 'sorted_schedule' field. "
            "This file may predate the --save-schedule-dir feature, or be "
            "a different kind of output file."
        )

    raw_schedule = payload['sorted_schedule']
    # JSON keys are always strings -- cast day and job ids back to int.
    return {
        int(day_str): [int(j) for j in job_list]
        for day_str, job_list in raw_schedule.items()
    }
