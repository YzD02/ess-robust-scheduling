"""Maintenance schedule utilities shared by all experiment scripts.

What this module provides
-------------------------
1. DEFAULT_MAINTENANCE_SCHEDULE — a fixed, deterministic maintenance plan
   that both run_single_case.py and run_grid_search.py use by default.
   Edit this dict to change the maintenance plan project-wide.

2. parse_maintenance_schedule()  — parses a CLI string into the same dict
   format, so users can override the default from the command line.

Maintenance map format
----------------------
A maintenance map is a plain Python dict:

    { day_number: (start_minute, end_minute), ... }

- day_number   : 1-based weekday index (1 = Monday of Week 1, 5 = Friday,
                 6 = Monday of Week 2, etc.)
- start_minute : minutes from shift start when maintenance begins
                 (0 = 08:00 if shift starts at 08:00)
- end_minute   : minutes from shift start when maintenance ends

Time conversion (assuming 08:00 shift start):

    08:00 – 10:00  →  (0,   120)
    10:00 – 12:00  →  (120, 240)
    12:00 – 14:00  →  (240, 360)
    14:00 – 16:00  →  (360, 480)

Day number reference (20-day horizon):

    Week 1 : Mon=1  Tue=2  Wed=3  Thu=4  Fri=5
    Week 2 : Mon=6  Tue=7  Wed=8  Thu=9  Fri=10
    Week 3 : Mon=11 Tue=12 Wed=13 Thu=14 Fri=15
    Week 4 : Mon=16 Tue=17 Wed=18 Thu=19 Fri=20
"""

from __future__ import annotations


# -----------------------------------------------------------------------
# Default fixed maintenance schedule
# -----------------------------------------------------------------------
# Edit the entries below to change the maintenance plan for all experiments.
# Set to None to use random maintenance windows in every run.
#
# Current plan:
#   Week 1  Wednesday  12:00–14:00   →  Day  3,  minutes 240–360
#   Week 2  Thursday   08:00–10:00   →  Day  9,  minutes   0–120
#   Week 3  Tuesday    10:00–12:00   →  Day 12,  minutes 120–240
#   Week 4  Wednesday  14:00–16:00   →  Day 18,  minutes 360–480

DEFAULT_MAINTENANCE_SCHEDULE: dict[int, tuple[float, float]] | None = {
    3:  (240, 360),   # Week 1 Wednesday  12:00–14:00
    9:  (0,   120),   # Week 2 Thursday   08:00–10:00
    12: (120, 240),   # Week 3 Tuesday    10:00–12:00
    18: (360, 480),   # Week 4 Wednesday  14:00–16:00
}


def parse_maintenance_schedule(text: str) -> dict[int, tuple[float, float]]:
    """Parse a CLI maintenance schedule string into a maintenance_map dict.

    Format
    ------
    "day:start_min:end_min,day:start_min:end_min,..."

    Examples
    --------
    Week 1 Wednesday 12:00-14:00 and Week 2 Thursday 08:00-10:00::

        "3:240:360,9:0:120"

    Parameters
    ----------
    text :
        Comma-separated entries, each in ``day:start_minute:end_minute`` form.

    Returns
    -------
    dict mapping day number → (start_minute, end_minute).

    Raises
    ------
    ValueError
        If any entry is malformed or has start >= end.
    """
    result: dict[int, tuple[float, float]] = {}
    for entry in text.split(','):
        entry = entry.strip()
        if not entry:
            continue
        parts = entry.split(':')
        if len(parts) != 3:
            raise ValueError(
                f"Invalid maintenance entry '{entry}'. "
                "Expected format: day:start_minute:end_minute  e.g. '3:240:360'"
            )
        day, start, end = int(parts[0]), float(parts[1]), float(parts[2])
        if start >= end:
            raise ValueError(
                f"Entry '{entry}': start_minute ({start}) must be less than end_minute ({end})."
            )
        result[day] = (start, end)
    return result


def resolve_maintenance_map(maintenance_arg: str | None) -> dict | None:
    """Resolve the --maintenance CLI argument into a maintenance_map.

    This function centralises the three-way decision logic used by both
    run_single_case.py and run_grid_search.py:

    - ``None``     → use DEFAULT_MAINTENANCE_SCHEDULE (defined above)
    - ``'random'`` → return None, triggering random window generation
    - any string   → parse as 'day:start:end,...' and return the dict

    Parameters
    ----------
    maintenance_arg :
        The raw string value of the --maintenance CLI argument, or None
        if the flag was not passed.

    Returns
    -------
    A maintenance_map dict, or None to signal random generation.
    """
    if maintenance_arg is None:
        return DEFAULT_MAINTENANCE_SCHEDULE
    if maintenance_arg.lower() == 'random':
        return None
    return parse_maintenance_schedule(maintenance_arg)
