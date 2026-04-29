"""Maintenance schedule utilities shared by all experiment scripts.

What this module provides
-------------------------
This module supports three maintenance modes, listed from most to least
constrained:

1. **Fixed schedule** (``DEFAULT_MAINTENANCE_SCHEDULE``)
   The exact day and start time of every maintenance window is predetermined.
   The same windows are used in every simulation replication.

2. **Constrained-random schedule** (``DEFAULT_MAINTENANCE_CANDIDATE_DAYS``)
   Each week's maintenance is allowed only on specific days, but the exact
   day within those candidates and the start time are drawn randomly per
   replication.  This is the middle ground between fully fixed and fully
   random.

3. **Fully random schedule** (pass ``--maintenance random`` or set both
   defaults to ``None``)
   Each week's maintenance falls on any of the 5 weekdays, chosen at random
   per replication.  This is the original behaviour.

Priority: if ``DEFAULT_MAINTENANCE_SCHEDULE`` is not None, it takes
precedence and ``DEFAULT_MAINTENANCE_CANDIDATE_DAYS`` is ignored.

Maintenance map format
----------------------
A maintenance map is a plain Python dict::

    { day_number: (start_minute, end_minute), ... }

- day_number   : 1-based weekday index (1 = Monday of Week 1, 5 = Friday,
                 6 = Monday of Week 2, etc.)
- start_minute : minutes from shift start when maintenance begins
                 (0 = 08:00 if shift starts at 08:00)
- end_minute   : minutes from shift start when maintenance ends

Candidate-days format
---------------------
A candidate-days dict maps week number (1-based) to a list of allowed
day-of-week positions within that week (1=Mon, 2=Tue, 3=Wed, 4=Thu, 5=Fri)::

    { week_number: [day_position, ...], ... }

Weeks not listed are unconstrained — maintenance may fall on any of the 5 days.

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
# Mode 1: Fixed maintenance schedule
# -----------------------------------------------------------------------
# Set each entry to the exact (start_minute, end_minute) for that day.
# Set to None to fall through to Mode 2 or Mode 3.
#
# Current plan:
#   Week 1  Wednesday  12:00–14:00   →  Day  3,  minutes 240–360
#   Week 2  Thursday   08:00–10:00   →  Day  9,  minutes   0–120
#   Week 3  Tuesday    10:00–12:00   →  Day 12,  minutes 120–240
#   Week 4  Wednesday  14:00–16:00   →  Day 18,  minutes 360–480

DEFAULT_MAINTENANCE_SCHEDULE: dict[int, tuple[float, float]] | None = None


# -----------------------------------------------------------------------
# Mode 2: Constrained-random maintenance schedule
# -----------------------------------------------------------------------
DEFAULT_MAINTENANCE_CANDIDATE_DAYS: dict[int, list[int]] | None = {
    1: [4, 5],       # Week 1: Thu or Fri
    2: [1, 2, 3],    # Week 2: Mon, Tue, or Wed
    3: [3, 4],       # Week 3: Wed or Thu
    4: [4, 5],       # Week 4: Thu or Fri
}

# -----------------------------------------------------------------------
# Unscheduled weeks policy
# -----------------------------------------------------------------------
# Controls what happens to weeks not listed in DEFAULT_MAINTENANCE_CANDIDATE_DAYS.
#
#   'random'  — schedule maintenance on a randomly chosen day (original behaviour)
#   'skip'    — no maintenance window for that week
#
# Only applies when using constrained-random mode (candidate_days is set).
# Has no effect when using a fixed schedule or fully random mode.

DEFAULT_UNSCHEDULED_WEEKS_POLICY: str = 'skip'


# -----------------------------------------------------------------------
# Parsing helpers
# -----------------------------------------------------------------------

def parse_maintenance_schedule(text: str) -> dict[int, tuple[float, float]]:
    """Parse a CLI fixed-schedule string into a maintenance_map dict.

    Format
    ------
    ``"day:start_min:end_min,day:start_min:end_min,..."``

    Example
    -------
    Week 1 Wednesday 12:00-14:00 and Week 2 Thursday 08:00-10:00::

        "3:240:360,9:0:120"

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


def parse_candidate_days(text: str) -> dict[int, list[int]]:
    """Parse a CLI candidate-days string into a candidate_days dict.

    Format
    ------
    ``"week:day,day,...;week:day,day,..."``

    Each semicolon-separated group is one week.  Within each group, the
    week number comes first, followed by a colon, then a comma-separated
    list of allowed day-of-week positions (1=Mon … 5=Fri).

    Example
    -------
    Week 1 only Thu/Fri, Week 2 only Mon/Tue/Wed::

        "1:4,5;2:1,2,3"

    Raises
    ------
    ValueError
        If any entry is malformed or contains out-of-range day positions.
    """
    result: dict[int, list[int]] = {}
    for group in text.split(';'):
        group = group.strip()
        if not group:
            continue
        if ':' not in group:
            raise ValueError(
                f"Invalid candidate-days group '{group}'. "
                "Expected format: week:day,day,...  e.g. '1:4,5'"
            )
        week_str, days_str = group.split(':', 1)
        week = int(week_str.strip())
        positions = [int(d.strip()) for d in days_str.split(',') if d.strip()]
        invalid = [p for p in positions if not 1 <= p <= 5]
        if invalid:
            raise ValueError(
                f"Week {week}: day positions {invalid} are out of range. "
                "Use 1=Mon, 2=Tue, 3=Wed, 4=Thu, 5=Fri."
            )
        result[week] = positions
    return result


def resolve_maintenance_map(
    maintenance_arg: str | None,
    unscheduled_override: str | None = None,
) -> tuple[dict | None, dict | None, str]:
    """Resolve CLI arguments into (maintenance_map, candidate_days, unscheduled_weeks_policy).

    Parameters
    ----------
    maintenance_arg :
        Value of the ``--maintenance`` CLI flag (or None if not passed).
    unscheduled_override :
        Value of the ``--unscheduled`` CLI flag (``'random'``, ``'skip'``,
        or None).  When provided, this overrides ``DEFAULT_UNSCHEDULED_WEEKS_POLICY``.
        Only meaningful when using constrained-random mode.

    Priority logic for maintenance_arg
    ------------------------------------
    - ``None``         → use DEFAULT_MAINTENANCE_SCHEDULE if set, otherwise
                         use DEFAULT_MAINTENANCE_CANDIDATE_DAYS
    - ``'random'``     → fully random; returns (None, None, 'random')
    - ``'candidates'`` → use DEFAULT_MAINTENANCE_CANDIDATE_DAYS
    - ``'day:s:e,...'``→ fixed schedule (no semicolons)
    - ``'week:d,...;'``→ constrained-random (contains semicolon)

    Returns
    -------
    (maintenance_map, candidate_days, unscheduled_weeks_policy)
    """
    # Determine the effective unscheduled policy: CLI flag takes priority.
    base_policy = unscheduled_override if unscheduled_override is not None \
        else DEFAULT_UNSCHEDULED_WEEKS_POLICY

    if maintenance_arg is None:
        if DEFAULT_MAINTENANCE_SCHEDULE is not None:
            return DEFAULT_MAINTENANCE_SCHEDULE, None, 'random'
        return None, DEFAULT_MAINTENANCE_CANDIDATE_DAYS, base_policy

    low = maintenance_arg.lower().strip()

    if low == 'random':
        return None, None, 'random'

    if low == 'candidates':
        return None, DEFAULT_MAINTENANCE_CANDIDATE_DAYS, base_policy

    if ';' in maintenance_arg:
        # Semicolons always indicate candidate-days format.
        return None, parse_candidate_days(maintenance_arg), base_policy

    # Distinguish single-entry candidate-days ("1:3,4,5") from fixed schedule
    # ("3:240:360") by counting colons in the first entry.
    # candidate-days entry: "week:day,day,..."  → exactly 1 colon
    # fixed schedule entry: "day:start:end"     → exactly 2 colons
    first_entry = maintenance_arg.split(',')[0].strip()
    if first_entry.count(':') == 1:
        # Single-week candidate-days without a trailing semicolon.
        return None, parse_candidate_days(maintenance_arg), base_policy

    return parse_maintenance_schedule(maintenance_arg), None, 'random'

