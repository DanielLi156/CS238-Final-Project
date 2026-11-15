"""
CSL rules (Season 1, Competition Format v7.1) encoded as data, plus:
  - score_race        : scores one race (place points, ties, jackpots, DSQ/DNS)
  - jackpot_threshold : jackpot margin for an event

Section numbers in comments refer to the CSL rules document.
The simulator should only ever read rules from this file.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional



# ---------------------------------------------------------------------------
# Constants (Sections 4, 5.2, 5.6)
# ---------------------------------------------------------------------------

TEAMS_PER_MATCH = 4
ROSTER_PER_GENDER = 14
ENTRIES_PER_TEAM = 2          # 2 swimmers (or 2 relay teams) per team per event -> 8 lanes
MAX_INDIVIDUAL_EVENTS = 3     # per swimmer
MAX_RELAY_EVENTS = 2          # per swimmer; Super Skins counts as a relay, Skins counts as nothing
RELAY_LEGS = 4
MEDLEY_ORDER = ("back", "breast", "fly", "free")   # Super Skins leg order (5.8.5)


class Gender(Enum):
    WOMEN = "W"
    MEN = "M"


class Kind(Enum):
    INDIVIDUAL = "individual"
    FREE_RELAY = "4x50 free relay"
    SKINS = "skins"               # individual elimination, 4 rounds of 50s
    SUPER_SKINS = "super skins"   # 200 medley relay elimination, 4 rounds


# ---------------------------------------------------------------------------
# Events (Section 5.7) - 30 single-gender events, women first in each pair
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Event:
    number: int
    gender: Gender
    name: str
    kind: Kind

    def __str__(self) -> str:
        return f"#{self.number} {self.gender.value} {self.name}"


_EVENT_ORDER = [
    ("200 Back", Kind.INDIVIDUAL),
    ("200 Breast", Kind.INDIVIDUAL),
    ("500 Free", Kind.INDIVIDUAL),
    ("4x50 Free Relay", Kind.FREE_RELAY),
    ("200 Fly", Kind.INDIVIDUAL),
    ("100 IM", Kind.INDIVIDUAL),
    ("100 Back", Kind.INDIVIDUAL),
    ("100 Free", Kind.INDIVIDUAL),
    ("Skins", Kind.SKINS),
    ("200 IM", Kind.INDIVIDUAL),
    ("100 Breast", Kind.INDIVIDUAL),
    ("50 Free", Kind.INDIVIDUAL),
    ("100 Fly", Kind.INDIVIDUAL),
    ("200 Free", Kind.INDIVIDUAL),
    ("Super Skins", Kind.SUPER_SKINS),
]

EVENTS: list[Event] = []
for _i, (_name, _kind) in enumerate(_EVENT_ORDER):
    EVENTS.append(Event(2 * _i + 1, Gender.WOMEN, _name, _kind))
    EVENTS.append(Event(2 * _i + 2, Gender.MEN, _name, _kind))

EVENT_BY_NUMBER: dict[int, Event] = {e.number: e for e in EVENTS}


# ---------------------------------------------------------------------------
# Points (Sections 6.1, 6.2, 5.8.11)
# ---------------------------------------------------------------------------

INDIVIDUAL_POINTS = [9, 7, 6, 5, 4, 3, 2, 1]
RELAY_POINTS = [18, 14, 12, 10, 8, 6, 4, 2]

# Skins / Super Skins: (places paid, number advancing) per round.
# Round 1 pays 8, Round 2 pays 6, Round 3 pays 4, Final pays 2.
ELIMINATION_ROUNDS = [(8, 6), (6, 4), (4, 2), (2, 1)]


# ---------------------------------------------------------------------------
# Jackpot thresholds in seconds, yards (Section 6.3) : (men, women)
# ---------------------------------------------------------------------------

JACKPOT = {
    "50 Free": (1.08, 1.22),
    "100 Free": (2.24, 2.50),
    "200 Free": (4.93, 5.39),
    "500 Free": (15.37, 17.03),
    "100 Back": (3.07, 3.47),
    "200 Back": (6.09, 6.76),
    "100 Breast": (4.04, 4.56),
    "200 Breast": (8.59, 9.88),
    "100 Fly": (2.86, 3.29),
    "200 Fly": (7.19, 8.05),
    "100 IM": (3.63, 4.14),
    "200 IM": (7.59, 8.57),
    "4x50 Free Relay": (5.45, 6.23),
    "Super Skins": (5.92, 6.73),        # 200 medley relay threshold, every round
}

SKINS_JACKPOT = {                        # 50-yard rounds, by selected stroke
    "free": (1.08, 1.22),
    "back": (1.43, 1.62),
    "breast": (1.87, 2.11),
    "fly": (1.29, 1.49),
}
SKINS_DEFAULT_STROKE = "fly"             # if the relay winner doesn't pick in 30 s (5.8.3)


def jackpot_threshold(event: Event, skins_stroke: Optional[str] = None) -> float:
    idx = 0 if event.gender == Gender.MEN else 1
    if event.kind == Kind.SKINS:
        return SKINS_JACKPOT[skins_stroke or SKINS_DEFAULT_STROKE][idx]
    return JACKPOT[event.name][idx]


# ---------------------------------------------------------------------------
# Scoring one race (Sections 6.1-6.3, 6.6, 6.9)
# ---------------------------------------------------------------------------

@dataclass
class RaceEntry:
    team: str
    who: object                  # athlete name, or tuple of 4 names for a relay
    time: Optional[float]        # None = DNS / DNF / DSQ


def score_race(entries: list[RaceEntry], table: list[int], threshold: float) -> list[float]:
    """
    Returns points for each entry, aligned with `entries`.

    - Places are by time (rounded to 0.01 s); ties share the average of the tied places.
    - Places beyond len(table) score 0 (used for later Skins rounds).
    - Jackpot: if the winner beats a slower finisher from ANOTHER team by >= threshold,
      the winner takes that finisher's points and the slower finisher gets 0.
    - DNS/DNF/DSQ entries take the lowest places and are automatically jackpotted to
      the winner; if the winner is from the same team, nobody gets those points.
    - Assumption for a rare edge case: if several entries tie for 1st, jackpotted
      points are split equally among the tied winners from other teams.
    """
    n = len(entries)
    pts = [0.0] * n
    alloc = lambda place: table[place] if place < len(table) else 0
    t = [None if e.time is None else round(e.time, 2) for e in entries]

    finishers = sorted((i for i in range(n) if t[i] is not None), key=lambda i: t[i])
    penalized = [i for i in range(n) if t[i] is None]
    if not finishers:
        return pts

    # Place points with tie averaging
    groups, pos = [], 0
    while pos < len(finishers):
        end = pos
        while end + 1 < len(finishers) and t[finishers[end + 1]] == t[finishers[pos]]:
            end += 1
        group = finishers[pos:end + 1]
        share = sum(alloc(p) for p in range(pos, end + 1)) / len(group)
        for i in group:
            pts[i] = share
        groups.append(group)
        pos = end + 1

    winners, win_time = groups[0], t[groups[0][0]]

    def give_to_winners(amount: float, victim_team: str) -> bool:
        eligible = [w for w in winners if entries[w].team != victim_team]
        for w in eligible:
            pts[w] += amount / len(eligible)
        return bool(eligible)

    # Margin jackpots
    for group in groups[1:]:
        for i in group:
            if t[i] - win_time >= threshold - 1e-9 and give_to_winners(pts[i], entries[i].team):
                pts[i] = 0.0

    # DNS / DNF / DSQ: lowest places, always jackpotted (or forfeited if same team)
    for k, i in enumerate(penalized):
        give_to_winners(alloc(len(finishers) + k), entries[i].team)
        pts[i] = 0.0

    return pts


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    assert len(EVENTS) == 30 and EVENT_BY_NUMBER[9].name == "200 Fly" and EVENT_BY_NUMBER[30].name == "Super Skins"

    # Rulebook example (6.3.1): Men's 100 Free, winner 41.39, last two 44.20 and 44.85 -> winner 12, last two 0
    m100 = EVENT_BY_NUMBER[16]
    times = [41.39, 42.0, 42.3, 42.6, 43.0, 43.5, 44.20, 44.85]
    race = [RaceEntry(f"T{i % 4}", f"S{i}", tm) for i, tm in enumerate(times)]
    race[0].team, race[6].team, race[7].team = "A", "B", "C"
    pts = score_race(race, INDIVIDUAL_POINTS, jackpot_threshold(m100))
    assert pts[0] == 12 and pts[6] == 0 and pts[7] == 0, pts

    # Tie for 2nd -> 6.5 each (6.1)
    race = [RaceEntry("A", "a", 50.0), RaceEntry("B", "b", 50.5), RaceEntry("C", "c", 50.5), RaceEntry("D", "d", 51.0)]
    assert score_race(race, INDIVIDUAL_POINTS, 99)[1:3] == [6.5, 6.5]

    # DSQ is auto-jackpotted to the winner; forfeited if the winner is a teammate (6.3)
    race = [RaceEntry("A", "a", 50.0), RaceEntry("B", "b", 50.1), RaceEntry("C", "c", None)]
    assert score_race(race, INDIVIDUAL_POINTS, 99) == [9 + 6, 7, 0]
    race[2].team = "A"
    assert score_race(race, INDIVIDUAL_POINTS, 99) == [9, 7, 0]

    print("All rule checks passed.")
