import numpy as np
import pandas as pd

from csl_rules import (
    EVENTS, INDIVIDUAL_POINTS, MAX_INDIVIDUAL_EVENTS, Gender, Kind, RaceEntry,
    jackpot_threshold, score_race,
)
from load_data import build_times, load_csl, load_current_swims

# times[swimmer][event] -> list of SCY times in seconds, oldest first
TIMES = build_times(load_current_swims(), course="SCY")

# CSL results for the meets Stanford swam in (csl_match_3, csl_match_5, csl_wild_card)
CSL_DF = load_csl()

rng = np.random.default_rng()


def predict_time(
    swimmer: str, event: str, noise: float = 0.01, fallback: float | None = None
) -> float:
    """Most recent time for `swimmer` in `event`, plus Gaussian noise.

    noise is the standard deviation as a fraction of the time (0.01 = 1%).
    fallback is used as the base time if the swimmer has no SCY history in `event`.
    """
    history = TIMES.get(swimmer, {}).get(event)
    last = history[-1] if history else fallback
    if last is None:
        raise KeyError(f"No {event} time for {swimmer} and no fallback given")
    return last * (1 + rng.normal(0, noise))


# Men's individual events in meet order (Skins/relays excluded)
MEN_INDIVIDUAL_EVENTS = [
    e.name for e in EVENTS if e.gender == Gender.MEN and e.kind == Kind.INDIVIDUAL
]


def greedy_lineup(times: dict[str, dict[str, list[float]]] = TIMES) -> list[tuple[str, ...]]:
    """Greedy policy: for each men's individual event in meet order, enter the two
    fastest swimmers (by most recent time) who still have < MAX_INDIVIDUAL_EVENTS events.

    Swimmers with no time in an event are not considered for it. If fewer than two
    are eligible, the tuple for that event is shorter.
    """
    counts = {swimmer: 0 for swimmer in times}
    lineup = []
    for event in MEN_INDIVIDUAL_EVENTS:
        eligible = [
            s for s in times
            if event in times[s] and counts[s] < MAX_INDIVIDUAL_EVENTS
        ]
        picks = sorted(eligible, key=lambda s: times[s][event][-1])[:2]
        for s in picks:
            counts[s] += 1
        lineup.append(tuple(picks))
    return lineup


MEN_INDIVIDUAL_EVENT_OBJS = [e for e in EVENTS if e.gender == Gender.MEN and e.kind == Kind.INDIVIDUAL]


def actual_lineup(match: str) -> list[tuple[str, ...]]:
    """Stanford's real men's individual-event lineup in `match`, in MEN_INDIVIDUAL_EVENTS order."""
    df = CSL_DF[(CSL_DF["match"] == match) & (CSL_DF["gender"] == "men") & (CSL_DF["team"] == "Stanford")]
    return [tuple(df.loc[df["event"] == event].iloc[:, 0]) for event in MEN_INDIVIDUAL_EVENTS]


def actual_times(match: str) -> dict[tuple[str, str], float]:
    """{(swimmer, event): actual CSL time} for Stanford men in `match` (used as a fallback)."""
    df = CSL_DF[(CSL_DF["match"] == match) & (CSL_DF["gender"] == "men") & (CSL_DF["team"] == "Stanford")]
    return {(r[0], r.event): float(r.time) for r in df.itertuples(index=False) if not pd.isna(r.time)}


def opponent_entries(match: str) -> dict[str, list[tuple[str, str, float | None]]]:
    """{event: [(team, swimmer, actual CSL time), ...]} for the non-Stanford men in `match`.

    A missing time (NaN) is returned as None, which score_race treats as DNS/DSQ.
    """
    df = CSL_DF[(CSL_DF["match"] == match) & (CSL_DF["gender"] == "men") & (CSL_DF["team"] != "Stanford")]
    out: dict[str, list] = {}
    for r in df.itertuples(index=False):
        t = None if pd.isna(r.time) else float(r.time)
        out.setdefault(r.event, []).append((r.team, r[0], t))
    return out


def simulate_meet(
    stanford_lineup: list[tuple[str, ...]], match: str, n_sims: int = 1000, noise: float = 0.01
) -> float:
    """Average Stanford men's individual-event points in `match` over `n_sims` simulations.

    stanford_lineup: one tuple of swimmers per event in MEN_INDIVIDUAL_EVENTS order
                     (e.g. the output of greedy_lineup()).
    match:           "csl_match_3", "csl_match_5", or "csl_wild_card".

    Stanford swimmers: most recent time + noise (predict_time).
    Opponents: their actual time in that CSL match + the same noise.
    Each race is scored with csl_rules.score_race (places, ties, jackpots).
    """
    opponents = opponent_entries(match)
    fallback = actual_times(match)
    totals = np.zeros(n_sims)
    for sim in range(n_sims):
        for event, pair in zip(MEN_INDIVIDUAL_EVENT_OBJS, stanford_lineup):
            race = [RaceEntry("Stanford", s, predict_time(s, event.name, noise, fallback.get((s, event.name))))
                    for s in pair]
            race += [
                RaceEntry(team, s, None if t is None else t * (1 + rng.normal(0, noise)))
                for team, s, t in opponents.get(event.name, [])
            ]
            pts = score_race(race, INDIVIDUAL_POINTS, jackpot_threshold(event))
            totals[sim] += sum(p for e, p in zip(race, pts) if e.team == "Stanford")
    return float(totals.mean())


if __name__ == "__main__":
    lineup = greedy_lineup()
    for event, pair in zip(MEN_INDIVIDUAL_EVENTS, lineup):
        print(f"{event:<11} {pair}")

    # Go Nagaoka opted out of CSL, so also run greedy without him
    lineup_no_go = greedy_lineup({s: t for s, t in TIMES.items() if s != "Go Nagaoka"})
    print("\nGreedy without Go Nagaoka (changed events only):")
    for event, pair, alt in zip(MEN_INDIVIDUAL_EVENTS, lineup, lineup_no_go):
        if pair != alt:
            print(f"{event:<11} {alt}")

    print("\nBaseline (Stanford's actual lineups) vs greedy:")
    for match in ["csl_match_3", "csl_match_5", "csl_wild_card"]:
        baseline = simulate_meet(actual_lineup(match), match)
        greedy = simulate_meet(lineup, match)
        greedy_no_go = simulate_meet(lineup_no_go, match)
        real = CSL_DF[(CSL_DF["match"] == match) & (CSL_DF["gender"] == "men")
                      & (CSL_DF["team"] == "Stanford") & CSL_DF["event"].isin(MEN_INDIVIDUAL_EVENTS)]["points"].sum()
        print(f"{match:<14} real result {real:6.1f} | actual lineup sim {baseline:6.1f} "
              f"| greedy sim {greedy:6.1f} | greedy (no Go) sim {greedy_no_go:6.1f}")
