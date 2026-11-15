import re
from pathlib import Path

import pandas as pd

SWIMS_CSV = "stanford_men_all_swims_2019_2026.csv"
ROSTER_TXT = "curr_roster.txt"
CSL_CSV = "csl_results.csv"


def load_roster(path: Path) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def build_times(df: pd.DataFrame, course: str = "SCY") -> dict[str, dict[str, list[float]]]:
    """Return {swimmer: {event: [times in seconds, oldest first]}} for one course.

    course: "SCY" (yards, NCAA season), "LCM" (long course meters), or "SCM".
    Mixing courses would put yard and meter times in the same list, so we pick one.
    """
    df = df[df["course"] == course].sort_values("date", kind="stable")
    times: dict[str, dict[str, list[float]]] = {}
    for (swimmer, event), group in df.groupby(["swimmer", "event"], sort=True):
        times.setdefault(swimmer, {})[event] = group["seconds"].tolist()
    return times


def meet_key(meet: str) -> str:
    """'CSL Match #3' -> 'csl_match_3', 'CSL Wild Card' -> 'csl_wild_card'."""
    return re.sub(r"[^a-z0-9]+", "_", meet.lower()).strip("_")


def load_csl(path: str = CSL_CSV, team: str = "Stanford") -> pd.DataFrame:
    """Load CSL results, keeping only the meets `team` swam in (all meets if team=None)."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["match"] = df["meet"].map(meet_key)
    df["gender"] = df["gender"].str.lower()
    if team is not None:
        df = df[df["match"].isin(df.loc[df["team"] == team, "match"])]
    return df


def build_csl_times(df: pd.DataFrame) -> dict[str, dict[str, dict[str, dict[str, dict[str, list[float]]]]]]:
    """Return {match: {school: {gender: {swimmer: {event: [times in seconds]}}}}}.

    Skins rounds stay separate events, e.g. "Skins 50 Free (R1)" ... "(R4)".
    """
    csl: dict = {}
    for r in df.itertuples(index=False):
        (csl.setdefault(r.match, {}).setdefault(r.team, {}).setdefault(r.gender, {})
            .setdefault(r[0], {}).setdefault(r.event, []).append(r.time))
    return csl


def build_csl_lineups(
    df: pd.DataFrame, exclude: tuple[str, ...] = ("Stanford",)
) -> dict[str, dict[str, dict[str, dict[str, list[str]]]]]:
    """Return {match: {school: {gender: {event: [swimmers who swam it]}}}}.

    Skins lineups are the Round 1 entrants, under e.g. "Skins 50 Free".
    """
    df = df[~df["event"].str.startswith("Skins") | df["event"].str.endswith("(R1)")]
    lineups: dict = {}
    for r in df.itertuples(index=False):
        if r.team in exclude:
            continue
        event = r.event.removesuffix(" (R1)")
        (lineups.setdefault(r.match, {}).setdefault(r.team, {}).setdefault(r.gender, {})
            .setdefault(event, []).append(r[0]))
    return lineups


def load_current_swims(swims_path: str = SWIMS_CSV, roster_path: str = ROSTER_TXT) -> pd.DataFrame:
    """Load the Stanford swims CSV, keeping only athletes on the current roster."""
    # utf-8-sig strips the BOM at the start of the CSV header
    swims = pd.read_csv(swims_path, encoding="utf-8-sig", low_memory=False)
    roster = load_roster(roster_path)

    # Case-insensitive match, renamed to the roster's spelling ("Jonathan TAN" -> "Jonathan Tan")
    roster_by_key = {name.casefold(): name for name in roster}
    roster_keys = set(roster_by_key)
    swimmer_keys = swims["swimmer"].str.strip().str.casefold()
    current = swims[swimmer_keys.isin(roster_keys)].copy()
    current["swimmer"] = swimmer_keys[current.index].map(roster_by_key)

    missing = roster_keys - set(current["swimmer"].str.casefold())
    if missing:
        print(f"Warning: no swims found for: {sorted(missing)}")
    return current


def main() -> None:
    current = load_current_swims()

    print("\nUnique athletes:")
    for name in sorted(current["swimmer"].unique()):
        print(f"  {name}")

    print(f"\n{len(current)} rows for {current['swimmer'].nunique()} current athletes")

    times = build_times(current, course="SCY")

    # CSL (only meets Stanford swam in): csl[match][school][gender][swimmer][event][i]
    csl_df = load_csl()
    csl = build_csl_times(csl_df)
    for match, teams in csl.items():
        print(f"{match}: {list(teams)}")
    print("Example:", csl["csl_match_3"]["Stanford"]["men"])

    # Lineups of the other teams: lineups[match][school][gender][event] -> swimmers
    lineups = build_csl_lineups(csl_df)
    print("csl_match_3 Cal men 100 Free:", lineups["csl_match_3"]["California"]["men"]["100 Free"])


if __name__ == "__main__":
    main()
