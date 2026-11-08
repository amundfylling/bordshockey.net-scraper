"""
Data checks on the results dataset, and the counts the PRs report.

Checks (they fail the run with --strict):
- no game has the same player id on both sides,
- no player id has two names in one tournament that share no word after folding accents.

Counts: empty id slots and the games they cost, synthetic ids (9,000,000 + ITHF and
local ids), walkovers, tournament names without a year.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from clean_results import has_year
from map_players import ITHF_ONLY_ID_OFFSET, LOCAL_ID_START, fold_name
from results_store import RESULTS_PATH, read_results


def self_matches(df: pd.DataFrame) -> pd.DataFrame:
    p1 = pd.to_numeric(df["Player1ID"], errors="coerce")
    p2 = pd.to_numeric(df["Player2ID"], errors="coerce")
    return df[p1.notna() & (p1 == p2)]


def _sides(df: pd.DataFrame) -> pd.DataFrame:
    parts = [
        pd.DataFrame(
            {
                "SourceTournamentID": df["SourceTournamentID"].astype(str),
                "PlayerName": df[name].astype("string").fillna("").str.strip(),
                "PlayerID": pd.to_numeric(df[pid], errors="coerce"),
            }
        )
        for name, pid in (("Player1", "Player1ID"), ("Player2", "Player2ID"))
    ]
    return pd.concat(parts, ignore_index=True)


def _share_no_word(names) -> bool:
    words = [set(fold_name(n).split()) for n in names]
    return any(not (a & b) for i, a in enumerate(words) for b in words[i + 1:])


def name_conflicts(df: pd.DataFrame, per_tournament: bool = True) -> pd.DataFrame:
    """Ids with two names that share no word (in one tournament, or anywhere)."""
    s = _sides(df).dropna(subset=["PlayerID"])
    s = s[s["PlayerName"] != ""].drop_duplicates()
    keys = ["SourceTournamentID", "PlayerID"] if per_tournament else ["PlayerID"]
    names = s.groupby(keys)["PlayerName"].agg(lambda x: sorted(set(x)))
    names = names[names.map(len) > 1]
    bad = names[names.map(_share_no_word)]
    return bad.rename("Names").reset_index()


def counts(df: pd.DataFrame) -> dict[str, int]:
    s = _sides(df)
    named = s["PlayerName"] != ""
    ids = s["PlayerID"]
    p1 = pd.to_numeric(df["Player1ID"], errors="coerce")
    p2 = pd.to_numeric(df["Player2ID"], errors="coerce")
    tournaments = df.drop_duplicates("SourceTournamentID")
    out = {
        "rows": len(df),
        "empty id slots": int(ids.isna().sum()),
        "empty id slots with a player name": int((ids.isna() & named).sum()),
        "games without both ids": int((p1.isna() | p2.isna()).sum()),
        "distinct ids": int(ids.nunique()),
        "ITHF-only ids (9,000,000 + ITHF)": int(ids[(ids >= ITHF_ONLY_ID_OFFSET) & (ids < LOCAL_ID_START)].nunique()),
        "local ids (10,000,001+)": int(ids[ids >= LOCAL_ID_START].nunique()),
        "slots with a synthetic id": int((ids >= ITHF_ONLY_ID_OFFSET).sum()),
        "self-matches": len(self_matches(df)),
        "name conflicts in one tournament": len(name_conflicts(df)),
        "ids with names sharing no word anywhere": len(name_conflicts(df, per_tournament=False)),
        "tournaments": len(tournaments),
        "tournament names without a year": int((~tournaments["TournamentName"].map(has_year)).sum()),
    }
    if "Walkover" in df.columns:
        out["walkover games"] = int((df["Walkover"].astype(str) == "Yes").sum())
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Check the results dataset.")
    parser.add_argument("--results", default=RESULTS_PATH)
    parser.add_argument("--strict", action="store_true", help="Exit 1 when a check fails.")
    parser.add_argument("--show", type=int, default=20, help="Failures to print per check.")
    args = parser.parse_args(argv)

    df = read_results(args.results)
    for key, value in counts(df).items():
        print(f"{key}: {value:,}")

    failed = False
    selfm = self_matches(df)
    if len(selfm):
        failed = True
        print(f"\nFAIL: {len(selfm)} games with the same id on both sides")
        print(selfm[["SourceTournamentID", "Player1", "Player2", "Player1ID"]].head(args.show).to_string(index=False))
    conflicts = name_conflicts(df)
    if len(conflicts):
        failed = True
        print(f"\nFAIL: {len(conflicts)} ids with two unrelated names in one tournament")
        print(conflicts.head(args.show).to_string(index=False))
    return 1 if failed and args.strict else 0


if __name__ == "__main__":
    sys.exit(main())
