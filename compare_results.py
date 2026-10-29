"""
Compare two results CSVs (e.g. the HTML scraper's and resultat_api_scraper.py's)
and print where they differ.

    python3 compare_results.py bordshockey_results.csv /tmp/api_results.csv
"""

from __future__ import annotations

import sys

import pandas as pd

MATCH_KEY = ["SourceTournamentID", "StageURL", "Stage", "RoundNumber", "PlayoffGameNumber", "Player1", "Player2"]
VALUE_COLUMNS = ["GoalsPlayer1", "GoalsPlayer2", "Overtime", "StageID", "StageSequence", "StageType",
                 "Date", "TournamentName", "TournamentID", "SourceStageID", "SourceMatchID"]


def _load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    for col in ("PlayoffGameNumber", "SourceMatchID", "SourceStageID", "RoundNumber"):
        df[col] = df[col].str.replace(r"\.0$", "", regex=True)
    return df


def _show(title: str, df: pd.DataFrame, limit: int = 15) -> None:
    print(f"\n## {title}: {len(df)}")
    if len(df):
        print(df.head(limit).to_string())


def main(old_path: str, new_path: str) -> int:
    old, new = _load(old_path), _load(new_path)
    tournaments = sorted(set(old.SourceTournamentID) | set(new.SourceTournamentID))

    print(f"old rows: {len(old):,}  new rows: {len(new):,}  tournaments: {len(tournaments)}")

    per_t = pd.DataFrame({
        "old": old.groupby("SourceTournamentID").size(),
        "new": new.groupby("SourceTournamentID").size(),
    }).fillna(0).astype(int)
    per_t["diff"] = per_t.new - per_t.old
    _show("tournaments with different row counts", per_t[per_t["diff"] != 0].sort_values("diff"), 60)

    # Match rows by occurrence within the key so repeated pairings line up.
    for df in (old, new):
        df["_n"] = df.groupby(MATCH_KEY).cumcount().astype(str)
    key = MATCH_KEY + ["_n"]
    merged = old.merge(new, on=key, how="outer", suffixes=("_old", "_new"), indicator=True)

    only_old = merged[merged._merge == "left_only"]
    only_new = merged[merged._merge == "right_only"]
    _show("rows only in old", only_old[key + ["GoalsPlayer1_old", "GoalsPlayer2_old"]])
    _show("rows only in new", only_new[key + ["GoalsPlayer1_new", "GoalsPlayer2_new"]])
    print("\nonly-old by tournament:\n", only_old.SourceTournamentID.value_counts().head(30).to_string())
    print("\nonly-new by tournament:\n", only_new.SourceTournamentID.value_counts().head(30).to_string())

    both = merged[merged._merge == "both"]
    print(f"\n## matched rows: {len(both):,}")
    for col in VALUE_COLUMNS:
        a, b = both[f"{col}_old"], both[f"{col}_new"]
        diff = both[(a != b) & ~((col == "SourceMatchID") & (a == ""))]
        if len(diff):
            _show(f"{col} differs", diff[key[:2] + [f"{col}_old", f"{col}_new"]], 8)
        else:
            print(f"{col}: identical")

    # Stage names that only exist on one side, per tournament (e.g. translated names).
    s_old = set(zip(old.SourceTournamentID, old.Stage))
    s_new = set(zip(new.SourceTournamentID, new.Stage))
    _show("stage names only in old", pd.DataFrame(sorted(s_old - s_new), columns=["t", "stage"]), 40)
    _show("stage names only in new", pd.DataFrame(sorted(s_new - s_old), columns=["t", "stage"]), 40)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:3]))
