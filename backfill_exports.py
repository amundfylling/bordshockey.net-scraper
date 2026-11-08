"""
Backfill facts that only the resultat API's tournament export and tournament list carry
onto the existing results:

- tournament_meta.csv: name, class, season, city and nation of every tournament.
- Walkover (Yes/No) on every row the export can be lined up with.
- row_player_overrides.csv: per-row ids for tournaments where two participants share
  a name (two "Igor Novikov" at Kursk Cup 2019). The dataset only stores names, so
  map_players.py can't tell them apart without this. Each participant is identified by
  the ITHF/SBHF ids the export gives it or, when it has none, by its games in the
  Sportscorpion match data (same date, opponent and score).

The daily fetch sets Walkover and the tournament metadata for new tournaments itself;
this script is for the rows fetched before that. Reads a saved dump (--dump, a gzipped
JSON with "tournaments" and "exports" as written by the player-matching investigation)
or fetches the exports from the API (--fetch).
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import os
import sys

import pandas as pd

from clean_results import TOURNAMENT_META_CSV, load_tournament_meta, merge_tournament_meta, save_tournament_meta
from export_index import export_records, join_results, records_frame
from map_players import ROW_OVERRIDES_CSV, ROW_OVERRIDE_COLUMNS, name_key, row_keys
from results_store import RESULTS_PATH, read_results, write_results

API_BASE = "https://resultat.bordshockey.se/api"
HEADERS = {"User-Agent": "bordshockey.net-scraper (+https://github.com/amundfylling/bordshockey.net-scraper)"}


def meta_from_tournaments(tournaments: list[dict]) -> pd.DataFrame:
    rows = [
        {
            "SourceTournamentID": f"{t['season']}/{t['permalink']}",
            "ApiTournamentID": t.get("id"),
            "Name": t.get("name") or "",
            "ClassName": t.get("className") or "",
            "Season": t.get("season") or "",
            "City": t.get("city") or "",
            "Nation": t.get("nation") or "",
        }
        for t in tournaments
    ]
    return pd.DataFrame(rows).astype(str)


def load_dump(path: str) -> tuple[list[dict], dict[str, dict]]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        dump = json.load(f)
    by_id = {str(t["id"]): f"{t['season']}/{t['permalink']}" for t in dump["tournaments"]}
    exports = {by_id[k]: v for k, v in dump["exports"].items() if k in by_id and isinstance(v, dict) and "_error" not in v}
    return dump["tournaments"], exports


def fetch_exports(source_tournament_ids: set[str]) -> tuple[list[dict], dict[str, dict]]:
    import requests

    tournaments, exports = [], {}
    with requests.Session() as s:
        seasons = [x["name"] for x in s.get(f"{API_BASE}/seasons", headers=HEADERS, timeout=60).json()]
        for season in seasons:
            tournaments += s.get(f"{API_BASE}/tournaments", params={"season": season}, headers=HEADERS, timeout=60).json()
        for t in tournaments:
            tid = f"{t['season']}/{t['permalink']}"
            if tid not in source_tournament_ids:
                continue
            r = s.get(f"{API_BASE}/tournaments/{t['id']}/export", headers=HEADERS, timeout=60)
            if r.ok:
                exports[tid] = r.json()
    return tournaments, exports


def scorpion_index(path: str) -> dict:
    """(date, name key, opponent name key, own goals, opponent goals) -> Counter of the Sportscorpion PlayerID."""
    m = pd.read_parquet(path, columns=["Player1", "Player1ID", "Player2", "Player2ID", "GoalsPlayer1", "GoalsPlayer2", "Date"])
    idx: dict = collections.defaultdict(collections.Counter)
    keys = {}
    for r in m.itertuples(index=False):
        a = keys.setdefault(r.Player1, name_key(r.Player1))
        b = keys.setdefault(r.Player2, name_key(r.Player2))
        idx[(r.Date, a, b, r.GoalsPlayer1, r.GoalsPlayer2)][int(r.Player1ID)] += 1
        idx[(r.Date, b, a, r.GoalsPlayer2, r.GoalsPlayer1)][int(r.Player2ID)] += 1
    return idx


def duplicate_name_overrides(results: pd.DataFrame, joined: pd.DataFrame, exports: dict[str, dict], scorpion: dict | None) -> pd.DataFrame:
    """Per-row ids for participants who share their name with another participant of the same tournament."""
    dup_names: dict[str, set[str]] = {}
    for tid, ex in exports.items():
        c = collections.Counter(str(p.get("name") or "").strip() for p in ex.get("participants") or [])
        names = {n for n, k in c.items() if k > 1 and n}
        if names:
            dup_names[tid] = names

    keys = row_keys(results)
    sides = []
    for side in (1, 2):
        part = pd.DataFrame(
            {
                "SourceTournamentID": results["SourceTournamentID"].astype(str),
                "RowKey": keys,
                "Side": side,
                "PlayerName": results[f"Player{side}"].fillna("").astype(str).str.strip(),
                "Opponent": results[f"Player{3 - side}"].fillna("").astype(str).str.strip(),
                "Goals": pd.to_numeric(results[f"GoalsPlayer{side}"], errors="coerce"),
                "OppGoals": pd.to_numeric(results[f"GoalsPlayer{3 - side}"], errors="coerce"),
                "Date": results["Date"].astype(str),
                "PKey": joined[f"P{side}Key"],
                "ITHF": joined[f"P{side}ITHF"],
                "SBHF": joined[f"P{side}SBHF"],
            },
            index=results.index,
        )
        sides.append(part)
    s = pd.concat(sides)
    s = s[[n in dup_names.get(t, ()) for t, n in zip(s["SourceTournamentID"], s["PlayerName"])]]
    s = s[s["PKey"].notna()]

    out = []
    for (tid, name), grp in s.groupby(["SourceTournamentID", "PlayerName"]):
        people = grp.groupby("PKey")
        if people.ngroups < 2:
            continue  # only one of the namesakes played
        idents = {k: (g["ITHF"].dropna().iloc[0] if g["ITHF"].notna().any() else None,
                      g["SBHF"].dropna().iloc[0] if g["SBHF"].notna().any() else None) for k, g in people}
        pids = {}
        if len(set(idents.values())) < len(idents) and scorpion is not None:
            for k, g in people:
                votes = collections.Counter()
                for r in g.itertuples():
                    votes.update(scorpion.get((r.Date, name_key(r.PlayerName), name_key(r.Opponent), r.Goals, r.OppGoals), {}))
                if votes:
                    best, n = votes.most_common(1)[0]
                    if n >= 3 and n >= 0.8 * sum(votes.values()):
                        pids[k] = best
            if len(set(pids.values())) < len(pids):
                pids = {}
        for k, g in people:
            ithf, sbhf = idents[k]
            pid = pids.get(k)
            unique_ids = list(idents.values()).count(idents[k]) == 1
            if not pid and not (unique_ids and (ithf or sbhf)):
                continue
            reason = "Sportscorpion games" if pid else "export participant ids"
            for r in g.itertuples():
                out.append(
                    {
                        "SourceTournamentID": tid,
                        "RowKey": r.RowKey,
                        "Side": r.Side,
                        "PlayerName": name,
                        "ITHF": "" if pid else (ithf or ""),
                        "SBHF": "" if pid else (sbhf or ""),
                        "PlayerID": pid or "",
                        "Reason": f"Two participants named {name}; {reason} ({k})",
                    }
                )
    return pd.DataFrame(out, columns=ROW_OVERRIDE_COLUMNS)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Backfill Walkover, tournament metadata and namesake ids from API exports.")
    parser.add_argument("--results", default=RESULTS_PATH)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--dump", help="Saved API dump (gzipped JSON with tournaments and exports).")
    src.add_argument("--fetch", action="store_true", help="Fetch the exports from the API.")
    parser.add_argument("--scorpion-matches", help="Sportscorpion scraped_matches.parquet, to tell id-less namesakes apart.")
    args = parser.parse_args(argv)

    results = read_results(args.results, as_text=True)
    if args.dump:
        tournaments, exports = load_dump(args.dump)
    else:
        tournaments, exports = fetch_exports(set(results["SourceTournamentID"]))
    print(f"{len(tournaments)} tournaments, {len(exports)} exports")

    meta = merge_tournament_meta(load_tournament_meta(), meta_from_tournaments(tournaments))
    save_tournament_meta(meta)
    print(f"Wrote {TOURNAMENT_META_CSV}: {len(meta)} tournaments")

    records = records_frame([r for tid, ex in exports.items() for r in export_records(tid, ex)])
    joined = join_results(results, records)
    matched = joined["Walkover"].notna()
    print(f"Rows lined up with an export match: {int(matched.sum())} of {len(results)}")
    wo = joined["Walkover"].map({True: "Yes", False: "No"})
    old = results["Walkover"] if "Walkover" in results.columns else pd.Series("", index=results.index)
    results["Walkover"] = wo.where(matched, old.replace("", pd.NA)).fillna("No")
    print(f"Walkover games: {int((results['Walkover'] == 'Yes').sum())}")
    write_results(args.results, results)

    scorpion = scorpion_index(args.scorpion_matches) if args.scorpion_matches else None
    overrides = duplicate_name_overrides(results, joined, exports, scorpion)
    if os.path.exists(ROW_OVERRIDES_CSV):
        old_ov = pd.read_csv(ROW_OVERRIDES_CSV, dtype=str, keep_default_na=False)
        overrides = pd.concat([old_ov, overrides.astype(str)]).drop_duplicates(["SourceTournamentID", "RowKey", "Side"], keep="last")
    overrides.to_csv(ROW_OVERRIDES_CSV, index=False, encoding="utf-8-sig")
    print(f"Wrote {ROW_OVERRIDES_CSV}: {len(overrides)} row sides")
    return 0


if __name__ == "__main__":
    sys.exit(main())
