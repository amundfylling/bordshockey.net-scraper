"""
Fetch the ITHF and SBHF ids of every participant in the tournaments of
the results dataset (data/bordshockey_results.parquet) from the resultat.bordshockey.se JSON API
(GET /api/tournaments/{id}/export, docs: https://resultat.bordshockey.se/api-docs.html)
and cache them in api_player_ids.csv, one row per tournament and player name.

Only tournaments not already in the cache are fetched. map_players.py uses the
ITHF id to link players to Scorpion's RankingID.
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

from results_store import RESULTS_PATH, read_results

API_BASE = "https://resultat.bordshockey.se/api"
HEADERS = {"User-Agent": "bordshockey.net-scraper (+https://github.com/amundfylling/bordshockey.net-scraper)"}

API_IDS_CSV = "api_player_ids.csv"
API_IDS_COLUMNS = ["SourceTournamentID", "PlayerName", "ITHF", "SBHF", "Nation", "Club"]


def _get_json(session: requests.Session, path: str, params: dict[str, object] | None = None) -> object:
    r = session.get(f"{API_BASE}{path}", params=params, headers=HEADERS, timeout=60)
    r.raise_for_status()
    return r.json()


def _clean_id(value: object) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def participants_from_export(source_tournament_id: str, export: dict) -> list[dict[str, object]]:
    """One row per named participant of a tournament export."""
    rows = []
    for p in export.get("participants") or []:
        name = str(p.get("name") or "").strip()
        if not name:
            continue
        ids = p.get("ids") or {}
        rows.append(
            {
                "SourceTournamentID": source_tournament_id,
                "PlayerName": name,
                "ITHF": _clean_id(ids.get("ithf")),
                "SBHF": _clean_id(ids.get("sbhf")),
                "Nation": p.get("nation"),
                "Club": p.get("club"),
            }
        )
    return rows


def api_tournament_ids(session: requests.Session, seasons: set[str]) -> dict[str, int]:
    """Maps "season/permalink" (SourceTournamentID) to the API's tournament id."""
    ids: dict[str, int] = {}
    for season in sorted(seasons):
        for t in _get_json(session, "/tournaments", {"season": season}):
            ids[f"{t['season']}/{t['permalink']}"] = t["id"]
    return ids


def fetch_missing(source_tournament_ids: set[str], max_workers: int = 8) -> pd.DataFrame:
    seasons = {tid.split("/", 1)[0] for tid in source_tournament_ids}
    with requests.Session() as session:
        api_ids = api_tournament_ids(session, seasons)
        missing = sorted(tid for tid in source_tournament_ids if tid not in api_ids)
        if missing:
            print(f"{len(missing)} tournaments not found in the API: {', '.join(missing[:10])}", file=sys.stderr)

        def fetch(tid: str) -> list[dict[str, object]]:
            try:
                return participants_from_export(tid, _get_json(session, f"/tournaments/{api_ids[tid]}/export"))
            except Exception as exc:
                print(f"[ERROR] {tid} -> {exc}", file=sys.stderr)
                return []

        found = sorted(tid for tid in source_tournament_ids if tid in api_ids)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            rows = [row for chunk in pool.map(fetch, found) for row in chunk]
    return pd.DataFrame(rows, columns=API_IDS_COLUMNS)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cache API participant ids for the tournaments in the results CSV.")
    parser.add_argument("--csv", default=RESULTS_PATH, help="Results file whose tournaments to look up.")
    parser.add_argument("--out", default=API_IDS_CSV, help="Participant id cache.")
    parser.add_argument("--refresh", action="store_true", help="Fetch every tournament again.")
    args = parser.parse_args(argv)

    results = read_results(args.csv, columns=["SourceTournamentID"], as_text=True)
    wanted = set(results["SourceTournamentID"]) - {""}

    cached = pd.DataFrame(columns=API_IDS_COLUMNS)
    if os.path.exists(args.out) and not args.refresh:
        cached = pd.read_csv(args.out, encoding="utf-8-sig", dtype=str)

    todo = wanted - set(cached["SourceTournamentID"])
    print(f"{len(wanted)} tournaments, {len(todo)} to fetch.")
    fresh = fetch_missing(todo) if todo else pd.DataFrame(columns=API_IDS_COLUMNS)

    frames = [df for df in (cached, fresh) if not df.empty]
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=API_IDS_COLUMNS)
    out = out.sort_values(["SourceTournamentID", "PlayerName"], kind="stable")
    out.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"Wrote {args.out}: {len(out)} participants, {out['ITHF'].notna().sum()} with ITHF id.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
