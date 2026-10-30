"""
Fetch tournament results from the JSON API behind resultat.bordshockey.se
(resultat.bordshockey.net redirects there) and write them in the same CSV
schema as bordshockey_scraper.py.

The API reads the same database as bordshockey.net, so stage ids and match ids
are the ones the HTML scraper finds in the page markup. API docs:
https://resultat.bordshockey.se/api-docs.html
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Iterable

import pandas as pd
import requests
from tqdm import tqdm

from results_store import RESULTS_PATH, read_results

from bordshockey_scraper import (
    OUTPUT_COLUMNS,
    SOURCE,
    _apply_player_map,
    _deduplicate_results,
    _existing_result_urls,
    _generate_id,
    _load_existing_csv,
    _load_player_map,
    _needs_rebuild,
    _write_results,
)

API_BASE = "https://resultat.bordshockey.se/api"
SITE_BASE = "https://bordshockey.net/tavlingar"

HEADERS = {"User-Agent": "bordshockey.net-scraper (+https://github.com/amundfylling/bordshockey.net-scraper)"}


@dataclass(frozen=True)
class ApiTournament:
    id: int
    name: str
    season: str
    permalink: str
    start_date: str

    @property
    def source_tournament_id(self) -> str:
        return f"{self.season}/{self.permalink}"

    @property
    def tournament_url(self) -> str:
        return f"{SITE_BASE}/{self.source_tournament_id}/"

    @property
    def result_url(self) -> str:
        return f"{self.tournament_url}resultat/"


# api

def _get_json(session: requests.Session, path: str, params: dict[str, object] | None = None) -> object:
    r = session.get(f"{API_BASE}{path}", params=params, headers=HEADERS, timeout=60)
    r.raise_for_status()
    return r.json()


def list_seasons(session: requests.Session) -> list[str]:
    """Season short names ("2526"), newest first."""
    return [s["name"] for s in _get_json(session, "/seasons")]


def list_tournaments(session: requests.Session, season: str) -> list[ApiTournament]:
    """Finished individual (non-team) tournaments with results in one season."""
    return [
        ApiTournament(
            id=t["id"],
            name=t["name"],
            season=t["season"],
            permalink=t["permalink"],
            start_date=t["startDate"] or "",
        )
        for t in _get_json(session, "/tournaments", {"season": season})
        if t.get("hasResults") and t.get("isFinished") and not t.get("external") and not t.get("isTeam")
    ]


# row building

def _row(
    tournament: ApiTournament,
    *,
    stage_id: int,
    stage_name: str,
    stage_sequence: int,
    stage_type: str,
    stage_url: str,
    source_stage_id: int,
    source_match_id: int | None,
    player1: str,
    player2: str,
    goals1: int,
    goals2: int,
    sudden_death: bool,
    round_number: int,
    playoff_game_number: int | None,
) -> dict[str, object]:
    return {
        "StageID": stage_id,
        "Player1": player1,
        "Player1ID": None,
        "Player2": player2,
        "Player2ID": None,
        "GoalsPlayer1": goals1,
        "GoalsPlayer2": goals2,
        "Overtime": "Yes" if sudden_death else "No",
        "Stage": stage_name,
        "RoundNumber": round_number,
        "PlayoffGameNumber": playoff_game_number,
        "Date": tournament.start_date,
        "TournamentName": tournament.name,
        "TournamentID": _generate_id(tournament.tournament_url),
        "StageSequence": stage_sequence,
        "StageType": stage_type,
        "TournamentURL": tournament.tournament_url,
        "ResultURL": tournament.result_url,
        "StageURL": stage_url,
        "SourceURL": stage_url,
        "Source": SOURCE,
        "SourceTournamentID": tournament.source_tournament_id,
        "SourceStageID": source_stage_id,
        "SourceMatchID": source_match_id,
    }


def _played(match: dict) -> bool:
    return match.get("homeScore") is not None and match.get("awayScore") is not None and not match.get("hidden")


def rows_from_group_stage(
    tournament: ApiTournament,
    stage: dict,
    detail: dict,
    names: dict[int, str],
) -> list[dict[str, object]]:
    """One row per played match. Carried-over matches (includedMatches) belong to the earlier stage."""
    rows: list[dict[str, object]] = []
    groups = detail.get("groups") or []
    single_group = len(groups) == 1

    for group in groups:
        if single_group:
            stage_name = stage["name"]
            stage_url = f"{tournament.result_url}{stage['permalink']}/?matcher=1"
        else:
            stage_name = f"{stage['name']} - {group['name']}"
            stage_url = f"{tournament.result_url}{stage['permalink']}/{group['permalink']}/?matcher=1"

        for rnd in group.get("rounds") or []:
            for match in rnd.get("matches") or []:
                if not _played(match):
                    continue
                rows.append(
                    _row(
                        tournament,
                        stage_id=_generate_id(stage_url),
                        stage_name=stage_name,
                        stage_sequence=stage["stageGroup"],
                        stage_type="round-robin",
                        stage_url=stage_url,
                        source_stage_id=stage["id"],
                        source_match_id=match["id"],
                        player1=names.get(match["homeId"], ""),
                        player2=names.get(match["awayId"], ""),
                        goals1=match["homeScore"],
                        goals2=match["awayScore"],
                        sudden_death=bool(match.get("suddenDeath")),
                        round_number=match.get("round") or rnd["round"],
                        playoff_game_number=None,
                    )
                )

    return rows


def rows_from_playoff_stage(
    tournament: ApiTournament,
    stage: dict,
    detail: dict,
    names: dict[int, str],
) -> list[dict[str, object]]:
    """One row per played game, scores from the series' home player's side."""
    rows: list[dict[str, object]] = []
    stage_url = f"{tournament.result_url}{stage['permalink']}/"

    for rnd in (detail.get("playoff") or {}).get("rounds") or []:
        stage_name = f"{stage['name']} {rnd['name']}".strip()
        for series in rnd.get("series") or []:
            home, away = series.get("homeId"), series.get("awayId")
            for game_no, game in enumerate(series.get("games") or [], start=1):
                if not _played(game):
                    continue
                g1, g2 = game["homeScore"], game["awayScore"]
                if game.get("homeId") == away and game.get("awayId") == home:
                    g1, g2 = g2, g1
                rows.append(
                    _row(
                        tournament,
                        stage_id=stage["id"],
                        stage_name=stage_name,
                        stage_sequence=stage["stageGroup"],
                        stage_type="playoff",
                        stage_url=stage_url,
                        source_stage_id=stage["id"],
                        source_match_id=game["id"],
                        player1=names.get(home, ""),
                        player2=names.get(away, ""),
                        goals1=g1,
                        goals2=g2,
                        sudden_death=bool(game.get("suddenDeath")),
                        round_number=1,
                        playoff_game_number=game.get("round") or game_no,
                    )
                )

    return rows


def fetch_tournament(session: requests.Session, tournament: ApiTournament) -> pd.DataFrame:
    stages = _get_json(session, f"/tournaments/{tournament.id}/stages")
    participants = _get_json(session, f"/tournaments/{tournament.id}/participants")
    names = {p["id"]: p["name"] for p in participants}

    rows: list[dict[str, object]] = []
    for stage in stages:
        if stage.get("kind") not in ("group", "playoff"):
            continue
        detail = _get_json(session, f"/tournaments/{tournament.id}/stages/{stage['id']}")
        if stage["kind"] == "group":
            rows.extend(rows_from_group_stage(tournament, stage, detail, names))
        else:
            rows.extend(rows_from_playoff_stage(tournament, stage, detail, names))

    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def _fetch_tournament_thread(tournament: ApiTournament) -> pd.DataFrame:
    with requests.Session() as session:
        return fetch_tournament(session, tournament)


def fetch_all(
    tournaments: list[ApiTournament],
    max_workers: int = 4,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_fetch_tournament_thread, t): t for t in tournaments}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Tournaments", unit="event"):
            t = futures[fut]
            try:
                frames.append(fut.result())
            except Exception as exc:
                print(f"[ERROR] {t.source_tournament_id} (id {t.id}) -> {exc}", file=sys.stderr)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=OUTPUT_COLUMNS)


def discover_tournaments(session: requests.Session, seasons: Iterable[str] | None = None) -> list[ApiTournament]:
    found: list[ApiTournament] = []
    for season in seasons or list_seasons(session):
        found.extend(list_tournaments(session, season))
    return found


# CLI entry point

def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch tournament results from the resultat.bordshockey.se API.")
    parser.add_argument("--csv", default=RESULTS_PATH, help="Output path: .parquet (default) or .csv.")
    parser.add_argument("--max-workers", type=int, default=4, help="Parallel tournament fetches.")
    parser.add_argument("--rebuild", action="store_true", help="Ignore existing CSV and fetch everything.")
    parser.add_argument("--season", action="append", help="Season short name such as 2526. Can be repeated.")
    parser.add_argument(
        "--only",
        action="append",
        help="Only this tournament, as season/permalink (e.g. 2526/swedish-masters). Can be repeated.",
    )
    parser.add_argument(
        "--only-from",
        help="Only tournaments whose SourceTournamentID appears in this results file (for comparing with the HTML scraper).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    only: set[str] = set(args.only or [])
    if args.only_from:
        only |= set(read_results(args.only_from, columns=["SourceTournamentID"], as_text=True)["SourceTournamentID"])
    seasons = args.season or (sorted({o.split("/", 1)[0] for o in only}) if only else None)

    player_map = _load_player_map()
    existing_df = _load_existing_csv(args.csv)
    existing_needs_rebuild = not existing_df.empty and _needs_rebuild(existing_df)
    if existing_needs_rebuild and (args.season or only) and not args.rebuild:
        print(
            "Refusing to replace an incomplete existing CSV with a filtered fetch. "
            "Use --rebuild to overwrite it, or pass --csv with a new/temporary output path.",
            file=sys.stderr,
        )
        return 2
    rebuild = args.rebuild or existing_needs_rebuild

    with requests.Session() as session:
        tournaments = discover_tournaments(session, seasons)
    if only:
        tournaments = [t for t in tournaments if t.source_tournament_id in only]

    frames: list[pd.DataFrame] = []
    if not rebuild and not existing_df.empty:
        frames.append(existing_df)
        skip = _existing_result_urls(existing_df)
        tournaments = [t for t in tournaments if t.result_url not in skip]

    print(f"Fetching {len(tournaments):,} tournaments from {API_BASE} ...", file=sys.stderr)
    frames.append(fetch_all(tournaments, max_workers=args.max_workers))

    combined = pd.concat(frames, ignore_index=True)
    combined = _deduplicate_results(combined)
    combined = _apply_player_map(combined, player_map)
    _write_results(args.csv, combined)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
