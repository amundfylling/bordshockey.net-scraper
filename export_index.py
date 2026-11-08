"""
Match-level data from the resultat API's tournament export
(GET /api/tournaments/{id}/export, schema tournament-0.1).

The export has per-match flags that the stage endpoints the fetcher reads do not
expose for old tournaments (walkover, carriedOver), and it lists every participant
with its own ids, including two participants who share a name. join_results()
lines export matches up with rows of the results dataset so those facts can be
copied onto the rows.
"""

from __future__ import annotations

import pandas as pd

RECORD_COLUMNS = [
    "SourceTournamentID",
    "Kind",
    "ExportStage",
    "Round",
    "Player1",
    "Player2",
    "GoalsPlayer1",
    "GoalsPlayer2",
    "Walkover",
    "CarriedOver",
    "SuddenDeath",
    "P1Key",
    "P2Key",
    "P1ITHF",
    "P2ITHF",
    "P1SBHF",
    "P2SBHF",
]

# Columns a results row and an export match must agree on. The stage name is left out on purpose:
# the HTML scraper wrote the old site's English stage names, the export has the new Swedish ones.
JOIN_KEYS = ["SourceTournamentID", "Kind", "Round", "Player1", "Player2", "GoalsPlayer1", "GoalsPlayer2"]


def _ids(participant: dict) -> tuple[str | None, str | None]:
    ids = participant.get("ids") or {}
    ithf = str(ids.get("ithf") or "").strip() or None
    sbhf = str(ids.get("sbhf") or "").strip() or None
    return ithf, sbhf


def export_records(source_tournament_id: str, export: dict) -> list[dict]:
    """One record per played match or playoff game in the export."""
    people = {}
    for p in export.get("participants") or []:
        people[p.get("id")] = (str(p.get("name") or "").strip(), *_ids(p))
    blank = ("", None, None)

    def side(key):
        return people.get(key, blank)

    rows: list[dict] = []
    for part in export.get("parts") or []:
        for stage in part.get("stages") or []:
            groups = stage.get("groups") or []
            for group in groups:
                stage_name = stage.get("name", "") if len(groups) == 1 else f"{stage.get('name', '')} - {group.get('name', '')}"
                for m in group.get("matches") or []:
                    h, a = side(m.get("home")), side(m.get("away"))
                    rows.append(
                        {
                            "SourceTournamentID": source_tournament_id,
                            "Kind": "round-robin",
                            "ExportStage": stage_name,
                            "Round": m.get("round"),
                            "Player1": h[0],
                            "Player2": a[0],
                            "GoalsPlayer1": m.get("homeScore"),
                            "GoalsPlayer2": m.get("awayScore"),
                            "Walkover": bool(m.get("walkover")),
                            "CarriedOver": bool(m.get("carriedOver")),
                            "SuddenDeath": bool(m.get("suddenDeath")),
                            "P1Key": m.get("home"),
                            "P2Key": m.get("away"),
                            "P1ITHF": h[1],
                            "P2ITHF": a[1],
                            "P1SBHF": h[2],
                            "P2SBHF": a[2],
                        }
                    )
            playoff = stage.get("playoff") or {}
            for rnd in playoff.get("rounds") or []:
                stage_name = f"{stage.get('name', '')} {rnd.get('name', '')}".strip()
                for series in rnd.get("series") or []:
                    h, a = side(series.get("home")), side(series.get("away"))
                    for game in series.get("games") or []:
                        rows.append(
                            {
                                "SourceTournamentID": source_tournament_id,
                                "Kind": "playoff",
                                "ExportStage": stage_name,
                                "Round": game.get("game"),
                                "Player1": h[0],
                                "Player2": a[0],
                                "GoalsPlayer1": game.get("homeScore"),
                                "GoalsPlayer2": game.get("awayScore"),
                                "Walkover": bool(game.get("walkover")),
                                "CarriedOver": False,
                                "SuddenDeath": bool(game.get("suddenDeath")),
                                "P1Key": series.get("home"),
                                "P2Key": series.get("away"),
                                "P1ITHF": h[1],
                                "P2ITHF": a[1],
                                "P1SBHF": h[2],
                                "P2SBHF": a[2],
                            }
                        )
    return rows


def records_frame(records: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(records, columns=RECORD_COLUMNS)


def _join_frame(df: pd.DataFrame, kind_col: str, round_col: str | None) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["SourceTournamentID"] = df["SourceTournamentID"].astype("string")
    out["Kind"] = df[kind_col].astype("string")
    if round_col is None:
        rnd = df["RoundNumber"].where(df["StageType"] == "round-robin", df["PlayoffGameNumber"])
    else:
        rnd = df[round_col]
    out["Round"] = pd.to_numeric(rnd, errors="coerce").astype("Int64")
    for col in ("Player1", "Player2"):
        out[col] = df[col].fillna("").astype(str).str.strip()
    for col in ("GoalsPlayer1", "GoalsPlayer2"):
        out[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    return out


def join_results(results: pd.DataFrame, records: pd.DataFrame) -> pd.DataFrame:
    """
    Export record columns for each results row (same index as results); rows without a match get NA.

    Rows and records are paired on JOIN_KEYS; when several share a key they are paired in order.
    Carried-over copies in the export are skipped, as the fetcher skips them.
    """
    left = _join_frame(results, "StageType", None)
    recs = records[~records["CarriedOver"].astype(bool)]
    right = _join_frame(recs, "Kind", "Round")
    extra = [c for c in RECORD_COLUMNS if c not in JOIN_KEYS]
    for col in extra:
        right[col] = recs[col].values
    left["_n"] = left.groupby(JOIN_KEYS, dropna=False).cumcount()
    right["_n"] = right.groupby(JOIN_KEYS, dropna=False).cumcount()
    left["_row"] = left.index
    merged = left.merge(right, on=JOIN_KEYS + ["_n"], how="left")
    merged = merged.set_index("_row")
    merged.index.name = None
    return merged.loc[results.index, extra]
