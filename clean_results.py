"""
Clean the results dataset after fetching and before player mapping.

1. Source corrections from source_corrections.json (rows the source has wrong,
   each with the evidence for the fix).
2. Drops games the source lists twice:
   - a placement game listed under two "Plats a-b" rounds of the same playoff,
   - final or placement groups that repeat the earlier group game of each pair
     (the earlier result carried over, but stored as new games without the
     carriedOver flag, e.g. Tornaligan 90 and Veberödsligan #2).
3. Clears Overtime on games that can't be sudden death: a tie, or a win by 2+.
4. Adds Country and City (from tournament_meta.csv, Country defaults to Sweden)
   and puts the season, and the class for class variants, into tournament names
   that lack them ("SM" in 2526/sm_veteran becomes "SM Veteran 2025/26").
5. Fills Walkover with "No" where the fetcher had no flag.

Every step is idempotent, so the daily workflow can run it on the whole dataset.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

import pandas as pd

from results_store import RESULTS_PATH, read_results, write_results

TOURNAMENT_META_CSV = "tournament_meta.csv"
TOURNAMENT_META_COLUMNS = ["SourceTournamentID", "ApiTournamentID", "Name", "ClassName", "Season", "City", "Nation"]
SOURCE_CORRECTIONS_FILE = "source_corrections.json"
DEFAULT_COUNTRY = "Sweden"
OPEN_CLASS = "Öppna"

# IOC/ISO codes used by the resultat API, in English.
COUNTRY_NAMES = {
    "SWE": "Sweden",
    "NOR": "Norway",
    "FIN": "Finland",
    "DEN": "Denmark",
    "ISL": "Iceland",
    "LAT": "Latvia",
    "LTU": "Lithuania",
    "EST": "Estonia",
    "RUS": "Russia",
    "BLR": "Belarus",
    "UKR": "Ukraine",
    "KAZ": "Kazakhstan",
    "CZE": "Czech Republic",
    "SVK": "Slovakia",
    "SLO": "Slovenia",
    "HUN": "Hungary",
    "AUT": "Austria",
    "SUI": "Switzerland",
    "GER": "Germany",
    "FRA": "France",
    "NED": "Netherlands",
    "GBR": "United Kingdom",
    "IRL": "Ireland",
    "POR": "Portugal",
    "PRT": "Portugal",
    "ESP": "Spain",
    "ITA": "Italy",
    "POL": "Poland",
    "SRB": "Serbia",
    "CRO": "Croatia",
    "ALB": "Albania",
    "ISR": "Israel",
    "USA": "United States",
    "CAN": "Canada",
    "BRA": "Brazil",
    "CHI": "Chile",
    "JPN": "Japan",
    "IND": "India",
    "PAK": "Pakistan",
    "NEP": "Nepal",
    "HKG": "Hong Kong",
    "IRN": "Iran",
    "SYR": "Syria",
    "MAR": "Morocco",
    "TRI": "Trinidad and Tobago",
}

_YEAR_RE = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)|(?<!\d)\d{2}/\d{2}(?!\d)")
_PLACE_RE = re.compile(r"(\d+)\s*-\s*(\d+)")


# --- tournament names and metadata ---

def country_name(code: object) -> str | None:
    text = str(code or "").strip().upper()
    if not text or text == "???":
        return None
    return COUNTRY_NAMES.get(text, text)


def season_label(source_tournament_id: object) -> str | None:
    """'1314/sm' -> '2013/14', '9900/x' -> '1999/00', '0001/x' -> '2000/01'."""
    code = str(source_tournament_id or "").split("/", 1)[0]
    if not re.fullmatch(r"\d{4}", code):
        return None
    first = int(code[:2])
    century = 1900 if first >= 50 else 2000
    return f"{century + first}/{code[2:]}"


def has_year(name: object) -> bool:
    return bool(_YEAR_RE.search(str(name or "")))


def display_name(name: object, source_tournament_id: object, class_name: object = None) -> str:
    """Adds the class (unless it is the open class or already named) and the season (unless a year is there)."""
    text = re.sub(r"\s+", " ", str(name or "")).strip()
    cls = str(class_name or "").strip()
    if cls and cls != OPEN_CLASS and cls.casefold() not in text.casefold():
        text = f"{text} {cls}".strip()
    season = season_label(source_tournament_id)
    if season and not has_year(text):
        text = f"{text} {season}".strip()
    return text


def load_tournament_meta(path: str = TOURNAMENT_META_CSV) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=TOURNAMENT_META_COLUMNS)
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def save_tournament_meta(meta: pd.DataFrame, path: str = TOURNAMENT_META_CSV) -> None:
    meta = meta.reindex(columns=TOURNAMENT_META_COLUMNS).fillna("")
    meta = meta.drop_duplicates("SourceTournamentID", keep="last").sort_values("SourceTournamentID")
    meta.to_csv(path, index=False, encoding="utf-8-sig")


def merge_tournament_meta(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """New rows win; tournaments only in old are kept."""
    return pd.concat([old, new], ignore_index=True).drop_duplicates("SourceTournamentID", keep="last")


def add_tournament_meta(df: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    by_tid = meta.set_index("SourceTournamentID") if not meta.empty else pd.DataFrame(columns=TOURNAMENT_META_COLUMNS[1:])
    tids = out["SourceTournamentID"].astype(str)

    def lookup(col):
        if col not in by_tid.columns:
            return pd.Series([""] * len(out), index=out.index)
        return tids.map(by_tid[col]).fillna("").astype(str)

    nation = lookup("Nation").map(country_name)
    city = lookup("City").str.strip()
    old_country = out["Country"] if "Country" in out.columns else pd.Series(pd.NA, index=out.index)
    old_city = out["City"] if "City" in out.columns else pd.Series(pd.NA, index=out.index)
    out["Country"] = nation.where(nation.notna(), old_country).fillna(DEFAULT_COUNTRY).replace("", DEFAULT_COUNTRY)
    out["City"] = city.where(city != "", old_city).replace("", pd.NA)

    classes = lookup("ClassName")
    names = {}
    for tid, name, cls in zip(tids, out["TournamentName"], classes):
        key = (tid, name, cls)
        if key not in names:
            names[key] = display_name(name, tid, cls)
    out["TournamentName"] = [names[(t, n, c)] for t, n, c in zip(tids, out["TournamentName"], classes)]
    return out


# --- source corrections ---

def load_source_corrections(path: str = SOURCE_CORRECTIONS_FILE) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f).get("corrections", [])


def apply_source_corrections(df: pd.DataFrame, corrections: list[dict]) -> tuple[pd.DataFrame, int]:
    """
    Each correction names a row by SourceTournamentID + SourceMatchID and either drops it ("action": "drop")
    or sets fields that still have the old value ("set": {col: {"from": old, "to": new}}).
    """
    out = df.copy()
    applied = 0
    for fix in corrections:
        mask = (out["SourceTournamentID"].astype(str) == str(fix["SourceTournamentID"])) & (
            out["SourceMatchID"].astype(str) == str(fix["SourceMatchID"])
        )
        for col, value in (fix.get("where") or {}).items():
            mask &= out[col].astype(str) == str(value)
        if fix.get("action") == "drop":
            applied += int(mask.sum())
            out = out[~mask]
            continue
        for col, change in (fix.get("set") or {}).items():
            hit = mask & (out[col].astype(str) == str(change["from"]))
            if hit.any():
                out.loc[hit, col] = change["to"]
                applied += int(hit.sum())
    return out, applied


# --- duplicate games ---

def placement_duplicates(df: pd.DataFrame) -> list:
    """
    Index labels of playoff games listed under two "Plats a-b" rounds of the same stage with identical players
    and score. Keeps the round whose places lie inside the bracket's own range ("17-22 Plats 21-22", not
    "17-22 Plats 23-24"), else the first.
    """
    po = df[df["StageType"] == "playoff"]
    keys = ["SourceTournamentID", "SourceStageID", "Player1", "Player2", "GoalsPlayer1", "GoalsPlayer2", "PlayoffGameNumber"]
    dup = po[po.duplicated(keys, keep=False)]
    drop = []
    for _, group in dup.groupby(keys, dropna=False, sort=False):
        if group["Stage"].nunique() < 2:
            continue

        def fits(stage):
            ranges = _PLACE_RE.findall(str(stage))
            if len(ranges) < 2:
                return False
            (lo, hi), (a, b) = (map(int, ranges[0]), map(int, ranges[-1]))
            return lo <= a and b <= hi

        ordered = sorted(group.index, key=lambda i: (not fits(group.at[i, "Stage"]), str(group.at[i, "SourceMatchID"])))
        drop.extend(ordered[1:])
    return drop


def carried_copies(df: pd.DataFrame, min_share: float = 0.75, min_hits: int = 3) -> tuple[list, list[tuple]]:
    """
    Index labels of games in a later round-robin stage that repeat an earlier stage's game of the same pair
    (same players and score). A stage only counts when at least min_share of its pairs that met earlier have
    such a copy, which chance doesn't produce (about 10% of groups reach 20%; none reach 50%). One copy per
    pair is dropped, the earliest round.
    """
    rr = df[(df["StageType"] == "round-robin")].copy()
    rr["StageSequence"] = pd.to_numeric(rr["StageSequence"], errors="coerce")
    rr["RoundNumber"] = pd.to_numeric(rr["RoundNumber"], errors="coerce")
    p1 = rr["Player1"].fillna("").astype(str)
    p2 = rr["Player2"].fillna("").astype(str)
    swap = p1 > p2
    rr["_a"], rr["_b"] = p1.where(~swap, p2), p2.where(~swap, p1)
    g1 = pd.to_numeric(rr["GoalsPlayer1"], errors="coerce")
    g2 = pd.to_numeric(rr["GoalsPlayer2"], errors="coerce")
    rr["_ga"], rr["_gb"] = g1.where(~swap, g2), g2.where(~swap, g1)
    rr = rr[(rr["_a"] != "") & (rr["_b"] != "")]

    drop, report = [], []
    for tid, t in rr.groupby("SourceTournamentID", sort=False):
        if t["StageSequence"].nunique() < 2:
            continue
        for stage, g in t.groupby("Stage", sort=False):
            earlier = t[t["StageSequence"] < g["StageSequence"].iloc[0]]
            if earlier.empty:
                continue
            seen = set(zip(earlier["_a"], earlier["_b"], earlier["_ga"], earlier["_gb"]))
            met = set(zip(earlier["_a"], earlier["_b"]))
            pairs = [(k, grp) for k, grp in g.groupby(["_a", "_b"]) if k in met]
            hits = []
            for _, grp in pairs:
                same = [i for i, a, b, ga, gb in zip(grp.index, grp["_a"], grp["_b"], grp["_ga"], grp["_gb"]) if (a, b, ga, gb) in seen]
                if same:
                    hits.append(min(same, key=lambda i: (grp.at[i, "RoundNumber"], i)))
            if pairs and len(hits) >= min_hits and len(hits) / len(pairs) >= min_share:
                drop.extend(hits)
                report.append((tid, stage, len(pairs), len(hits)))
    return drop, report


# --- overtime ---

def impossible_overtime(df: pd.DataFrame) -> pd.Series:
    """Overtime games that are tied or won by 2+ goals: sudden death ends at the first goal."""
    diff = (pd.to_numeric(df["GoalsPlayer1"], errors="coerce") - pd.to_numeric(df["GoalsPlayer2"], errors="coerce")).abs()
    return (df["Overtime"].astype(str) == "Yes") & diff.notna() & (diff != 1)


# --- pipeline ---

def clean(df: pd.DataFrame, meta: pd.DataFrame, corrections: list[dict], verbose: bool = True) -> pd.DataFrame:
    def say(msg):
        if verbose:
            print(msg)

    out, applied = apply_source_corrections(df, corrections)
    say(f"Source corrections applied: {applied}")

    drop = placement_duplicates(out)
    say(f"Placement games listed twice: {len(drop)} dropped")
    out = out.drop(index=drop)

    drop, report = carried_copies(out)
    for tid, stage, pairs, hits in report:
        say(f"  {tid} {stage}: {hits} of {pairs} pairs repeat an earlier group game")
    say(f"Repeated group games: {len(drop)} dropped")
    out = out.drop(index=drop)

    bad_ot = impossible_overtime(out)
    say(f"Overtime flag cleared on {int(bad_ot.sum())} tied or 2+ goal games")
    out.loc[bad_ot, "Overtime"] = "No"

    out = add_tournament_meta(out, meta)
    if "Walkover" not in out.columns:
        out["Walkover"] = "No"
    out["Walkover"] = out["Walkover"].replace("", pd.NA).fillna("No")
    added = ["Country", "City", "Walkover"]
    out = out[[c for c in out.columns if c not in added] + added]
    return out.reset_index(drop=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Clean the results dataset (see module docstring).")
    parser.add_argument("--results", default=RESULTS_PATH)
    parser.add_argument("--meta", default=TOURNAMENT_META_CSV)
    parser.add_argument("--corrections", default=SOURCE_CORRECTIONS_FILE)
    args = parser.parse_args(argv)

    df = read_results(args.results, as_text=True)
    before = len(df)
    out = clean(df, load_tournament_meta(args.meta), load_source_corrections(args.corrections))
    write_results(args.results, out)
    print(f"Rows: {before} -> {len(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
