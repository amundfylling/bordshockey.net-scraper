"""
Assign Scorpion PlayerIDs to the players in bordshockey_results.csv.

Each player entry (one player name in one tournament) is matched in this order:

1. Manual override by ITHF id ("ithf:<id>" keys in manual_mapping.json).
2. The entry's ITHF id from the resultat API (api_player_ids.csv) to the Scorpion
   player with that RankingID.
3. The name mapping (manual by name, exact, accent-folded, then fuzzy), unless the Scorpion
   player it points to has a different RankingID than the entry's ITHF id.
4. Optionally, for players Scorpion does not have: ITHF_ONLY_ID_OFFSET + ITHF id.

Writes player_entry_map.csv (per entry), player_id_map.csv (per name),
unmapped_players.csv, and Player1ID/Player2ID in bordshockey_results.csv.
"""

import argparse
import difflib
import json
import os
import re
import sys
import unicodedata

import pandas as pd

# --- Configuration ---
RESULTS_CSV = "bordshockey_results.csv"
GITHUB_CSV = "players_data.csv"
API_IDS_CSV = "api_player_ids.csv"
MANUAL_MAPPING_FILE = "manual_mapping.json"
OUTPUT_MAPPING_CSV = "player_id_map.csv"
ENTRY_MAPPING_CSV = "player_entry_map.csv"
UNMAPPED_CSV = "unmapped_players.csv"
# Players with an ITHF id but no Scorpion record get PlayerID = offset + ITHF id.
# None leaves them unmapped. Scorpion PlayerIDs (< 20,000) and ITHF ids (< 1,000,000)
# keep this range apart.
ITHF_ONLY_ID_OFFSET = 1_000_000
# ---------------------

MANUAL_ITHF_PREFIX = "ithf:"


def load_manual_mapping(filepath):
    """Loads manual mapping from JSON file."""
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Could not decode {filepath}. Starting with empty manual map.", file=sys.stderr)
    return {}


def norm_name(name):
    return re.sub(r"\s+", " ", str(name).strip()).casefold()


def fold_name(name):
    """Name without accents, case or punctuation (as the h2h site's unique-name lookup)."""
    text = unicodedata.normalize("NFKD", str(name).strip())
    text = "".join(ch for ch in text if not unicodedata.combining(ch)).casefold()
    return re.sub(r"[\W_]+", " ", text).strip()


def _id_text(value):
    """'672107', 672107 and 672107.0 all become '672107'; blanks become None."""
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return str(int(float(text)))
    except ValueError:
        return text


def map_names(names, github_df, manual_map):
    """The name-only mapping: {name: (PlayerID or None, method, matched name)} for mapped names."""
    # Dictionary: {lowercase_name: PlayerID}
    # Note: If multiple players have same name, this simple approach takes the last one encountered.
    # The ITHF id step in map_entries takes precedence where the API knows the player.
    github_lookup = {}
    github_names_original = [] # For fuzzy matching, keep original casing
    folded_lookup = {} # {folded_name: {PlayerID, ...}}

    for _, row in github_df.iterrows():
        name = str(row['Name']).strip()
        pid = row['PlayerID']
        if name and name.lower() != "nan":
            github_lookup[name.lower()] = pid
            github_names_original.append(name)
            folded_lookup.setdefault(fold_name(name), set()).add(pid)

    mapped = {}
    for scraped_name in names:
        scraped_lower = scraped_name.lower()

        # A. Manual Mapping
        if scraped_name in manual_map:
            mapped[scraped_name] = (manual_map[scraped_name], "Manual", scraped_name)

        # B. Exact Match (Case-Insensitive)
        elif scraped_lower in github_lookup:
            mapped[scraped_name] = (github_lookup[scraped_lower], "Exact", scraped_name)

        # C. Same name without accents/punctuation, if only one player has it
        elif len(folded_lookup.get(fold_name(scraped_name), ())) == 1:
            pid = next(iter(folded_lookup[fold_name(scraped_name)]))
            mapped[scraped_name] = (pid, "Folded", scraped_name)

        # D. Fuzzy Match
        else:
            # cutoff=0.90 is a strict threshold to avoid false positives
            matches = difflib.get_close_matches(scraped_name, github_names_original, n=1, cutoff=0.90)
            if matches:
                best_match = matches[0]
                mapped[scraped_name] = (github_lookup[best_match.lower()], "Fuzzy", best_match)
    return mapped


def player_entries(results_df):
    """Unique (SourceTournamentID, PlayerName) with how many match rows each appears in."""
    sides = [
        results_df[["SourceTournamentID", col]].set_axis(["SourceTournamentID", "PlayerName"], axis=1)
        for col in ("Player1", "Player2")
    ]
    entries = pd.concat(sides, ignore_index=True)
    entries["PlayerName"] = entries["PlayerName"].fillna("").astype(str).str.strip()
    entries = entries[entries["PlayerName"] != ""]
    return entries.groupby(["SourceTournamentID", "PlayerName"]).size().rename("Rows").reset_index()


def attach_api_ids(entries, api_df):
    """Adds ITHF/SBHF from the tournament's API participant with the same name; ambiguous names get none."""
    lookup = {}
    for row in api_df.itertuples(index=False):
        key = (row.SourceTournamentID, norm_name(row.PlayerName))
        lookup.setdefault(key, set()).add((_id_text(row.ITHF), _id_text(row.SBHF)))

    ithf, sbhf = [], []
    for row in entries.itertuples(index=False):
        found = lookup.get((row.SourceTournamentID, norm_name(row.PlayerName)), set())
        ids = next(iter(found)) if len(found) == 1 else (None, None)
        ithf.append(ids[0])
        sbhf.append(ids[1])
    out = entries.copy()
    out["ITHF"] = ithf
    out["SBHF"] = sbhf
    return out


def map_entries(entries, github_df, name_map, manual_map, ithf_only_offset=ITHF_ONLY_ID_OFFSET):
    """Adds PlayerID and Method to each entry (see module docstring for the order)."""
    ranked = github_df.assign(RankingKey=github_df["RankingID"].map(_id_text)).dropna(subset=["RankingKey"])
    # A RankingID held by two Scorpion players identifies neither.
    ranked = ranked.drop_duplicates("RankingKey", keep=False)
    ranking_to_pid = dict(zip(ranked["RankingKey"], ranked["PlayerID"]))
    pid_to_ranking = {
        _id_text(pid): _id_text(rk)
        for pid, rk in zip(github_df["PlayerID"], github_df["RankingID"])
        if _id_text(rk)
    }
    manual_ithf = {
        key[len(MANUAL_ITHF_PREFIX):].strip(): pid
        for key, pid in manual_map.items()
        if key.startswith(MANUAL_ITHF_PREFIX)
    }

    pids, methods = [], []
    for row in entries.itertuples(index=False):
        ithf = _id_text(row.ITHF)
        pid, method = None, ""
        if ithf and ithf in manual_ithf:
            pid, method = manual_ithf[ithf], "Manual"
        elif ithf and ithf in ranking_to_pid:
            pid, method = ranking_to_pid[ithf], "ITHF"
        elif row.PlayerName in name_map and name_map[row.PlayerName][0] is not None:
            name_pid, name_method, _ = name_map[row.PlayerName]
            ranking = pid_to_ranking.get(_id_text(name_pid))
            if ithf and ranking and ranking != ithf:
                method = "Conflict"
            else:
                pid, method = name_pid, name_method
        if pid is None and ithf and ithf_only_offset is not None:
            pid, method = ithf_only_offset + int(ithf), "ITHF-only"
        pids.append(pid)
        methods.append(method)

    out = entries.copy()
    out["PlayerID"] = pd.array([_id_text(p) for p in pids], dtype="string").astype("Int64")
    out["Method"] = methods

    # A name-matched Scorpion player without RankingID should be one ITHF player.
    # If entries with different ITHF ids landed on it, keep the id with the most rows.
    by_name = out[out["Method"].isin(["Manual", "Exact", "Folded", "Fuzzy"]) & out["ITHF"].notna()]
    for pid, group in by_name.groupby("PlayerID"):
        if group["ITHF"].nunique() < 2:
            continue
        keep = group.groupby("ITHF")["Rows"].sum().idxmax()
        drop = group.index[group["ITHF"] != keep]
        out.loc[drop, "PlayerID"] = (
            pd.NA if ithf_only_offset is None else out.loc[drop, "ITHF"].astype(int) + ithf_only_offset
        )
        out.loc[drop, "Method"] = "Conflict" if ithf_only_offset is None else "ITHF-only"
    return out


def apply_entry_map(results_df, entry_map):
    """Sets Player1ID/Player2ID from the entry map; entries it leaves unmapped get no id."""
    lookup = {
        (row.SourceTournamentID, row.PlayerName): row.PlayerID
        for row in entry_map.itertuples(index=False)
        if not pd.isna(row.PlayerID)
    }
    out = results_df.copy()
    for player_col, id_col in (("Player1", "Player1ID"), ("Player2", "Player2ID")):
        keys = zip(out["SourceTournamentID"], out[player_col].fillna("").astype(str).str.strip())
        out[id_col] = [str(int(lookup[k])) if k in lookup else "" for k in keys]
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description="Map bordshockey.net players to Scorpion PlayerIDs.")
    parser.add_argument("--results", default=RESULTS_CSV)
    parser.add_argument("--players", default=GITHUB_CSV)
    parser.add_argument("--api-ids", default=API_IDS_CSV)
    parser.add_argument("--no-write-results", action="store_true", help="Only write the mapping files.")
    args = parser.parse_args(argv)

    # 1. Validation
    for path in (args.results, args.players):
        if not os.path.exists(path):
            print(f"Error: {path} not found.", file=sys.stderr)
            return 1

    print("Loading data...")
    # Read everything as text so the rewrite only touches the id columns.
    results_df = pd.read_csv(args.results, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    github_df = pd.read_csv(args.players, encoding="utf-8-sig")
    api_df = (
        pd.read_csv(args.api_ids, encoding="utf-8-sig", dtype=str)
        if os.path.exists(args.api_ids)
        else pd.DataFrame(columns=["SourceTournamentID", "PlayerName", "ITHF", "SBHF"])
    )
    if api_df.empty:
        print(f"Warning: no API ids in {args.api_ids}; matching by name only.", file=sys.stderr)
    manual_map = load_manual_mapping(MANUAL_MAPPING_FILE)

    # 2. Name mapping
    entries = player_entries(results_df)
    names = sorted(entries["PlayerName"].unique())
    print(f"Mapping {len(names)} player names in {len(entries)} tournament entries...")
    name_map = map_names(names, github_df, manual_map)

    # 3. Entry mapping
    entries = attach_api_ids(entries, api_df)
    entry_map = map_entries(entries, github_df, name_map, manual_map)

    # 4. Save Results
    entry_map.to_csv(ENTRY_MAPPING_CSV, index=False, encoding="utf-8-sig")

    name_rows = [
        {"ScrapedName": name, "PlayerID": pid, "Method": method, "MatchedName": matched}
        for name, (pid, method, matched) in name_map.items()
    ]
    pd.DataFrame(name_rows, columns=["ScrapedName", "PlayerID", "Method", "MatchedName"]).to_csv(
        OUTPUT_MAPPING_CSV, index=False, encoding="utf-8-sig"
    )

    unmapped = sorted(entry_map.loc[entry_map["PlayerID"].isna(), "PlayerName"].unique())
    with open(UNMAPPED_CSV, 'w', encoding='utf-8') as f:
        f.write("PlayerName\n")
        for name in unmapped:
            f.write(f"{name}\n")

    if not args.no_write_results:
        apply_entry_map(results_df, entry_map).to_csv(args.results, index=False, encoding="utf-8-sig")

    # 5. Summary
    total = len(entry_map)
    mapped = entry_map["PlayerID"].notna()
    print("-" * 30)
    print(f"Tournament entries: {total}, with ITHF id: {entry_map['ITHF'].notna().sum()}")
    print(f"Mapped: {mapped.sum()} ({mapped.sum()/total:.1%})")
    for method, count in entry_map.loc[mapped, "Method"].value_counts().items():
        print(f"  - {method}: {count}")
    print(f"Rejected name matches (ITHF id conflicts): {(entry_map['Method'] == 'Conflict').sum()}")
    print("-" * 30)
    print(f"Entry mapping saved to: {ENTRY_MAPPING_CSV}")
    print(f"Unmapped list saved to: {UNMAPPED_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
