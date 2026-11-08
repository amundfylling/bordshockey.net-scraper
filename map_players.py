"""
Assign player ids to the players in data/bordshockey_results.parquet.

Every player entry (one player name in one tournament) gets an id:

- a Sportscorpion PlayerID when the player is in Sportscorpion,
- else 9,000,000 + ITHF id when the player has an ITHF id,
- else a local id from 10,000,001 up, kept in local_player_ids.csv so it never
  changes or gets reused.

The ITHF id of an entry comes from the resultat API participant with that name in
that tournament (api_player_ids.csv), corrected by ithf_corrections.csv. Entries
without one borrow it from the same SBHF id in other tournaments, or from the same
name (accents and word order folded) when only one ITHF id ever has that name, the
nations agree and the seasons are close.

Entries are matched to Sportscorpion in this order:

1. Manual override by ITHF id ("ithf:<id>" keys in manual_mapping.json).
2. ITHF id = Sportscorpion RankingID.
3. The name mapping (manual by name, exact, accent-folded, then fuzzy), unless the
   Sportscorpion player it points to has a different RankingID than the entry's ITHF id.
4. A Sportscorpion name that is unique after folding accents and word order, with the
   same nation when both sides have one, whose RankingID (if any) is not someone
   else's ITHF id in the data.
5. ITHF_ONLY_ID_OFFSET + ITHF id, then a local id.

Two entries of one tournament never share an id: the weaker match falls back to its
next option. row_player_overrides.csv sets per-row ids where two participants of a
tournament share a name.

Writes player_entry_map.csv (per entry), player_id_map.csv (per name),
unmapped_players.csv, local_player_ids.csv, Player1ID/Player2ID in the results, and
appends ids that moved entirely to data/id_changes.csv.
"""

import argparse
import difflib
import json
import os
import re
import sys
import unicodedata

import numpy as np
import pandas as pd

from results_store import RESULTS_PATH, read_results, results_exist, write_results

# --- Configuration ---
RESULTS_CSV = RESULTS_PATH
GITHUB_CSV = "players_data.csv"
API_IDS_CSV = "api_player_ids.csv"
MANUAL_MAPPING_FILE = "manual_mapping.json"
OUTPUT_MAPPING_CSV = "player_id_map.csv"
ENTRY_MAPPING_CSV = "player_entry_map.csv"
UNMAPPED_CSV = "unmapped_players.csv"
# Players with an ITHF id but no Scorpion record get PlayerID = offset + ITHF id.
# None leaves them unmapped. 9,000,000 is the rule shared with table-hockey.eu and the
# table-hockey-scrapers repo, so the same player gets the same id in every dataset.
# Ids below 9,000,000 are real Scorpion PlayerIDs.
ITHF_ONLY_ID_OFFSET = 9_000_000
# Players with neither a Sportscorpion record nor an ITHF id get a local id from this range.
LOCAL_ID_START = 10_000_001
LOCAL_IDS_CSV = "local_player_ids.csv"
ITHF_CORRECTIONS_CSV = "ithf_corrections.csv"
ROW_OVERRIDES_CSV = "row_player_overrides.csv"
ROW_OVERRIDE_COLUMNS = ["SourceTournamentID", "RowKey", "Side", "PlayerName", "ITHF", "SBHF", "PlayerID", "Reason"]
ID_CHANGES_CSV = os.path.join("data", "id_changes.csv")
# An entry borrows an ITHF id by name only from players who played within this many seasons of it.
NAME_ITHF_MAX_SEASON_GAP = 5
FUZZY_CUTOFF = 0.90
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


# Letters NFKD doesn't split into a base letter and an accent.
_FOLD_LETTERS = str.maketrans({"ø": "o", "Ø": "O", "æ": "ae", "Æ": "AE", "ß": "ss", "ł": "l", "Ł": "L", "đ": "d", "Đ": "D", "œ": "oe", "Œ": "OE", "þ": "th", "Þ": "Th"})


def fold_name(name):
    """Name without accents, case or punctuation (as the h2h site's unique-name lookup)."""
    text = unicodedata.normalize("NFKD", str(name).strip().translate(_FOLD_LETTERS))
    text = "".join(ch for ch in text if not unicodedata.combining(ch)).casefold()
    return re.sub(r"[\W_]+", " ", text).strip()


def name_key(name):
    """fold_name with the words sorted, so "Gunnarsson Fredrik" and "Fredrik Gunnarsson" are one key."""
    return " ".join(sorted(fold_name(name).split()))


def short_name_key(name):
    """name_key of the first and last word only, so "Trond Gangsoy" meets "Trond Ove Gangsøy"."""
    words = fold_name(name).split()
    return " ".join(sorted([words[0], words[-1]])) if len(words) >= 2 else " ".join(words)


# Names that stand for an empty slot, not a player. They get no id.
PLACEHOLDER_NAMES = {"", "struken", "strucken", "walkover", "wo", "w o", "vakant", "bye", "frirond", "reserv", "okand", "unknown"}


def is_placeholder(name):
    folded = fold_name(name)
    return folded in PLACEHOLDER_NAMES or folded.startswith("reserv ")


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


class FuzzyMatcher:
    """
    Exact drop-in for difflib.get_close_matches(name, names, n=1, cutoff=cutoff), but faster.

    get_close_matches skips any candidate whose real_quick_ratio() or quick_ratio() (both upper
    bounds on ratio(), computed from lengths and shared character counts) is below the cutoff.
    We compute those same bounds for all candidates at once with numpy, then let
    get_close_matches judge only the survivors. Duplicate names are dropped; they can't change
    the best match.
    """

    def __init__(self, names, cutoff=FUZZY_CUTOFF):
        self.cutoff = cutoff
        self.names = list(dict.fromkeys(names))
        alphabet = sorted({ch for name in self.names for ch in name})
        self.char_index = {ch: i for i, ch in enumerate(alphabet)}
        self.counts = np.zeros((len(self.names), len(alphabet)), dtype=np.int32)
        for row, name in enumerate(self.names):
            for ch in name:
                self.counts[row, self.char_index[ch]] += 1
        self.lengths = np.array([len(name) for name in self.names], dtype=np.int64)

    def match(self, name):
        query = np.zeros(self.counts.shape[1], dtype=np.int32)
        for ch in name:
            i = self.char_index.get(ch)
            if i is not None:  # characters no candidate has can't contribute matches
                query[i] += 1
        totals = self.lengths + len(name)
        # Same formulas as difflib's real_quick_ratio() and quick_ratio().
        with np.errstate(divide="ignore", invalid="ignore"):
            real_quick = np.where(totals > 0, 2.0 * np.minimum(self.lengths, len(name)) / totals, 1.0)
            quick = np.where(totals > 0, 2.0 * np.minimum(self.counts, query).sum(axis=1) / totals, 1.0)
        keep = np.flatnonzero((real_quick >= self.cutoff) & (quick >= self.cutoff))
        candidates = [self.names[i] for i in keep]
        return difflib.get_close_matches(name, candidates, n=1, cutoff=self.cutoff)


def map_names(names, github_df, manual_map):
    """The name-only mapping: {name: (PlayerID or None, method, matched name)} for mapped names."""
    # Dictionary: {lowercase_name: PlayerID}
    # Note: If multiple players have same name, this simple approach takes the last one encountered.
    # The ITHF id step in map_entries takes precedence where the API knows the player.
    github_lookup = {}
    github_names_original = [] # For fuzzy matching, keep original casing
    folded_lookup = {} # {folded_name: {PlayerID, ...}}

    for name, pid in zip(github_df['Name'], github_df['PlayerID']):
        name = str(name).strip()
        if name and name.lower() != "nan":
            github_lookup[name.lower()] = pid
            github_names_original.append(name)
            folded_lookup.setdefault(fold_name(name), set()).add(pid)

    fuzzy = FuzzyMatcher(github_names_original)

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
            # Same result as difflib.get_close_matches(..., n=1, cutoff=0.90), see FuzzyMatcher
            # cutoff=0.90 is a strict threshold to avoid false positives
            matches = fuzzy.match(scraped_name)
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
    """Adds ITHF/SBHF/Nation from the tournament's API participant with the same name; ambiguous names get none."""
    has_nation = "Nation" in api_df.columns
    lookup = {}
    for row in api_df.itertuples(index=False):
        key = (row.SourceTournamentID, norm_name(row.PlayerName))
        nation = getattr(row, "Nation", None) if has_nation else None
        nation = None if nation is None or pd.isna(nation) else str(nation)
        lookup.setdefault(key, set()).add((_id_text(row.ITHF), _id_text(row.SBHF), nation))

    ithf, sbhf, nations = [], [], []
    for row in entries.itertuples(index=False):
        found = lookup.get((row.SourceTournamentID, norm_name(row.PlayerName)), set())
        ids = next(iter(found)) if len(found) == 1 else (None, None, None)
        ithf.append(ids[0])
        sbhf.append(ids[1])
        nations.append(ids[2])
    out = entries.copy()
    out["ITHF"] = ithf
    out["SBHF"] = sbhf
    out["Nation"] = nations
    return out


# --- nations ---

_SCORPION_COUNTRY_ALIASES = {
    "republic of lithuania": "lithuania",
    "united states of america": "united states",
    "россия": "russia",
    "росиия": "russia",
    "казакстан": "kazakhstan",
    "kazakistan": "kazakhstan",
    "republic of moldova": "moldova",
}


def _country(value):
    """Country name or IOC code -> lower-case English country name, None when unknown."""
    from clean_results import country_name

    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text in ("-", "???"):
        return None
    if len(text) == 3 and text.isalpha():
        text = country_name(text) or text
    text = text.casefold()
    return _SCORPION_COUNTRY_ALIASES.get(text, text)


def _same_nation(a, b):
    a, b = _country(a), _country(b)
    return a is None or b is None or a == b


def _season_year(source_tournament_id):
    from clean_results import season_label

    label = season_label(source_tournament_id)
    return int(label[:4]) if label else None


# --- ITHF ids for entries the API gives none ---

def load_ithf_corrections(path=ITHF_CORRECTIONS_CSV):
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    return {(r.SourceTournamentID, r.PlayerName): (_id_text(r.ITHF) or None) for r in df.itertuples(index=False)}


def fill_ithf(entries, manual_map=None, corrections=None):
    """
    Adds ITHFSource and fills ITHF where the API gives none (see module docstring). Corrections replace the
    API's ITHF id for one entry (an empty ITHF there removes it).
    """
    manual_map = manual_map or {}
    out = entries.copy()
    out["ITHF"] = out["ITHF"].map(_id_text)
    out["SBHF"] = out["SBHF"].map(_id_text)
    if "Nation" not in out.columns:
        out["Nation"] = None
    source = pd.Series(["API" if _id_text(i) else "" for i in out["ITHF"]], index=out.index, dtype=object)

    for (tid, name), ithf in (corrections or {}).items():
        hit = (out["SourceTournamentID"] == tid) & (out["PlayerName"] == name)
        out.loc[hit, "ITHF"] = ithf
        source[hit] = "Correction"

    # Same SBHF id in other tournaments.
    pairs = out.dropna(subset=["ITHF", "SBHF"]).groupby("SBHF")["ITHF"].agg(lambda s: set(s))
    sbhf_ithf = {k: next(iter(v)) for k, v in pairs.items() if len(v) == 1}
    fill = out["ITHF"].isna() & out["SBHF"].isin(sbhf_ithf.keys())
    out.loc[fill, "ITHF"] = out.loc[fill, "SBHF"].map(sbhf_ithf)
    source[fill] = "SBHF"

    # Same name, when one ITHF id has it; then the same for SBHF ids, so namesakes without any id share one.
    keys = out["PlayerName"].map(name_key)
    short = out["PlayerName"].map(short_name_key)
    years = out["SourceTournamentID"].map(_season_year)
    ambiguous = {norm_name(k) for k, v in manual_map.items() if v is None and not str(k).startswith(MANUAL_ITHF_PREFIX)}
    for col, label in (("ITHF", "Name"), ("SBHF", "Name")):
        known = out[col].notna()
        holders, short_holders = {}, {}
        for key, skey, value, nation, year in zip(keys[known], short[known], out.loc[known, col], out.loc[known, "Nation"], years[known]):
            for index, k in ((holders, key), (short_holders, skey)):
                info = index.setdefault(k, {}).setdefault(value, {"nations": set(), "years": []})
                if _country(nation):
                    info["nations"].add(_country(nation))
                if year is not None:
                    info["years"].append(year)
        taken = set(zip(out.loc[known, "SourceTournamentID"], out.loc[known, col]))
        todo = out.index[~known & out["ITHF"].isna()] if col == "SBHF" else out.index[~known]
        for i in todo:
            name = out.at[i, "PlayerName"]
            if norm_name(name) in ambiguous:
                continue
            h = holders.get(keys[i]) or short_holders.get(short[i], {})
            if len(h) != 1:
                continue
            value, info = next(iter(h.items()))
            nation = _country(out.at[i, "Nation"])
            if nation and info["nations"] and nation not in info["nations"]:
                continue
            year = years[i]
            if year is None or not info["years"] or min(abs(year - y) for y in info["years"]) > NAME_ITHF_MAX_SEASON_GAP:
                continue
            if (out.at[i, "SourceTournamentID"], value) in taken:
                continue
            out.at[i, col] = value
            if col == "ITHF":
                source[i] = label
    out["ITHFSource"] = source
    return out


# --- local ids ---

class LocalIds:
    """Stable ids for players with neither a Sportscorpion record nor an ITHF id. Ids are never reused."""

    def __init__(self, path=LOCAL_IDS_CSV):
        self.path = path
        self.ids = {}
        self.names = {}
        self.next_id = LOCAL_ID_START
        if path and os.path.exists(path):
            df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
            for r in df.itertuples(index=False):
                self.ids[r.Key] = int(r.PlayerID)
                self.names[r.Key] = r.Name
            if self.ids:
                self.next_id = max(max(self.ids.values()) + 1, LOCAL_ID_START)

    @staticmethod
    def key(sbhf, name, nation=None):
        if sbhf:
            return f"sbhf:{sbhf}"
        return f"name:{name_key(name)}"

    def get(self, key, name):
        if key not in self.ids:
            self.ids[key] = self.next_id
            self.names[key] = name
            self.next_id += 1
        return self.ids[key]

    def save(self):
        if not self.path:
            return
        rows = sorted(self.ids.items(), key=lambda kv: kv[1])
        pd.DataFrame(
            [{"Key": k, "PlayerID": v, "Name": self.names.get(k, "")} for k, v in rows], columns=["Key", "PlayerID", "Name"]
        ).to_csv(self.path, index=False, encoding="utf-8-sig")


# --- matching ---

# Lower ranks win when two entries of one tournament land on the same id.
METHOD_RANK = {"Manual": 0, "ITHF": 1, "Exact": 2, "Folded": 2, "ScorpionName": 3, "Fuzzy": 4, "ITHF-only": 5, "Local": 6}
NAME_METHODS = ["Manual", "Exact", "Folded", "Fuzzy", "ScorpionName"]


class Matcher:
    def __init__(self, github_df, entries=None, ithf_only_offset=ITHF_ONLY_ID_OFFSET):
        self.offset = ithf_only_offset
        ranked = github_df.assign(RankingKey=github_df["RankingID"].map(_id_text)).dropna(subset=["RankingKey"])
        # A RankingID held by two Scorpion players identifies neither.
        ranked = ranked.drop_duplicates("RankingKey", keep=False)
        self.ranking_to_pid = dict(zip(ranked["RankingKey"], ranked["PlayerID"]))
        self.pid_to_ranking = {
            _id_text(pid): _id_text(rk) for pid, rk in zip(github_df["PlayerID"], github_df["RankingID"]) if _id_text(rk)
        }
        countries = github_df["Country"] if "Country" in github_df.columns else pd.Series(None, index=github_df.index)
        self.by_key = {}
        for pid, name, rk, country in zip(github_df["PlayerID"], github_df["Name"], github_df["RankingID"], countries):
            if pd.isna(name) or not str(name).strip():
                continue
            self.by_key.setdefault(name_key(name), []).append((pid, _id_text(rk), country))
        # Name keys each ITHF id is seen with, to tell whether a RankingID belongs to someone else.
        self.ithf_keys = {}
        if entries is not None:
            for ithf, name in zip(entries["ITHF"], entries["PlayerName"]):
                if _id_text(ithf):
                    self.ithf_keys.setdefault(_id_text(ithf), set()).add(name_key(name))

    def scorpion_by_name(self, name, nation, ithf=None):
        """
        The one Scorpion player with this name (accents and word order folded) and a compatible nation. Its
        RankingID must not be another player's ITHF id, nor, for an entry with an ITHF id, a different ITHF id
        seen in the data (two namesakes with their own ITHF ids).
        """
        key = name_key(name)
        cands = [c for c in self.by_key.get(key, []) if _same_nation(c[2], nation)]
        if len(self.by_key.get(key, [])) != 1 or len(cands) != 1:
            return None
        pid, ranking, _ = cands[0]
        if ranking and not self.ithf_keys.get(ranking, set()) <= {key}:
            return None  # its RankingID is another player's ITHF id
        if ithf and ranking and ranking != ithf and ranking in self.ithf_keys:
            return None
        return pid

    def by_ithf(self, ithf, name=None, nation=None):
        """Scorpion id for an ITHF id (RankingID, else unique name), else offset + ITHF; method too."""
        if ithf in self.ranking_to_pid:
            return self.ranking_to_pid[ithf], "ITHF"
        if name is not None:
            pid = self.scorpion_by_name(name, nation, ithf)
            if pid is not None:
                return pid, "ScorpionName"
        if self.offset is not None:
            return self.offset + int(ithf), "ITHF-only"
        return None, ""


def map_entries(entries, github_df, name_map, manual_map, ithf_only_offset=ITHF_ONLY_ID_OFFSET, local_ids=None):
    """Adds PlayerID and Method to each entry (see module docstring for the order)."""
    entries = entries.copy()
    if "Nation" not in entries.columns:
        entries["Nation"] = None
    matcher = Matcher(github_df, entries, ithf_only_offset)
    manual_ithf = {
        key[len(MANUAL_ITHF_PREFIX):].strip(): pid
        for key, pid in manual_map.items()
        if key.startswith(MANUAL_ITHF_PREFIX)
    }

    # One SBHF id with two ITHF ids is one player the source has twice; the ITHF id Scorpion knows wins.
    sbhf_ranked = {}
    for sbhf, group in entries.dropna(subset=["SBHF", "ITHF"]).groupby("SBHF"):
        ranked = {i for i in group["ITHF"].map(_id_text) if i in matcher.ranking_to_pid}
        if len(ranked) == 1:
            sbhf_ranked[_id_text(sbhf)] = next(iter(ranked))

    pids, methods = [], []
    for row in entries.itertuples(index=False):
        ithf = _id_text(row.ITHF)
        pid, method = None, ""
        if ithf and ithf not in matcher.ranking_to_pid and _id_text(row.SBHF) in sbhf_ranked:
            ithf = sbhf_ranked[_id_text(row.SBHF)]
        if is_placeholder(row.PlayerName):
            method = "Placeholder"
        elif ithf and ithf in manual_ithf:
            pid, method = manual_ithf[ithf], "Manual"
        elif ithf and ithf in matcher.ranking_to_pid:
            pid, method = matcher.ranking_to_pid[ithf], "ITHF"
        elif row.PlayerName in name_map and name_map[row.PlayerName][0] is not None:
            name_pid, name_method, _ = name_map[row.PlayerName]
            ranking = matcher.pid_to_ranking.get(_id_text(name_pid))
            if ithf and ranking and ranking != ithf:
                method = "Conflict"
            else:
                pid, method = name_pid, name_method
        if pid is None and method != "Placeholder":
            scorpion = matcher.scorpion_by_name(row.PlayerName, row.Nation, ithf)
            if scorpion is not None:
                pid, method = scorpion, "ScorpionName"
        if pid is None and ithf and ithf_only_offset is not None and method != "Placeholder":
            pid, method = ithf_only_offset + int(ithf), "ITHF-only"
        if pid is None and local_ids is not None and method != "Placeholder":
            pid, method = local_ids.get(LocalIds.key(_id_text(row.SBHF), row.PlayerName, row.Nation), row.PlayerName), "Local"
        pids.append(pid)
        methods.append(method)

    out = entries.copy()
    out["PlayerID"] = pd.array([_id_text(p) for p in pids], dtype="string").astype("Int64")
    out["Method"] = methods

    def fallback(i):
        """Next id for an entry whose match was rejected."""
        ithf = _id_text(out.at[i, "ITHF"])
        if ithf and ithf_only_offset is not None and out.at[i, "Method"] != "ITHF-only":
            return ithf_only_offset + int(ithf), "ITHF-only"
        if local_ids is not None and out.at[i, "Method"] != "Local":
            key = LocalIds.key(_id_text(out.at[i, "SBHF"]), out.at[i, "PlayerName"], out.at[i, "Nation"])
            return local_ids.get(key, out.at[i, "PlayerName"]), "Local"
        return pd.NA, "Conflict"

    # A name-matched Scorpion player without RankingID should be one ITHF player.
    # If entries with different ITHF ids landed on it, keep the id with the most rows.
    by_name = out[out["Method"].isin(NAME_METHODS) & out["ITHF"].notna()]
    for pid, group in by_name.groupby("PlayerID"):
        if group["ITHF"].nunique() < 2:
            continue
        keep = group.groupby("ITHF")["Rows"].sum().idxmax()
        for i in group.index[group["ITHF"] != keep]:
            out.at[i, "PlayerID"], out.at[i, "Method"] = fallback(i)

    # Two entries (names) of one tournament never share an id: they play each other.
    for _ in range(3):
        clash = out[out["PlayerID"].notna() & out.duplicated(["SourceTournamentID", "PlayerID"], keep=False)]
        if clash.empty:
            break
        for _, group in clash.groupby(["SourceTournamentID", "PlayerID"]):
            order = sorted(group.index, key=lambda i: (METHOD_RANK.get(out.at[i, "Method"], 9), -int(out.at[i, "Rows"])))
            for i in order[1:]:
                out.at[i, "PlayerID"], out.at[i, "Method"] = fallback(i)
    return out


# --- applying ids to the results ---

def row_keys(results_df):
    """A key per results row that survives refetches: the source match id, else the match tuple."""
    smid = results_df["SourceMatchID"].fillna("").astype(str).str.strip() if "SourceMatchID" in results_df else ""
    cols = ["Stage", "RoundNumber", "PlayoffGameNumber", "Player1", "Player2", "GoalsPlayer1", "GoalsPlayer2"]
    tuple_key = results_df[cols].astype("string").fillna("").astype(str).agg("|".join, axis=1)
    return ("m:" + smid).where(smid != "", "t:" + tuple_key)


def load_row_overrides(path=ROW_OVERRIDES_CSV):
    if not os.path.exists(path):
        return pd.DataFrame(columns=ROW_OVERRIDE_COLUMNS)
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def resolve_overrides(overrides, matcher, sbhf_ithf=None, local_ids=None):
    """{(SourceTournamentID, RowKey, Side): PlayerID} from the override rows."""
    sbhf_ithf = sbhf_ithf or {}
    out = {}
    for r in overrides.itertuples(index=False):
        pid = _id_text(r.PlayerID)
        ithf = _id_text(r.ITHF) or sbhf_ithf.get(_id_text(r.SBHF))
        if pid:
            pid = int(pid)
        elif ithf:
            pid, _ = matcher.by_ithf(ithf, r.PlayerName)
        elif _id_text(r.SBHF) and local_ids is not None:
            pid = local_ids.get(LocalIds.key(_id_text(r.SBHF), r.PlayerName, None), r.PlayerName)
        if pid is not None:
            out[(r.SourceTournamentID, r.RowKey, int(r.Side))] = int(pid)
    return out


def apply_entry_map(results_df, entry_map, row_ids=None):
    """Sets Player1ID/Player2ID from the entry map, then per-row overrides; entries left unmapped get no id."""
    lookup = {
        (row.SourceTournamentID, row.PlayerName): row.PlayerID
        for row in entry_map.itertuples(index=False)
        if not pd.isna(row.PlayerID)
    }
    out = results_df.copy()
    keys = row_keys(out) if row_ids else None
    for side, (player_col, id_col) in enumerate((("Player1", "Player1ID"), ("Player2", "Player2ID")), start=1):
        names = zip(out["SourceTournamentID"], out[player_col].fillna("").astype(str).str.strip())
        ids = [str(int(lookup[k])) if k in lookup else "" for k in names]
        if row_ids:
            for n, (tid, rk) in enumerate(zip(out["SourceTournamentID"], keys)):
                pid = row_ids.get((tid, rk, side))
                if pid is not None:
                    ids[n] = str(pid)
        out[id_col] = ids
    return out


def id_changes(old_df, new_df):
    """
    (old_id, new_id) for every id whose rows all moved to one other id, and per-entry detail of every move.
    Rows that had no id before are not changes.
    """
    sides = []
    for col, id_col in (("Player1", "Player1ID"), ("Player2", "Player2ID")):
        sides.append(
            pd.DataFrame(
                {
                    "SourceTournamentID": new_df["SourceTournamentID"].astype(str).values,
                    "PlayerName": new_df[col].astype(str).values,
                    "Old": pd.to_numeric(old_df[id_col].replace("", pd.NA), errors="coerce").values,
                    "New": pd.to_numeric(new_df[id_col].replace("", pd.NA), errors="coerce").values,
                }
            )
        )
    s = pd.concat(sides).dropna(subset=["Old"])
    s["Old"] = s["Old"].astype("int64")
    s["New"] = s["New"].astype("Int64")
    moved = s[s["New"].isna() | (s["Old"] != s["New"])]
    detail = (
        moved.groupby(["Old", "New", "PlayerName", "SourceTournamentID"], dropna=False).size().rename("Slots").reset_index()
    )
    whole = []
    for old, grp in s.groupby("Old"):
        news = set(grp["New"].dropna().tolist())
        if len(news) == 1 and old not in news and grp["New"].notna().all():
            whole.append((int(old), int(next(iter(news)))))
    return pd.DataFrame(whole, columns=["old_id", "new_id"]), detail


def append_id_changes(changes, path=ID_CHANGES_CSV):
    if changes.empty:
        return 0
    old = pd.read_csv(path, dtype="int64") if os.path.exists(path) else pd.DataFrame(columns=["old_id", "new_id"])
    merged = pd.concat([old, changes], ignore_index=True).drop_duplicates()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    merged.to_csv(path, index=False)
    return len(merged) - len(old)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Map bordshockey.net players to Scorpion PlayerIDs.")
    parser.add_argument("--results", default=RESULTS_CSV)
    parser.add_argument("--players", default=GITHUB_CSV)
    parser.add_argument("--api-ids", default=API_IDS_CSV)
    parser.add_argument("--no-write-results", action="store_true", help="Only write the mapping files.")
    args = parser.parse_args(argv)

    # 1. Validation
    for path, exists in ((args.results, results_exist(args.results)), (args.players, os.path.exists(args.players))):
        if not exists:
            print(f"Error: {path} not found.", file=sys.stderr)
            return 1

    print("Loading data...")
    # Read everything as text so the rewrite only touches the id columns.
    results_df = read_results(args.results, as_text=True)
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
    entries = fill_ithf(entries, manual_map, load_ithf_corrections())
    local_ids = LocalIds()
    entry_map = map_entries(entries, github_df, name_map, manual_map, local_ids=local_ids)
    matcher = Matcher(github_df, entry_map)
    sbhf_pairs = entry_map.dropna(subset=["ITHF", "SBHF"]).groupby("SBHF")["ITHF"].agg(lambda x: set(x))
    sbhf_ithf = {k: next(iter(v)) for k, v in sbhf_pairs.items() if len(v) == 1}
    row_ids = resolve_overrides(load_row_overrides(), matcher, sbhf_ithf, local_ids)
    local_ids.save()

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
        new_df = apply_entry_map(results_df, entry_map, row_ids)
        changes, _ = id_changes(results_df, new_df)
        added = append_id_changes(changes)
        print(f"Ids that moved entirely: {len(changes)} ({added} new rows in {ID_CHANGES_CSV})")
        write_results(args.results, new_df)

    # 5. Summary
    total = len(entry_map)
    mapped = entry_map["PlayerID"].notna()
    print("-" * 30)
    print(f"Tournament entries: {total}, with ITHF id: {entry_map['ITHF'].notna().sum()}")
    print(f"Mapped: {mapped.sum()} ({mapped.sum()/total:.1%})")
    for method, count in entry_map.loc[mapped, "Method"].value_counts().items():
        print(f"  - {method}: {count}")
    print(f"Rejected name matches (ITHF id conflicts): {(entry_map['Method'] == 'Conflict').sum()}")
    print(f"ITHF ids borrowed: {entry_map['ITHFSource'].isin(['SBHF', 'Name']).sum()} "
          f"(SBHF {(entry_map['ITHFSource'] == 'SBHF').sum()}, name {(entry_map['ITHFSource'] == 'Name').sum()}), "
          f"corrected: {(entry_map['ITHFSource'] == 'Correction').sum()}")
    print(f"Row overrides for namesakes: {len(row_ids)} row sides")
    print("-" * 30)
    print(f"Entry mapping saved to: {ENTRY_MAPPING_CSV}")
    print(f"Unmapped list saved to: {UNMAPPED_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
