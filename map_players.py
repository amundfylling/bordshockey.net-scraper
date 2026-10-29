import numpy as np
import pandas as pd
import json
import difflib
import sys
import os

# --- Configuration ---
PLAYERS_CSV = "players.csv"
GITHUB_CSV = "players_data.csv"
MANUAL_MAPPING_FILE = "manual_mapping.json"
OUTPUT_MAPPING_CSV = "player_id_map.csv"
UNMAPPED_CSV = "unmapped_players.csv"
FUZZY_CUTOFF = 0.90
# ---------------------

def load_manual_mapping(filepath):
    """Loads manual mapping from JSON file."""
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Could not decode {filepath}. Starting with empty manual map.", file=sys.stderr)
    return {}

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

def main():
    # 1. Validation
    if not os.path.exists(PLAYERS_CSV):
        print(f"Error: {PLAYERS_CSV} not found.", file=sys.stderr)
        return
    if not os.path.exists(GITHUB_CSV):
        print(f"Error: {GITHUB_CSV} not found.", file=sys.stderr)
        return

    print("Loading data...")
    scraped_df = pd.read_csv(PLAYERS_CSV)
    github_df = pd.read_csv(GITHUB_CSV)

    # 2. Preprocess GitHub Data for Lookup
    # Dictionary: {lowercase_name: PlayerID}
    # Note: If multiple players have same name, this simple approach takes the last one encountered.
    # Ideally we'd have more distinguishing info, but name is all we have from scraped data.
    github_lookup = {}
    github_names_original = [] # For fuzzy matching, keep original casing

    for name, pid in zip(github_df['Name'], github_df['PlayerID']):
        name = str(name).strip()
        if name and name.lower() != "nan":
            github_lookup[name.lower()] = pid
            github_names_original.append(name)

    fuzzy = FuzzyMatcher(github_names_original)

    # 3. Load Manual Mapping
    manual_map = load_manual_mapping(MANUAL_MAPPING_FILE)

    mapped_results = []
    unmapped_list = []

    print(f"Mapping {len(scraped_df)} players...")

    total = len(scraped_df)
    for i, row in scraped_df.iterrows():
        scraped_name = str(row['PlayerName']).strip()
        scraped_lower = scraped_name.lower()
        
        match_found = False
        pid = None
        method = ""
        matched_name = ""

        # A. Manual Mapping
        if scraped_name in manual_map:
            pid = manual_map[scraped_name]
            method = "Manual"
            matched_name = scraped_name # Assuming manual map is correct
            match_found = True

        # B. Exact Match (Case-Insensitive)
        elif scraped_lower in github_lookup:
            pid = github_lookup[scraped_lower]
            method = "Exact"
            matched_name = scraped_name 
            match_found = True
        
        # C. Fuzzy Match
        else:
            # Same result as difflib.get_close_matches(..., n=1, cutoff=0.90), see FuzzyMatcher
            # cutoff=0.90 is a strict threshold to avoid false positives
            matches = fuzzy.match(scraped_name)
            if matches:
                 best_match = matches[0]
                 pid = github_lookup[best_match.lower()]
                 method = "Fuzzy"
                 matched_name = best_match
                 match_found = True

        if match_found:
            mapped_results.append({
                "ScrapedName": scraped_name,
                "PlayerID": pid,
                "Method": method,
                "MatchedName": matched_name
            })
        else:
            unmapped_list.append(scraped_name)

    # 4. Save Results
    res_df = pd.DataFrame(mapped_results)
    res_df.to_csv(OUTPUT_MAPPING_CSV, index=False, encoding='utf-8-sig')

    with open(UNMAPPED_CSV, 'w', encoding='utf-8') as f:
        f.write("PlayerName\n")
        for name in sorted(unmapped_list):
            f.write(f"{name}\n")

    # 5. Summary
    print("-" * 30)
    print(f"Total Scraped Players: {total}")
    print(f"Mapped: {len(mapped_results)} ({len(mapped_results)/total:.1%})")
    print(f"Unmapped: {len(unmapped_list)} ({len(unmapped_list)/total:.1%})")
    print(f"  - Exact/Manual: {len([r for r in mapped_results if r['Method'] in ['Exact', 'Manual']])}")
    print(f"  - Fuzzy: {len([r for r in mapped_results if r['Method'] == 'Fuzzy'])}")
    print("-" * 30)
    print(f"Mapping saved to: {OUTPUT_MAPPING_CSV}")
    print(f"Unmapped list saved to: {UNMAPPED_CSV}")

if __name__ == "__main__":
    main()
