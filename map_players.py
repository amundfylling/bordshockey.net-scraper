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
    
    for _, row in github_df.iterrows():
        name = str(row['Name']).strip()
        pid = row['PlayerID']
        if name and name.lower() != "nan":
            github_lookup[name.lower()] = pid
            github_names_original.append(name)

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
            # difflib.get_close_matches returns a list of best matches
            # cutoff=0.85 is a strict threshold to avoid false positives
            matches = difflib.get_close_matches(scraped_name, github_names_original, n=1, cutoff=0.85)
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
