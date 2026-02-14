from __future__ import annotations

import re
import sys
import os
import time
import hashlib
from pathlib import Path
from typing import Iterable, List
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup, Tag
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from tqdm import tqdm
from datetime import date

# ──────────────────────────── constants ───────────────────────────────────── #

_tqdm = tqdm

ARCHIVE_URL = "https://bordshockey.net/tavlingar/tavlingsarkiv/"
PLAYER_MAP_FILE = "player_id_map.csv"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    )
}

ROUND_RX = re.compile(r"Omgång\s+(\d+)", re.I)           # “Omgång 7 av 21”
DATE_RX  = re.compile(r"(\d{1,2})\s+([A-Za-zÅÄÖåäö]+)\s+(\d{4})")

# Swedish month names  → number
SV_MONTH = {
    "januari": 1, "februari": 2, "mars": 3, "april": 4, "maj": 5,
    "juni": 6, "juli": 7, "augusti": 8, "september": 9,
    "oktober": 10, "november": 11, "december": 12,
}


# ──────────────────────────── helpers ─────────────────────────────────────── #

def _get_soup(session: requests.Session, url: str) -> BeautifulSoup:
    """HTTP-GET wrapper returning BeautifulSoup (raises on HTTP errors)."""
    r = session.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return BeautifulSoup(r.text, "lxml")


def _archive_links(session: requests.Session) -> list[str]:
    """
    Return *every* distinct tournament results URL found under ARCHIVE_URL.
    A results URL is defined by the suffix `/resultat/`.
    """
    soup = _get_soup(session, ARCHIVE_URL)

    # The archive page is huge – simply pick every <a href="…/resultat/">
    links = {
        urljoin(ARCHIVE_URL, a["href"])
        for a in soup.find_all("a", href=lambda h: h and h.rstrip("/").endswith("/resultat"))
    }

    return sorted(links)

def _parse_date_sv(text: str) -> str | None:
    """
    Return first Swedish/English date in *text* as ISO 'YYYY-MM-DD'.
    """
    m = DATE_RX.search(text)
    if not m:
        return None

    day, month_name, year = m.groups()
    month = (
        SV_MONTH.get(month_name.lower())  # Swedish
        or date.fromisoformat(f"2000-{month_name.title()}-01").month  # English
    )
    return date(int(year), month, int(day)).isoformat()


def _slug_after_resultat(url: str) -> str:
    """
    “…/resultat/kvalgrupper/grupp-1/…” → “Kvalgrupper”,
    “…/resultat/a-slutspel/”            → “A Slutspel”.
    """
    parts = [p for p in urlparse(url).path.split("/") if p]
    if "resultat" in parts:
        idx = parts.index("resultat")
        if idx + 1 < len(parts):
            slug = parts[idx + 1]
        else:
            slug = "Resultat"
    else:
        slug = parts[-1]

    return slug.replace("-", " ").title()


def _overtime(cell: Tag | None) -> str:
    """Detect '(SD)' flag inside a <td>."""
    if cell and "sd" in cell.get_text(strip=True).lower():
        return "Yes"
    return "No"


def _generate_id(text: str) -> int:
    """Generate a consistent integer ID from a string using MD5 and modulo."""
    return int(hashlib.md5(text.encode('utf-8')).hexdigest(), 16) % (10**9) # 9 digit ID

# --------------------------------------------------------------------------- #
# ROUND-ROBIN (“TABELL”) PAGES
# --------------------------------------------------------------------------- #

def _scrape_group_page(
    session: requests.Session,
    url: str,
    t_name: str,
    t_date: str,
    t_id: int
) -> List[List]:
    """Return rows from one 'Tabell och matcher' page."""
    soup  = _get_soup(session, url)
    stage = _slug_after_resultat(url)
    stage_id = _generate_id(f"{t_name}_{stage}")
    
    out   : List[List] = []

    for div in soup.select("div.round_matches"):
        head = div.select_one("tr.round_header")
        if not head:
            continue
        m_r = ROUND_RX.search(head.get_text(strip=True))
        if not m_r:
            continue
        round_no = int(m_r.group(1))

        for tr in div.select("tbody tr"):
            try:
                p1 = tr.select_one("td.home_name").get_text(strip=True)
                p2 = tr.select_one("td.away_name").get_text(strip=True)
                g1 = tr.select_one("td.home_score").get_text(strip=True)
                g2 = tr.select_one("td.away_score").get_text(strip=True)
            except AttributeError:
                continue                      # malformed row

            if not (g1.isdigit() and g2.isdigit()):
                continue                      # unplayed / walk-over

            otime = _overtime(tr.select_one("td.matchinfo"))

            # Schema: StageID, Player1, Player1ID, Player2, Player2ID, GoalsPlayer1, GoalsPlayer2, Overtime, Stage, RoundNumber, PlayoffGameNumber, Date, TournamentName, TournamentID, StageSequence
            # IDs will be filled later, StageSequence default 1
            out.append(
                [stage_id, p1, None, p2, None, int(g1), int(g2),
                 otime, stage, float(round_no), None, t_date, t_name, t_id, 1]
            )
    return out


# --------------------------------------------------------------------------- #
# PLAY-OFF PAGES
# --------------------------------------------------------------------------- #

def _scrape_playoff_page(
    session: requests.Session,
    url: str,
    t_name: str,
    t_date: str,
    t_id: int
) -> List[List]:
    """Drill into an '…-slutspel' page and return one row *per game*."""
    soup       = _get_soup(session, url)
    stage_slug = _slug_after_resultat(url)
    
    out        : List[List] = []

    for pr in soup.select("div.playoff_round"):
        round_header = pr.select_one("h3")
        sub_stage    = round_header.get_text(strip=True) if round_header else ""
        stage_name   = f"{stage_slug} {sub_stage}".strip()
        stage_id = _generate_id(f"{t_name}_{stage_name}")

        table = pr.select_one("table.playoff_round")
        if not table:
            continue

        for series in table.select("tbody tr.match"):
            try:
                p1 = series.select_one("td.home_name").get_text(strip=True)
                p2 = series.select_one("td.away_name").get_text(strip=True)
            except AttributeError:
                continue

            # iterate over game-cells  (td.match 1-7)
            for g_no, cell in enumerate(series.select("td.match"), start=1):
                res_span = cell.select_one("span.result")
                if not res_span:
                    continue            # blank → game not played

                score_txt = res_span.get_text(strip=True)
                if "-" not in score_txt:
                    continue

                g1_txt, g2_txt = [s.strip() for s in score_txt.split("-")]
                if not (g1_txt.isdigit() and g2_txt.isdigit()):
                    continue

                otime = _overtime(cell)

                # Schema: StageID, Player1, Player1ID, Player2, Player2ID, GoalsPlayer1, GoalsPlayer2, Overtime, Stage, RoundNumber, PlayoffGameNumber, Date, TournamentName, TournamentID, StageSequence
                out.append(
                   [stage_id, p1, None, p2, None, int(g1_txt), int(g2_txt),
                    otime, stage_name, 1.0, float(g_no), t_date, t_name, t_id, 1]
                )
    return out


# --------------------------------------------------------------------------- #
# METADATA
# --------------------------------------------------------------------------- #

def _tournament_meta(session: requests.Session, result_url: str) -> tuple[str, str, int]:
    """
    Read the tournament *landing* page to get name + calendar date.
    """
    base = result_url.rstrip("/").rsplit("/resultat", 1)[0] + "/"
    soup = _get_soup(session, base)
    name = soup.select_one("h1").get_text(strip=True)
    date_iso = _parse_date_sv(soup.get_text(" ")) or ""
    t_id = _generate_id(f"{name}_{date_iso}")
    return name, date_iso, t_id

# -----------------------------------------------------------------------------------------------
# 1) helper: scrape ONE tournament  (unchanged logic, but with a local Session)
# -----------------------------------------------------------------------------------------------
def _scrape_tournament_thread(result_url: str, existing_keys: set[tuple[str, str]] | None = None) -> pd.DataFrame:
    """
    Worker-wrapper that-creates its own Session and calls scrape_tournament.
    """
    with requests.Session() as sess:
        return scrape_tournament(result_url, session=sess, existing_keys=existing_keys)

# --------------------------------------------------------------------------- #
# PUBLIC ENTRY
# --------------------------------------------------------------------------- #

# -----------------------------------------------------------------------------------------------
# 2) scrape_tournament(): allow an *external* Session + existing_keys check
# -----------------------------------------------------------------------------------------------
def scrape_tournament(result_url: str, session: requests.Session | None = None, existing_keys: set[tuple[str, str]] | None = None) -> pd.DataFrame:
    """
    If *session* is given we reuse it, otherwise we make a one-shot Session.
    Checks existing_keys (Name, Date) to skip already scraped tournaments.
    """
    own = False
    if session is None:
        session = requests.Session()
        own = True

    try:
        t_name, t_date, t_id = _tournament_meta(session, result_url)

        # Check for duplicates
        if existing_keys is not None and (t_name, t_date) in existing_keys:
            # print(f"Skipping {t_name} ({t_date}) - already scraped.", file=sys.stderr)
            return pd.DataFrame()

        main = _get_soup(session, result_url)

        tabell_links = {
            urljoin(result_url, a["href"])
            for a in main.find_all("a", string=lambda s: s and "Tabell" in s)
        }

        slutspel_links = {
            urljoin(result_url, a["href"])
            for a in main.find_all("a")
            if a.has_attr("href") and (
                "slutspel" in (href := a["href"].lower())
                or "play-off" in href
                or "slutspel" in a.get_text(strip=True).lower()
            )
        }

        rows: list[list] = []
        for link in tabell_links:
            rows.extend(_scrape_group_page(session, link, t_name, t_date, t_id))
        for link in slutspel_links:
            rows.extend(_scrape_playoff_page(session, link, t_name, t_date, t_id))

        return pd.DataFrame(
            rows,
            columns=[
                "StageID", "Player1", "Player1ID", "Player2", "Player2ID", 
                "GoalsPlayer1", "GoalsPlayer2", "Overtime", "Stage", 
                "RoundNumber", "PlayoffGameNumber", "Date", "TournamentName", 
                "TournamentID", "StageSequence"
            ],
        )
    finally:
        if own:
            session.close()


# -----------------------------------------------------------------------------------------------
# 3) NEW fast scrape_archive()
# -----------------------------------------------------------------------------------------------
def scrape_archive(max_workers: int = 12, existing_keys: set[tuple[str, str]] | None = None) -> pd.DataFrame:
    """
    Crawl the whole archive in parallel *and* show a progress bar.
    The bar advances once per finished tournament.
    """
    with requests.Session() as s:
        urls = _archive_links(s)

    print(f"Found {len(urls):,} tournaments – scraping …", file=sys.stderr)
    if existing_keys:
        print(f"(Skipping {len(existing_keys)} already scraped tournaments based on Name/Date)", file=sys.stderr)

    frames: list[pd.DataFrame] = []
    worker = partial(_scrape_tournament_thread, existing_keys=existing_keys)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_url = {pool.submit(worker, u): u for u in urls}

        for fut in _tqdm(as_completed(future_to_url),
                         total=len(urls),
                         desc="Tournaments",
                         unit="event"):
            url = future_to_url[fut]
            try:
                frames.append(fut.result())
            except Exception as exc:
                print(f"[ERROR] {url} → {exc}", file=sys.stderr)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ──────────────────────────── CLI entry-point ─────────────────────────────── #

if __name__ == "__main__":
    csv_file = "bordshockey_results.csv"
    existing_keys = set()
    existing_df = pd.DataFrame()

    # Rename old columns to new standard if needed, or just handle migration
    column_mapping = {
        "Tournament": "TournamentName",
        "Round": "RoundNumber_Old" # Temporary to avoid conflict? Or parse?
    }
    
    # 1. Load Player Map
    player_map = {}
    if os.path.exists(PLAYER_MAP_FILE):
        try:
            pm_df = pd.read_csv(PLAYER_MAP_FILE)
            # Create a lookup: ScrapedName -> PlayerID
            for _, row in pm_df.iterrows():
                player_map[str(row['ScrapedName']).strip()] = row['PlayerID']
            print(f"Loaded {len(player_map)} player mappings.")
        except Exception as e:
            print(f"Warning: Could not load player map: {e}", file=sys.stderr)

    # 2. Check Existing CSV
    if os.path.exists(csv_file):
        try:
            existing_df = pd.read_csv(csv_file)
            # Support incremental scrape for OLD format or NEW format?
            # Assuming we are transitioning. If old columns exist, we might need to be careful.
            # But the requirement is likely for future scraping.
            # If "TournamentName" exists, it's new format. If "Tournament" exists, it's old.
            
            t_col = "TournamentName" if "TournamentName" in existing_df.columns else "Tournament"
            d_col = "Date"
            
            if t_col in existing_df.columns and d_col in existing_df.columns:
                existing_keys = set(zip(existing_df[t_col], existing_df[d_col]))
                print(f"Loaded {len(existing_df)} rows from existing CSV.")
                
        except Exception as e:
            print(f"Warning: Could not read existing CSV: {e}", file=sys.stderr)

    # 3. Scrape
    new_results = scrape_archive(existing_keys=existing_keys)

    if not new_results.empty:
        print(f"Scraped {len(new_results)} new rows.")
        
        # 4. Fill Player IDs
        print("Mapping Player IDs...")
        new_results['Player1ID'] = new_results['Player1'].map(player_map)
        new_results['Player2ID'] = new_results['Player2'].map(player_map)
        
        # Convert IDs to Int64 (nullable int) to handle NaNs gracefully
        new_results['Player1ID'] = new_results['Player1ID'].astype('Int64')
        new_results['Player2ID'] = new_results['Player2ID'].astype('Int64')

        # 5. Append
        if not existing_df.empty:
            # If existing DF is old format, we might have a schema mismatch. 
            # Ideally, we should migrate the OLD data too, but that's a huge task.
            # For now, we will concat. Pandas handles missing columns by adding NaNs.
            # But we want to enforce the NEW schema for the file if possible.
            # If the user wants a full overwrite, they should delete the CSV.
            # Here we just append.
            final_df = pd.concat([existing_df, new_results], ignore_index=True)
        else:
            final_df = new_results
        
        # Ensure column order matches requirements
        target_columns = [
            "StageID", "Player1", "Player1ID", "Player2", "Player2ID", 
            "GoalsPlayer1", "GoalsPlayer2", "Overtime", "Stage", 
            "RoundNumber", "PlayoffGameNumber", "Date", "TournamentName", 
            "TournamentID", "StageSequence"
        ]
        
        # Reorder if columns exist (handled by concat generally, but good to be explicit for new file)
        # Only keep columns that are in target or exist in dataframe
        cols_to_write = [c for c in target_columns if c in final_df.columns]
        # Append any extra columns from old format if we want to keep them
        extra_cols = [c for c in final_df.columns if c not in target_columns]
        
        final_df = final_df[cols_to_write + extra_cols]
        
        final_df.to_csv(csv_file, index=False, mode="w", encoding="utf-8-sig")
        print(f"Updated {csv_file} with new data. Total rows: {len(final_df)}")
    else:
        print("No new tournaments found to scrape.")
