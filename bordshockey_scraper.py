from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date
from functools import partial
from typing import Iterable
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

import pandas as pd
import requests
from bs4 import BeautifulSoup, Tag
from tqdm import tqdm

# constants

_tqdm = tqdm

ARCHIVE_URL = "https://bordshockey.net/tavlingar/tavlingsarkiv/"
PLAYER_MAP_FILE = "player_id_map.csv"
SOURCE = "bordshockey.net"

BASE_COLUMNS = [
    "StageID",
    "Player1",
    "Player1ID",
    "Player2",
    "Player2ID",
    "GoalsPlayer1",
    "GoalsPlayer2",
    "Overtime",
    "Stage",
    "RoundNumber",
    "PlayoffGameNumber",
    "Date",
    "TournamentName",
    "TournamentID",
    "StageSequence",
]

ADDED_COLUMNS = [
    "StageType",
    "TournamentURL",
    "ResultURL",
    "StageURL",
    "SourceURL",
    "Source",
    "SourceTournamentID",
    "SourceStageID",
    "SourceMatchID",
]

OUTPUT_COLUMNS = BASE_COLUMNS + ADDED_COLUMNS

REQUIRED_POPULATED_COLUMNS = [
    "StageID",
    "TournamentID",
    "StageSequence",
    "StageType",
    "TournamentURL",
    "ResultURL",
    "StageURL",
    "SourceURL",
    "Source",
    "SourceTournamentID",
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    )
}

ROUND_RX = re.compile(r"Omgång\s+(\d+)", re.I)  # "Omgång 7 av 21"
DATE_RX = re.compile(r"(\d{1,2})\s+([A-Za-zÅÄÖåäö]+)\s+(\d{4})")
MATCH_CLASS_RX = re.compile(r"^match-(\d+)$")
STAGE_CLASS_RX = re.compile(r"^stage-(\d+)$")
OT_RX = re.compile(r"\(\s*sd\s*\)|\bsd\b", re.I)
SCORE_RX = re.compile(r"(\d+)\s*-\s*(\d+)")

# Swedish/English month names to number.
MONTHS = {
    "januari": 1,
    "februari": 2,
    "mars": 3,
    "april": 4,
    "maj": 5,
    "juni": 6,
    "juli": 7,
    "augusti": 8,
    "september": 9,
    "oktober": 10,
    "november": 11,
    "december": 12,
    "january": 1,
    "february": 2,
    "march": 3,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "october": 10,
}


@dataclass(frozen=True)
class TournamentMeta:
    name: str
    date: str
    tournament_id: int
    tournament_url: str
    result_url: str
    source_tournament_id: str


@dataclass(frozen=True)
class StageMeta:
    name: str
    stage_type: str
    stage_url: str
    stage_id: int
    stage_sequence: int
    source_stage_id: str | None = None


# helpers

def _get_soup(session: requests.Session, url: str) -> BeautifulSoup:
    """HTTP-GET wrapper returning BeautifulSoup (raises on HTTP errors)."""
    r = session.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return BeautifulSoup(r.text, "lxml")


def _archive_links(session: requests.Session) -> list[str]:
    """
    Return every distinct tournament results URL found under ARCHIVE_URL.
    A results URL is defined by the suffix `/resultat/`.
    """
    soup = _get_soup(session, ARCHIVE_URL)

    links = {
        _canonical_url(urljoin(ARCHIVE_URL, a["href"]))
        for a in soup.find_all("a", href=lambda h: h and h.rstrip("/").endswith("/resultat"))
    }

    return sorted(links)


def _parse_date_sv(text: str) -> str | None:
    """Return the first Swedish/English date in text as ISO 'YYYY-MM-DD'."""
    m = DATE_RX.search(text)
    if not m:
        return None

    day, month_name, year = m.groups()
    month = MONTHS.get(month_name.lower())
    if not month:
        return None
    return date(int(year), month, int(day)).isoformat()


def _generate_id(text: str) -> int:
    """Generate a consistent integer ID from a string using MD5 and modulo."""
    return int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16) % (10**9)


def _canonical_url(url: str) -> str:
    """Normalize bordshockey URLs before using them as source identifiers."""
    parsed = urlparse(url)
    path = re.sub(r"/+", "/", parsed.path or "/")
    if not path.endswith("/"):
        path = f"{path}/"

    query_items = sorted(parse_qsl(parsed.query, keep_blank_values=True))
    query = urlencode(query_items)

    return urlunparse(
        (
            parsed.scheme.lower() or "https",
            parsed.netloc.lower(),
            path,
            "",
            query,
            "",
        )
    )


def _tournament_url_from_result(result_url: str) -> str:
    parsed = urlparse(_canonical_url(result_url))
    path = parsed.path
    marker = "/resultat/"
    if marker in path:
        path = path.split(marker, 1)[0] + "/"
    return urlunparse((parsed.scheme, parsed.netloc, path, "", "", ""))


def _result_url_from_any(url: str) -> str:
    canonical = _canonical_url(url)
    parsed = urlparse(canonical)
    path = parsed.path
    marker = "/resultat/"

    if marker in path:
        path = path.split(marker, 1)[0] + marker
    elif path.endswith("/resultat/"):
        pass
    else:
        path = path.rstrip("/") + "/resultat/"

    return urlunparse((parsed.scheme, parsed.netloc, path, "", "", ""))


def _source_tournament_id(tournament_url: str) -> str:
    parts = [p for p in urlparse(_canonical_url(tournament_url)).path.split("/") if p]
    if "tavlingar" in parts:
        idx = parts.index("tavlingar")
        source_parts = parts[idx + 1 :]
        if source_parts:
            return "/".join(source_parts)
    return urlparse(_canonical_url(tournament_url)).path.strip("/")


def _slug_to_title(slug: str) -> str:
    return slug.replace("-", " ").strip().title()


def _stage_name_from_url(url: str) -> str:
    parts = [p for p in urlparse(_canonical_url(url)).path.split("/") if p]
    if "resultat" in parts:
        parts = parts[parts.index("resultat") + 1 :]

    if len(parts) >= 2 and parts[-1].startswith("grupp-"):
        return f"{_slug_to_title(parts[-2])} - {_slug_to_title(parts[-1])}"
    if parts:
        return _slug_to_title(parts[-1])
    return "Resultat"


def _class_id(tag: Tag | None, prefix: str, rx: re.Pattern[str]) -> str | None:
    if not tag:
        return None
    for cls in tag.get("class", []):
        m = rx.match(cls)
        if m and cls.startswith(prefix):
            return m.group(1)
    return None


def _stage_id_for_url(stage_url: str, source_stage_id: str | None, *, direct_source_stage: bool) -> int:
    if direct_source_stage and source_stage_id and source_stage_id.isdigit():
        return int(source_stage_id)
    return _generate_id(_canonical_url(stage_url))


def _overtime(*nodes: Tag | str | None) -> str:
    """Detect the bordshockey sudden-death marker."""
    for node in nodes:
        if node is None:
            continue
        text = node if isinstance(node, str) else node.get_text(" ", strip=True)
        if OT_RX.search(text):
            return "Yes"
    return "No"


def _match_id_from_row(row: Tag) -> str | None:
    return _class_id(row, "match-", MATCH_CLASS_RX)


def _first_link_url(parent: Tag | None, base_url: str, *, prefer_matcher: bool = False) -> str | None:
    if not parent:
        return None

    links = [a for a in parent.find_all("a", href=True)]
    if prefer_matcher:
        for a in links:
            href = urljoin(base_url, a["href"])
            if "matcher=1" in urlparse(href).query:
                return _canonical_url(href)

    if links:
        return _canonical_url(urljoin(base_url, links[0]["href"]))
    return None


def _blankish(value: object) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except TypeError:
        pass
    return str(value).strip() == ""


def _key_text(value: object) -> str | None:
    if _blankish(value):
        return None
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


# --------------------------------------------------------------------------- #
# RESULT OVERVIEW
# --------------------------------------------------------------------------- #

def _classify_stage_type(row: Tag, stage_url: str | None) -> str | None:
    classes = set(row.get("class", []))
    text = row.get_text(" ", strip=True).lower()
    parsed = urlparse(stage_url or "")

    if "playoff_stage" in classes or "slutspel" in text:
        return "playoff"
    if "matcher=1" in parsed.query or "tabell och matcher" in text:
        return "round-robin"

    return None


def _parse_result_overview(soup: BeautifulSoup, result_url: str) -> list[StageMeta]:
    """Extract source stage ids, sequence numbers, URLs, and stage type."""
    result_url = _canonical_url(result_url)
    stages: list[StageMeta] = []
    current_sequence = 1
    parent_name: str | None = None
    parent_source_stage_id: str | None = None

    for row in soup.select("tr.stage_group, tr.stage, tr.group"):
        classes = set(row.get("class", []))

        if "stage_group" in classes:
            raw_sequence = row.get_text(" ", strip=True)
            try:
                current_sequence = int(raw_sequence)
            except ValueError:
                current_sequence += 1
            parent_name = None
            parent_source_stage_id = None
            continue

        if "stage" in classes:
            name_cell = row.select_one("td.stage_name") or row.find("td")
            link_cell = row.select_one("td.links")
            stage_name = name_cell.get_text(" ", strip=True) if name_cell else _stage_name_from_url(result_url)
            stage_url = _first_link_url(name_cell, result_url) or _first_link_url(link_cell, result_url, prefer_matcher=True)
            source_stage_id = _class_id(row, "stage-", STAGE_CLASS_RX)

            parent_name = stage_name
            parent_source_stage_id = source_stage_id

            stage_type = _classify_stage_type(row, stage_url)
            if not stage_url or not stage_type:
                continue

            direct_source_stage = "matcher=1" not in urlparse(stage_url).query
            stages.append(
                StageMeta(
                    name=stage_name,
                    stage_type=stage_type,
                    stage_url=stage_url,
                    stage_id=_stage_id_for_url(
                        stage_url,
                        source_stage_id,
                        direct_source_stage=direct_source_stage,
                    ),
                    stage_sequence=current_sequence,
                    source_stage_id=source_stage_id,
                )
            )
            continue

        if "group" in classes:
            name_cell = row.select_one("td.group_name") or row.find("td")
            link_cell = row.select_one("td.links")
            stage_url = (
                _first_link_url(name_cell, result_url, prefer_matcher=True)
                or _first_link_url(link_cell, result_url, prefer_matcher=True)
            )
            if not stage_url:
                continue

            group_name = name_cell.get_text(" ", strip=True) if name_cell else _stage_name_from_url(stage_url)
            stage_name = f"{parent_name} - {group_name}" if parent_name else group_name
            stage_type = _classify_stage_type(row, stage_url) or "round-robin"

            stages.append(
                StageMeta(
                    name=stage_name,
                    stage_type=stage_type,
                    stage_url=stage_url,
                    stage_id=_stage_id_for_url(
                        stage_url,
                        parent_source_stage_id,
                        direct_source_stage=False,
                    ),
                    stage_sequence=current_sequence,
                    source_stage_id=parent_source_stage_id,
                )
            )

    return stages


def _infer_stage_meta(stage_url: str, soup: BeautifulSoup | None = None) -> StageMeta:
    stage_url = _canonical_url(stage_url)
    source_stage_id = _class_id(soup.select_one("div.stage") if soup else None, "stage-", STAGE_CLASS_RX)

    if soup and (soup.select_one("div.playoff_stage, table.playoff_round") or "slutspel" in soup.get_text(" ", strip=True).lower()):
        stage_type = "playoff"
    elif "matcher=1" in urlparse(stage_url).query or (soup and soup.select_one("div.round_matches")):
        stage_type = "round-robin"
    else:
        stage_type = "round-robin"

    return StageMeta(
        name=_stage_name_from_url(stage_url),
        stage_type=stage_type,
        stage_url=stage_url,
        stage_id=_stage_id_for_url(stage_url, source_stage_id, direct_source_stage=stage_type == "playoff"),
        stage_sequence=1,
        source_stage_id=source_stage_id,
    )


# --------------------------------------------------------------------------- #
# ROW BUILDING
# --------------------------------------------------------------------------- #

def _base_match_row(
    tournament: TournamentMeta,
    stage: StageMeta,
    *,
    player1: str,
    player2: str,
    goals1: int,
    goals2: int,
    overtime: str,
    round_number: int | float | None,
    playoff_game_number: int | float | None,
    stage_name: str | None = None,
    source_match_id: str | None = None,
) -> dict[str, object]:
    return {
        "StageID": stage.stage_id,
        "Player1": player1,
        "Player1ID": None,
        "Player2": player2,
        "Player2ID": None,
        "GoalsPlayer1": goals1,
        "GoalsPlayer2": goals2,
        "Overtime": overtime,
        "Stage": stage_name or stage.name,
        "RoundNumber": round_number,
        "PlayoffGameNumber": playoff_game_number,
        "Date": tournament.date,
        "TournamentName": tournament.name,
        "TournamentID": tournament.tournament_id,
        "StageSequence": stage.stage_sequence,
        "StageType": stage.stage_type,
        "TournamentURL": tournament.tournament_url,
        "ResultURL": tournament.result_url,
        "StageURL": stage.stage_url,
        "SourceURL": stage.stage_url,
        "Source": SOURCE,
        "SourceTournamentID": tournament.source_tournament_id,
        "SourceStageID": stage.source_stage_id,
        "SourceMatchID": source_match_id,
    }


def _rows_from_group_soup(
    soup: BeautifulSoup,
    stage: StageMeta,
    tournament: TournamentMeta,
) -> list[dict[str, object]]:
    """Return rows from one round-robin 'Tabell och matcher' page."""
    rows: list[dict[str, object]] = []

    for div in soup.select("div.round_matches"):
        head = div.select_one("tr.round_header")
        if not head:
            continue
        m_r = ROUND_RX.search(head.get_text(" ", strip=True))
        if not m_r:
            continue
        round_no = int(m_r.group(1))

        for row in div.select("tbody tr"):
            if row.select_one("p.noplay"):
                continue

            try:
                p1 = row.select_one("td.home_name").get_text(" ", strip=True)
                p2 = row.select_one("td.away_name").get_text(" ", strip=True)
                g1 = row.select_one("td.home_score").get_text(" ", strip=True)
                g2 = row.select_one("td.away_score").get_text(" ", strip=True)
            except AttributeError:
                continue

            if not (g1.isdigit() and g2.isdigit()):
                continue

            rows.append(
                _base_match_row(
                    tournament,
                    stage,
                    player1=p1,
                    player2=p2,
                    goals1=int(g1),
                    goals2=int(g2),
                    overtime=_overtime(row.select_one("td.matchinfo")),
                    round_number=round_no,
                    playoff_game_number=None,
                    source_match_id=_match_id_from_row(row),
                )
            )

    return rows


def _scrape_group_page(
    session: requests.Session,
    stage: StageMeta,
    tournament: TournamentMeta,
) -> list[dict[str, object]]:
    soup = _get_soup(session, stage.stage_url)
    inferred = stage
    if stage.stage_type != "round-robin":
        inferred = StageMeta(
            name=stage.name,
            stage_type="round-robin",
            stage_url=stage.stage_url,
            stage_id=stage.stage_id,
            stage_sequence=stage.stage_sequence,
            source_stage_id=stage.source_stage_id,
        )
    return _rows_from_group_soup(soup, inferred, tournament)


def _rows_from_playoff_soup(
    soup: BeautifulSoup,
    stage: StageMeta,
    tournament: TournamentMeta,
) -> list[dict[str, object]]:
    """Return one row per played playoff game."""
    rows: list[dict[str, object]] = []

    for playoff_round in soup.select("div.playoff_round"):
        round_header = playoff_round.select_one("h3")
        round_name = round_header.get_text(" ", strip=True) if round_header else ""
        stage_name = f"{stage.name} {round_name}".strip()

        table = playoff_round.select_one("table.playoff_round")
        if not table:
            continue

        for series in table.select("tbody tr.match"):
            if "mobile_matches" in series.get("class", []):
                continue

            try:
                p1 = series.select_one("td.home_name").get_text(" ", strip=True)
                p2 = series.select_one("td.away_name").get_text(" ", strip=True)
            except AttributeError:
                continue

            for game_no, cell in enumerate(series.select("td.match"), start=1):
                res_span = cell.select_one("span.result")
                score_txt = res_span.get_text(" ", strip=True) if res_span else cell.get_text(" ", strip=True)
                m_score = SCORE_RX.search(score_txt)
                if not m_score:
                    continue

                rows.append(
                    _base_match_row(
                        tournament,
                        stage,
                        player1=p1,
                        player2=p2,
                        goals1=int(m_score.group(1)),
                        goals2=int(m_score.group(2)),
                        overtime=_overtime(cell),
                        round_number=1,
                        playoff_game_number=game_no,
                        stage_name=stage_name,
                        source_match_id=None,
                    )
                )

    return rows


def _scrape_playoff_page(
    session: requests.Session,
    stage: StageMeta,
    tournament: TournamentMeta,
) -> list[dict[str, object]]:
    soup = _get_soup(session, stage.stage_url)
    inferred = stage
    source_stage_id = stage.source_stage_id or _class_id(soup.select_one("div.stage"), "stage-", STAGE_CLASS_RX)
    if stage.stage_type != "playoff" or source_stage_id != stage.source_stage_id:
        inferred = StageMeta(
            name=stage.name,
            stage_type="playoff",
            stage_url=stage.stage_url,
            stage_id=_stage_id_for_url(stage.stage_url, source_stage_id, direct_source_stage=True),
            stage_sequence=stage.stage_sequence,
            source_stage_id=source_stage_id,
        )
    return _rows_from_playoff_soup(soup, inferred, tournament)


# --------------------------------------------------------------------------- #
# METADATA
# --------------------------------------------------------------------------- #

def _tournament_meta(session: requests.Session, result_url: str) -> TournamentMeta:
    """Read the tournament landing page to get name, date, and stable ids."""
    result_url = _result_url_from_any(result_url)
    tournament_url = _tournament_url_from_result(result_url)
    soup = _get_soup(session, tournament_url)

    name_el = soup.select_one("h1")
    name = name_el.get_text(" ", strip=True) if name_el else _source_tournament_id(tournament_url)
    date_iso = _parse_date_sv(soup.get_text(" ", strip=True)) or ""
    source_tournament_id = _source_tournament_id(tournament_url)

    return TournamentMeta(
        name=name,
        date=date_iso,
        tournament_id=_generate_id(tournament_url),
        tournament_url=tournament_url,
        result_url=result_url,
        source_tournament_id=source_tournament_id,
    )


# -----------------------------------------------------------------------------------------------
# scrape helpers
# -----------------------------------------------------------------------------------------------

def _scrape_tournament_thread(result_url: str, existing_result_urls: set[str] | None = None) -> pd.DataFrame:
    """Worker wrapper that creates its own Session and calls scrape_tournament."""
    with requests.Session() as sess:
        return scrape_tournament(result_url, session=sess, existing_result_urls=existing_result_urls)


def _stage_meta_for_url(
    session: requests.Session,
    stage_url: str,
    tournament: TournamentMeta,
) -> StageMeta:
    overview = _get_soup(session, tournament.result_url)
    stages = _parse_result_overview(overview, tournament.result_url)
    canonical = _canonical_url(stage_url)
    for stage in stages:
        if stage.stage_url == canonical:
            return stage

    stage_soup = _get_soup(session, canonical)
    return _infer_stage_meta(canonical, stage_soup)


# --------------------------------------------------------------------------- #
# PUBLIC ENTRY
# --------------------------------------------------------------------------- #

def scrape_tournament(
    result_url: str,
    session: requests.Session | None = None,
    existing_result_urls: set[str] | None = None,
) -> pd.DataFrame:
    """
    Scrape one tournament result overview and all discovered match pages.
    existing_result_urls is only a tournament-level skip for archive mode.
    Row-level duplicate handling happens before CSV write.
    """
    own = False
    if session is None:
        session = requests.Session()
        own = True

    try:
        tournament = _tournament_meta(session, result_url)
        if existing_result_urls is not None and tournament.result_url in existing_result_urls:
            return pd.DataFrame(columns=OUTPUT_COLUMNS)

        overview = _get_soup(session, tournament.result_url)
        stages = _parse_result_overview(overview, tournament.result_url)

        rows: list[dict[str, object]] = []
        for stage in stages:
            if stage.stage_type == "round-robin":
                rows.extend(_scrape_group_page(session, stage, tournament))
            elif stage.stage_type == "playoff":
                rows.extend(_scrape_playoff_page(session, stage, tournament))

        return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    finally:
        if own:
            session.close()


def scrape_url(url: str, session: requests.Session | None = None) -> pd.DataFrame:
    """Scrape a tournament result URL, tournament landing URL, or one stage URL."""
    own = False
    if session is None:
        session = requests.Session()
        own = True

    try:
        canonical = _canonical_url(url)
        result_url = _result_url_from_any(canonical)

        if canonical == result_url or canonical == _tournament_url_from_result(result_url):
            return scrape_tournament(result_url, session=session)

        tournament = _tournament_meta(session, result_url)
        stage = _stage_meta_for_url(session, canonical, tournament)

        if stage.stage_type == "playoff":
            rows = _scrape_playoff_page(session, stage, tournament)
        else:
            rows = _scrape_group_page(session, stage, tournament)

        return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    finally:
        if own:
            session.close()


def scrape_archive(max_workers: int = 12, existing_result_urls: set[str] | None = None) -> pd.DataFrame:
    """Crawl the whole archive in parallel and show a progress bar."""
    with requests.Session() as s:
        urls = _archive_links(s)

    if existing_result_urls:
        print(
            f"Found {len(urls):,} tournaments; skipping {len(existing_result_urls):,} already present result URLs.",
            file=sys.stderr,
        )
    else:
        print(f"Found {len(urls):,} tournaments - scraping ...", file=sys.stderr)

    frames: list[pd.DataFrame] = []
    worker = partial(_scrape_tournament_thread, existing_result_urls=existing_result_urls)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_url = {pool.submit(worker, u): u for u in urls}

        for fut in _tqdm(
            as_completed(future_to_url),
            total=len(urls),
            desc="Tournaments",
            unit="event",
        ):
            url = future_to_url[fut]
            try:
                frames.append(fut.result())
            except Exception as exc:
                print(f"[ERROR] {url} -> {exc}", file=sys.stderr)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=OUTPUT_COLUMNS)


# --------------------------------------------------------------------------- #
# CSV compatibility, mapping, and dedupe
# --------------------------------------------------------------------------- #

def _load_player_map(path: str = PLAYER_MAP_FILE) -> dict[str, object]:
    player_map: dict[str, object] = {}
    if not os.path.exists(path):
        return player_map

    try:
        pm_df = pd.read_csv(path)
        for _, row in pm_df.iterrows():
            player_map[str(row["ScrapedName"]).strip()] = row["PlayerID"]
        print(f"Loaded {len(player_map)} player mappings.")
    except Exception as exc:
        print(f"Warning: Could not load player map: {exc}", file=sys.stderr)

    return player_map


def _ensure_output_schema(df: pd.DataFrame) -> pd.DataFrame:
    for col in OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df[OUTPUT_COLUMNS].copy()


def _needs_rebuild(df: pd.DataFrame) -> bool:
    missing = [col for col in OUTPUT_COLUMNS if col not in df.columns]
    if missing:
        print(
            f"Existing CSV is missing required columns ({', '.join(missing)}); rebuilding instead of appending.",
            file=sys.stderr,
        )
        return True

    for col in REQUIRED_POPULATED_COLUMNS:
        if df[col].map(_blankish).any():
            print(
                f"Existing CSV has blank values in {col}; rebuilding instead of appending.",
                file=sys.stderr,
            )
            return True

    return False


def _dedupe_key(row: pd.Series) -> tuple[object, ...]:
    source_match_id = _key_text(row.get("SourceMatchID"))
    if source_match_id is not None:
        return (
            "source-match",
            _key_text(row.get("Source")),
            _key_text(row.get("SourceTournamentID")),
            source_match_id,
        )

    return (
        "match-tuple",
        row.get("TournamentID"),
        row.get("StageID"),
        row.get("StageURL"),
        row.get("RoundNumber"),
        row.get("PlayoffGameNumber"),
        row.get("Player1"),
        row.get("Player2"),
        row.get("GoalsPlayer1"),
        row.get("GoalsPlayer2"),
    )


def _deduplicate_results(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return _ensure_output_schema(df)

    out = _ensure_output_schema(df)
    keys = out.apply(_dedupe_key, axis=1)
    out = out.loc[~keys.duplicated(keep="last")].copy()
    return out.reset_index(drop=True)


def _apply_player_map(df: pd.DataFrame, player_map: dict[str, object]) -> pd.DataFrame:
    if df.empty or not player_map:
        return df

    out = df.copy()
    for player_col, id_col in (("Player1", "Player1ID"), ("Player2", "Player2ID")):
        mapped = out[player_col].map(player_map)
        if id_col in out.columns:
            out[id_col] = mapped.combine_first(out[id_col])
        else:
            out[id_col] = mapped
        out[id_col] = out[id_col].astype("Int64")

    return out


def _existing_result_urls(df: pd.DataFrame) -> set[str]:
    if "ResultURL" not in df.columns:
        return set()
    return {
        _canonical_url(str(url))
        for url in df["ResultURL"].dropna().unique()
        if str(url).strip()
    }


def _load_existing_csv(csv_file: str) -> pd.DataFrame:
    if not os.path.exists(csv_file):
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    try:
        df = pd.read_csv(csv_file)
        print(f"Loaded {len(df)} rows from existing CSV.")
        return df
    except Exception as exc:
        print(f"Warning: Could not read existing CSV: {exc}", file=sys.stderr)
        return pd.DataFrame(columns=OUTPUT_COLUMNS)


def _write_results(csv_file: str, df: pd.DataFrame) -> None:
    final_df = _ensure_output_schema(df)
    final_df.to_csv(csv_file, index=False, encoding="utf-8-sig")
    print(f"Wrote {csv_file}. Total rows: {len(final_df)}")


# CLI entry point

def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Scrape bordshockey.net tournament match results.")
    parser.add_argument("--csv", default="bordshockey_results.csv", help="Output CSV path.")
    parser.add_argument("--max-workers", type=int, default=12, help="Parallel workers for archive scrape.")
    parser.add_argument("--rebuild", action="store_true", help="Ignore existing CSV and rebuild the full archive.")
    parser.add_argument(
        "--url",
        action="append",
        help="Scrape one tournament/result/stage URL. Can be passed multiple times.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    player_map = _load_player_map()
    existing_df = _load_existing_csv(args.csv)

    rebuild = args.rebuild
    existing_needs_rebuild = not existing_df.empty and _needs_rebuild(existing_df)
    if existing_needs_rebuild and args.url and not rebuild:
        print(
            "Refusing to append URL scrape results to an incomplete existing CSV. "
            "Use --rebuild to overwrite it, or pass --csv with a new/temporary output path.",
            file=sys.stderr,
        )
        return 2
    if existing_needs_rebuild:
        rebuild = True

    frames: list[pd.DataFrame] = []
    if not rebuild and not existing_df.empty:
        frames.append(existing_df)

    if args.url:
        with requests.Session() as session:
            for url in args.url:
                print(f"Scraping {url}")
                frames.append(scrape_url(url, session=session))
    else:
        skip_urls = None if rebuild else _existing_result_urls(existing_df)
        frames.append(scrape_archive(max_workers=args.max_workers, existing_result_urls=skip_urls))

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=OUTPUT_COLUMNS)
    combined = _deduplicate_results(combined)
    combined = _apply_player_map(combined, player_map)
    _write_results(args.csv, combined)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
