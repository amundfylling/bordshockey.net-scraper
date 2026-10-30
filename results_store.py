"""
Read and write the results dataset.

The dataset lives in data/bordshockey_results.parquet (a few MB, where the CSV
would be well over GitHub's 100 MB file limit). bordshockey_results.csv is the
frozen legacy copy that older consumers still read: nothing writes it, and it is
only read to seed the Parquet file when that does not exist yet.

Any path ending in .csv is read and written as a plain CSV instead.
"""

from __future__ import annotations

import os

import pandas as pd

RESULTS_PATH = os.path.join("data", "bordshockey_results.parquet")
LEGACY_CSV = "bordshockey_results.csv"

INT_COLUMNS = [
    "StageID",
    "Player1ID",
    "Player2ID",
    "GoalsPlayer1",
    "GoalsPlayer2",
    "RoundNumber",
    "PlayoffGameNumber",
    "TournamentID",
    "StageSequence",
]
# Ids that are not always numeric stay text, matching how consumers read the CSV.
ID_TEXT_COLUMNS = ["SourceTournamentID", "SourceStageID", "SourceMatchID"]


def _is_csv(path: str) -> bool:
    return path.lower().endswith(".csv")


def _source(path: str) -> str | None:
    if os.path.exists(path):
        return path
    if path == RESULTS_PATH and os.path.exists(LEGACY_CSV):
        return LEGACY_CSV
    return None


def results_exist(path: str = RESULTS_PATH) -> bool:
    return _source(path) is not None


def read_results(path: str = RESULTS_PATH, columns: list[str] | None = None, as_text: bool = False) -> pd.DataFrame:
    """
    Read the dataset. as_text returns every column as str with "" for missing values.
    Raises FileNotFoundError when there is no dataset.
    """
    source = _source(path)
    if source is None:
        raise FileNotFoundError(path)

    if _is_csv(source):
        if as_text:
            return pd.read_csv(source, encoding="utf-8-sig", usecols=columns, dtype=str, keep_default_na=False)
        return pd.read_csv(source, encoding="utf-8-sig", usecols=columns)

    df = pd.read_parquet(source, columns=columns)
    if as_text:
        df = df.astype("string").fillna("").astype(str)
    return df


def _id_text(value: object) -> object:
    if value is None or (isinstance(value, float) and pd.isna(value)) or value is pd.NA:
        return pd.NA
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text or pd.NA


def normalize_types(df: pd.DataFrame) -> pd.DataFrame:
    """Give every column a fixed type so the Parquet schema stays stable between runs."""
    out = df.copy()
    for col in out.columns:
        if col in INT_COLUMNS:
            out[col] = pd.to_numeric(out[col].replace("", pd.NA), errors="coerce").astype("Int64")
        elif col in ID_TEXT_COLUMNS:
            out[col] = out[col].map(_id_text).astype("string")
        else:
            out[col] = out[col].astype("string").replace("", pd.NA)
    return out


def write_results(path: str, df: pd.DataFrame) -> None:
    if _is_csv(path):
        df.to_csv(path, index=False, encoding="utf-8-sig")
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    normalize_types(df).to_parquet(path, index=False, compression="zstd")
