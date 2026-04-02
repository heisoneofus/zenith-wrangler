from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd


def read_csv(path: Path, sample_rows: int | None = None) -> pd.DataFrame:
    if sample_rows:
        return pd.read_csv(path, nrows=sample_rows)
    return pd.read_csv(path)


def read_excel(path: Path, sample_rows: int | None = None) -> pd.DataFrame:
    if sample_rows:
        return pd.read_excel(path, nrows=sample_rows)
    return pd.read_excel(path)


def read_parquet(path: Path, sample_rows: int | None = None) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if sample_rows:
        return df.head(sample_rows)
    return df


def detect_loader(path: Path) -> Literal["read_csv", "read_excel", "read_parquet"]:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return "read_excel"
    if suffix == ".parquet":
        return "read_parquet"
    return "read_csv"
