"""Small I/O helpers: parquet round-trips and a DuckDB view registry."""
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pandas as pd

from .config import DUCKDB_PATH, PROCESSED


def save(df: pd.DataFrame, name: str) -> Path:
    """Write a processed table as parquet and return its path."""
    PROCESSED.mkdir(parents=True, exist_ok=True)
    path = PROCESSED / f"{name}.parquet"
    df.to_parquet(path, index=False)
    return path


def load(name: str) -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / f"{name}.parquet")


def exists(name: str) -> bool:
    return (PROCESSED / f"{name}.parquet").exists()


def register_duckdb(names: list[str] | None = None) -> None:
    """Create (or refresh) a DuckDB file whose views point at every processed parquet table."""
    PROCESSED.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DUCKDB_PATH))
    paths = sorted(PROCESSED.glob("*.parquet")) if names is None else [PROCESSED / f"{n}.parquet" for n in names]
    for p in paths:
        con.execute(f'CREATE OR REPLACE VIEW "{p.stem}" AS SELECT * FROM read_parquet(\'{p.as_posix()}\')')
    con.close()


def write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))
