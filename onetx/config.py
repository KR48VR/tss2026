"""Paths and constants shared by every module."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("ONETX_ROOT", Path(__file__).resolve().parents[1]))
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
CACHE = DATA / "cache"
DUCKDB_PATH = PROCESSED / "onetx.duckdb"

# O*NET releases we load. 30.1 (Dec 2025) still uses the classic DWA/IWA identifiers
# (4.A.1.a.1.I01.D01) that the Anthropic, Microsoft and robot-exposure files reference.
# 31.0 (Aug 2026) renumbered DWAs/IWAs and split "Skills" into Essential and Transferable skills.
ONET_BASE_VERSION = "30.1"
ONET_LATEST_VERSION = "31.0"
ONET_DIRS = {
    "30.1": RAW / "onet" / "db_30_1_text",
    "31.0": RAW / "onet" / "db_31_0_text",
}

# Static embedding model (fast, CPU-only). Swap for a sentence-transformers model if you have a GPU.
EMBED_MODEL = os.environ.get("ONETX_EMBED_MODEL", "minishlab/potion-base-32M")

# Anthropic Economic Index vintages (data collection window midpoint, used as the time axis).
AEI_VINTAGES = [
    # vintage label, release folder, file(s), mid-date of the usage window, schema family
    {"vintage": "2024-12", "release": "release_2025_02_10", "mid_date": "2025-01-01", "schema": "task_pct"},
    {"vintage": "2025-03", "release": "release_2025_03_27", "mid_date": "2025-03-01", "schema": "task_pct"},
    {"vintage": "2025-08", "release": "release_2025_09_15", "mid_date": "2025-08-08", "schema": "raw_weekly"},
    {"vintage": "2025-11", "release": "release_2026_01_15", "mid_date": "2025-11-17", "schema": "raw_weekly"},
    {"vintage": "2026-02", "release": "release_2026_03_24", "mid_date": "2026-02-09", "schema": "raw_weekly"},
    {"vintage": "2026-04", "release": "release_2026_06_26", "mid_date": "2026-04-15", "schema": "monthly"},
    {"vintage": "2026-05", "release": "release_2026_06_26", "mid_date": "2026-05-15", "schema": "monthly"},
]


def ensure_dirs() -> None:
    for p in (RAW, PROCESSED, CACHE):
        p.mkdir(parents=True, exist_ok=True)
