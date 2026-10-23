"""Download every public source into data/raw and record a manifest with SHA-256 hashes.

Sources that cannot be fetched automatically from some networks (the SkillsFuture portal pages are
behind a bot check, but the files' go.gov.sg short links worked for us) are listed with instructions
in docs/DATA_SOURCES.md. Existing files are not re-downloaded; delete a file to refresh it.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

from .config import RAW, ensure_dirs

UA = {"User-Agent": "Mozilla/5.0 (onetx data fetcher)"}
HF = "https://huggingface.co/datasets/Anthropic/EconomicIndex/resolve/main"

SOURCES: dict[str, str] = {
    # O*NET (CC BY 4.0)
    "onet/db_30_1_text.zip": "https://www.onetcenter.org/dl_files/database/db_30_1_text.zip",
    "onet/db_31_0_text.zip": "https://www.onetcenter.org/dl_files/database/db_31_0_text.zip",
    # crosswalks
    "crosswalks/onetsoc2019_to_soc2018.csv": "https://www.onetcenter.org/taxonomy/2019/soc/2019_to_SOC_Crosswalk.csv?fmt=csv",
    "crosswalks/onetsoc2010_to_onetsoc2019.csv": "https://www.onetcenter.org/taxonomy/2019/walk/2010_to_2019_Crosswalk.csv?fmt=csv",
    "crosswalks/esco_to_onetsoc2019.xlsx": "https://www.onetcenter.org/crosswalks/esco/ESCO_to_ONET-SOC.xlsx",
    "crosswalks/soc2010_to_isco08.csv": "https://danielruss.github.io/codingsystems/soc2010_isco2008.csv",
    "crosswalks/isco08_to_soc2010.csv": "https://danielruss.github.io/codingsystems/isco2008_soc2010.csv",
    "crosswalks/soc2018_all.csv": "https://danielruss.github.io/codingsystems/soc2018_all.csv",
    # Singapore: SingStat SSOC 2024 / 2020
    "singapore/ssoc2024_structure.xlsx": "https://www.singstat.gov.sg/files/b4cb5e81-91bd-4b21-9745-e09383ee6d8e.xlsx",
    "singapore/ssoc2024_type_of_change.xlsx": "https://www.singstat.gov.sg/files/33ad1368-4e1f-4a93-be04-5dd2c5422b9f.xlsx",
    "singapore/ssoc2024_detailed_definitions.xlsx": "https://www.singstat.gov.sg/files/456a230e-fa6a-41f6-9e7f-6d3f0ffcf521.xlsx",
    "singapore/ssoc2024_alpha_index.xlsx": "https://www.singstat.gov.sg/files/e1468cad-4d50-412e-97b7-40d3621a4953.xlsx",
    "singapore/ssoc2024_to_ssoc2020.xlsx": "https://www.singstat.gov.sg/files/9c84e968-93bb-44e2-8e96-2c7142c1cf42.xlsx",
    "singapore/ssoc2024_to_isco08.xlsx": "https://www.singstat.gov.sg/files/64fec1e2-8a1f-426b-87f7-07b29aae61b3.xlsx",
    "singapore/ssoc2020_structure.xlsx": "https://www.singstat.gov.sg/files/3276886b-d6a1-4c6a-989e-716a54e189e8.xlsx",
    "singapore/ssoc2020_detailed_definitions.xlsx": "https://www.singstat.gov.sg/files/bbd1b6a5-9246-4f73-8d57-34fbd0a3c084.xlsx",
    "singapore/ssoc2020_alpha_index.xlsx": "https://www.singstat.gov.sg/files/03ca0c44-8d8d-4c58-8b51-3025d06df82d.xlsx",
    "singapore/ssoc2020_to_isco08.xlsx": "https://www.singstat.gov.sg/files/e4e5da20-9922-463d-bf23-a1f2a17ecc79.xlsx",
    # Singapore: SkillsFuture Skills Framework (Jobs-Skills Portal, go.gov.sg short links)
    "singapore/jobsandskills-skillsfuture-skills-framework-dataset.xlsx": "https://file.go.gov.sg/jobsandskills-skillsfuture-skills-framework-dataset.xlsx",
    "singapore/jobsandskills-skillsfuture-unique-skills-list.xlsx": "https://file.go.gov.sg/jobsandskills-skillsfuture-unique-skills-list.xlsx",
    "singapore/jobsandskills-skillsfuture-tsc-to-unique-skills-mapping.xlsx": "https://file.go.gov.sg/jobsandskills-skillsfuture-tsc-to-unique-skills-mapping.xlsx",
    # Microsoft "Working with AI"
    **{f"microsoft/{f}": f"https://raw.githubusercontent.com/microsoft/working-with-ai/main/{f}" for f in ["iwa_metrics.csv", "ai_applicability_scores.csv", "soc_metrics.csv", "soc_iwa_weights.csv", "soc_to_iwas.csv", "physical_tasks.csv", "soc_iwa_nonphysical_weights.csv"]},
    # OpenAI GDPval
    "openai/gdpval.parquet": "https://huggingface.co/datasets/openai/gdpval/resolve/main/data/train-00000-of-00001.parquet",
    # METR
    "metr/release_dates.yaml": "https://raw.githubusercontent.com/METR/eval-analysis-public/main/data/external/release_dates.yaml",
    "metr/runs_th11.jsonl": "https://raw.githubusercontent.com/METR/eval-analysis-public/main/reports/time-horizon-1-1/data/raw/runs.jsonl",
    # Anthropic Economic Index
    **{f"aei/{p}": f"{HF}/{p}" for p in [
        "release_2025_02_10/onet_task_mappings.csv", "release_2025_02_10/onet_task_statements.csv", "release_2025_02_10/SOC_Structure.csv",
        "release_2025_03_27/README.md", "release_2025_03_27/task_pct_v1.csv", "release_2025_03_27/task_pct_v2.csv", "release_2025_03_27/automation_vs_augmentation_by_task.csv", "release_2025_03_27/task_thinking_fractions.csv", "release_2025_03_27/cluster_level_data/cluster_level_dataset.tsv",
        "release_2025_09_15/data_documentation.md", "release_2025_09_15/data/intermediate/aei_raw_claude_ai_2025-08-04_to_2025-08-11.csv", "release_2025_09_15/data/intermediate/aei_raw_1p_api_2025-08-04_to_2025-08-11.csv", "release_2025_09_15/data/intermediate/onet_task_statements.csv",
        "release_2026_01_15/data_documentation.md", "release_2026_01_15/data/intermediate/aei_raw_claude_ai_2025-11-13_to_2025-11-20.csv", "release_2026_01_15/data/intermediate/aei_raw_1p_api_2025-11-13_to_2025-11-20.csv",
        "release_2026_03_24/data_documentation.md", "release_2026_03_24/data/aei_raw_claude_ai_2026-02-05_to_2026-02-12.csv", "release_2026_03_24/data/aei_raw_1p_api_2026-02-05_to_2026-02-12.csv",
        "release_2026_06_26/data_documentation.md", "release_2026_06_26/data/aei_claude_ai_2026-06-26.csv", "release_2026_06_26/data/aei_1p_api_2026-06-26.csv",
        "labor_market_impacts/job_exposure.csv", "labor_market_impacts/task_penetration.csv",
        "robot_exposure/README.md", "robot_exposure/robot_exposure_occupations.csv", "robot_exposure/robot_exposure_tasks.csv",
    ]},
}


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_one(rel: str, url: str, force: bool = False) -> Path:
    dest = RAW / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        return dest
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=600) as r, open(dest, "wb") as fh:
        shutil.copyfileobj(r, fh)
    if dest.suffix == ".zip":
        with zipfile.ZipFile(dest) as z:
            z.extractall(dest.parent)
    return dest


def fetch_all(force: bool = False) -> dict:
    ensure_dirs()
    manifest = {}
    for rel, url in SOURCES.items():
        try:
            p = fetch_one(rel, url, force)
            manifest[rel] = {"url": url, "sha256": _sha256(p), "bytes": p.stat().st_size}
            print(f"ok    {rel}")
        except Exception as e:  # keep going; report at the end
            manifest[rel] = {"url": url, "error": str(e)}
            print(f"FAIL  {rel}: {e}")
    (RAW / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
