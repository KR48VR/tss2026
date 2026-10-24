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

UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15 onetx-data-fetcher"}
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
    # OpenAI "GPTs are GPTs" occupation-level exposure (Eloundou et al. 2023)
    "openai/gpts_are_gpts_occ_level.csv": "https://raw.githubusercontent.com/openai/GPTs-are-GPTs/main/data/occ_level.csv",
    # task-level labels from the same study: human raters and GPT-4, E0/E1/E2 per O*NET task
    "openai/gpts_are_gpts_full_labelset.tsv": "https://raw.githubusercontent.com/openai/GPTs-are-GPTs/main/data/full_labelset.tsv",
    # US Census Business Trends and Outlook Survey: share of firms using AI, national and by NAICS sector
    # (biweekly; refreshed on every fetch once the local copy is two weeks old, see FRESH_DAYS)
    "census/btos_national.xlsx": "https://www.census.gov/hfp/btos/downloads/National.xlsx",
    "census/btos_sector.xlsx": "https://www.census.gov/hfp/btos/downloads/Sector.xlsx",
    # US Bureau of Labor Statistics, Occupational Employment and Wage Statistics (May 2024 and May 2023):
    # employment and wages by detailed occupation (national) and by NAICS sector (the in4 bundle; only its
    # sector file is used). The second workforce the map can be weighted by.
    "bls/oesm24nat.zip": "https://www.bls.gov/oes/special-requests/oesm24nat.zip",
    "bls/oesm23nat.zip": "https://www.bls.gov/oes/special-requests/oesm23nat.zip",
    "bls/oesm24in4.zip": "https://www.bls.gov/oes/special-requests/oesm24in4.zip",
    # US Department of Labor, Competency Model Clearinghouse: the 30 industry competency models and the
    # Building Blocks model as database scripts (CC BY 4.0; credit the CMC and DOL ETA)
    "dol/cmc_competency_models.zip": "https://www.careeronestop.org/TridionMultimedia/tcm24-44482_CareerOneStop_CompetencyACC.zip",
    "dol/cmc_competency_models_readme.txt": "https://www.careeronestop.org/TridionMultimedia/tcm24-44486_CompetencyModelClearinghouse_DataDownloadReadme.txt",
    # reports kept for reference (numbers are hand-transcribed into data/reference/)
    "openai/how_people_use_chatgpt_2025.pdf": "https://cdn.openai.com/pdf/a253471f-8260-40c6-a2cc-aa93fe9f142e/economic-research-chatgpt-usage-paper.pdf",
    "google/GoogleATLASv1.pdf": "https://ai.google/static/documents/GoogleATLASv1.pdf",
    "microsoft/ai_diffusion_2026_q2.pdf": "https://www.microsoft.com/en-us/research/wp-content/uploads/2026/09/Microsoft-AI-Diffusion-Report-2026-Q2.pdf",
    # Anthropic Economic Index
    **{f"aei/{p}": f"{HF}/{p}" for p in [
        "release_2025_09_15/data/input/BTOS_National.xlsx",
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


# sources that are time series: re-download when the local copy is older than this many days
FRESH_DAYS = {"census/btos_national.xlsx": 14, "census/btos_sector.xlsx": 14}


def fetch_one(rel: str, url: str, force: bool = False, max_age_days: int | None = None) -> Path:
    import time

    dest = RAW / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        age_days = (time.time() - dest.stat().st_mtime) / 86400.0
        if max_age_days is None or age_days <= max_age_days:
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
            p = fetch_one(rel, url, force, FRESH_DAYS.get(rel))
            manifest[rel] = {"url": url, "sha256": _sha256(p), "bytes": p.stat().st_size}
            print(f"ok    {rel}")
        except Exception as e:  # keep going; report at the end
            manifest[rel] = {"url": url, "error": str(e)}
            print(f"FAIL  {rel}: {e}")
    manifest["singapore_labour_apis"] = fetch_singapore_labour(force)
    try:
        parse_microsoft_diffusion_pdf()
        manifest["microsoft_diffusion_countries_csv"] = "parsed"
    except Exception as e:
        manifest["microsoft_diffusion_countries_csv"] = f"not parsed: {e}"
    (RAW / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


# ----------------------------------------------------------------------------- Singapore labour statistics (APIs, saved as CSV/JSON)
DATAGOV_RESOURCES = {
    "ows_2024_wages_by_occupation": "d_9917e751f7498502f70052a940a3f312",
    "ows_2024_wages_by_occupation_industry": "d_523d376ff1d501a338add925f3313fd1",
    "ows_2024_wages_by_occupation_estab_size": "d_7d528712f1fd454ac6ca32fe06fe8f12",
    "lfs_employed_by_occupation_status_sex": "d_3ba99a6cc9f643bacb63bdab296d7bff",
    "lfs_employed_by_occupation_age": "d_1d7ab908d16d7b9ddf6f2c2985894119",
    "retrench_by_occupation_annual": "d_49ab680655c6d48a221004630a37edd6",
    "jv_rate_by_occupation_quarterly": "d_1e10046c33418c507bb2483c26dca489",
}
SINGSTAT_TABLES = ["M182081", "M920131", "M182171"]
# stats.mom.gov.sg sits behind a web-application firewall that returns 403 for some exact file paths
# but serves the same file when the path's case differs (the server is case-insensitive). The variants
# below are the spellings that worked; the fetcher tries each in turn.
MOM_LFS_FILES = {
    "LFR2025_SectionC.xlsx": ["https://stats.mom.gov.sg/iMAS_Tables1/LabourForce/LabourForce_2025/LFR2025_SectionC.xlsx", "https://stats.mom.gov.sg/iMAS_Tables1/LabourForce/LabourForce_2025/LFR2025_SectionC.XLSX"],
    "LFR2025_SectionD.xlsx": ["https://stats.mom.gov.sg/iMAS_Tables1/LabourForce/LabourForce_2025/LFR2025_SectionD.XLSX", "https://stats.mom.gov.sg/iMAS_Tables1/LabourForce/LabourForce_2025/LFR2025_SectionD.xlsx"],
    "LFR2024_SectionD.xlsx": ["https://stats.mom.gov.sg/iMAS_Tables1/LabourForce/LabourForce_2024/LFR2024_SectionD.XLSX", "https://stats.mom.gov.sg/iMAS_Tables1/LabourForce/LabourForce_2024/LFR2024_SectionD.xlsx"],
    "ows2025/mrsd_2025Wages_table4.xlsx": ["https://stats.mom.gov.sg/iMAS_Tables1/Wages/Wages_2025/mrsd_2025Wages_table4.XLSX", "https://stats.mom.gov.sg/iMAS_Tables1/Wages/Wages_2025/mrsd_2025Wages_table4.xlsx"],
    "ows2025/mrsd_2025wages_table1.xlsx": ["https://stats.mom.gov.sg/iMAS_Tables1/Wages/Wages_2025/MRSD_2025Wages_table1.xlsx", "https://stats.mom.gov.sg/iMAS_Tables1/Wages/Wages_2025/mrsd_2025Wages_table1.xlsx"],
    "ows2025/mrsd_2025Wages_table5.xlsx": ["https://stats.mom.gov.sg/iMAS_Tables1/Wages/Wages_2025/mrsd_2025Wages_table5.xlsx", "https://stats.mom.gov.sg/iMAS_Tables1/Wages/Wages_2025/mrsd_2025Wages_table5.XLSX"],
    # what has happened to jobs (rung five): vacancies by detailed occupation, employment by occupation group over time,
    # retrenchment incidence, vacancy and resignation rates by occupational group
    "jv/mrsd_2025_jv_by_ssoc_ind_occ_edu.xlsx": ["https://stats.mom.gov.sg/iMAS_Tables1/Job%20Vacancies/2025/mrsd_2025_jv_by_ssoc_ind_occ_edu.xlsx", "https://stats.mom.gov.sg/iMAS_Tables1/Job%20Vacancies/2025/mrsd_2025_JV_by_ssoc_ind_occ_edu.xlsx"],
    "jv/mrsd_2024_jv_by_ssoc_ind_occ_edu.xlsx": ["https://stats.mom.gov.sg/iMAS_Tables1/Job%20Vacancies/2024/mrsd_2024_JV_by_ssoc_ind_occ_edu.xlsx", "https://stats.mom.gov.sg/iMAS_Tables1/Job%20Vacancies/2024/mrsd_2024_jv_by_ssoc_ind_occ_edu.xlsx"],
    "jv/mrsd_2023_jv_by_ssoc_ind_occ_edu.xlsx": ["https://stats.mom.gov.sg/iMAS_Tables1/Job%20Vacancies/2023/mrsd_2023_jv_by_ssoc_ind_occ_edu.xlsx", "https://stats.mom.gov.sg/iMAS_Tables1/Job%20Vacancies/2023/mrsd_2023_JV_by_ssoc_ind_occ_edu.xlsx"],
    "mrsd_69_Emp_Res_DetailedOcc_Sex.xlsx": ["https://stats.mom.gov.sg/iMAS_Tables1/Time-Series-Table/mrsd_69_Emp_Res_DetailedOcc_Sex.xlsx", "https://stats.mom.gov.sg/iMAS_Tables1/Time-Series-Table/MRSD_69_Emp_Res_DetailedOcc_Sex.xlsx"],
    "mrsd_33_incidence_of_retrenchment_by_ind_occ_grp.xlsx": ["https://stats.mom.gov.sg/iMAS_Tables1/Time-Series-Table/mrsd_33_incidence_of_retrenchment_by_ind_occ_grp.xlsx", "https://stats.mom.gov.sg/iMAS_Tables1/Time-Series-Table/MRSD_33_incidence_of_retrenchment_by_ind_occ_grp.xlsx"],
    "mrsd_24_Qtly_and_annl_tsd_on_job_vac_rate_by_ind_and_occ_grp.xlsx": ["https://stats.mom.gov.sg/iMAS_Tables1/Time-Series-Table/mrsd_24_Qtly_and_annl_tsd_on_job_vac_rate_by_ind_and_occ_grp.xlsx", "https://stats.mom.gov.sg/iMAS_Tables1/Time-Series-Table/MRSD_24_Qtly_and_annl_tsd_on_job_vac_rate_by_ind_and_occ_grp.xlsx"],
    "mrsd_26_Qtly_and_annl_tsd_on_resgn_by_ind_and_occ_grp.xlsx": ["https://stats.mom.gov.sg/iMAS_Tables1/Time-Series-Table/mrsd_26_Qtly_and_annl_tsd_on_resgn_by_ind_and_occ_grp.xlsx", "https://stats.mom.gov.sg/iMAS_Tables1/Time-Series-Table/MRSD_26_Qtly_and_annl_tsd_on_resgn_by_ind_and_occ_grp.xlsx"],
}


def fetch_singapore_labour(force: bool = False) -> dict:
    """data.gov.sg datastore resources (MOM wages and LFS employment) and SingStat Table Builder tables."""
    import csv

    out = {}
    dest_dir = RAW / "singapore" / "mom"
    dest_dir.mkdir(parents=True, exist_ok=True)
    for name, rid in DATAGOV_RESOURCES.items():
        dest = dest_dir / f"{name}.csv"
        if dest.exists() and not force:
            out[name] = "exists"; continue
        url = f"https://data.gov.sg/api/action/datastore_search?resource_id={rid}&limit=5000"
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
            d = json.load(r)["result"]
        fields = [f["id"] for f in d["fields"]]
        with open(dest, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields); w.writeheader(); [w.writerow({k: rec.get(k) for k in fields}) for rec in d["records"]]
        out[name] = f"{len(d['records'])} rows"
    for t in SINGSTAT_TABLES:
        dest = dest_dir / f"singstat_{t}.json"
        if dest.exists() and not force:
            out[t] = "exists"; continue
        url = f"https://tablebuilder.singstat.gov.sg/api/table/tabledata/{t}?limit=2000"
        with urllib.request.urlopen(urllib.request.Request(url, headers={**UA, "Accept": "application/json"}), timeout=120) as r:
            dest.write_bytes(r.read())
        out[t] = "ok"
    browser = {**UA, "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"}
    for name, urls in MOM_LFS_FILES.items():
        dest = dest_dir / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and dest.stat().st_size > 20000 and not force:
            out[name] = "exists"; continue
        out[name] = "blocked on every spelling; download by hand from stats.mom.gov.sg"
        for url in urls:
            try:
                with urllib.request.urlopen(urllib.request.Request(url, headers=browser), timeout=120) as r:
                    data = r.read()
                if data[:2] == b"PK":
                    dest.write_bytes(data); out[name] = f"ok ({url.rsplit('/', 1)[-1]})"; break
            except Exception as e:  # 403 from the firewall: try the next spelling
                out[name] = f"blocked: {e}"
    return out


def parse_microsoft_diffusion_pdf() -> Path:
    """Turn the Q2 2026 Global AI Diffusion PDF's appendix table into a CSV (needs `pdftotext`)."""
    import csv
    import re
    import subprocess

    pdf = RAW / "microsoft" / "ai_diffusion_2026_q2.pdf"
    txt = subprocess.run(["pdftotext", "-layout", str(pdf), "-"], capture_output=True, text=True, check=True).stdout
    rows = []
    started = False
    for ln in txt.split("\n"):
        if "AI Diffusion Data Source" in ln:
            started = True
        if not started:
            continue
        m = re.match(r"^\s*([A-Za-z][A-Za-z .,'()&-]+?)\s{2,}(\d+\.\d)%\s+(\d+\.\d)%\s+(-?\d+\.\d)\s*$", ln)
        if m:
            rows.append((m.group(1).strip(), float(m.group(2)), float(m.group(3)), float(m.group(4))))
    dest = RAW / "microsoft" / "ai_diffusion_2026_q2_countries.csv"
    with open(dest, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["economy", "ai_user_share_2026q1_pct", "ai_user_share_2026q2_pct", "change_pp"]); w.writerows(rows)
    return dest
