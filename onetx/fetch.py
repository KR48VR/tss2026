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

BROWSER_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
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
    # the same question by employment-size class (A 1-4 ... G 250+), for a US comparator matched to the firm
    # sizes other countries' surveys cover (10+ in Europe and the UK, 100+ in Japan)
    "census/btos_empsize.xlsx": "https://www.census.gov/hfp/btos/downloads/Employment%20Size%20Class.xlsx",
    # US Bureau of Labor Statistics, Occupational Employment and Wage Statistics (May 2024 and May 2023):
    # employment and wages by detailed occupation (national) and by NAICS sector (the in4 bundle; only its
    # sector file is used). The second workforce the map can be weighted by.
    "bls/oesm25nat.zip": "https://www.bls.gov/oes/special-requests/oesm25nat.zip",   # May 2025; the loaders use the latest edition on disk (BLS refuses downloads from cloud addresses: fetch these two in a browser if the fetch fails)
    "bls/oesm25in4.zip": "https://www.bls.gov/oes/special-requests/oesm25in4.zip",
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
    # ---- Australia (ABS) and the United Kingdom (ONS, Skills England): the third and fourth workforces
    "abs/lfs_EQ08.xlsx": "https://www.abs.gov.au/statistics/labour/employment-and-unemployment/labour-force-australia-detailed/mar-2026/EQ08.xlsx",
    "abs/lfs_EQ09.xlsx": "https://www.abs.gov.au/statistics/labour/employment-and-unemployment/labour-force-australia-detailed/mar-2026/EQ09.xlsx",
    "abs/eeh_63060DO011_202505.xlsx": "https://www.abs.gov.au/statistics/labour/earnings-and-working-conditions/employee-earnings-and-hours-australia/may-2025/63060DO011_202505.xlsx",
    "abs/eeh_63060DO011_202305.xlsx": "https://www.abs.gov.au/statistics/labour/earnings-and-working-conditions/employee-earnings-and-hours-australia/may-2023/63060DO011_202305.xlsx",
    "abs/cab_BCSDC06_2024-25.xlsx": "https://www.abs.gov.au/statistics/industry/technology-and-innovation/characteristics-australian-business/2024-25/BCSDC06_2024%E2%80%9325.xlsx",
    "abs/cab_81670DO007_202122.xlsx": "https://www.abs.gov.au/statistics/industry/technology-and-innovation/characteristics-australian-business/2021-22/81670DO007_202122.xlsx",
    "abs/anzsco2022_structure.xlsx": "https://www.abs.gov.au/statistics/classifications/anzsco-australian-and-new-zealand-standard-classification-occupations/2022/anzsco%202022%20structure%20062023.xlsx",
    "abs/anzsco2022_index.xlsx": "https://www.abs.gov.au/statistics/classifications/anzsco-australian-and-new-zealand-standard-classification-occupations/2022/anzsco%202022%20index%20of%20principal%20titles%2C%20alternative%20titles%20and%20specialisations%20062023.xlsx",
    "abs/anzsco2022_correspondence.xlsx": "https://www.abs.gov.au/statistics/classifications/anzsco-australian-and-new-zealand-standard-classification-occupations/2022/anzsco%202022%20correspondence%20tables.xlsx",
    "ons/bics/bicswave165.xlsx": "https://www.ons.gov.uk/file?uri=/economy/economicoutputandproductivity/output/datasets/businessinsightsandimpactontheukeconomy/bicswave165/bicswave1652final.xlsx",
    "ons/ashe/ashetable142026provisional.zip": "https://www.ons.gov.uk/file?uri=/employmentandlabourmarket/peopleinwork/earningsandworkinghours/datasets/occupation4digitsoc2010ashetable14/2026provisional/ashetable142026provisional.zip",
    "ons/ashe/ashetable142025revised.zip": "https://www.ons.gov.uk/file?uri=/employmentandlabourmarket/peopleinwork/earningsandworkinghours/datasets/occupation4digitsoc2010ashetable14/2025revised/ashetable142025revised.zip",
    "ons/ashe/ashetable142025provisional.zip": "https://www.ons.gov.uk/file?uri=/employmentandlabourmarket/peopleinwork/earningsandworkinghours/datasets/occupation4digitsoc2010ashetable14/2025provisional/ashetable142025provisional.zip",
    "ons/ashe/ashetable142024revised.zip": "https://www.ons.gov.uk/file?uri=/employmentandlabourmarket/peopleinwork/earningsandworkinghours/datasets/occupation4digitsoc2010ashetable14/2024revised/ashetable142024revised.zip",
    "ons/soc2020_coding_index.xlsx": "https://www.ons.gov.uk/file?uri=/methodology/classificationsandstandards/standardoccupationalclassificationsoc/soc2020/soc2020volume2codingrulesandconventions/soc2020volume2thecodingindexexcel20260827v2.xlsx",
    "ons/soc2020_structure.xlsx": "https://www.ons.gov.uk/file?uri=/methodology/classificationsandstandards/standardoccupationalclassificationsoc/soc2020/soc2020volume1structureanddescriptionsofunitgroups/soc2020volume1structureanddescriptionofunitgroupsexcel20260827.xlsx",
    "ons/census2021_occupation_industry.json": "https://api.beta.ons.gov.uk/v1/population-types/UR/census-observations?area-type=nat&dimensions=occupation_current_105a,industry_current_88a",
    # ---- Canada (Statistics Canada, ESDC): the fifth workforce
    # New Zealand: Stats NZ tables republished by Figure.NZ (CC BY 4.0), the source file and its tidy CSV
    # Eurostat (JSON-stat from the dissemination API) and Statistics Norway (JSON-stat 2.0 from data.ssb.no; see fetch_ssb)
    "eurostat/isoc_eb_ai.json": "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/isoc_eb_ai?format=JSON&lang=EN&indic_is=E_AI_TANY&unit=PC_ENT&size_emp=GE10",
    "eurostat/isoc_eb_ain2.json": "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/isoc_eb_ain2?format=JSON&lang=EN&indic_is=E_AI_TANY&unit=PC_ENT&size_emp=GE10",
    "eurostat/lfsa_egai2d.json": "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/lfsa_egai2d?format=JSON&lang=EN&sex=T&age=Y15-64&time=2024&time=2025",
    "eurostat/earn_ses22_14.json": "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/earn_ses22_14?format=JSON&lang=EN&sex=T&age=TOTAL&sizeclas=TOTAL&sizeclas=GE10&indic_se=ERN&unit=EUR&unit=NAC",
    "eurostat/earn_ses18_14.json": "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/earn_ses18_14?format=JSON&lang=EN&sex=T&age=TOTAL&sizeclas=TOTAL&sizeclas=GE10&indic_se=ERN&unit=EUR&unit=NAC",
    "eurostat/lfsa_eisn2.json": "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/lfsa_eisn2?format=JSON&lang=EN&sex=T&time=2024",
    "statsnz/census2023_occupation_sex_region.xlsx": "https://figure.nz/table/93JjK3JipxibN5z0/download-source-dataset",
    "statsnz/census_occupation_2013_2018_2023.csv": "https://figure.nz/table/rjkP7FejTiMetM8I/download",
    "statsnz/census_industry_2013_2018_2023.csv": "https://figure.nz/table/5NUkKEe3d3IlNraI/download",
    "statsnz/hlfs_earnings_by_occupation.csv": "https://figure.nz/table/tANdfeZpRDSl8UII/download",
    "statcan/noc2021_structure.csv": "https://www.statcan.gc.ca/en/subjects/standard/noc/2021/indexV1/noc-2021-v1.0-classification-structure.csv",
    "statcan/noc2021_elements.csv": "https://www.statcan.gc.ca/en/subjects/standard/noc/2021/indexV1/noc-2021-v1.0-elements.csv",
    "statcan/soc2018us-noc2016v1_3-eng.csv": "https://www.statcan.gc.ca/en/statistical-programs/document/soc2018us-noc2016v1_3-eng.csv",
    "statcan/noc2016v1_3-noc2021v1_0-eng.csv": "https://www.statcan.gc.ca/en/statistical-programs/document/noc2016v1_3-noc2021v1_0-eng.csv",
    "statcan/33100825.zip": "https://www150.statcan.gc.ca/n1/tbl/csv/33100825-eng.zip",
    "statcan/33101004.zip": "https://www150.statcan.gc.ca/n1/tbl/csv/33101004-eng.zip",
    "statcan/33101167.zip": "https://www150.statcan.gc.ca/n1/tbl/csv/33101167-eng.zip",
    "statcan/14100416.zip": "https://www150.statcan.gc.ca/n1/tbl/csv/14100416-eng.zip",
    "esdc/wages/wages2025.csv": "https://open.canada.ca/data/dataset/adad580f-76b0-4502-bd05-20c125de9116/resource/9da94d63-b178-4a64-aeb3-b6a3bd721ad2/download/2a71-das-wage2025opendata-esdc-all-19nov2025-vf.csv",
    "esdc/wages/wages2024.csv": "https://open.canada.ca/data/dataset/adad580f-76b0-4502-bd05-20c125de9116/resource/d16e10ea-77bf-4db8-bdb5-adc709e6cada/download/2a71-das-wage2024opendata-esdc-all-11dec2024-vf.csv",
    "esdc/wages/wages2023.csv": "https://open.canada.ca/data/dataset/adad580f-76b0-4502-bd05-20c125de9116/resource/ff45366b-1c17-4862-8325-f6e7797c7c56/download/2a71-das-wage2023opendata-esdc-all-08dec2023-vf.csv",
    "esdc/oasis/labels.csv": "https://open.canada.ca/data/dataset/10ce43bd-fb58-4969-806b-4bffebc87bec/resource/af53ca1d-5efc-4e52-ad42-9b858a4507c6/download/labels_oasis_2025_v1.1.csv",
    "esdc/oasis/lead_statement.csv": "https://open.canada.ca/data/dataset/10ce43bd-fb58-4969-806b-4bffebc87bec/resource/d2a00c9b-2b7d-458f-a41f-10011d8164c7/download/lead-statement_oasis_2025_v1.0.csv",
    "esdc/oasis/main_duties.csv": "https://open.canada.ca/data/dataset/10ce43bd-fb58-4969-806b-4bffebc87bec/resource/92e4107e-a804-4126-85ec-1fca23b82d81/download/main-duties_oasis-2025_v1.1.csv",
    "esdc/oasis/example_titles.csv": "https://open.canada.ca/data/dataset/10ce43bd-fb58-4969-806b-4bffebc87bec/resource/3484364c-75a4-4e22-ae9a-d12ea72569d4/download/example-titles_oasis_2025_v1.0.csv",
    "esdc/oasis/skills.csv": "https://open.canada.ca/data/dataset/10ce43bd-fb58-4969-806b-4bffebc87bec/resource/bd9cf3f2-4cd5-47b1-ba74-ab68cd0a941b/download/skills_oasis_2025_v1.1.csv",
    "esdc/oasis/knowledge.csv": "https://open.canada.ca/data/dataset/10ce43bd-fb58-4969-806b-4bffebc87bec/resource/b5c3eb08-dda8-4929-ba5d-b90582db6c70/download/knowledge_oasis_2025_v.1.1.csv",
    "esdc/oasis/work_activities.csv": "https://open.canada.ca/data/dataset/10ce43bd-fb58-4969-806b-4bffebc87bec/resource/44e666cc-f12d-4b3f-9369-6e0843d2bb48/download/work-activities_oasis_2025_v1.1.csv",
    "esdc/oasis/abilities.csv": "https://open.canada.ca/data/dataset/10ce43bd-fb58-4969-806b-4bffebc87bec/resource/f015095d-15e7-4ef7-97d1-e5fed3486e40/download/abilities_oasis_2025_v1.1.csv",
    "esdc/oasis/employment_requirements.csv": "https://open.canada.ca/data/dataset/10ce43bd-fb58-4969-806b-4bffebc87bec/resource/33b4ccb4-e725-4618-ab0b-f64c979ab9cc/download/employment-requirements_oasis_2025_v1.0.csv",
    "esdc/oasis/taxonomy.csv": "https://open.canada.ca/data/dataset/10ce43bd-fb58-4969-806b-4bffebc87bec/resource/2a7a17bf-b67c-4fc2-b636-959c7f1d7ae4/download/guide_oasis_2025_v4.0.csv",
    "ifate/apprenticeship_standards.json": "https://www.instituteforapprenticeships.org/api/apprenticeshipstandards",
}


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# sources that are time series: re-download when the local copy is older than this many days
FRESH_DAYS = {"census/btos_national.xlsx": 14, "census/btos_sector.xlsx": 14, "census/btos_empsize.xlsx": 14}


def fetch_one(rel: str, url: str, force: bool = False, max_age_days: int | None = None) -> Path:
    import time

    dest = RAW / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        age_days = (time.time() - dest.stat().st_mtime) / 86400.0
        if max_age_days is None or age_days <= max_age_days:
            return dest
    headers = {**UA, "User-Agent": BROWSER_UA} if rel.split("/")[0] in ("abs", "ons", "ifate", "statcan", "esdc", "statsnz", "eurostat") else UA
    req = urllib.request.Request(url, headers=headers)
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
        manifest["further_offices"] = fetch_further_offices(force)
    except Exception as e:
        manifest["further_offices"] = f"failed: {e}"
    try:
        manifest["estat"] = fetch_estat(force)
    except Exception as e:
        manifest["estat"] = f"failed: {e}"
    try:
        manifest["ssb"] = fetch_ssb(force)
    except Exception as e:
        manifest["ssb"] = f"failed: {e}"
    try:
        manifest["statsnz_ade"] = fetch_statsnz_ade(force=force)   # only with STATSNZ_API_KEY set; otherwise records why it was skipped
    except Exception as e:
        manifest["statsnz_ade"] = f"failed: {e}"
    try:
        manifest["kosis"] = fetch_kosis(force)
    except Exception as e:
        manifest["kosis"] = f"failed: {e}"
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


# ----------------------------------------------------------------------------- the ABS's ANZSCO pages (one per unit group)
def fetch_anzsco_pages(force: bool = False, pause: float = 0.7) -> dict:
    """Download the ABS classification page of every ANZSCO 2022 unit group (364 pages, about five minutes)
    into data/raw/abs/anzsco_pages/<unit>.html. The pages carry the official 'Tasks Include' lists."""
    import time

    from .au_labour import ABS, load_anzsco_structure

    out_dir = ABS / "anzsco_pages"
    out_dir.mkdir(parents=True, exist_ok=True)
    units = load_anzsco_structure()
    ok, failed = 0, []
    for r in units.itertuples():
        dest = out_dir / f"{r.anzsco_code}.html"
        if dest.exists() and dest.stat().st_size > 20000 and not force:
            ok += 1
            continue
        url = f"https://www.abs.gov.au/statistics/classifications/anzsco-australian-and-new-zealand-standard-classification-occupations/2022/browse-classification/{r.major_group}/{r.submajor}/{r.minor}/{r.anzsco_code}"
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={**UA, "User-Agent": BROWSER_UA}), timeout=120) as resp:
                dest.write_bytes(resp.read())
            ok += 1
        except Exception as e:  # keep going; report at the end
            failed.append({"unit": r.anzsco_code, "error": str(e)})
        time.sleep(pause)
    return {"pages": ok, "failed": failed}


# ----------------------------------------------------------------------------- Statistics Canada cells pulled through the WDS API
def _post_json(url: str, body: dict, timeout: int = 600, headers: dict | None = None) -> bytes:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json", "User-Agent": BROWSER_UA, **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _pxweb_query(url: str, selections: dict, fmt: str = "json-stat2") -> bytes:
    """A PxWeb v1 query (Statistics Sweden, Statistics Finland): {variable code: [values] | "*"}."""
    q = [{"code": k, "selection": {"filter": "all", "values": ["*"]} if v == "*" else {"filter": "item", "values": list(v)}} for k, v in selections.items()]
    return _post_json(url, {"query": q, "response": {"format": fmt}})


def fetch_further_offices(force: bool = False) -> dict:
    """The open tables behind Sweden, Finland, Denmark, the Netherlands and Ireland, as JSON-stat or JSON under
    data/raw/<office>/. Each pull is one query to the office's own API; the file name says what it holds."""
    jobs = {
        # Sweden (SCB PxWeb): employees by 4-digit SSYK 2012 (register, 2022-2024), salary by SSYK (2023-2025), AI use by study domain (2021-2025)
        "scb/yreg50bas_employees_by_ssyk.json": ("https://api.scb.se/OV0104/v1/doris/en/ssd/AM/AM0208/AM0208E/YREG50BAS", {"Yrke2012": "*", "ArbetsSektor": "*", "Kon": "*", "ContentsCode": "*", "Tid": ["2022", "2023", "2024"]}),
        "scb/lone_salary_by_ssyk.json": ("https://api.scb.se/OV0104/v1/doris/en/ssd/AM/AM0110/AM0110A/LoneSpridSektYrk4AN", {"Sektor": ["0"], "Yrke2012": "*", "Kon": ["1+2"], "ContentsCode": ["000007CD", "000007CE"], "Tid": "*"}),
        "scb/ai_use_by_domain.json": ("https://api.scb.se/OV0104/v1/doris/en/ssd/NV/NV0116/NV0116M/AiTeknikerTypN", {"TypAvTeknik": ["0080"], "Redovisningsgrupp": "*", "ContentsCode": ["000007JC"], "Tid": "*"}),
        # Finland (StatFin PxWeb): employed by occupation (register), structure of earnings by occupation, ICT use in enterprises (AI)
        "statfin/tyokay_115q_employed_by_occupation.json": ("https://pxdata.stat.fi/PxWeb/api/v1/en/StatFin/tyokay/115q.px", {"ammatti_104_20161021": "*", "sukupuoli_9_20180101": ["SSS"], "ammattiasema_7_20121201": ["SS"], "timeperiod_y": ["2022", "2023", "2024"], "contentscode": ["tyokay-tyolliset2"]}),
        "statfin/pra_15ay_earnings_by_occupation.json": ("https://pxdata.stat.fi/PxWeb/api/v1/en/StatFin/pra/15ay.px", {"timeperiod_y": "*", "sektoriluokitus_7_20230101": ["S0"], "ammatti_19_20180101": "*", "tyoaika_4_20180128": ["SSS"], "contentscode": "*"}),
        "statfin/icte_14yc_ict_in_enterprises.json": ("https://pxdata.stat.fi/PxWeb/api/v1/en/StatFin/icte/14yc.px", {"timeperiod_y": "*", "toimiala_79_20180101": "*", "contentscode": ["icte22"]}),
    }
    out = {}
    for rel, (url, sel) in jobs.items():
        f = RAW / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        if f.exists() and not force:
            out[rel] = "cached"; continue
        try:
            f.write_bytes(_pxweb_query(url, sel)); out[rel] = f"{f.stat().st_size} bytes"
        except Exception as e:
            out[rel] = f"failed: {e}"
    # Denmark (Statistics Denmark API): earnings by DISCO-08 occupation; enterprises' use of AI by industry
    dk = {
        "dst/lons20_earnings_by_occupation.json": ("LONS20", {"ARBF": ["*"], "SEKTOR": ["1000"], "AFLOEN": ["TIFA"], "LONGRP": ["LTOT"], "LØNMÅL": ["FORINKL"], "KØN": ["TOT"], "Tid": ["2023", "2024", "2025"]}),
        "dst/itav28_ai_use_by_industry.json": ("ITAV28", {"VIRKSTRRDA": ["*"], "BRANCHEDB25UDVALG": ["*"], "ANVEND": ["1680"], "Tid": ["*"]}),
    }
    for rel, (table, variables) in dk.items():
        f = RAW / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        if f.exists() and not force:
            out[rel] = "cached"; continue
        try:
            body = {"table": table, "format": "JSONSTAT", "lang": "en", "variables": [{"code": k, "values": v} for k, v in variables.items()]}
            f.write_bytes(_post_json("https://api.statbank.dk/v1/data", body)); out[rel] = f"{f.stat().st_size} bytes"
        except Exception as e:
            out[rel] = f"failed: {e}"
    # Netherlands (CBS StatLine OData): employed by occupation group, hourly wage by occupation group, ICT use by industry (AI) 2023-2025
    nl = {
        "cbs/86320NED_employed_by_occupation.json": "https://opendata.cbs.nl/ODataApi/odata/86320NED/TypedDataSet?$format=json&$filter=Geslacht%20eq%20%27T001038%27%20and%20Persoonskenmerken%20eq%20%27T009002%27",
        "cbs/86320NED_Beroep.json": "https://opendata.cbs.nl/ODataApi/odata/86320NED/Beroep?$format=json",
        "cbs/86355NED_hourly_wage_by_occupation.json": "https://opendata.cbs.nl/ODataApi/odata/86355NED/TypedDataSet?$format=json",
        "cbs/86355NED_Beroep.json": "https://opendata.cbs.nl/ODataApi/odata/86355NED/Beroep?$format=json",
        "cbs/86120NED_ict_by_industry_2025.json": "https://opendata.cbs.nl/ODataApi/odata/86120NED/TypedDataSet?$format=json",
        "cbs/86120NED_industries.json": "https://opendata.cbs.nl/ODataApi/odata/86120NED/BedrijfstakkenBranchesSBI2008?$format=json",
        "cbs/86075NED_ict_by_industry_2024.json": "https://opendata.cbs.nl/ODataApi/odata/86075NED/TypedDataSet?$format=json",
        "cbs/86075NED_industries.json": "https://opendata.cbs.nl/ODataApi/odata/86075NED/BedrijfstakkenBranchesSBI2008?$format=json",
        "cbs/85735NED_ict_by_industry_2023.json": "https://opendata.cbs.nl/ODataApi/odata/85735NED/TypedDataSet?$format=json",
        "cbs/85735NED_industries.json": "https://opendata.cbs.nl/ODataApi/odata/85735NED/BedrijfstakkenBranchesSBI2008?$format=json",
        "cbs/85735NED_sizes.json": "https://opendata.cbs.nl/ODataApi/odata/85735NED/Bedrijfsgrootte?$format=json",
    }
    for rel, url in nl.items():
        f = RAW / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        if f.exists() and not force:
            out[rel] = "cached"; continue
        try:
            req = urllib.request.Request(url, headers={"User-Agent": BROWSER_UA, "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=600) as r:
                f.write_bytes(r.read())
            out[rel] = f"{f.stat().st_size} bytes"
        except Exception as e:
            out[rel] = f"failed: {e}"
    # Ireland (CSO PxStat JSON-RPC): Census labour force by detailed occupational group; enterprise use of AI
    ie = {"cso/F7042_labour_force_by_detailed_occupation.json": "F7042", "cso/ICA237_enterprise_ai_use.json": "ICA237"}
    for rel, matrix in ie.items():
        f = RAW / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        if f.exists() and not force:
            out[rel] = "cached"; continue
        try:
            body = {"jsonrpc": "2.0", "method": "PxStat.Data.Cube_API.ReadDataset", "params": {"class": "query", "id": [], "dimension": {}, "extension": {"pivot": None, "codes": False, "language": {"code": "en"}, "format": {"type": "JSON-stat", "version": "2.0"}, "matrix": matrix}, "version": "2.0"}}
            f.write_bytes(_post_json("https://ws.cso.ie/public/api.jsonrpc", body)); out[rel] = f"{f.stat().st_size} bytes"
        except Exception as e:
            out[rel] = f"failed: {e}"
    return out


ESTAT_FILES = {
    # e-Stat file ids (fileKind 4 = Excel for the wage survey tables, 1 = CSV for the communications survey)
    "estat/wage2024_t1_occupation_small.xlsx": ("000040247854", 4),
    "estat/wage2024_t20_occupation_major_industry.xlsx": ("000040247875", 4),
    "estat/wage2025_t1_occupation_small.xlsx": ("000040421116", 4),          # 令和7年 (2025) edition, released 2026-03-24
    "estat/wage2025_t20_occupation_major_industry.xlsx": ("000040421137", 4),
    "estat/mic_iot_ai_adoption_2018.csv": ("000031831159", 1), "estat/mic_iot_ai_adoption_2019.csv": ("000031951199", 1), "estat/mic_iot_ai_adoption_2020.csv": ("000032097896", 1),
    "estat/mic_iot_ai_adoption_2021.csv": ("000032203902", 1), "estat/mic_iot_ai_adoption_2022.csv": ("000040056827", 1), "estat/mic_iot_ai_adoption_2023.csv": ("000040185424", 1),
    "estat/mic_iot_ai_adoption_2024.csv": ("000040278570", 1), "estat/mic_iot_ai_adoption_2025.csv": ("000040458767", 1),
}


def fetch_estat(force: bool = False) -> dict:
    """Japan's files from e-Stat's file-download endpoint (no key needed): the wage survey tables and the
    communications survey's IoT/AI question by industry, 2018 to 2025."""
    out = {}
    for rel, (sid, kind) in ESTAT_FILES.items():
        f = RAW / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        if f.exists() and not force:
            out[rel] = "cached"; continue
        try:
            req = urllib.request.Request(f"https://www.e-stat.go.jp/stat-search/file-download?statInfId={sid}&fileKind={kind}", headers={"User-Agent": BROWSER_UA, "Referer": "https://www.e-stat.go.jp/stat-search/files", "Accept-Language": "ja,en"})
            with urllib.request.urlopen(req, timeout=300) as r:
                f.write_bytes(r.read())
            out[rel] = f"{f.stat().st_size} bytes"
        except Exception as e:
            out[rel] = f"failed: {e}"
    return out


def fetch_ssb(force: bool = False) -> dict:
    """Statistics Norway's three tables through the data.ssb.no JSON query API (POST), saved as JSON-stat 2.0:
    11658 (employees, jobs and earnings by occupation, second quarters 2024-2026), 13265 and 14942 (AI use by
    industry, 2021-2025 and 2026)."""
    dest = RAW / "ssb"
    dest.mkdir(parents=True, exist_ok=True)
    queries = {
        "11658_employees_earnings_by_occupation.json": (11658, [{"code": "Kjonn", "selection": {"filter": "item", "values": ["0"]}}, {"code": "Alder", "selection": {"filter": "item", "values": ["999D"]}}, {"code": "Yrke", "selection": {"filter": "all", "values": ["*"]}},
                                                                {"code": "ContentsCode", "selection": {"filter": "item", "values": ["Lonsstakere", "AntArbForhold", "HeltidsEkvMnd", "GjMdTotal", "MedianMndLonn", "GjAvtaltMdlonn"]}}, {"code": "Tid", "selection": {"filter": "item", "values": ["2024K2", "2025K2", "2026K2"]}}]),
        "13265_ai_use_by_industry_2021_2025.json": (13265, [{"code": "SyssGrpIKT", "selection": {"filter": "all", "values": ["*"]}}, {"code": "NACE2007", "selection": {"filter": "all", "values": ["*"]}}, {"code": "ContentsCode", "selection": {"filter": "item", "values": ["BrukarKunstigIntelli"]}}, {"code": "Tid", "selection": {"filter": "all", "values": ["*"]}}]),
        "14942_ai_use_by_industry_2026.json": (14942, [{"code": "NACE2025", "selection": {"filter": "all", "values": ["*"]}}, {"code": "SyssGrpIKT", "selection": {"filter": "all", "values": ["*"]}}, {"code": "ContentsCode", "selection": {"filter": "item", "values": ["BrukarKunstigIntelli"]}}, {"code": "Tid", "selection": {"filter": "all", "values": ["*"]}}]),
    }
    out = {}
    for name, (table, query) in queries.items():
        f = dest / name
        if f.exists() and not force:
            out[name] = "cached"
            continue
        body = json.dumps({"query": query, "response": {"format": "json-stat2"}}).encode()
        req = urllib.request.Request(f"https://data.ssb.no/api/v0/en/table/{table}", data=body, headers={"Content-Type": "application/json", "User-Agent": BROWSER_UA})
        with urllib.request.urlopen(req, timeout=600) as r:
            f.write_bytes(r.read())
        out[name] = f"{f.stat().st_size} bytes"
    return out


def fetch_statsnz_ade(key: str | None = None, force: bool = False) -> dict:
    """The two Census 2023 tables New Zealand lacks openly, from the Aotearoa Data Explorer's SDMX service,
    which needs a (free, registered) Stats NZ API subscription key: income bands by occupation (CEN23_INC_007)
    and occupation by industry (CEN23_WRK_008). Pass the key or set STATSNZ_API_KEY. Written as CSV with
    labels to data/raw/statsnz/ade/; the New Zealand loaders use them when present."""
    import os

    key = key or os.environ.get("STATSNZ_API_KEY")
    out = {}
    if not key:
        return {"skipped": "no STATSNZ_API_KEY; New Zealand pay stays at the major-group level and no industry mix is used"}
    dest = RAW / "statsnz" / "ade"
    dest.mkdir(parents=True, exist_ok=True)
    for flow in ("CEN23_INC_007", "CEN23_WRK_008"):
        f = dest / f"{flow}.csv"
        if f.exists() and not force:
            out[flow] = "cached"
            continue
        url = f"https://api.data.stats.govt.nz/rest/data/STATSNZ,{flow},1.0/all?dimensionAtObservation=AllDimensions&format=csvfilewithlabels"
        req = urllib.request.Request(url, headers={"User-Agent": BROWSER_UA, "Ocp-Apim-Subscription-Key": key, "Accept": "text/csv"})
        with urllib.request.urlopen(req, timeout=600) as r:
            f.write_bytes(r.read())
        out[flow] = f"{f.stat().st_size} bytes"
    return out


def fetch_statcan_api(force: bool = False) -> dict:
    """The Canada-level cells of four Statistics Canada tables whose full downloads run to gigabytes:
    Census 2021 employment by NOC unit group (98-10-0449), employment income by unit group (98-10-0586),
    employment by minor group and NAICS sector (98-10-0594), and job vacancies by unit group (14-10-0444).
    Writes data/raw/statcan/api/api_<table>_<name>.csv plus the dimension member lists."""
    import csv
    import time

    out_dir = RAW / "statcan" / "api"
    out_dir.mkdir(parents=True, exist_ok=True)
    hdr = {**UA, "User-Agent": BROWSER_UA, "Content-Type": "application/json"}
    pids = [98100449, 98100586, 98100594, 14100444]
    req = urllib.request.Request("https://www150.statcan.gc.ca/t1/wds/rest/getCubeMetadata", data=json.dumps([{"productId": p} for p in pids]).encode(), headers=hdr)
    with urllib.request.urlopen(req, timeout=180) as r:
        meta = json.load(r)
    dims = {str(x["object"]["productId"]): x["object"]["dimension"] for x in meta}
    for pid in pids:
        (out_dir / f"members_{pid}.json").write_text(json.dumps([{"dim": d["dimensionNameEn"], "pos": d["dimensionPositionId"], "members": [{"id": m["memberId"], "name": m["memberNameEn"], "parent": m.get("parentMemberId")} for m in d["member"]]} for d in dims[str(pid)]]))

    def pull(pid, coords, name, latest=1):
        dest = out_dir / f"api_{pid}_{name}.csv"
        if dest.exists() and not force:
            return str(dest)
        rows = []
        for i in range(0, len(coords), 250):
            body = json.dumps([{"productId": pid, "coordinate": c, "latestN": latest} for c in coords[i:i + 250]]).encode()
            with urllib.request.urlopen(urllib.request.Request("https://www150.statcan.gc.ca/t1/wds/rest/getDataFromCubePidCoordAndLatestNPeriods", data=body, headers=hdr), timeout=180) as r:
                for x in json.load(r):
                    o = x.get("object") or {}
                    for p in o.get("vectorDataPoint", []):
                        rows.append({"coordinate": o.get("coordinate"), "refPer": p.get("refPer"), "value": p.get("value"), "status": x.get("status")})
            time.sleep(0.5)
        with open(dest, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["coordinate", "refPer", "value", "status"]); w.writeheader(); w.writerows(rows)
        return str(dest)

    mem = lambda pid, i: [m["memberId"] for m in dims[str(pid)][i]["member"]]
    out = {}
    out["employed"] = pull(98100449, [f"1.1.1.1.{o}.2.0.0.0.0" for o in mem(98100449, 4)], "employed")
    wa = dims["98100586"][4]["member"]
    fyft = next((m["memberId"] for m in wa if "full year" in m["memberNameEn"].lower() and "full time" in m["memberNameEn"].lower()), 1)
    out["income"] = pull(98100586, [f"1.1.1.1.{w}.1.{o}.{s}.0.0" for w in (1, fyft) for o in mem(98100586, 6) for s in (2, 3)], "income")
    out["vacancies"] = pull(14100444, [f"1.{o}.{s}.0.0.0.0.0.0.0" for o in mem(14100444, 1) for s in (1, 2)], "vacancies", latest=9)
    out["occ_industry"] = pull(98100594, [f"1.{i}.1.1.1.{o}.2.0.0.0" for i in mem(98100594, 1) for o in mem(98100594, 5)], "occ_industry")
    return out


# ----------------------------------------------------------------------------- South Korea: KOSIS tables without an API key
# KOSIS membership (and so an API key) needs a Korean mobile phone, but any visitor can download a table from its page.
# Each entry: table id -> (organisation id, classification levels to tick per tree, how many latest periods, note).
# levels: {tree number: [levels]} ticks every node at those levels in that classification tree; None keeps the page's default.
KOSIS_TABLES = {
    "DT_1ES3B32J": ("101", {"1": [3]}, 5, "Local Area Labour Force Survey: employed by three-digit occupation, half-yearly"),
    "DT_1ES3B33J": ("101", None, 1, "Local Area Labour Force Survey: employed by industry section and occupation major group"),
    "DT_118N_PAYM41": ("118", {"1": [3], "2": [1]}, 2, "Survey on Labour Conditions by Employment Type: workers, wages and hours by three-digit occupation and sex, yearly"),
    "DT_1EP1237_1": ("101", {"1": [1, 2]}, None, "Survey of Business Activities: fourth-industrial-revolution technologies by industry, 2017 to 2022"),
    "DT_1EP1337_1": ("101", {"1": [1, 2]}, None, "Survey of Business Activities: the same, 2023 onward"),
}


async def _kosis_download(org: str, tbl: str, out_csv: Path, levels: dict | None, n_periods: int | None, log=print) -> dict:
    """Open the table page in a headless browser, tick the classification levels and periods in its query settings
    the way a visitor does, run the search, then save the page's own CSV download (with codes). The page's
    classification tree (Korean and English titles per code) is saved beside the CSV."""
    import json as _json
    from playwright.async_api import async_playwright
    async with async_playwright() as p:
        b = await p.chromium.launch(headless=True, executable_path=CHROME_PATH if Path(CHROME_PATH).exists() else None, args=["--no-sandbox"])
        ctx = await b.new_context(viewport={"width": 1500, "height": 1000}, accept_downloads=True); page = await ctx.new_page()
        searches, tree = [], {}
        async def on_resp(r):
            if r.url.endswith("/statHtml/html.do"): searches.append(r.url)
            if r.url.endswith("/statHtml/treeInfo.do") and "tree" not in tree: tree["tree"] = await r.text()
        page.on("response", on_resp)
        for attempt in range(4):
            try:
                await page.goto(f"https://kosis.kr/statHtml/statHtml.do?orgId={org}&tblId={tbl}", wait_until="load", timeout=120000); break
            except Exception as e:
                log(f"{tbl}: page load failed ({str(e)[:60]}), retry {attempt + 1}"); await page.wait_for_timeout(4000 * (attempt + 1))
                if attempt == 3: raise
        fr = None
        for _ in range(90):
            fr = next((f for f in page.frames if f.name == "iframe_centerMenu"), None)
            if fr is not None and searches:
                try:
                    if await fr.evaluate("typeof fn_search==='function' && typeof $!=='undefined' && !!$.ui.fancytree.getTree('#fancytree_1')"): break
                except Exception: pass
            await page.wait_for_timeout(1000)
        else:
            raise RuntimeError(f"{tbl}: the table page did not finish loading")
        await page.wait_for_timeout(2000)
        if levels:
            for i, keep in levels.items():
                await fr.evaluate("([i,keep])=>{ const t=$.ui.fancytree.getTree('#fancytree_'+i); t.visit(x=>{ const l=x.data.lvl||x.getLevel(); x.setSelected(keep.includes(l)); }); }", [int(i), keep])
        n0 = len(searches)
        ttype = await fr.evaluate("(typeof timeTreeList!=='undefined' && timeTreeList.length) ? timeTreeList[0] : 'H'")
        avail = await fr.evaluate("(tt)=>{ const t=$.ui.fancytree.getTree('#timePopList'+tt); const a=[]; t.visit(x=>a.push(x.data.prdDe||x.key)); return a; }", ttype)
        want = avail if n_periods is None else avail[:n_periods]
        await fr.evaluate("([want,tt])=>{ const t=$.ui.fancytree.getTree('#timePopList'+tt); t.visit(x=>x.setSelected(want.includes(x.data.prdDe||x.key))); }", [want, ttype])
        await fr.evaluate("fn_searchPopPrd('prd')"); await page.wait_for_timeout(1500)   # the panel's own 'apply periods' action, which also runs the search
        if len(searches) == n0:
            await fr.evaluate("fn_search()")
        for _ in range(240):
            if len(searches) > n0: break
            await page.wait_for_timeout(500)
        await page.wait_for_timeout(3000)
        await fr.evaluate("popupControl('pop_downgrid','show','modal')"); await page.wait_for_timeout(800)
        await fr.evaluate("(()=>{ const c=document.getElementById('csvradio'); if(c){ c.checked=true; if(typeof fn_clickFileType==='function') fn_clickFileType(); } const k=document.getElementById('codeYn'); if(k) k.checked=true; const s=document.getElementById('csvType'); if(s){ for(const o of s.options){ if(/utf/i.test(o.textContent)) s.value=o.value; } } })()")
        async with page.expect_download(timeout=240000) as dl:
            await fr.evaluate("fn_downGridSubmit()")
        d = await dl.value; out_csv.parent.mkdir(parents=True, exist_ok=True); await d.save_as(out_csv)
        await b.close()
    info = {"file": str(out_csv), "periods": want}
    if tree.get("tree"):
        i = tree["tree"].find('var source = [{"title":'); j = tree["tree"].find("];", i)
        try:
            src = _json.loads(tree["tree"][i + len("var source = "):j + 1])
            flat = []
            def walk(ns):
                for n in ns:
                    flat.append({"code": n["itmId"], "lvl": n.get("lvl"), "ko": n["title"], "en": n.get("scrEng", "")}); walk(n.get("children") or [])
            walk(src)
            if flat and flat[0].get("code") == "0":   # the first classification tree is the occupation (or industry) tree
                (out_csv.parent / "classification_titles.json").write_text(_json.dumps(flat, ensure_ascii=False), encoding="utf-8")
                info["classification_titles"] = len(flat)
        except Exception as e:
            info["classification_titles"] = f"not parsed: {e}"
    return info


CHROME_PATH = "/opt/pw-browsers/chromium-1243/chrome-linux64/chrome"


def fetch_kosis_table(org: str, tbl: str, out_dir: Path | None = None, levels: dict | None = None, n_periods: int | None = None) -> dict:
    """Download one KOSIS table through its page (see _kosis_download). Returns what was saved."""
    import asyncio
    out_dir = out_dir or (RAW / "kosis" / tbl)
    return asyncio.run(_kosis_download(org, tbl, out_dir / f"{tbl}.csv", levels, n_periods))


def fetch_kosis(force: bool = False) -> dict:
    """South Korea's tables, one by one with a pause between them (the site flags bursts as abnormal use)."""
    import time
    out = {}
    for tbl, (org, levels, n_periods, note) in KOSIS_TABLES.items():
        f = RAW / "kosis" / tbl / f"{tbl}.csv"
        if f.exists() and not force:
            out[tbl] = "cached"; continue
        try:
            out[tbl] = fetch_kosis_table(org, tbl, levels=levels, n_periods=n_periods)
            if tbl == "DT_1ES3B32J":   # the occupation tree with English titles feeds the KSCO titles
                src = f.parent / "classification_titles.json"
                if src.exists():
                    src.replace(f.parent / "ksco7_titles.json")
            time.sleep(5)
        except Exception as e:
            out[tbl] = f"failed: {e}"
    return out
