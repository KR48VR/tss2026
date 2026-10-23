# Data sources

All files are public. `python -m onetx fetch` downloads them into `data/raw/` and writes `data/raw/manifest.json` with SHA-256 hashes. Nothing under `data/` is committed.

| Source | Files | Licence / terms | Notes |
|---|---|---|---|
| **O*NET 30.1** (Dec 2025) and **31.0** (Aug 2026), National Center for O*NET Development | `db_30_1_text.zip`, `db_31_0_text.zip` from onetcenter.org | CC BY 4.0 | 31.0 renumbers DWAs/IWAs and introduces Essential/Transferable skills; both are loaded |
| **O*NET crosswalks** | O*NET-SOC 2019 ↔ SOC 2018; O*NET-SOC 2010 ↔ 2019; ESCO ↔ O*NET-SOC | CC BY 4.0 | ESCO codes carry the ISCO-08 unit group |
| **BLS SOC 2010 ↔ ISCO-08** | mirrored as CSV in Daniel Russ's `codingsystems` repo (bls.gov blocks automated downloads) | public domain (US gov) | the only SOC→ISCO table the BLS publishes is for SOC 2010; we bridge via O*NET's 2010→2019 walk |
| **SingStat SSOC 2024 and 2020** | classification structure, detailed definitions (with unit-group task bullets), alphabetical index (tags Skills Framework roles), SSOC↔ISCO-08 and SSOC 2024↔2020 correspondence tables | Singapore Open Data Licence | from the SSOC page at singstat.gov.sg (go.gov.sg/ssoc) |
| **SkillsFuture Skills Framework dataset** (Q3 2026), **Unique Skills List**, **TSC-to-Unique-Skills mapping** | three `.xlsx` from the Jobs-Skills Portal's go.gov.sg short links | SSG terms of use on the portal | the portal pages themselves sit behind a bot check; the `file.go.gov.sg` links worked. If they stop working, download manually from jobsandskills.swda.gov.sg → Skills Frameworks → "Download the Latest Skills Framework Dataset" and drop the files into `data/raw/singapore/` with the names in `onetx/fetch.py` |
| **Anthropic Economic Index** | releases 2025-02-10, 2025-03-27, 2025-09-15, 2026-01-15, 2026-03-24, 2026-06-26; `labor_market_impacts/`; `robot_exposure/` | CC BY (data), MIT (code) | Hugging Face `Anthropic/EconomicIndex`. Task-level usage is keyed by task text in the 2025 files and by task id from 2026-06 |
| **Microsoft "Working with AI"** (Tomlinson et al., NeurIPS 2026) | `iwa_metrics.csv`, `soc_metrics.csv`, `ai_applicability_scores.csv`, `physical_tasks.csv`, weights | MIT (repo) | GitHub `microsoft/working-with-ai`, v1.1 |
| **OpenAI GDPval** | `data/train-00000-of-00001.parquet` (220 tasks, 44 occupations) | see dataset card | Hugging Face `openai/gdpval`; we use only occupation/sector/prompt |
| **OpenAI "GPTs are GPTs"** (Eloundou et al., 2023) | `data/occ_level.csv` | MIT (repo) | occupation-level exposure (alpha/beta/gamma, GPT-4 and human raters); task-level labels are not in the public repo |
| **OpenAI "How People Use ChatGPT"** (Chatterji et al., Sept 2025) | PDF; GWA shares hand-transcribed into `data/reference/openai_chatgpt_gwa_shares_2025.csv` | OpenAI | shares of 1.1M messages by O*NET generalized work activity (all and work-only); no task-level file published. OpenAI's 2026 "Work at the Frontier" posts likewise publish no data files |
| **Google AI & Economy ATLAS v1.0** (July 2026, arXiv 2608.00038) | PDF; lists and headline statistics hand-transcribed into `data/reference/google_atlas_v1_*` | Google | 14.65M Gemini interactions mapped to O*NET; only occupation lists and summary statistics are published |
| **Microsoft Global AI Diffusion, Q2 2026** | PDF; appendix table parsed by `fetch.parse_microsoft_diffusion_pdf` into `ai_diffusion_2026_q2_countries.csv` | Microsoft AI Economy Institute | share of 15–64 year-olds using generative AI, 147 economies (Singapore 64.3%, US 33.0%) |
| **US Census BTOS** (Business Trends and Outlook Survey) | `BTOS_National.xlsx` as shipped in AEI release 2025-09-15 | public domain (US gov) | biweekly share of firms using AI in production, Sept 2023 – Aug 2025; anchors the adoption curve |
| **MOM Occupational Wage Survey, June 2024** | data.gov.sg datastore `d_9917e751…` (by occupation), `d_523d376f…` (by industry), `d_7d528712…` (by establishment size) | Singapore Open Data Licence | median and quartile gross monthly wages for 562 occupations (private establishments with ≥ 25 employees); matched to SSOC codes by title |
| **MOM Labour Force Survey** | data.gov.sg `d_1d7ab908…` (employed residents by occupation and age), `d_3ba99a6c…` (by occupation, status, sex); SingStat Table Builder `M182081`, `M182171`, `M920131` | Singapore Open Data Licence | resident employment by major occupational group (1-digit) and median income by group. The 2025 report's Section D (`LFR2025_SectionD.xlsx`, finer occupation detail) is behind a bot check: download it by hand from stats.mom.gov.sg into `data/raw/singapore/mom/` |
| **METR time horizons** (Time Horizon 1.1) | `reports/time-horizon-1-1/data/raw/runs.jsonl`, `data/external/release_dates.yaml` | MIT (repo) | GitHub `METR/eval-analysis-public`; we re-fit the per-model logistic ourselves (approximate) |

## Things we looked for and did not use (yet)

* ESCO skills ↔ occupations — esco.ec.europa.eu blocks automated download; would give a second skills taxonomy with ISCO codes.
* MOM job-vacancy and retrenchment tables by occupation (data.gov.sg `d_77a848ab…`, `d_49ab6806…`) — a demand-side view to pair with exposure.
* IMDA business surveys on AI adoption — would replace the US firm-adoption curve for Singapore if published as a time series.
* Remote Labor Index, GDPval win rates by occupation — published as PDFs/blogs; would strengthen the capability calibration.
