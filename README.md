# onetx — a crosswalked O*NET for Singapore, joined to the AI frontier

**What this is.** A data pipeline that builds one connected map of work:

```
O*NET tasks  ──►  detailed / intermediate / generalized work activities  ──►  skills, knowledge, abilities
     │                                                                          ▲
     ▼                                                                          │
O*NET occupations ──► SOC 2018 ──► ISCO-08 ──► SSOC 2024 (Singapore) ──► Skills Framework job roles
                                                                            ├── critical work functions / key tasks  ──► O*NET tasks (text match)
                                                                            └── TSCs / CCS / unique skills            ──► O*NET activities + skills (text match)
```

…and then lays frontier-AI evidence on top of it:

| Signal | Source | Level | What it tells you |
|---|---|---|---|
| Where AI is being used, and whether it does or helps | Anthropic Economic Index, 7 monthly points Dec 2024 – May 2026, Claude.ai and API | O*NET task | observed usage share, automation vs augmentation, AI autonomy, time a human would need |
| Second opinion on usage | Microsoft "Working with AI" (Bing Copilot) | O*NET intermediate work activity | activity share, completion rate, scope of impact; AI applicability per occupation |
| Labour-market exposure | Anthropic "Labor market impacts" | task and SOC | task penetration, observed job exposure |
| Other providers' views | OpenAI "GPTs are GPTs" (2023) and "How People Use ChatGPT" (2025); Google AI & Economy ATLAS (2026) | occupation / GWA | exposure ratings, message shares by work activity, over- and under-represented occupations; combined into a provider-consensus count |
| How fast adoption moves | US Census BTOS firm survey; Microsoft Global AI Diffusion (147 economies) | firms / people | the adoption curve the timing model rides, and Singapore's lead on it |
| Singapore people and pay | MOM Occupational Wage Survey June 2025 (with SSOC codes); MOM Labour Force in Singapore 2025, Table D8 | SSOC occupation / 2-digit group | employment- and wage-weighted exposure for the resident workforce |
| Physical work | Anthropic "What work can robots do?" | O*NET task | physical flag and robot-exposure tier E0–E3 |
| What the labs benchmark | OpenAI GDPval | occupation | 220 expert tasks across 44 occupations |
| How fast capability moves | METR time horizons (re-fitted here) | model / date | task length an agent completes at 50% / 80% reliability; doubling time |

From that it produces, for every task, activity, skill, occupation, SSOC code and Skills Framework role:

* **frontier scores** — how much observed AI use looks like *doing* the work rather than *helping*;
* **timing** — a central year, with a fast and a slow case, by which the item is "disruptable" (AI can do it at workplace reliability *and* typical organisations have had time to adopt it).

Everything is a planning signal with a range. Nothing here forecasts job loss.

## Two ways in

**No code.** The web page built from this pipeline lets anyone search a Singapore job title or Skills Framework role and see which of its tasks AI already does, which stay human, and the range of years by which half the job could be disruptable, with sector, skill and whole-economy views. Published copies: [the map](https://claude.ai/artifact/FC2mCKb7exU2u9jmVbFNPS) and its [companion page](https://claude.ai/artifact/Pxd31ubFiwTQjdTSBsu7qZ) (how it was built, the QA log, and how King and Claude worked together; source in `site/companion.html`). Rebuild the map with `python -m onetx site`, which writes `site/index.html` plus the files it loads on demand: `kt.json` (key tasks per role), `search.json` (alternative titles and the O*NET task index) and the meaning-search model and vectors (`model.*`, `emb.*`, from `onetx/site_search.py`). The page's "Ask Claude" button uses the artifact `sample` capability and only appears inside a Claude viewer. A scenario switch (fast, central, slow) at the top changes every year and curve on the page; each job shows what its years rest on (seen in AI use, rated doable, newly doable, raters disagree, unproven, physical) and an "evidence as of" line gives the date of every input. A "Reading the news" view places any report about AI and jobs on a five-rung ladder (rated, seen in use, AI mostly doing it, adopted, job changed), translates a report's headline number down the ladder with conversion rates measured on this map's own tasks, and scores every source behind the map on its track record (`onetx/validation.py`).

**Code.** Run the pipeline to rebuild every table, change the assumptions, or query the DuckDB file.

## Quick start

```bash
pip install -e .            # Python 3.11+
python -m onetx fetch       # ~700 MB of public files into data/raw (see docs/DATA_SOURCES.md)
python -m onetx build       # ~5 minutes on a laptop; writes data/processed/*.parquet + onetx.duckdb + docs/REPORT.md
python -m onetx query "select ssoc_code, ssoc_title, half_year_p50 from forecast_ssoc2024 order by 3 limit 10"
python -m onetx rate        # (optional, ~25 min, ~US$15 of Claude use) re-rate every task on onetx/rubric_2026.md; cached in data/reference
python -m onetx forecast    # re-run only the timing model (fast); use after `rate` or to try parameters
python -m onetx site        # the browsable page, from the processed tables
```

Or open `data/processed/onetx.duckdb` in any DuckDB client; every processed table is a view.

## What you get (tables in `data/processed/`)

| Group | Tables |
|---|---|
| O*NET core | `onet_occupations`, `onet_tasks`, `onet_task_dwa`, `onet_activities` (DWA→IWA→GWA, classic ids with 31.0 ids alongside), `onet_descriptor_ratings` (+`_v31` with the new Essential/Transferable skills), `onet_descriptor_to_gwa`, `onet_technology_skills`, `onet_job_zones`, `onet_alternate_titles`, `onet_related_occupations`, `onet_emerging_tasks_v31` |
| Singapore | `ssoc2024_*` / `ssoc2020_*` (structure, definitions, unit-group task bullets, alphabetical index), `sfw_roles`, `sfw_key_tasks`, `sfw_role_tscs`, `sfw_tsc_key`, `sfw_tsc_knowledge_abilities`, `sfw_unique_skills`, `sfw_tsc_to_unique_skill`, `sg_ows_wages_ssoc2024`, `sg_lfs_employment_major`, `sg_ssoc2024_weights` |
| Crosswalks | `xw_onetsoc_soc2018`, `xw_onetsoc_isco08`, `xw_isco08_ssoc2024/2020`, `xw_onetsoc_ssoc2024/2020`, `xw_ssoc2024_ssoc2020`, `xw_sfw_role_ssoc2024`, `xw_sfw_keytask_onet_task`, `xw_sfw_keytask_onet_dwa`, `xw_sfw_skill_onet_iwa`, `xw_sfw_skill_onet_descriptor`, `xw_ssoc2024_task_onet_task` |
| AI signals | `ai_aei_task_metrics` (one row per vintage × source × task), `ai_aei_task_penetration`, `ai_aei_job_exposure`, `ai_robot_exposure_*`, `ai_microsoft_iwa`, `ai_microsoft_soc`, `ai_gdpval_tasks`, `ai_metr_horizons`, `ai_openai_gpts_exposure`, `ai_openai_chatgpt_gwa_shares`, `ai_google_atlas_occupations`, `ai_microsoft_diffusion_countries`, `ai_btos_firm_adoption` |
| Frontier | `frontier_tasks`, `frontier_task_barriers`, `frontier_activities`, `frontier_occupations` (with provider consensus), `frontier_skills` (+`_v31`), `frontier_ssoc2024`, `frontier_sfw_roles`, `frontier_sfw_key_tasks` |
| Timing | `forecast_tasks`, `forecast_activities`, `forecast_occupations`, `forecast_ssoc2024`, `forecast_sfw_roles`, `forecast_skills` |
| Singapore economy | `sg_ssoc2024_weighted`, `sg_economy_exposure` (employment- and wage-weighted, by major group), `sg_sfw_sectors` |

`docs/REPORT.md` is regenerated on every build with the headline tables; `data/processed/build_stats.json` records the parameters and checks.

## Read next

* `docs/DESIGN.md` — how the graph is built and why each join is done the way it is
* `docs/DATA_SOURCES.md` — every file, where it comes from, licence, and what to do if a download is blocked
* `docs/METHOD.md` — the frontier scores and the timing model, parameter by parameter, with the caveats
* `docs/REPORT.md` — what the current build says

## Status

Second version. Compared with the first: four providers instead of one at occupation level (Anthropic, Microsoft, OpenAI, Google) with a consensus count; a per-task barrier index (interpersonal, in-person, accountability, hand-over) in place of a flat lag; an adoption anchor fitted to the US Census firm-adoption series with Singapore's lead applied; and MOM wages (June 2025, coded to SSOC) and employment (2-digit groups) so Singapore results can be read as shares of people and pay. Known gaps are listed at the end of `docs/METHOD.md`.
