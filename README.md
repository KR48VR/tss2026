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
| Physical work | Anthropic "What work can robots do?" | O*NET task | physical flag and robot-exposure tier E0–E3 |
| What the labs benchmark | OpenAI GDPval | occupation | 220 expert tasks across 44 occupations |
| How fast capability moves | METR time horizons (re-fitted here) | model / date | task length an agent completes at 50% / 80% reliability; doubling time |

From that it produces, for every task, activity, skill, occupation, SSOC code and Skills Framework role:

* **frontier scores** — how much observed AI use looks like *doing* the work rather than *helping*;
* **timing** — a central year, with a fast and a slow case, by which the item is "disruptable" (AI can do it at workplace reliability *and* typical organisations have had time to adopt it).

Everything is a planning signal with a range. Nothing here forecasts job loss.

## Quick start

```bash
pip install -e .            # Python 3.11+
python -m onetx fetch       # ~700 MB of public files into data/raw (see docs/DATA_SOURCES.md)
python -m onetx build       # ~5 minutes on a laptop; writes data/processed/*.parquet + onetx.duckdb + docs/REPORT.md
python -m onetx query "select ssoc_code, ssoc_title, half_year_p50 from forecast_ssoc2024 order by 3 limit 10"
```

Or open `data/processed/onetx.duckdb` in any DuckDB client; every processed table is a view.

## What you get (tables in `data/processed/`)

| Group | Tables |
|---|---|
| O*NET core | `onet_occupations`, `onet_tasks`, `onet_task_dwa`, `onet_activities` (DWA→IWA→GWA, classic ids with 31.0 ids alongside), `onet_descriptor_ratings` (+`_v31` with the new Essential/Transferable skills), `onet_descriptor_to_gwa`, `onet_technology_skills`, `onet_job_zones`, `onet_alternate_titles`, `onet_related_occupations`, `onet_emerging_tasks_v31` |
| Singapore | `ssoc2024_*` / `ssoc2020_*` (structure, definitions, unit-group task bullets, alphabetical index), `sfw_roles`, `sfw_key_tasks`, `sfw_role_tscs`, `sfw_tsc_key`, `sfw_tsc_knowledge_abilities`, `sfw_unique_skills`, `sfw_tsc_to_unique_skill` |
| Crosswalks | `xw_onetsoc_soc2018`, `xw_onetsoc_isco08`, `xw_isco08_ssoc2024/2020`, `xw_onetsoc_ssoc2024/2020`, `xw_ssoc2024_ssoc2020`, `xw_sfw_role_ssoc2024`, `xw_sfw_keytask_onet_task`, `xw_sfw_keytask_onet_dwa`, `xw_sfw_skill_onet_iwa`, `xw_sfw_skill_onet_descriptor`, `xw_ssoc2024_task_onet_task` |
| AI signals | `ai_aei_task_metrics` (one row per vintage × source × task), `ai_aei_task_penetration`, `ai_aei_job_exposure`, `ai_robot_exposure_*`, `ai_microsoft_iwa`, `ai_microsoft_soc`, `ai_gdpval_tasks`, `ai_metr_horizons` |
| Frontier | `frontier_tasks`, `frontier_activities`, `frontier_occupations`, `frontier_skills` (+`_v31`), `frontier_ssoc2024`, `frontier_sfw_roles`, `frontier_sfw_key_tasks` |
| Timing | `forecast_tasks`, `forecast_activities`, `forecast_occupations`, `forecast_ssoc2024`, `forecast_sfw_roles`, `forecast_skills` |

`docs/REPORT.md` is regenerated on every build with the headline tables; `data/processed/build_stats.json` records the parameters and checks.

## Read next

* `docs/DESIGN.md` — how the graph is built and why each join is done the way it is
* `docs/DATA_SOURCES.md` — every file, where it comes from, licence, and what to do if a download is blocked
* `docs/METHOD.md` — the frontier scores and the timing model, parameter by parameter, with the caveats
* `docs/REPORT.md` — what the current build says

## Status

First working version. Known gaps are listed at the end of `docs/METHOD.md`. The biggest ones: usage data come from one provider; Singapore links are official where SingStat coded them and text-matched elsewhere (scores are kept so you can filter); the adoption lag is an assumption you should set to your own view.
