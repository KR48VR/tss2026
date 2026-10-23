# Method: frontier scores and the timing model

## 1. Frontier scores (what is AI doing today?)

Per O*NET task (`frontier_tasks`):

| Column | Meaning |
|---|---|
| `usage_latest` | share (%) of Claude.ai conversations mapped to the task, average of April and May 2026. 0 means not published (below Anthropic's privacy threshold or unused) |
| `usage_index` | percentile rank of `usage_latest` among tasks with any usage; 0 for the rest |
| `usage_trend` | yearly change in log usage share across the vintages (needs ≥ 3 points). Positive = the task is gaining share of AI use |
| `automation_share` | among classified conversations, the share with an automation-style pattern (directive or feedback loop) as opposed to augmentation (validation, task iteration, learning). Imputed from the task's DWA, then IWA, then the global mean when unpublished (`automation_imputed`) |
| `ai_autonomy` | 1–5 mean autonomy rating (Jan 2026 release onwards) |
| `human_only_hours` | Anthropic's estimate of the time a person would need without AI |
| `human_only_ability_pct` | share of conversations where a human could have done the task alone |
| `penetration` | Anthropic labour-market "task penetration" |
| `ms_share`, `ms_completion`, `ms_scope` | Microsoft Copilot activity share, completion rate and impact scope, averaged over the task's IWAs |
| `physical`, `robot_tier` | from the robot-exposure release (E0–E3) |
| `observed_automation` | `1{usage_latest > 0} × automation_share` — Project Iceberg's "observed automation" signal |
| `status` | `physical_robotics`, `observed_automation` (usage and automation_share ≥ 0.5), `observed_augmentation`, `not_observed` |

Roll-ups are weighted means: tasks → activities and occupations use `task_weight` (importance share within the occupation); occupations → SSOC use the crosswalk's `weight_rev`; SSOC → Skills Framework roles use the role's SSOC weights; key tasks use their top-3 matched O*NET tasks weighted by match score. Skills get an importance-weighted mean over the occupations that need them (importance ≥ 3) and, separately, the mean over the GWAs O*NET links them to.

Sanity check: our task-weighted `observed_automation` per occupation correlates 0.67 with Anthropic's own occupation-level `observed_exposure` and 0.49 with Microsoft's AI applicability score (different data, different method), which is the kind of agreement one would hope for and no more.

## 2. Timing model (when could it be disrupted?)

Two components and an explicit lag, computed per task and rolled up as "share of task weight disruptable by year".

### 2a. Capability

*Non-physical tasks.* AI is already used to do tasks of a few hours. We calibrate today's **delegation horizon** on the usage data: fit `automation_share ~ sigmoid(a + b·log2 minutes)` on observed non-physical tasks and take the length at which the fitted share crosses 50%. In the current build that is about 4.3 hours (`delegation_horizon_now_min` in `build_stats.json`). The horizon then grows at METR's measured frontier doubling time (our re-fit of Time Horizon 1.1 runs: about 106 days since 2024; fast case 85 days; slow case 212 days, METR's long-run 2019–2025 figure). A task becomes "capable" when the horizon reaches its length. Lengths come from Anthropic's per-task estimate where published (5,180 tasks), else the DWA / IWA / GWA median, else the Job Zone median (`minutes_source`).

A task already observed with AI mostly doing it (`status = observed_automation`) is capable now regardless.

*Physical tasks.* The robot-exposure tier sets years-from-now to capability (fast / central / slow): E1 purpose-built cell 2 / 4 / 8; E2 structured human facility 4 / 7 / 12; E3 unstructured setting 8 / 12 / 20; E0 beyond the horizon.

### 2b. Adoption lag

From "capable" to "typical organisations have adopted it": central 3 years, range 1.5–6 (firm-level AI adoption has been climbing from single digits since 2023; this is an assumption, set it to your own view). Adjustments:

* minus 1 year where the task is already observed being automated;
* plus 1 year where `human_only_ability_pct ≥ 95` (people rarely hand it over even when AI is involved);
* plus up to 2 years, scaled by the occupation's in-person index (O*NET work context: physical proximity, face-to-face discussion, contact with others), for tasks not yet observed in usage data.

### 2c. Outputs

Per task: `capable_year_{fast,central,slow}`, `disrupt_year_{p10,p50,p90}`, `timing_bucket`, `beyond_horizon`. Per occupation / SSOC / role / activity / skill (`forecast_*`): `share_by_2026 … share_by_2040` (share of task weight whose central year ≤ Y), `share_beyond_horizon`, `share_already_observed`, and `half_year_{p10,p50,p90}` (first year at which half the weight is disruptable under each case). Skills Framework roles are timed through their key tasks (top-1 matched O*NET task each).

### 2d. How to read it

* A half-way year of 2029 means: under the central assumptions, by 2029 AI could be doing half of this job's task weight and organisations will have had time to adopt it. The p10/p90 columns give the fast and slow cases; the honest statement is the range.
* Jobs re-bundle. A role whose tasks are disruptable does not disappear; it changes. Pair these numbers with demand (vacancies, wages) before drawing conclusions.
* The model has no notion of regulation, liability, unions, or price. Those shift adoption lags; change the parameters.

## 3. Known gaps and next steps

1. **One usage provider.** Claude users skew to coding and knowledge work. Microsoft's IWA data are a second opinion; OpenAI publishes only aggregates. The not-observed category mixes "AI cannot" with "Claude users do not".
2. **Relative, not absolute, usage.** Usage shares sum to 100% within a vintage; a falling share can still be rising use. Treat `usage_trend` as where the frontier is moving, not as adoption.
3. **Length is a weak discriminator.** Most O*NET tasks take 1–6 hours and the delegation horizon is already in that range; timing differences come mostly from physical work, observation status and the in-person/hand-over modifiers. Add task-level barriers (judgement, accountability, tacit context) when a source for them appears.
4. **Singapore wages and employment.** Add MOM's occupational wage and vacancy tables by SSOC to compute wage-value exposure (as in Project Iceberg) and to weight sectors.
5. **Text matches are approximate.** Filter on `score` / `in_mapped_occ`; hand-check the roles that matter to you. The evaluation in `build_stats.json` tells you the matcher's hit rate on roles SingStat coded.
6. **Capability calibration.** The 50% automation-share point is a proxy for "AI can do it". Real-world agent benchmarks (Remote Labor Index, GDPval win rates) would give a better anchor; wire them in when task-level files are published.
