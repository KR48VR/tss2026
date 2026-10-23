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

Per occupation (`frontier_occupations`) the other providers are joined: Microsoft's `ai_applicability_score`, OpenAI's 2023 `openai_exposure_beta_gpt4` / `_human` (Eloundou et al.), Google ATLAS's list membership (`google_atlas_flag`), Anthropic's `aei_job_exposure`, GDPval coverage and the robot index. `n_providers_positive` counts how many of Anthropic (≥ 25% of task weight observed), Microsoft (top third of applicability), OpenAI (beta ≥ 0.5) and Google (listed as over-represented or saturated) point the same way.

Sanity check: our task-weighted `observed_automation` per occupation correlates 0.67 with Anthropic's own occupation-level `observed_exposure` and 0.49 with Microsoft's AI applicability score (different data, different method), which is the kind of agreement one would hope for and no more.

## 2. Timing model (when could it be disrupted?)

Three components per task, each anchored on published data, each with a fast and a slow case.

### 2a. Capability — when could AI do it?

*Non-physical tasks.* AI is already used to do tasks of a few hours. We calibrate today's **delegation horizon** on the usage data: fit `automation_share ~ sigmoid(a + b·log2 minutes)` on observed non-physical tasks and take the length at which the fitted share crosses 50% (about 4.3 hours in the current build; `delegation_horizon_now_min` in `build_stats.json`). The horizon grows at METR's measured frontier doubling time (our re-fit of Time Horizon 1.1: about 106 days since 2024; fast 85 days; slow 212 days). A task is capable when the horizon reaches its length. Lengths come from Anthropic's per-task estimate where published (5,180 tasks), else the DWA / IWA / GWA median, else the Job Zone median (`minutes_source`). A task already observed with AI mostly doing it is capable now.

*Physical tasks.* The robot-exposure tier sets years-from-now to capability (fast / central / slow): E1 purpose-built cell 2 / 4 / 8; E2 structured human facility 4 / 7 / 12; E3 unstructured setting 8 / 12 / 20; E0 beyond the horizon.

### 2b. Adoption — when do organisations let it?

The US Census Business Trends and Outlook Survey asks firms every two weeks whether they used AI in producing goods or services (51 points, Sept 2023 to Aug 2025, 3.7% to 9.7%). We fit a logistic curve with a 90% ceiling and read off when it crosses 50% of firms: **2030.4** in the central case (growth rate as fitted), 2028.2 if growth is 1.5x faster, 2034.6 if 0.6x slower. That date is the **adoption anchor**: when a typical, barrier-free task that is capable today would be in majority use.

Singapore runs ahead of the US on individual AI use: Microsoft's diffusion measure puts 64% of Singapore's working-age people on generative AI versus 33% in the US (Q2 2026), and Anthropic's per-capita usage index is about 1.5x the US figure. The geometric mean (1.7x) converted onto the same curve is a **lead of about one year**; set `geo="USA"` to switch it off. This is the one place a national figure enters the model.

### 2c. Barriers — what else holds it back?

`frontier.task_barriers` scores every task on four things, each 0–1:

| component | what it measures | source |
|---|---|---|
| interpersonal | share of the task's detailed activities in caring, selling, negotiating, teaching, supervising, public-facing GWAs | O*NET task→DWA→GWA links |
| in_person | occupation percentile of physical proximity, face-to-face discussion, contact with others, dealing with the public | O*NET work context |
| accountability | occupation percentile of consequence of error, impact and frequency of decisions | O*NET work context |
| handover | how often people kept the task even with AI in the loop (human_only_ability), 50–100% → 0–1 | Anthropic usage data |

`barrier_index` is a weighted sum (0.35 / 0.25 / 0.25 / 0.15) and delays adoption by `barrier_index × max_barrier_years` (fast 2, central 4, slow 6 years). Physical tasks are handled by the robot path. `barrier_type` names the dominant component so a reader can see *why* a task is late.

### 2d. Putting it together

```
disrupt_year = adoption anchor − Singapore lead + max(0, capable year − today) + barrier delay − 1 year if already observed being automated
```

Per task: `capable_year_{fast,central,slow}`, `barrier_years_{fast,central,slow}`, `disrupt_year_{p10,p50,p90}`, `timing_bucket`. Per occupation / SSOC / role / activity / skill (`forecast_*`): `share_by_2026 … share_by_2040`, `share_beyond_horizon`, `share_already_observed`, mean `barrier_index`, and `half_year_{p10,p50,p90}` (first year at which half the weight is disruptable). Skills Framework roles are timed through their key tasks (top-1 matched O*NET task each).

### 2e. Singapore, weighted by people and pay

`sg_ssoc2024_weights` gives every SSOC 2024 code a wage and an employment figure. Wages: MOM Occupational Wage Survey June 2025, which carries official SSOC codes for 523 occupations (no title matching needed); codes it lacks take the 2024 survey matched by title, then the 2-digit group median, then the major-group median income (`wage_source` says which). Employment: MOM Labour Force in Singapore 2025, Table D8, resident employment for 43 SSOC sub-major groups (2-digit), split evenly across the surveyed occupations within each group because no finer official count is published. `sg_economy_exposure` then reports observed automation and the share of work disruptable by year for the whole resident workforce, each major group and each sub-major group, weighted by employment and by wage value (employment × wage). Economy and group figures are robust to the even split; single-occupation wage values are not.

### 2f. How to read it

* A half-way year of 2031 means: under the central assumptions, by 2031 AI could be doing half of this job's task weight and organisations will have had time to adopt it. p10/p90 give the fast and slow cases; the honest statement is the range.
* Jobs re-bundle. A role whose tasks are disruptable does not disappear; it changes. Pair these numbers with demand (vacancies, wages) before drawing conclusions.
* Regulation, liability and price are not modelled beyond the accountability barrier. Change the parameters if you know better.

## 3. Known gaps and next steps

1. **Usage data are still provider-skewed.** Anthropic is the only provider publishing task-level usage. Microsoft (activity level), OpenAI (occupation-level exposure from 2023, GWA shares from 2025) and Google (lists of over- and under-represented occupations, 2026) are second opinions, not substitutes; `n_providers_positive` shows where they agree. OpenAI's 2026 "Work at the Frontier" series and Google's ATLAS publish no downloadable task tables yet; wire them in when they do.
2. **Relative, not absolute, usage.** Usage shares sum to 100% within a vintage; a falling share can still be rising use. Treat `usage_trend` as where the frontier is moving, not as adoption.
3. **Barriers are proxies.** The barrier index uses O*NET's occupation-level work context and activity links; it cannot see regulation or liability for a specific task. Where you know better (a licensed act, a statutory sign-off), override `barrier_index` for those tasks.
4. **Adoption is a US firm curve.** Singapore's lead is inferred from individual usage, not from a Singapore firm-adoption series. IMDA's business surveys would be the right replacement when a time series is available.
5. **Employment detail.** Official resident employment stops at the 2-digit SSOC group (43 groups). Below that, the even split across surveyed occupations is the weakest link in the wage-value view; the Occupational Employment Dataset that would fix it is not published.
6. **Text matches are approximate.** Filter on `score` / `in_mapped_occ`; hand-check the roles that matter to you.
7. **Capability calibration.** The 50% automation-share point is a proxy for "AI can do it". Real-world agent benchmarks (Remote Labor Index, GDPval win rates) would be a better anchor when task-level files are published.
