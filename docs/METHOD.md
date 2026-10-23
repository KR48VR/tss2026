# Method: frontier scores and the timing model

## 1. Frontier scores (what is AI doing today?)

Per O*NET task (`frontier_tasks`):

| Column | Meaning |
|---|---|
| `usage_latest` | share (%) of Claude.ai conversations mapped to the task, mean of the task's two most recent releases with published usage (`usage_last_vintage` says which). The 2026 monthly releases list about 1,100 tasks against 3,600 in the first release, so taking only the newest releases would turn thousands of tasks that were seen in use into "not observed". 0 means never published (below Anthropic's privacy threshold or unused) |
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
| `status` | `physical_robotics`, `observed_automation` (seen in any release and automation_share ≥ 0.5), `observed_augmentation` (seen), `not_observed` (never published) |

Roll-ups are weighted means: tasks → activities and occupations use `task_weight` (importance share within the occupation); occupations → SSOC use the crosswalk's `weight_rev`; SSOC → Skills Framework roles use the role's SSOC weights; key tasks use their top-3 matched O*NET tasks weighted by match score. Skills get an importance-weighted mean over the occupations that need them (importance ≥ 3) and, separately, the mean over the GWAs O*NET links them to.

Per occupation (`frontier_occupations`) the other providers are joined: Microsoft's `ai_applicability_score`, OpenAI's 2023 `openai_exposure_beta_gpt4` / `_human` (Eloundou et al.), Google ATLAS's list membership (`google_atlas_flag`), Anthropic's `aei_job_exposure`, GDPval coverage and the robot index. `n_providers_positive` counts how many of Anthropic (≥ 25% of task weight observed), Microsoft (top third of applicability), OpenAI (beta ≥ 0.5) and Google (listed as over-represented or saturated) point the same way.

Sanity check: our task-weighted `observed_automation` per occupation correlates 0.67 with Anthropic's own occupation-level `observed_exposure` and 0.49 with Microsoft's AI applicability score (different data, different method), which is the kind of agreement one would hope for and no more.

## 2. Timing model (when could it be disrupted?)

Three components per task, each anchored on published data, each with a fast and a slow case.

### 2a. Capability — when could AI do it?

*Non-physical tasks.* AI is already used to do tasks of a few hours. We calibrate today's **delegation horizon** on the usage data: fit `automation_share ~ sigmoid(a + b·log2 minutes)` on observed non-physical tasks and take the length at which the fitted share crosses 50% (about 4.3 hours in the current build; `delegation_horizon_now_min` in `build_stats.json`). The horizon grows at METR's measured frontier doubling time (our re-fit of Time Horizon 1.1: about 106 days since 2024; fast 85 days; slow 212 days). A task is capable when the horizon reaches its length. Lengths come from Anthropic's per-task estimate where published (5,180 tasks), else the DWA / IWA / GWA median, else the Job Zone median (`minutes_source`). A task already observed with AI mostly doing it is capable now.

*Evidence tiers and the evidence lag.* Task length alone says "capable now" for almost every desk task, because most tasks are shorter than the delegation horizon. So a desk task is only treated as capable today when there is evidence for it. Each task gets an evidence tier (`frontier_task_evidence`, `forecast.evidence_tiers`): **seen** (AI use on the task has been published in at least one Economic Index release), **rated** (not seen, but rated doable, i.e. E1 or E2, by the 2023 raters in OpenAI's "GPTs are GPTs" study, human or GPT-4, and by Claude's October 2026 rating), **newly_rated** (the 2023 raters both said no, the 2026 rating says yes), **mixed** (the raters disagree) and **unproven** (every rater said no); physical tasks follow the robot path. A task that is not seen waits a lag before it counts as capable. The lag is measured, not assumed (`forecast.calibrate_evidence`): for desk tasks, the share of each rating group ever seen in use is tracked release by release (seven releases, December 2024 to May 2026); the rated group's coverage grows at about 8 percentage points a year, and a group whose coverage today equals what the rated group had k years ago lags by k years. Current central lags: mixed 1.0 years, unproven 1.9 years, newly rated 3.1 years (fast = half, slow = double). The newly-rated lag is the longest: a fresh "yes" from one model against both 2023 raters has so far been the weakest predictor of actual use. "Unproven" never means "AI cannot".

*The 2026 rating.* `python -m onetx rate` has Claude (Sonnet, through the Claude Code command line, batches of 80 tasks, about 25 minutes and US$15) rate every O*NET task on `onetx/rubric_2026.md`: exposure on the 2023 E0/E1/E2 scale, end-to-end autonomy A0–A3 and the main barrier. Results are cached in `data/reference/ai_task_ratings_<version>.csv` and used once at least 90% of tasks are rated. Validation (October 2026 run): the 2026 yes/no agrees with the 2023 human raters on 80% of tasks (kappa 0.60) and with GPT-4 on 84% (kappa 0.69), while the two 2023 raters agreed with each other on 82% (kappa 0.64). 90% of robot-flagged tasks are rated with a physical barrier and 4.5% of non-physical tasks are. Among seen desk tasks the autonomy rating tracks the observed automation share (A3 tasks 59%, A0 tasks 41%; correlation 0.22), which the exposure rating does not. Against the 2023 raters, 26% of desk tasks both called "no" are now rated doable and 17% of tasks at least one called "yes" are now rated not. A repeat rating of 300 random tasks in fresh batches agreed with the first on 84% of exposure labels (kappa 0.73; 89% on yes/no), 78% of autonomy levels (99.7% within one level) and 77% of barrier reasons (kappa 0.71). Re-run with a new rubric version to measure how the frontier moves.

*Physical tasks.* The robot-exposure tier sets years-from-now to capability (fast / central / slow): E1 purpose-built cell 2 / 4 / 8; E2 structured human facility 4 / 7 / 12; E3 unstructured setting 8 / 12 / 20; E0 beyond the horizon.

### 2b. Adoption — when do organisations let it?

*US curve, refreshed every build.* The US Census Business Trends and Outlook Survey asks firms every two weeks whether they used AI in producing goods or services. `python -m onetx fetch` downloads the national and sector files from census.gov whenever the local copy is two weeks old, so the curve never again runs on a stale copy. Two wordings are spliced: the original "Artificial Intelligence (AI)" question (51 readings, Sept 2023 to Aug 2025, 3.7% to 9.7%, from the copy in the AEI release) and the reworded "Super Intelligence or SI (formerly referred to as Artificial Intelligence or AI)" question (Nov 2025 onward; 24 readings to 2026-10-04, 24.6%). The level jumped at the rewording (9.7% in Aug 2025 to 17.3% in Nov 2025, more than the whole prior two years), so the fit `logit(p/K) = a + b·t + c·[new wording]` carries a step: growth 0.506 a year on all 75 readings, step 0.57. The curve used runs at the new-wording level and crosses half of US firms in **2029.2** (central), 2027.5 with growth 1.5x faster, 2032.7 at 0.6x; the old-wording level would say 2030.3. The ceiling K is 90% by assumption. A stale copy fitted to Aug 2025 had predicted 15% of firms for Oct 2026; the Census reported 24.6%, which is why the feed is live.

*Singapore lead, measured from firms.* IMDA's annual survey of enterprises (`data/reference/singapore_ai_adoption_imda.csv`, hand-transcribed from the Singapore Digital Economy Reports with a source per row) gives the share of Singapore firms that used AI: 2024: 14.7%, 2025: 23.5% (all enterprises; SMEs 4.2% in 2023, 14.5% in 2024, 23.4% in 2025; large local enterprises 44.0%, 62.5%, 70.4%). For each point the date at which the fitted US curve reaches the same share is read off; Singapore's annual figure is dated at year end, the conservative choice because a firm that used AI at any time in the year counts while the Census asks about the last two weeks. The mean gap over [2024, 2025] is a **lead of 0.64 years**, so half of Singapore firms by about 2028.6 in the central case. Singapore's own three points fitted alone put the half-way point at 2027.2, earlier still, but that fit leans on a derived 2023 figure. The earlier method (individual usage ratio converted through the growth rate, about 1.1 years) remains as the fallback when no firm series is available (`geo_lead_years`).

*Sector shifts.* The Census publishes the same question by NAICS sector. Holding the national growth rate, each sector's level offset converts into years ahead of or behind the national curve (`ai_signals.fit_sector_levels`), and `sector_adoption.py` carries those shifts to Singapore: NAICS sectors map to the industry groups MOM uses in Labour Force Table D2; an SSOC occupation takes a blend of its major group's industry mix (Table D2, employed residents by occupation group and industry) and the industries of the Skills Framework roles tagged to it (half each where tags exist); an O*NET occupation takes the crosswalk-weighted mean over its SSOC codes; a Skills Framework role takes its sector's industry (cross-sector frameworks such as Human Resource stay national). Current shifts run from -2.1 years (information and communications) to +2.0 years (accommodation and food services); SSOC codes range -1.4 to +1.7. As a check, the US-derived ordering of industries agrees with IMDA's 2023 sector adoption rates (Spearman -0.93, where -1 is perfect agreement). Shifts are divided by the growth multiplier in the fast and slow cases. Roll-ups re-anchor task years to the entity's own shift (`forecast.reshift_years`), so the same O*NET task can carry a different year in a bank's role and a restaurant chain's role.

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
disrupt_year = adoption anchor − Singapore lead + industry shift / growth multiplier + max(0, capable year − today) + barrier delay − 1 year if already observed being automated

where, for a desk task that is not seen, capable year is at least today + the evidence lag of its tier
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
4. **Adoption is still shaped by the US curve.** Its speed comes from the US series; Singapore's level comes from IMDA's annual points (three so far, one of them derived), and the industry shifts are US sector levels carried over. IMDA's sector rates agree on the ordering, not yet on the size. A longer Singapore series would let the speed be fitted here too.
5. **Employment detail.** Official resident employment stops at the 2-digit SSOC group (43 groups). Below that, the even split across surveyed occupations is the weakest link in the wage-value view; the Occupational Employment Dataset that would fix it is not published.
6. **Text matches are approximate.** Filter on `score` / `in_mapped_occ`; hand-check the roles that matter to you.
7. **Capability evidence is still indirect.** "Seen in use" reflects who uses Claude; the 2023 and 2026 ratings are judgements, not demonstrations; the evidence lags are measured on usage coverage, which grows partly because the sample grows. Real-world agent benchmarks (Remote Labor Index, GDPval win rates) would be a better anchor when task-level files are published. The repeat-rating check in the QA log gives the 2026 rating's own reliability.
8. **One model rating its own kind.** The 2026 rating is Claude's view of what AI can do. Its agreement with the 2023 human raters is the guard, and the measured lag for "newly rated" tasks shows the data are allowed to overrule it.
