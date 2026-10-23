# Design: the connected map of work

## The idea in one paragraph

O*NET already links occupations to tasks, tasks to work activities, and occupations to skills. What it does not do is (a) speak Singapore's language (SSOC codes, Skills Framework roles, Critical Work Functions, Key Tasks, TSCs) and (b) say anything about AI. This project adds both. Official code tables carry the occupation links (O*NET-SOC → SOC → ISCO-08 → SSOC → Skills Framework roles). Where there is no official table (Singapore key tasks and skills versus O*NET tasks and activities) we match on meaning and keep the score. Then every AI signal that is published at task, activity or occupation level is attached to the node it names, and pushed along the links as weighted averages.

## Nodes

| Node | Count | Source |
|---|---|---|
| O*NET occupation (O*NET-SOC 2019) | 1,016 | O*NET 30.1 (Dec 2025); 31.0 (Aug 2026) loaded alongside |
| O*NET task | 18,796 | O*NET 30.1 Task Statements + Task Ratings |
| Detailed / intermediate / generalized work activity | 2,087 / 332 / 37 | O*NET 30.1 DWA/IWA references; 31.0 renumbered ids are mapped back by name |
| Skills, knowledge, abilities, work activities (descriptors) | 35 / 33 / 52 / 41 (30.1); Essential 10 + Transferable 25 skills (31.0) | O*NET descriptor files |
| SOC 2018 | 867 | O*NET crosswalk |
| ISCO-08 unit group | 436 | ILO (via BLS and ESCO tables) |
| SSOC 2024 occupation (5-digit) | 1,006 | SingStat |
| SSOC unit-group task bullet | 2,819 | SingStat detailed definitions (4-digit level) |
| Skills Framework job role | 2,030 across 39 sectors | SkillsFuture Jobs-Skills Portal dataset (Q3 2026) |
| Skills Framework key task | 40,367 | same |
| TSC / CCS (sector codes) | 12,007 | same |
| Unique skill (SFw 2.0 language) | 2,388 | SkillsFuture Unique Skills List |

## Edges

### Occupation chain (official tables)

1. **O*NET-SOC 2019 → SOC 2018**: O*NET's own file. One-to-one up to the `.xx` specialisations.
2. **O*NET-SOC 2019 → ISCO-08**: two independent routes, combined.
   * *ESCO route*: O*NET publishes an ESCO↔O*NET-SOC crosswalk. ESCO occupation codes start with their ISCO-08 unit group (`2512.1` → `2512`).
   * *SOC-2010 route*: O*NET's 2010→2019 walk, SOC 2010 = first seven characters, then the BLS SOC 2010 → ISCO-08 table.
   * Each route gives equal weights across its targets; the two are averaged and renormalised. `n_routes` = 2 marks the links both tables agree on (about 20% of links, carrying most of the weight).
3. **ISCO-08 → SSOC 2024 / 2020**: SingStat's correspondence tables (both directions, union), `part` flags partial correspondences.
4. **O*NET-SOC → SSOC**: composition. `weight` sums to 1 per O*NET occupation (push US-anchored scores to Singapore); `weight_rev` sums to 1 per SSOC code (pull them into each Singapore occupation).
5. **Skills Framework role → SSOC**: SingStat tags Skills Framework roles in its SSOC alphabetical index as "Role title (SFw-Sector)". We parse those and join to the dataset's roles by sector and title (exact, exact-with-track, or fuzzy ≥ 92). That codes 1,556 of 2,030 roles officially. The remaining 474 are matched by description to SSOC definitions with a hybrid text matcher; on the officially-coded roles that matcher recovers the exact 5-digit code 37% of the time and has the right code in its top five 64% of the time, so filter on `match_score` for those.

### Meaning-based edges (text matching)

The matcher blends a static sentence embedding (model2vec `potion-base-32M`, CPU-fast) with TF-IDF cosine, 60/40, and adds a small bonus when the candidate belongs to an occupation the query's role or SSOC code already maps to. Results keep the raw score and an `in_mapped_occ` flag.

* **Key task → O*NET task** (top 3) and **→ DWA** (top 1).
* **Unique skill → O*NET IWA** (top 3) and **→ O*NET descriptor** (skills, knowledge, abilities, 30.1 and 31.0; top 3).
* **SSOC unit-group task bullet → O*NET task** (top 3).

### AI-signal edges

Signals attach at the level they are published and are pushed along the graph as weighted means:

* task-level (Anthropic usage, penetration, robots) → DWA/IWA/GWA (task weight) → occupation (task importance share) → SOC/ISCO/SSOC (crosswalk weights) → Skills Framework roles (via SSOC) and directly via key-task matches;
* IWA-level (Microsoft) → tasks that map to the IWA → everything above;
* occupation-level (Microsoft applicability, Anthropic job exposure, OpenAI 2023 exposure, Google ATLAS lists, GDPval coverage, robot index) → SSOC → roles, with a provider-consensus count;
* Singapore weights (MOM wages and employment by SSOC) turn SSOC-level scores into employment- and wage-weighted economy views;
* skills get two views: through the occupations that need them (importance-weighted), and through the GWAs O*NET links them to.

## Why two O*NET versions

O*NET 31.0 (August 2026) renumbered DWA/IWA ids and replaced the 35 "Skills" with 10 Essential and 25 Transferable skills. Every AI dataset still uses the classic ids. So 30.1 is the spine and 31.0 is loaded alongside with the id mapping done by name (all 2,087 DWA names match). When the AI datasets move to 31.0, flip `ONET_BASE_VERSION` in `onetx/config.py`.

## Code map

```
onetx/config.py      paths, versions, AEI vintages
onetx/fetch.py       download + manifest
onetx/onet.py        O*NET loaders
onetx/singapore.py   SSOC + Skills Framework loaders
onetx/singapore_labour.py   MOM wages and employment matched to SSOC codes
onetx/singapore_economy.py  employment- and wage-weighted economy views
onetx/crosswalks.py  official occupation crosswalks
onetx/embed.py       embeddings + hybrid matcher
onetx/match.py       meaning-based crosswalks + evaluation
onetx/ai_signals.py  AEI / Microsoft / OpenAI / METR loaders
onetx/frontier.py    per-task frontier scores and roll-ups
onetx/forecast.py    timing model
onetx/build.py       the pipeline; onetx/report.py the summary
```
