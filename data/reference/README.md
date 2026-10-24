# Hand-transcribed reference tables

Small tables typed in from reports that publish numbers only inside PDFs. Each file names its source.

* `openai_chatgpt_gwa_shares_2025.csv` — Chatterji, Cunningham, Deming et al., "How People Use ChatGPT" (OpenAI / NBER WP 34255, Sept 2025), Figures 14 and 15: share of 1.1M consumer messages (May 2024 – June 2025) by O*NET generalized work activity, for all messages and for the ~366k work-classified messages. GWAs with fewer than 100 users are suppressed in the source.
* `google_atlas_v1_occupations.csv` and `google_atlas_v1_headline.json` — Google's AI & Economy ATLAS v1.0 (July 2026), Tables 1 and 2 and the headline coverage statistics. Google publishes no task- or occupation-level numbers, only these lists.

## Added in round three

* `singapore_ai_adoption_imda.csv`: AI adoption rate of Singapore enterprises by year (all, SMEs, non-SMEs, sectors), hand-transcribed from IMDA's Singapore Digital Economy Reports 2024, 2025 and 2026 with a source per row. The 2023 all-enterprise figure is derived (SMEs are about 99% of enterprises) and marked as such.
* `ai_task_ratings_2026.10a.csv`: Claude's October 2026 rating of every O*NET task on `onetx/rubric_2026.md` (exposure E0/E1/E2, autonomy A0-A3, main barrier), produced by `python -m onetx rate` and committed so a build is reproducible without re-rating. Validation is in `docs/METHOD.md`.

- `onet_career_clusters_2026.csv`: O*NET OnLine "Browse by Career Cluster" (National Career Clusters Framework, 14 clusters), collected from the 14 cluster pages on 24 Oct 2026: 1,264 cluster-to-occupation links covering 997 of the 1,016 O*NET occupations (the 19 without a cluster are military); 186 occupations sit in more than one cluster. Not part of the O*NET database download, so refresh by re-collecting the pages. O*NET content is CC BY 4.0.

- `cmc_models_to_career_clusters.csv`: hand-made match of the 30 DOL industry competency models to the 14 career clusters (Arts, Education and Agriculture have no industry model and show the generic Building Blocks tiers). The models themselves come from the Competency Model Clearinghouse data download (`python -m onetx fetch`, `dol/`), CC BY 4.0, credit the CMC and DOL ETA.
