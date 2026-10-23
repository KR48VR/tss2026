"""onetx: a crosswalked O*NET joined to Singapore's occupation and skills frameworks and to
frontier-AI usage and capability signals.

Modules
-------
config      paths and constants
fetch       download every public source into data/raw (with a manifest)
onet        load O*NET text releases into tidy tables
singapore   load SingStat SSOC tables and the SkillsFuture Skills Framework dataset
crosswalks  occupation crosswalks: O*NET-SOC -> SOC -> ISCO-08 -> SSOC -> Skills Framework roles
embed       text embeddings and hybrid (semantic + lexical) matching helpers
match       text-based crosswalks: Skills Framework key tasks and TSCs to O*NET tasks, activities, skills
ai_signals  frontier-AI usage and capability signals (Anthropic Economic Index, Microsoft, OpenAI, METR)
frontier    per-task "AI frontier" scores and their propagation to activities, skills, occupations
forecast    disruption-timing model with explicit assumptions and ranges
build       end-to-end pipeline that writes data/processed and a DuckDB file
"""
__version__ = "0.1.0"
