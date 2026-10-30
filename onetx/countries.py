"""Further countries on the same map. Everything a country needs is listed in one place (a CountrySpec),
and the same steps run for each:

  1. a crosswalk from O*NET occupations to the national classification through ISCO-08
     (O*NET -> ISCO from crosswalks.py; ISCO -> ANZSCO from the ABS table, ISCO -> SOC 2020 from the
     ONS coding index), with `weight` summing to 1 per O*NET occupation and `weight_rev` to 1 per
     national code
  2. national weights (employment and pay per code) and their O*NET-level mirror
  3. the country's own firm-adoption series against the fitted US curve: a lead in years
     (ai_signals.singapore_firm_lead, generalised) and each industry's level against the national one,
     carried to occupations through the country's occupation-by-industry mix
  4. the timing model re-run with that lead and those shifts (forecast.forecast_tasks), rolled up to
     national codes and to the whole economy, and crossed with the other countries' weights for the
     job-mix / timing decomposition
  5. rung five: employment and pay change by national code against exposure
  6. the national skill levels (ANZSCO 1-5, SOC 1-4) against AI, beside the O*NET skill-level test
  7. the official task list per national code matched to O*NET tasks, so each code's page shows when AI
     could do each part of what the statistics office says the job involves; for the UK also the
     apprenticeship standards' duties
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from . import au_labour as au
from . import nz_labour as nz
from . import no_labour as no
from . import eu_labour as el
from . import jp_labour as jp
from . import kr_labour as kr
from . import ca_labour as ca
from . import forecast as fcm
from . import io
from . import uk_labour as uk
from . import us_economy as ue
from . import singapore_economy as se
from .ai_signals import _logit
from .forecast import YEARS
from .frontier import _weighted_mean
from .match import _rerank_with_bonus
from .outcomes import _reading, _spearman


@dataclass
class CountrySpec:
    code: str                      # ISO3, used in ForecastParams.geo and the page
    name: str
    adjective: str
    prefix: str                    # processed-table prefix: au_, uk_
    code_col: str                  # national code column: anzsco_code, soc_code
    title_col: str
    classification: str            # plain-words name of the national classification
    major_names: dict
    skill_level_names: dict
    industry_names: dict           # industry code -> label (the sector lens)
    industry_label: str
    currency: str
    wage_label: str
    emp_label: str
    wage_period: str
    adoption_source: str
    group_col: str                 # the level at which the occupation-by-industry mix is published
    weights: Callable[[], pd.DataFrame]
    outcomes: Callable[[], pd.DataFrame]
    isco_link: Callable[[set], pd.DataFrame]
    industry_mix: Callable[[], pd.DataFrame]     # columns: group_col, industry, employed
    ai_points: Callable[[], pd.DataFrame]        # columns: date, group_type (all/industry/size), industry, label, pct
    tasks: Callable[[], pd.DataFrame]            # columns: code_col, task_seq, task
    outcome_labels: dict = field(default_factory=dict)
    min_lead_year: float = 2024.0
    group_len: int = 3                 # digits of the national code that name the mix group (1 = major group)
    onet_link: Callable[[set], pd.DataFrame] | None = None   # a direct O*NET -> national crosswalk (onet_soc_code, nat_code, weight), used instead of the ISCO route
    descriptions: Callable[[], pd.DataFrame] | None = None   # per code: description, skill_text (for the job pages)
    titles: Callable[[], pd.DataFrame] | None = None         # per code: alternative titles (for the search)
    ratings: Callable[[], pd.DataFrame] | None = None        # national skill ratings per code: family, element_name, level
    level_descending: bool = False                           # True when a lower national skill-level number means higher skill (ANZSCO, NOC TEER)
    outcome_units: Callable[[], pd.DataFrame] | None = None  # per-code units for the outcome cells (emp_unit, wage_unit) where they are not thousands and the wage period
    lead_proxy: str | None = None                            # a country whose measured lead stands in when this one has no firm survey (marked on the page)
    depth: str = "full"                                      # full | medium | limited: how fine the country's own data is, marked on the page
    depth_note: str = ""                                     # one sentence on what is coarse or borrowed
    what_exists: str = ""                                    # for the leaders card: what the statistics office publishes
    lvl_kind: str = "isco"                                   # anzsco | teer | soc | isco: which skill-level scale the national level uses
    rating_bands: tuple = ()           # ((band, name, lo, hi), ...) on the national rating scale
    framework_name: str = ""           # what stands in for the Skills Framework
    survey_firms: str = "all"          # all | 10plus | 100plus: the firm sizes the national AI survey covers, which picks the US comparator curve the lead is read against
    survey_question: str = "production"   # production (used AI in producing goods or services, the Census question) | checklist (uses at least one AI technology from a list) | iot_ai (has introduced IoT or AI systems)
    survey_note: str = ""              # one sentence for the page on the survey's population and question, and how to read the lead


def _au_mix():
    d = au.load_industry_by_major(2025)
    return d.rename(columns={"employed_thousands": "employed"})[["major_group", "industry", "employed"]]


def _uk_mix():
    d = uk.load_census_occupation_industry()
    return d.rename(columns={"section": "industry"})[["minor", "industry", "employed"]]


def _au_isco(unit_codes):
    return au.load_isco_to_anzsco(unit_codes).rename(columns={"anzsco_code": "nat_code"})[["isco_code", "nat_code", "weight"]]


def _uk_isco(unit_codes):
    d = uk.load_soc_to_isco()
    d = d[d.soc_code.isin(unit_codes)] if unit_codes else d
    d = d.drop(columns=["weight"]).rename(columns={"soc_code": "nat_code", "weight_rev": "weight"})[["isco_code", "nat_code", "weight"]]
    d["weight"] = d.weight / d.groupby("isco_code").weight.transform("sum")
    return d


def _ca_mix():
    d = ca.load_industry_mix()
    return d[["minor", "industry", "employed"]]


SPECS: dict[str, CountrySpec] = {
    "CAN": CountrySpec(code="CAN", survey_firms="all", survey_question="production", survey_note="Statistics Canada asks businesses of every size whether they used AI in producing goods or delivering services in the last 12 months: the same question as the US survey, over a longer window, so the lead is a fair measurement.", name="Canada", adjective="Canadian", prefix="ca_", code_col="noc_code", title_col="noc_title", classification="NOC 2021 unit group", level_descending=True, lvl_kind="teer", what_exists="Statistics Canada's Census employment and income by NOC unit group, its quarterly vacancy survey, its business conditions survey with AI since 2024, the Census occupation-by-industry table, and OaSIS with duties, titles and skill ratings per unit group.",
                       major_names=ca.BROAD_NAMES, skill_level_names={k: v for k, v in ca.TEER_NAMES.items()}, industry_names=ca.NAICS_SECTORS, industry_label="NAICS sector", currency="C$",
                       wage_label="median employment income of full-year full-time workers, 2020 (Census 2021)", emp_label="employed persons, May 2021 (Census 2021)", wage_period="2020",
                       adoption_source="Canadian Survey on Business Conditions (Statistics Canada)", group_col="minor", group_len=4,
                       weights=ca.build_noc_weights, outcomes=ca.employment_change, outcome_units=ca.outcome_units, isco_link=lambda codes: pd.DataFrame(columns=["isco_code", "nat_code", "weight"]), onet_link=ca.load_onet_to_noc,
                       industry_mix=_ca_mix, ai_points=ca.load_csbc_ai_use, tasks=ca.load_oasis_duties, descriptions=ca.load_oasis_descriptions, titles=ca.load_oasis_titles, ratings=ca.load_oasis_ratings,
                       rating_bands=(("L", "not needed or basic (level 0 to 1)", 0.0, 1.5), ("M", "working level (2 to 3)", 1.5, 3.5), ("H", "advanced (4 to 5)", 3.5, 9.0)), framework_name="OaSIS, the Occupational and Skills Information System",
                       outcome_labels={"emp": "job vacancies change, four quarters to mid-2026 against the four before (Statistics Canada Job Vacancy and Wage Survey)", "wage": "Job Bank median wage change, 2024 to 2025 releases"}),
    "AUS": CountrySpec(code="AUS", survey_firms="all", survey_question="checklist", survey_note="The ABS asks businesses of every size which technologies they used in the year, with AI on the list: a looser question than the US survey's, so the lead is an upper bound.", name="Australia", adjective="Australian", prefix="au_", code_col="anzsco_code", title_col="anzsco_title", classification="ANZSCO unit group", level_descending=True, lvl_kind="anzsco",
                       major_names=au.MAJOR_GROUP_NAMES, skill_level_names=au.SKILL_LEVEL_NAMES, industry_names=au.DIVISIONS, industry_label="ANZSIC industry division", currency="A$",
                       wage_label="average weekly total cash earnings, May 2025 (ABS Employee Earnings and Hours)", emp_label="employed persons, 2025 average (ABS Labour Force)", wage_period="May 2025",
                       adoption_source="ABS Characteristics of Australian Business", group_col="major_group", group_len=1,
                       weights=au.build_anzsco_weights, outcomes=au.employment_change, isco_link=_au_isco, industry_mix=_au_mix, ai_points=au.load_cab_ai_use, tasks=au.load_anzsco_tasks,
                       outcome_labels={"emp": "employment change, 2024 to 2025 (ABS Labour Force, yearly averages)", "wage": "average weekly earnings change, May 2023 to May 2025 (ABS)"}),
    "NZL": CountrySpec(code="NZL", survey_firms="all", survey_question="checklist", survey_note="No New Zealand firm survey has asked about AI yet; Australia's reading stands in, with Australia's basis.", name="New Zealand", adjective="New Zealand", prefix="nz_", code_col="anzsco_code", title_col="anzsco_title", classification="ANZSCO unit group", level_descending=True, lvl_kind="anzsco", depth="limited", depth_note="Pay is the major group's median (eight figures), Australia's AI survey stands in for a New Zealand one, and there is no industry lens; the detailed Census pay and industry tables sit behind a Stats NZ API key.",
                       major_names=au.MAJOR_GROUP_NAMES, skill_level_names=au.SKILL_LEVEL_NAMES, industry_names=au.DIVISIONS, industry_label="ANZSIC industry division", currency="NZ$",
                       wage_label="median weekly earnings of the major occupation group, year to June 2026 (Stats NZ Household Labour Force Survey; Stats NZ publishes no finer breakdown openly)", emp_label="employed people, March 2023 (2023 Census)", wage_period="June 2026",
                       adoption_source="no New Zealand firm survey has asked about AI use yet; the Australian ABS readings stand in", group_col="major_group", group_len=1, lead_proxy="AUS",
                       weights=nz.build_anzsco_weights, outcomes=nz.employment_change, outcome_units=nz.outcome_units, isco_link=_au_isco, industry_mix=nz.load_industry_mix, ai_points=nz.load_ai_points, tasks=au.load_anzsco_tasks,
                       descriptions=nz.load_descriptions, titles=nz.load_titles,
                       outcome_labels={"emp": "employment change, 2018 to 2023 Censuses (Stats NZ; the 2023 Census changed its occupation coding, so small changes mean little)", "wage": "median weekly earnings change of the major group, June 2025 to June 2026 (Stats NZ HLFS)"}),
    "NOR": CountrySpec(code="NOR", survey_firms="10plus", survey_question="checklist", survey_note="The survey covers enterprises with 10 or more employees and asks which AI technologies they use, from a list: a looser question than the US survey's, so the lead is an upper bound; it is measured against US firms of the same sizes.", name="Norway", adjective="Norwegian", prefix="no_", code_col="styrk_code", title_col="styrk_title", classification="STYRK-08 occupation (ISCO-08 unit group)", depth_note="No official task list per occupation: job pages carry O*NET's tasks through the ISCO code.",
                       major_names=no.MAJOR_GROUP_NAMES, skill_level_names=no.SKILL_LEVEL_NAMES, industry_names=no.INDUSTRIES, industry_label="industry group", currency="NOK ",
                       wage_label="median monthly earnings, second quarter of 2026 (Statistics Norway register-based earnings)", emp_label="employees, second quarter of 2026 (Statistics Norway)", wage_period="2026 Q2",
                       adoption_source="Statistics Norway's survey of ICT usage in enterprises", group_col="major_group", group_len=1,
                       weights=no.build_styrk_weights, outcomes=no.employment_change, outcome_units=no.outcome_units, isco_link=no.load_isco_link, industry_mix=no.load_industry_mix, ai_points=no.load_ai_use, tasks=no.no_tasks, titles=no.load_titles,
                       outcome_labels={"emp": "employees change, second quarter of 2025 to second quarter of 2026 (Statistics Norway)", "wage": "median monthly earnings change, second quarter of 2025 to 2026 (Statistics Norway)"}),
    "SWE": CountrySpec(code="SWE", survey_firms="10plus", survey_question="checklist", survey_note="The survey covers enterprises with 10 or more employees and asks which AI technologies they use, from a list: a looser question than the US survey's, so the lead is an upper bound; it is measured against US firms of the same sizes.", name="Sweden", adjective="Swedish", prefix="se_", code_col="ssyk_code", title_col="ssyk_title", classification="SSYK 2012 occupation",
                       major_names=no.MAJOR_GROUP_NAMES, skill_level_names=no.SKILL_LEVEL_NAMES, industry_names=el.eu.NACE_SECTIONS, industry_label="NACE section", currency="SEK ",
                       wage_label="median monthly salary, 2025 (Statistics Sweden salary structure statistics)", emp_label="employees, 2024 (Statistics Sweden occupational register)", wage_period="2025",
                       adoption_source="Statistics Sweden's survey of ICT usage in enterprises", group_col="major_group", group_len=1,
                       weights=el.se_weights, outcomes=el.se_outcomes, outcome_units=el.se_units, isco_link=el.se_isco_link, industry_mix=lambda: el._mix("SE"), ai_points=el.se_ai_points, tasks=el.empty_tasks("ssyk_code"), titles=el.se_titles,
                       depth="full", depth_note="The link to O*NET runs through Statistics Sweden's own key, which pairs each ISCO minor group (three digits) with the SSYK classes it maps to; no official task list.",
                       what_exists="Statistics Sweden's occupational register (employees by four-digit SSYK 2012 occupation, yearly), its salary structure statistics at the same detail, its ICT-use survey with AI since 2021 by industry, and the EU Labour Force Survey's occupation-by-industry table.",
                       outcome_labels={"emp": "employees change, 2023 to 2024 (Statistics Sweden occupational register)", "wage": "median monthly salary change, 2024 to 2025 (Statistics Sweden)"}),
    "FIN": CountrySpec(code="FIN", survey_firms="10plus", survey_question="checklist", survey_note="The survey covers enterprises with 10 or more employees and asks which AI technologies they use, from a list: a looser question than the US survey's, so the lead is an upper bound; it is measured against US firms of the same sizes.", name="Finland", adjective="Finnish", prefix="fi_", code_col="tk_code", title_col="tk_title", classification="Classification of Occupations 2010 (ISCO-08 unit group)",
                       major_names=no.MAJOR_GROUP_NAMES, skill_level_names=no.SKILL_LEVEL_NAMES, industry_names=el.eu.NACE_SECTIONS, industry_label="NACE section", currency="€",
                       wage_label="median hourly earnings, 2024 (Statistics Finland structure of earnings)", emp_label="employed persons, 2024 (Statistics Finland employment statistics)", wage_period="2024",
                       adoption_source="Statistics Finland's survey of ICT use in enterprises", group_col="major_group", group_len=1,
                       weights=el.fi_weights, outcomes=el.fi_outcomes, outcome_units=el.fi_units, isco_link=el.fi_isco_link, industry_mix=lambda: el._mix("FI"), ai_points=el.fi_ai_points, tasks=el.empty_tasks("tk_code"), titles=el.fi_titles,
                       depth="full", depth_note="Finland's classification carries ISCO-08's own four-digit codes, so O*NET links directly; the earnings structure is published for one year so far, so there is no pay-change test; no official task list.",
                       what_exists="Statistics Finland's register-based employment statistics (employed by occupation to five digits, yearly), its structure of earnings by four-digit occupation, its ICT-use survey with AI since 2021 by industry, and the EU Labour Force Survey's occupation-by-industry table.",
                       outcome_labels={"emp": "employed persons change, 2023 to 2024 (Statistics Finland employment statistics)", "wage": "pay change not testable yet (one year of the earnings structure)"}),
    "DNK": CountrySpec(code="DNK", survey_firms="10plus", survey_question="checklist", survey_note="The survey covers enterprises with 10 or more employees and asks which AI technologies they use, from a list: a looser question than the US survey's, so the lead is an upper bound; it is measured against US firms of the same sizes.", name="Denmark", adjective="Danish", prefix="dk_", code_col="disco_code", title_col="disco_title", classification="DISCO-08 occupation (ISCO-08 unit group)",
                       major_names=no.MAJOR_GROUP_NAMES, skill_level_names=no.SKILL_LEVEL_NAMES, industry_names=el.eu.NACE_SECTIONS, industry_label="NACE section", currency="DKK ",
                       wage_label="earnings per hour worked, 2025 (Statistics Denmark earnings statistics)", emp_label="employed, 2025 (EU Labour Force Survey two-digit groups, split across four-digit occupations in Norway's register proportions)", wage_period="2025",
                       adoption_source="Statistics Denmark's survey of ICT usage in enterprises", group_col="major_group", group_len=1,
                       weights=el.dk_weights, outcomes=el.dk_outcomes, outcome_units=el.dk_units, isco_link=el.dk_isco_link, industry_mix=lambda: el._mix("DK"), ai_points=el.dk_ai_points, tasks=el.empty_tasks("disco_code"), titles=el.dk_titles,
                       depth="medium", depth_note="Pay is at four digits but employment by detailed occupation is not in the open API: each two-digit group's EU-LFS employment is split across its four-digit occupations in the proportions of Norway's register, the nearest country with those codes; no employment-change test; no official task list.",
                       what_exists="Statistics Denmark's earnings statistics by four-digit DISCO-08 occupation (yearly) and its ICT-use survey with AI since 2023 by industry; register employment is published by industry rather than occupation in the open API.",
                       outcome_labels={"emp": "employment change not testable (no employment by detailed occupation in the open API)", "wage": "hourly earnings change, 2024 to 2025 (Statistics Denmark)"}),
    "NLD": CountrySpec(code="NLD", survey_firms="10plus", survey_question="checklist", survey_note="The survey covers enterprises with 10 or more employees and asks which AI technologies they use, from a list: a looser question than the US survey's, so the lead is an upper bound; it is measured against US firms of the same sizes.", name="Netherlands", adjective="Dutch", prefix="nl_", code_col="brc_code", title_col="brc_title", classification="BRC 2014 occupation group",
                       major_names=no.MAJOR_GROUP_NAMES, skill_level_names=no.SKILL_LEVEL_NAMES, industry_names=el.eu.NACE_SECTIONS, industry_label="NACE section", currency="€",
                       wage_label="median hourly wage of employees, 2025 (Statistics Netherlands)", emp_label="employed persons, 2025 (Statistics Netherlands labour force survey)", wage_period="2025",
                       adoption_source="Statistics Netherlands' survey of ICT use by businesses", group_col="major_group", group_len=1,
                       weights=el.nl_weights, outcomes=el.nl_outcomes, outcome_units=el.nl_units, isco_link=el.nl_isco_link, industry_mix=lambda: el._mix("NL"), ai_points=el.nl_ai_points, tasks=el.empty_tasks("brc_code"), titles=el.nl_titles,
                       depth="medium", depth_note="The 114 BRC 2014 groups sit between two and three ISCO digits (a group often holds several ISCO occupations), titles are Dutch, and a few small groups take their segment's median wage; the ISCO composition of every group gives the link to O*NET; no official task list.",
                       what_exists="Statistics Netherlands' labour force survey employment and median hourly wages for the 114 BRC 2014 occupation groups, its ICT-use survey with AI since 2023 by industry, and the ROA-CBS publication of the ISCO 2008 composition of every group.",
                       outcome_labels={"emp": "employed persons change, 2024 to 2025 (Statistics Netherlands)", "wage": "median hourly wage change, 2024 to 2025 (Statistics Netherlands)"}),
    "IRL": CountrySpec(code="IRL", survey_firms="10plus", survey_question="checklist", survey_note="The CSO's survey covers enterprises with 10 or more employees and asks which AI technologies they use, from a list: a looser question than the US survey's, so the lead is an upper bound; it is measured against US firms of the same sizes.", name="Ireland", adjective="Irish", prefix="ie_", code_col="soc10_code", title_col="soc10_title", classification="SOC 2010 unit group (Census of Ireland)",
                       major_names={"1": "Managers, directors and senior officials", "2": "Professional occupations", "3": "Associate professional and technical", "4": "Administrative and secretarial", "5": "Skilled trades", "6": "Caring, leisure and other service", "7": "Sales and customer service", "8": "Process, plant and machine operatives", "9": "Elementary occupations"},
                       skill_level_names=no.SKILL_LEVEL_NAMES, industry_names=el.eu.NACE_SECTIONS, industry_label="NACE section", currency="€",
                       wage_label="mean hourly earnings of the major group, 2022 (Eurostat Structure of Earnings Survey; Ireland publishes no pay by occupation)", emp_label="persons in the labour force, April 2022 (Census 2022)", wage_period="2022",
                       adoption_source="CSO survey of ICT usage in enterprises (national reading only)", group_col="major_group", group_len=1,
                       weights=el.ie_weights, outcomes=el.ie_outcomes, outcome_units=el.ie_units, isco_link=el.ie_isco_link, industry_mix=lambda: el._mix("IE"), ai_points=el.ie_ai_points, tasks=el.empty_tasks("soc10_code"), titles=el.ie_titles,
                       depth="medium", depth_note="Employment is the Census 2022 labour force by 329 detailed occupations (people at work or looking for work), pay is the major group's mean from Eurostat (nine figures), and the AI survey is published nationally only, so there are no industry shifts; the link to O*NET runs through the ONS SOC 2010 mapping; no official task list.",
                       what_exists="The CSO's Census 2022 labour force by detailed occupational group (UK SOC 2010 codes) and its enterprise AI-use survey since 2023; pay by occupation is not published, and the AI survey has no industry breakdown.",
                       outcome_labels={"emp": "labour force change by occupation, Census 2016 to Census 2022 (CSO)", "wage": "pay change not testable (no pay by occupation)"}),
    "JPN": CountrySpec(code="JPN", survey_firms="100plus", survey_question="iot_ai", survey_note="The MIC survey covers enterprises with 100 or more employees and asks whether they have introduced IoT or AI systems: a looser question than the US survey's, so the lead is an upper bound; it is measured against US firms of the same sizes.", name="Japan", adjective="Japanese", prefix="jp_", code_col="jp_code", title_col="jp_title", classification="wage survey occupation class",
                       major_names=jp.MAJOR_GROUP_NAMES, skill_level_names=jp.SKILL_LEVEL_NAMES, industry_names=jp.INDUSTRIES, industry_label="industry group", currency="¥",
                       wage_label=f"monthly scheduled earnings, June {jp.WAGE_YEAR} (Basic Survey on Wage Structure, regular workers in enterprises with 10 or more employees)", emp_label=f"regular workers covered by the Basic Survey on Wage Structure, June {jp.WAGE_YEAR}", wage_period=f"June {jp.WAGE_YEAR}",
                       adoption_source="MIC Communications Usage Trend Survey, enterprises with 100 or more employees that have introduced IoT or AI systems", group_col="major_group", group_len=1,
                       weights=jp.build_weights, outcomes=jp.employment_change, outcome_units=jp.outcome_units, isco_link=jp.load_isco_link, industry_mix=jp.load_industry_mix, ai_points=jp.load_ai_points, tasks=jp.no_tasks, titles=jp.load_titles,
                       depth="medium", depth_note="The 144 occupation classes are the wage survey's (regular employees in enterprises with 10 or more employees, not all workers), the firm survey asks about IoT and AI together, and the link to ISCO-08 is a table written by hand for this map, since Japan publishes no official correspondence; no official task list." + ("" if len(jp._wage_years()) >= 2 else " No outcome test yet (one year of the wage survey loaded)."),
                       what_exists="The Basic Survey on Wage Structure's workers and monthly earnings by 144 occupation classes and by occupation group and industry, and the Communications Usage Trend Survey's yearly question on IoT or AI systems by industry since 2018; the Census publishes occupation by small group only in sample tabulations by prefecture.",
                       outcome_labels=({"emp": f"workers represented change, June {jp.WAGE_YEAR - 1} to June {jp.WAGE_YEAR} (Basic Survey on Wage Structure)", "wage": f"monthly scheduled earnings change, June {jp.WAGE_YEAR - 1} to June {jp.WAGE_YEAR} (Basic Survey on Wage Structure)"} if len(jp._wage_years()) >= 2 else {"emp": "employment change not testable yet (one year of the wage survey loaded)", "wage": "pay change not testable yet"})),
    "KOR": CountrySpec(code="KOR", survey_firms="50plus", survey_question="checklist", survey_note="Statistics Korea's Survey of Business Activities covers enterprises with 50 or more regular employees and asks which fourth-industrial-revolution technologies they develop or use, with artificial intelligence on the list: a looser question than the US survey's, so the lead is an upper bound; it is measured against US firms of the same sizes.", name="South Korea", adjective="Korean", prefix="kr_", code_col="ksco_code", title_col="ksco_title", classification="KSCO 2024 minor group (three digits)",
                       major_names=kr.MAJOR_GROUP_NAMES, skill_level_names=no.SKILL_LEVEL_NAMES, industry_names=el.eu.NACE_SECTIONS, industry_label="KSIC section", currency="₩",
                       wage_label="mean monthly total wages of employees, 2025 (Survey on Labour Conditions by Employment Type, carried to the 2024 classification)", emp_label="employed persons, first half of 2026 (Local Area Labour Force Survey)", wage_period="2025",
                       adoption_source="Statistics Korea's Survey of Business Activities, enterprises with 50 or more regular employees developing or using artificial intelligence", group_col="major_group", group_len=1,
                       weights=kr.build_weights, outcomes=kr.employment_change, outcome_units=kr.outcome_units, isco_link=kr.load_isco_link, industry_mix=kr.load_industry_mix, ai_points=kr.load_ai_points, tasks=kr.no_tasks, titles=kr.load_titles,
                       depth="medium", depth_note="Occupations are the 163 three-digit groups of the 2024 classification rather than unit groups; pay is employees' pay from a survey still on the 2017 classification, carried across through the official link; the firm survey covers only enterprises with 50 or more employees; no official task list. KOSIS has no API key for this record (membership needs a Korean mobile phone), so its tables are downloaded from the website.",
                       what_exists="KOSIS carries the Local Area Labour Force Survey (employed by three-digit occupation every half year, and by industry and occupation group), the Ministry of Employment and Labour's wage survey by three-digit occupation, and the Survey of Business Activities' yearly question on fourth-industrial-revolution technologies, including AI, by industry since 2017; Statistics Korea publishes official links from its occupation classification to ISCO-08.",
                       outcome_labels={"emp": "employed persons change, first half of 2025 to first half of 2026 (Local Area Labour Force Survey)", "wage": "mean monthly wage change, 2024 to 2025 (Survey on Labour Conditions by Employment Type)"}),
    "GBR": CountrySpec(code="GBR", survey_firms="10plus", survey_question="checklist", survey_note="The ONS asks businesses with 10 or more employees which AI technologies they currently use, from a list: a looser question than the US survey's, so the lead is an upper bound; it is measured against US firms of the same sizes.", name="United Kingdom", adjective="UK", prefix="uk_", code_col="soc_code", title_col="soc_title", classification="SOC 2020 unit group", lvl_kind="soc",
                       major_names=uk.MAJOR_GROUP_NAMES, skill_level_names=uk.SKILL_LEVEL_NAMES, industry_names=uk.SECTIONS, industry_label="SIC industry section", currency="£",
                       wage_label=f"median gross annual pay, April {uk.ASHE_YEAR} (ONS ASHE, provisional)", emp_label=f"employee jobs, April {uk.ASHE_YEAR} (ONS ASHE)", wage_period=f"April {uk.ASHE_YEAR}",
                       adoption_source="ONS Business Insights and Conditions Survey", group_col="minor", group_len=3,
                       weights=uk.build_soc_weights, outcomes=uk.employment_change, isco_link=_uk_isco, industry_mix=_uk_mix, ai_points=uk.load_bics_ai_use, tasks=uk.load_soc_tasks,
                       outcome_labels={"emp": f"employee jobs change, April {uk.ASHE_YEAR - 1} to April {uk.ASHE_YEAR} (ONS ASHE)", "wage": f"median gross annual pay change, April {uk.ASHE_YEAR - 1} to April {uk.ASHE_YEAR} (ONS ASHE)"}),
}


# ----------------------------------------------------------------------------- 1. crosswalk and 2. weights
def national_crosswalk(spec: CountrySpec, unit_codes: set[str] | None = None) -> pd.DataFrame:
    """Compose O*NET -> ISCO-08 with ISCO-08 -> national code. Columns: onet_soc_code, nat_code, weight
    (per O*NET occupation), weight_rev (per national code), n_isco_paths."""
    if spec.onet_link is not None:
        g = spec.onet_link(unit_codes).rename(columns={"weight": "path_w"})
        g["n_isco_paths"] = 0
    else:
        a = io.load("xw_onetsoc_isco08")[["onet_soc_code", "isco_code", "weight"]].rename(columns={"weight": "w1"})
        b = spec.isco_link(unit_codes).rename(columns={"weight": "w2"})
        m = a.merge(b, on="isco_code", how="inner")
        m["path_w"] = m.w1 * m.w2
        g = m.groupby(["onet_soc_code", "nat_code"]).agg(path_w=("path_w", "sum"), n_isco_paths=("isco_code", "nunique")).reset_index()
    g["weight"] = g.path_w / g.groupby("onet_soc_code").path_w.transform("sum")
    g["weight_rev"] = g.path_w / g.groupby("nat_code").path_w.transform("sum")
    g["country"] = spec.code
    return g.drop(columns="path_w").sort_values(["onet_soc_code", "weight"], ascending=[True, False]).reset_index(drop=True)


def onet_weights_from_national(xw: pd.DataFrame, natw: pd.DataFrame, spec: CountrySpec) -> pd.DataFrame:
    """O*NET-level mirror of the national weights: each national code's employment is shared over the
    O*NET occupations it maps to by weight_rev; pay is the employment-weighted mean. The occupation's
    group labels come from the national code carrying most of its workers."""
    n = natw.rename(columns={spec.code_col: "nat_code"})
    m = xw.merge(n[["nat_code", "employment_thousands", "wage_value", "major_group", "submajor_label", "submajor_name", "skill_level"]], on="nat_code", how="inner")
    m["emp"] = m.employment_thousands * m.weight_rev
    m["wv"] = m.wage_value * m.weight_rev
    g = m.groupby("onet_soc_code").agg(employment_thousands=("emp", "sum"), wage_value=("wv", "sum"), n_nat_codes=("nat_code", "nunique")).reset_index()
    top = m.sort_values("emp", ascending=False).drop_duplicates("onet_soc_code").set_index("onet_soc_code")
    g["major_group"] = g.onet_soc_code.map(top.major_group)
    g["submajor_label"] = g.onet_soc_code.map(top.submajor_label)
    g["submajor_name"] = g.onet_soc_code.map(top.submajor_name)
    g["skill_level"] = g.onet_soc_code.map(top.skill_level)
    g["nat_code_top"] = g.onet_soc_code.map(top.nat_code)
    g["wage_median"] = g.wage_value / g.employment_thousands.where(g.employment_thousands > 0)
    g["employment_method"] = "national_code_employment_split_by_crosswalk_weight"
    g["country"] = spec.code
    return g


def push_frontier(occf: pd.DataFrame, xw: pd.DataFrame) -> pd.DataFrame:
    """National-code frontier scores = weight_rev-weighted means over the O*NET occupations mapping in."""
    cols = [c for c in ["observed_automation", "usage_index", "automation_share", "share_weight_observed", "share_weight_physical", "aei_job_exposure", "ai_applicability_score", "robot_exposure_index"] if c in occf.columns]
    m = xw.merge(occf[["onet_soc_code", *cols]], on="onet_soc_code", how="inner")
    out = pd.DataFrame({"n_onet_occupations": m.groupby("nat_code").onet_soc_code.nunique()})
    for c in cols:
        out[c] = _weighted_mean(m, "nat_code", c, "weight_rev")
    return out.reset_index()


def national_curve(fc: pd.DataFrame, xw: pd.DataFrame, nat_shift: pd.Series | None, params: fcm.ForecastParams, now_y: float) -> pd.DataFrame:
    """Disruption curve per national code from the task forecast: tasks weighted by task weight times the
    crosswalk share, moved from their occupation's industry shift to the national code's own."""
    f = fc.merge(xw[["onet_soc_code", "nat_code", "weight_rev"]], on="onet_soc_code")
    f["w"] = f.task_weight * f.weight_rev
    if nat_shift is not None and len(nat_shift):
        f = fcm.reshift_years(f, f.nat_code.map(nat_shift), params, now_y)
    return fcm.disruption_curve(f, "nat_code", weight="w")


def weighted_table(natw: pd.DataFrame, frontier_nat: pd.DataFrame, curve_nat: pd.DataFrame, spec: CountrySpec) -> pd.DataFrame:
    """One row per national code: weights, frontier scores and timing, shaped for economy_exposure."""
    share_cols = [c for c in curve_nat.columns if c.startswith("share_by_") or c.startswith("done_by_") or c.startswith("half_year_") or c.startswith("quarter_done_year_") or c.startswith("half_done_year_") or c in ("share_beyond_horizon", "share_already_observed", "ceiling_ai_share", "barrier_index", "n_tasks")]
    t = natw.rename(columns={spec.code_col: "nat_code", spec.title_col: "nat_title"}).merge(frontier_nat, on="nat_code", how="left").merge(curve_nat[["nat_code", *share_cols]], on="nat_code", how="left")
    t["major_group_name"] = t.major_group.map(spec.major_names)
    t["has_scores"] = t.observed_automation.notna()
    t["country"] = spec.code
    return t


def economy(t: pd.DataFrame, spec: CountrySpec) -> pd.DataFrame:
    out = se.economy_exposure(t)
    out["country"] = spec.code
    out.loc[out.scope == "economy", "scope_name"] = f"Whole economy ({spec.name})"
    maj = out.level == "major_group"
    out.loc[maj, "scope_name"] = out.loc[maj, "scope"].str[-1].map(spec.major_names).fillna(out.loc[maj, "scope_name"])
    return out


# ----------------------------------------------------------------------------- 3. the country's own adoption: lead and industry shifts
def firm_lead(fit: dict, points: pd.DataFrame, min_year: float = 2024.0) -> dict:
    """Years the country's firms run ahead of (positive) or behind (negative) US firms at matched adoption
    levels: for each national point the date at which the fitted US curve reaches the same share is read
    off, and the lead is the mean gap over points dated from `min_year` (the generative-AI years)."""
    K, a, b = fit["saturation"], fit["intercept"], fit["growth_per_year"]
    pts = points[points.group_type == "all"].dropna(subset=["pct", "date"]).sort_values("date")
    rows = []
    for r in pts.itertuples():
        p = min(max(float(r.pct) / 100.0, 0.002), K - 0.002)
        us_year = 2024.0 + (float(_logit(np.array(p), K)) - a) / b
        rows.append({"date": float(r.date), "label": str(getattr(r, "period", "") or getattr(r, "wave", "")), "pct": float(r.pct), "us_curve_reaches_same_share": us_year, "lead_years": us_year - float(r.date)})
    use = [x for x in rows if x["date"] >= min_year]
    lead = float(np.mean([x["lead_years"] for x in use])) if use else None
    own = {}
    if len(pts) >= 2:
        t = pts.date.values.astype(float) - 2024.0
        yy = _logit(np.clip(pts.pct.values / 100.0, 0.002, K - 0.002), K)
        bs, as_ = np.polyfit(t, yy, 1)
        own = {"own_growth_per_year": float(bs), "own_year_cross_50pct": float(2024.0 + (float(_logit(np.array(0.5), K)) - as_) / bs) if bs > 0 else None, "own_n_points": int(len(pts))}
    return {"points": rows, "lead_years": lead, "n_points_used": len(use), "latest_pct": float(pts.pct.iloc[-1]) if len(pts) else None, "latest_date": float(pts.date.iloc[-1]) if len(pts) else None, **own}


def lead_basis(spec: CountrySpec, cmp: dict, lead_all) -> dict:
    """What the lead was measured against and how to read it: the US comparator curve (all firms, or the
    firm sizes the national survey covers), the kind of question the survey asks, whether the lead is an
    upper bound (a checklist or an IoT-and-AI question counts more firms as users than the Census question
    on AI in producing goods or services, so it can only overstate a country's position), and the lead
    against all US firms for reference."""
    q = spec.survey_question
    return {"comparator": cmp.get("comparator", "all"), "comparator_label": cmp.get("label", "all US firms"), "comparator_fallback": bool(cmp.get("fallback", False)),
            "question": q, "upper_bound": q != "production", "lead_years_all_firms": lead_all, "survey_firms": spec.survey_firms, "survey_note": spec.survey_note}


def industry_offsets(points: pd.DataFrame, fit: dict, min_national_pct: float = 3.0) -> pd.DataFrame:
    """Each industry's adoption level against the country's own national level at the same date, in
    logit units averaged over the survey dates where the national share is at least `min_national_pct`
    (earlier readings of 0-1% are too small to carry a level). Converted to years with the US growth
    rate: shift = -offset / b (negative = earlier than the national curve)."""
    K, b = fit["saturation"], fit["growth_per_year"]
    nat = points[points.group_type == "all"].dropna(subset=["pct"]).set_index("date").pct
    d = points[(points.group_type == "industry")].dropna(subset=["pct"]).copy()
    d["national_pct"] = d.date.map(nat)
    d = d[d.national_pct >= min_national_pct]
    if d.empty:
        return pd.DataFrame(columns=["industry", "label", "n_points", "latest_date", "latest_pct", "national_pct_latest", "level_offset", "anchor_shift_years"])
    clip = lambda p: np.clip(p / 100.0, 0.002, K - 0.002)
    d["resid"] = _logit(clip(d.pct.values), K) - _logit(clip(d.national_pct.values), K)
    g = d.sort_values("date").groupby("industry")
    out = pd.DataFrame({"label": g.label.first(), "n_points": g.size(), "latest_date": g.date.last(), "latest_pct": g.pct.last(), "national_pct_latest": g.national_pct.last(), "level_offset": g.resid.mean()}).reset_index()
    out["anchor_shift_years"] = -out.level_offset / b
    return out.sort_values("anchor_shift_years").reset_index(drop=True)


def occupation_shifts(mix: pd.DataFrame, ind_shift: pd.DataFrame, natw: pd.DataFrame, xw: pd.DataFrame, spec: CountrySpec) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(group shifts, national-code shifts, O*NET shifts). A group's shift is its industry mix times the
    industry shifts (industries without a survey reading sit at the national level, 0); every code in the
    group inherits it; an O*NET occupation takes the weight_rev-weighted mean of its codes."""
    gcol = spec.group_col
    m = mix.merge(ind_shift[["industry", "anchor_shift_years"]], on="industry", how="left")
    m["anchor_shift_years"] = pd.to_numeric(m.anchor_shift_years, errors="coerce").fillna(0.0).astype(float)   # an empty shift table arrives as object dtype
    m["share"] = m.employed / m.groupby(gcol).employed.transform("sum")
    grp = m.assign(_x=m.share * m.anchor_shift_years).groupby(gcol).agg(adoption_shift_years=("_x", "sum"), employed=("employed", "sum")).reset_index()
    codes = natw[[spec.code_col]].rename(columns={spec.code_col: "nat_code"}).copy()
    codes["group"] = codes.nat_code.str[:spec.group_len]
    codes["adoption_shift_years"] = pd.to_numeric(codes.group.map(grp.set_index(gcol).adoption_shift_years), errors="coerce").fillna(0.0).astype(float)
    o = xw[["onet_soc_code", "nat_code", "weight_rev"]].merge(codes[["nat_code", "adoption_shift_years"]], on="nat_code", how="left")
    o["adoption_shift_years"] = pd.to_numeric(o.adoption_shift_years, errors="coerce").fillna(0.0).astype(float)
    o["_x"] = o.weight_rev * o.adoption_shift_years
    onet = o.groupby("onet_soc_code").agg(_x=("_x", "sum"), w=("weight_rev", "sum")).reset_index()
    onet["adoption_shift_years"] = np.where(onet.w > 0, onet._x / onet.w, 0.0)
    return grp, codes.drop(columns="group"), onet[["onet_soc_code", "adoption_shift_years"]]


# ----------------------------------------------------------------------------- 5. rung five and 6. skill levels
def outcome_tests(change: pd.DataFrame, t: pd.DataFrame, spec: CountrySpec) -> dict:
    """Does employment or pay change by national code relate to exposure? Rank correlations plus the
    employment-weighted change by exposure tercile (a negative rho is the sign disruption would show)."""
    c = change.rename(columns={spec.code_col: "nat_code"})
    d = c.merge(t[["nat_code", "employment_thousands", "done_by_2035", "share_by_2035", "share_weight_observed", "observed_automation"]], on="nat_code", how="inner")
    yf, yt = int(c.year_from.iloc[0]), int(c.year_to.iloc[0])
    tests = []
    for outcome, label in [("emp_change_pct", spec.outcome_labels.get("emp", f"employment change {yf} to {yt}")), ("wage_change_pct", spec.outcome_labels.get("wage", f"pay change {yf} to {yt}"))]:
        for sig, sig_label in [("done_by_2035", "AI doing by 2035 (central)"), ("share_weight_observed", "share of work seen in AI use"), ("observed_automation", "AI already doing the work")]:
            dd = d.dropna(subset=[outcome, sig])
            dd = dd[np.isfinite(dd[outcome])]
            rho, n = _spearman(dd[outcome], dd[sig])
            tests.append({"level": f"detailed occupation ({spec.classification})", "outcome": label, "signal": sig_label, "rho": rho, "n": n, "reading": _reading(rho, n, -1)})
    occ_rhos = [x["rho"] for x in tests if x["rho"] is not None]
    dd = d.dropna(subset=["done_by_2035", f"emp_{yf}", f"emp_{yt}"])
    dd = dd[(dd[f"emp_{yf}"] > 0)]
    terc = {}
    if len(dd) >= 9:
        dd = dd.assign(tercile=pd.qcut(dd.done_by_2035.rank(method="first"), 3, labels=["least exposed third", "middle third", "most exposed third"]))
        terc = {str(k): float(v) for k, v in dd.groupby("tercile", observed=True).apply(lambda g: 100.0 * (g[f"emp_{yt}"].sum() / g[f"emp_{yf}"].sum() - 1.0), include_groups=False).items()}
    return {"tests": tests, "headline": {"occupation": {"median_signed_rho": float(np.median(occ_rhos)) if occ_rhos else None, "n_tests": len(occ_rhos)}}, "n_codes": int(len(d)), "years": [yf, yt], "wage_years": str(c.wage_years.iloc[0]) if "wage_years" in c else f"{yf}-{yt}",
            "total_employment_change_pct": float(100.0 * (dd[f"emp_{yt}"].sum() / dd[f"emp_{yf}"].sum() - 1.0)) if len(dd) else None, "emp_change_by_exposure_tercile": terc, "labels": spec.outcome_labels}


def skill_level_bands(t: pd.DataFrame, spec: CountrySpec) -> tuple[pd.DataFrame, dict]:
    """Exposure by the national skill level (employment-weighted), and the rank correlation between a
    code's level and 'AI doing by 2035' (positive = AI reaches the higher-skilled work first)."""
    d = t.dropna(subset=["skill_level"]).copy()
    d["band"] = d.skill_level.round().astype(int)
    rows = []
    for lvl, g in d.groupby("band"):
        if g.employment_thousands.sum() <= 0:
            continue
        row = {"level": int(lvl), "level_name": spec.skill_level_names.get(int(lvl), f"level {lvl}"), "n_codes": int(len(g)), "employment_thousands": float(g.employment_thousands.sum()), "employment_share": float(g.employment_thousands.sum() / d.employment_thousands.sum())}
        for c in ["observed_automation", "share_weight_observed", "share_weight_physical", "done_by_2035", "done_by_2035_p10", "done_by_2035_p90", "share_by_2035", "done_by_2030", "ceiling_ai_share", "quarter_done_year_p10", "quarter_done_year_p50", "quarter_done_year_p90"]:
            if c in g.columns:
                row[c] = se._wavg(g, c, "employment_thousands")
        rows.append(row)
    dd = d.dropna(subset=["done_by_2035"])
    rho, n = _spearman(dd.skill_level, dd.done_by_2035)
    rho_seen, _ = _spearman(dd.skill_level, dd.share_weight_observed)
    # ANZSCO and the NOC TEER count down (1 or 0 = highest skill), SOC counts up (4 = highest): read the sign accordingly
    higher_first = (rho is not None) and ((rho < -0.15) if spec.level_descending else (rho > 0.15))
    lower_first = (rho is not None) and ((rho > 0.15) if spec.level_descending else (rho < -0.15))
    reading = "no clear pattern across skill levels" if rho is None or (not higher_first and not lower_first) else ("AI reaches the higher-skilled work first" if higher_first else "AI reaches the lower-skilled work first")
    return pd.DataFrame(rows), {"rho_level_vs_done_2035": rho, "rho_level_vs_seen": rho_seen, "n_codes": n, "reading": reading}


def rating_level_exposure(t: pd.DataFrame, ratings: pd.DataFrame, spec: CountrySpec, families: tuple = ("skill",), min_codes: int = 5) -> dict:
    """The country's own skill ratings against AI, as us_economy.skill_level_exposure does with O*NET levels:
    codes are banded by the level each skill needs (spec.rating_bands), exposure is employment-weighted
    within each band, and a rank correlation between level and 'AI doing by 2035' is given per skill and
    for the average level over all the family's skills."""
    r = ratings[ratings.family.isin(families)].rename(columns={"noc_code": "nat_code"}) if "noc_code" in ratings.columns else ratings[ratings.family.isin(families)]
    base = t[["nat_code", "employment_thousands", "share_weight_observed", "observed_automation", "done_by_2035", "done_by_2035_p10", "done_by_2035_p90", "quarter_done_year_p10", "quarter_done_year_p50", "quarter_done_year_p90"]].dropna(subset=["done_by_2035"])
    def band_of(level):
        for b, name, lo, hi in spec.rating_bands:
            if lo <= level < hi:
                return b
        return None
    rows, corr = [], []
    def bands_for(d, element_name, family):
        d = d.assign(band=d.level.map(band_of)).dropna(subset=["band"])
        for b, name, lo, hi in spec.rating_bands:
            g = d[d.band == b]
            if len(g) < min_codes or g.employment_thousands.sum() <= 0:
                continue
            row = {"family": family, "element_name": element_name, "band": b, "band_name": name, "n_occupations": int(len(g))}
            for c in ["share_weight_observed", "observed_automation", "done_by_2035", "done_by_2035_p10", "done_by_2035_p90", "quarter_done_year_p10", "quarter_done_year_p50", "quarter_done_year_p90"]:
                row[c] = se._wavg(g, c, "employment_thousands")
            row["quarter_done_year_fast_from_curve"], row["quarter_done_year_central_from_curve"], row["quarter_done_year_slow_from_curve"] = row["quarter_done_year_p10"], row["quarter_done_year_p50"], row["quarter_done_year_p90"]
            rows.append(row)
        rho, n = _spearman(d.level, d.done_by_2035); rho_s, _ = _spearman(d.level, d.share_weight_observed)
        corr.append({"family": family, "element_name": element_name, "rho_level_vs_done_2035": rho, "rho_level_vs_seen": rho_s, "n_occupations": n})
    for (fam, name), g in r.groupby(["family", "element_name"]):
        d = base.merge(g[["nat_code", "level"]], on="nat_code", how="inner")
        if len(d) >= 3 * min_codes:
            bands_for(d, name, fam)
    idx = r[r.family == families[0]].groupby("nat_code")["level"].mean().reset_index()
    d = base.merge(idx, on="nat_code", how="inner")
    bands_for(d, f"All {families[0].replace('_', ' ')}s (average level)", "index")
    return {"bands": pd.DataFrame(rows), "correlations": pd.DataFrame(corr)}


def industry_view(mix: pd.DataFrame, ind_shift: pd.DataFrame, points: pd.DataFrame, t: pd.DataFrame, spec: CountrySpec) -> pd.DataFrame:
    """The sector lens: per industry, its share of employment, the firm AI-use readings, its shift against
    the national curve, and exposure through its occupation mix (group-level, employment-weighted)."""
    if mix is None or mix.empty:   # no occupation-by-industry table (New Zealand without an API key): no industry lens
        return pd.DataFrame(columns=["industry", "label", "employment_share", "ai_use_latest_pct", "ai_use_latest_date", "ai_use_first_pct", "ai_use_first_date", "anchor_shift_years", "share_of_groups_scored"])
    gcol = spec.group_col
    key = t.nat_code.str[:spec.group_len]
    cols = ["observed_automation", "share_weight_observed", "share_weight_physical", "done_by_2030", "done_by_2035", "done_by_2035_p10", "done_by_2035_p90", "share_by_2035", "ceiling_ai_share", "quarter_done_year_p10", "quarter_done_year_p50", "quarter_done_year_p90"]
    cols = [c for c in cols if c in t.columns]
    grp = pd.DataFrame({c: t.assign(_g=key).groupby("_g").apply(lambda g: se._wavg(g, c, "employment_thousands"), include_groups=False) for c in cols})
    m = mix.copy()
    m["_g"] = m[gcol].astype(str)
    m["share_in_industry"] = m.employed / m.groupby("industry").employed.transform("sum")
    tot = m.employed.sum()
    rows = []
    pts = points[points.group_type == "industry"].dropna(subset=["pct"]).sort_values("date")
    for ind, g in m.groupby("industry"):
        row = {"industry": ind, "label": spec.industry_names.get(ind, ind), "employment_share": float(g.employed.sum() / tot) if tot else np.nan}
        gg = g.merge(grp, left_on="_g", right_index=True, how="left")
        for c in cols:
            ok = gg[c].notna()
            row[c] = float(np.average(gg.loc[ok, c], weights=gg.loc[ok, "share_in_industry"])) if ok.any() and gg.loc[ok, "share_in_industry"].sum() > 0 else np.nan
        row["share_of_groups_scored"] = float(gg.loc[gg[cols[0]].notna(), "share_in_industry"].sum()) if cols else np.nan
        p = pts[pts.industry == ind]
        row["ai_use_latest_pct"] = float(p.pct.iloc[-1]) if len(p) else np.nan
        row["ai_use_latest_date"] = float(p.date.iloc[-1]) if len(p) else np.nan
        row["ai_use_first_pct"] = float(p.pct.iloc[0]) if len(p) > 1 else np.nan
        row["ai_use_first_date"] = float(p.date.iloc[0]) if len(p) > 1 else np.nan
        s = ind_shift[ind_shift.industry == ind]
        row["anchor_shift_years"] = float(s.anchor_shift_years.iloc[0]) if len(s) else 0.0
        row["surveyed"] = bool(len(s))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("employment_share", ascending=False).reset_index(drop=True)


# ----------------------------------------------------------------------------- 7. the official task lists against O*NET tasks
def match_tasks(tasks: pd.DataFrame, xw: pd.DataFrame, spec: CountrySpec, k_global: int = 20, k_out: int = 3, bonus: float = 0.10) -> pd.DataFrame:
    """Each official task item -> its closest O*NET tasks (hybrid text similarity), with a bonus for
    tasks of the O*NET occupations the code maps to, as match.match_ssoc_tasks does for Singapore."""
    from . import onet
    from .embed import HybridMatcher

    ot = onet.load_tasks()
    neigh = xw.groupby("nat_code").onet_soc_code.apply(set).to_dict()
    t = tasks.rename(columns={spec.code_col: "nat_code"}).reset_index(drop=True)
    if t.empty:  # a country with no official task list (Norway): nothing to match
        return pd.DataFrame({"nat_code": pd.Series(dtype=str), "task_seq": pd.Series(dtype=int), "rank": pd.Series(dtype=int), "task_id": pd.Series(dtype=int), "onet_soc_code": pd.Series(dtype=str), "score": pd.Series(dtype=float), "in_mapped_occ": pd.Series(dtype=bool), "country": pd.Series(dtype=str)})
    m = HybridMatcher(ot.task_statement.tolist())
    idx, sc = m.query(t.task.tolist(), k=k_global)
    allowed = [neigh.get(c, set()) for c in t.nat_code]
    ridx, rsc, rin = _rerank_with_bonus(idx, sc, allowed, ot.onet_soc_code.values, bonus=bonus, k_out=k_out)
    rows = [(t.nat_code.iat[i], int(t.task_seq.iat[i]), r + 1, int(ot.task_id.iat[ridx[i, r]]), ot.onet_soc_code.iat[ridx[i, r]], float(rsc[i, r]), bool(rin[i, r])) for i in range(len(t)) for r in range(k_out)]
    out = pd.DataFrame(rows, columns=["nat_code", "task_seq", "rank", "task_id", "onet_soc_code", "score", "in_mapped_occ"])
    out["country"] = spec.code
    return out


def match_duties(standards: pd.DataFrame, std_soc: pd.DataFrame, xw: pd.DataFrame, k_global: int = 20, bonus: float = 0.10) -> pd.DataFrame:
    """UK apprenticeship duties -> the closest O*NET task, with the neighbourhood bonus for the O*NET
    occupations of the standard's SOC codes."""
    from . import onet
    from .embed import HybridMatcher

    ot = onet.load_tasks()
    soc_neigh = xw.groupby("nat_code").onet_soc_code.apply(set).to_dict()
    std_neigh = {}
    for r in std_soc.itertuples():
        std_neigh.setdefault(r.reference, set()).update(soc_neigh.get(r.soc_code, set()))
    rows_q = [(s.reference, i, d) for s in standards.itertuples() for i, d in enumerate(s.duties) if d]
    if not rows_q:
        return pd.DataFrame(columns=["reference", "duty_seq", "task_id", "onet_soc_code", "score", "in_mapped_occ"])
    m = HybridMatcher(ot.task_statement.tolist())
    idx, sc = m.query([q[2][:600] for q in rows_q], k=k_global)
    allowed = [std_neigh.get(q[0], set()) for q in rows_q]
    ridx, rsc, rin = _rerank_with_bonus(idx, sc, allowed, ot.onet_soc_code.values, bonus=bonus, k_out=1)
    return pd.DataFrame([(rows_q[i][0], rows_q[i][1], int(ot.task_id.iat[ridx[i, 0]]), ot.onet_soc_code.iat[ridx[i, 0]], float(rsc[i, 0]), bool(rin[i, 0])) for i in range(len(rows_q))], columns=["reference", "duty_seq", "task_id", "onet_soc_code", "score", "in_mapped_occ"])


# ----------------------------------------------------------------------------- orchestration
def build_base(spec: CountrySpec) -> dict:
    """The slow, model-independent tables for one country: weights, crosswalk, task matches, industry mix,
    adoption readings (and for the UK the apprenticeship standards). `python -m onetx countries`."""
    p = spec.prefix
    natw = spec.weights(); io.save(natw, f"{p}weights")
    codes = set(natw[spec.code_col].astype(str))
    xw = national_crosswalk(spec, codes); io.save(xw, f"{p}onet_crosswalk")
    io.save(onet_weights_from_national(xw, natw, spec), f"{p}onet_weights")
    io.save(spec.industry_mix(), f"{p}industry_mix")
    io.save(spec.ai_points(), f"{p}ai_adoption")
    tasks = spec.tasks(); io.save(tasks, f"{p}tasks")
    links = match_tasks(tasks, xw, spec); io.save(links, f"{p}task_onet_task")
    stats = {"codes": int(len(natw)), "codes_linked_to_onet": int(xw.nat_code.nunique()), "onet_occupations_linked": int(xw.onet_soc_code.nunique()), "employment_thousands_total": float(natw.employment_thousands.sum()),
             "wage_sources": natw.wage_source.value_counts(dropna=False).to_dict(), "official_tasks": int(len(tasks)), "codes_with_tasks": int(tasks[spec.code_col].nunique()), "task_match_mean_score": float(links[links["rank"] == 1].score.mean()) if len(links) else None, "task_match_in_mapped_occ_share": float(links[links["rank"] == 1].in_mapped_occ.mean())}
    if spec.descriptions is not None:
        io.save(spec.descriptions(), f"{p}descriptions")
    if spec.titles is not None:
        io.save(spec.titles(), f"{p}titles")
    if spec.ratings is not None:
        rt = spec.ratings(); io.save(rt, f"{p}ratings")
        stats["national_ratings"] = {"codes": int(rt[spec.code_col].nunique() if spec.code_col in rt.columns else rt.nat_code.nunique()), "families": rt.family.value_counts().to_dict()}
    if spec.code == "AUS":
        io.save(au.load_anzsco_pages().drop_duplicates("anzsco_code")[["anzsco_code", "description", "skill_text"]], "au_descriptions")
        io.save(au.load_anzsco_structure(), "au_structure")
    if spec.code == "GBR":
        st = uk.load_soc_structure(); io.save(st, "uk_structure")
        stds = uk.load_ifate_standards(); io.save(stds, "uk_ifate_standards")
        ss = uk.match_standards_to_soc(stds, st); io.save(ss, "uk_ifate_standard_soc")
        io.save(match_duties(stds, ss, xw), "uk_ifate_duty_onet_task")
        stats["apprenticeship_standards"] = {"approved": int(len(stds)), "matched_by_title": int(ss[ss.method == "title_in_coding_index"].reference.nunique()), "matched_by_text": int(ss[ss.method == "text_match"].reference.nunique()), "routes": int(stds.route.nunique()), "duties": int(stds.duties.map(len).sum())}
    return stats


def build_country(spec: CountrySpec, tf, barriers, params, now, tiers, occ, occf, btos_fit, o2s, ssoc_w, frontier_ssoc, econ_sg, fc_sg, us_w, econ_us, fc_us, others: dict | None = None, comparators: dict | None = None) -> tuple[dict, dict]:
    """The timing run and every view for one country. Returns (stats block, context for cross runs).
    `comparators` (ai_signals.btos_comparators) holds the US curves by firm size; the lead is read against
    the one matching the national survey's coverage (spec.survey_firms)."""
    p = spec.prefix
    now_y = fcm._frac_year(pd.Timestamp(now))
    natw = io.load(f"{p}weights"); xw = io.load(f"{p}onet_crosswalk"); mix = io.load(f"{p}industry_mix"); points = io.load(f"{p}ai_adoption"); onet_w = io.load(f"{p}onet_weights")
    # 3. lead and sector shifts from the country's own survey
    cmp = (comparators or {}).get(spec.survey_firms) or btos_fit
    lead = firm_lead(cmp, points, spec.min_lead_year)
    lead_all = lead["lead_years"] if cmp is btos_fit else firm_lead(btos_fit, points, spec.min_lead_year)["lead_years"]
    lead.update(lead_basis(spec, cmp, lead_all))
    if lead["lead_years"] is None and spec.lead_proxy and others and spec.lead_proxy in others:
        # no national survey: the proxy country's measured lead stands in, with its basis, and the page says so
        lead = {**others[spec.lead_proxy]["lead"], "proxy": spec.lead_proxy, "own_points": int(len(points))}
    ind = industry_offsets(points, cmp); io.save(ind, f"{p}adoption_shift_industry")
    grp_shift, nat_shift, onet_shift = occupation_shifts(mix, ind, natw, xw, spec)
    io.save(nat_shift, f"{p}adoption_shift_national"); io.save(onet_shift, f"{p}adoption_shift_onet")
    # 4. the timing model on this country's lead and shifts
    params_c = fcm.ForecastParams(**{**fcm.asdict(params), "geo": spec.code, "geo_lead_years_measured": lead["lead_years"] if lead["lead_years"] is not None else 0.0})
    fc = fcm.forecast_tasks(tf, barriers, params_c, now=now, tiers=tiers, occ_shift=onet_shift); io.save(fc, f"forecast_tasks_{p.rstrip('_')}")
    occ_curve = fcm.disruption_curve(fc, "onet_soc_code").merge(occ[["onet_soc_code", "title"]], on="onet_soc_code", how="left"); io.save(occ_curve, f"forecast_occupations_{p.rstrip('_')}")
    ns = nat_shift.set_index("nat_code").adoption_shift_years
    curve = national_curve(fc, xw, ns, params_c, now_y); io.save(curve, f"forecast_{p}national")
    fr_nat = push_frontier(occf, xw); io.save(fr_nat, f"{p}frontier_national")
    t = weighted_table(natw, fr_nat, curve, spec); io.save(t, f"{p}weighted")
    econ = economy(t, spec); io.save(econ, f"{p}economy_exposure")
    # cross runs: this country's weights on the others' timing and theirs on this one
    def own_weights_on(fc_other):
        return economy(weighted_table(natw, fr_nat, national_curve(fc_other, xw, None, params_c, now_y), spec), spec)
    f2s = fc.merge(o2s[["onet_soc_code", "ssoc_code", "ssoc_title", "weight_rev"]], on="onet_soc_code"); f2s["w"] = f2s.task_weight * f2s.weight_rev
    e_sgw_ct = se.economy_exposure(se.ssoc_weighted_table(frontier_ssoc, fcm.disruption_curve(f2s, "ssoc_code", weight="w"), ssoc_w))
    e_usw_ct = ue.economy_exposure(ue.onet_weighted_table(occf, fcm.disruption_curve(fc, "onet_soc_code"), us_w), "USA")
    decomps = {}
    for w in ["wage_value", "employment"]:
        decomps[f"SGP_{spec.code}_{w}"] = pair_decomposition(econ_sg, econ, own_weights_on(fc_sg), e_sgw_ct, w, "SGP", spec.code)
        decomps[f"USA_{spec.code}_{w}"] = pair_decomposition(econ_us, econ, own_weights_on(fc_us), e_usw_ct, w, "USA", spec.code)
    for oc, octx in (others or {}).items():
        for w in ["wage_value", "employment"]:
            e_ow_ct = economy(weighted_table(octx["natw"], octx["fr_nat"], national_curve(fc, octx["xw"], None, params_c, now_y), octx["spec"]), octx["spec"])
            decomps[f"{oc}_{spec.code}_{w}"] = pair_decomposition(octx["econ"], econ, own_weights_on(octx["fc"]), e_ow_ct, w, oc, spec.code)
    # 5. rung five, 6. skill levels, the sector lens
    try:
        change = spec.outcomes(); io.save(change, f"{p}outcomes")
        out_stats = outcome_tests(change, t, spec)
    except FileNotFoundError as e:
        out_stats = {"error": str(e)}
    bands, band_stats = skill_level_bands(t, spec); io.save(bands, f"{p}skill_level_bands")
    lv = ue.skill_level_exposure(onet_w, io.load("onet_descriptor_ratings_v31"), occf, occ_curve)
    io.save(lv["bands"], f"{p}onet_skill_level_exposure"); io.save(lv["correlations"], f"{p}onet_skill_level_correlations")
    iv = industry_view(mix, ind, points, t, spec); io.save(iv, f"{p}industry_view")
    rating_stats = {}
    if spec.ratings is not None and io.exists(f"{p}ratings"):
        rl = rating_level_exposure(t, io.load(f"{p}ratings"), spec)
        io.save(rl["bands"], f"{p}rating_level_exposure"); io.save(rl["correlations"], f"{p}rating_level_correlations")
        ix = rl["correlations"][rl["correlations"].family == "index"]
        rating_stats = {"index_rho": float(ix.rho_level_vs_done_2035.iloc[0]) if len(ix) and pd.notna(ix.rho_level_vs_done_2035.iloc[0]) else None, "skills": int((rl["correlations"].family != "index").sum())}
    route_stats = {}
    if spec.code == "GBR" and io.exists("uk_ifate_standards"):
        rv = route_view(io.load("uk_ifate_standards"), io.load("uk_ifate_standard_soc"), t); io.save(rv, "uk_ifate_route_view")
        route_stats = {"routes": int(len(rv)), "earliest": rv.iloc[0].route if len(rv) else None, "latest": rv.iloc[-1].route if len(rv) else None}
    e = econ[econ.scope == "economy"].set_index("weighting")
    stats = {"name": spec.name, "classification": spec.classification, "lead": lead, "geo_lead_years": lead["lead_years"], "industry_shifts": {r.industry: round(float(r.anchor_shift_years), 2) for r in ind.itertuples()},
             "shift_range": [float(nat_shift.adoption_shift_years.min()), float(nat_shift.adoption_shift_years.max())] if len(nat_shift) else [0.0, 0.0],
             "weights": {"employment_thousands_total": float(natw.employment_thousands.sum()), "wage_value_total": float(natw.wage_value.sum()), "codes": int(len(natw)), "wage_sources": natw.wage_source.value_counts(dropna=False).to_dict(), "edition": int(natw.edition.iloc[0]) if "edition" in natw else None},
             "economy": {w: {"observed_automation": float(e.loc[w, "observed_automation"]), "share_weight_observed": float(e.loc[w, "share_weight_observed"]), "done_by_2035": float(e.loc[w, "done_by_2035"]), "share_by_2035": float(e.loc[w, "share_by_2035"]), "quarter_done_year_central": float(e.loc[w, "quarter_done_year_central_from_curve"]) if pd.notna(e.loc[w, "quarter_done_year_central_from_curve"]) else None} for w in ["employment", "wage_value"] if w in e.index},
             "outcomes": out_stats, "skill_levels_national": band_stats, "skill_levels_onet": {"index_rho": float(lv["correlations"].set_index("element_name").loc["All essential skills (average level)", "rho_level_vs_done_2035"]) if "All essential skills (average level)" in set(lv["correlations"].element_name) else None},
             "apprenticeship_routes": route_stats, "national_ratings": rating_stats, "framework_name": spec.framework_name, "decompositions": decomps}
    ctx = {"spec": spec, "natw": natw, "xw": xw, "fr_nat": fr_nat, "econ": econ, "fc": fc, "params": params_c, "lead": lead}
    return stats, ctx


def saved_ctx(spec: CountrySpec, cstats: dict | None) -> dict | None:
    """The context a later country needs from an earlier one (weights, crosswalk, frontier, economy, task
    forecast, lead), read back from the saved tables so a partial re-run can skip countries already built."""
    p = spec.prefix
    needed = [f"{p}weights", f"{p}onet_crosswalk", f"{p}frontier_national", f"{p}economy_exposure", f"forecast_tasks_{p.rstrip('_')}"]
    if not all(io.exists(n) for n in needed) or not cstats or "lead" not in cstats:
        return None
    return {"spec": spec, "natw": io.load(f"{p}weights"), "xw": io.load(f"{p}onet_crosswalk"), "fr_nat": io.load(f"{p}frontier_national"), "econ": io.load(f"{p}economy_exposure"),
            "fc": io.load(f"forecast_tasks_{p.rstrip('_')}"), "params": None, "lead": cstats["lead"]}


def route_view(standards: pd.DataFrame, std_soc: pd.DataFrame, t: pd.DataFrame) -> pd.DataFrame:
    """The 15 apprenticeship routes as a sector lens: each route's standards, the SOC codes they lead to,
    and the exposure of those codes weighted by employment times the standard's share in the code."""
    m = std_soc.merge(standards[["reference", "route", "title", "level"]], on="reference", how="left").merge(t[["nat_code", "employment_thousands", "observed_automation", "share_weight_observed", "share_weight_physical", "done_by_2030", "done_by_2035", "done_by_2035_p10", "done_by_2035_p90", "share_by_2035", "ceiling_ai_share", "quarter_done_year_p10", "quarter_done_year_p50", "quarter_done_year_p90"]], left_on="soc_code", right_on="nat_code", how="left")
    m["w"] = m.share * m.employment_thousands.fillna(0.0)
    rows = []
    for route, g in m.groupby("route"):
        row = {"route": route, "n_standards": int(g.reference.nunique()), "n_soc_codes": int(g.soc_code.nunique()), "employment_thousands": float(g.drop_duplicates("soc_code").employment_thousands.sum()), "levels": ", ".join(str(int(x)) for x in sorted(g.level.dropna().unique()))}
        for c in ["observed_automation", "share_weight_observed", "share_weight_physical", "done_by_2030", "done_by_2035", "done_by_2035_p10", "done_by_2035_p90", "share_by_2035", "ceiling_ai_share", "quarter_done_year_p10", "quarter_done_year_p50", "quarter_done_year_p90"]:
            row[c] = se._wavg(g, c, "w")
        rows.append(row)
    return pd.DataFrame(rows).sort_values("quarter_done_year_p50").reset_index(drop=True)


def pair_decomposition(e_a: pd.DataFrame, e_b: pd.DataFrame, e_aw_bt: pd.DataFrame, e_bw_at: pd.DataFrame, weighting: str, a: str, b: str) -> dict:
    """Split the A-minus-B gap in 'AI doing by year' into a job-mix part and a timing part (the mean of
    the two orderings, which add up exactly), for the central, fast and slow cases."""
    def row(e):
        return e[(e.scope == "economy") & (e.weighting == weighting)].iloc[0]
    ra, rb, x, y = row(e_a), row(e_b), row(e_aw_bt), row(e_bw_at)
    out = {"weighting": weighting, "a": a, "b": b, "by_year": {}}
    for q, suf in [("central", ""), ("fast", "_p10"), ("slow", "_p90")]:
        for yv in (2030, 2035, 2040):
            c = f"done_by_{yv}{suf}"
            gap = float(ra[c] - rb[c])
            jobmix = 0.5 * (float(x[c] - rb[c]) + float(ra[c] - y[c]))
            out["by_year"][f"{q}_{yv}"] = {a: float(ra[c]), b: float(rb[c]), "gap": gap, "job_mix": jobmix, "timing": gap - jobmix, "a_weights_b_timing": float(x[c]), "b_weights_a_timing": float(y[c])}
    for q in ("central", "fast", "slow"):
        c = f"quarter_done_year_{q}_from_curve"
        if c in ra and c in rb:
            out[f"quarter_done_year_{q}"] = {a: float(ra[c]) if pd.notna(ra[c]) else None, b: float(rb[c]) if pd.notna(rb[c]) else None}
    return out
