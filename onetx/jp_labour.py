"""Japan (medium depth): the Basic Survey on Wage Structure by occupation, the communications survey's IoT/AI
question, and a hand-made link from Japan's occupation classes to ISCO-08.

Inputs (`data/raw/estat/`, downloaded from e-Stat by file id, `fetch.fetch_estat`):
  wage2024_t1_occupation_small.xlsx          Basic Survey on Wage Structure 2024, table 1 by occupation (small
                                             classification, 144 classes): age, tenure, hours, monthly contractual
                                             cash earnings, scheduled earnings, bonus and number of workers (tens),
                                             regular workers in enterprises with 10 or more employees
  wage2024_t20_occupation_major_industry.xlsx table 20: the same by occupation major group and industry (JSIC
                                             divisions C to R), the occupation-by-industry mix
  mic_iot_ai_adoption_<year>.csv             MIC Communications Usage Trend Survey, enterprise edition 2018-2025:
                                             share of enterprises (100 or more employees) that have introduced IoT or
                                             AI systems or services, by industry grouping

What is coarse (the page says so): the occupation classes are survey classes rather than a census of all workers
(about 144 classes covering regular employees in enterprises with 10 or more employees), the firm survey asks about
IoT and AI together, pay is monthly scheduled earnings before bonuses, and the link to ISCO-08 is a table written by
hand for this map (`JSOC_TO_ISCO`), one or several ISCO unit groups per class with equal weight. Japan publishes no
official correspondence from its occupation classification to ISCO-08 and no task list per occupation.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import RAW

ESTAT = RAW / "estat"
SURVEY_DATE_IN_YEAR = 0.6   # the communications survey is conducted at the end of August

MAJOR_GROUP_NAMES = {"A": "Administrative and managerial workers", "B": "Professional and engineering workers", "C": "Clerical workers", "D": "Sales workers", "E": "Service workers",
                     "F": "Security workers", "G": "Agriculture, forestry and fishery workers", "H": "Manufacturing process workers", "I": "Transport and machine operation workers",
                     "J": "Construction and mining workers", "K": "Carrying, cleaning, packaging and related workers"}
SKILL_LEVEL_NAMES = {1: "simple physical or manual tasks (primary education)", 2: "operating machinery, clerical, sales and service work (secondary education)",
                     3: "complex technical and practical tasks (post-secondary education)", 4: "complex problem-solving, decision-making and creative work (degree level)"}
# the communications survey's industry groupings, keyed to JSIC divisions (the wage survey's industry letters)
INDUSTRIES = {"D": "Construction", "E": "Manufacturing", "H": "Transport and postal services", "I": "Wholesale and retail trade", "J": "Finance and insurance", "K": "Real estate",
              "G": "Information and communications", "S": "Services and other industries"}
_MIC_TO_KEY = {"建設業": "D", "製造業": "E", "運輸業・郵便業": "H", "運輸業,郵便業": "H", "卸売・小売業": "I", "卸売業,小売業": "I", "金融・保険業": "J", "金融業,保険業": "J", "不動産業": "K", "情報通信業": "G"}
_JSIC_TO_KEY = {"C": "S", "D": "D", "E": "E", "F": "S", "G": "G", "H": "H", "I": "I", "J": "J", "K": "K", "L": "S", "M": "S", "N": "S", "O": "S", "P": "S", "Q": "S", "R": "S"}

# (Japanese class, English title, major group, ISCO-08 unit groups it maps to)
JSOC_TO_ISCO = [
    ("管理的職業従事者", "Administrative and managerial workers", "A", ["1120", "1211", "1212", "1213", "1219", "1221", "1222", "1223", "1321", "1323", "1324", "1330", "1341", "1342", "1343", "1344", "1345", "1346", "1349", "1411", "1412", "1420", "1431", "1439"]),
    ("研究者", "Researchers", "B", ["2111", "2112", "2113", "2114", "2131", "2132", "2133", "2631", "2632", "2633", "2634"]),
    ("電気・電子・電気通信技術者（通信ネットワーク技術者を除く）", "Electrical, electronic and telecommunications engineers", "B", ["2151", "2152", "2153"]),
    ("機械技術者", "Mechanical engineers", "B", ["2144"]),
    ("輸送用機器技術者", "Transport equipment engineers", "B", ["2144"]),
    ("金属技術者", "Metallurgical engineers", "B", ["2146"]),
    ("化学技術者", "Chemical engineers", "B", ["2145"]),
    ("建築技術者", "Architects and building engineers", "B", ["2161", "2142"]),
    ("土木技術者", "Civil engineers", "B", ["2142"]),
    ("測量技術者", "Surveying engineers", "B", ["2165"]),
    ("システムコンサルタント・設計者", "Systems consultants and designers", "B", ["2511"]),
    ("ソフトウェア作成者", "Software developers", "B", ["2512", "2514"]),
    ("その他の情報処理・通信技術者", "Other information processing and communications engineers", "B", ["2513", "2519", "2521", "2522", "2523", "2529"]),
    ("他に分類されない技術者", "Engineers not elsewhere classified", "B", ["2141", "2143", "2149"]),
    ("医師", "Medical doctors", "B", ["2211", "2212"]),
    ("歯科医師", "Dentists", "B", ["2261"]),
    ("獣医師", "Veterinarians", "B", ["2250"]),
    ("薬剤師", "Pharmacists", "B", ["2262"]),
    ("保健師", "Public health nurses", "B", ["2221"]),
    ("助産師", "Midwives", "B", ["2222"]),
    ("看護師", "Nurses", "B", ["2221"]),
    ("准看護師", "Assistant nurses", "B", ["3221"]),
    ("診療放射線技師", "Radiological technologists", "B", ["3211"]),
    ("臨床検査技師", "Clinical laboratory technologists", "B", ["3212"]),
    ("理学療法士，作業療法士，言語聴覚士，視能訓練士", "Physiotherapists, occupational therapists, speech therapists and orthoptists", "B", ["2264", "2269"]),
    ("歯科衛生士", "Dental hygienists", "B", ["3251"]),
    ("歯科技工士", "Dental technicians", "B", ["3214"]),
    ("栄養士", "Dietitians", "B", ["2265"]),
    ("その他の保健医療従事者", "Other health professionals", "B", ["2269", "3259"]),
    ("保育士", "Childcare workers (nursery teachers)", "B", ["5311", "2342"]),
    ("介護支援専門員（ケアマネージャー）", "Care managers", "B", ["3412"]),
    ("その他の社会福祉専門職業従事者", "Other social welfare professionals", "B", ["2635", "3412"]),
    ("法務従事者", "Legal professionals", "B", ["2611", "2612", "2619"]),
    ("公認会計士，税理士", "Certified public accountants and tax accountants", "B", ["2411"]),
    ("その他の経営・金融・保険専門職業従事者", "Other business, finance and insurance professionals", "B", ["2412", "2413", "2421", "2423", "2424"]),
    ("幼稚園教員，保育教諭", "Kindergarten teachers", "B", ["2342"]),
    ("小・中学校教員", "Primary and lower secondary school teachers", "B", ["2341", "2330"]),
    ("高等学校教員", "Upper secondary school teachers", "B", ["2330"]),
    ("大学教授（高専含む）", "University professors", "B", ["2310"]),
    ("大学准教授（高専含む）", "University associate professors", "B", ["2310"]),
    ("大学講師・助教（高専含む）", "University lecturers and assistant professors", "B", ["2310"]),
    ("その他の教員", "Other teachers", "B", ["2320", "2351", "2352", "2353", "2359"]),
    ("宗教家", "Religious professionals", "B", ["2636"]),
    ("著述家，記者，編集者", "Authors, journalists and editors", "B", ["2641", "2642"]),
    ("美術家，写真家，映像撮影者", "Artists, photographers and camera operators", "B", ["2651", "3431", "3521"]),
    ("デザイナー", "Designers", "B", ["2163", "2166"]),
    ("音楽家，舞台芸術家", "Musicians and performing artists", "B", ["2652", "2653", "2655"]),
    ("個人教師", "Private tutors and instructors", "B", ["2355", "2353", "3422"]),
    ("他に分類されない専門的職業従事者", "Professional workers not elsewhere classified", "B", ["2422", "2431", "2432", "2434", "2643", "3433", "3435"]),
    ("庶務・人事事務員", "General affairs and personnel clerks", "C", ["4110", "4416", "3343"]),
    ("企画事務員", "Planning clerks", "C", ["2422", "3343", "4110"]),
    ("受付・案内事務員", "Receptionists and information clerks", "C", ["4226", "4224", "4225"]),
    ("秘書", "Secretaries", "C", ["3343", "4120"]),
    ("電話応接事務員", "Telephone and contact centre clerks", "C", ["4222", "4223"]),
    ("総合事務員", "General clerks", "C", ["4110", "4419"]),
    ("その他の一般事務従事者", "Other general clerical workers", "C", ["4110", "4412", "4415", "4419"]),
    ("会計事務従事者", "Accounting clerks", "C", ["4311", "4313", "3313"]),
    ("生産関連事務従事者", "Production-related clerks", "C", ["4322", "4321"]),
    ("営業・販売事務従事者", "Sales and marketing clerks", "C", ["4311", "3339", "4110"]),
    ("外勤事務従事者", "Field clerical workers (collectors and the like)", "C", ["4214", "4227"]),
    ("運輸・郵便事務従事者", "Transport and postal clerks", "C", ["4323", "4412"]),
    ("事務用機器操作員", "Office machine operators", "C", ["4131", "4132"]),
    ("販売店員", "Shop sales assistants", "D", ["5223"]),
    ("その他の商品販売従事者", "Other goods sales workers", "D", ["5211", "5212", "5221", "5222", "5230", "5242", "5243"]),
    ("販売類似職業従事者", "Sales-related workers (brokers, buyers, agents)", "D", ["3324", "3323", "3339", "3334"]),
    ("自動車営業職業従事者", "Motor vehicle sales representatives", "D", ["5223", "3322"]),
    ("機械器具・通信・システム営業職業従事者（自動車を除く）", "Machinery, communications and systems sales representatives", "D", ["2433", "2434", "3322"]),
    ("金融営業職業従事者", "Financial sales representatives", "D", ["3312", "3311"]),
    ("保険営業職業従事者", "Insurance sales representatives", "D", ["3321"]),
    ("その他の営業職業従事者", "Other sales representatives", "D", ["3322", "2431"]),
    ("介護職員（医療・福祉施設等）", "Care workers in medical and welfare facilities", "E", ["5321", "5329"]),
    ("訪問介護従事者", "Home care workers", "E", ["5322"]),
    ("看護助手", "Nursing assistants", "E", ["5321"]),
    ("その他の保健医療サービス職業従事者", "Other health service workers", "E", ["5329", "5321"]),
    ("理容・美容師", "Barbers and hairdressers", "E", ["5141"]),
    ("美容サービス・浴場従事者（美容師を除く）", "Beauty service and bathhouse workers", "E", ["5142"]),
    ("クリーニング職，洗張職", "Laundry workers", "E", ["9121", "8157"]),
    ("飲食物調理従事者", "Cooks", "E", ["5120", "3434"]),
    ("飲食物給仕従事者", "Waiters and food service attendants", "E", ["5131", "5132", "5246"]),
    ("航空機客室乗務員", "Flight attendants", "E", ["5111"]),
    ("身の回り世話従事者", "Personal care and attendance workers", "E", ["5322", "5162", "5169"]),
    ("娯楽場等接客員", "Attendants in amusement and leisure facilities", "E", ["5169", "4212", "9629"]),
    ("居住施設・ビル等管理人", "Building and residential facility caretakers", "E", ["5153"]),
    ("その他のサービス職業従事者", "Other service workers", "E", ["5169", "5151", "5152", "5163", "5164", "5165"]),
    ("警備員", "Security guards", "F", ["5414"]),
    ("その他の保安職業従事者", "Other security workers", "F", ["5411", "5412", "5413", "5419"]),
    ("農林漁業従事者", "Agriculture, forestry and fishery workers", "G", ["6111", "6112", "6113", "6121", "6210", "6221", "6222", "6223", "9211", "9212", "9213", "9215"]),
    ("製銑・製鋼・非鉄金属製錬従事者", "Iron, steel and non-ferrous metal smelting workers", "H", ["8121", "3135"]),
    ("鋳物製造・鍛造従事者", "Casting and forging workers", "H", ["7211", "7221", "7223"]),
    ("金属工作機械作業従事者", "Metal machine tool operators", "H", ["7223", "8122"]),
    ("金属プレス従事者", "Metal press workers", "H", ["7223", "8122"]),
    ("鉄工，製缶従事者", "Structural metal and boiler workers", "H", ["7214", "7213"]),
    ("板金従事者", "Sheet metal workers", "H", ["7213"]),
    ("金属彫刻・表面処理従事者", "Metal engraving and surface treatment workers", "H", ["7224", "8122", "7313"]),
    ("金属溶接・溶断従事者", "Welders and flame cutters", "H", ["7212"]),
    ("その他の製品製造・加工処理従事者（金属製品）", "Other metal product manufacturing workers", "H", ["7222", "7233", "8211", "8122"]),
    ("化学製品製造従事者", "Chemical product manufacturing workers", "H", ["8131", "8132", "3133"]),
    ("窯業・土石製品製造従事者", "Ceramic, stone and clay product workers", "H", ["8114", "7314", "7315"]),
    ("食料品・飲料・たばこ製造従事者", "Food, beverage and tobacco manufacturing workers", "H", ["8160", "7511", "7512", "7513", "7514", "7515", "7516"]),
    ("紡織・衣服・繊維製品製造従事者", "Textile, clothing and fibre product workers", "H", ["8151", "8152", "8153", "8154", "8155", "8156", "8159", "7531", "7532", "7533"]),
    ("木・紙製品製造従事者", "Wood and paper product workers", "H", ["7522", "7523", "8141", "8143", "8172"]),
    ("印刷・製本従事者", "Printing and bookbinding workers", "H", ["7321", "7322", "7323"]),
    ("ゴム・プラスチック製品製造従事者", "Rubber and plastic product workers", "H", ["8141", "8142"]),
    ("その他の製品製造・加工処理従事者（金属製品を除く）", "Other product manufacturing workers (non-metal)", "H", ["8189", "7549", "8183", "7319"]),
    ("はん用・生産用・業務用機械器具組立従事者", "General, production and business machinery assemblers", "H", ["8211"]),
    ("電気機械器具組立従事者", "Electrical machinery assemblers", "H", ["8212"]),
    ("自動車組立従事者", "Motor vehicle assemblers", "H", ["8211"]),
    ("その他の機械組立従事者", "Other machinery assemblers", "H", ["8219", "8211"]),
    ("はん用・生産用・業務用機械器具・電気機械器具整備・修理従事者", "Machinery and electrical equipment maintenance and repair workers", "H", ["7233", "7412", "7421"]),
    ("自動車整備・修理従事者", "Motor vehicle mechanics and repairers", "H", ["7231"]),
    ("その他の機械整備・修理従事者", "Other machinery maintenance and repair workers", "H", ["7232", "7233", "7234"]),
    ("製品検査従事者（金属製品）", "Product inspectors (metal products)", "H", ["7543", "3115"]),
    ("製品検査従事者（金属製品を除く）", "Product inspectors (non-metal products)", "H", ["7543", "3122"]),
    ("機械検査従事者", "Machinery inspectors", "H", ["3115", "7543"]),
    ("画工，塗装・看板制作従事者", "Painters, coaters and sign makers", "H", ["7132", "7131", "7316"]),
    ("製図その他生産関連・生産類似作業従事者", "Draughtspersons and other production-related workers", "H", ["3118", "3119", "7549"]),
    ("鉄道運転従事者", "Railway drivers", "I", ["8311"]),
    ("バス運転者", "Bus drivers", "I", ["8331"]),
    ("タクシー運転者", "Taxi drivers", "I", ["8322"]),
    ("乗用自動車運転者（タクシー運転者を除く）", "Car drivers (other than taxi)", "I", ["8322"]),
    ("営業用大型貨物自動車運転者", "Heavy truck drivers (commercial)", "I", ["8332"]),
    ("営業用貨物自動車運転者（大型車を除く）", "Truck drivers (commercial, other than heavy)", "I", ["8332"]),
    ("自家用貨物自動車運転者", "Truck drivers (own-account)", "I", ["8332"]),
    ("その他の自動車運転従事者", "Other motor vehicle drivers", "I", ["8321", "8322", "8332"]),
    ("航空機操縦士", "Aircraft pilots", "I", ["3153"]),
    ("車掌", "Train conductors", "I", ["5112"]),
    ("他に分類されない輸送従事者", "Transport workers not elsewhere classified", "I", ["8312", "8350", "3152", "3154"]),
    ("発電員，変電員", "Power generation and substation operators", "I", ["3131", "8182"]),
    ("クレーン・ウインチ運転従事者", "Crane and winch operators", "I", ["8343"]),
    ("建設・さく井機械運転従事者", "Construction and drilling machinery operators", "I", ["8342", "8113"]),
    ("その他の定置・建設機械運転従事者", "Other stationary and construction machinery operators", "I", ["8341", "8344", "8181", "8189"]),
    ("建設躯体工事従事者", "Building frame construction workers", "J", ["7114", "7115", "7121"]),
    ("大工", "Carpenters", "J", ["7115"]),
    ("配管従事者", "Plumbers and pipe fitters", "J", ["7126"]),
    ("その他の建設従事者", "Other construction workers", "J", ["7112", "7113", "7119", "7122", "7123", "7124", "7125", "7127"]),
    ("電気工事従事者", "Electrical installation workers", "J", ["7411", "7413"]),
    ("土木従事者，鉄道線路工事従事者", "Civil engineering and railway track workers", "J", ["9312", "9313", "7119"]),
    ("ダム・トンネル掘削従事者，採掘従事者", "Tunnelling and mining workers", "J", ["8111", "7111", "9311"]),
    ("船内・沿岸荷役従事者", "Dock and shipboard cargo handlers", "K", ["9333"]),
    ("その他の運搬従事者", "Other carrying and freight handling workers", "K", ["9333", "9329", "9331", "9332", "8344"]),
    ("ビル・建物清掃員", "Building cleaners", "K", ["9112"]),
    ("清掃員（ビル・建物を除く），廃棄物処理従事者", "Other cleaners and refuse workers", "K", ["9611", "9612", "9613", "9129"]),
    ("包装従事者", "Packers", "K", ["9321"]),
    ("他に分類されない運搬・清掃・包装等従事者", "Carrying, cleaning and packaging workers not elsewhere classified", "K", ["9329", "9622", "9629", "9621"]),
]


def _norm(s: str) -> str:
    return re.sub(r"\s", "", str(s).replace("\n", "")).replace("男女計", "").replace("男", "").replace("女", "")


def _wage_years() -> list:
    """The editions of the wage survey's table 1 on disk, oldest to newest."""
    return sorted({int(m.group(1)) for p in ESTAT.glob("wage*_t1_occupation_small.xlsx") for m in [re.search(r"wage(\d{4})_", p.name)] if m})


WAGE_YEAR = (_wage_years() or [2024])[-1]   # the latest edition is the weights year; the one before it gives the outcome change


def load_wage_table(year: int = WAGE_YEAR) -> pd.DataFrame:
    """Table 1 (both sexes block): jp_code (two-digit major letter + running number), class (Japanese), title,
    major_group, workers (persons), monthly scheduled earnings (yen), monthly contractual cash earnings (yen).
    Rows are found by class name, so the one-row shift between the 2024 and 2025 layouts does not matter."""
    x = pd.ExcelFile(ESTAT / f"wage{year}_t1_occupation_small.xlsx")
    df = x.parse(x.sheet_names[0], header=None)
    key = {_norm(j): (j, e, m, iscos) for j, e, m, iscos in JSOC_TO_ISCO}
    rows, seen = [], set()
    for i in range(9, len(df)):
        raw = df.iloc[i, 1]
        if not isinstance(raw, str):
            continue
        name = _norm(raw)
        if name in seen or name not in key:
            continue
        seen.add(name)
        if name == "不詳":
            continue
        j, e, m, iscos = key[name]
        workers = pd.to_numeric(df.iloc[i, 10], errors="coerce")
        pay = pd.to_numeric(df.iloc[i, 8], errors="coerce")
        cash = pd.to_numeric(df.iloc[i, 7], errors="coerce")
        rows.append({"jp_class": j, "jp_title": e, "major_group": m, "workers": float(workers) * 10.0 if pd.notna(workers) else np.nan, "monthly_scheduled_yen": float(pay) * 1000.0 if pd.notna(pay) else np.nan, "monthly_cash_yen": float(cash) * 1000.0 if pd.notna(cash) else np.nan})
        if len(seen) >= len(JSOC_TO_ISCO):
            break
    out = pd.DataFrame(rows)
    out["jp_code"] = [f"{m}{n:03d}" for n, m in enumerate(out.major_group, start=1)]
    return out


def build_weights(year: int = WAGE_YEAR) -> pd.DataFrame:
    w = load_wage_table(year)
    df = pd.DataFrame({"jp_code": w.jp_code, "jp_title": w.jp_title, "employment_thousands": w.workers / 1000.0, "wage_monthly": w.monthly_scheduled_yen, "major_group": w.major_group})
    df["submajor_label"] = df.major_group
    df["submajor_name"] = df.major_group.map(MAJOR_GROUP_NAMES)
    iscos = {j: iscos for j, e, m, iscos in JSOC_TO_ISCO}
    from .no_labour import _skill_level
    df["skill_level"] = [float(np.mean([_skill_level(c) for c in iscos[j]])) for j in w.jp_class]
    df["wage_source"] = f"wage_structure_survey_{year}_scheduled_monthly"
    df["wage_value"] = df.employment_thousands * df.wage_monthly
    df["employment_method"] = f"wage_structure_survey_{year}_workers"
    df["edition"] = year
    return df[(df.employment_thousands > 0) & df.wage_monthly.notna()].reset_index(drop=True)


def load_isco_link(unit_codes: set[str] | None = None) -> pd.DataFrame:
    w = load_wage_table(); code_of = dict(zip(w.jp_class, w.jp_code))
    rows = []
    for j, e, m, iscos in JSOC_TO_ISCO:
        c = code_of.get(j)
        if c is None or (unit_codes and c not in unit_codes):
            continue
        rows += [{"isco_code": i, "nat_code": c, "weight": 1.0} for i in iscos]
    k = pd.DataFrame(rows)
    k["weight"] = k.weight / k.groupby("isco_code").weight.transform("sum")   # an ISCO group named by several classes is shared equally
    return k.reset_index(drop=True)


def load_ai_points() -> pd.DataFrame:
    """Share of enterprises that have introduced IoT or AI systems or services, nationally and by industry grouping, 2018 to 2025."""
    rows = []
    for f in sorted(ESTAT.glob("mic_iot_ai_adoption_*.csv")):
        year = int(re.search(r"(\d{4})", f.name).group(1))
        txt = f.read_bytes().decode("cp932", "replace")
        for line in txt.splitlines():
            cells = line.split(",")
            if len(cells) < 5 or cells[0] not in ("全体", "産業分類"):
                continue
            try:
                pct = float(cells[4])
            except ValueError:
                continue
            label = cells[1].replace("　", "").strip()
            if cells[0] == "全体":
                rows.append({"date": year + SURVEY_DATE_IN_YEAR, "period": str(year), "pct": pct, "group_type": "all", "industry": None, "label": "Enterprises with 100 or more employees (IoT or AI introduced)"})
            else:
                key = _MIC_TO_KEY.get(label, "S" if label.startswith("サービス") else None)
                if key:
                    rows.append({"date": year + SURVEY_DATE_IN_YEAR, "period": str(year), "pct": pct, "group_type": "industry", "industry": key, "label": INDUSTRIES[key]})
    return pd.DataFrame(rows).sort_values(["group_type", "industry", "date"]).reset_index(drop=True)


def load_industry_mix(year: int = WAGE_YEAR) -> pd.DataFrame:
    """Workers by occupation major group and the survey's industry groupings (table 20, both sexes, all sizes)."""
    x = pd.ExcelFile(ESTAT / f"wage{year}_t20_occupation_major_industry.xlsx")
    df = x.parse("男女計", header=None)
    majors = {_norm(k): v for k, v in {"管理的職業従事者": "A", "専門的・技術的職業従事者": "B", "専門的・技術的従事者": "B", "事務従事者": "C", "販売従事者": "D", "サービス職業従事者": "E", "保安職業従事者": "F", "農林漁業従事者": "G", "生産工程従事者": "H", "輸送・機械運転従事者": "I", "建設・採掘従事者": "J", "運搬・清掃・包装等従事者": "K"}.items()}
    rows, cur = [], None
    for i in range(10, len(df)):
        raw = df.iloc[i, 1]
        if not isinstance(raw, str):
            continue
        name = re.sub(r"\s", "", raw)
        m = re.match(r"^(.+?)Ｔ産業計$", name)
        if m:
            cur = majors.get(_norm(m.group(1)))
            continue
        m2 = re.match(r"^([Ｃ-Ｒ])(.+)$", name)
        if m2 and cur:
            letter = chr(ord(m2.group(1)) - ord("Ａ") + ord("A"))
            workers = pd.to_numeric(df.iloc[i, 10], errors="coerce")
            if pd.notna(workers):
                rows.append({"major_group": cur, "industry": _JSIC_TO_KEY.get(letter, "S"), "employed": float(workers) * 10.0})
    return pd.DataFrame(rows).groupby(["major_group", "industry"]).employed.sum().reset_index()


def employment_change(year_to: int = WAGE_YEAR) -> pd.DataFrame:
    """Rung five for Japan: the change in workers represented and in monthly scheduled earnings per occupation
    class between the two latest editions of the wage survey (June of each year). With one edition on disk the
    table keeps its shape with missing values, so the page says the test is not available yet."""
    years = _wage_years()
    y1 = year_to; y0 = max([y for y in years if y < y1], default=None)
    w1 = load_wage_table(y1)
    if y0 is None:
        return pd.DataFrame({"jp_code": w1.jp_code, f"emp_{y1 - 1}": np.nan, f"emp_{y1}": np.nan, "emp_change_pct": np.nan, f"wage_{y1 - 1}": np.nan, f"wage_{y1}": np.nan, "wage_change_pct": np.nan, "year_from": y1 - 1, "year_to": y1, "wage_years": "none"})
    w0 = load_wage_table(y0).set_index("jp_class")
    df = pd.DataFrame({"jp_code": w1.jp_code.values, "jp_class": w1.jp_class.values})
    df[f"emp_{y0}"] = df.jp_class.map(w0.workers).values
    df[f"emp_{y1}"] = w1.workers.values
    df["emp_change_pct"] = 100.0 * (df[f"emp_{y1}"] / df[f"emp_{y0}"].where(df[f"emp_{y0}"] > 0) - 1.0)
    df[f"wage_{y0}"] = df.jp_class.map(w0.monthly_scheduled_yen).values
    df[f"wage_{y1}"] = w1.monthly_scheduled_yen.values
    df["wage_change_pct"] = 100.0 * (df[f"wage_{y1}"] / df[f"wage_{y0}"].where(df[f"wage_{y0}"] > 0) - 1.0)
    df["year_from"], df["year_to"], df["wage_years"] = y0, y1, f"{y0}-{y1}"
    return df.drop(columns=["jp_class"])


def outcome_units() -> pd.DataFrame:
    w = build_weights(); return pd.DataFrame({"jp_code": w.jp_code.values, "emp_unit": "people", "wage_unit": "month"})


def load_titles() -> pd.DataFrame:
    w = load_wage_table()
    return pd.concat([pd.DataFrame({"jp_code": w.jp_code, "title": w.jp_title, "category": "principal"}), pd.DataFrame({"jp_code": w.jp_code, "title": w.jp_class, "category": "japanese"})], ignore_index=True)


def no_tasks() -> pd.DataFrame:
    return pd.DataFrame(columns=["jp_code", "task_seq", "task"])
