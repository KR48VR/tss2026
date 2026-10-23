"""Text-based crosswalks between Singapore's frameworks and O*NET.

Where no official code-to-code table exists, we match on meaning:

* Skills Framework key tasks  -> O*NET tasks (top-k) and O*NET detailed work activities (top-1)
* Skills Framework TSCs/CCS   -> O*NET intermediate work activities, and O*NET skill/knowledge/ability
                                 descriptors (30.1 classic and 31.0 essential/transferable skills)
* SSOC unit-group task bullets-> O*NET tasks
* Skills Framework roles that SingStat has not coded -> SSOC occupations (by description)

Every match carries a similarity score and, where an occupation crosswalk exists, a flag saying
whether the matched O*NET item belongs to an occupation the role/SSOC code maps to. Downstream code
can threshold on either. The occupation-neighbourhood bonus pulls matches toward the right field
without forbidding cross-field matches (many Singapore roles have no exact US counterpart).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import crosswalks as xw
from . import onet
from . import singapore as sg
from .embed import HybridMatcher


def _rerank_with_bonus(idx, sc, allowed_sets, cand_occ, bonus=0.10, k_out=3):
    """Given global top-K candidates, add `bonus` to those whose occupation is in the query's allowed
    set and keep the best k_out."""
    out_idx = np.zeros((idx.shape[0], k_out), dtype=np.int64)
    out_sc = np.zeros((idx.shape[0], k_out), dtype=np.float32)
    out_in = np.zeros((idx.shape[0], k_out), dtype=bool)
    for i in range(idx.shape[0]):
        allowed = allowed_sets[i]
        s = sc[i].astype(np.float32).copy()
        inset = np.array([cand_occ[j] in allowed for j in idx[i]]) if allowed else np.zeros(len(idx[i]), dtype=bool)
        s2 = s + bonus * inset
        order = np.argsort(-s2)[:k_out]
        out_idx[i] = idx[i][order]
        out_sc[i] = s[order]
        out_in[i] = inset[order]
    return out_idx, out_sc, out_in


# ----------------------------------------------------------------------------- role -> O*NET occupation neighbourhood
def role_onet_neighbourhood(role_ssoc: pd.DataFrame, edition: int = 2024) -> dict[str, set]:
    """role_id -> set of O*NET-SOC codes reachable through SSOC -> ISCO -> O*NET."""
    o2s = xw.onetsoc_to_ssoc(edition)[["onet_soc_code", "ssoc_code"]]
    m = role_ssoc[["role_id", "ssoc_code"]].merge(o2s, on="ssoc_code", how="inner")
    return m.groupby("role_id").onet_soc_code.apply(set).to_dict()


# ----------------------------------------------------------------------------- SFw key tasks -> O*NET tasks / DWAs
def match_sfw_key_tasks(role_ssoc: pd.DataFrame, k_global: int = 20, k_out: int = 3, bonus: float = 0.10) -> tuple[pd.DataFrame, pd.DataFrame]:
    tasks = onet.load_tasks()
    kt = sg.load_sfw_key_tasks()
    neigh = role_onet_neighbourhood(role_ssoc)
    matcher = HybridMatcher(tasks.task_statement.tolist())
    idx, sc = matcher.query(kt.key_task.tolist(), k=k_global)
    allowed = [neigh.get(r, set()) for r in kt.role_id]
    cand_occ = tasks.onet_soc_code.values
    ridx, rsc, rin = _rerank_with_bonus(idx, sc, allowed, cand_occ, bonus=bonus, k_out=k_out)
    rows = []
    for i in range(len(kt)):
        for r in range(k_out):
            rows.append((kt.key_task_id.iat[i], r + 1, int(tasks.task_id.iat[ridx[i, r]]), tasks.onet_soc_code.iat[ridx[i, r]], float(rsc[i, r]), bool(rin[i, r])))
    task_links = pd.DataFrame(rows, columns=["key_task_id", "rank", "task_id", "onet_soc_code", "score", "in_mapped_occ"])
    # DWA level (2,087 generic activities): top-1 is usually a safe, coarse anchor
    dwa = onet.load_dwa_hierarchy()
    dm = HybridMatcher(dwa.dwa_name.tolist())
    didx, dsc = dm.query(kt.key_task.tolist(), k=1)
    dwa_links = pd.DataFrame({"key_task_id": kt.key_task_id.values, "dwa_id": dwa.dwa_id.values[didx[:, 0]], "iwa_id": dwa.iwa_id.values[didx[:, 0]], "gwa_id": dwa.gwa_id.values[didx[:, 0]], "score": dsc[:, 0]})
    return task_links, dwa_links


# ----------------------------------------------------------------------------- TSC / unique skills -> O*NET
def _onet_descriptor_corpus() -> pd.DataFrame:
    cm30 = onet.load_content_model("30.1")
    cm31 = onet.load_content_model("31.0")
    fam = {"2.A": "skill", "2.B": "skill", "2.C": "knowledge", "1.A": "ability", "4.A": "work_activity"}
    def pick(cm, version):
        d = cm[cm.element_id.str.match(r"^(2\.A\.\d\.[a-z]|2\.B\.\d\.[a-z]|2\.C\.\d+\.[a-z]|1\.A\.\d\.[a-z]\.\d|4\.A\.\d\.[a-z]\.\d)$")].copy()
        d["family"] = d.element_id.str[:3].map(fam)
        d["onet_version"] = version
        return d
    a = pick(cm30, "30.1")
    b = pick(cm31, "31.0")
    b = b[~b.element_id.isin(set(a.element_id))]  # only the new 31.0 skills
    out = pd.concat([a, b], ignore_index=True)
    out["text"] = out.element_name + ": " + out.description.fillna("")
    return out[["element_id", "element_name", "description", "family", "onet_version", "text"]]


def match_unique_skills(k: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Unique skills (Skills Framework 2.0 language, ~2.4k) -> O*NET IWAs and O*NET descriptors."""
    us = sg.load_unique_skills()
    q = (us.skill_title + ": " + us.skill_description.fillna("")).tolist()
    iwa = onet.load_dwa_hierarchy()[["iwa_id", "iwa_name", "gwa_id"]].drop_duplicates("iwa_id").reset_index(drop=True)
    m = HybridMatcher(iwa.iwa_name.tolist())
    idx, sc = m.query(q, k=k)
    rows = [(us.skill_id.iat[i], r + 1, iwa.iwa_id.iat[idx[i, r]], iwa.gwa_id.iat[idx[i, r]], float(sc[i, r])) for i in range(len(us)) for r in range(k)]
    iwa_links = pd.DataFrame(rows, columns=["skill_id", "rank", "iwa_id", "gwa_id", "score"])
    desc = _onet_descriptor_corpus()
    m2 = HybridMatcher(desc.text.tolist())
    idx2, sc2 = m2.query(q, k=k)
    rows = [(us.skill_id.iat[i], r + 1, desc.element_id.iat[idx2[i, r]], desc.family.iat[idx2[i, r]], desc.onet_version.iat[idx2[i, r]], float(sc2[i, r])) for i in range(len(us)) for r in range(k)]
    desc_links = pd.DataFrame(rows, columns=["skill_id", "rank", "element_id", "family", "onet_version", "score"])
    return iwa_links, desc_links


# ----------------------------------------------------------------------------- SSOC task bullets -> O*NET tasks
def match_ssoc_tasks(edition: int = 2024, k_global: int = 20, k_out: int = 3, bonus: float = 0.10) -> pd.DataFrame:
    tasks = onet.load_tasks()
    st = sg.load_ssoc_tasks(edition)
    o2s = xw.onetsoc_to_ssoc(edition)[["onet_soc_code", "ssoc_code"]]
    o2s["unit"] = o2s.ssoc_code.str[:4]
    neigh = o2s.groupby("unit").onet_soc_code.apply(set).to_dict()
    m = HybridMatcher(tasks.task_statement.tolist())
    idx, sc = m.query(st.ssoc_task.tolist(), k=k_global)
    allowed = [neigh.get(c, set()) for c in st.ssoc_code]
    ridx, rsc, rin = _rerank_with_bonus(idx, sc, allowed, tasks.onet_soc_code.values, bonus=bonus, k_out=k_out)
    rows = [(st.ssoc_task_id.iat[i], st.ssoc_code.iat[i], r + 1, int(tasks.task_id.iat[ridx[i, r]]), tasks.onet_soc_code.iat[ridx[i, r]], float(rsc[i, r]), bool(rin[i, r])) for i in range(len(st)) for r in range(k_out)]
    return pd.DataFrame(rows, columns=["ssoc_task_id", "ssoc_unit", "rank", "task_id", "onet_soc_code", "score", "in_mapped_occ"])


# ----------------------------------------------------------------------------- SFw roles without a SingStat code -> SSOC by description
def _ssoc_corpus(edition: int = 2024) -> pd.DataFrame:
    d = sg.load_ssoc_definitions(edition)
    d = d[d.level == 5].copy()
    alt = sg.load_ssoc_alpha_index(edition)
    alt = alt[alt.title_type != "SFw Title"].groupby("ssoc_code").index_title.apply(lambda s: "; ".join(s.head(12))).rename("alt_titles")
    d = d.merge(alt, on="ssoc_code", how="left")
    d["text"] = d.ssoc_title + ". " + d.definition.fillna("") + " Also called: " + d.alt_titles.fillna("")
    return d[["ssoc_code", "ssoc_title", "text"]].reset_index(drop=True)


def match_roles_to_ssoc_by_text(roles: pd.DataFrame, edition: int = 2024, k: int = 3) -> pd.DataFrame:
    corpus = _ssoc_corpus(edition)
    m = HybridMatcher(corpus.text.tolist(), alpha=0.6)
    q = (roles.job_role + " (" + roles.sector + "). " + roles.job_role_description.fillna("")).tolist()
    idx, sc = m.query(q, k=k)
    rows = [(roles.role_id.iat[i], r + 1, corpus.ssoc_code.iat[idx[i, r]], corpus.ssoc_title.iat[idx[i, r]], float(sc[i, r])) for i in range(len(roles)) for r in range(k)]
    return pd.DataFrame(rows, columns=["role_id", "rank", "ssoc_code", "ssoc_title", "score"])


def evaluate_role_text_matcher(edition: int = 2024, k: int = 5) -> dict:
    """Hold-out check: for roles SingStat did code, how often does the text matcher recover that code?"""
    coded, _ = xw.sfw_role_to_ssoc(edition)
    truth = coded.groupby("role_id").ssoc_code.apply(set)
    roles = sg.load_sfw_roles()
    roles = roles[roles.role_id.isin(truth.index)].reset_index(drop=True)
    pred = match_roles_to_ssoc_by_text(roles, edition, k=k)
    top1 = pred[pred["rank"] == 1].set_index("role_id").ssoc_code
    hit1 = np.mean([top1[r] in truth[r] for r in roles.role_id])
    hitk = np.mean([bool(set(pred[pred.role_id == r].ssoc_code) & truth[r]) for r in roles.role_id])
    unit1 = np.mean([top1[r][:4] in {c[:4] for c in truth[r]} for r in roles.role_id])
    return {"n_roles": int(len(roles)), "top1_5digit": float(hit1), f"top{k}_5digit": float(hitk), "top1_4digit_unit": float(unit1)}


def build_role_ssoc_crosswalk(edition: int = 2024, text_threshold: float = 0.0) -> pd.DataFrame:
    """Final role -> SSOC table: SingStat's codes where they exist, else the text matcher's top-1
    (with its score, so users can filter)."""
    coded, _ = xw.sfw_role_to_ssoc(edition)
    roles = sg.load_sfw_roles()
    missing = roles[~roles.role_id.isin(coded.role_id)].reset_index(drop=True)
    coded = coded[["role_id", "ssoc_code", "weight", "match_method", "match_score"]].copy()
    coded["source"] = "singstat_index"
    if len(missing):
        t = match_roles_to_ssoc_by_text(missing, edition, k=1)
        t = t[t.score >= text_threshold].rename(columns={"score": "match_score"})
        t["weight"] = 1.0
        t["match_method"] = "text_top1"
        t["source"] = "text_matcher"
        coded = pd.concat([coded, t[["role_id", "ssoc_code", "weight", "match_method", "match_score", "source"]]], ignore_index=True)
    coded["edition"] = edition
    return coded
