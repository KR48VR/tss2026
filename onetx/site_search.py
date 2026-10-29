"""Search assets for the web page: alternative titles, an O*NET task index, and meaning-search
embeddings (a small static embedding model exported for the browser, plus quantised item vectors).

Files written to site/:
  search.json   alternative SSOC titles and the O*NET task index (loaded on first task search)
  model.json    tokenizer vocabulary and model shape (loaded on first meaning search)
  model.bin.txt int8 token-embedding table, one float32 scale per row (model_scales.bin.txt)
  emb.json      what each item vector refers to (type and index)
  emb.bin.txt   int8 item vectors, one float32 scale per row (emb_scales.bin.txt)

The binary tables are written as base64 text (.txt) because the artifact host serves text and
media types only; the page decodes them with atob() when meaning search is first used.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import onet
from . import singapore as sg
from .config import ROOT
from .io import load
from .site import STATUS_CODE, TIER_CODE, _r

SITE = ROOT / "site"
MODEL_NAME = "minishlab/potion-base-4M"


def _quantise(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scales = np.abs(mat).max(axis=1) / 127.0
    scales[scales == 0] = 1.0
    q = np.round(mat / scales[:, None]).clip(-127, 127).astype(np.int8)
    return q, scales.astype(np.float32)


def _write_b64(path: Path, raw: bytes) -> None:
    """Write bytes as one line of base64 text (served as text/plain by the artifact host)."""
    path.write_text(base64.b64encode(raw).decode("ascii"))


def build_search_assets(ssoc_order: list[str], role_ids: list[str], key_tasks: list[list], onet_order: list[str] | None = None, national: dict | None = None) -> dict:
    """ssoc_order: SSOC codes in the page's order; role_ids: role ids in the page's order;
    key_tasks: the per-role grouped key tasks exactly as written to kt.json."""
    SITE.mkdir(parents=True, exist_ok=True)
    ssoc_idx = {c: i for i, c in enumerate(ssoc_order)}
    role_idx = {r: i for i, r in enumerate(role_ids)}

    # ---- alternative titles (SingStat alphabetical index, principal and alternative titles)
    alpha = sg.load_ssoc_alpha_index(2024)
    alpha = alpha[alpha.title_type.isin(["Alternative Title", "Principal Title"]) & alpha.ssoc_code.isin(ssoc_idx)]
    ssoc_titles = load("sg_ssoc2024_weighted").set_index("ssoc_code").ssoc_title
    alt = []
    seen = set()
    for _, r in alpha.iterrows():
        t = str(r.index_title).strip()
        if t.casefold() == str(ssoc_titles.get(r.ssoc_code, "")).casefold():
            continue
        key = (r.ssoc_code, t.casefold())
        if key in seen:
            continue
        seen.add(key)
        alt.append([ssoc_idx[r.ssoc_code], t])

    # ---- US job titles: O*NET alternate titles -> occupation index (the page's O*NET order)
    onet_idx = {c: i for i, c in enumerate(onet_order or [])}
    us_alt = []
    if onet_idx:
        at = load("onet_alternate_titles")
        occ_titles = load("onet_occupations").set_index("onet_soc_code").title
        seen_us = set()
        for _, r in at[at.onet_soc_code.isin(onet_idx)].iterrows():
            t = str(r.alt_title).strip()
            key = (r.onet_soc_code, t.casefold())
            if key in seen_us or t.casefold() == str(occ_titles.get(r.onet_soc_code, "")).casefold():
                continue
            seen_us.add(key)
            us_alt.append([onet_idx[r.onet_soc_code], t])

    # ---- O*NET task index: statement, status, central year, occupation, up to 3 SSOC codes, tier, occupation index
    tasks = load("onet_tasks")[["task_id", "onet_soc_code", "task_statement"]]
    fc = load("forecast_tasks").set_index("task_id")
    occ = load("onet_occupations").set_index("onet_soc_code").title
    o2s = load("xw_onetsoc_ssoc2024").sort_values("weight", ascending=False)
    top_ssoc = o2s.groupby("onet_soc_code").ssoc_code.apply(lambda s: [ssoc_idx[c] for c in s.head(3) if c in ssoc_idx]).to_dict()
    onet_tasks = []
    for _, t in tasks.iterrows():
        f = fc.loc[t.task_id]
        onet_tasks.append([str(t.task_statement)[:160], STATUS_CODE.get(f.status, 3), _r(f.disrupt_year_p50, 1), occ.get(t.onet_soc_code, ""), top_ssoc.get(t.onet_soc_code, []), TIER_CODE.get(f.get("evidence_tier"), None), onet_idx.get(t.onet_soc_code)])
    # ---- further countries: alternative titles per national code (ABS index of titles; ONS coding index)
    nat_alt, nat_titles, nat_codes = {}, {}, list((national or {}).keys())
    for code, order in (national or {}).items():
        idx = {c: i for i, c in enumerate(order)}
        rows, seen_n = [], set()
        if code == "AUS":
            from . import au_labour as au
            w = load("au_weights").set_index("anzsco_code").anzsco_title
            for _, r in au.load_title_index().iterrows():
                if r.unit in idx and (r.unit, r.title.casefold()) not in seen_n and r.title.casefold() != str(w.get(r.unit, "")).casefold():
                    seen_n.add((r.unit, r.title.casefold())); rows.append([idx[r.unit], r.title])
            nat_titles[code] = [str(w.get(c, c)) for c in order]
        elif code == "GBR":
            from . import uk_labour as uk
            w = load("uk_weights").set_index("soc_code").soc_title
            for _, r in uk.coding_index_titles().iterrows():
                if r.soc_code in idx and (r.soc_code, r.key) not in seen_n and r.title.casefold() != str(w.get(r.soc_code, "")).casefold():
                    seen_n.add((r.soc_code, r.key)); rows.append([idx[r.soc_code], r.title])
            nat_titles[code] = [str(w.get(c, c)) for c in order]
        else:
            # any further country with a titles table (code column first, then title)
            from . import countries as cn
            spec = cn.SPECS[code]
            w = load(f"{spec.prefix}weights").set_index(spec.code_col)[spec.title_col]
            if load.__module__ and (ROOT / "data" / "processed" / f"{spec.prefix}titles.parquet").exists():
                for _, r in load(f"{spec.prefix}titles").iterrows():
                    c = str(r[spec.code_col]); t = str(r.title).strip()
                    if c in idx and (c, t.casefold()) not in seen_n and t.casefold() != str(w.get(c, "")).casefold():
                        seen_n.add((c, t.casefold())); rows.append([idx[c], t])
            nat_titles[code] = [str(w.get(c, c)) for c in order]
        nat_alt[code] = rows
    (SITE / "search.json").write_text(json.dumps({"alt": alt, "onet": onet_tasks, "us": us_alt, "nat": nat_alt}, ensure_ascii=True, separators=(",", ":")))

    # ---- items to embed
    from model2vec import StaticModel

    m = StaticModel.from_pretrained(MODEL_NAME)
    texts, refs = [], []
    ssocw = load("sg_ssoc2024_weighted").set_index("ssoc_code")
    for c in ssoc_order:
        texts.append(str(ssocw.loc[c].ssoc_title)); refs.append([0, ssoc_idx[c]])
    for i, (si, t) in enumerate(alt):
        texts.append(t); refs.append([1, i])
    roles = load("sfw_roles").set_index("role_id")
    for rid in role_ids:
        r = roles.loc[rid]
        texts.append(f"{r.job_role} ({r.track})" if isinstance(r.track, str) and r.track else str(r.job_role)); refs.append([2, role_idx[rid]])
    for ri, groups in enumerate(key_tasks):
        for gi, (cwf, items) in enumerate(groups):
            for ti, it in enumerate(items):
                texts.append(it[0]); refs.append([3, ri, gi, ti])
    for i, t in enumerate(onet_tasks):
        texts.append(t[0]); refs.append([4, i])
    if onet_idx:
        occ_titles = load("onet_occupations").set_index("onet_soc_code").title
        for c in onet_order:
            texts.append(str(occ_titles.get(c, c))); refs.append([5, onet_idx[c]])
    # national titles are all embedded; alternative titles up to 12 per code (the UK coding index has 29,000,
    # which would push the vector file past the 16 MB the artifact host allows per file); keyword search
    # still covers every alternative title through search.json
    for k, code in enumerate(nat_codes):
        for i, t in enumerate(nat_titles.get(code, [])):
            texts.append(t); refs.append([6, k, i])
        per_code: dict[int, int] = {}
        for i, (ni, t) in enumerate(nat_alt.get(code, [])):
            if per_code.get(ni, 0) >= 12:
                continue
            per_code[ni] = per_code.get(ni, 0) + 1
            texts.append(t); refs.append([7, k, i])
    vecs = m.encode(texts, normalize=True).astype(np.float32)
    q, sc = _quantise(vecs)
    _write_b64(SITE / "emb.bin.txt", q.tobytes())
    _write_b64(SITE / "emb_scales.bin.txt", sc.tobytes())
    (SITE / "emb.json").write_text(json.dumps({"n": len(refs), "dim": int(vecs.shape[1]), "refs": refs, "nat": nat_codes}, separators=(",", ":")))

    # ---- model export: vocabulary + int8 table
    tok = json.loads(m.tokenizer.to_str())
    vocab = tok["model"]["vocab"]
    table = m.embedding.astype(np.float32)
    tq, tsc = _quantise(table)
    _write_b64(SITE / "model.bin.txt", tq.tobytes())
    _write_b64(SITE / "model_scales.bin.txt", tsc.tobytes())
    (SITE / "model.json").write_text(json.dumps({"name": MODEL_NAME, "n": int(table.shape[0]), "dim": int(table.shape[1]), "unk_id": int(m.unk_token_id) if m.unk_token_id is not None else 1, "vocab": vocab, "lowercase": tok["normalizer"].get("lowercase", True)}, separators=(",", ":")))
    return {"alt_titles": len(alt), "us_alt_titles": len(us_alt), "national_alt_titles": {k: len(v) for k, v in nat_alt.items()}, "onet_tasks": len(onet_tasks), "items_embedded": len(refs), "dim": int(vecs.shape[1]), "vocab": len(vocab)}
