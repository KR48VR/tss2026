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


def build_search_assets(ssoc_order: list[str], role_ids: list[str], key_tasks: list[list]) -> dict:
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

    # ---- O*NET task index: statement, status, central year, occupation, up to 3 SSOC codes
    tasks = load("onet_tasks")[["task_id", "onet_soc_code", "task_statement"]]
    fc = load("forecast_tasks").set_index("task_id")
    occ = load("onet_occupations").set_index("onet_soc_code").title
    o2s = load("xw_onetsoc_ssoc2024").sort_values("weight", ascending=False)
    top_ssoc = o2s.groupby("onet_soc_code").ssoc_code.apply(lambda s: [ssoc_idx[c] for c in s.head(3) if c in ssoc_idx]).to_dict()
    onet_tasks = []
    for _, t in tasks.iterrows():
        f = fc.loc[t.task_id]
        onet_tasks.append([str(t.task_statement)[:160], STATUS_CODE.get(f.status, 3), _r(f.disrupt_year_p50, 1), occ.get(t.onet_soc_code, ""), top_ssoc.get(t.onet_soc_code, []), TIER_CODE.get(f.get("evidence_tier"), None)])
    (SITE / "search.json").write_text(json.dumps({"alt": alt, "onet": onet_tasks}, ensure_ascii=True, separators=(",", ":")))

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
    vecs = m.encode(texts, normalize=True).astype(np.float32)
    q, sc = _quantise(vecs)
    _write_b64(SITE / "emb.bin.txt", q.tobytes())
    _write_b64(SITE / "emb_scales.bin.txt", sc.tobytes())
    (SITE / "emb.json").write_text(json.dumps({"n": len(refs), "dim": int(vecs.shape[1]), "refs": refs}, separators=(",", ":")))

    # ---- model export: vocabulary + int8 table
    tok = json.loads(m.tokenizer.to_str())
    vocab = tok["model"]["vocab"]
    table = m.embedding.astype(np.float32)
    tq, tsc = _quantise(table)
    _write_b64(SITE / "model.bin.txt", tq.tobytes())
    _write_b64(SITE / "model_scales.bin.txt", tsc.tobytes())
    (SITE / "model.json").write_text(json.dumps({"name": MODEL_NAME, "n": int(table.shape[0]), "dim": int(table.shape[1]), "unk_id": int(m.unk_token_id) if m.unk_token_id is not None else 1, "vocab": vocab, "lowercase": tok["normalizer"].get("lowercase", True)}, separators=(",", ":")))
    return {"alt_titles": len(alt), "onet_tasks": len(onet_tasks), "items_embedded": len(refs), "dim": int(vecs.shape[1]), "vocab": len(vocab)}
