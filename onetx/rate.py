"""Fresh capability ratings for every O*NET task, produced by Claude through the Claude Code CLI.

Each task gets three ratings on a fixed rubric (onetx/rubric_2026.md): exposure (E0/E1/E2, the same
scale Eloundou et al. used in 2023 so the two vintages can be compared), end-to-end autonomy (A0-A3)
and the main barrier. Ratings are cached in data/reference/ai_task_ratings_<version>.csv so a build
never re-rates a task; `python -m onetx rate` resumes where it left off and can be re-run with a new
rubric version to measure how the frontier moved.

The runner shells out to `claude -p` (non-interactive) because that is the authenticated path in the
environments this project runs in; set ONETX_RATE_CMD to override the command.
"""
from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .config import ROOT

RUBRIC_PATH = Path(__file__).with_name("rubric_2026.md")
REF_DIR = ROOT / "data" / "reference"
EXPOSURE = {"E0", "E1", "E2"}
AUTONOMY = {"A0", "A1", "A2", "A3"}
BARRIERS = {"none", "physical", "in_person", "relationship", "accountability", "system_access", "judgement", "data_absent"}
COLUMNS = ["task_id", "onet_soc_code", "exposure", "autonomy", "barrier", "model", "rubric_version", "rated_at", "batch"]


def rubric() -> tuple[str, str]:
    text = RUBRIC_PATH.read_text()
    m = re.search(r"RUBRIC_VERSION:\s*(\S+)", text)
    version = m.group(1) if m else "unversioned"
    body = text.split("\n", 1)[1].strip()
    return version, body


def ratings_path(version: str | None = None) -> Path:
    version = version or rubric()[0]
    return REF_DIR / f"ai_task_ratings_{version}.csv"


def load_ratings(version: str | None = None) -> pd.DataFrame:
    p = ratings_path(version)
    if not p.exists():
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(p, dtype=str)
    return df.drop_duplicates("task_id", keep="last")


def task_table() -> pd.DataFrame:
    """Tasks to rate: id, occupation code and title, statement (from the processed O*NET tables)."""
    from . import io

    t = io.load("frontier_tasks")[["task_id", "onet_soc_code", "task_statement"]]
    occ = io.load("onet_occupations")[["onet_soc_code", "title"]]
    t = t.merge(occ, on="onet_soc_code", how="left")
    t["task_id"] = t.task_id.astype(str)
    return t.sort_values(["onet_soc_code", "task_id"]).reset_index(drop=True)


def build_prompt(body: str, batch: pd.DataFrame) -> str:
    lines = [body, "", "Tasks:"]
    for n, r in enumerate(batch.itertuples(index=False), start=1):
        lines.append(f"{n}. [{r.onet_soc_code} {r.title}] {str(r.task_statement).strip()}")
    lines.append("")
    lines.append(f"Reply with a JSON array of exactly {len(batch)} objects.")
    return "\n".join(lines)


def _parse(text: str, n: int) -> list[dict]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end < 0:
        raise ValueError("no JSON array in reply")
    arr = json.loads(text[start : end + 1])
    if not isinstance(arr, list):
        raise ValueError("reply is not a list")
    out = {}
    for item in arr:
        try:
            k = int(item["n"])
        except Exception:
            continue
        e, a, b = str(item.get("exposure", "")).upper(), str(item.get("autonomy", "")).upper(), str(item.get("barrier", "")).lower()
        if e in EXPOSURE and a in AUTONOMY and b in BARRIERS:
            out[k] = {"exposure": e, "autonomy": a, "barrier": b}
    return [out.get(i) for i in range(1, n + 1)]


def call_claude(prompt: str, model: str, timeout: int = 600, cwd: Path | None = None) -> tuple[str, float]:
    """Run one non-interactive CLI call; return (reply text, cost in USD)."""
    cmd = os.environ.get("ONETX_RATE_CMD")
    if cmd:
        argv = cmd.split() + [prompt]
    else:
        argv = ["claude", "-p", "--no-session-persistence", "--tools", "", "--model", model, "--output-format", "json", prompt]
    r = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, cwd=str(cwd or tempfile.gettempdir()))
    if r.returncode != 0:
        raise RuntimeError(f"claude exited {r.returncode}: {r.stderr[:400]}")
    try:
        d = json.loads(r.stdout)
        return str(d.get("result", "")), float(d.get("total_cost_usd") or 0.0)
    except json.JSONDecodeError:
        return r.stdout, 0.0


def rate_all(model: str = "sonnet", batch_size: int = 80, workers: int = 4, limit: int | None = None, task_ids: list[str] | None = None, retries: int = 2, log=print) -> dict:
    version, body = rubric()
    tasks = task_table()
    done = set(load_ratings(version).task_id)
    todo = tasks[~tasks.task_id.isin(done)]
    if task_ids is not None:
        todo = todo[todo.task_id.isin(set(map(str, task_ids)))]
    if limit:
        todo = todo.head(limit)
    log(f"rubric {version}: {len(done)} rated, {len(todo)} to rate with {model}, batches of {batch_size}, {workers} workers")
    if todo.empty:
        return {"version": version, "rated": len(done), "new": 0, "cost_usd": 0.0}
    REF_DIR.mkdir(parents=True, exist_ok=True)
    path = ratings_path(version)
    new_file = not path.exists()
    lock = threading.Lock()
    stats = {"new": 0, "cost_usd": 0.0, "failed_batches": 0}
    batches = [todo.iloc[i : i + batch_size] for i in range(0, len(todo), batch_size)]
    work_dir = Path(tempfile.mkdtemp(prefix="onetx-rate-"))

    def run(bi: int, batch: pd.DataFrame):
        prompt = build_prompt(body, batch)
        last_err = None
        for attempt in range(retries + 1):
            try:
                text, cost = call_claude(prompt, model, cwd=work_dir)
                parsed = _parse(text, len(batch))
                rows = []
                stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
                for r, p in zip(batch.itertuples(index=False), parsed):
                    if p:
                        rows.append([r.task_id, r.onet_soc_code, p["exposure"], p["autonomy"], p["barrier"], model, version, stamp, bi])
                if len(rows) < 0.8 * len(batch):
                    raise ValueError(f"only {len(rows)} of {len(batch)} parsed")
                return rows, cost
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"batch {bi} failed: {last_err}")

    with open(path, "a", newline="") as f, ThreadPoolExecutor(max_workers=workers) as ex:
        w = csv.writer(f)
        if new_file:
            w.writerow(COLUMNS)
            f.flush()
        futs = {ex.submit(run, i, b): i for i, b in enumerate(batches)}
        t0 = time.time()
        for k, fut in enumerate(as_completed(futs), start=1):
            try:
                rows, cost = fut.result()
            except Exception as e:  # noqa: BLE001
                stats["failed_batches"] += 1
                log(f"  ! {e}")
                continue
            with lock:
                w.writerows(rows)
                f.flush()
                stats["new"] += len(rows)
                stats["cost_usd"] += cost
            if k % 10 == 0 or k == len(batches):
                log(f"  {k}/{len(batches)} batches, {stats['new']} tasks, ${stats['cost_usd']:.2f}, {time.time() - t0:.0f}s")
    return {"version": version, "rated": len(done) + stats["new"], **stats}


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description="rate every O*NET task on the 2026 rubric with Claude")
    ap.add_argument("--model", default="sonnet")
    ap.add_argument("--batch", type=int, default=80)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="rate only this many tasks (pilot)")
    a = ap.parse_args(argv)
    print(json.dumps(rate_all(a.model, a.batch, a.workers, a.limit), indent=1))


if __name__ == "__main__":
    sys.exit(main())
