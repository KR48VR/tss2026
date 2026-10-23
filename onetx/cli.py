"""Command line: `python -m onetx build` / `python -m onetx report` / `python -m onetx fetch`."""
from __future__ import annotations

import argparse
import json
import sys


def main(argv=None):
    ap = argparse.ArgumentParser(prog="onetx", description="Crosswalked O*NET + Singapore frameworks + AI frontier signals")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch", help="download all public sources into data/raw")
    b = sub.add_parser("build", help="build every processed table, the DuckDB file and the report")
    b.add_argument("--now", default="2026-10-01", help="date treated as 'today' by the timing model")
    sub.add_parser("report", help="write docs/REPORT.md from the processed tables")
    f = sub.add_parser("forecast", help="re-run only the timing model (fast) and the report")
    f.add_argument("--now", default="2026-10-01")
    st = sub.add_parser("site", help="build the self-contained web page (site/index.html + site/kt.json)")
    st.add_argument("--companion", default="https://claude.ai/artifact/Pxd31ubFiwTQjdTSBsu7qZ", help="URL of the companion page linked from the method view ('' for none)")
    q = sub.add_parser("query", help="run SQL against the DuckDB file")
    q.add_argument("sql")
    rt = sub.add_parser("rate", help="rate every O*NET task on the 2026 rubric with Claude (resumes; cached in data/reference)")
    rt.add_argument("--model", default="sonnet")
    rt.add_argument("--batch", type=int, default=80)
    rt.add_argument("--workers", type=int, default=4)
    rt.add_argument("--limit", type=int, default=None, help="rate only this many tasks (pilot)")
    args = ap.parse_args(argv)
    if args.cmd == "fetch":
        from .fetch import fetch_all

        fetch_all()
    elif args.cmd == "build":
        from .build import build
        from .report import write_report

        stats = build(now=args.now)
        print(json.dumps(stats, indent=2, default=str))
        write_report()
    elif args.cmd == "forecast":
        from .build import build_forecasts
        from .report import write_report
        from .config import PROCESSED
        from .io import write_json

        stats = json.loads((PROCESSED / "build_stats.json").read_text()) if (PROCESSED / "build_stats.json").exists() else {"now": args.now}
        stats["now"] = args.now
        stats.update(build_forecasts(now=args.now))
        write_json(stats, PROCESSED / "build_stats.json")
        print(json.dumps(stats, indent=2, default=str))
        write_report()
    elif args.cmd == "report":
        from .report import write_report

        write_report()
    elif args.cmd == "site":
        from .site import build_site

        print(build_site(args.companion or None))
    elif args.cmd == "rate":
        from .rate import rate_all

        print(json.dumps(rate_all(args.model, args.batch, args.workers, args.limit), indent=1))
    elif args.cmd == "query":
        import duckdb

        from .config import DUCKDB_PATH

        con = duckdb.connect(str(DUCKDB_PATH), read_only=True)
        print(con.sql(args.sql).df().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
