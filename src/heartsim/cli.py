"""Command line:  python -m heartsim <command> ...

  env                       print hardware/software report and sizing notes
  smoke --data-root DIR     run the baseline harness on <DIR>/<CLASS>/*.wav
  verify RUN_DIR            independently re-check a run folder (exit code 1 on FAIL)
  compare RUN_A RUN_B       check two runs produced identical predictions (same seed/config)
  report RUN_DIR            write an audio-free zip of the run folder
"""
from __future__ import annotations

import argparse
import json
import sys

from . import __version__


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="heartsim", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("env")

    s = sub.add_parser("smoke")
    s.add_argument("--data-root", required=True)
    s.add_argument("--out-root", default="runs")
    s.add_argument("--name", default="smoke")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--n-splits", type=int, default=5)
    s.add_argument("--max-per-class", type=int, default=None)
    s.add_argument("--workers", type=int, default=1)
    s.add_argument("--groups-csv", default=None, help="csv with columns file,group (patient/database ids)")
    s.add_argument("--grid-n", type=int, default=None)
    s.add_argument("--band-hi", type=float, default=None)

    v = sub.add_parser("verify")
    v.add_argument("run_dir")
    v.add_argument("--data-root", default=None)

    c = sub.add_parser("compare")
    c.add_argument("run_a")
    c.add_argument("run_b")

    r = sub.add_parser("report")
    r.add_argument("run_dir")
    r.add_argument("--out", default=None)

    a = ap.parse_args(argv)

    if a.cmd == "env":
        from .env import env_info, sizing_advice
        info = env_info(".")
        print(json.dumps(info, indent=2))
        print()
        for n in sizing_advice(info):
            print("-", n)
        return 0

    if a.cmd == "smoke":
        from .smoke import run_smoke
        sim = {"grid_n": a.grid_n} if a.grid_n else {}
        pre = {"band_hi_hz": a.band_hi} if a.band_hi else {}
        run_smoke(a.data_root, a.out_root, a.name, a.seed, a.n_splits, a.max_per_class,
                  a.workers, a.groups_csv, sim, pre, repo_dir=".")
        return 0

    if a.cmd == "verify":
        from .verify import verify_run
        rep = verify_run(a.run_dir, a.data_root)
        print(rep.text())
        return 0 if rep.ok else 1

    if a.cmd == "compare":
        from .verify import compare_runs
        rep = compare_runs(a.run_a, a.run_b)
        print(rep.text())
        return 0 if rep.ok else 1

    if a.cmd == "report":
        from .report import make_report
        print(make_report(a.run_dir, a.out))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
