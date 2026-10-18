"""Package a run folder into a small, audio-free zip that can be pushed to git or dropped into
a chat for review.  Allow-list only: nothing outside the listed files is ever included."""
from __future__ import annotations

import zipfile
from pathlib import Path

ALLOWED = ["config.json", "env.json", "manifest.csv", "predictions.csv", "metrics.json",
           "quality.json", "log.txt", "verify_report.json"]
MAX_BYTES = 8 * 1024 * 1024


def make_report(run_dir, out_path=None) -> Path:
    run = Path(run_dir)
    out = Path(out_path) if out_path else run.with_suffix(".report.zip")
    total = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for name in ALLOWED:
            p = run / name
            if p.exists():
                total += p.stat().st_size
                if total > MAX_BYTES:
                    raise SystemExit(f"report would exceed {MAX_BYTES//2**20} MB; refusing")
                z.write(p, arcname=f"{run.name}/{name}")
    return out
