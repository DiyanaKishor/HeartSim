# Working loop: run locally, get independent review

Principle: results computed on the author's machine are the only results that count. A reviewer
(a person or Claude) works from the run folder, never from numbers typed into a chat.

## One experiment

1. `git status` must be clean (otherwise `env.json` records `+dirty` and `verify` warns).
2. Run: `python -m heartsim smoke --data-root data/yaseen --workers 2 --name yaseen_v1`
3. Check yourself: `python -m heartsim verify runs/<run> --data-root data/yaseen` (must exit 0).
4. Share: `python -m heartsim report runs/<run>` produces `runs/<run>.report.zip`
   (no audio, a few MB). Either attach it in the chat, or commit it to a `reports/` folder
   (`reports/` is not git-ignored; `runs/` is).
5. Ask the reviewer to recompute the headline numbers from `predictions.csv` and to read
   `verify_report.json`, `config.json` warnings and the `trivial` baseline before discussing any result.

## Cadence for long runs

Push or attach `log.txt` partway through (it is flushed after every line) so problems such as
NaNs, absurd timings or a wrong data path are caught early rather than after hours.

## Trust but verify, cheaply

* Every few experiments, re-run one with the same seed and `python -m heartsim compare` it with the original.
* Any number that goes into a paper is re-derived from `predictions.csv` by a script in `reports/`,
  not copied from a terminal.
* A result that does not clearly beat the `trivial` baseline, or whose shuffled-label control is above chance, is a bug
  or a shortcut until proven otherwise.

## Order of work (from the audit)

1. Smoke test on Yaseen to confirm plumbing and see the shortcut level on your machine.
2. Segmentation of S1/S2 (use the CirCor `.tsv` annotations first, then an automatic method) -> per-cycle montages.
3. CirCor with patient ids via `--groups-csv`; PhysioNet 2016 with database ids and leave-one-database-out.
4. Add baselines in the same harness: spectrogram CNN, CWT, CyTex-style (after reading that paper in full),
   and a random radial filter-bank control.
5. Only then the CNN on simulated-pattern montages, and ablations over drive point, grid, damping, cycles.
