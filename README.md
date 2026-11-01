# HeartSim

Audio of a heartbeat goes into a simulated membrane (2D wave equation, driven at the centre,
open/absorbing edges). The resulting height field is summarised and used for classification.
The code is arranged so that **every reported number can be re-derived and sanity-checked
by someone who does not trust the person or model that produced it.**

Status: research scaffold, version 0.2. It runs end to end on a laptop CPU with classical
classifiers, on Yaseen (smoke test) and CirCor (patient-grouped). It does **not** yet contain per-cycle
montages or a CNN, and nothing in
this repository demonstrates diagnostic value yet. See *Known limitations*.

## Install

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest -q            # ~30 s; add  -m "not slow"  to skip the two boundary tests
python -m heartsim env         # hardware report + sizing notes (paste this to your reviewer)
```

`make` is optional; every Makefile target is just a `python -m heartsim ...` command.

## Data layout

```
data/yaseen/N/*.wav   data/yaseen/AS/*.wav   data/yaseen/MR/*.wav   data/yaseen/MS/*.wav   data/yaseen/MVP/*.wav
```

CirCor DigiScope (the dataset to use for real claims; open access, about 450 MB zipped, the public
60% training part, 3,163 recordings):

```
data/circor/training_data.csv
data/circor/training_data/<patient>_<AV|MV|PV|TV|...>.wav / .tsv / .hea
```

Download: `https://physionet.org/content/circor-heart-sound/1.0.3/` (zip is served from S3; `curl.exe -L -C -`
resumes an interrupted download).

The Yaseen files ship as `.rar` archives in
[the dataset repo](https://github.com/yaseen21khan/Classification-of-Heart-Sound-Signal-Using-Multiple-Features-).
Extract with 7-Zip/WinRAR, or `node scripts/extract_yaseen.cjs <dir with rars> data/yaseen`
(needs `npm i node-unrar-js`). `data/`, `runs/` and all audio are git-ignored.

## Commands

| Command | What it does |
|---|---|
| `python -m heartsim smoke --data-root data/yaseen --workers 2` | features + grouped cross-validation + controls, writes `runs/<time>_smoke/` |
| `python -m heartsim verify runs/<run> --data-root data/yaseen` | independent re-check of a run folder, exit code 1 on FAIL |
| `python -m heartsim compare runs/<a> runs/<b>` | same seed/config/data must give identical predictions |
| `python -m heartsim report runs/<run>` | audio-free zip (config, env, manifest, predictions, metrics, log) |
| `python -m heartsim env` | CPU/RAM/GPU/package versions and sizing advice |

CirCor run (patient-grouped folds, one `--excerpt-s` excerpt per recording starting at the first annotated S1,
a `meta` baseline that uses only age/sex/height/weight/pregnancy/campaign/location, and per-patient scores):

```
python -m heartsim smoke --dataset circor --target outcome --data-root data/circor --workers 4 --name circor_outcome
python -m heartsim smoke --dataset circor --target murmur  --data-root data/circor --workers 4 --name circor_murmur
```

`--target outcome` (Normal/Abnormal) is the screening label; `murmur` drops the Unknown rows. Patients that
appear under two ids (CSV column *Additional ID*) are merged into one group. A patient's prediction is the class
with the highest mean probability over their recordings (`metrics.aggregate_patients`).
Read the `patient_level` numbers, not the per-recording ones, and read them next to `trivial` and `meta`.
`--balance auto` (default for CirCor) grows class-balanced forests and corrects the decision for the training class
frequencies, because the murmur target is about 20% Present and a plain forest answers "Absent" almost every time.
Every model also reports **AUROC** (one-vs-rest, macro), which does not depend on a decision threshold; `verify`
recomputes it with a separate rank-based implementation, per recording and per patient.

Options worth knowing: `--grid-n`, `--band-hi`, `--groups-csv file.csv` (columns `file,group`;
provide patient or recording-database ids whenever the dataset has them), `--max-per-class N`
for a quick first pass. Features are cached in `runs/_cache`, so a re-run only recomputes what changed.

## What a run folder contains

`config.json` (parameters, seed, qualifying flags) · `env.json` (versions, hardware, git commit,
`+dirty` marker) · `manifest.csv` (id, path, sha256, label, group) · `predictions.csv` (one row per
model and sample, out-of-fold) · `metrics.json` (numbers as reported by training code) · `log.txt`.

## What `verify` checks, and what it cannot

Checks: all files present and schema-valid; ids unique and consistent; `y_true` matches the manifest;
every headline metric recomputed from `predictions.csv` by a separate numpy implementation (the training code uses
scikit-learn) and compared; no group (patient/database) is test data in two folds; duplicate audio
across folds; shuffled-label controls stay near chance; the model is compared with the
non-cardiac-features baseline; data files still match their sha256; seed and git commit recorded.

Cannot: prove the training happened as logged (re-run one fold yourself occasionally and use
`compare`), prove the labels are right, or prove the result is scientifically meaningful. A PASS
means "internally consistent and the guards held".

## Known limitations (from the audit of the prototype; measured, not assumed)

* **Grid resolution.** At the default grid (140) the scheme's phase speed is within 5% only up to
  about 75 Hz (10%: about 106 Hz). Published murmur components reach 300-700 Hz. Content above the
  reliable limit is dispersed, not lost. The run log and `config.json` warn when `--band-hi` exceeds it.
  Raising resolution costs roughly grid^3.
* **The model is linear and has no threshold**; real Faraday-wave patterns respond at half the drive
  frequency and need a minimum drive amplitude. Treat the simulator as a designed feature transform,
  not a physical replica. In an open domain it is a fixed radial filter of the audio.
* **Radial structure dominates.** On one real recording a radial profile alone explained a median 88%
  of field energy. The `sim` features here are radial ring summaries, so they are a baseline, not the end state.
* **Yaseen is a smoke test only.** It has no patient ids, clips from varied sources, and a random forest on
  non-cardiac features (loudness, duration, silence, DC) reached 97% in 5-fold CV. Always read results next to the
  `trivial` model. Use a dataset with patient/database ids (CirCor, PhysioNet 2016) for any real claim.
* **CirCor caveats.** Only the public 60% is used, so numbers are not comparable with the Challenge's hidden
  set. Murmur and Outcome are patient-level labels (a recording from a patient with a murmur may not contain one).
  Only one excerpt per recording is simulated; the S1/S2 `.tsv` is used to place it, not yet to build per-cycle montages.
* **Not implemented yet:** per-cycle montages from the S1/S2 annotations, leave-one-database-out runs,
  random-radial-filter control (does the physics matter at all?), CNN training, calibration to hardware.
* Frame storage is not needed and not done: about 70 MB per second of audio at grid 140. Features are streamed.

## Layout

```
src/heartsim/  simulator.py preprocess.py features.py splits.py metrics.py runlog.py verify.py smoke.py circor.py env.py report.py cli.py
tests/         regression tests, incl. the boundary-reflection and frame-timing defects found in the audit
docs/WORKFLOW.md   the local-run + review loop
scripts/       extract_yaseen.cjs
```
