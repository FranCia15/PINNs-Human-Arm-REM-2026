"""
export_splits.py

Writes the exact data splits of the paper to results/splits.json, as trial
target numbers (tgt_number) per subject:
  train       the training trials of the paper model's nine subjects
  evaluation  the three Table I trial sets (common.SCENARIOS): training,
              heldout (unseen trials of training subjects), unseen (subjects)
Trials are the valid trials of DBAS22's initial-acquisition phase after
the dataset's exclusion files, in dataset order.

Usage (from the repository root): python src/export_splits.py
"""

import json

from common import N_TRIALS, PAPER_MODEL, RESULTS, SCENARIOS, model_config, scenario_trials, subject_trials
from dataset import PRIMARY_PHASE


def main():
    counts = model_config(PAPER_MODEL)["subject_trial_counts"]
    tgts = lambda trials: [int(t["tgt_number"]) for t in trials]
    split = {
        "dataset": "DBAS22 / 3D-ARM-Gaze",
        "phase": PRIMARY_PHASE,
        "rule": "valid trials after the dataset's exclusion files, in dataset order; training = first n trials "
                f"of each training subject; heldout = the next {N_TRIALS}; unseen = first {N_TRIALS} trials",
        "train": {s: tgts(subject_trials(s)[:n]) for s, n in sorted(counts.items(), key=lambda kv: int(kv[0][1:]))},
        "evaluation": {sc: {s: tgts(scenario_trials(sc, s, counts)) for s in subjects} for sc, subjects in SCENARIOS.items()},
    }
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "splits.json"
    out.write_text(json.dumps(split, indent=1) + "\n")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
