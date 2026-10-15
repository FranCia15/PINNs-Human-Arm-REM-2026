"""
common.py

Repository paths and the evaluation trial sets of the paper (Table I), in
one place so every script uses the same definition.

Trial sets (20 trials per subject, every recorded frame):
  training  S1, S12, S13, S20 -- their first 20 trials, used for training
  heldout   S1, S12, S13, S20 -- the 20 trials right after the training ones
  unseen    S8, S11, S16, S17 -- (not used for training) their first 20 trials
"Trials" are the valid trials of DBAS22's initial-acquisition phase after
the dataset's own exclusion files, in dataset order (dataset.extract_trials).
"""

import json
from pathlib import Path

from dataset import DATASET_ROOT, PRIMARY_PHASE, load_exclusions, extract_trials

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
PAPER_MODEL = "paper_model"            # models/paper_model: the model of the paper
ABLATION_MODEL = "ablation_noeffort"   # models/ablation_noeffort: same, trained without the effort term
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
ROBODK = ROOT / "robodk"

N_TRIALS = 20
SCENARIOS = {
    "training": ["s1", "s12", "s13", "s20"],
    "heldout": ["s1", "s12", "s13", "s20"],
    "unseen": ["s8", "s11", "s16", "s17"],
}


def model_config(model: str = PAPER_MODEL) -> dict:
    return json.loads((MODELS / model / "config.json").read_text())


def load_model(model: str = PAPER_MODEL):
    """Trained network of models/<model> (eval mode, float64)."""
    import torch
    from pinn_model import CoordinateNet
    from train_pinn import DTYPE
    cfg = model_config(model)
    net = CoordinateNet(hidden=cfg["hidden"], n_layers=cfg["n_layers"], dtype=DTYPE)
    net.load_state_dict(torch.load(MODELS / model / "model.pt", weights_only=True))
    net.eval()
    return net


def subject_trials(subject: str) -> list:
    d = DATASET_ROOT / subject
    return extract_trials(d, PRIMARY_PHASE, load_exclusions(d, PRIMARY_PHASE))


def scenario_trials(scenario: str, subject: str, counts: dict) -> list:
    """Evaluated trials of one subject; counts = the model's subject_trial_counts."""
    trials = subject_trials(subject)
    if scenario == "training":
        return trials[:counts[subject]][:N_TRIALS]
    if scenario == "heldout":
        return trials[counts[subject]:counts[subject] + N_TRIALS]
    return trials[:N_TRIALS]


def training_pool(counts: dict) -> list:
    """Training trials of all training subjects: the data that calibrates the IK fixed posture."""
    return [t for s in counts for t in subject_trials(s)[:counts[s]]]
