"""
paired_tests.py

Paired statistical tests of Sec. V-A on the Table I trial sets: per trial,
the mean absolute joint-angle error over all frames and joints, PINN vs. IK
and PINN vs. dynamics baseline, Wilcoxon signed-rank test over the 80 trials
of each scenario (4 subjects x 20 trials). The same test is run on the
per-trial swivel-angle error. Per-subject mean errors are printed too.

Note: trials are nested in subjects, so the per-trial test treats them as
independent.

Input:  results/joint_errors_paper_model.npz, results/dynamics_<scenario>.npz,
        results/cartesian_errors.npz

Usage (from the repository root): python src/paired_tests.py
"""

import numpy as np
from scipy.stats import wilcoxon

from common import PAPER_MODEL, RESULTS, SCENARIOS


def per_trial(frames, lens):
    """(N_frames, ...) per-frame errors -> per-trial means (over frames and joints)."""
    edges = np.cumsum(np.r_[0, lens])
    return np.array([frames[a:b].mean() for a, b in zip(edges[:-1], edges[1:])])


def test(name, a, b):
    p = wilcoxon(a, b).pvalue
    print(f"  {name:22s} PINN {a.mean():6.2f}  other {b.mean():6.2f}  "
          f"PINN lower in {(a < b).sum():2d}/{len(a)} trials, Wilcoxon p = {p:.1e}")


def main():
    je = np.load(RESULTS / f"joint_errors_{PAPER_MODEL}.npz")
    cart = np.load(RESULTS / "cartesian_errors.npz")
    for sc, subjects in SCENARIOS.items():
        dyn = np.load(RESULTS / f"dynamics_{sc}.npz")
        err = {m: [] for m in ("pinn", "ik", "dyn")}
        rows = []
        for s in subjects:
            lens = je[f"{sc}_{s}_lens"]
            e = {"pinn": per_trial(je[f"{sc}_{s}_err_pinn"], lens), "ik": per_trial(je[f"{sc}_{s}_err_ik"], lens),
                 "dyn": per_trial(dyn[f"{s}_dyn"], lens)}
            for m in err:
                err[m].append(e[m])
            rows.append(f"{s}: " + " ".join(f"{m} {e[m].mean():5.2f}" for m in e))
        err = {m: np.concatenate(v) for m, v in err.items()}
        print(f"== {sc} ({len(err['pinn'])} trials) | per subject: " + " | ".join(rows))
        test("joint error vs IK", err["pinn"], err["ik"])
        test("joint error vs Dyn", err["pinn"], err["dyn"])
        sw = {m: per_trial(cart[f"{sc}_{m}_swivel"], cart[f"{sc}_lens"]) for m in ("pinn", "ik", "dyn")}
        test("swivel error vs IK", sw["pinn"], sw["ik"])
        test("swivel error vs Dyn", sw["pinn"], sw["dyn"])


if __name__ == "__main__":
    main()
