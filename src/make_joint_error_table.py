"""
make_joint_error_table.py

Joint-angle error table of the paper (Table I): mean absolute joint-angle error (deg, mean +- standard
deviation over all frames) per joint and overall, for the PINN, the IK
baseline and the dynamics baseline, on the three trial sets of
common.SCENARIOS. '*' marks the lowest mean per joint and scenario.

Input:  results/joint_errors_paper_model.npz (eval_joint_errors.py),
        results/dynamics_<scenario>.npz (eval_dynamics.py)
Output: printed table and results/joint_error_table.csv

Usage (from the repository root): python src/make_joint_error_table.py
"""

import numpy as np

from common import PAPER_MODEL, RESULTS, SCENARIOS
from dataset import JOINT_COLS

METHODS = ("PINN", "IK", "Dynamics")


def scenario_errors(scenario: str) -> dict:
    """Per method, (N_frames, 7) abs errors in degrees, all subjects of the scenario."""
    je = np.load(RESULTS / f"joint_errors_{PAPER_MODEL}.npz")
    dyn = np.load(RESULTS / f"dynamics_{scenario}.npz")
    err = {m: [] for m in METHODS}
    for s in SCENARIOS[scenario]:
        assert (je[f"{scenario}_{s}_tgts"] == dyn[f"{s}_tgts"]).all(), (scenario, s)
        err["PINN"].append(je[f"{scenario}_{s}_err_pinn"])
        err["IK"].append(je[f"{scenario}_{s}_err_ik"])
        err["Dynamics"].append(dyn[f"{s}_dyn"])
    return {m: np.concatenate(v) for m, v in err.items()}


def main():
    rows = ["scenario,method," + ",".join(f"{j}_mean,{j}_std" for j in JOINT_COLS) + ",mean,std"]
    for sc in SCENARIOS:
        err = scenario_errors(sc)
        means = np.array([np.r_[e.mean(0), e.mean()] for e in err.values()])
        best = means.argmin(0)
        print(f"\n{sc} ({', '.join(SCENARIOS[sc])}; {len(err['PINN'])} frames)")
        print(f"{'':9s}" + "".join(f"{c:>13s}" for c in (*JOINT_COLS, "mean")))
        for i, (m, e) in enumerate(err.items()):
            stds = np.r_[e.std(0), e.std()]
            print(f"{m:9s}" + "".join(f"{mu:6.2f}±{sd:5.2f}{'*' if best[k] == i else ' '}"
                                      for k, (mu, sd) in enumerate(zip(means[i], stds))))
            rows.append(f"{sc},{m}," + ",".join(f"{mu:.4f},{sd:.4f}" for mu, sd in zip(means[i], stds)))
    out = RESULTS / "joint_error_table.csv"
    out.write_text("\n".join(rows) + "\n")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
