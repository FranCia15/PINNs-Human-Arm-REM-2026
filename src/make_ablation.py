"""
make_ablation.py

Ablation of the effort term (Sec. V-B): the paper model against the same
model trained without the effort term (models/ablation_noeffort), on the
Table I trial sets.

  joint-angle accuracy  overall mean abs error (deg, all frames) per
                        scenario, with vs. without the effort term
  torque                per trial and joint, against the torque derived
                        from the recorded motion (eval_torque_metrics.py):
                          ripple  RMS torque rate of change / reference's
                          nrmse   RMSE / range of the reference torque
                          corr    Pearson correlation with the reference
                        summarised per trial by the median over joints,
                        then over trials (with the number of trials where
                        the effort model is better and a paired Wilcoxon
                        test); plus the median per-trial reduction of the
                        nRMSE.

Input:  results/joint_errors_paper_model.npz, results/joint_errors_ablation_noeffort.npz
        (eval_joint_errors.py), results/torque_metrics.npz (eval_torque_metrics.py)

Usage (from the repository root): python src/make_ablation.py
"""

import numpy as np
from scipy.stats import wilcoxon

from common import ABLATION_MODEL, PAPER_MODEL, RESULTS, SCENARIOS


def main():
    print("Joint-angle error (deg, all frames): with / without effort term")
    files = {m: np.load(RESULTS / f"joint_errors_{m}.npz") for m in (PAPER_MODEL, ABLATION_MODEL)}
    for sc, subjects in SCENARIOS.items():
        mean = {m: np.concatenate([f[f"{sc}_{s}_err_pinn"] for s in subjects]).mean() for m, f in files.items()}
        print(f"  {sc:8s} {mean[PAPER_MODEL]:6.2f} / {mean[ABLATION_MODEL]:6.2f}")

    tm = np.load(RESULTS / "torque_metrics.npz")
    get = lambda key: np.median(np.concatenate([tm[f"{sc}_{key}"] for sc in SCENARIOS]), axis=1)
    print("\nTorque vs. the recorded-motion reference, all trials (median over joints, then over trials):")
    for metric in ("ripple", "nrmse", "corr"):
        e, n = get(f"effort_{metric}"), get(f"noeffort_{metric}")
        better = (e > n) if metric == "corr" else (e < n)
        print(f"  {metric:6s} with {np.median(e):5.2f} / without {np.median(n):5.2f} | effort better in "
              f"{better.sum()}/{len(e)} trials, Wilcoxon p = {wilcoxon(e, n).pvalue:.1e}")
    e, n = get("effort_nrmse"), get("noeffort_nrmse")
    print(f"  nRMSE reduction by the effort term: {100 * np.median(1 - e / n):.0f}% in the median trial")


if __name__ == "__main__":
    main()
