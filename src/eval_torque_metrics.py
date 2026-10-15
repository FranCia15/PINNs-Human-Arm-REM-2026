"""
eval_torque_metrics.py

Torque metrics of Sec. V-B on the Table I trial sets (common.SCENARIOS),
PINN with vs. without the effort term (models/paper_model,
models/ablation_noeffort). Torque as in torque.py (paper setting): the
reference from the recorded joint angles and the PINN from its predicted
trajectory, both with the 6 Hz Butterworth low-pass filter (reference
joint angles / PINN hand input). Per trial and joint:
  ripple  RMS of d(tau)/dt, PINN / reference (1 = as smooth as the reference)
  nrmse   RMS(tau_pinn - tau_ref) / range(tau_ref)
  corr    Pearson correlation with the reference

Runtime: ~10 min with 12 processes (240 trials x 2 models).

Output: results/torque_metrics.npz with <scenario>_<model>_<metric>
(n_trials, 7), <scenario>_subj and <scenario>_tgts; summary printed
(make_ablation.py prints the numbers reported in the paper).

Usage (from the repository root): python src/eval_torque_metrics.py [--workers 12]
"""

import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.stats import wilcoxon

from common import ABLATION_MODEL, PAPER_MODEL, RESULTS, SCENARIOS, model_config, scenario_trials
from dataset import JOINT_COLS

RUNS = {"effort": PAPER_MODEL, "noeffort": ABLATION_MODEL}
N_WORKERS = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 12


def trial_tgts(scenario, s, counts):
    return [t["tgt_number"] for t in scenario_trials(scenario, s, counts)]


def metrics_one(s, tgt):
    import torch
    torch.set_num_threads(1)
    from torque import compute_gt_and_pinn_torque
    out = {}
    for model, run in RUNS.items():
        t, tau_ref, tau = compute_gt_and_pinn_torque(run, s, tgt, smooth_hand=True, filt="butter")
        rough = lambda x: np.sqrt((np.gradient(x, t, axis=0) ** 2).mean(0))
        out[f"{model}_ripple"] = rough(tau) / rough(tau_ref)
        out[f"{model}_nrmse"] = np.sqrt(((tau - tau_ref) ** 2).mean(0)) / np.ptp(tau_ref, axis=0)
        out[f"{model}_corr"] = np.array([np.corrcoef(tau[:, j], tau_ref[:, j])[0, 1] for j in range(7)])
    return out


def main():
    counts = model_config()["subject_trial_counts"]
    jobs = [(sc, s, tgt) for sc, subs in SCENARIOS.items() for s in subs for tgt in trial_tgts(sc, s, counts)]
    unique = sorted({(s, tgt) for _, s, tgt in jobs})
    print(f"{len(jobs)} trials ({len(unique)} unique), 2 models, {N_WORKERS} workers", flush=True)

    t0, res = time.perf_counter(), {}
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(metrics_one, s, tgt): (s, tgt) for s, tgt in unique}
        for k, f in enumerate(futs, 1):
            res[futs[f]] = f.result()
            if k % 20 == 0 or k == len(futs):
                print(f"  {k}/{len(futs)} done, {(time.perf_counter() - t0) / 60:.1f} min", flush=True)

    out = {}
    for sc in SCENARIOS:
        keys = [(s, tgt) for c, s, tgt in jobs if c == sc]
        out[f"{sc}_subj"] = np.array([s for s, _ in keys])
        out[f"{sc}_tgts"] = np.array([tgt for _, tgt in keys])
        for m in res[keys[0]]:
            out[f"{sc}_{m}"] = np.stack([res[k][m] for k in keys])
    out_path = RESULTS / "torque_metrics.npz"
    np.savez(out_path, joints=np.array(JOINT_COLS), **out)
    print(f"Saved {out_path}\n")

    # summary: per scenario, median over trials of the per-trial median over joints
    for sc in list(SCENARIOS) + ["all"]:
        scs = list(SCENARIOS) if sc == "all" else [sc]
        get = lambda m: np.concatenate([out[f"{c}_{m}"] for c in scs])
        print(f"== {sc} ({len(get('effort_ripple'))} trials)")
        for metric in ("ripple", "nrmse", "corr"):
            e, n = np.median(get(f"effort_{metric}"), 1), np.median(get(f"noeffort_{metric}"), 1)
            better = (e < n) if metric != "corr" else (e > n)
            print(f"  {metric:6s} effort {np.median(e):5.2f}  no effort {np.median(n):5.2f}  "
                  f"effort better in {better.sum()}/{len(e)} trials, Wilcoxon p = {wilcoxon(e, n).pvalue:.1e}")
        print("  per joint, median over trials (effort / no effort):")
        for metric in ("ripple", "nrmse", "corr"):
            e, n = get(f"effort_{metric}"), get(f"noeffort_{metric}")
            print(f"    {metric:6s} " + "  ".join(f"{c} {np.median(e[:, j]):.2f}/{np.median(n[:, j]):.2f}"
                                                for j, c in enumerate(JOINT_COLS)))


if __name__ == "__main__":
    main()
