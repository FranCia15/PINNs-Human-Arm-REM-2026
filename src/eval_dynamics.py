"""
eval_dynamics.py

Classical dynamics baseline (baseline_dynamics.solve_trial) on the Table I
trial sets (common.SCENARIOS), every frame of each trial.

Initial pose: one IK solve on each trial's first frame, with the fixed
posture (ay and the Newton initial guess) calibrated once for everyone on
the training trials of all the paper model's training subjects (as for the
IK baseline in eval_joint_errors.py; no access to the evaluated trials).

Trials run in parallel processes (the solver is per-frame autodiff and
barely benefits from torch threads). Each finished trial is cached in
results/dyn_cache/, so an interrupted run resumes where it stopped.
Runtime: ~0.4 s per frame and process, ~50 min in total with 12 processes.

Output: results/dynamics_<scenario>.npz with, per subject, <s>_dyn
(N_frames, 7) abs joint error in degrees, <s>_q (rad), <s>_tau (N*m),
<s>_track (N_frames, 6) tracking residual (m / rotvec rad), <s>_tgts and
<s>_lens.

Usage (from the repository root):
    python src/eval_dynamics.py [--workers 12] [--max-trials K] [scenario ...]
--max-trials K runs only the first K trials of each scenario (smoke test,
no output npz).
"""

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from common import RESULTS, SCENARIOS, model_config, scenario_trials, training_pool
from dataset import DATASET_ROOT, load_subject_info
from kinematics import load_segment_constants
from anthropometry import estimate_body_mass
from baseline_ik import compute_fixed_posture, solve_ik_batch, pack_joint_angles, AY_INDEX


_argv = sys.argv[1:]
_opt = lambda k, default: _argv.pop(_argv.index(k) + 1) if k in _argv else default
N_WORKERS = int(_opt("--workers", 12))
MAX_TRIALS = _opt("--max-trials", None)
_args = [a for a in _argv if a not in ("--workers", "--max-trials")]
SELECTED = _args or list(SCENARIOS)


def solve_one(s: str, trial: dict, fixed_posture: np.ndarray, cache_path: Path) -> tuple:
    """Worker: initial IK pose + dynamics baseline for one trial, cached to disk."""
    import torch
    torch.set_num_threads(1)
    from dynamics import build_segment_params
    from baseline_dynamics import solve_trial, DTYPE

    t0 = time.perf_counter()
    d = DATASET_ROOT / s
    seg_dims, quat_wri2arrow, _ = load_segment_constants(d)
    info = load_subject_info(d)
    seg_params = build_segment_params(info.u_arm, info.f_arm, float(np.linalg.norm(seg_dims[2])),
                                      estimate_body_mass(info.height), dtype=DTYPE)
    ay_fixed = fixed_posture[AY_INDEX]
    free_idx = [i for i in range(7) if i != AY_INDEX]
    q_free0, _ = solve_ik_batch(trial["hand_pos"][:1], trial["hand_quat"][:1],
                                trial["arm_root_pos"][:1], trial["arm_root_quat"][:1],
                                seg_dims, quat_wri2arrow, ay_fixed, fixed_posture[free_idx])
    q0 = pack_joint_angles(q_free0, ay_fixed)[0]

    res = solve_trial(trial, seg_dims, quat_wri2arrow, seg_params, q0)
    err = np.degrees(np.abs(res["q"] - trial["joint_angles"]))
    tmp = cache_path.with_suffix(".tmp.npz")
    np.savez(tmp, err=err, q=res["q"], tau=res["tau"], track=res["track_residual"])
    os.replace(tmp, cache_path)
    return s, trial["tgt_number"], len(trial["timestamps"]), time.perf_counter() - t0, err.mean()


def main():
    counts = model_config()["subject_trial_counts"]
    pool = training_pool(counts)
    fixed_posture = compute_fixed_posture(pool)  # shared by every scenario
    print(f"fixed posture (deg, ik_pool over {len(pool)} trials):", np.round(np.degrees(fixed_posture), 2))

    cache_dir = RESULTS / "dyn_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # job list: (scenario, subject, trial); the same trial is solved once even if in two scenarios
    plan = {sc: {s: scenario_trials(sc, s, counts) for s in SCENARIOS[sc]} for sc in SELECTED}
    if MAX_TRIALS:
        plan = {sc: {s: v[:int(MAX_TRIALS)] for s, v in list(plan[sc].items())[:1]} for sc in plan}
    jobs, todo_frames = {}, 0
    for sc in plan:
        for s, trials in plan[sc].items():
            for t in trials:
                path = cache_dir / f"{s}_tgt{t['tgt_number']}.npz"
                if not path.exists() and path not in jobs:
                    jobs[path] = (s, t)
                    todo_frames += len(t["timestamps"])
    n_total = sum(len(v) for sc in plan for v in plan[sc].values())
    print(f"scenarios {list(plan)}: {n_total} trials, {len(jobs)} still to solve ({todo_frames} frames), "
          f"{N_WORKERS} workers", flush=True)

    t_start, done_frames = time.perf_counter(), 0
    # longest trials first, for better load balancing
    order = sorted(jobs.items(), key=lambda kv: -len(kv[1][1]["timestamps"]))
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = [ex.submit(solve_one, s, t, fixed_posture, path) for path, (s, t) in order]
        for k, f in enumerate(as_completed(futs), 1):
            s, tgt, n, dt, mean_err = f.result()
            done_frames += n
            elapsed = time.perf_counter() - t_start
            eta = elapsed / done_frames * (todo_frames - done_frames)
            print(f"[{k}/{len(futs)}] {s} tgt {tgt:3d}: {n:4d} frames {dt:6.1f}s  mean err {mean_err:6.2f} deg | "
                  f"elapsed {elapsed / 60:5.1f} min, ETA {eta / 60:5.1f} min", flush=True)

    if MAX_TRIALS:
        print("smoke test done (no output npz written)")
        return
    for sc in plan:
        out = {}
        for s, trials in plan[sc].items():
            parts = [np.load(cache_dir / f"{s}_tgt{t['tgt_number']}.npz") for t in trials]
            out[f"{s}_dyn"] = np.concatenate([p["err"] for p in parts])
            out[f"{s}_q"] = np.concatenate([p["q"] for p in parts])
            out[f"{s}_tau"] = np.concatenate([p["tau"] for p in parts])
            out[f"{s}_track"] = np.concatenate([p["track"] for p in parts])
            out[f"{s}_tgts"] = np.array([t["tgt_number"] for t in trials])
            out[f"{s}_lens"] = np.array([len(t["timestamps"]) for t in trials])
            print(f"{sc} {s}: {len(trials)} trials, tgts {out[f'{s}_tgts'].tolist()}, "
                  f"mean err {out[f'{s}_dyn'].mean():.2f} deg")
        out_path = RESULTS / f"dynamics_{sc}.npz"
        np.savez(out_path, subjects=np.array(SCENARIOS[sc]), fixed_posture=fixed_posture, **out)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
