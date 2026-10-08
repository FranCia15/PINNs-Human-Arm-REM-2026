"""
eval_joint_errors.py

Joint-angle predictions and errors of the PINN and of the IK baseline on the
Table I trial sets (common.SCENARIOS), every recorded frame.

IK baseline: fixed arm yaw (swivel) and Newton initial guess = per-joint
median posture over the training trials of all the model's training
subjects (baseline_ik.compute_fixed_posture), shared by all subjects and
scenarios -- the same data the PINN is trained on, no access to the
evaluated trials.

Output: results/joint_errors_<model>.npz with, per scenario and subject,
  <scenario>_<subject>_q_pinn, _q_ik      (N_frames, 7) predicted angles, rad
  <scenario>_<subject>_err_pinn, _err_ik  (N_frames, 7) abs error, deg
  <scenario>_<subject>_tgts, _lens        trial target numbers, frames per trial
(the _ik keys only for the paper model; IK does not depend on the network).

Usage (from the repository root):
    python src/eval_joint_errors.py [paper_model | ablation_noeffort]
"""

import sys

import numpy as np
import torch

from common import PAPER_MODEL, RESULTS, SCENARIOS, DATASET_ROOT, load_model, model_config, scenario_trials, training_pool
from dataset import load_subject_info
from kinematics import load_segment_constants
from baseline_ik import compute_fixed_posture, solve_ik_batch, pack_joint_angles, AY_INDEX
from pinn_model import query_trajectory_batch
from trajectory_interp import HandTrajectorySpline
from train_pinn import DTYPE

MODEL = sys.argv[1] if len(sys.argv) > 1 else PAPER_MODEL


def pinn_angles(net, trials, subject_feats) -> np.ndarray:
    """(N_frames, 7) PINN joint angles (rad), every frame of every trial, in trial order."""
    out = []
    for t in trials:
        spline = HandTrajectorySpline(t["timestamps"], t["hand_pos"], t["hand_quat"], dtype=DTYPE)
        q, _, _ = query_trajectory_batch(net, spline, subject_feats, torch.tensor(t["timestamps"], dtype=DTYPE))
        out.append(q.detach().numpy())
    return np.concatenate(out)


def ik_angles(fixed_posture, trials, seg_dims, quat_wri2arrow) -> np.ndarray:
    """(N_frames, 7) IK joint angles (rad) with the fixed arm yaw of fixed_posture."""
    ay = fixed_posture[AY_INDEX]
    cat = lambda k: np.concatenate([t[k] for t in trials])
    q_free, _ = solve_ik_batch(cat("hand_pos"), cat("hand_quat"), cat("arm_root_pos"), cat("arm_root_quat"),
                               seg_dims, quat_wri2arrow, ay, fixed_posture[[i for i in range(7) if i != AY_INDEX]])
    return pack_joint_angles(q_free, ay)


def main():
    counts = model_config(MODEL)["subject_trial_counts"]
    net = load_model(MODEL)
    with_ik = MODEL == PAPER_MODEL
    fixed_posture = compute_fixed_posture(training_pool(counts)) if with_ik else None
    out = {}
    for sc, subjects in SCENARIOS.items():
        for s in subjects:
            d = DATASET_ROOT / s
            trials = scenario_trials(sc, s, counts)
            seg_dims, quat_wri2arrow, _ = load_segment_constants(d)
            info = load_subject_info(d)
            q_gt = np.concatenate([t["joint_angles"] for t in trials])
            k = f"{sc}_{s}"
            out[f"{k}_q_pinn"] = pinn_angles(net, trials, torch.tensor([info.u_arm, info.f_arm, info.height], dtype=DTYPE))
            out[f"{k}_err_pinn"] = np.degrees(np.abs(out[f"{k}_q_pinn"] - q_gt))
            if with_ik:
                out[f"{k}_q_ik"] = ik_angles(fixed_posture, trials, seg_dims, quat_wri2arrow)
                out[f"{k}_err_ik"] = np.degrees(np.abs(out[f"{k}_q_ik"] - q_gt))
            out[f"{k}_tgts"] = np.array([t["tgt_number"] for t in trials])
            out[f"{k}_lens"] = np.array([len(t["timestamps"]) for t in trials])
            print(f"{sc:8s} {s:3s}: {len(trials)} trials, {len(q_gt)} frames | mean error PINN "
                  f"{out[f'{k}_err_pinn'].mean():.2f} deg" + (f", IK {out[f'{k}_err_ik'].mean():.2f} deg" if with_ik else ""),
                  flush=True)
    RESULTS.mkdir(exist_ok=True)
    out_path = RESULTS / f"joint_errors_{MODEL}.npz"
    np.savez(out_path, **out)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
