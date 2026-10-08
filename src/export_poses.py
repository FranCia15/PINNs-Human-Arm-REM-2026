"""
export_poses.py

Exports the poses of the RoboDK pose comparison (Fig. 3 of the paper): the
recorded, IK-baseline, dynamics-baseline and PINN joint angles of one trial
at three recorded frames, initial, centre and final. The centre frame is
the midpoint of the hand path by arc length (not by time), so the frames
depend on the hand path only, not on any method's error.

IK calibration: the fixed posture is here the median over all of the
subject's trials, not over the training trials as in the evaluation
(eval_joint_errors.py); kept so that the published figure is reproduced
exactly.

Output: robodk/poses/<subject>_tgt<N>_compare_{groundtruth,ik,dynamics,pinn}.csv
        (read by robodk/capture_poses.py, check_rig_rendering.py and
        make_pose_figure.py)
Usage (from the repository root):
    python src/export_poses.py [model] [subject] [tgt_number]
Defaults (the paper's figure): paper_model, s1, tgt 10.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from scipy.interpolate import interp1d

from dataset import DATASET_ROOT, PRIMARY_PHASE, JOINT_COLS, load_exclusions, extract_trials, load_subject_info
from kinematics import load_segment_constants
from anthropometry import estimate_body_mass
from dynamics import build_segment_params
from baseline_ik import compute_fixed_posture, solve_ik_batch, pack_joint_angles, AY_INDEX
from baseline_dynamics import solve_trial as solve_trial_dynamics
from trajectory_interp import HandTrajectorySpline
from pinn_model import query_trajectory_batch
from train_pinn import DTYPE
from common import PAPER_MODEL, ROBODK, load_model
from tool_box.rot_quat_utils import change_quat_ref_handeness

MODEL = sys.argv[1] if len(sys.argv) > 1 else PAPER_MODEL
SUBJECT = sys.argv[2] if len(sys.argv) > 2 else "s1"
TGT_NUMBER = int(sys.argv[3]) if len(sys.argv) > 3 else 10
N_DENSE = 200  # scan resolution for locating the arc-length midpoint

OUT_DIR = ROBODK / "poses"


def main():
    net = load_model(MODEL)

    subject_dir = DATASET_ROOT / SUBJECT
    exclusions = load_exclusions(subject_dir, PRIMARY_PHASE)
    all_trials = extract_trials(subject_dir, PRIMARY_PHASE, exclusions)
    trial = next((t for t in all_trials if t["tgt_number"] == TGT_NUMBER), None)
    if trial is None:
        raise SystemExit(f"No trial with tgt_number={TGT_NUMBER} for {SUBJECT}")

    info = load_subject_info(subject_dir)
    subject_feats = torch.tensor([info.u_arm, info.f_arm, info.height], dtype=DTYPE)
    seg_dims, quat_wri2arrow, arm_side = load_segment_constants(subject_dir)

    t0, t1 = trial["timestamps"][0], trial["timestamps"][-1]
    t_dense = np.linspace(t0, t1, N_DENSE)

    spline = HandTrajectorySpline(trial["timestamps"], trial["hand_pos"], trial["hand_quat"], dtype=DTYPE)

    # hand path on a dense grid, only to locate the arc-length midpoint
    pos_interp = interp1d(trial["timestamps"], trial["hand_pos"], axis=0)
    hand_pos_dense = pos_interp(t_dense)

    fixed_posture = compute_fixed_posture(all_trials)
    ay_fixed = fixed_posture[AY_INDEX]
    free_idx = [i for i in range(7) if i != AY_INDEX]

    # arc-length midpoint on the dense grid, snapped to the nearest
    # recorded frame (the model is queried at recorded timestamps only)
    seg_lengths = np.linalg.norm(np.diff(hand_pos_dense, axis=0), axis=1)
    cum_length = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total_length = cum_length[-1]
    idx_center_dense = int(np.searchsorted(cum_length, total_length / 2.0))
    t_center = t_dense[idx_center_dense]

    t_native = trial["timestamps"]
    idx_center_native = int(np.argmin(np.abs(t_native - t_center)))
    chosen_idx = sorted({0, idx_center_native, len(t_native) - 1})
    labels = ["initial", "center", "final"]
    print("Chosen frames (index, t, fraction of total path length):")
    for label, i in zip(labels, chosen_idx):
        print(f"  {label:8s} idx={i:3d}  t={t_native[i] - t0:.2f}s  "
              f"path_frac={cum_length[idx_center_dense] / total_length if label == 'center' else (0.0 if label == 'initial' else 1.0):.2f}")

    # the four methods at the three frames
    t_chosen = t_native[chosen_idx]
    t_chosen_torch = torch.tensor(t_chosen, dtype=DTYPE)

    q_gt_chosen = trial["joint_angles"][chosen_idx]  # (3, 7) radians, exact recorded values
    root_quat_chosen = trial["arm_root_quat"][chosen_idx]

    q_pinn_chosen, _, _ = query_trajectory_batch(net, spline, subject_feats, t_chosen_torch)
    q_pinn_chosen = q_pinn_chosen.detach().numpy()  # (3, 7) radians

    q_free_chosen, _ = solve_ik_batch(trial["hand_pos"][chosen_idx], trial["hand_quat"][chosen_idx],
                                       trial["arm_root_pos"][chosen_idx], trial["arm_root_quat"][chosen_idx],
                                       seg_dims, quat_wri2arrow, ay_fixed, fixed_posture[free_idx])
    q_ik_chosen = pack_joint_angles(q_free_chosen, ay_fixed)  # (3, 7) radians

    # dynamics baseline: solved over the whole trial (it is causal), then
    # taken at the three frames
    body_mass = estimate_body_mass(info.height)
    hand_length = float(np.linalg.norm(seg_dims[2]))
    seg_params = build_segment_params(info.u_arm, info.f_arm, hand_length, body_mass, dtype=DTYPE)
    q_free0, _ = solve_ik_batch(trial["hand_pos"][0:1], trial["hand_quat"][0:1],
                                 trial["arm_root_pos"][0:1], trial["arm_root_quat"][0:1],
                                 seg_dims, quat_wri2arrow, ay_fixed, fixed_posture[free_idx])
    q0_dyn = pack_joint_angles(q_free0, ay_fixed)[0]
    dyn_result = solve_trial_dynamics(trial, seg_dims, quat_wri2arrow, seg_params, q0_dyn)
    q_dyn_chosen = dyn_result["q"][chosen_idx]  # (3, 7) radians

    # one 3-row CSV per method
    header = "t," + ",".join(JOINT_COLS) + ",rqx,rqy,rqz,rqw\n"

    def export(name, q_all_rad):
        path = OUT_DIR / f"{SUBJECT}_tgt{TGT_NUMBER}_compare_{name}.csv"
        root_quat_rh = change_quat_ref_handeness(root_quat_chosen.copy())
        with open(path, "w") as f:
            f.write(header)
            for row_i, (label, i) in enumerate(zip(labels, chosen_idx)):
                row_q = np.degrees(q_all_rad[row_i])
                row_rq = root_quat_rh[row_i]
                f.write(f"{t_native[i] - t0:.4f}," + ",".join(f"{v:.4f}" for v in row_q)
                        + "," + ",".join(f"{v:.6f}" for v in row_rq) + "\n")
        print(f"Exported {path}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    export("groundtruth", q_gt_chosen)
    export("ik", q_ik_chosen)
    export("dynamics", q_dyn_chosen)
    export("pinn", q_pinn_chosen)


if __name__ == "__main__":
    main()
