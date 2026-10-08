"""
export_trajectory.py

Exports one trial for playback on the RoboDK mannequin (robodk/play_trajectory.py,
Sec. VI): the PINN-predicted joint trajectory and the recorded one, at the
trial's native recorded timestamps, as plain CSV files.

CSV columns: t (s from trial start), sp, sr, ay, ep, fy, hp, hr (deg, the
order of dataset.JOINT_COLS), rqx, rqy, rqz, rqw (trunk / arm-root
orientation, right-handed: DBAS22's shoulder angles are relative to it).

Output: robodk/trajectories/<subject>_tgt<N>_{pinn,groundtruth}.csv
Usage (from the repository root):
    python src/export_trajectory.py [model] [subject] [tgt_number]
Defaults: paper_model, s1, tgt 10.
"""

import sys

import numpy as np
import torch

from common import PAPER_MODEL, ROBODK, load_model, subject_trials
from dataset import DATASET_ROOT, JOINT_COLS, load_subject_info
from pinn_model import query_trajectory_batch
from trajectory_interp import HandTrajectorySpline
from train_pinn import DTYPE
from tool_box.rot_quat_utils import change_quat_ref_handeness

MODEL = sys.argv[1] if len(sys.argv) > 1 else PAPER_MODEL
SUBJECT = sys.argv[2] if len(sys.argv) > 2 else "s1"
TGT_NUMBER = int(sys.argv[3]) if len(sys.argv) > 3 else 10
OUT_DIR = ROBODK / "trajectories"


def write_csv(path, t, q_rad, root_quat):
    with open(path, "w") as f:
        f.write("t," + ",".join(JOINT_COLS) + ",rqx,rqy,rqz,rqw\n")
        for row_t, row_q, row_rq in zip(t, np.degrees(q_rad), root_quat):
            f.write(f"{row_t:.4f}," + ",".join(f"{v:.4f}" for v in row_q) + "," + ",".join(f"{v:.6f}" for v in row_rq) + "\n")
    print(f"Exported {len(t)} frames to {path}")


def main():
    trial = next((t for t in subject_trials(SUBJECT) if t["tgt_number"] == TGT_NUMBER), None)
    if trial is None:
        raise SystemExit(f"No valid trial with tgt_number={TGT_NUMBER} for {SUBJECT}")
    info = load_subject_info(DATASET_ROOT / SUBJECT)
    net = load_model(MODEL)
    spline = HandTrajectorySpline(trial["timestamps"], trial["hand_pos"], trial["hand_quat"], dtype=DTYPE)
    q_pinn, _, _ = query_trajectory_batch(net, spline, torch.tensor([info.u_arm, info.f_arm, info.height], dtype=DTYPE),
                                          torch.tensor(trial["timestamps"], dtype=DTYPE))
    t_rel = trial["timestamps"] - trial["timestamps"][0]
    root_quat = change_quat_ref_handeness(trial["arm_root_quat"].copy())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_DIR / f"{SUBJECT}_tgt{TGT_NUMBER}_pinn.csv", t_rel, q_pinn.detach().numpy(), root_quat)
    write_csv(OUT_DIR / f"{SUBJECT}_tgt{TGT_NUMBER}_groundtruth.csv", t_rel, trial["joint_angles"], root_quat)


if __name__ == "__main__":
    main()
