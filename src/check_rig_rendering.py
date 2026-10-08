"""
check_rig_rendering.py

Offline check, without RoboDK, of how the mannequin renders the poses of
the pose comparison (Fig. 3; robodk/poses/, s1 tgt 10, 3 frames x 4
methods) with the mapping of robodk/rig_mapping.py, on the rig geometry measured in RoboDK
(robodk/dump_rig_geometry.py -> robodk/rig_geometry.json).

Prints:
  - that rig_mapping's segment chain reproduces config2quats_multi;
  - the rendering error of the recorded pose (segment-direction error);
  - per method and frame, the distance of the rendered elbow and wrist
    (mm) and the angle of the rendered hand (deg) from the rendered
    recorded pose, i.e. what the figure compares visually, next to the
    same quantities in the true geometry (the subject's own segments).
    The two differ only through the rig's generic segment lengths.

load_csv and true_points are also used by make_pose_figure.py.

Usage (from the repository root): python src/check_rig_rendering.py [subject] [tgt]
"""

import csv
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "robodk"))
from rig_mapping import load_rig, rendered_points, segment_rotations  # noqa: E402
from dataset import DATASET_ROOT  # noqa: E402
from kinematics import load_segment_constants  # noqa: E402
from tool_box.rot_quat_utils import config2quats_multi  # noqa: E402

TRAJ_DIR = Path(__file__).resolve().parents[1] / "robodk" / "poses"
METHODS = ("groundtruth", "ik", "dynamics", "pinn")
POSES = ("initial", "center", "final")
JOINTS = ("sp", "sr", "ay", "ep", "fy", "hp", "hr")


def load_csv(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    q_deg = np.array([[float(r[j]) for j in JOINTS] for r in rows])
    rq = np.array([[float(r[k]) for k in ("rqx", "rqy", "rqz", "rqw")] for r in rows])
    return q_deg, rq  # rq already right-handed (export_poses.py)


def true_points(q_deg, rq, seg_dims):
    """True DBAS22 elbow, wrist (relative to the shoulder, mm) and hand orientation."""
    up, fore, hand = (R.from_quat(x) for x in config2quats_multi(np.radians(q_deg), prev_rot=R.from_quat(rq)))
    elbow = up.apply(seg_dims[0]) * 1e3
    return elbow, elbow + fore.apply(seg_dims[1]) * 1e3, hand


def main():
    subject = sys.argv[1] if len(sys.argv) > 1 else "s1"
    tgt = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    seg_dims, _, _ = load_segment_constants(DATASET_ROOT / subject)
    rig = load_rig()
    data = {m: load_csv(TRAJ_DIR / f"{subject}_tgt{tgt}_compare_{m}.csv") for m in METHODS}
    unit = lambda v: v / np.linalg.norm(v, axis=-1, keepdims=True)
    print(f"{subject} tgt {tgt} | segments (mm) subject {np.linalg.norm(seg_dims[0]) * 1e3:.0f}/"
          f"{np.linalg.norm(seg_dims[1]) * 1e3:.0f}, rig {np.linalg.norm(rig['b_up']):.0f}/{np.linalg.norm(rig['b_fore']):.0f}")
    for v, name in ((seg_dims[0], "upper"), (seg_dims[1], "fore")):
        assert np.degrees(np.arccos(-unit(v)[1])) < 0.01, f"{name} rest segment not straight down"

    # rig_mapping's own chain reproduces config2quats_multi
    q, rq = data["pinn"]
    ref = config2quats_multi(np.radians(q), prev_rot=R.from_quat(rq))
    dev = max(np.degrees((R.from_quat(a).inv() * b).magnitude()).max() for a, b in zip(ref, segment_rotations(q, rq)))
    print(f"rig_mapping chain vs config2quats_multi: max {dev:.1e} deg")

    truth = {m: true_points(*data[m], seg_dims) for m in METHODS}
    ren = {m: rendered_points(*data[m], rig) for m in METHODS}
    gt_r, gt_t = ren["groundtruth"], truth["groundtruth"]
    fid = [np.degrees(np.arccos(np.clip(np.sum(unit(gt_r[0]) * unit(gt_t[0]), 1), -1, 1))),
           np.degrees(np.arccos(np.clip(np.sum(unit(gt_r[1] - gt_r[0]) * unit(gt_t[1] - gt_t[0]), 1), -1, 1)))]
    print(f"\nrecorded pose rendering, segment-direction error (deg): "
          f"upper {np.round(fid[0], 1)}, fore {np.round(fid[1], 1)}")
    print("   vs. recorded pose, per frame (initial/center/final):  RIG render  |  TRUE geometry")
    for m in METHODS[1:]:
        d = lambda a, b, k: np.linalg.norm(a[k] - b[k], axis=1).round(0).astype(int).tolist()
        ang = lambda a, b: np.degrees((a[2].inv() * b[2]).magnitude()).round(0).astype(int).tolist()
        print(f"   {m:9s} elbow {d(ren[m], gt_r, 0)} wrist {d(ren[m], gt_r, 1)} hand {ang(ren[m], gt_r)} deg"
              f"  |  elbow {d(truth[m], gt_t, 0)} wrist {d(truth[m], gt_t, 1)} hand {ang(truth[m], gt_t)} deg")


if __name__ == "__main__":
    main()
