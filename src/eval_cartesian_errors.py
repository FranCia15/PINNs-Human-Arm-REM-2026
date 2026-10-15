"""
eval_cartesian_errors.py

Cartesian errors of the elbow, wrist and hand, hand orientation error and
swivel-angle error of the PINN, the IK baseline and the dynamics baseline,
on the Table I trial sets (common.SCENARIOS), every frame.

Joint positions come from the forward-kinematics chain (arm root ->
shou2elb -> elb2wri -> wri2arrow, as in kinematics.py) applied to each
method's joint angles and to the recorded ones; errors are Euclidean
distances to the latter (mm), plus the hand orientation error (deg) and
the swivel-angle error (deg): the elbow's rotation about each method's own
shoulder-wrist axis, measured from the downward vertical, which isolates
the redundant DOF independently of how well the hand itself is tracked.

Input:  results/joint_errors_paper_model.npz (PINN and IK angles),
        results/dynamics_<scenario>.npz (dynamics-baseline angles)
Output: results/cartesian_errors.npz with <scenario>_<method>_<point>
        (N_frames,) errors (subjects and trials in common.SCENARIOS order)
        and <scenario>_lens (frames per trial); printed summary.

Usage (from the repository root): python src/eval_cartesian_errors.py
"""

import numpy as np
from scipy.spatial.transform import Rotation as R

from common import PAPER_MODEL, RESULTS, SCENARIOS, DATASET_ROOT, model_config, scenario_trials
from kinematics import load_segment_constants
from tool_box.rot_quat_utils import config2quats_multi, change_quat_ref_handeness

METHODS = ("pinn", "ik", "dyn")
POINTS = ("elbow", "wrist", "hand", "hand_ori", "swivel")


def swivel(shoulder, elbow, wrist):
    """Swivel angle (rad): elbow direction about the shoulder-wrist axis, from the downward vertical (-y)."""
    u = wrist - shoulder
    u = u / np.linalg.norm(u, axis=1, keepdims=True)
    perp = lambda v: v - np.sum(v * u, axis=1, keepdims=True) * u
    e, g = perp(elbow - shoulder), perp(np.broadcast_to([0.0, -1.0, 0.0], u.shape))
    return np.arctan2(np.sum(np.cross(g, e) * u, axis=1), np.sum(g * e, axis=1))


def arm_points(q, arm_root_pos, arm_root_quat, seg_dims, quat_wri2arrow):
    """(T, 7) joint angles (rad) -> elbow, wrist, hand positions (T, 3), hand rotation, swivel angle."""
    shou2elb, elb2wri, wri2arrow = seg_dims
    rot_ref = R.from_quat(change_quat_ref_handeness(np.array(arm_root_quat, copy=True)))
    q_up, q_fore, q_hand = config2quats_multi(np.asarray(q), prev_rot=rot_ref)
    root = np.array(arm_root_pos, copy=True)
    root[:, 0] *= -1  # right-handed frame, as in kinematics.forward_kinematics
    elbow = root + R.from_quat(q_up).apply(shou2elb)
    wrist = elbow + R.from_quat(q_fore).apply(elb2wri)
    r_hand = R.from_quat(q_hand)
    hand = wrist + r_hand.apply(wri2arrow)
    return elbow, wrist, hand, r_hand * R.from_quat(np.tile(quat_wri2arrow, (len(q), 1))), swivel(root, elbow, wrist)


def main():
    counts = model_config()["subject_trial_counts"]
    je = np.load(RESULTS / f"joint_errors_{PAPER_MODEL}.npz")
    out = {}
    for sc, subjects in SCENARIOS.items():
        dyn = np.load(RESULTS / f"dynamics_{sc}.npz")
        acc = {(m, p): [] for m in METHODS for p in POINTS}
        lens = []
        for s in subjects:
            trials = scenario_trials(sc, s, counts)
            assert [t["tgt_number"] for t in trials] == je[f"{sc}_{s}_tgts"].tolist() == dyn[f"{s}_tgts"].tolist()
            seg_dims, quat_wri2arrow, _ = load_segment_constants(DATASET_ROOT / s)
            cat = lambda k: np.concatenate([t[k] for t in trials])
            root_p, root_q = cat("arm_root_pos"), cat("arm_root_quat")
            q = {"pinn": je[f"{sc}_{s}_q_pinn"], "ik": je[f"{sc}_{s}_q_ik"], "dyn": dyn[f"{s}_q"]}
            gt = arm_points(cat("joint_angles"), root_p, root_q, seg_dims, quat_wri2arrow)
            for m in METHODS:
                pts = arm_points(q[m], root_p, root_q, seg_dims, quat_wri2arrow)
                for i, p in enumerate(POINTS[:3]):
                    acc[(m, p)].append(np.linalg.norm(pts[i] - gt[i], axis=1) * 1e3)
                acc[(m, "hand_ori")].append(np.degrees((pts[3].inv() * gt[3]).magnitude()))
                acc[(m, "swivel")].append(np.degrees(np.abs(np.angle(np.exp(1j * (pts[4] - gt[4]))))))
            lens.append(je[f"{sc}_{s}_lens"])
        out[f"{sc}_lens"] = np.concatenate(lens)
        print(f"\n=== {sc} ({', '.join(subjects)}): mean ± std (median); elbow/wrist/hand in mm, angles in deg")
        print(f"{'':6s}" + "".join(f"{p:>24s}" for p in POINTS))
        for m in METHODS:
            row = ""
            for p in POINTS:
                e = np.concatenate(acc[(m, p)])
                out[f"{sc}_{m}_{p}"] = e
                row += f"{e.mean():9.1f} ±{e.std():5.1f} (med{np.median(e):6.1f})"
            print(f"{m:6s}{row}")
    out_path = RESULTS / "cartesian_errors.npz"
    np.savez(out_path, **out)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
