# -*- coding: utf-8 -*-
"""
capture_poses.py

Poses the mannequin of Simulation_roboDK.rdk for the 12 screenshots of the
pose comparison figure (Fig. 3: recorded pose, IK, dynamics baseline, PINN
at the initial / centre / final frames of the exported poses), with the
mapping of rig_mapping.py, pausing for a manual screenshot after each pose.

Before each screenshot it reads the rendered elbow / wrist / hand back from
RoboDK (PoseAbs) and prints:
  check   distance from the offline prediction (should be ~0 mm / ~0 deg;
          if not, the station differs from rig_geometry.json)
  vs rec  distance from the rendered recorded pose at the same frame
          (expected: IK wrist <= ~20 mm and hand 0 deg -- the residual is the
          rig's generic segment lengths; PINN wrist ~35-55 mm)

Run with RoboDK open, the station loaded, the mannequin intact
(fix_segments.py if needed) and AT REST, from this folder, with any Python
that has robodk, numpy and scipy (pip install robodk numpy scipy):
    python capture_poses.py [subject] [tgt_number]
Defaults: s1, tgt 10. Reads rig_geometry.json and
poses/<subject>_tgt<N>_compare_*.csv (src/export_poses.py).
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
from robolink import Robolink, ITEM_TYPE_FRAME
from robodk.robomath import Mat
from scipy.spatial.transform import Rotation as R

from rig_mapping import GEOMETRY, SEGMENT_FRAMES, REST_FRAMES, load_rig, frame_rotations, rendered_points

TRAJ_DIR = Path(__file__).resolve().parent / "poses"
LABELS = ["initial", "center", "final"]
METHODS = [("recorded", "groundtruth"), ("IK baseline", "ik"),
           ("dynamics baseline", "dynamics"), ("PINN", "pinn")]
JOINTS = ("sp", "sr", "ay", "ep", "fy", "hp", "hr")


def load_csv(path):
    """Exported pose CSV -> joint angles (deg), right-handed root quaternions, times."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return (np.array([[float(r[j]) for j in JOINTS] for r in rows]),
            np.array([[float(r[k]) for k in ("rqx", "rqy", "rqz", "rqw")] for r in rows]),
            [float(r["t"]) for r in rows])


def rot4(m):
    return Mat([[m[0][0], m[0][1], m[0][2], 0.0], [m[1][0], m[1][1], m[1][2], 0.0],
                [m[2][0], m[2][1], m[2][2], 0.0], [0.0, 0.0, 0.0, 1.0]])


def abs_np(fr):
    p = fr.PoseAbs()
    return np.array([[p[i, j] for j in range(4)] for i in range(4)])


def main():
    subject = sys.argv[1] if len(sys.argv) > 1 else "s1"
    tgt = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    rig = load_rig()

    data = {}
    for _, key in METHODS:
        path = TRAJ_DIR / f"{subject}_tgt{tgt}_compare_{key}.csv"
        if not path.exists():
            sys.exit(f"Missing {path} -- run src/export_poses.py first.")
        data[key] = load_csv(path)

    rdk = Robolink()
    print("Station:", rdk.ActiveStation().Name())
    names = SEGMENT_FRAMES + REST_FRAMES
    frames = {n: rdk.Item(n, ITEM_TYPE_FRAME) for n in names}
    missing = [n for n, fr in frames.items() if not fr.Valid()]
    if missing:
        sys.exit(f"Frames not found: {missing}")

    # the mapping assumes the station is at the rest pose measured in rig_geometry.json
    measured = json.loads(Path(GEOMETRY).read_text())["probes"]["rest"]
    for n, fr in frames.items():
        if np.abs(abs_np(fr) - np.array(measured[n])).max() > 1e-3:
            sys.exit(f"'{n}' is not at the measured rest pose -- reopen the station (and run fix_segments.py) first.")
    rest = {n: fr.Pose() for n, fr in frames.items()}
    shoulder = rdk.Item("shoulderR_rotation", ITEM_TYPE_FRAME)

    def apply(q_deg, rq):
        for n, m in frame_rotations(q_deg, rq, rig).items():
            frames[n].setPose(rest[n] * rot4(m))
        rdk.Render(True)
        s = abs_np(shoulder)[:3, 3]
        hand = abs_np(frames["wristR_deviation"])
        return (abs_np(frames["elbowR_flexion"])[:3, 3] - s, abs_np(frames["wristR_flexion"])[:3, 3] - s,
                R.from_matrix(hand[:3, :3]))

    try:
        for i, label in enumerate(LABELS[:len(data["groundtruth"][0])]):
            rec = None
            for method_label, key in METHODS:
                q, rq, t = data[key]
                elbow, wrist, hand = apply(q[i], rq[i])
                pe, pw, ph = rendered_points(q[i:i + 1], rq[i:i + 1], rig)
                check = (np.linalg.norm(elbow - pe[0]), np.linalg.norm(wrist - pw[0]),
                         np.degrees((hand.inv() * ph[0]).magnitude()))
                if rec is None:
                    rec = (elbow, wrist, hand)
                vs = (np.linalg.norm(elbow - rec[0]), np.linalg.norm(wrist - rec[1]),
                      np.degrees((hand.inv() * rec[2]).magnitude()))
                input(f"[{label:7s}] {method_label:17s} t={t[i]:5.2f}s | check {check[0]:.1f}/{check[1]:.1f} mm "
                      f"{check[2]:.1f} deg | vs rec: elbow {vs[0]:4.0f} mm, wrist {vs[1]:4.0f} mm, "
                      f"hand {vs[2]:4.0f} deg -- screenshot, then Enter...")
    finally:
        for n, fr in frames.items():
            fr.setPose(rest[n])
        rdk.Render(True)
        print("Reset to rest.")


if __name__ == "__main__":
    main()
