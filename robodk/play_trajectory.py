# -*- coding: utf-8 -*-
"""
play_trajectory.py

Plays an exported joint trajectory (src/export_trajectory.py) on the
mannequin of Simulation_roboDK.rdk (Sec. VI): each frame's joint angles are
applied through the rig mapping of rig_mapping.py, and frames are paced by
their own recorded timestamps, so the mannequin moves at the trajectory's
natural speed (the actual rate is capped by RoboDK's rendering time).

Run with RoboDK open, the station loaded, the mannequin intact
(fix_segments.py if needed) and at rest, from this folder, with any Python
that has robodk, numpy and scipy (pip install robodk numpy scipy):
    python play_trajectory.py trajectories/s1_tgt10_pinn.csv
The arm is reset to rest at the end (also if interrupted).
"""
import csv
import sys
import time

import numpy as np
from robolink import Robolink, ITEM_TYPE_FRAME
from robodk.robomath import Mat

from rig_mapping import SEGMENT_FRAMES, REST_FRAMES, load_rig, frame_rotations

JOINTS = ("sp", "sr", "ay", "ep", "fy", "hp", "hr")


def rot4(m):
    return Mat([[m[0][0], m[0][1], m[0][2], 0.0], [m[1][0], m[1][1], m[1][2], 0.0],
                [m[2][0], m[2][1], m[2][2], 0.0], [0.0, 0.0, 0.0, 1.0]])


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    with open(sys.argv[1], newline="") as f:
        rows = list(csv.DictReader(f))
    t = np.array([float(r["t"]) for r in rows])
    q = np.array([[float(r[j]) for j in JOINTS] for r in rows])
    rq = np.array([[float(r[k]) for k in ("rqx", "rqy", "rqz", "rqw")] for r in rows])
    rig = load_rig()

    rdk = Robolink()
    frames = {n: rdk.Item(n, ITEM_TYPE_FRAME) for n in SEGMENT_FRAMES + REST_FRAMES}
    missing = [n for n, fr in frames.items() if not fr.Valid()]
    if missing:
        sys.exit(f"Frames not found: {missing}")
    rest = {n: fr.Pose() for n, fr in frames.items()}
    print(f"Playing {len(t)} frames, {t[-1]:.2f} s")
    try:
        start = time.perf_counter()
        for i in range(len(t)):
            for n, m in frame_rotations(q[i], rq[i], rig).items():
                frames[n].setPose(rest[n] * rot4(m))
            rdk.Render(True)
            wait = t[i] - (time.perf_counter() - start)
            if wait > 0:
                time.sleep(wait)
        print(f"Done in {time.perf_counter() - start:.2f} s (recorded {t[-1]:.2f} s)")
    finally:
        for n, fr in frames.items():
            fr.setPose(rest[n])
        rdk.Render(True)


if __name__ == "__main__":
    main()
