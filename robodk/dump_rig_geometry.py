# -*- coding: utf-8 -*-
"""
dump_rig_geometry.py

Non-interactive dump of the mannequin rig's geometry (rig_geometry.json),
from which rig_mapping.py takes the rest bones of the rig, and with which
src/check_rig_rendering.py checks the rendering offline. Does not change the
station: every frame touched is restored to its saved pose.

Writes rig_geometry.json next to this script with:
  items   every frame/object in the station: name, type, parent, relative
          pose (Pose) and absolute pose (PoseAbs), 4x4, mm
  probes  for each arm joint frame and each local axis x/y/z: absolute
          poses (4x4) of all arm frames after a +20 deg rotation about that
          axis (rest values under "rest"), which give each joint's
          rotation axis and sign, and the bone offsets

Run with RoboDK open, Simulation_roboDK.rdk loaded, the mannequin intact
(fix_segments.py if needed) and at rest, from this folder:
    python dump_rig_geometry.py
"""
import json
import math
from pathlib import Path

from robolink import Robolink, ITEM_TYPE_FRAME, ITEM_TYPE_OBJECT
from robodk.robomath import rotx, roty, rotz

ARM_FRAMES = ["shoulderR_abduction", "shoulderR_flexion", "shoulderR_rotation", "elbowR_flexion",
              "forearmR_rotation", "wristR_flexion", "wristR_deviation"]
ROT = {"x": rotx, "y": roty, "z": rotz}
STEP_DEG = 20.0
OUT = Path(__file__).resolve().parent / "rig_geometry.json"


def mat(pose):
    return [[float(pose[i, j]) for j in range(4)] for i in range(4)]


def main():
    rdk = Robolink()
    station = rdk.ActiveStation()
    print("Station:", station.Name())

    items = []
    for it in rdk.ItemList(filter=-1):
        if it.Type() not in (ITEM_TYPE_FRAME, ITEM_TYPE_OBJECT):
            continue
        parent = it.Parent()
        items.append({"name": it.Name(), "type": "frame" if it.Type() == ITEM_TYPE_FRAME else "object",
                      "parent": parent.Name() if parent.Valid() else None,
                      "pose": mat(it.Pose()), "pose_abs": mat(it.PoseAbs())})
    print(f"{len(items)} frames/objects")

    frames = {}
    for name in ARM_FRAMES:
        fr = rdk.Item(name, ITEM_TYPE_FRAME)
        if not fr.Valid():
            raise SystemExit(f"frame '{name}' not found -- is the right station open?")
        frames[name] = fr
    rest = {n: fr.Pose() for n, fr in frames.items()}
    poses_abs = lambda: {n: mat(fr.PoseAbs()) for n, fr in frames.items()}

    probes = {"rest": poses_abs()}
    try:
        for name, fr in frames.items():
            for axis in ("x", "y", "z"):
                fr.setPose(rest[name] * ROT[axis](math.radians(STEP_DEG)))
                probes[f"{name}:{axis}"] = poses_abs()
                fr.setPose(rest[name])
    finally:
        for n, fr in frames.items():
            fr.setPose(rest[n])
        rdk.Render(True)

    OUT.write_text(json.dumps({"station": station.Name(), "step_deg": STEP_DEG,
                               "arm_frames": ARM_FRAMES, "items": items, "probes": probes}, indent=1))
    print(f"Saved {OUT}; all arm frames restored to rest.")


if __name__ == "__main__":
    main()
