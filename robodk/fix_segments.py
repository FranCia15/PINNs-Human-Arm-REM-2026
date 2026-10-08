# -*- coding: utf-8 -*-
"""
fix_segments.py

Repairs the mannequin of Simulation_roboDK.rdk, whose segments come apart
when the station is reopened, by re-attaching each segment to its frame.

Cause: for the mannequin's sub-solids (item type 7), the .rdk file saves
the original raw points but not the compensation set by setParentStatic
when the mannequin was rigged; on opening, RoboDK applies the parent
frame's transformation to the raw points, i.e. a second time. The frames
themselves are saved correctly.

Fix, per segment:
  1) store each frame's relative pose and set the frame to its build state
     (translation only, identity rotation);
  2) setParent(station), keeping the relative pose (the segment returns to
     its raw, imported position), then setParentStatic(frame), keeping the
     absolute pose (the compensation is set again, as when rigging);
  3) restore the stored frame poses; the geometry follows the frames.

Run once after each opening of the station (saving loses the fix), with
RoboDK open and rendering active: inside Render(False) the re-attachment
has no effect. Not idempotent: on an intact station it would break the
mannequin, so a check on a trunk segment stops the script if the station
is intact (--force skips the check).

Usage (from this folder):
    python fix_segments.py [--skip name1,name2] [--force]
    --skip   segments to leave untouched
    --force  skip the check
"""
import math
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from robolink import Robolink, ITEM_TYPE_FRAME
from robodk.robomath import transl

# import anchors (mm) of the trunk segments, for the check
TRUNK_ANCHORS = {"tete": (20, 1584, 30), "torso": (15, 1260, 0), "bassin": (0, 958, 0)}

skip = set()
if "--skip" in sys.argv:
    skip = set(sys.argv[sys.argv.index("--skip") + 1].split(","))

t0 = time.time()
RDK = Robolink()
station = RDK.ActiveStation()
print("Station:", station.Name())


def _d(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def station_is_broken():
    """True if the mannequin has come apart (double transformation)."""
    for it in RDK.ItemList():
        if it.Type() != 7:
            continue
        part = it.Name().split("-", 1)[-1]
        if part not in TRUNK_ANCHORS:
            continue
        bb = it.setParam("BoundingBox")
        c = [(bb["min"][i] + bb["max"][i]) / 2.0 for i in range(3)]
        a = TRUNK_ANCHORS[part]
        F = it.Parent().PoseAbs().rows
        fa = [sum(F[i][j] * a[j] for j in range(3)) + F[i][3] for i in range(3)]
        d_intact, d_broken = _d(c, a), _d(c, fa)
        state = "broken (double transformation)" if d_broken < d_intact else "intact"
        print(f"Check ({part}): {d_intact:.0f} mm from intact, {d_broken:.0f} mm from broken -> station {state}")
        return d_broken < d_intact
    sys.exit("Check impossible: no trunk segment (tete/torso/bassin) found.")


if "--force" not in sys.argv and not station_is_broken():
    sys.exit("Nothing to do: the station is intact (--force skips the check).")

frames = RDK.ItemList(ITEM_TYPE_FRAME)
solids = [it for it in RDK.ItemList() if it.Type() == 7]
if not solids:
    sys.exit("Nothing to repair: no type-7 sub-solid in the station.")
parent_of = {}
for s in solids:
    nm = s.Name()
    if nm in skip or nm.split("-", 1)[-1] in skip:
        print(f"  (skipped: {nm})")
        continue
    p = s.Parent()
    if p.Type() == 3:
        parent_of[s] = p
    else:
        print(f"  (ignored: {nm} is not under a frame)")
print(f"{len(frames)} frames, {len(parent_of)} segments to re-attach")

# 1) frames to their build state (translation only)
saved = []
for fr in frames:
    pose = fr.Pose()
    saved.append((fr, pose))
    r = pose.rows
    fr.setPose(transl(r[0][3], r[1][3], r[2][3]))

# 2) detach keeping the relative pose, re-attach keeping the absolute pose
for s, fr in parent_of.items():
    s.setParent(station)
    s.setParentStatic(fr)
    print(f"  re-attached: {s.Name():26s} -> {fr.Name()}")

# 3) restore the frame poses
for fr, pose in saved:
    fr.setPose(pose)

print(f"Done: {len(parent_of)} segments re-attached in {time.time() - t0:.1f} s")
