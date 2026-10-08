# -*- coding: utf-8 -*-
"""
rig_mapping.py

DBAS22 joint angles -> mannequin rig mapping (Sec. VI, "Calibration"),
used by play_trajectory.py, capture_poses.py and src/check_rig_rendering.py.

Measured rig geometry (dump_rig_geometry.py -> rig_geometry.json): every
arm frame has, at rest, the station's own orientation, and the station is
DBAS22's right-handed world (Y up, mannequin facing +Z, right arm at -X).
So instead of mapping individual joint angles onto rig axes, each rig
segment is given the absolute orientation of the corresponding DBAS22
segment (same chain as tool_box.rot_quat_utils.config2quats_multi),
composed with a fixed per-segment rest correction G (minimal rotation
taking DBAS22's rest segment direction, straight down, onto the rig's
measured rest bone):
    A_upper = r_upper G_up^-1,  A_fore = r_fore G_fore^-1,  A_hand = r_hand G_fore^-1
so each rendered bone points exactly along the true segment. The rotation
of each segment is put on one frame per segment (shoulderR_abduction,
elbowR_flexion, wristR_flexion); the other arm frames stay at rest.

Remaining, unavoidable difference: the rig's generic segment lengths
(about 248 / 227 mm) are not the subject's.

Needs only numpy + scipy (available in RoboDK's embedded Python).
"""
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

GEOMETRY = Path(__file__).resolve().parent / "rig_geometry.json"
DOWN = np.array([0.0, -1.0, 0.0])  # DBAS22 rest direction of upper arm and forearm
SEGMENT_FRAMES = ("shoulderR_abduction", "elbowR_flexion", "wristR_flexion")
REST_FRAMES = ("shoulderR_flexion", "shoulderR_rotation", "forearmR_rotation", "wristR_deviation")


def _min_rotation(a, b):
    """Smallest rotation taking unit vector a onto unit vector b."""
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    axis = np.cross(a, b)
    s, c = np.linalg.norm(axis), float(np.dot(a, b))
    return R.identity() if s < 1e-12 else R.from_rotvec(axis / s * np.arctan2(s, c))


def load_rig(path=GEOMETRY):
    """Rest bone vectors (station, mm) and per-segment rest corrections."""
    rest = json.loads(Path(path).read_text())["probes"]["rest"]
    pos = lambda n: np.array(rest[n])[:3, 3]
    for n, pose in rest.items():
        assert np.allclose(np.array(pose)[:3, :3], np.eye(3), atol=1e-6), f"{n} not station-aligned at rest"
    b_up = pos("elbowR_flexion") - pos("shoulderR_rotation")
    b_fore = pos("wristR_flexion") - pos("forearmR_rotation")
    return {"b_up": b_up, "b_fore": b_fore, "G_up": _min_rotation(DOWN, b_up), "G_fore": _min_rotation(DOWN, b_fore)}


def segment_rotations(q_deg, root_quat):
    """DBAS22 absolute segment orientations; q_deg (N, 7) sp..hr in degrees, root_quat (N, 4) right-handed."""
    q = np.atleast_2d(q_deg)
    r_up = R.from_quat(np.atleast_2d(root_quat)) * R.from_euler("XZY", q[:, 0:3], degrees=True)
    r_fore = r_up * R.from_euler("X", q[:, 3:4], degrees=True)
    r_hand = r_fore * R.from_euler("YXZ", q[:, 4:7], degrees=True)
    return r_up, r_fore, r_hand


def rig_absolute(q_deg, root_quat, rig):
    """Absolute rig orientations (upper arm, forearm, hand)."""
    r_up, r_fore, r_hand = segment_rotations(q_deg, root_quat)
    return r_up * rig["G_up"].inv(), r_fore * rig["G_fore"].inv(), r_hand * rig["G_fore"].inv()


def frame_rotations(q_deg, root_quat, rig):
    """Local rotation (3x3) to apply on top of each arm frame's rest pose, for ONE frame of data."""
    a_up, a_fore, a_hand = rig_absolute(q_deg, root_quat, rig)
    local = {"shoulderR_abduction": a_up, "elbowR_flexion": a_up.inv() * a_fore,
             "wristR_flexion": a_fore.inv() * a_hand}
    out = {n: r.as_matrix()[0] for n, r in local.items()}
    out.update({n: np.eye(3) for n in REST_FRAMES})
    return out


def rendered_points(q_deg, root_quat, rig):
    """Rendered elbow and wrist relative to the shoulder (station mm) and hand orientation."""
    a_up, a_fore, a_hand = rig_absolute(q_deg, root_quat, rig)
    elbow = a_up.apply(rig["b_up"])
    return elbow, elbow + a_fore.apply(rig["b_fore"]), a_hand
