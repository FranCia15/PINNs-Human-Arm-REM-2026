"""
dataset.py

Reads the DBAS22 / 3D-ARM-Gaze reaching data of one subject: per trial
(one valid target reach), the recorded hand pose (endEffVirtPos,
endEffVirtQuat), the arm-root pose, and the 7 joint angles (sp, sr, ay, ep,
fy, hp, hr) computed from the recorded segment orientations with
quats2config (tool_box.rot_quat_utils); and the subject's anthropometry.

Unity's world frame is left-handed, while quats2config (scipy rotations)
assumes right-handed rotations: all world-frame quaternions go through
change_quat_ref_handeness first, and the shoulder angles are relative to
the arm root (armRootQuat), not to the world.
"""

import json
from pathlib import Path
from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation as R

# Authors' conversion utilities (HYBRID Team INCIA, CNRS / Univ. Bordeaux,
# Apache 2.0 licensed).
from tool_box.rot_quat_utils import quats2config, change_quat_ref_handeness


DATASET_ROOT = Path(__file__).resolve().parents[1] / "DBAS22_DataOnline"  # one folder per subject (s1 ... s20)
PRIMARY_PHASE = "initial_acquisition"     # acquisition phase used throughout


# ---------------------------------------------------------------------------
# Per-subject anthropometry
# ---------------------------------------------------------------------------

@dataclass
class SubjectInfo:
    subject_id: str
    arm_side: str
    u_arm: float          # upper-arm length, m
    f_arm: float          # forearm length, m
    height: float         # "size" field, m


def load_subject_info(subject_dir: Path) -> SubjectInfo:
    with open(subject_dir / "subj_info.json") as f:
        info = json.load(f)

    return SubjectInfo(
        subject_id=subject_dir.name,
        arm_side=info["armSide"],
        u_arm=info["uArm"],
        f_arm=info["fArm"],
        height=info["size"],
    )


# ---------------------------------------------------------------------------
# Exclusion handling
# ---------------------------------------------------------------------------

def _flatten(x, out: list) -> None:
    """Recursively flattens arbitrarily nested lists into a flat list."""
    if isinstance(x, list):
        for item in x:
            _flatten(item, out)
    else:
        out.append(x)


def load_exclusions(subject_dir: Path, phase: str) -> dict:
    """
    Loads the three exclusion files for a phase.

    File contents:
      exclude_times_PHASE.json   -> {"indices": [flat, alternating start/end...]}
      exclude_indexs_PHASE.json  -> {"indexs": [int positions...], "timestamps": [float...]}
      exclude_targets_PHASE.json -> {"targets": [int tgtNumbers...]}

    Returns dict with:
      - excluded_targets: set of tgtNumber to drop entirely
      - excluded_time_ranges: list of (start, end) second pairs, end exclusive
      - excluded_positions: set of integer positions in the RAW samples
        list (before any filtering) to drop -- matched by list position,
        not by float timestamp, since float equality is fragile.
    """
    def _load(path):
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return {}

    times_data = _load(subject_dir / f"exclude_times_{phase}.json")
    indexs_data = _load(subject_dir / f"exclude_indexs_{phase}.json")
    targets_data = _load(subject_dir / f"exclude_targets_{phase}.json")

    flat: list = []
    _flatten(times_data.get("indices", []), flat)
    if len(flat) % 2 != 0:
        raise ValueError(
            f"exclude_times_{phase}.json: odd number of values ({len(flat)}) "
            f"after flattening -- cannot pair into (start, end) ranges."
        )
    excluded_time_ranges = list(zip(flat[0::2], flat[1::2]))

    return {
        "excluded_targets": set(targets_data.get("targets", [])),
        "excluded_time_ranges": excluded_time_ranges,
        "excluded_positions": set(indexs_data.get("indexs", [])),
    }


def is_excluded(position: int, timestamp: float, tgt_number: int,
                 exclusions: dict) -> bool:
    """
    position: index of this frame in the RAW (unfiltered) samples list --
    must match the indexing exclude_indexs_PHASE.json was built against.
    """
    if tgt_number in exclusions["excluded_targets"]:
        return True
    if position in exclusions["excluded_positions"]:
        return True
    for start, end in exclusions["excluded_time_ranges"]:
        if start <= timestamp < end:
            return True
    return False


# ---------------------------------------------------------------------------
# Quaternion -> 7DOF joint angles
# ---------------------------------------------------------------------------

def quats_to_7dof(quats_shou: np.ndarray, quats_elb: np.ndarray,
                   quats_wri: np.ndarray, quats_arm_root: np.ndarray,
                   degrees: bool = False) -> np.ndarray:
    """
    Converts a trial's worth of shouVirtQuat / elbVirtQuat / wriVirtQuat
    into [sp, sr, ay, ep, fy, hp, hr] per frame, using the authors'
    quats2config(), with shoulder angles computed relative to the
    trunk-referenced armRoot frame rather than world.

    World-frame quaternions are first made right-handed (module docstring).

    Parameters
    ----------
    quats_shou, quats_elb, quats_wri : (T, 4) arrays
        shouVirtQuat, elbVirtQuat, wriVirtQuat for the trial, world frame.
    quats_arm_root : (T, 4) array
        armRootQuat for the trial -- the trunk-referenced frame.
    degrees : bool
        Output in degrees instead of radians.

    Returns
    -------
    (T, 7) array: sp, sr, ay, ep, fy, hp, hr
    """
    quats_shou = change_quat_ref_handeness(np.array(quats_shou, copy=True))
    quats_elb = change_quat_ref_handeness(np.array(quats_elb, copy=True))
    quats_wri = change_quat_ref_handeness(np.array(quats_wri, copy=True))
    quats_arm_root = change_quat_ref_handeness(np.array(quats_arm_root, copy=True))

    rot_ref = R.from_quat(quats_arm_root)
    return quats2config(quats_shou, quats_elb, quats_wri,
                         rot_ref=rot_ref, degrees=degrees)


JOINT_COLS = ("sp", "sr", "ay", "ep", "fy", "hp", "hr")


def compute_functional_joint_limits(trials: list[dict], lo_pct: float = 1.0,
                                     hi_pct: float = 99.0) -> dict:
    """
    Joint limits of the training loss: the lo_pct-hi_pct percentiles of the
    subject's recorded joint angles (natural reaching often exceeds the
    dataset's calibration range of motion; percentiles, so that a single
    glitch frame does not set a bound). Returns {col: (lo, hi)}.
    """
    all_angles = np.concatenate([t["joint_angles"] for t in trials], axis=0)
    limits = {}
    for i, col in enumerate(JOINT_COLS):
        col_angles = all_angles[:, i]
        limits[col] = (
            np.percentile(col_angles, lo_pct),
            np.percentile(col_angles, hi_pct),
        )
    return limits


# ---------------------------------------------------------------------------
# Trial extraction
# ---------------------------------------------------------------------------

NEEDED_FIELDS = (
    "timestamp", "tgtNumber", "tgtType", "expeInPause",
    "endEffVirtPos", "endEffVirtQuat",
    "shouVirtQuat", "elbVirtQuat", "wriVirtQuat",
    "armRootPos", "armRootQuat",
)


def _slim_frame(frame: dict) -> dict:
    """Keeps only the fields the pipeline actually uses, dropping eyes,
    cubes, gaze, custom-arm duplicates, trunk/neck/opposite-shoulder, etc."""
    return {k: frame[k] for k in NEEDED_FIELDS}


def _positions_to_drop_after_pause(raw_samples: list[dict]) -> set[int]:
    """
    Positions (RAW list indices) of the 2 samples following the end of
    each pause block: per the dataset documentation, "the two samples
    following a pause should be considered as part of the pause", and
    expeInPause flags only the pause frames themselves.
    """
    to_drop = set()
    countdown = 0
    for position, raw_s in enumerate(raw_samples):
        if raw_s.get("expeInPause", False):
            countdown = 2
            continue
        if countdown > 0:
            to_drop.add(position)
            countdown -= 1
    return to_drop


def extract_trials(subject_dir: Path, phase: str, exclusions: dict) -> list[dict]:
    """
    Returns a list of trial dicts, one per validated target reach.
    Frames flagged in exclusions (by RAW list position), frames where
    expeInPause is True, and the 2 frames following each pause, are
    dropped, as is the last sample of the file (dataset documentation).
    """
    with open(subject_dir / f"{phase}.json") as f:
        data = json.load(f)
    raw_samples = data["samples"]
    post_pause_drop = _positions_to_drop_after_pause(raw_samples)

    trials_by_target: dict[int, list[dict]] = {}
    # Enumerate over the RAW list (before dropping anything) so that
    # `position` matches what exclude_indexs_PHASE.json was built against.
    n = len(raw_samples)
    for position, raw_s in enumerate(raw_samples):
        if position == n - 1:
            continue  # documented: drop the last sample of the file
        s = _slim_frame(raw_s)
        if s.get("expeInPause", False):
            continue
        if position in post_pause_drop:
            continue
        ts = s["timestamp"]
        tgt = s["tgtNumber"]
        if is_excluded(position, ts, tgt, exclusions):
            continue
        trials_by_target.setdefault(tgt, []).append(s)

    trials = []
    for tgt, frames in trials_by_target.items():
        if not frames:
            continue
        timestamps = np.array([f["timestamp"] for f in frames])
        hand_pos = np.array([[f["endEffVirtPos"]["x"],
                               f["endEffVirtPos"]["y"],
                               f["endEffVirtPos"]["z"]] for f in frames])
        hand_quat = np.array([[f["endEffVirtQuat"]["x"],
                                f["endEffVirtQuat"]["y"],
                                f["endEffVirtQuat"]["z"],
                                f["endEffVirtQuat"]["w"]] for f in frames])

        def _quat_arr(key):
            return np.array([[f[key]["x"], f[key]["y"],
                               f[key]["z"], f[key]["w"]] for f in frames])

        joint_angles = quats_to_7dof(
            quats_shou=_quat_arr("shouVirtQuat"),
            quats_elb=_quat_arr("elbVirtQuat"),
            quats_wri=_quat_arr("wriVirtQuat"),
            quats_arm_root=_quat_arr("armRootQuat"),
        )

        arm_root_pos = np.array([[f["armRootPos"]["x"],
                                   f["armRootPos"]["y"],
                                   f["armRootPos"]["z"]] for f in frames])

        trials.append({
            "subject_id": subject_dir.name,
            "tgt_number": tgt,
            "timestamps": timestamps,
            "hand_pos": hand_pos,
            "hand_quat": hand_quat,
            "joint_angles": joint_angles,
            "arm_root_pos": arm_root_pos,
            "arm_root_quat": _quat_arr("armRootQuat"),
        })

    return trials

