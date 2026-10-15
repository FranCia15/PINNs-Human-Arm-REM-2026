"""
kinematics.py

Forward kinematics of the 7-DOF arm model of DBAS22: joint angles (sp, sr,
ay, ep, fy, hp, hr) and the subject's segment constants -> hand
(end-effector) position and orientation in Unity's world frame, as the
recorded endEffVirtPos / endEffVirtQuat. Built on config2quats_multi of
tool_box.rot_quat_utils (DBAS22 authors' code).
"""

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from tool_box.rot_quat_utils import config2quats_multi, change_quat_ref_handeness


SEG_DIM_KEYS = ("shou2elb", "elb2wri", "wri2arrow")


def load_segment_constants(subject_dir: Path, change_ref_handeness: bool = True):
    """
    Reads shou2elb / elb2wri / wri2arrow / quatWri2Arrow / armSide from
    subj_info.json, expressed by default in a right-handed frame.

    Returns
    -------
    seg_dims : (3, 3) array -- [shou2elb, elb2wri, wri2arrow] vectors
    quat_wri2arrow : (4,) array -- constant wrist-to-endEffector rotation offset
    arm_side : str
    """
    with open(subject_dir / "subj_info.json") as f:
        info = json.load(f)

    seg_dims = np.array([list(info[key].values()) for key in SEG_DIM_KEYS])
    quat_wri2arrow = np.atleast_2d(list(info["quatWri2Arrow"].values()))
    arm_side = info.get("armSide", "R")

    if change_ref_handeness ^ (arm_side == "L"):
        quat_wri2arrow = change_quat_ref_handeness(quat_wri2arrow)[0]
        seg_dims = seg_dims.copy()
        seg_dims[:, 0] *= -1  # express segment dimensions in a right-handed frame
    else:
        quat_wri2arrow = quat_wri2arrow[0]

    return seg_dims, quat_wri2arrow, arm_side


def forward_kinematics(joint_angles: np.ndarray, arm_root_pos: np.ndarray,
                        arm_root_quat: np.ndarray, seg_dims: np.ndarray,
                        quat_wri2arrow: np.ndarray) -> tuple:
    """
    joint_angles -> (hand_pos, hand_quat), in Unity's native (left-handed)
    world frame, matching endEffVirtPos / endEffVirtQuat.

    Parameters
    ----------
    joint_angles : (T, 7) array -- sp, sr, ay, ep, fy, hp, hr, radians
    arm_root_pos : (T, 3) array -- armRootPos, world frame, metres
    arm_root_quat : (T, 4) array -- armRootQuat, world frame (left-handed)
    seg_dims : (3, 3) array -- from load_segment_constants
    quat_wri2arrow : (4,) array -- from load_segment_constants

    Returns
    -------
    hand_pos : (T, 3) array
    hand_quat : (T, 4) array
    """
    shou2elb, elb2wri, wri2arrow = seg_dims

    rot_ref = R.from_quat(change_quat_ref_handeness(np.array(arm_root_quat, copy=True)))
    quats_upper, quats_fore, quats_hand = config2quats_multi(
        np.asarray(joint_angles), prev_rot=rot_ref
    )

    r_upper = R.from_quat(quats_upper)
    r_fore = R.from_quat(quats_fore)
    r_hand = R.from_quat(quats_hand)

    arm_root_pos_rh = np.array(arm_root_pos, copy=True)
    arm_root_pos_rh[:, 0] *= -1

    hand_pos_rh = (
        arm_root_pos_rh
        + r_upper.apply(shou2elb)
        + r_fore.apply(elb2wri)
        + r_hand.apply(wri2arrow)
    )
    hand_pos = hand_pos_rh.copy()
    hand_pos[:, 0] *= -1

    n_frames = len(joint_angles)
    r_arrow_offset = R.from_quat(np.tile(quat_wri2arrow, (n_frames, 1)))
    hand_quat_rh = (r_hand * r_arrow_offset).as_quat()
    hand_quat = change_quat_ref_handeness(hand_quat_rh)

    return hand_pos, hand_quat
