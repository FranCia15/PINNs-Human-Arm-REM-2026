"""
baseline_ik.py

IK baseline (current practice): the swivel angle (arm yaw, ay) is fixed to
a constant posture value, and the other 6 DOF are solved by numerical
inverse kinematics against the recorded hand pose (3 position + 3
orientation constraints per frame). Solved by a batched Levenberg-Marquardt
iteration over all frames at once, with kinematics.forward_kinematics as
the forward model.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R

from dataset import JOINT_COLS
from kinematics import forward_kinematics

FREE_COLS = [c for c in JOINT_COLS if c != "ay"]  # sp, sr, ep, fy, hp, hr
AY_INDEX = JOINT_COLS.index("ay")


def compute_fixed_posture(trials: list) -> np.ndarray:
    """
    Per-joint median of the joint angles over the given trials: ay is held
    at its value (the fixed posture), the other joints start from it.
    """
    all_angles = np.concatenate([t["joint_angles"] for t in trials], axis=0)
    return np.median(all_angles, axis=0)  # (7,)


def pack_joint_angles(q_free: np.ndarray, ay_fixed: float) -> np.ndarray:
    """Reinserts the fixed ay column into a (T, 6) free-DOF array -> (T, 7)."""
    n = q_free.shape[0]
    q = np.empty((n, 7))
    q[:, AY_INDEX] = ay_fixed
    free_idx = [i for i in range(7) if i != AY_INDEX]
    q[:, free_idx] = q_free
    return q


def _residual(q_free: np.ndarray, ay_fixed: float, arm_root_pos: np.ndarray,
              arm_root_quat: np.ndarray, seg_dims: np.ndarray,
              quat_wri2arrow: np.ndarray, hand_pos_target: np.ndarray,
              hand_quat_target: np.ndarray) -> np.ndarray:
    q = pack_joint_angles(q_free, ay_fixed)
    hand_pos_pred, hand_quat_pred = forward_kinematics(
        q, arm_root_pos, arm_root_quat, seg_dims, quat_wri2arrow
    )
    pos_res = hand_pos_pred - hand_pos_target
    r_err = R.from_quat(hand_quat_pred) * R.from_quat(hand_quat_target).inv()
    rot_res = r_err.as_rotvec()
    return np.concatenate([pos_res, rot_res], axis=1)  # (T, 6)


def solve_ik_batch(hand_pos_target: np.ndarray, hand_quat_target: np.ndarray,
                    arm_root_pos: np.ndarray, arm_root_quat: np.ndarray,
                    seg_dims: np.ndarray, quat_wri2arrow: np.ndarray,
                    ay_fixed: float, q0_free: np.ndarray,
                    n_iters: int = 30, fd_eps: float = 1e-5,
                    damping: float = 1e-3, max_step: float = 0.2):
    """
    Batched Levenberg-Marquardt over all frames, with a finite-difference
    Jacobian. The damping is scaled by diag(J^T J) (Marquardt), and each
    update is clipped to max_step per joint, so that frames the fixed ay
    cannot reach end with a residual instead of diverging angles.

    Returns
    -------
    q_free : (T, 6) sp, sr, ep, fy, hp, hr, rad
    final_residual : (T, 6) hand residual (position m, rotation vector rad)
    """
    T = hand_pos_target.shape[0]
    q_free = np.tile(q0_free, (T, 1)).astype(float)

    args = (ay_fixed, arm_root_pos, arm_root_quat, seg_dims, quat_wri2arrow,
            hand_pos_target, hand_quat_target)

    for _ in range(n_iters):
        res0 = _residual(q_free, *args)
        J = np.empty((T, 6, 6))
        for k in range(6):
            dq = np.zeros_like(q_free)
            dq[:, k] = fd_eps
            res_p = _residual(q_free + dq, *args)
            J[:, :, k] = (res_p - res0) / fd_eps

        JT = np.transpose(J, (0, 2, 1))
        JTJ = JT @ J
        JTr = (JT @ res0[..., None])[..., 0]
        diag = np.eye(6)[None, :, :] * JTJ * np.eye(6)[None, :, :]
        A = JTJ + damping * diag
        # explicit trailing axis: numpy>=2 no longer treats a (T, 6) b as
        # stacked vectors (same result as numpy 1.x's implicit behavior)
        delta = np.linalg.solve(A, -JTr[..., None])[..., 0]
        delta = np.clip(delta, -max_step, max_step)
        q_free = q_free + delta

    final_residual = _residual(q_free, *args)
    return q_free, final_residual
