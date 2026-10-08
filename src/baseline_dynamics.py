"""
baseline_dynamics.py

Dynamics baseline: resolves the arm's redundancy frame by frame with the
gradient projection method for redundant manipulators (Liegeois 1977),
using the same anthropometry-based dynamics model as the PINN
(dynamics.py):

    q_dot = J+ @ x_dot_target + alpha * n

J+ is the damped pseudo-inverse of the (6,7) hand Jacobian, so the first
term follows the recorded hand velocity; n spans the 1-D null space of
the Jacobian, so alpha changes the arm posture without moving the hand.
At each frame alpha is set by a few Adam steps on

    ||tau(t) - tau(t-1)||^2 + alpha_reg * alpha^2

(torque change, plus a small term that keeps alpha bounded). The
objective is greedy and causal: unlike the PINN's effort term (mean
tau^2 over the whole trajectory), it is not minimized globally.

q is integrated with forward Euler at the trial's recorded timesteps,
from the IK baseline's pose at the first frame (q0, see eval_dynamics.py).
"""

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from dynamics import segment_chain, inverse_dynamics_single
from tool_box.rot_quat_utils import change_quat_ref_handeness

DTYPE = torch.float64


def hand_pose_from_q(q, rot_ref, shou2elb, elb2wri, wri2arrow, quat_wri2arrow_rotmat):
    """q: (7,) torch tensor, differentiable -- see dynamics.segment_chain
    for units/frame convention. Returns (hand_pos (3,), hand_rotmat
    (3,3)), relative to the shoulder (arm_root_pos not added -- add
    externally). com_fracs=1.0 for every segment is the same trick
    train_pinn.py's hand_pose_from_q uses to get the exact end-effector
    point instead of a segment's center of mass."""
    com_fracs = {"upper_arm": 1.0, "forearm": 1.0, "hand": 1.0}
    out = segment_chain(q, rot_ref, shou2elb, elb2wri, wri2arrow, com_fracs)
    hand_pos = out["com_hand"]
    hand_rotmat = out["R_hand"] @ quat_wri2arrow_rotmat
    return hand_pos, hand_rotmat


def geometric_jacobian(q, rot_ref, shou2elb, elb2wri, wri2arrow, quat_wri2arrow_rotmat):
    """(6,7) geometric Jacobian at q: rows 0-2 = d(hand position)/dq,
    rows 3-5 = angular-velocity Jacobian (column i = the world-frame
    angular velocity produced by q_dot_i = 1, all others 0), via the
    standard skew(R_dot @ R^T) = hat(omega) identity -- same identity
    dynamics.py's _omega_from_Rdot uses, just per-column via jacrev
    instead of a single JVP along a specific qd direction."""
    def pos_fn(qq):
        p, _ = hand_pose_from_q(qq, rot_ref, shou2elb, elb2wri, wri2arrow, quat_wri2arrow_rotmat)
        return p

    def rot_fn(qq):
        _, R = hand_pose_from_q(qq, rot_ref, shou2elb, elb2wri, wri2arrow, quat_wri2arrow_rotmat)
        return R

    J_pos = torch.func.jacrev(pos_fn)(q)  # (3,7)
    dR_dq = torch.func.jacrev(rot_fn)(q)  # (3,3,7)
    R = rot_fn(q)
    RT = R.transpose(-1, -2)
    skew = torch.einsum("ijk,jl->ilk", dR_dq, RT)  # (3,3,7), skew[:,:,i] = dR/dq_i @ R^T
    J_ang = torch.stack([skew[2, 1, :], skew[0, 2, :], skew[1, 0, :]], dim=0)  # (3,7)
    return torch.cat([J_pos, J_ang], dim=0)  # (6,7)


def _rotation_log(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> rotation vector (axis*angle), for finite-
    differencing angular velocity from consecutive recorded orientations."""
    return Rotation.from_matrix(R).as_rotvec()


def solve_trial(trial: dict, seg_dims: np.ndarray, quat_wri2arrow: np.ndarray,
                 seg_params: dict, q0: np.ndarray,
                 n_null_steps: int = 15, null_lr: float = 0.1,
                 damping: float = 1e-3, alpha_reg: float = 1.0) -> dict:
    """
    Runs the gradient-projection controller over one trial's recorded
    hand trajectory. Returns {"q": (T,7) rad, "tau": (T,7) N*m,
    "track_residual": (T,6)}; track_residual is the gap between the
    controller's hand pose and the recorded one at each frame (position
    m / rotation vector rad), i.e. how well J+ tracking with Euler
    integration followed the target.

    damping: Tikhonov damping of J+.
    alpha_reg: weight of the alpha^2 term of the null-space cost. Without
    it, ||tau(t) - tau(t-1)||^2 has no bounded minimum in alpha and the
    optimizer drifts along the null space.
    """
    shou2elb_np, elb2wri_np, wri2arrow_np = seg_dims
    shou2elb = torch.tensor(shou2elb_np, dtype=DTYPE)
    elb2wri = torch.tensor(elb2wri_np, dtype=DTYPE)
    wri2arrow = torch.tensor(wri2arrow_np, dtype=DTYPE)
    quat_wri2arrow_rotmat = torch.tensor(
        Rotation.from_quat(quat_wri2arrow).as_matrix(), dtype=DTYPE
    )

    t = trial["timestamps"]
    T = len(t)
    # dynamics.segment_chain works in the right-handed frame (as rot_ref,
    # seg_dims and quat_wri2arrow, see kinematics.load_segment_constants);
    # the recorded poses are in Unity's left-handed frame: X-flip the
    # positions and convert the quaternions.
    hand_pos_rh = trial["hand_pos"].copy()
    hand_pos_rh[:, 0] *= -1
    arm_root_pos_rh = trial["arm_root_pos"].copy()
    arm_root_pos_rh[:, 0] *= -1
    hand_pos_rel = hand_pos_rh - arm_root_pos_rh  # shoulder-relative, world-RH frame

    hand_rotmat = Rotation.from_quat(
        change_quat_ref_handeness(trial["hand_quat"].copy())
    ).as_matrix()  # (T,3,3)
    rot_ref_all = Rotation.from_quat(
        change_quat_ref_handeness(trial["arm_root_quat"].copy())
    ).as_matrix()  # (T,3,3)

    q = torch.tensor(q0, dtype=DTYPE)
    q_traj = np.empty((T, 7))
    tau_traj = np.empty((T, 7))
    track_residual = np.empty((T, 6))
    tau_prev = torch.zeros(7, dtype=DTYPE)
    qd_prev = torch.zeros(7, dtype=DTYPE)
    n_hat_prev = None  # null-space basis vector, previous frame (for sign/warm-start continuity)
    alpha_prev = torch.zeros((), dtype=DTYPE)  # scalar null-space coefficient, previous frame

    for i in range(T):
        rot_ref = torch.tensor(rot_ref_all[i], dtype=DTYPE)

        # Target hand velocity: central difference where possible.
        i0, i1 = max(i - 1, 0), min(i + 1, T - 1)
        dt = t[i1] - t[i0]
        v_target = torch.tensor((hand_pos_rel[i1] - hand_pos_rel[i0]) / dt, dtype=DTYPE)
        # World-frame angular velocity, as in geometric_jacobian's angular
        # rows: skew(omega) = Rdot @ R^T, so rotvec(R2 @ R1^T)/dt ~ omega.
        w_target_np = _rotation_log(hand_rotmat[i1] @ hand_rotmat[i0].T) / dt
        w_target = torch.tensor(w_target_np, dtype=DTYPE)
        x_dot_target = torch.cat([v_target, w_target])

        J = geometric_jacobian(q, rot_ref, shou2elb, elb2wri, wri2arrow, quat_wri2arrow_rotmat)
        JJt = J @ J.T
        J_pinv = J.T @ torch.linalg.solve(JJt + damping * torch.eye(6, dtype=DTYPE), torch.eye(6, dtype=DTYPE))

        # The null space of the full-row-rank (6,7) J is 1-D: its basis
        # vector is the last right-singular vector, and only its scalar
        # coefficient alpha is optimized (equivalent to (I - J+ @ J) @ qd0).
        _, _, Vh = torch.linalg.svd(J, full_matrices=True)
        n_hat = Vh[-1]
        if n_hat_prev is not None and torch.dot(n_hat, n_hat_prev) < 0:
            n_hat = -n_hat  # keep the basis vector's sign continuous frame-to-frame

        alpha = alpha_prev.clone().requires_grad_(True)
        opt = torch.optim.Adam([alpha], lr=null_lr)
        for _ in range(n_null_steps):
            opt.zero_grad()
            qd = J_pinv @ x_dot_target + alpha * n_hat
            qdd = (qd - qd_prev) / dt
            tau = inverse_dynamics_single(q, qd, qdd, rot_ref, shou2elb, elb2wri, wri2arrow, seg_params)
            cost = ((tau - tau_prev) ** 2).sum() + alpha_reg * alpha ** 2
            cost.backward()
            opt.step()

        with torch.no_grad():
            qd = J_pinv @ x_dot_target + alpha * n_hat
            qdd = (qd - qd_prev) / dt
            tau = inverse_dynamics_single(q, qd, qdd, rot_ref, shou2elb, elb2wri, wri2arrow, seg_params)

            hand_pos_actual, hand_rot_actual = hand_pose_from_q(
                q, rot_ref, shou2elb, elb2wri, wri2arrow, quat_wri2arrow_rotmat)
            pos_res = hand_pos_actual.numpy() - hand_pos_rel[i]
            rot_res = _rotation_log(hand_rot_actual.numpy().T @ hand_rotmat[i])
            track_residual[i] = np.concatenate([pos_res, rot_res])

            q_traj[i] = q.numpy()
            tau_traj[i] = tau.numpy()

            if i < T - 1:
                dt_fwd = t[i + 1] - t[i]
                q = q + qd * dt_fwd
            qd_prev = qd
            tau_prev = tau
            n_hat_prev = n_hat
            alpha_prev = alpha.detach()

    return {"q": q_traj, "tau": tau_traj, "track_residual": track_residual}
