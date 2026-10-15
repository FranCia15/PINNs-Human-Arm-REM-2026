"""
dynamics.py

Differentiable (PyTorch) rigid-body model of the 7-DOF arm: upper arm,
forearm and hand, joints sp, sr, ay (shoulder), ep (elbow), fy, hp, hr
(forearm and wrist). Gradients flow from the joint torques back to q, qd,
qdd, as the PINN's effort term requires.

  segment_chain       forward kinematics: the rotation chain of
                      tool_box.rot_quat_utils.config2quats (intrinsic Euler
                      angles "XZY" shoulder, "X" elbow, "YXZ" wrist,
                      relative to the arm root), with torch rotation
                      matrices instead of scipy rotations
  inverse_dynamics_*  Euler-Lagrange: tau = d/dt(dL/dqd) - dL/dq, L = T - V
                      from the segments' centre-of-mass velocities and
                      angular velocities (torch.func.jvp through the chain)
                      and their heights, differentiated with torch.func

Modelling assumptions:
  - the arm root (trunk) is a fixed base: its own acceleration is neglected;
  - each segment's local Y axis is its long axis; the sagittal and
    transverse moments of inertia are assigned to local X and Z;
  - the hand segment runs from the wrist to the tracked end-effector
    marker (wri2arrow): DBAS22 gives no anatomical hand segment;
  - no payload and no contact forces.
"""

import torch


def _rotmat_x(a: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(a), torch.sin(a)
    one, zero = torch.ones_like(a), torch.zeros_like(a)
    row0 = torch.stack([one, zero, zero], dim=-1)
    row1 = torch.stack([zero, c, -s], dim=-1)
    row2 = torch.stack([zero, s, c], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def _rotmat_y(a: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(a), torch.sin(a)
    one, zero = torch.ones_like(a), torch.zeros_like(a)
    row0 = torch.stack([c, zero, s], dim=-1)
    row1 = torch.stack([zero, one, zero], dim=-1)
    row2 = torch.stack([-s, zero, c], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def _rotmat_z(a: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(a), torch.sin(a)
    one, zero = torch.ones_like(a), torch.zeros_like(a)
    row0 = torch.stack([c, -s, zero], dim=-1)
    row1 = torch.stack([s, c, zero], dim=-1)
    row2 = torch.stack([zero, zero, one], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


_AXIS_FN = {"X": _rotmat_x, "Y": _rotmat_y, "Z": _rotmat_z}


def _compose_intrinsic_euler(order: str, angles: torch.Tensor) -> torch.Tensor:
    """
    Matches scipy Rotation.from_euler(order, angles) for an intrinsic
    (uppercase) sequence: R = R_axis0(a0) @ R_axis1(a1) @ ... (matrix
    product in the same left-to-right order as the axis letters).

    angles: (..., len(order))
    Returns (..., 3, 3)
    """
    R = None
    for i, axis in enumerate(order):
        Ri = _AXIS_FN[axis](angles[..., i])
        R = Ri if R is None else R @ Ri
    return R


def segment_chain(q: torch.Tensor, rot_ref: torch.Tensor,
                   shou2elb: torch.Tensor, elb2wri: torch.Tensor,
                   wri2arrow: torch.Tensor, com_fracs: dict) -> dict:
    """
    Forward kinematics of the 3-segment chain, in the right-handed frame of
    rot_ref.

    Parameters
    ----------
    q : (..., 7) joint angles [sp, sr, ay, ep, fy, hp, hr], rad
    rot_ref : (..., 3, 3) arm-root rotation matrix
    shou2elb, elb2wri, wri2arrow : (3,) segment vectors in the segments'
        local frames, right-handed (kinematics.load_segment_constants)
    com_fracs : {"upper_arm", "forearm", "hand": float} centre-of-mass
        position as a fraction of segment length from the proximal joint;
        1.0 for every segment gives the joint and hand positions

    Returns
    -------
    dict with R_upper, R_forearm, R_hand (..., 3, 3) and p_elbow, p_wrist,
    com_upper, com_forearm, com_hand (..., 3), relative to the shoulder.
    """
    R_upper = rot_ref @ _compose_intrinsic_euler("XZY", q[..., 0:3])
    R_forearm = R_upper @ _compose_intrinsic_euler("X", q[..., 3:4])
    R_hand = R_forearm @ _compose_intrinsic_euler("YXZ", q[..., 4:7])

    p_elbow = torch.einsum("...ij,j->...i", R_upper, shou2elb)
    p_wrist = p_elbow + torch.einsum("...ij,j->...i", R_forearm, elb2wri)

    com_upper = torch.einsum("...ij,j->...i", R_upper, shou2elb * com_fracs["upper_arm"])
    com_forearm = p_elbow + torch.einsum(
        "...ij,j->...i", R_forearm, elb2wri * com_fracs["forearm"]
    )
    com_hand = p_wrist + torch.einsum(
        "...ij,j->...i", R_hand, wri2arrow * com_fracs["hand"]
    )

    return {
        "R_upper": R_upper, "R_forearm": R_forearm, "R_hand": R_hand,
        "p_elbow": p_elbow, "p_wrist": p_wrist,
        "com_upper": com_upper, "com_forearm": com_forearm, "com_hand": com_hand,
    }


# ---------------------------------------------------------------------------
# Anthropometric segment parameters, packaged for the energy/torque layer
# ---------------------------------------------------------------------------

def build_segment_params(u_arm_length: float, f_arm_length: float,
                          hand_length: float, body_mass: float,
                          dtype=torch.float64) -> dict:
    """
    {segment: {"mass": 0-d tensor, "com_frac": float, "I_body_diag":
    (3,) tensor}} for upper_arm, forearm, hand, from anthropometry.py.
    I_body_diag: principal moments about local (X, Y, Z) = (sagittal,
    longitudinal, transverse).
    """
    from anthropometry import segment_inertial_params

    lengths = {"upper_arm": u_arm_length, "forearm": f_arm_length, "hand": hand_length}
    params = {}
    for seg, length in lengths.items():
        p = segment_inertial_params(seg, length, body_mass)
        i_sag, i_trans, i_long = p["inertia"]
        params[seg] = {
            "mass": torch.tensor(p["mass"], dtype=dtype),
            "com_frac": p["com_frac"],
            "I_body_diag": torch.tensor([i_sag, i_long, i_trans], dtype=dtype),
        }
    return params


# ---------------------------------------------------------------------------
# Energy and inverse dynamics via autodiff (torch.func: jvp/jacrev/vmap)
# ---------------------------------------------------------------------------

def _omega_from_Rdot(Rmat: torch.Tensor, Rdot: torch.Tensor) -> torch.Tensor:
    """Angular velocity (world frame) from a rotation matrix and its time
    derivative, via the standard identity skew(omega) = Rdot @ R^T."""
    skew = Rdot @ Rmat.transpose(-1, -2)
    return torch.stack([skew[..., 2, 1], skew[..., 0, 2], skew[..., 1, 0]], dim=-1)


def kinetic_energy(q: torch.Tensor, qd: torch.Tensor, rot_ref: torch.Tensor,
                    shou2elb: torch.Tensor, elb2wri: torch.Tensor,
                    wri2arrow: torch.Tensor, seg_params: dict) -> torch.Tensor:
    """Single frame: q, qd are (7,); rot_ref is (3,3). Returns a scalar."""
    com_fracs = {k: v["com_frac"] for k, v in seg_params.items()}

    def chain_fn(qq):
        out = segment_chain(qq, rot_ref, shou2elb, elb2wri, wri2arrow, com_fracs)
        return (out["com_upper"], out["com_forearm"], out["com_hand"],
                out["R_upper"], out["R_forearm"], out["R_hand"])

    primals, tangents = torch.func.jvp(chain_fn, (q,), (qd,))
    com_u, com_f, com_h, R_u, R_f, R_h = primals
    v_u, v_f, v_h, Rd_u, Rd_f, Rd_h = tangents

    T = q.new_zeros(())
    for v, Rm, Rd, seg in [(v_u, R_u, Rd_u, "upper_arm"),
                           (v_f, R_f, Rd_f, "forearm"),
                           (v_h, R_h, Rd_h, "hand")]:
        m = seg_params[seg]["mass"]
        I_body = torch.diag(seg_params[seg]["I_body_diag"])
        I_world = Rm @ I_body @ Rm.transpose(-1, -2)
        w = _omega_from_Rdot(Rm, Rd)
        T = T + 0.5 * m * (v * v).sum() + 0.5 * (w @ I_world @ w)
    return T


def potential_energy(q: torch.Tensor, rot_ref: torch.Tensor,
                      shou2elb: torch.Tensor, elb2wri: torch.Tensor,
                      wri2arrow: torch.Tensor, seg_params: dict,
                      g: float = 9.81) -> torch.Tensor:
    """Single frame: q is (7,); rot_ref is (3,3). Returns a scalar.
    Y is up (Unity's world Y axis; the change to a right-handed frame
    flips X only)."""
    com_fracs = {k: v["com_frac"] for k, v in seg_params.items()}
    out = segment_chain(q, rot_ref, shou2elb, elb2wri, wri2arrow, com_fracs)

    V = q.new_zeros(())
    for com, seg in [(out["com_upper"], "upper_arm"),
                      (out["com_forearm"], "forearm"),
                      (out["com_hand"], "hand")]:
        V = V + seg_params[seg]["mass"] * g * com[1]
    return V


def inverse_dynamics_single(q: torch.Tensor, qd: torch.Tensor, qdd: torch.Tensor,
                             rot_ref: torch.Tensor, shou2elb: torch.Tensor,
                             elb2wri: torch.Tensor, wri2arrow: torch.Tensor,
                             seg_params: dict, g: float = 9.81) -> torch.Tensor:
    """
    Single frame inverse dynamics: tau = d/dt(dL/dqd) - dL/dq, q/qd/qdd
    all (7,), rot_ref (3,3). Returns (7,) joint torques.
    """
    def dL_dqd(q_, qd_):
        return torch.func.jacrev(
            lambda qq, qqd: kinetic_energy(qq, qqd, rot_ref, shou2elb, elb2wri, wri2arrow, seg_params)
            - potential_energy(qq, rot_ref, shou2elb, elb2wri, wri2arrow, seg_params, g),
            argnums=1,
        )(q_, qd_)

    _, dp_dt = torch.func.jvp(dL_dqd, (q, qd), (qd, qdd))

    dL_dq = torch.func.jacrev(
        lambda qq: kinetic_energy(qq, qd, rot_ref, shou2elb, elb2wri, wri2arrow, seg_params)
        - potential_energy(qq, rot_ref, shou2elb, elb2wri, wri2arrow, seg_params, g)
    )(q)

    return dp_dt - dL_dq


def inverse_dynamics_batch(q: torch.Tensor, qd: torch.Tensor, qdd: torch.Tensor,
                            rot_ref: torch.Tensor, shou2elb: torch.Tensor,
                            elb2wri: torch.Tensor, wri2arrow: torch.Tensor,
                            seg_params: dict, g: float = 9.81) -> torch.Tensor:
    """Batched over frames: q, qd, qdd are (T,7), rot_ref is (T,3,3).
    shou2elb/elb2wri/wri2arrow/seg_params are shared (not batched)."""
    fn = torch.func.vmap(
        lambda qi, qdi, qddi, ri: inverse_dynamics_single(
            qi, qdi, qddi, ri, shou2elb, elb2wri, wri2arrow, seg_params, g
        )
    )
    return fn(q, qd, qdd, rot_ref)
