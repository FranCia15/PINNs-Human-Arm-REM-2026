"""
trajectory_interp.py

Differentiable (PyTorch) cubic-spline interpolation of a trial's recorded
hand pose, the input of the coordinate network (pinn_model.py). The spline
is fitted with scipy; evaluating it is a torch function of t, so dq/dt
includes the dependence of q on the hand pose at t.

Quaternions are splined component-wise and renormalized, adequate for the
small orientation changes of a reach.
"""

import numpy as np
import torch
from scipy.interpolate import CubicSpline


class HandTrajectorySpline:
    """Fits once (numpy/scipy) at construction; position()/orientation()
    are torch functions of a scalar or batched query time, differentiable
    via autograd."""

    def __init__(self, timestamps: np.ndarray, hand_pos: np.ndarray,
                 hand_quat: np.ndarray, dtype=torch.float64):
        self.t0 = float(timestamps[0])
        self.t1 = float(timestamps[-1])
        t = timestamps - self.t0  # trial-relative time, avoids large-t numerical issues

        # Limitation: the recorded quaternions are not always
        # sign-continuous (q and -q are the same rotation), and the spline
        # is fitted through the signs as recorded; at a sign change,
        # orientation(t) has a narrow spike over one frame interval. The
        # released models were trained on this input, so it is kept as is;
        # making the signs continuous would require retraining.
        pos_splines = [CubicSpline(t, hand_pos[:, i]) for i in range(3)]
        quat_splines = [CubicSpline(t, hand_quat[:, i]) for i in range(4)]

        # scipy CubicSpline.c has shape (4, n_segments): value =
        # c[0]*dt^3 + c[1]*dt^2 + c[2]*dt + c[3], dt = t - breakpoint.
        self.breakpoints = torch.tensor(pos_splines[0].x, dtype=dtype)
        self._pos_coeffs = torch.stack(
            [torch.tensor(s.c, dtype=dtype) for s in pos_splines], dim=0
        )  # (3, 4, n_segments)
        self._quat_coeffs = torch.stack(
            [torch.tensor(s.c, dtype=dtype) for s in quat_splines], dim=0
        )  # (4, 4, n_segments)

    def _eval(self, coeffs: torch.Tensor, t_rel: torch.Tensor) -> torch.Tensor:
        """coeffs: (D, 4, n_segments), t_rel: (...) trial-relative time
        -> (..., D). The spline segment is selected by a one-hot weighted
        sum: torch.searchsorted and advanced indexing do not work under
        torch.func.vmap, as used by pinn_model.py."""
        bp_start = self.breakpoints[:-1]
        n_seg = coeffs.shape[-1]

        idx = (t_rel.detach().unsqueeze(-1) >= bp_start).sum(-1) - 1
        idx = idx.clamp(0, n_seg - 1)
        one_hot = (idx.unsqueeze(-1) == torch.arange(n_seg, dtype=idx.dtype)).to(coeffs.dtype)

        bp_sel = torch.einsum("...s,s->...", one_hot, bp_start)
        dt = t_rel - bp_sel

        c = torch.einsum("...s,dks->...dk", one_hot, coeffs)  # (..., D, 4)
        val = c[..., 0]
        for k in range(1, 4):
            val = val * dt.unsqueeze(-1) + c[..., k]
        return val  # (..., D)

    def position(self, t: torch.Tensor) -> torch.Tensor:
        t_rel = (t - self.t0).clamp(0.0, self.t1 - self.t0)
        return self._eval(self._pos_coeffs, t_rel)

    def orientation(self, t: torch.Tensor) -> torch.Tensor:
        t_rel = (t - self.t0).clamp(0.0, self.t1 - self.t0)
        q = self._eval(self._quat_coeffs, t_rel)
        return q / q.norm(dim=-1, keepdim=True)
