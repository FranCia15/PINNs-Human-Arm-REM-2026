"""
pinn_model.py

Coordinate network of the PINN: q(t) = MLP(Fourier features of t, hand pose
interpolated at t, subject anthropometry) -> 7 joint angles, for a
continuous time t within a trial. qd and qdd are exact derivatives through
the network and the hand-pose spline (nested torch.func.jacrev).

Tanh activations: a ReLU network is piecewise linear, so its second
derivative, and with it the effort term, would vanish almost everywhere.
"""

import math

import torch
import torch.nn as nn

from trajectory_interp import HandTrajectorySpline


class CoordinateNet(nn.Module):
    def __init__(self, n_fourier: int = 3, hidden: int = 128, n_layers: int = 4,
                 n_subject_feats: int = 3, dtype=torch.float64):
        super().__init__()
        # freqs are in Hz (t is in seconds): 2**arange(3) = 1,2,4 Hz.
        # Higher caps (8 Hz and above) let the network, supervised only at
        # sparse points, produce noisy velocities and accelerations between
        # them; 4 Hz covers human reaching motion.
        freqs = 2.0 ** torch.arange(n_fourier, dtype=dtype)  # fixed, not learned
        self.register_buffer("freqs", freqs)

        input_dim = 2 * n_fourier + 3 + 4 + n_subject_feats  # fourier(t), hand_pos, hand_quat, subject
        layers = []
        d = input_dim
        for _ in range(n_layers):
            layers += [nn.Linear(d, hidden, dtype=dtype), nn.Tanh()]
            d = hidden
        layers += [nn.Linear(d, 7, dtype=dtype)]
        self.mlp = nn.Sequential(*layers)

    def fourier_features(self, t: torch.Tensor) -> torch.Tensor:
        angles = t.unsqueeze(-1) * self.freqs * (2 * math.pi)
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

    def forward(self, t: torch.Tensor, hand_pos: torch.Tensor,
                hand_quat: torch.Tensor, subject_feats: torch.Tensor) -> torch.Tensor:
        """t: scalar. hand_pos: (3,). hand_quat: (4,). subject_feats: (n_subject_feats,).
        Returns (7,) joint angles. Single-example; batch via vmap."""
        ff = self.fourier_features(t)
        x = torch.cat([ff, hand_pos, hand_quat, subject_feats], dim=-1)
        return self.mlp(x)


def query_trajectory(net: CoordinateNet, spline: HandTrajectorySpline,
                      subject_feats: torch.Tensor, t: torch.Tensor):
    """Single query time t (scalar). Returns q, qd, qdd, each (7,)."""

    def q_fn(tt):
        pos = spline.position(tt)
        quat = spline.orientation(tt)
        return net(tt, pos, quat, subject_feats)

    q = q_fn(t)
    qd_fn = lambda tt: torch.func.jacrev(q_fn)(tt)
    qd = qd_fn(t)
    qdd = torch.func.jacrev(qd_fn)(t)
    return q, qd, qdd


def query_trajectory_batch(net: CoordinateNet, spline: HandTrajectorySpline,
                            subject_feats: torch.Tensor, t_batch: torch.Tensor):
    """t_batch: (N,). Returns q, qd, qdd, each (N, 7)."""
    fn = torch.func.vmap(lambda tt: query_trajectory(net, spline, subject_feats, tt))
    return fn(t_batch)
