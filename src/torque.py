"""
torque.py

Joint torques of one trial through the same inverse dynamics
(dynamics.inverse_dynamics_batch), from
  - the recorded motion (reference; the dataset has no measured torque):
    low-pass filtered recorded joint angles, cubic spline for q, qd, qdd;
  - the PINN: q, qd, qdd by automatic differentiation through the network
    and its interpolated hand-pose input.

Paper setting (filt="butter", smooth_hand=True): 6 Hz zero-lag Butterworth
low-pass on the reference joint angles and on the PINN's hand-pose input.
The tracked hand position has ~1-1.5 mm of frame-to-frame jitter, which,
differentiated twice, produces narrow torque spikes; the filter affects the
torques only, not the joint-angle results. filt="savgol" uses a
Savitzky-Golay filter instead.

Torques are evaluated at the recorded timestamps: between them, the second
derivative through the hand-pose spline can show isolated numerical spikes.
"""

import numpy as np
import torch
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter, butter, filtfilt
from scipy.spatial.transform import Rotation

from common import load_model
from dataset import DATASET_ROOT, PRIMARY_PHASE, load_exclusions, extract_trials, load_subject_info
from kinematics import load_segment_constants
from anthropometry import estimate_body_mass
from dynamics import build_segment_params, inverse_dynamics_batch
from trajectory_interp import HandTrajectorySpline
from pinn_model import query_trajectory_batch
from train_pinn import DTYPE
from tool_box.rot_quat_utils import change_quat_ref_handeness

BUTTER_FC = 6.0  # Hz


def lowpass(timestamps, x, filt):
    """(T, D) -> (T, D) filtered, at the recorded timestamps. savgol:
    window 21 frames, order 3. butter: the timestamps are irregular (~84 Hz
    with jitter), so the signal is resampled to a uniform grid at the median
    rate, filtered forward-backward and interpolated back."""
    if filt == "savgol":
        window = min(21, len(timestamps) // 2 * 2 - 1)  # odd, capped at 21
        return savgol_filter(x, window_length=window, polyorder=3, axis=0)
    n = int(round((timestamps[-1] - timestamps[0]) / np.median(np.diff(timestamps)))) + 1
    grid = np.linspace(timestamps[0], timestamps[-1], n)
    fs = (n - 1) / (timestamps[-1] - timestamps[0])
    b, a = butter(2, BUTTER_FC / (fs / 2))
    x_f = filtfilt(b, a, CubicSpline(timestamps, x, axis=0)(grid), axis=0)
    return CubicSpline(grid, x_f, axis=0)(timestamps)


def smooth_hand_pose(timestamps, hand_pos, hand_quat, filt):
    """Same filter as the reference, on the recorded hand pose.
    Quaternions are made sign-continuous before filtering (filtering across
    a q -> -q flip creates a huge artifact), renormalized, and given back
    each frame's recorded sign."""
    sign = np.ones(len(hand_quat))
    for i in range(1, len(hand_quat)):
        sign[i] = sign[i - 1] * (1.0 if np.dot(hand_quat[i], hand_quat[i - 1]) >= 0 else -1.0)
    quat = lowpass(timestamps, hand_quat * sign[:, None], filt)
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    pos = lowpass(timestamps, hand_pos, filt)
    return pos, quat * sign[:, None]


def compute_gt_and_pinn_torque(model, subject, tgt_number, smooth_hand=True, filt="butter"):
    """Torque of one trial from the recorded motion (reference) and from the
    PINN of models/<model>. Paper setting: smooth_hand=True, filt="butter".
    Returns (t_rel, tau_gt, tau_pinn), (T,) s and (T, 7) N*m."""
    net = load_model(model)

    subject_dir = DATASET_ROOT / subject
    exclusions = load_exclusions(subject_dir, PRIMARY_PHASE)
    all_trials = extract_trials(subject_dir, PRIMARY_PHASE, exclusions)
    trial = next(t for t in all_trials if t["tgt_number"] == tgt_number)

    seg_dims, quat_wri2arrow, arm_side = load_segment_constants(subject_dir)
    shou2elb_np, elb2wri_np, wri2arrow_np = seg_dims
    info = load_subject_info(subject_dir)
    body_mass = estimate_body_mass(info.height)
    hand_length = float(np.linalg.norm(wri2arrow_np))
    seg_params = build_segment_params(info.u_arm, info.f_arm, hand_length, body_mass, dtype=DTYPE)
    shou2elb = torch.tensor(shou2elb_np, dtype=DTYPE)
    elb2wri = torch.tensor(elb2wri_np, dtype=DTYPE)
    wri2arrow = torch.tensor(wri2arrow_np, dtype=DTYPE)
    subject_feats = torch.tensor([info.u_arm, info.f_arm, info.height], dtype=DTYPE)

    # recorded timestamps (see module docstring)
    t0, t1 = trial["timestamps"][0], trial["timestamps"][-1]
    t_dense = trial["timestamps"]
    t_dense_torch = torch.tensor(t_dense, dtype=DTYPE)
    t_rel = t_dense - t0

    rot_ref_dense = torch.tensor(
        Rotation.from_quat(change_quat_ref_handeness(trial["arm_root_quat"].copy())).as_matrix(),
        dtype=DTYPE,
    )

    # PINN: q, qd, qdd by automatic differentiation
    hand_pos, hand_quat = trial["hand_pos"], trial["hand_quat"]
    if smooth_hand:
        hand_pos, hand_quat = smooth_hand_pose(trial["timestamps"], hand_pos, hand_quat, filt)
    spline_hand = HandTrajectorySpline(trial["timestamps"], hand_pos, hand_quat, dtype=DTYPE)
    q_pinn, qd_pinn, qdd_pinn = query_trajectory_batch(net, spline_hand, subject_feats, t_dense_torch)
    tau_pinn = inverse_dynamics_batch(q_pinn, qd_pinn, qdd_pinn, rot_ref_dense,
                                       shou2elb, elb2wri, wri2arrow, seg_params)
    tau_pinn = tau_pinn.detach().numpy()

    # Reference: low-pass filtered recorded joint angles (an interpolating
    # spline alone passes the recording noise on to qdd), then a cubic
    # spline for the derivatives.
    q_smooth = lowpass(trial["timestamps"], trial["joint_angles"], filt)
    cs = CubicSpline(trial["timestamps"], q_smooth, axis=0)
    q_gt = cs(t_dense)
    qd_gt = cs(t_dense, 1)
    qdd_gt = cs(t_dense, 2)
    tau_gt = inverse_dynamics_batch(
        torch.tensor(q_gt, dtype=DTYPE), torch.tensor(qd_gt, dtype=DTYPE), torch.tensor(qdd_gt, dtype=DTYPE),
        rot_ref_dense, shou2elb, elb2wri, wri2arrow, seg_params
    ).detach().numpy()

    return t_rel, tau_gt, tau_pinn
