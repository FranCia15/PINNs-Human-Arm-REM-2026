"""
train_pinn.py

Trains the PINN (pinn_model.CoordinateNet) on the training trials of the
subjects of a config and saves model.pt, history.npz and config.json to
runs/<timestamp>_<tag>/ (the layout of models/).

Usage (from the repository root):
    python src/train_pinn.py [--config configs/paper_model.json]
configs/paper_model.json and configs/ablation_noeffort.json are the
configurations of the two models in models/ (paper model, and the same
model trained without the effort term).

Loss per trial, each term a mean (so the weights do not depend on the
number of points): joint-angle supervision at the labelled points
(normalized by each joint's range), and hand position and orientation
tracking, joint-limit penalty (1st-99th percentiles of the subject's
recorded angles) and effort term mean(tau^2) at the labelled and
collocation points. The effort weight ramps up linearly from 0 over the
first half of training: the large accelerations of an untrained network
would otherwise make the effort term dominate.
"""

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from dataset import (
    DATASET_ROOT, PRIMARY_PHASE, JOINT_COLS,
    load_exclusions, extract_trials, load_subject_info, compute_functional_joint_limits,
)
from kinematics import load_segment_constants
from anthropometry import estimate_body_mass
from dynamics import build_segment_params, segment_chain, inverse_dynamics_batch
from trajectory_interp import HandTrajectorySpline
from pinn_model import CoordinateNet, query_trajectory_batch
from tool_box.rot_quat_utils import change_quat_ref_handeness

RUNS_ROOT = Path(__file__).resolve().parents[1] / "runs"
DTYPE = torch.float64

CHECKPOINT_EVERY = 25  # steps (and at the last step)

# Defaults for keys missing from a config (the configs in configs/ specify
# every key).
DEFAULT_CFG = {
    "subject_trial_counts": {
        "s1": 7, "s3": 7, "s5": 7, "s7": 7, "s9": 7,
        "s11": 7, "s13": 7, "s15": 7, "s17": 7, "s19": 7,
    },
    "n_query_per_trial": 60,
    "n_collocation_per_trial": 60,
    "n_steps": 300,
    "seed": 1,
    "w_sup": 1.0, "w_track": 1.0, "w_track_rot": 1.0, "w_limit": 1.0,
    "w_effort_max": 1e-4,
    "lr": 1e-3,
    "hidden": 128,
    "n_layers": 4,
    "tag": "",  # optional suffix of the run directory name
}


def hand_pose_from_q(q, arm_root_pos_rh, rot_ref, shou2elb, elb2wri, wri2arrow, r_arrow_offset):
    """Differentiable hand pose via dynamics.segment_chain (com_fracs = 1
    gives the hand position): hand_pos in Unity's left-handed frame, as
    trial["hand_pos"]; hand_rotmat in the right-handed frame, as the
    hand_rotmat_true of prepare_subject_trials."""
    out = segment_chain(q, rot_ref, shou2elb, elb2wri, wri2arrow,
                         {"upper_arm": 1.0, "forearm": 1.0, "hand": 1.0})
    hand_pos_rh = out["com_hand"] + arm_root_pos_rh
    hand_pos = hand_pos_rh.clone()
    hand_pos[:, 0] *= -1
    hand_rotmat = out["R_hand"] @ r_arrow_offset
    return hand_pos, hand_rotmat


def joint_limit_penalty(q, lo, hi):
    return (torch.relu(q - hi) ** 2 + torch.relu(lo - q) ** 2).mean()


def prepare_subject_trials(subject: str, n_trials: int, n_query: int, n_colloc: int) -> list[dict]:
    """One dict per training trial of the subject, with the subject's
    segment and dynamics constants."""
    subject_dir = DATASET_ROOT / subject
    exclusions = load_exclusions(subject_dir, PRIMARY_PHASE)
    all_trials = extract_trials(subject_dir, PRIMARY_PHASE, exclusions)
    trials = all_trials[:n_trials]

    seg_dims, quat_wri2arrow, arm_side = load_segment_constants(subject_dir)
    shou2elb_np, elb2wri_np, wri2arrow_np = seg_dims
    info = load_subject_info(subject_dir)

    functional_limits = compute_functional_joint_limits(all_trials)
    lo = torch.tensor([functional_limits[c][0] for c in JOINT_COLS], dtype=DTYPE)
    hi = torch.tensor([functional_limits[c][1] for c in JOINT_COLS], dtype=DTYPE)
    # Per-joint range, for normalizing sup_loss so sp's ~1.75 rad range
    # doesn't dominate the gradient over ep/hp's ~0.45 rad range.
    joint_range = hi - lo

    body_mass = estimate_body_mass(info.height)
    hand_length = float(np.linalg.norm(wri2arrow_np))
    seg_params = build_segment_params(info.u_arm, info.f_arm, hand_length, body_mass, dtype=DTYPE)

    shou2elb = torch.tensor(shou2elb_np, dtype=DTYPE)
    elb2wri = torch.tensor(elb2wri_np, dtype=DTYPE)
    wri2arrow = torch.tensor(wri2arrow_np, dtype=DTYPE)
    subject_feats = torch.tensor([info.u_arm, info.f_arm, info.height], dtype=DTYPE)
    # constant wrist-to-marker rotation, right-handed (as in
    # kinematics.forward_kinematics)
    r_arrow_offset = torch.tensor(R.from_quat(quat_wri2arrow).as_matrix(), dtype=DTYPE)

    prepared = []
    for t in trials:
        # labelled (idx) and collocation (idx_collo) points interleaved on
        # one grid, so that they do not coincide
        n_total = n_query + n_colloc
        dense_idx = np.linspace(0, len(t["timestamps"]) - 1, n_total).astype(int)
        idx = dense_idx[0::2][:n_query]
        idx_collo = dense_idx[1::2][:n_colloc]

        def _rot_ref_and_pos(indices):
            rh_arm_root_quat = change_quat_ref_handeness(t["arm_root_quat"][indices].copy())
            rot_ref = torch.tensor(R.from_quat(rh_arm_root_quat).as_matrix(), dtype=DTYPE)
            arm_root_pos_rh = t["arm_root_pos"][indices].copy()
            arm_root_pos_rh[:, 0] *= -1
            return rot_ref, torch.tensor(arm_root_pos_rh, dtype=DTYPE)

        def _hand_rotmat_true(indices):
            # recorded hand orientation, right-handed (as hand_pose_from_q)
            rh_quat = change_quat_ref_handeness(t["hand_quat"][indices].copy())
            return torch.tensor(R.from_quat(rh_quat).as_matrix(), dtype=DTYPE)

        spline = HandTrajectorySpline(t["timestamps"], t["hand_pos"], t["hand_quat"], dtype=DTYPE)
        rot_ref, arm_root_pos_rh = _rot_ref_and_pos(idx)
        rot_ref_collo, arm_root_pos_rh_collo = _rot_ref_and_pos(idx_collo)

        prepared.append({
            "subject": subject,
            "tgt_number": t["tgt_number"],
            "spline": spline,
            "rot_ref": rot_ref,
            "q_true": torch.tensor(t["joint_angles"][idx], dtype=DTYPE),
            "t_query": torch.tensor(t["timestamps"][idx], dtype=DTYPE),
            "arm_root_pos_rh": arm_root_pos_rh,
            "hand_pos_true": torch.tensor(t["hand_pos"][idx], dtype=DTYPE),
            "hand_rotmat_true": _hand_rotmat_true(idx),
            "rot_ref_collo": rot_ref_collo,
            "t_query_collo": torch.tensor(t["timestamps"][idx_collo], dtype=DTYPE),
            "arm_root_pos_rh_collo": arm_root_pos_rh_collo,
            "hand_pos_true_collo": torch.tensor(t["hand_pos"][idx_collo], dtype=DTYPE),
            "hand_rotmat_true_collo": _hand_rotmat_true(idx_collo),
            "subject_feats": subject_feats,
            "shou2elb": shou2elb, "elb2wri": elb2wri, "wri2arrow": wri2arrow,
            "r_arrow_offset": r_arrow_offset,
            "seg_params": seg_params, "lo": lo, "hi": hi, "joint_range": joint_range,
        })
    return prepared


def save_checkpoint(run_dir: Path, net, history: dict, total_time: float, cfg: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), run_dir / "model.pt")
    np.savez(run_dir / "history.npz", **{k: np.array(v) for k, v in history.items()})
    config = {
        "subject_trial_counts": cfg["subject_trial_counts"],
        "n_query_per_trial": cfg["n_query_per_trial"],
        "n_collocation_per_trial": cfg["n_collocation_per_trial"],
        "n_steps": cfg["n_steps"],
        "seed": cfg["seed"],
        "w_sup": cfg["w_sup"], "w_track": cfg["w_track"], "w_track_rot": cfg["w_track_rot"], "w_limit": cfg["w_limit"],
        "w_effort_max": cfg["w_effort_max"], "effort_warmup_steps": cfg["n_steps"] // 2,
        "lr": cfg["lr"], "hidden": cfg["hidden"], "n_layers": cfg["n_layers"], "tag": cfg["tag"],
        "dtype": str(DTYPE),
        "total_time_s": total_time,
        "steps_completed": len(history["step"]),
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)


def run(cfg: dict | None = None, resume_from: Path | None = None) -> Path:
    """Trains one model per cfg (missing keys from DEFAULT_CFG) and returns
    the run directory. resume_from: a run directory to continue from its
    last checkpoint (weights and history; the Adam state is not saved, so
    this is a warm restart)."""
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    effort_warmup_steps = cfg["n_steps"] // 2

    def effort_weight(step):
        return cfg["w_effort_max"] * min(1.0, step / effort_warmup_steps)

    torch.manual_seed(cfg["seed"])
    net = CoordinateNet(hidden=cfg["hidden"], n_layers=cfg["n_layers"], dtype=DTYPE)
    opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"])

    prepared = []
    for subject, n_trials in cfg["subject_trial_counts"].items():
        prepared.extend(prepare_subject_trials(
            subject, n_trials, cfg["n_query_per_trial"], cfg["n_collocation_per_trial"]))
    n_trials_total = len(prepared)

    print(f"Training run: cfg={ {k: v for k, v in cfg.items() if k != 'subject_trial_counts'} }, "
          f"subjects={cfg['subject_trial_counts']}, {n_trials_total} trials total\n")

    start_step = 0
    prior_time = 0.0
    if resume_from is not None:
        net.load_state_dict(torch.load(resume_from / "model.pt", weights_only=True))
        old_hist = np.load(resume_from / "history.npz")
        history = {k: list(old_hist[k]) for k in old_hist.files}
        start_step = len(history["step"])
        with open(resume_from / "config.json") as f:
            prior_time = json.load(f).get("total_time_s", 0.0)
        run_dir = resume_from
        print(f"Resuming from {run_dir.resolve()} at step {start_step} "
              f"(prior wall time {prior_time:.0f}s)\n")
    else:
        history = {"step": [], "total": [], "sup": [], "track": [], "orient": [], "limit": [], "effort": [], "w_effort": []}
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        if cfg["tag"]:
            run_id = f"{run_id}_{cfg['tag']}"
        run_dir = RUNS_ROOT / run_id
        print(f"Run dir: {run_dir.resolve()} (checkpointing every {CHECKPOINT_EVERY} steps)\n")

    run_start = time.perf_counter()
    for step in range(start_step, cfg["n_steps"]):
        step_start = time.perf_counter()
        opt.zero_grad()
        total_loss_value = 0.0
        sup_l = track_l = orient_l = limit_l = effort_l = 0.0
        w_effort = effort_weight(step)

        for p in prepared:
            q_pred, qd_pred, qdd_pred = query_trajectory_batch(net, p["spline"], p["subject_feats"], p["t_query"])

            sup_loss = (((q_pred - p["q_true"]) / p["joint_range"]) ** 2).mean()

            # Tracking, joint-limit and effort terms on the labelled and
            # collocation points (both are recorded frames; only the
            # labelled ones are supervised on the joint angles).
            q_pred_c, qd_pred_c, qdd_pred_c = query_trajectory_batch(
                net, p["spline"], p["subject_feats"], p["t_query_collo"]
            )
            q_all = torch.cat([q_pred, q_pred_c], dim=0)
            qd_all = torch.cat([qd_pred, qd_pred_c], dim=0)
            qdd_all = torch.cat([qdd_pred, qdd_pred_c], dim=0)
            rot_ref_all = torch.cat([p["rot_ref"], p["rot_ref_collo"]], dim=0)
            arm_root_pos_rh_all = torch.cat([p["arm_root_pos_rh"], p["arm_root_pos_rh_collo"]], dim=0)
            hand_pos_true_all = torch.cat([p["hand_pos_true"], p["hand_pos_true_collo"]], dim=0)
            hand_rotmat_true_all = torch.cat([p["hand_rotmat_true"], p["hand_rotmat_true_collo"]], dim=0)

            hand_pos_pred, hand_rotmat_pred = hand_pose_from_q(
                q_all, arm_root_pos_rh_all, rot_ref_all,
                p["shou2elb"], p["elb2wri"], p["wri2arrow"], p["r_arrow_offset"]
            )
            track_loss = ((hand_pos_pred - hand_pos_true_all) ** 2).mean()
            # Chordal (Frobenius) distance, not the geodesic angle: smooth
            # and well-defined everywhere, unlike arccos-based angle error
            # which has a singular gradient at zero error.
            orient_loss = ((hand_rotmat_pred - hand_rotmat_true_all) ** 2).mean()

            limit_loss = joint_limit_penalty(q_all, p["lo"], p["hi"])

            tau = inverse_dynamics_batch(q_all, qd_all, qdd_all, rot_ref_all,
                                          p["shou2elb"], p["elb2wri"], p["wri2arrow"], p["seg_params"])
            effort_loss = (tau ** 2).mean()

            trial_loss = (cfg["w_sup"] * sup_loss + cfg["w_track"] * track_loss
                          + cfg["w_track_rot"] * orient_loss
                          + cfg["w_limit"] * limit_loss + w_effort * effort_loss)
            # Backward per trial: the gradients accumulate as for the summed
            # loss, but each trial's graph is freed at once (memory).
            scaled_loss = trial_loss / n_trials_total
            scaled_loss.backward()
            total_loss_value += scaled_loss.item()
            sup_l += sup_loss.item() / n_trials_total
            track_l += track_loss.item() / n_trials_total
            orient_l += orient_loss.item() / n_trials_total
            limit_l += limit_loss.item() / n_trials_total
            effort_l += effort_loss.item() / n_trials_total

        opt.step()

        history["step"].append(step)
        history["total"].append(total_loss_value)
        history["sup"].append(sup_l)
        history["track"].append(track_l)
        history["orient"].append(orient_l)
        history["limit"].append(limit_l)
        history["effort"].append(effort_l)
        history["w_effort"].append(w_effort)

        step_time = time.perf_counter() - step_start
        if step % 10 == 0 or step == cfg["n_steps"] - 1:
            print(f"step {step:3d}: total={total_loss_value:10.4f}  "
                  f"sup={sup_l:9.4f}  track={track_l:9.4f}  orient={orient_l:9.4f}  "
                  f"limit={limit_l:9.4f}  effort={effort_l:14.2f}  "
                  f"w_effort={w_effort:.2e}  step_time={step_time:.2f}s")

        if step % CHECKPOINT_EVERY == 0 or step == cfg["n_steps"] - 1:
            save_checkpoint(run_dir, net, history, prior_time + time.perf_counter() - run_start, cfg)

    total_time = prior_time + time.perf_counter() - run_start
    n_steps_this_call = cfg["n_steps"] - start_step
    print(f"\nTotal: {total_time:.1f}s for {cfg['n_steps']} steps ({n_steps_this_call} this call, "
          f"{n_trials_total} trials/step)")
    print(f"Saved run to {run_dir.resolve()}")
    return run_dir


if __name__ == "__main__":
    import sys
    cfg_path = sys.argv[sys.argv.index("--config") + 1] if "--config" in sys.argv else         Path(__file__).resolve().parents[1] / "configs" / "paper_model.json"
    run(json.loads(Path(cfg_path).read_text()))
