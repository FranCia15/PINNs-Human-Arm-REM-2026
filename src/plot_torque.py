"""
plot_torque.py

Torque figure of the paper (Fig. 2): joint torques derived from the
recorded motion and from the PINN with and without the effort term
(models/paper_model, models/ablation_noeffort), in torque.py's paper
setting.

Output: figures/torque_<subject>_tgt<N>_<joints>.pdf
Usage (from the repository root):
    python src/plot_torque.py [subject] [tgt_number] [joint joint ...]
Defaults (the paper's figure): s20, tgt 36, joints ay sp.
"""

import sys

import matplotlib.pyplot as plt

from common import ABLATION_MODEL, FIGURES, PAPER_MODEL
from dataset import JOINT_COLS
from torque import compute_gt_and_pinn_torque

COLOR_GT = "black"
COLOR_PINN = "#56B4E9"  # Okabe-Ito sky blue
COLOR_NO_EFFORT = "#E69F00"  # Okabe-Ito orange
FS_LABEL_C, FS_TICK_C, FS_LEGEND_C = 15, 13, 11


def main():
    args = sys.argv[1:]
    subject = args[0] if len(args) > 0 else "s20"
    tgt_number = int(args[1]) if len(args) > 1 else 36
    joints = args[2:] or ["ay", "sp"]

    t_rel, tau_gt, tau_pinn = compute_gt_and_pinn_torque(PAPER_MODEL, subject, tgt_number, smooth_hand=True, filt="butter")
    _, _, tau_no_effort = compute_gt_and_pinn_torque(ABLATION_MODEL, subject, tgt_number, smooth_hand=True, filt="butter")

    fig, axes = plt.subplots(len(joints), 1, figsize=(4.5, 3.4), sharex=True, constrained_layout=True)
    for ax, col in zip(axes, joints):
        j = JOINT_COLS.index(col)
        ax.axhline(0, color="0.85", linewidth=0.8, zorder=1)
        ax.plot(t_rel, tau_gt[:, j], color=COLOR_GT, linewidth=2.2, label="Recorded motion", zorder=3)
        ax.plot(t_rel, tau_pinn[:, j], color=COLOR_PINN, linewidth=1.8, label="PINN", zorder=2)
        ax.plot(t_rel, tau_no_effort[:, j], color=COLOR_NO_EFFORT, linewidth=1.4, linestyle="--",
                label="PINN w/o effort", zorder=2)
        ax.set_ylabel(f"{col}\n(N·m)", fontsize=FS_LABEL_C)
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=FS_TICK_C)
    axes[-1].set_xlabel("time (s)", fontsize=FS_LABEL_C)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", fontsize=FS_LEGEND_C,
               ncol=3, frameon=False, columnspacing=1.0, handlelength=1.6)

    FIGURES.mkdir(exist_ok=True)
    out_path = FIGURES / f"torque_{subject}_tgt{tgt_number}_{'_'.join(joints)}.pdf"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
