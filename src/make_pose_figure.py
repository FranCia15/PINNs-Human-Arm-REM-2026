"""
make_pose_figure.py

Assembles the pose comparison figure of the paper (Fig. 3) from the 12
RoboDK screenshots taken with robodk/capture_poses.py: rows recorded motion
/ PINN / IK / dynamics baseline, columns start / waypoint / end,
collaborative workcell below.

The screenshots share the camera zoom but were cropped by hand. Each one is
aligned on the mannequin's head (top of the head, horizontal centre); each
column takes the union of its four screenshots' horizontal extents, and all
panels share one height chosen so that no hand is cut. Where a screenshot
does not cover the window, it is padded by repeating its edge pixels
(background on the sides; a few rows of torso at the bottom of the end
column). The workcell screenshot is cropped by WORKCELL_CROP.

Method panels are annotated with
  - a dashed outline of the recorded pose (silhouette of the ground-truth
    screenshot of the same column, same alignment);
  - their swivel-angle error with respect to the recorded pose at that
    frame (elbow rotation about the shoulder-wrist axis, from the downward
    vertical, as in eval_cartesian_errors.py; true geometry of the
    exported poses), as a number and as the colour of the panel border,
    on a continuous scale shown by the colour bar.
--plain leaves all annotations out.

Input:  robodk/screenshots/{GT,PINN,IK,DYN}{1,2,3}.png, workcell.png,
        robodk/poses/s1_tgt10_compare_*.csv (src/export_poses.py)
Output: figures/pose_comparison[_plain].{pdf,png}

Usage (from the repository root): python src/make_pose_figure.py [--plain]
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from PIL import Image
from scipy import ndimage

from check_rig_rendering import load_csv, true_points
from common import FIGURES, ROBODK
from dataset import DATASET_ROOT
from eval_cartesian_errors import swivel
from kinematics import load_segment_constants

PLAIN = "--plain" in sys.argv
SHOTS = ROBODK / "screenshots"
POSES = ROBODK / "poses"
OUT = FIGURES / ("pose_comparison" + ("_plain" if PLAIN else ""))
ROWS = [("GT", "Ground truth", "groundtruth"), ("PINN", "PINN", "pinn"),
        ("IK", "IK baseline", "ik"), ("DYN", "Dynamics baseline", "dynamics")]
COLS = ["Start position", "Waypoint", "End position"]
WORKCELL_CROP = {"t": 0.23937, "r": 0.02549, "b": 0.16999}  # cropped fractions: top, right, bottom
CMAP = ListedColormap(colormaps["YlOrRd"](np.linspace(0.22, 1.0, 256)))  # from a visible light yellow
NORM = Normalize(0, 60)  # swivel-angle error, deg
OUTLINE = "#00e5ff"


def body_mask(a):
    """Mannequin pixels: farther than a threshold from their row's background colour (the
    background is a vertical gradient; reference = median of the 8 outermost columns)."""
    a = a.astype(float)
    row_bg = np.median(np.concatenate([a[:, :8], a[:, -8:]], axis=1), axis=1)[:, None, :]
    mask = np.linalg.norm(a - row_bg, axis=2) > 30
    return ndimage.binary_fill_holes(ndimage.binary_opening(mask, iterations=1))


def head_anchor(a):
    """(x centre, top) of the mannequin's head: the topmost light-grey (non-blue) pixels."""
    a = a.astype(int)
    body = (a.min(2) > 170) & (np.abs(a[..., 0] - a[..., 2]) < 60)
    top = np.where(body.any(1))[0][0]
    xs = np.where(body[top:top + 70].any(0))[0]
    return (xs.min() + xs.max()) // 2, top


def window(a, cx, top, left, right, up, down):
    """Crop [cx-left, cx+right) x [top-up, top+down), padding with edge pixels where needed."""
    h, w = a.shape[:2]
    pad = ((max(0, up - top), max(0, top + down - h)), (max(0, left - cx), max(0, cx + right - w)), (0, 0))
    a = np.pad(a, pad, mode="edge")
    y0, x0 = top - up + pad[0][0], cx - left + pad[1][0]
    return a[y0:y0 + up + down, x0:x0 + left + right]


def swivel_errors():
    """Per method, swivel-angle error (deg) w.r.t. the recorded pose at the 3 frames; and the frame times."""
    seg_dims, _, _ = load_segment_constants(DATASET_ROOT / "s1")
    data = {key: load_csv(POSES / f"s1_tgt10_compare_{key}.csv") for _, _, key in ROWS}
    sw = {}
    for key, (q, rq) in data.items():
        elbow, wrist, _ = true_points(q, rq, seg_dims)
        sw[key] = swivel(np.zeros_like(elbow), elbow, wrist)
    err = {key: np.degrees(np.abs(np.angle(np.exp(1j * (s - sw["groundtruth"]))))) for key, s in sw.items()}
    with open(POSES / "s1_tgt10_compare_groundtruth.csv") as f:
        times = [float(line.split(",")[0]) for line in f.readlines()[1:]]
    return err, times


def main():
    imgs = {(r, c): np.asarray(Image.open(SHOTS / f"{r}{c + 1}.png").convert("RGB")) for r, _, _ in ROWS for c in range(3)}
    anchors = {k: head_anchor(a) for k, a in imgs.items()}
    up = max(top for _, top in anchors.values())
    # height: the bottom of the shortest screenshot of the start/waypoint columns (keeps their hands)
    down = min(a.shape[0] - anchors[k][1] for k, a in imgs.items() if k[1] < 2)
    crops, widths = {}, []
    for c in range(3):
        col = [(r, c) for r, _, _ in ROWS]
        left = max(anchors[k][0] for k in col)
        right = max(imgs[k].shape[1] - anchors[k][0] for k in col)
        widths.append(left + right)
        for k in col:
            crops[k] = window(imgs[k], *anchors[k], left, right, up, down)
    errors, times = swivel_errors()
    for _, _, key in ROWS[1:]:
        print(f"{key:9s} swivel error (deg): {np.round(errors[key], 1)}")

    work = Image.open(SHOTS / "workcell.png").convert("RGB")
    w, h = work.size
    work = work.crop((0, int(WORKCELL_CROP["t"] * h), int(w * (1 - WORKCELL_CROP["r"])), int(h * (1 - WORKCELL_CROP["b"]))))
    fracs = np.array(widths) / sum(widths)  # column widths, proportional to the windows
    plt.rcParams.update({"font.family": "serif", "font.size": 8,
                         "pdf.fonttype": 42, "ps.fonttype": 42})  # TrueType, IEEE PDF eXpress rejects Type 3
    width = 3.5  # in, one IEEE column
    head_h, title_h, cbar_h = 0.30, 0.16, 0.0 if PLAIN else 0.50
    cell_h = width * (up + down) / sum(widths)
    work_h = width * work.size[1] / work.size[0]
    height = head_h + 5 * title_h + 4 * cell_h + cbar_h + work_h
    fig = plt.figure(figsize=(width, height))
    y = height - head_h
    for c, name in enumerate(COLS):
        x = fracs[:c].sum() + fracs[c] / 2
        fig.text(x, (y + 0.19) / height, name, ha="center", va="center", weight="bold")
        fig.text(x, (y + 0.07) / height, f"$t$ = {times[c]:.1f} s", ha="center", va="center", size=7)
    for r, label, key in ROWS:
        fig.text(0.5, (y - title_h / 2) / height, label, ha="center", va="center")
        y -= title_h + cell_h
        for c in range(3):
            ax = fig.add_axes([fracs[:c].sum(), y / height, fracs[c], cell_h / height])
            ax.imshow(crops[(r, c)])
            ax.axis("off")
            if PLAIN or key == "groundtruth":
                continue
            ax.contour(body_mask(crops[("GT", c)]), levels=[0.5], colors=OUTLINE, linewidths=0.5, linestyles="--")
            e = errors[key][c]
            hh, ww = crops[(r, c)].shape[:2]
            ax.add_patch(Rectangle((1.5, 1.5), ww - 4, hh - 4, fill=False, lw=2.2, ec=CMAP(NORM(e))))
            ax.text(ww - 12, 14, f"{e:.0f}°", ha="right", va="top", size=7, weight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=CMAP(NORM(e)), lw=1))
    if not PLAIN:
        cax = fig.add_axes([0.05, (y - 0.17) / height, 0.58, 0.07 / height])
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=NORM, cmap=CMAP), cax=cax, orientation="horizontal",
                          extend="max")
        cb.set_label("Swivel-angle error (°)", size=7, labelpad=1)
        cb.ax.tick_params(labelsize=6.5, length=2, pad=1)
        fig.legend(handles=[Line2D([], [], ls="--", color=OUTLINE, lw=1.0)], labels=["recorded pose"],
                   loc="center", bbox_to_anchor=(0.83, (y - 0.16) / height), frameon=False, fontsize=7,
                   handlelength=2.2)
        y -= cbar_h
    fig.text(0.5, (y - title_h / 2) / height, "Collaborative workcell", ha="center", va="center")
    y -= title_h + work_h
    ax = fig.add_axes([0, y / height, 1, work_h / height])
    ax.imshow(work)
    ax.axis("off")
    # border around the whole figure
    lw = 0.8
    dx, dy = lw / 72 / 2 / width, lw / 72 / 2 / height  # inset by half the line width, so it is not clipped
    fig.patches.append(Rectangle((dx, dy), 1 - 2 * dx, 1 - 2 * dy, transform=fig.transFigure, fill=False,
                                 ec="black", lw=lw, zorder=10, clip_on=False))
    FIGURES.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT.with_suffix(f".{ext}"), dpi=300)
    print(f"saved {OUT}.pdf/.png")


if __name__ == "__main__":
    main()
