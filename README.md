# PINNs for Human Arm Motion Reconstruction in Digital Twins of Human-Robot Collaborative Systems

Code, trained models and results for the paper

> Ciampi, F.G., Diallo, T.M.L., and Plateaux, R., *A Physics-Informed Neural Network for Human Arm Motion Reconstruction in Digital Twins of Human-Robot Collaborative Systems*, 24th IEEE International Conference on Research & Education in Mechatronics, 12–13 November 2026, Eindhoven, The Netherlands, 2026.

A physics-informed neural network (PINN) reconstructs the 7-DOF human arm joint trajectory (shoulder pitch/roll, arm yaw, elbow pitch, forearm yaw, wrist pitch/roll) from a tracked hand trajectory and subject-specific anthropometry. A minimum-effort term based on a rigid-body (Euler–Lagrange) model of the arm resolves the arm's kinematic redundancy. The PINN is compared with a fixed-swivel inverse-kinematics baseline and a gradient-projection dynamics baseline on the DBAS22 / 3D-ARM-Gaze reaching dataset, and the reconstructed motion is played on a mannequin in a RoboDK digital twin.

## Repository layout

```
configs/      training configurations of the two models
models/       trained models: paper_model (the model of the paper) and
              ablation_noeffort (same, trained without the effort term)
src/          training, baselines, evaluation, figures
robodk/       RoboDK station, rig mapping, Fig. 3 poses and screenshots
results/      computed results used in the paper (see below)
figures/      Fig. 2 and Fig. 3, written by the scripts (not included)
```

## Setup

```
conda env create -f environment.yml
conda activate pinn_arm_rem2026
```

### Dataset

DBAS22 / 3D-ARM-Gaze is third-party data and is not redistributed here. 
Download it from Zenodo, [doi:10.5281/zenodo.10567365](https://doi.org/10.5281/zenodo.10567365), and place the per-subject folders directly under `DBAS22_DataOnline/` at the repository root (`DBAS22_DataOnline/s1/subj_info.json`, ...). 
The data of the initial-acquisition phase are used, after the dataset's own exclusion files.

All commands below are run from the repository root.

## Data splits

`results/splits.json` (written by `python src/export_splits.py`) lists, as
trial target numbers per subject, the training trials of the nine training
subjects (S1, S2, S7, S9, S12–S15, S20; their first 20 valid trials) and
the three evaluation sets of Table I (20 trials per subject, every frame):

| scenario | subjects | trials |
|---|---|---|
| training | S1, S12, S13, S20 | their 20 training trials |
| heldout (unseen trials) | S1, S12, S13, S20 | the 20 trials after the training ones |
| unseen (subjects) | S8, S11, S16, S17 | their first 20 trials |

The sets are defined once, in `src/common.py`.

## Reproducing the paper

The results files in `results/` are included, so tables and figures can be
regenerated in seconds; the "from scratch" commands recompute them.

The figure files themselves are not included, since the copyright of the
published figures belongs to the publisher; the scripts below write them to
`figures/`. See the paper for the published versions.

| paper item | from the included results | from scratch (runtime, CPU) |
|---|---|---|
| Table I | `python src/make_joint_error_table.py` | `python src/eval_joint_errors.py` (~5 min) and `python src/eval_dynamics.py` (~50 min with 12 processes) |
| Sec. V-A, paired tests | `python src/paired_tests.py` | the above, then `python src/eval_cartesian_errors.py` (~1 min) |
| Sec. V-A, swivel-angle and hand errors | (printed by) `python src/eval_cartesian_errors.py` | as above |
| Sec. V-B, ablation of the effort term | `python src/make_ablation.py` | `python src/eval_joint_errors.py ablation_noeffort` (~2 min) and `python src/eval_torque_metrics.py` (~10 min with 12 processes) |
| Fig. 2 | – | `python src/plot_torque.py` (~1 min) |
| Fig. 3 | `python src/make_pose_figure.py` (`--plain` without annotations) | `python src/export_poses.py`, then the RoboDK screenshots (below) |

Results files:

| file | content | written by |
|---|---|---|
| `joint_errors_paper_model.npz` | PINN and IK joint angles and errors, Table I trials | `eval_joint_errors.py` |
| `joint_errors_ablation_noeffort.npz` | same, model without the effort term (PINN only) | `eval_joint_errors.py ablation_noeffort` |
| `dynamics_{training,heldout,unseen}.npz` | dynamics-baseline joint angles, errors, torques | `eval_dynamics.py` |
| `cartesian_errors.npz` | elbow / wrist / hand, hand orientation and swivel-angle errors | `eval_cartesian_errors.py` |
| `torque_metrics.npz` | torque ripple, nRMSE, correlation vs. the recorded-motion reference | `eval_torque_metrics.py` |
| `joint_error_table.csv`, `splits.json` | Table I; data splits | `make_joint_error_table.py`; `export_splits.py` |

## Training

```
python src/train_pinn.py --config configs/paper_model.json
python src/train_pinn.py --config configs/ablation_noeffort.json
```

Each run writes `runs/<timestamp>_<tag>/` (`model.pt`, `history.npz`,
`config.json`, the same layout as `models/`). The paper's models took about
5 and 7 hours on a CPU. Training is seeded, but results can differ slightly
across hardware and library versions; the evaluation scripts use the
models in `models/`.

## RoboDK digital twin (Sec. VI, Fig. 3)

`robodk/Simulation_roboDK.rdk` is the station with the mannequin. The
scripts in `robodk/` connect to a running RoboDK through its Python API
(`pip install robodk`, included in `environment.yml`) and are run from the
`robodk/` folder:

| script | purpose |
|---|---|
| `fix_segments.py` | re-attaches the mannequin's segments after the station is opened (RoboDK does not save them correctly); run once after each opening |
| `rig_mapping.py` | maps the dataset's joint angles onto the mannequin: each segment takes the orientation given by the dataset's forward kinematics, corrected by the rig's measured rest bones |
| `dump_rig_geometry.py` | measures the rig (`rig_geometry.json`) |
| `capture_poses.py` | poses the mannequin for the 12 screenshots of Fig. 3 (taken manually) and checks each rendered pose against the offline prediction |
| `play_trajectory.py` | plays a trajectory exported by `src/export_trajectory.py`, paced by its recorded timestamps |

`src/check_rig_rendering.py` checks the rendering offline, without RoboDK.
`robodk/screenshots/` holds the raw screenshots that `src/make_pose_figure.py`
assembles into Fig. 3, and the workcell view; `robodk/poses/` the exported
Fig. 3 poses.

Note: the IK baseline of the Fig. 3 poses is calibrated on all of S1's
trials (the calibration in use when the figure was produced), not on the
training data as in Table I. It is kept so that the published figure is
reproduced exactly.

## License

MIT, except `src/tool_box/rot_quat_utils.py` from the DBAS22 authors'
code release (Apache 2.0); see [LICENSE](LICENSE).
