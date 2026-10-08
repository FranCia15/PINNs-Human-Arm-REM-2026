"""
anthropometry.py

Per-segment inertial parameters (mass, center-of-mass location, radius
of gyration) for the 3-rigid-body arm chain (upper arm, forearm, hand)
used by dynamics.py.

Values are from de Leva (1996), "Adjustments to Zatsiorsky-Seluyanov's
segment inertia parameters", J. Biomech. 29(9):1223-1230, Table 4 --
gender-averaged (mean of the male/female rows), since DBAS22 does not
record subject gender. 

Note: the "hand" row is defined wrist-to-3rd-metacarpal
(a knuckle landmark), not wrist-to-fingertip, and is not the same
landmark as wri2arrow (wrist-to-VR-end-effector-marker) used elsewhere
in this codebase; wri2arrow is a stand-in approximation, see
dynamics.py's docstring.

DBAS22 also does not record body mass, estimated here from height via
a population-average BMI. This is the least reliable input.
"""

import numpy as np

ASSUMED_BMI = 22.5  # kg/m^2, healthy adult population average

# De Leva (1996) Table 4, gender-averaged. Each entry:
#   mass_frac: segment mass / total body mass
#   com_frac:  center-of-mass distance from the proximal joint, as a
#              fraction of segment length
#   rog_frac:  radius of gyration about the (sagittal, transverse,
#              longitudinal) principal axes through the COM, as a
#              fraction of segment length

SEGMENT_PARAMS = {
    "upper_arm": {
        "mass_frac": 0.0263,
        "com_frac": 0.5763,
        "rog_frac": (0.2815, 0.2645, 0.1530),
    },
    "forearm": {
        "mass_frac": 0.0150,
        "com_frac": 0.45665,
        "rog_frac": (0.2685, 0.2610, 0.1075),
    },
    "hand": {
        "mass_frac": 0.00585,
        "com_frac": 0.7687,
        "rog_frac": (0.5795, 0.4835, 0.3680),
    },
}


def estimate_body_mass(height_m: float, bmi: float = ASSUMED_BMI) -> float:
    """Population-average mass estimate from height alone (no measured
    body mass in DBAS22)."""
    return bmi * height_m ** 2


def segment_inertial_params(segment: str, length_m: float, body_mass_kg: float) -> dict:
    """
    Returns {"mass": kg, "com_frac": fraction of length from the
    proximal joint, "inertia": (3,) principal moments of inertia
    (sagittal, transverse, longitudinal), kg*m^2} for one segment.
    """
    p = SEGMENT_PARAMS[segment]
    mass = p["mass_frac"] * body_mass_kg
    rog = np.array(p["rog_frac"]) * length_m
    inertia = mass * rog ** 2
    return {"mass": mass, "com_frac": p["com_frac"], "inertia": inertia}
