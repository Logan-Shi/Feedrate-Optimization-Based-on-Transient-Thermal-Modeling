from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from moving_heat import MovingHeatSource

CASES = ("GeoA", "GeoB", "GeoC_p45", "GeoC_0", "GeoC_m45")

# Parameters used for the paper's feedrate-optimization examples.
THERMAL = {
    "rho": 1300.0,
    "c": 916.0,
    "kappa": 0.045,
    "beta": 4.22,
    "ratio_sigma": 1.0,
    "ratio_q": 0.56,
    "z0": 1.0e-6,
    "integrand_ceiling": 25.7838653902,
}

TEMP_SCALE = {
    "GeoA": 1.0,
    "GeoB": 1.0,
    "GeoC_p45": 1.0,
    "GeoC_0": 1.0,
    "GeoC_m45": 1.0,
}

PLANNER_CONFIG = {
    "GeoA": {
        "v_range": (0.4, 3.0), "p_a_max": 300.0,
        "p_v_min": 0.0, "p_v_max": 0.0075, "default_v": 1.0,
    },
    "GeoB": {
        "v_range": (0.4, 3.0), "p_a_max": 300.0,
        "p_v_min": 0.0, "p_v_max": 0.0075, "default_v": 1.0,
    },
    "GeoC_p45": {
        "v_range": (0.4, 3.0), "p_a_max": 300.0,
        "p_v_min": 0.0, "p_v_max": 0.0075, "default_v": 1.0,
    },
    "GeoC_0": {
        "v_range": (0.4, 3.0), "p_a_max": 300.0,
        "p_v_min": 0.0, "p_v_max": 0.0075, "default_v": 1.0,
    },
    "GeoC_m45": {
        "v_range": (0.4, 3.0), "p_a_max": 300.0,
        "p_v_min": 0.0, "p_v_max": 0.0075, "default_v": 1.0,
    },
}


def load_model(case):
    model = MovingHeatSource(
        THERMAL["rho"],
        THERMAL["c"],
        THERMAL["kappa"],
        THERMAL["beta"],
        ratio_sigma=THERMAL["ratio_sigma"],
        ratio_q=THERMAL["ratio_q"],
    )
    model.load_gaussian(ROOT / "data" / "heat_flux_fits" / f"{case}.npz")
    return model
