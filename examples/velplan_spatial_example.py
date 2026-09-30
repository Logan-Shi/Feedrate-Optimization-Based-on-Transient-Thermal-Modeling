"""Run a short VelPlanSpatial feedrate example for one dataset segment."""

import argparse

import numpy as np

from _common import CASES, PLANNER_CONFIG, TEMP_SCALE, THERMAL, load_model
from velPlanSpatial import VelPlanSpatial


def make_inputs(model, segment, nodes):
    s_start, s_end = model.pod.get_seg_range(segment)
    coarse_s = np.linspace(s_start, s_end, 50)
    centers = model.pod.get_params_list(coarse_s)[:, 1:3]
    path_length = float(np.sum(np.linalg.norm(np.diff(centers, axis=0), axis=1)))
    if path_length <= 0:
        raise ValueError("The selected heat-flux segment has zero path length")

    s_nodes = np.linspace(s_start, s_end, nodes)
    nominal_path_speed = 0.01  # m/s, used to set the time-step scale
    mean_parameter_speed = nominal_path_speed * (s_end - s_start) / path_length
    ds = float(s_nodes[1] - s_nodes[0])
    dt = ds / mean_parameter_speed

    params = {
        "mu_x": model.pod.get_spline(1, segment),
        "mu_y": model.pod.get_spline(2, segment),
        "Q": model.pod.get_spline(0, segment),
        "sigma_x": model.pod.get_spline(3, segment),
        "sigma_y": model.pod.get_spline(4, segment),
        "theta": model.pod.get_spline(5, segment),
        "pod": model.pod,
        "segment": int(segment),
        "s_range": (float(s_start), float(s_end)),
    }
    return s_start, s_end, s_nodes, path_length, ds, dt, params


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, default="GeoC_p45")
    parser.add_argument("--segment", type=int, default=0)
    parser.add_argument("--nodes", type=int, default=12)
    parser.add_argument("--optimize", action="store_true",
                        help="run sequential feedrate optimization; otherwise simulate a constant profile")
    args = parser.parse_args()
    if args.nodes < 2:
        parser.error("--nodes must be at least 2")

    model = load_model(args.case)
    segments = sorted(model.pod.s_range_by_seg)
    if args.segment not in segments:
        parser.error(f"segment {args.segment} not found; available segments: {segments}")

    s_start, s_end, s_nodes, path_length, ds, dt, params = make_inputs(
        model, args.segment, args.nodes
    )
    config = PLANNER_CONFIG[args.case]
    planner = VelPlanSpatial(
        S_INIT=s_start,
        T0=25.0,
        model=model,
        thermal_integrand_ceiling=THERMAL["integrand_ceiling"],
    )
    alpha = THERMAL["kappa"] / (THERMAL["rho"] * THERMAL["c"])

    if args.optimize:
        result = planner.optimize_velocity_profile_sequential(
            n_variables=args.nodes,
            params_spline=params,
            z0=THERMAL["z0"],
            alpha=alpha,
            kappa=THERMAL["kappa"],
            beta=THERMAL["beta"],
            ds=ds,
            dt=dt,
            v_range=config["v_range"],
            T_ref=150.0,
            temperature_margin=50.0,
            temperature_lower=120.0,
            temperature_upper=200.0,
            p_a_max=config["p_a_max"],
            p_v_min=config["p_v_min"],
            p_v_max=config["p_v_max"],
            S_FINAL=s_end,
            default_v=config["default_v"],
            temp_scale=TEMP_SCALE[args.case],
            enable_backtracking=True,
            max_backtracking_steps=20,
            solver_maxiter=12,
            segment_timeout_s=180.0,
            show_progress=True,
        )
        temperatures, feedrates = result["temp"], result["v"]
        label = "optimized"
        elapsed = result["metrics"].get("total_computation_time_s")
    else:
        history_s = []
        history_v = []
        temperatures = []
        feedrates = []
        nominal_v = config["default_v"]
        for _ in range(args.nodes):
            temp, _, history_s, history_v = planner.temp_next_v(
                history_s,
                history_v,
                nominal_v,
                params,
                THERMAL["z0"],
                alpha,
                THERMAL["kappa"],
                THERMAL["beta"],
                ds=ds,
                dt=dt,
                use_sqrt_gauss=True,
                sqrt_gauss_nodes=8,
                temp_scale=TEMP_SCALE[args.case],
            )
            temperatures.append(temp)
            feedrates.append(nominal_v)
        temperatures = np.asarray(temperatures)
        feedrates = np.asarray(feedrates)
        label = "constant feedrate"
        elapsed = None

    print(f"Case: {args.case}; segment: {args.segment}; mode: {label}")
    print(f"Path length: {path_length:.4f} m; parameter range: [{s_start:.3f}, {s_end:.3f}]")
    if temperatures.size:
        print(f"Contact temperature: start={temperatures[0]:.2f} °C, "
              f"final={temperatures[-1]:.2f} °C, max={temperatures.max():.2f} °C")
    if feedrates.size:
        print(f"Feedrate: min={feedrates.min():.4f}, max={feedrates.max():.4f}, "
              f"mean={feedrates.mean():.4f} (ds/dt)")
    if elapsed is not None:
        print(f"Optimizer computation time: {elapsed:.2f} s")


if __name__ == "__main__":
    main()
