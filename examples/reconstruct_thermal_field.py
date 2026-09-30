"""Reconstruct a thermal field and print its temperature range."""

import argparse

import numpy as np

from _common import CASES, THERMAL, load_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, default="GeoC_p45")
    parser.add_argument("--segment", type=int, default=0)
    parser.add_argument("--time", type=float, help="evaluation time in seconds")
    parser.add_argument("--resolution", type=int, default=25)
    args = parser.parse_args()

    model = load_model(args.case)
    segments = sorted(model.pod.s_range_by_seg)
    if args.segment not in segments:
        parser.error(f"segment {args.segment} not found; available segments: {segments}")

    t_start, t_end = model.pod.get_seg_range(args.segment)
    t_eval = args.time if args.time is not None else 0.5 * (t_start + t_end)
    if not t_start <= t_eval <= t_end:
        parser.error(f"time must lie in segment range [{t_start:.3f}, {t_end:.3f}] s")

    x = np.linspace(model.pod.x_coords.min(), model.pod.x_coords.max(), args.resolution)
    y = np.linspace(model.pod.y_coords.min(), model.pod.y_coords.max(), args.resolution)
    xx, yy = np.meshgrid(x, y)
    temperature = np.empty_like(xx)

    for row in range(args.resolution):
        for col in range(args.resolution):
            rise = model.get_temp_gaussian(
                (xx[row, col], yy[row, col], THERMAL["z0"], t_eval),
            )
            temperature[row, col] = 25.0 + rise

    print(f"Case: {args.case}; segment: {args.segment}; time: {t_eval:.3f} s")
    print(f"Field grid: {args.resolution} x {args.resolution}")
    print(f"Temperature: min={temperature.min():.2f} °C, "
          f"max={temperature.max():.2f} °C, mean={temperature.mean():.2f} °C")


if __name__ == "__main__":
    main()
