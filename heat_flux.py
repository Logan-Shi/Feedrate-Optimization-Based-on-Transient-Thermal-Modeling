"""Load and interpolate the fitted Gaussian heat-flux datasets."""

import numpy as np
from scipy.interpolate import UnivariateSpline


class GaussianHeatFlux:
    """Segmented spline interpolator for saved Gaussian heat-flux fits."""

    @classmethod
    def load(cls, filename):
        """Load a fitted model from an ``.npz`` file."""
        with np.load(filename, allow_pickle=False) as data:
            instance = object.__new__(cls)
            instance.params_all = data["params_all"]
            instance.s_samples = data["s_samples"]
            instance.x_coords = data["x_coords"]
            instance.y_coords = data["y_coords"]
            instance.n_gaussians = int(data["n_gaussians"])
            instance.N = instance.params_all.shape[0]
            instance.nx = instance.x_coords.size
            instance.ny = instance.y_coords.size
            seg_idx = data.get("seg_idx")
            instance.seg_idx = None if seg_idx is None or seg_idx.size == 0 else seg_idx.astype(int)

        instance._build_segment_groups()
        instance._fit_splines()
        return instance

    def _build_segment_groups(self):
        if self.seg_idx is None:
            self.unique_segments = np.array([0], dtype=int)
            indices_by_segment = {0: np.arange(self.N)}
        else:
            self.unique_segments = np.unique(self.seg_idx)
            indices_by_segment = {
                int(segment): np.flatnonzero(self.seg_idx == segment)
                for segment in self.unique_segments
            }

        self.s_by_seg = {}
        self.params_by_seg = {}
        self.s_range_by_seg = {}
        for segment in self.unique_segments:
            indices = indices_by_segment[int(segment)]
            order = np.argsort(self.s_samples[indices])
            s_values = self.s_samples[indices][order]
            params = self.params_all[indices][order]
            self.s_by_seg[int(segment)] = s_values
            self.params_by_seg[int(segment)] = params
            self.s_range_by_seg[int(segment)] = (
                float(s_values[0]), float(s_values[-1])
            )

    def _fit_splines(self):
        valid = ~np.any(np.isnan(self.params_all), axis=1)
        if not np.all(valid):
            self.params_all = self.params_all[valid]
            self.s_samples = self.s_samples[valid]
            if self.seg_idx is not None:
                self.seg_idx = self.seg_idx[valid]
            self.N = len(self.s_samples)
            self._build_segment_groups()

        self.param_splines_by_seg = {}
        for segment in self.unique_segments:
            segment = int(segment)
            s_values = self.s_by_seg[segment]
            params = self.params_by_seg[segment]
            order = 3 if len(s_values) >= 4 else 1
            self.param_splines_by_seg[segment] = [
                UnivariateSpline(s_values, params[:, index], k=order, s=0)
                for index in range(6 * self.n_gaussians)
            ]

    def _select_segment(self, s_value):
        if self.seg_idx is None or len(self.unique_segments) == 1:
            return int(self.unique_segments[0])
        for segment, (s_min, s_max) in self.s_range_by_seg.items():
            if s_min <= s_value <= s_max:
                return segment
        return min(
            self.s_range_by_seg,
            key=lambda segment: min(
                abs(s_value - self.s_range_by_seg[segment][0]),
                abs(s_value - self.s_range_by_seg[segment][1]),
            ),
        )

    def _eval_params(self, s_value):
        s_value = float(np.asarray(s_value).item())
        segment = self._select_segment(s_value)
        s_min, s_max = self.s_range_by_seg[segment]
        if not s_min <= s_value <= s_max:
            return np.zeros(6 * self.n_gaussians)

        params = np.array([
            spline(s_value)
            for spline in self.param_splines_by_seg[segment]
        ])
        for index in range(self.n_gaussians):
            params[6 * index + 3] = max(params[6 * index + 3], 1e-6)
            params[6 * index + 4] = max(params[6 * index + 4], 1e-6)
        return params

    def get_spline(self, index, segment):
        """Return the fitted parameter spline for one segment."""
        segment = int(segment)
        if segment not in self.param_splines_by_seg:
            segment -= 1
        return self.param_splines_by_seg[segment][index]

    def get_params_list(self, s_values):
        """Evaluate all fitted Gaussian parameters at the supplied samples."""
        s_values = np.atleast_1d(np.asarray(s_values, dtype=float))
        result = np.zeros((len(s_values), 6 * self.n_gaussians))
        for segment in self.unique_segments:
            segment = int(segment)
            s_min, s_max = self.s_range_by_seg[segment]
            mask = (s_values >= s_min) & (s_values <= s_max)
            if np.any(mask):
                splines = self.param_splines_by_seg[segment]
                result[mask] = np.column_stack([
                    spline(s_values[mask]) for spline in splines
                ])

        for index in range(self.n_gaussians):
            result[:, 6 * index + 3] = np.maximum(result[:, 6 * index + 3], 1e-6)
            result[:, 6 * index + 4] = np.maximum(result[:, 6 * index + 4], 1e-6)
        return result

    def get_seg_range(self, segment):
        """Return the interpolation range for a segment index."""
        segment = int(segment)
        if segment not in self.s_range_by_seg:
            segment -= 1
        return self.s_range_by_seg[segment]
