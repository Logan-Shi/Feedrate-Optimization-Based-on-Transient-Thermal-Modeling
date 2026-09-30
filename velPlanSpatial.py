import time
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import erfc
from tqdm import tqdm

maximum_temp_history_time = 900


class VelPlanSpatial:
    """
    Class for velocity planning using spatial parameterization v(s).
    """
    _PARAMETER_NAMES = ("Q", "mu_x", "mu_y", "sigma_x", "sigma_y", "theta")
    _SQRT_GAUSS_CACHE = {}

    def __init__(self, T0=25.0, S_INIT=0.0, model=None,
                 thermal_integrand_ceiling=None):
        self.T0 = T0
        self.S_INIT = S_INIT
        self.model = model
        self.ratio_q = model.ratio_q if model is not None else 1.0
        self.ratio_sigma = model.ratio_sigma if model is not None else 1.0
        self.thermal_integrand_ceiling = self._normalize_thermal_integrand_ceiling(
            thermal_integrand_ceiling
        )

    @staticmethod
    def _normalize_thermal_integrand_ceiling(ceiling):
        if ceiling is None:
            return None
        ceiling = float(ceiling)
        if not np.isfinite(ceiling) or ceiling <= 0.0:
            raise ValueError("thermal_integrand_ceiling must be a positive finite value")
        return ceiling

    def _apply_thermal_integrand_ceiling(self, integrand, derivative=None):
        ceiling = self.thermal_integrand_ceiling
        if ceiling is None:
            if derivative is None:
                return integrand
            return integrand, derivative
        capped_mask = np.asarray(integrand, dtype=float) > ceiling
        capped_integrand = np.minimum(integrand, ceiling)
        if derivative is None:
            return capped_integrand
        capped_derivative = np.where(capped_mask, 0.0, derivative)
        return capped_integrand, capped_derivative

    @classmethod
    def _sqrt_gauss_rule(cls, n_nodes):
        n_nodes = int(n_nodes)
        if n_nodes <= 0:
            raise ValueError("sqrt_gauss_nodes must be positive")
        rule = cls._SQRT_GAUSS_CACHE.get(n_nodes)
        if rule is None:
            rule = np.polynomial.legendre.leggauss(n_nodes)
            cls._SQRT_GAUSS_CACHE[n_nodes] = rule
        return rule

    @staticmethod
    def _segment_range_from_params(params_spline):
        if not isinstance(params_spline, dict):
            return None
        if "s_range" in params_spline:
            s_range = params_spline["s_range"]
        elif "segment_range" in params_spline:
            s_range = params_spline["segment_range"]
        elif "S_INIT" in params_spline and "S_FINAL" in params_spline:
            s_range = (params_spline["S_INIT"], params_spline["S_FINAL"])
        else:
            return None
        if s_range is None or len(s_range) != 2:
            return None
        s_min, s_max = map(float, s_range)
        if not (np.isfinite(s_min) and np.isfinite(s_max)):
            return None
        return (min(s_min, s_max), max(s_min, s_max))

    @staticmethod
    def _params_bundle(mu_x, mu_y, Q_spline, sigma_x_spline, sigma_y_spline,
                       theta_spline, params_spline=None):
        if isinstance(params_spline, dict):
            return params_spline
        return {
            "mu_x": mu_x,
            "mu_y": mu_y,
            "Q": Q_spline,
            "sigma_x": sigma_x_spline,
            "sigma_y": sigma_y_spline,
            "theta": theta_spline,
        }

    def _eval_pod_params_on_s(self, s_values, pod):
        s_values = np.asarray(s_values, dtype=float)
        out = {
            name: np.zeros(s_values.shape, dtype=float)
            for name in self._PARAMETER_NAMES
        }
        if not hasattr(self, "_params_cache"):
            self._params_cache = {}
        for idx, st in enumerate(s_values):
            cache_key = (id(pod), float(st))
            if cache_key in self._params_cache:
                params_array = self._params_cache[cache_key]
            else:
                params_array = np.asarray(pod._eval_params(float(st)), dtype=float)
                self._params_cache[cache_key] = params_array
            if params_array.size < 6:
                continue
            out["Q"][idx] = params_array[0]
            out["mu_x"][idx] = params_array[1]
            out["mu_y"][idx] = params_array[2]
            out["sigma_x"][idx] = params_array[3]
            out["sigma_y"][idx] = params_array[4]
            out["theta"][idx] = params_array[5]
        return out

    def _eval_segmented_params_on_s(self, s_values, params_spline):
        s_values = np.asarray(s_values, dtype=float)
        original_shape = s_values.shape
        flat_s = s_values.reshape(-1)
        out = {
            name: np.zeros(flat_s.shape, dtype=float)
            for name in self._PARAMETER_NAMES
        }
        if flat_s.size == 0:
            return {name: values.reshape(original_shape) for name, values in out.items()}

        pod = None
        if self.model is not None:
            pod = getattr(self.model, "pod", None)
        if pod is None and isinstance(params_spline, dict):
            pod = params_spline.get("pod")
        if pod is not None and hasattr(pod, "_eval_params"):
            out = self._eval_pod_params_on_s(flat_s, pod)
        elif isinstance(params_spline, dict):
            valid_mask = np.ones(flat_s.shape, dtype=bool)
            s_range = self._segment_range_from_params(params_spline)
            if s_range is not None:
                s_min, s_max = s_range
                valid_mask = (flat_s >= s_min) & (flat_s <= s_max)

            if np.any(valid_mask):
                valid_s = flat_s[valid_mask]
                for name in self._PARAMETER_NAMES:
                    out[name][valid_mask] = np.asarray(params_spline[name](valid_s), dtype=float)
        else:
            raise TypeError("params_spline must be a dictionary of parameter splines")

        # Override the source-center trajectory with the recorded contact point
        # (c_x, c_y) when provided, so temperature evaluation uses the toolpath
        # position instead of the fitted Gaussian mu_x/mu_y.
        if isinstance(params_spline, dict):
            c_x = params_spline.get("c_x")
            c_y = params_spline.get("c_y")
            if c_x is not None and c_y is not None:
                out["mu_x"] = np.asarray(c_x(flat_s), dtype=float)
                out["mu_y"] = np.asarray(c_y(flat_s), dtype=float)

        return {name: values.reshape(original_shape) for name, values in out.items()}

    def _source_segment_ids_on_s(self, s_values, params_spline):
        s_values = np.asarray(s_values, dtype=float)
        original_shape = s_values.shape
        flat_s = s_values.reshape(-1)
        segment_ids = np.full(flat_s.shape, -1, dtype=int)
        if flat_s.size == 0:
            return segment_ids.reshape(original_shape)

        pod = None
        if self.model is not None:
            pod = getattr(self.model, "pod", None)
        if pod is None and isinstance(params_spline, dict):
            pod = params_spline.get("pod")
        if pod is not None and hasattr(pod, "s_range_by_seg"):
            for seg, (s_min, s_max) in pod.s_range_by_seg.items():
                s_min = float(s_min)
                s_max = float(s_max)
                mask = (flat_s >= min(s_min, s_max)) & (flat_s <= max(s_min, s_max))
                segment_ids[mask] = int(seg)
            return segment_ids.reshape(original_shape)

        s_range = self._segment_range_from_params(params_spline)
        if s_range is not None:
            s_min, s_max = s_range
            segment_ids[(flat_s >= s_min) & (flat_s <= s_max)] = 0
        else:
            segment_ids[:] = 0
        return segment_ids.reshape(original_shape)

    # ------------------------------------------------------------
    # Green's function G1
    # ------------------------------------------------------------
    def G1(self, z0, dtau, alpha, kappa, beta):
        a = alpha
        t = np.maximum(dtau, 1e-12)
        term1 = np.exp(-z0**2/(4*a*t)) / (4*kappa*np.sqrt(a)*(np.pi*t)**1.5)
        term2 = beta*np.exp((t*beta**2*a)/(kappa**2) + (beta*z0)/kappa) / (4*kappa**2*np.pi*t)
        arg = z0/(2*np.sqrt(a*t)) + np.sqrt(a*t)*beta/kappa
        return term1 - term2 * erfc(arg)

    # ------------------------------------------------------------
    # Forward model with control s(t) (uniform dt discrete sum)
    # ------------------------------------------------------------
    def forward_temperature_sqrt_gauss(self, s, mu_x, mu_y, Q_spline, sigma_x_spline, sigma_y_spline, theta_spline,
                                       tgrid, z0, alpha, kappa, beta, eps=1e-12,
                                       sqrt_gauss_nodes=2, params_spline=None):
        """
        Continuous-time integral with eta = t_end - tau = u^2:
            T = integral 2u * G1(z0, u^2) * J(t_end - u^2) du
        This composite fixed-node rule applies the transformed integral on
        each existing time-grid interval, avoiding the singular endpoint
        sample while retaining the toolpath time resolution.
        """
        t_end = float(tgrid[-1])
        if t_end <= 0.0:
            return 0.0

        t_arr = np.asarray(tgrid, dtype=float)
        s_arr = np.asarray(s, dtype=float)
        if t_arr.size < 2:
            return 0.0

        tau_start = max(float(t_arr[0]), t_end - maximum_temp_history_time)
        tau_lo = np.maximum(t_arr[:-1], tau_start)
        tau_hi = np.minimum(t_arr[1:], t_end)
        active = tau_hi > tau_lo
        if not np.any(active):
            return 0.0

        tau_lo = tau_lo[active]
        tau_hi = tau_hi[active]
        u_lo = np.sqrt(np.maximum(t_end - tau_hi, 0.0))
        u_hi = np.sqrt(np.maximum(t_end - tau_lo, 0.0))

        nodes, weights = self._sqrt_gauss_rule(sqrt_gauss_nodes)
        u_mid = 0.5 * (u_hi + u_lo)
        u_half_width = 0.5 * (u_hi - u_lo)
        u_arr = (u_mid[:, None] + u_half_width[:, None] * nodes[None, :]).reshape(-1)
        w_arr = (u_half_width[:, None] * weights[None, :]).reshape(-1)
        dtau_arr = np.maximum(u_arr * u_arr, eps)
        tau_arr = t_end - dtau_arr

        s_tau_arr = np.interp(tau_arr, t_arr, s_arr)

        params_bundle = self._params_bundle(
            mu_x, mu_y, Q_spline, sigma_x_spline, sigma_y_spline,
            theta_spline, params_spline
        )
        params_at_tau = self._eval_segmented_params_on_s(s_tau_arr, params_bundle)
        source_segment_ids = self._source_segment_ids_on_s(s_tau_arr, params_bundle)

        final_params = self._eval_segmented_params_on_s([s_arr[-1]], params_bundle)
        phi_final = final_params["mu_x"][0]
        psi_final = final_params["mu_y"][0]

        params_Q = params_at_tau["Q"]
        params_sigma_x = params_at_tau["sigma_x"]
        params_sigma_y = params_at_tau["sigma_y"]
        params_mu_x = params_at_tau["mu_x"]
        params_mu_y = params_at_tau["mu_y"]
        params_theta = params_at_tau["theta"]

        A = params_Q * self.ratio_q
        sigma_x = params_sigma_x * self.ratio_sigma
        sigma_y = params_sigma_y * self.ratio_sigma

        Q_arr = A * 1e6 * (2 * np.pi * sigma_x * sigma_y)
        cos_t = np.cos(params_theta)
        sin_t = np.sin(params_theta)

        x_shift = phi_final - params_mu_x
        y_shift = psi_final - params_mu_y

        x_rot = cos_t * x_shift + sin_t * y_shift
        y_rot = -sin_t * x_shift + cos_t * y_shift

        denom = np.sqrt(sigma_x**2 + 2*alpha*dtau_arr) * np.sqrt(sigma_y**2 + 2*alpha*dtau_arr)
        prefactor = Q_arr * (2*alpha*dtau_arr) / denom
        exponent = -x_rot**2 / (2*sigma_x**2 + 2*alpha*dtau_arr) - y_rot**2 / (2*sigma_y**2 + 2*alpha*dtau_arr)

        J_arr = prefactor * np.exp(exponent)
        J_arr[A <= 0.0] = 0.0

        base_integrand = self.G1(z0, dtau_arr, alpha, kappa, beta) * J_arr
        valid_mask = (
            (source_segment_ids >= 0)
            & np.isfinite(base_integrand)
            & (base_integrand > 0.0)
        )
        base_integrand = np.where(valid_mask, base_integrand, 0.0)
        base_integrand = self._apply_thermal_integrand_ceiling(base_integrand)
        return float(np.sum(w_arr * 2.0 * u_arr * base_integrand))

    def forward_temperature_trapz(self, s, mu_x, mu_y, Q_spline, sigma_x_spline, sigma_y_spline, theta_spline,
                                 tgrid, z0, alpha, kappa, beta, params_spline=None):
        """
        Continuous-time integral:
            T = ∫_0^{t_end} G1(z0, t_end - τ) * J(τ) dτ
        Uses fast vectorized np.trapz with parameters caching.
        """
        t_end = float(tgrid[-1])
        s_end = s[-1]

        t_arr = np.asarray(tgrid)
        s_arr = np.asarray(s)

        # Only integrate over the history window
        valid_idx = np.searchsorted(t_arr, t_end - maximum_temp_history_time)
        tau_arr = t_arr[valid_idx:]
        s_tau_arr = s_arr[valid_idx:]

        params_bundle = self._params_bundle(
            mu_x, mu_y, Q_spline, sigma_x_spline, sigma_y_spline,
            theta_spline, params_spline
        )
        params_at_tau = self._eval_segmented_params_on_s(s_tau_arr, params_bundle)
        params_Q = params_at_tau["Q"]
        params_sigma_x = params_at_tau["sigma_x"]
        params_sigma_y = params_at_tau["sigma_y"]
        params_mu_x = params_at_tau["mu_x"]
        params_mu_y = params_at_tau["mu_y"]
        params_theta = params_at_tau["theta"]

        final_params = self._eval_segmented_params_on_s([s_end], params_bundle)
        phi_final = final_params["mu_x"][0]
        psi_final = final_params["mu_y"][0]

        dtau_arr = t_end - tau_arr
        dtau_arr = np.maximum(dtau_arr, 1e-12)

        A = params_Q * self.ratio_q
        sigma_x = params_sigma_x * self.ratio_sigma
        sigma_y = params_sigma_y * self.ratio_sigma
        
        Q_arr = A * 1e6 * (2 * np.pi * sigma_x * sigma_y)
        cos_t = np.cos(params_theta)
        sin_t = np.sin(params_theta)
        
        x_shift = phi_final - params_mu_x
        y_shift = psi_final - params_mu_y
        
        x_rot = cos_t * x_shift + sin_t * y_shift
        y_rot = -sin_t * x_shift + cos_t * y_shift
        
        denom = np.sqrt(sigma_x**2 + 2*alpha*dtau_arr) * np.sqrt(sigma_y**2 + 2*alpha*dtau_arr)
        prefactor = Q_arr * (2*alpha*dtau_arr) / denom
        exponent = -x_rot**2 / (2*sigma_x**2 + 2*alpha*dtau_arr) - y_rot**2 / (2*sigma_y**2 + 2*alpha*dtau_arr)
        
        J_arr = prefactor * np.exp(exponent)
        J_arr[A <= 0.0] = 0.0
        
        g_arr = self.G1(z0, dtau_arr, alpha, kappa, beta)
        
        integrand_arr = g_arr * J_arr
        integrand_arr[t_end - tau_arr <= 0.0] = 0.0
        integrand_arr = self._apply_thermal_integrand_ceiling(integrand_arr)
        
        if tau_arr.size < 2:
            return 0.0
        segment_ids = self._source_segment_ids_on_s(s_tau_arr, params_bundle)
        same_source_segment = (
            (segment_ids[:-1] >= 0)
            & (segment_ids[1:] >= 0)
            & (segment_ids[:-1] == segment_ids[1:])
        )
        interval_values = 0.5 * (integrand_arr[:-1] + integrand_arr[1:]) * np.diff(tau_arr)
        val = float(np.sum(interval_values[same_source_segment]))
        return val

    def _validate_discrete_nodes(self, s_nodes, v_nodes):
        s_nodes = np.asarray(s_nodes, dtype=float)
        v_nodes = np.asarray(v_nodes, dtype=float)
        if s_nodes.ndim != 1 or v_nodes.ndim != 1:
            raise ValueError("s_nodes and v_nodes must be one-dimensional arrays")
        if s_nodes.size != v_nodes.size:
            raise ValueError("s_nodes and v_nodes must have the same length")
        if s_nodes.size == 0:
            raise ValueError("at least one node is required")
        if np.any(np.diff(s_nodes) <= 0.0):
            raise ValueError("s_nodes must be strictly increasing")
        if np.any(v_nodes <= 0.0):
            raise ValueError("v_nodes must be strictly positive")
        return s_nodes, v_nodes

    def discrete_time_map(self, s_nodes, v_nodes):
        """
        Node times from the trapezoidal discretization of dt/ds = 1/v(s).
        """
        s_nodes, v_nodes = self._validate_discrete_nodes(s_nodes, v_nodes)
        if s_nodes.size == 1:
            return np.array([0.0], dtype=float)
        ds = np.diff(s_nodes)
        dt = 0.5 * ds * (1.0 / v_nodes[:-1] + 1.0 / v_nodes[1:])
        return np.concatenate(([0.0], np.cumsum(dt)))

    def discrete_time_jacobian(self, s_nodes, v_nodes):
        """
        Analytic Jacobian J[n, j] = d t_n / d v_j for the node time map.
        """
        s_nodes, v_nodes = self._validate_discrete_nodes(s_nodes, v_nodes)
        n_nodes = s_nodes.size
        jac = np.zeros((n_nodes, n_nodes), dtype=float)
        for segment_idx, ds in enumerate(np.diff(s_nodes), start=1):
            left = segment_idx - 1
            right = segment_idx
            jac[segment_idx:, left] += -0.5 * ds / v_nodes[left] ** 2
            jac[segment_idx:, right] += -0.5 * ds / v_nodes[right] ** 2
        return jac

    def _eval_params_on_s(self, s_values, params_spline):
        return self._eval_segmented_params_on_s(s_values, params_spline)

    def _green_g1_and_derivative(self, dtau, z0, alpha, kappa, beta):
        dtau = np.asarray(dtau, dtype=float)
        value = np.zeros_like(dtau, dtype=float)
        derivative = np.zeros_like(dtau, dtype=float)
        mask = dtau > 0.0
        if not np.any(mask):
            return value, derivative

        t = np.maximum(dtau[mask], 1e-12)
        a = alpha
        b = z0**2 / (4.0 * a)
        term1 = np.exp(-b / t) / (4.0 * kappa * np.sqrt(a) * (np.pi * t) ** 1.5)
        d_term1 = term1 * (b / t**2 - 1.5 / t)

        lam = beta**2 * a / kappa**2
        exp_beta = np.exp(lam * t + beta * z0 / kappa)
        c2 = beta * exp_beta / (4.0 * kappa**2 * np.pi * t)
        arg = z0 / (2.0 * np.sqrt(a * t)) + np.sqrt(a * t) * beta / kappa
        d_arg = (
            -z0 / (4.0 * np.sqrt(a) * t**1.5)
            + beta * np.sqrt(a) / (2.0 * kappa * np.sqrt(t))
        )
        erfc_arg = erfc(arg)
        d_erfc = -2.0 / np.sqrt(np.pi) * np.exp(-arg**2) * d_arg
        d_c2 = c2 * (lam - 1.0 / t)
        term2 = c2 * erfc_arg
        d_term2 = d_c2 * erfc_arg + c2 * d_erfc

        value[mask] = term1 - term2
        derivative[mask] = d_term1 - d_term2
        return value, derivative

    def _heat_flux_j_and_derivative(self, dtau, params, alpha, phi_final, psi_final):
        dtau = np.asarray(dtau, dtype=float)
        value = np.zeros_like(dtau, dtype=float)
        derivative = np.zeros_like(dtau, dtype=float)

        q_density = np.asarray(params["Q"], dtype=float) * self.ratio_q
        sigma_x = np.asarray(params["sigma_x"], dtype=float) * self.ratio_sigma
        sigma_y = np.asarray(params["sigma_y"], dtype=float) * self.ratio_sigma
        mask = (dtau > 0.0) & (q_density > 0.0) & (sigma_x > 0.0) & (sigma_y > 0.0)
        if not np.any(mask):
            return value, derivative

        t = np.maximum(dtau[mask], 1e-12)
        sx = sigma_x[mask]
        sy = sigma_y[mask]
        theta = np.asarray(params["theta"], dtype=float)[mask]
        mu_x = np.asarray(params["mu_x"], dtype=float)[mask]
        mu_y = np.asarray(params["mu_y"], dtype=float)[mask]

        q_total = q_density[mask] * 1e6 * (2.0 * np.pi * sx * sy)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        x_shift = phi_final - mu_x
        y_shift = psi_final - mu_y
        x_rot = cos_t * x_shift + sin_t * y_shift
        y_rot = -sin_t * x_shift + cos_t * y_shift

        denom_x = sx**2 + 2.0 * alpha * t
        denom_y = sy**2 + 2.0 * alpha * t
        exp_x = sx**2 + alpha * t
        exp_y = sy**2 + alpha * t
        prefactor = q_total * (2.0 * alpha * t) / (np.sqrt(denom_x) * np.sqrt(denom_y))
        exponent = -x_rot**2 / (2.0 * exp_x) - y_rot**2 / (2.0 * exp_y)
        j_value = prefactor * np.exp(exponent)

        d_log_j = (
            1.0 / t
            - alpha / denom_x
            - alpha / denom_y
            + alpha * x_rot**2 / (2.0 * exp_x**2)
            + alpha * y_rot**2 / (2.0 * exp_y**2)
        )

        value[mask] = j_value
        derivative[mask] = j_value * d_log_j
        return value, derivative

    def thermal_integrand_and_derivative(self, dtau, params, z0, alpha, kappa, beta,
                                         phi_final=None, psi_final=None):
        """
        Return F = G1 * J and dF/d(t - tau) for the Eq. 22 sensitivity term.
        """
        if phi_final is None:
            phi_final = float(np.asarray(params["mu_x"], dtype=float)[-1])
        if psi_final is None:
            psi_final = float(np.asarray(params["mu_y"], dtype=float)[-1])
        g_value, g_derivative = self._green_g1_and_derivative(dtau, z0, alpha, kappa, beta)
        j_value, j_derivative = self._heat_flux_j_and_derivative(
            dtau, params, alpha, phi_final, psi_final
        )
        integrand = g_value * j_value
        integrand_dtau = g_derivative * j_value + g_value * j_derivative
        return self._apply_thermal_integrand_ceiling(integrand, integrand_dtau)

    def terminal_temperature_discrete(self, s_nodes, v_nodes, params_spline,
                                      z0, alpha, kappa, beta, temp_scale=None):
        """
        Fast node-trapezoid terminal temperature for a fixed spatial velocity profile.
        """
        s_nodes, v_nodes = self._validate_discrete_nodes(s_nodes, v_nodes)
        if s_nodes.size == 1:
            temperature = self.T0
        else:
            t_nodes = self.discrete_time_map(s_nodes, v_nodes)
            params = self._eval_params_on_s(s_nodes, params_spline)
            dtau = t_nodes[-1] - t_nodes
            integrand, _ = self.thermal_integrand_and_derivative(
                dtau,
                params,
                z0=z0,
                alpha=alpha,
                kappa=kappa,
                beta=beta,
                phi_final=float(params["mu_x"][-1]),
                psi_final=float(params["mu_y"][-1]),
            )
            if hasattr(np, "trapezoid"):
                integral = np.trapezoid(integrand, t_nodes)
            else:
                integral = np.trapz(integrand, t_nodes)
            temperature = integral + self.T0
        if temp_scale is None:
            return float(temperature)
        return float((temperature - 25.0) * temp_scale + 25.0)

    def terminal_temperature_gradient_discrete(self, s_nodes, v_nodes, params_spline,
                                               z0, alpha, kappa, beta, temp_scale=None):
        """
        Analytic dT_end/dv_j for the node-trapezoid terminal temperature.
        """
        s_nodes, v_nodes = self._validate_discrete_nodes(s_nodes, v_nodes)
        if s_nodes.size == 1:
            return np.zeros_like(v_nodes, dtype=float)

        t_nodes = self.discrete_time_map(s_nodes, v_nodes)
        time_jacobian = self.discrete_time_jacobian(s_nodes, v_nodes)
        params = self._eval_params_on_s(s_nodes, params_spline)
        dtau = t_nodes[-1] - t_nodes
        integrand, integrand_dtau = self.thermal_integrand_and_derivative(
            dtau,
            params,
            z0=z0,
            alpha=alpha,
            kappa=kappa,
            beta=beta,
            phi_final=float(params["mu_x"][-1]),
            psi_final=float(params["mu_y"][-1]),
        )

        dtau_jacobian = time_jacobian[-1][None, :] - time_jacobian
        integrand_jacobian = integrand_dtau[:, None] * dtau_jacobian
        segment_dt = np.diff(t_nodes)
        segment_dt_jacobian = np.diff(time_jacobian, axis=0)

        gradient = np.zeros_like(v_nodes, dtype=float)
        for idx, dt_segment in enumerate(segment_dt):
            gradient += 0.5 * (
                integrand_jacobian[idx] + integrand_jacobian[idx + 1]
            ) * dt_segment
            gradient += 0.5 * (
                integrand[idx] + integrand[idx + 1]
            ) * segment_dt_jacobian[idx]

        if temp_scale is not None:
            gradient *= temp_scale
        return gradient

    def sensitivity_backtracking_update(self, s_nodes, v_nodes, params_spline,
                                        z0, alpha, kappa, beta,
                                        temperature, temperature_bound,
                                        v_min, v_max, gamma=0.005,
                                        temp_scale=None, exclude_last=True,
                                        candidate_s_range=None,
                                        min_gradient=1e-12,
                                        min_velocity_change=1e-12):
        """
        Update the upstream velocity with the largest useful analytic sensitivity.
        """
        s_nodes, v_nodes = self._validate_discrete_nodes(s_nodes, v_nodes)
        gradient = self.terminal_temperature_gradient_discrete(
            s_nodes,
            v_nodes,
            params_spline,
            z0=z0,
            alpha=alpha,
            kappa=kappa,
            beta=beta,
            temp_scale=temp_scale,
        )
        candidate_count = v_nodes.size - 1 if exclude_last else v_nodes.size
        candidate_count = max(candidate_count, 0)
        updated_v = v_nodes.copy()
        if candidate_count == 0:
            return {
                "success": False,
                "reason": "no upstream velocity is available",
                "updated_v": updated_v,
                "gradient_vector": gradient,
            }

        gap = float(temperature_bound - temperature)
        candidates = np.arange(candidate_count)
        raw_steps = np.zeros(candidate_count, dtype=float)
        valid = np.isfinite(gradient[candidates]) & (np.abs(gradient[candidates]) > min_gradient)
        if candidate_s_range is not None:
            s_lower, s_upper = sorted(map(float, candidate_s_range))
            valid &= (s_nodes[candidates] >= s_lower) & (s_nodes[candidates] <= s_upper)
        raw_steps[valid] = gamma * gap / gradient[candidates][valid]
        proposed = np.clip(v_nodes[candidates] + raw_steps, v_min, v_max)
        valid &= np.abs(proposed - v_nodes[candidates]) > min_velocity_change
        if not np.any(valid):
            return {
                "success": False,
                "reason": "no velocity can be changed by the sensitivity update",
                "updated_v": updated_v,
                "gradient_vector": gradient,
                "gap": gap,
            }

        local_index = int(np.argmax(np.where(valid, np.abs(gradient[candidates]), -np.inf)))
        selected_index = int(candidates[local_index])
        old_v = float(v_nodes[selected_index])
        new_v = float(proposed[local_index])
        updated_v[selected_index] = new_v
        return {
            "success": True,
            "index": selected_index,
            "old_v": old_v,
            "new_v": new_v,
            "step": new_v - old_v,
            "gradient": float(gradient[selected_index]),
            "gradient_vector": gradient,
            "updated_v": updated_v,
            "temperature": float(temperature),
            "temperature_bound": float(temperature_bound),
            "gap": gap,
        }
    
    def get_s_t(self,
                s_grid,
                v_s_grid,
                dt=None,
                ds_max=None,
                vel_grad_tol=0.15,
                v_min=1e-10,
                max_steps=200000,
                enforce_monotonic=True,
                dt_cap=True):
        """
        Robust / higher‑accuracy construction of (t, s(t)) from node velocities v(s).

        Integrates ds/dt = v(s) by stepping in s (space) with adaptive ds so that
        velocity variation across each interval stays below vel_grad_tol (relative).
        Uses trapezoidal rule for dt increment (second order in ds):

            dt_i ≈ ∫_{s_i}^{s_{i+1}} ds / v(s)  ≈ 0.5*Δs*(1/v_i + 1/v_{i+1})

        Args:
            s_grid        : 1D array of s nodes (monotonic, final target = s_grid[-1])
            v_s_grid      : velocities defined at s_grid
            dt            : optional initial time step hint (unused for stepping,
                            only used if ds_max not supplied)
            ds_max        : cap for spatial increment; default = min node spacing
            vel_grad_tol  : relative change threshold for |v(s+Δs)-v(s)| / max(v,1e-12)
            v_min         : floor velocity to avoid divide-by-zero
            max_steps     : safety iteration cap
            enforce_monotonic : clamp any negative/zero velocities to v_min

        Returns:
            t_arr (np.ndarray), s_arr (np.ndarray)
        """
        s_end = float(s_grid[-1])
        # Precompute min spacing
        if len(s_grid) > 1:
            base_ds = np.min(np.diff(s_grid))
        else:
            base_ds = s_end if s_end > 0 else 1e-3
        if ds_max is None:
            ds_max = base_ds
        # Fallback avg velocity for dt hint
        if dt is None:
            v_mean = np.clip(np.mean(v_s_grid), v_min, None)
            dt_hint = ds_max / v_mean
        else:
            dt_hint = dt

        t_list = [0.0]
        s_list = [float(self.S_INIT)]

        # Fast linear interpolation helpers
        s_nodes = np.asarray(s_grid)
        v_nodes = np.asarray(v_s_grid)

        def v_of_s(s):
            if s <= s_nodes[0]:
                v = v_nodes[0]
            elif s >= s_nodes[-1]:
                v = v_nodes[-1]
            else:
                v = np.interp(s, s_nodes, v_nodes)
            if enforce_monotonic and v <= v_min:
                return v_min
            return v

        steps = 0
        while s_list[-1] < s_end and steps < max_steps:
            s_curr = s_list[-1]
            v_curr = v_of_s(s_curr)

            # Start with proposed ds
            ds_try = min(ds_max, s_end - s_curr)
            # Refine ds until velocity variation acceptable
            while True:
                s_next = s_curr + ds_try
                v_next = v_of_s(s_next)
                rel_change = abs(v_next - v_curr) / max(v_curr, v_min)
                if rel_change > vel_grad_tol and ds_try > 1e-12:
                    ds_try *= 0.5
                    continue
                break

            # Trapezoidal dt (bounded)
            inv_v_curr = 1.0 / max(v_curr, v_min)
            inv_v_next = 1.0 / max(v_next, v_min)
            dt_inc = 0.5 * ds_try * (inv_v_curr + inv_v_next)

            if dt_inc > dt and dt_cap:
                scale = dt / dt_inc
                ds_try *= max(scale,1e-6)
                v_next = v_of_s(s_curr + ds_try)
                inv_v_next = 1.0 / max(v_next, v_min)
                dt_inc = 0.5 * ds_try * (inv_v_curr + inv_v_next)

            # Optional guard against pathological tiny dt (fallback to hint)
            if dt_inc < 1e-12:
                dt_inc = dt_hint

            t_list.append(t_list[-1] + dt_inc)
            s_list.append(s_curr + ds_try)
            steps += 1

        if steps >= max_steps and s_list[-1] < s_end:
            # Force final point if loop exited early
            s_list.append(s_end)
            t_list.append(t_list[-1] + (s_end - s_list[-2]) / v_of_s(s_list[-2]))

        return np.array(t_list), np.array(s_list)

    def get_temp(self, t_grid, s_t_grid, params_spline,
                 z0, alpha, kappa, beta,
                 use_sqrt_gauss=False, sqrt_gauss_nodes=2):
        """Evaluate contact temperature using the selected quadrature rule."""
        if use_sqrt_gauss:
            return self.forward_temperature_sqrt_gauss(
                s_t_grid,
                params_spline["mu_x"], params_spline["mu_y"],
                params_spline["Q"], params_spline["sigma_x"],
                params_spline["sigma_y"], params_spline["theta"],
                tgrid=t_grid,
                z0=z0, alpha=alpha, kappa=kappa, beta=beta,
                sqrt_gauss_nodes=sqrt_gauss_nodes,
                params_spline=params_spline
            ) + self.T0
        return self.forward_temperature_trapz(
            s_t_grid,
            params_spline["mu_x"], params_spline["mu_y"],
            params_spline["Q"], params_spline["sigma_x"],
            params_spline["sigma_y"], params_spline["theta"],
            tgrid=t_grid,
            z0=z0, alpha=alpha, kappa=kappa, beta=beta,
            params_spline=params_spline,
        ) + self.T0

    # ============================================================
    # Local (sequential) optimizer:
    #   At each step k -> choose v_s_{k+1} in [v_min, v_max]
    #   to minimize (T_{k+1} - T_ref_target[k+1])^2
    # ============================================================
    def temp_next_v(
            self, s_grid, v_s_grid,
            next_v,
            params_spline,
            z0, alpha, kappa, beta,
            ds=1e-3, dt=0.05,
            use_sqrt_gauss=False, sqrt_gauss_nodes=2,
            temp_scale=None):
        """
        Pass use_sqrt_gauss=True for fixed-node sqrt-substitution quadrature;
        otherwise the vectorized trapezoidal rule is used.
        Pass temp_scale to scale the simulated temperature about 25 C.
        """
        if len(s_grid) == 0:
            s_grid = np.array([self.S_INIT + ds/100])
            v_s_grid = np.array([next_v])
        else:
            s_grid = np.append(s_grid, s_grid[-1] + ds)
            v_s_grid = np.append(v_s_grid, next_v)

        t_grid, s_t_grid = self.get_s_t(s_grid, v_s_grid, dt=dt)
        # TODO: add t_grid, s_t_grid from previous segments
        sim_f = self.get_temp(t_grid, s_t_grid, params_spline,
                              z0, alpha, kappa, beta,
                              use_sqrt_gauss=use_sqrt_gauss,
                              sqrt_gauss_nodes=sqrt_gauss_nodes)
        ratio = temp_scale if temp_scale is not None else 1.0
        scaled_sim = (sim_f - 25.0) * ratio + 25.0
        return scaled_sim, t_grid[-1], s_grid, v_s_grid

    def get_v_bounds(self, s_list, v_list, v_min, v_max, mu_x, mu_y,
                     p_a_max, p_v_max, S_FINAL, ds=1e-6, p_v_min=0.0,
                     use_acceleration_bounds=True, s_step=None):
        """
        Compute dynamic bounds for next velocity based on acceleration limits.

        Args:
            s_list : list of s positions so far
            v_list : list of velocities so far
            v_min  : minimum allowed velocity
            v_max  : maximum allowed velocity
            mu_x   : spline function for x(s)
            mu_y   : spline function for y(s)
            p_a_max: maximum allowed acceleration (m/s^2)
            p_v_min: absolute minimum velocity (m/s)
            p_v_max: absolute maximum velocity (m/s)
            S_FINAL: final target s position

        Returns:
            (v_min_next, v_max_next) : tuple of new velocity bounds for next step
        """

        s_last = s_list[-1] if len(s_list) else self.S_INIT
        v_last = v_list[-1] if len(v_list) else None

        # Compute spatial derivatives
        delta_s = ds
        next_step = delta_s if s_step is None else float(s_step)
        s_eval = s_last + next_step if len(s_list) else self.S_INIT + next_step / 100.0
        mu_x_last = mu_x(s_eval)
        mu_y_last = mu_y(s_eval)
        mu_x_curr = mu_x(s_eval + delta_s)
        mu_y_curr = mu_y(s_eval + delta_s)

        dx_ds = (mu_x_curr - mu_x_last) / delta_s
        dy_ds = (mu_y_curr - mu_y_last) / delta_s

        v_min_feedrate, v_max_feedrate = v_min, v_max

        v_min_next = v_min_feedrate
        v_max_next = v_max_feedrate

        if not use_acceleration_bounds or v_last is None:
            if v_max_next < v_min_next:
                print(f"Warning: Infeasible velocity bounds at s={s_last:.4f}.")
            return v_min_next, v_max_next

        d2x_ds2 = (mu_x(s_eval + 2*delta_s) - 2*mu_x_curr + mu_x_last) / (delta_s**2)
        d2y_ds2 = (mu_y(s_eval + 2*delta_s) - 2*mu_y_curr + mu_y_last) / (delta_s**2)
        
        A = np.sqrt(max(0, (p_a_max+dx_ds*v_last**2/2/delta_s)/(d2x_ds2 + dx_ds/2/delta_s)))
        B = np.sqrt(max(0, (-p_a_max+dx_ds*v_last**2/2/delta_s)/(d2x_ds2 + dx_ds/2/delta_s)))
        C = np.sqrt(max(0, (p_a_max+dy_ds*v_last**2/2/delta_s)/(d2y_ds2 + dy_ds/2/delta_s)))
        D = np.sqrt(max(0, (-p_a_max+dy_ds*v_last**2/2/delta_s)/(d2y_ds2 + dy_ds/2/delta_s)))

        v_min_next = max(v_min_feedrate, min(A, B), min(C, D))
        
        v_max_next = min(v_max_feedrate, max(A, B), max(C, D))
        
        if v_max_next < v_min_next:
            print(f"Warning: Infeasible velocity bounds at s={s_last:.4f}, v_last={v_last:.4f}.")

        return v_min_next, v_max_next

    def _replay_velocity_sequence(self, velocity_sequence, params_spline,
                                  z0, alpha, kappa, beta, ds, dt,
                                  use_sqrt_gauss=False, sqrt_gauss_nodes=2,
                                  temp_scale=None):
        s_list = []
        v_list = []
        temp_list = []
        t_list = []
        for velocity in velocity_sequence:
            temp, t_value, s_list, v_list = self.temp_next_v(
                s_list,
                v_list,
                float(velocity),
                params_spline,
                z0,
                alpha,
                kappa,
                beta,
                ds=ds,
                dt=dt,
                use_sqrt_gauss=use_sqrt_gauss,
                sqrt_gauss_nodes=sqrt_gauss_nodes,
                temp_scale=temp_scale,
            )
            temp_list.append(temp)
            t_list.append(t_value)
        return temp_list, t_list, s_list, v_list

    def optimize_velocity_profile_sequential(self, n_variables, params_spline,
                                             z0, alpha, kappa, beta,
                                             ds, dt, v_range, T_ref,
                                             p_a_max, p_v_max, S_FINAL,
                                             p_a_max_backtracking=None,
                                             lam_smooth=0.0, default_v=None,
                                             temp_scale=None,
                                             optimize_s_range=None,
                                             enable_backtracking=True,
                                             temperature_margin=150.0,
                                             temperature_lower=None,
                                             temperature_upper=None,
                                             sensitivity_gamma=0.005,
                                             solver_xatol=1e-3,
                                             solver_maxiter=50,
                                             max_backtracking_steps=500,
                                             segment_timeout_s=None,
                                             p_v_min=0.0,
                                             use_acceleration_bounds=True,
                                             use_sqrt_gauss=False,
                                             sqrt_gauss_nodes=2,
                                             strict_feasibility_failure=False,
                                             show_progress=True):
        """
        Sequential velocity planner with optional analytic-sensitivity backtracking.
        """
        n_variables = int(n_variables)
        if n_variables <= 0:
            raise ValueError("n_variables must be positive")
        v_lower, v_upper = map(float, v_range)
        if p_a_max_backtracking is None:
            p_a_max_backtracking = 10.0 * float(p_a_max)
        p_a_max_backtracking = float(p_a_max_backtracking)
        use_sqrt_gauss = bool(use_sqrt_gauss)
        sqrt_gauss_nodes = int(sqrt_gauss_nodes)
        if sqrt_gauss_nodes <= 0:
            raise ValueError("sqrt_gauss_nodes must be positive")
        thermal_integration = "sqrt_gauss" if use_sqrt_gauss else "trapezoid"
        strict_feasibility_failure = bool(strict_feasibility_failure)
        if default_v is None:
            default_v = 0.5 * (v_lower + v_upper)
        default_v = float(np.clip(default_v, v_lower, v_upper))
        if optimize_s_range is not None:
            optimize_s_range = tuple(sorted(map(float, optimize_s_range)))
        mu_x = params_spline["mu_x"]
        mu_y = params_spline["mu_y"]
        lower_temperature = None if temperature_lower is None else float(temperature_lower)
        upper_temperature = (
            float(T_ref + temperature_margin)
            if temperature_upper is None
            else float(temperature_upper)
        )

        metrics = {
            "optimization_variables": n_variables,
            "solver": "scipy.optimize.minimize_scalar(method='bounded')",
            "convergence_criteria": {
                "solver_xatol": solver_xatol,
                "solver_maxiter": solver_maxiter,
                "temperature_target": float(T_ref),
                "temperature_lower_bound": lower_temperature,
                "temperature_upper_bound": upper_temperature,
                "backtracking_enabled": bool(enable_backtracking),
                "backtracking_gain": sensitivity_gamma,
                "max_backtracking_steps": max_backtracking_steps,
                "physical_feedrate_lower_bound_m_s": float(p_v_min),
                "physical_feedrate_upper_bound_m_s": float(p_v_max),
                "acceleration_bounds_enabled": bool(use_acceleration_bounds),
                "sqrt_gauss_quadrature": use_sqrt_gauss,
                "sqrt_gauss_nodes": sqrt_gauss_nodes if use_sqrt_gauss else None,
                "thermal_integration": thermal_integration,
                "strict_feasibility_failure": strict_feasibility_failure,
                "thermal_integrand_ceiling": self.thermal_integrand_ceiling,
                "optimize_s_range": (
                    None if optimize_s_range is None else [float(optimize_s_range[0]), float(optimize_s_range[1])]
                ),
            },
            "accepted_steps": 0,
            "scalar_solves": 0,
            "solver_iterations": 0,
            "function_evaluations": 0,
            "backtracking_steps": 0,
            "successful_backtracking_steps": 0,
            "failed_backtracking_steps": 0,
            "temperature_bound_events": 0,
            "temperature_upper_bound_events": 0,
            "temperature_lower_bound_events": 0,
            "temperature_violations_without_backtracking": 0,
            "temperature_upper_violations_without_backtracking": 0,
            "temperature_lower_violations_without_backtracking": 0,
            "infeasible_velocity_bounds": 0,
            "infeasible_velocity_bounds_nonfinite": 0,
            "infeasible_velocity_bounds_min_above_global_upper": 0,
            "infeasible_velocity_bounds_max_below_global_lower": 0,
            "infeasible_velocity_bounds_inverted_internal": 0,
            "infeasible_velocity_bound_examples": [],
            "timed_out": False,
            "optimization_success": True,
            "strict_feasibility_failure": False,
            "failure_reason": None,
            "failure_step": None,
            "failure_kind": None,
            "failure_probe_velocity": None,
            "failure_temperature": None,
            "failure_bound": None,
            "failure_raw_v_min": None,
            "failure_raw_v_max": None,
        }

        temp_list = []
        t_list = []
        s_list = []
        v_list = []
        v_min_list = []
        v_max_list = []
        t_min_list = []
        t_max_list = []
        start_time = time.perf_counter()
        pbar = None
        if show_progress:
            pbar = tqdm(total=n_variables, desc="Sequential planner",
                        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

        def progress_to(value):
            if pbar is None:
                return
            pbar.n = int(value)
            pbar.refresh()

        def get_smoothness_penalty(velocity):
            previous = v_list[-1] if len(v_list) else v_lower
            return lam_smooth * (velocity - previous) ** 2

        def temperature_for_candidate(velocity):
            return self.temp_next_v(
                s_list,
                v_list,
                float(velocity),
                params_spline,
                z0,
                alpha,
                kappa,
                beta,
                ds=ds,
                dt=dt,
                use_sqrt_gauss=use_sqrt_gauss,
                sqrt_gauss_nodes=sqrt_gauss_nodes,
                temp_scale=temp_scale,
            )[0]

        def next_s_candidate():
            if len(s_list) == 0:
                return float(self.S_INIT + ds / 100.0)
            return float(s_list[-1] + ds)

        def optimization_is_active(s_value):
            if optimize_s_range is None:
                return True
            return optimize_s_range[0] <= s_value <= optimize_s_range[1]

        def mark_strict_failure(reason, bound_violation):
            metrics["optimization_success"] = False
            metrics["strict_feasibility_failure"] = True
            metrics["failure_reason"] = reason
            metrics["failure_step"] = int(k)
            metrics["failure_kind"] = bound_violation["kind"]
            metrics["failure_probe_velocity"] = float(bound_violation["probe_velocity"])
            metrics["failure_temperature"] = float(bound_violation["probe_temperature"])
            metrics["failure_bound"] = float(bound_violation["temperature_bound"])

        def mark_velocity_bound_failure(case, raw_v_min, raw_v_max):
            metrics["optimization_success"] = False
            metrics["strict_feasibility_failure"] = True
            metrics["failure_reason"] = "infeasible_velocity_bounds"
            metrics["failure_step"] = int(k)
            metrics["failure_kind"] = case
            metrics["failure_raw_v_min"] = raw_v_min
            metrics["failure_raw_v_max"] = raw_v_max

        expanded_accel_until = None
        k = 0
        while k < n_variables:
            if segment_timeout_s is not None and time.perf_counter() - start_time > segment_timeout_s:
                metrics["timed_out"] = True
                metrics["optimization_success"] = False
                metrics["failure_reason"] = "segment_timeout"
                metrics["failure_step"] = int(k)
                break

            s_candidate = next_s_candidate()
            optimization_active = optimization_is_active(s_candidate)
            if not optimization_active:
                active_p_a_max = p_a_max_backtracking
            elif expanded_accel_until is not None and k <= expanded_accel_until:
                active_p_a_max = 10.0 * p_a_max
            else:
                active_p_a_max = p_a_max
                if expanded_accel_until is not None:
                    expanded_accel_until = None
            v_min, v_max = self.get_v_bounds(
                s_list,
                v_list,
                v_lower,
                v_upper,
                mu_x,
                mu_y,
                active_p_a_max,
                p_v_max,
                S_FINAL,
                ds=ds / 1000.0,
                p_v_min=p_v_min,
                use_acceleration_bounds=use_acceleration_bounds,
                s_step=ds,
            )
            if (not np.isfinite(v_min)) or (not np.isfinite(v_max)) or v_max < v_min:
                metrics["infeasible_velocity_bounds"] += 1
                raw_v_min = float(v_min) if np.isfinite(v_min) else None
                raw_v_max = float(v_max) if np.isfinite(v_max) else None
                if np.isfinite(v_min) and np.isfinite(v_max):
                    if v_min > v_upper:
                        repair_case = "min_above_global_upper"
                        metrics["infeasible_velocity_bounds_min_above_global_upper"] += 1
                        v_min, v_max = v_upper, v_upper
                    elif v_max < v_lower:
                        repair_case = "max_below_global_lower"
                        metrics["infeasible_velocity_bounds_max_below_global_lower"] += 1
                        v_min, v_max = v_lower, v_lower
                    else:
                        repair_case = "inverted_internal"
                        metrics["infeasible_velocity_bounds_inverted_internal"] += 1
                        repaired = float(np.clip(0.5 * (v_min + v_max), v_lower, v_upper))
                        v_min, v_max = repaired, repaired
                else:
                    repair_case = "nonfinite"
                    metrics["infeasible_velocity_bounds_nonfinite"] += 1
                    v_min, v_max = v_lower, v_upper
                if len(metrics["infeasible_velocity_bound_examples"]) < 20:
                    metrics["infeasible_velocity_bound_examples"].append(
                        {
                            "index": int(k),
                            "case": repair_case,
                            "raw_v_min": raw_v_min,
                            "raw_v_max": raw_v_max,
                            "repaired_v_min": float(v_min),
                            "repaired_v_max": float(v_max),
                        }
                    )
                if strict_feasibility_failure:
                    mark_velocity_bound_failure(repair_case, raw_v_min, raw_v_max)
                    break
            v_min = float(np.clip(v_min, v_lower, v_upper))
            v_max = float(np.clip(v_max, v_min, v_upper))
            bounds_collapsed = abs(v_max - v_min) <= 1e-12

            t_min = temperature_for_candidate(v_max)
            t_max = temperature_for_candidate(v_min)

            bound_violation = None
            if optimization_active and t_min > upper_temperature:
                bound_violation = {
                    "kind": "upper",
                    "probe_velocity": v_max,
                    "probe_temperature": t_min,
                    "temperature_bound": upper_temperature,
                }
            elif optimization_active and lower_temperature is not None and t_max < lower_temperature:
                bound_violation = {
                    "kind": "lower",
                    "probe_velocity": v_min,
                    "probe_temperature": t_max,
                    "temperature_bound": lower_temperature,
                }

            if bound_violation is not None:
                metrics["temperature_bound_events"] += 1
                if bound_violation["kind"] == "upper":
                    metrics["temperature_upper_bound_events"] += 1
                else:
                    metrics["temperature_lower_bound_events"] += 1
                if enable_backtracking:
                    if len(v_list) > 0 and metrics["backtracking_steps"] < max_backtracking_steps:
                        _, _, s_probe, v_probe = self.temp_next_v(
                            s_list,
                            v_list,
                            bound_violation["probe_velocity"],
                            params_spline,
                            z0,
                            alpha,
                            kappa,
                            beta,
                            ds=ds,
                            dt=dt,
                            use_sqrt_gauss=use_sqrt_gauss,
                            sqrt_gauss_nodes=sqrt_gauss_nodes,
                            temp_scale=temp_scale,
                        )
                        update = self.sensitivity_backtracking_update(
                            s_probe,
                            v_probe,
                            params_spline,
                            z0=z0,
                            alpha=alpha,
                            kappa=kappa,
                            beta=beta,
                            temperature=bound_violation["probe_temperature"],
                            temperature_bound=bound_violation["temperature_bound"],
                            v_min=v_lower,
                            v_max=v_upper,
                            gamma=sensitivity_gamma,
                            temp_scale=temp_scale,
                            exclude_last=True,
                            candidate_s_range=optimize_s_range,
                        )
                        metrics["backtracking_steps"] += 1
                        if update.get("success", False):
                            expanded_accel_until = int(k)
                            selected = int(update["index"])
                            retained_v = update["updated_v"][:selected + 1]
                            temp_list, t_list, s_list, v_list = self._replay_velocity_sequence(
                                retained_v,
                                params_spline,
                                z0,
                                alpha,
                                kappa,
                                beta,
                                ds,
                                dt,
                                use_sqrt_gauss=use_sqrt_gauss,
                                sqrt_gauss_nodes=sqrt_gauss_nodes,
                                temp_scale=temp_scale,
                            )
                            v_min_list = v_min_list[:len(v_list)]
                            v_max_list = v_max_list[:len(v_list)]
                            t_min_list = t_min_list[:len(v_list)]
                            t_max_list = t_max_list[:len(v_list)]
                            k = len(v_list)
                            metrics["successful_backtracking_steps"] += 1
                            progress_to(k)
                            continue
                        metrics["failed_backtracking_steps"] += 1
                        if strict_feasibility_failure:
                            mark_strict_failure("backtracking_update_failed", bound_violation)
                            break
                    elif strict_feasibility_failure:
                        reason = (
                            "max_backtracking_steps_reached"
                            if metrics["backtracking_steps"] >= max_backtracking_steps
                            else "no_upstream_velocity_for_backtracking"
                        )
                        mark_strict_failure(reason, bound_violation)
                        break
                else:
                    metrics["temperature_violations_without_backtracking"] += 1
                    if bound_violation["kind"] == "upper":
                        metrics["temperature_upper_violations_without_backtracking"] += 1
                    else:
                        metrics["temperature_lower_violations_without_backtracking"] += 1
                    if strict_feasibility_failure:
                        mark_strict_failure("temperature_bound_violation_without_backtracking", bound_violation)
                        break

            if len(s_list) == 0:
                next_v = float(np.clip(default_v, v_min, v_max))
            elif bounds_collapsed:
                next_v = v_min
            else:
                def objective(velocity):
                    temperature = temperature_for_candidate(velocity)
                    return (temperature - T_ref) ** 2 + get_smoothness_penalty(velocity)

                res = minimize_scalar(
                    objective,
                    bounds=(v_min, v_max),
                    method='bounded',
                    options={'xatol': solver_xatol, 'maxiter': solver_maxiter},
                )
                next_v = float(res.x)
                metrics["scalar_solves"] += 1
                metrics["solver_iterations"] += int(getattr(res, "nit", 0) or 0)
                metrics["function_evaluations"] += int(getattr(res, "nfev", 0) or 0)

            v_min_list.append(v_min)
            v_max_list.append(v_max)
            t_min_list.append(t_min)
            t_max_list.append(t_max)

            temp, t_value, s_list, v_list = self.temp_next_v(
                s_list,
                v_list,
                next_v,
                params_spline,
                z0,
                alpha,
                kappa,
                beta,
                ds=ds,
                dt=dt,
                use_sqrt_gauss=use_sqrt_gauss,
                sqrt_gauss_nodes=sqrt_gauss_nodes,
                temp_scale=temp_scale,
            )
            temp_list.append(temp)
            t_list.append(t_value)
            k = len(v_list)
            progress_to(k)

        if pbar is not None:
            pbar.close()

        metrics["accepted_steps"] = len(v_list)
        metrics["total_computation_time_s"] = time.perf_counter() - start_time
        temp_arr = np.asarray(temp_list, dtype=float)
        t_arr = np.asarray(t_list, dtype=float)
        s_arr = np.asarray(s_list, dtype=float)
        v_arr = np.asarray(v_list, dtype=float)
        if temp_arr.size:
            metrics["max_temperature"] = float(np.max(temp_arr))
            metrics["final_temperature"] = float(temp_arr[-1])
        else:
            metrics["max_temperature"] = None
            metrics["final_temperature"] = None
        if v_arr.size:
            metrics["min_velocity"] = float(np.min(v_arr))
            metrics["max_velocity"] = float(np.max(v_arr))
            metrics["mean_velocity"] = float(np.mean(v_arr))
        else:
            metrics["min_velocity"] = None
            metrics["max_velocity"] = None
            metrics["mean_velocity"] = None

        return {
            "temp": temp_arr,
            "time": t_arr,
            "s": s_arr,
            "v": v_arr,
            "v_min": np.asarray(v_min_list[:len(v_arr)], dtype=float),
            "v_max": np.asarray(v_max_list[:len(v_arr)], dtype=float),
            "t_min": np.asarray(t_min_list[:len(v_arr)], dtype=float),
            "t_max": np.asarray(t_max_list[:len(v_arr)], dtype=float),
            "metrics": metrics,
        }
