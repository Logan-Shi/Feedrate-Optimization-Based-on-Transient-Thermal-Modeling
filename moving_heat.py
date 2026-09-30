"""Transient temperature reconstruction from fitted Gaussian heat flux."""

import numpy as np
from scipy.special import erfcx

from heat_flux import GaussianHeatFlux


class MovingHeatSource:
    """Evaluate transient temperature rise for a moving heat source."""

    def __init__(self, rho, c, kappa, beta, ratio_sigma=1.0, ratio_q=0.56):
        self.rho = rho
        self.c = c
        self.kappa = kappa
        self.beta = beta
        self.alpha = kappa / (rho * c)
        self.ratio_sigma = ratio_sigma
        self.ratio_q = ratio_q
        self.dt = 0.05
        self.pod = None

    def load_gaussian(self, filename):
        """Load a fitted Gaussian heat-flux model from an ``.npz`` file."""
        self.pod = GaussianHeatFlux.load(filename)

    def get_temp_gaussian(self, state, start_t=0.0):
        """Return temperature rise at ``(x, y, z, t)`` in metres and seconds."""
        if self.pod is None:
            raise ValueError("Load a fitted Gaussian heat-flux model first")
        x, y, z, t = state
        if t <= start_t:
            return 0.0

        n_steps = max(200, int((t - start_t) / self.dt))
        tau = np.linspace(start_t, t, n_steps)[:-1]
        params = self.pod.get_params_list(tau)
        amplitude, x0, y0, sigma_x, sigma_y, theta = params[:, :6].T

        diffusion = np.sqrt(self.alpha * (t - tau))
        beta_diffusion = self.beta / self.kappa * diffusion
        sigma_x *= self.ratio_sigma
        sigma_y *= self.ratio_sigma
        power = amplitude * 1e6 * (2 * np.pi * sigma_x * sigma_y) * self.ratio_q
        valid = power >= 1e-6
        if not np.any(valid):
            return 0.0

        denominator = np.sqrt(sigma_x**2 + 2 * diffusion**2) * np.sqrt(
            sigma_y**2 + 2 * diffusion**2
        )
        cosine, sine = np.cos(theta), np.sin(theta)
        x_shift, y_shift = x - x0, y - y0
        x_rotated = cosine * x_shift + sine * y_shift
        y_rotated = -sine * x_shift + cosine * y_shift
        source = (
            self.alpha / (2 * self.kappa) * power
            * np.exp(
                -x_rotated**2 / (2 * sigma_x**2 + 4 * diffusion**2)
                -y_rotated**2 / (2 * sigma_y**2 + 4 * diffusion**2)
            )
        )
        previous = np.exp(-z**2 / (4 * diffusion**2)) / (
            diffusion * np.pi**1.5
        )
        if z < 1e-6:
            boundary = self.beta / (self.kappa * np.pi) * erfcx(beta_diffusion)
        else:
            boundary_arg = beta_diffusion + z / (2 * diffusion)
            boundary = (
                self.beta / (self.kappa * np.pi)
                * np.exp(beta_diffusion**2 + self.beta * z / self.kappa)
                * erfcx(boundary_arg)
            )

        rate = (previous - boundary) * source / denominator
        rate[~valid] = 0.0
        return float(np.trapz(rate, tau))
