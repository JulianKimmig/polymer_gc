import secanalysis.sec_formats
import secanalysis.sec_formats.base
import pandas as pd
from typing import Optional, Tuple
from scipy.ndimage import gaussian_filter1d
import numpy as np
from secanalysis.sec_formats.base import SECDataBase


class SimSEC(SECDataBase):
    DEFAULT_CALIBRATION_PARAMS: Tuple[float, float] = (-0.45, 10.5)

    def __init__(
        self,
        raw_data: pd.DataFrame,
        *,
        calibration_params: Tuple[float, float] | None = None,
    ) -> None:
        super().__init__(raw_data)

        self.calibration_params = calibration_params
        a, b = calibration_params or self.DEFAULT_CALIBRATION_PARAMS
        # The calibration function must vectorise over *v* for bulk operations
        self.set_calibration_function(lambda v, _a=a, _b=b: _a * v + _b, (a, b))

    def from_string(self, string):
        raise NotImplementedError("SECDataBase.from_string() not implemented")

    @classmethod
    def from_mn_mw(
        cls,
        Mn: float,
        Mw: float,
        *,
        n_molecules: int = 1_000_000,
        n_points: int = 5_000,
        n_points_detect: Optional[int] = None,
        volume_range: Tuple[float, float] = (0.0, 30.0),
        calibration_params: Tuple[float, float] | None = None,
        noise_level: float = 0.0,
        baseline_level: float = 0.0,
        smooth_sigma: float | None = 3.0,
        asymmetry: float = 0.0,
        max_iter: int = 10,
        tol: float = 0.005,
        random_state: Optional[int | np.random.Generator] = None,
    ) -> "SimSEC":
        """Build a skewed SEC trace whose calculated Mn/Mw match the targets."""
        if Mw < Mn:
            raise ValueError("Mw must exceed Mn.")

        a, b = calibration_params or cls.DEFAULT_CALIBRATION_PARAMS
        if np.isclose(a, 0.0):
            raise ValueError("Calibration slope 'a' cannot be zero.")

        rng_master = (
            np.random.default_rng(random_state)
            if not isinstance(random_state, np.random.Generator)
            else random_state
        )

        if n_points_detect is None:
            n_points_detect = int(n_points / 10)

        # --------------------------------------------------------------
        # Helper: simulate moments with given μ, σ on a *small* sample
        # --------------------------------------------------------------
        def _simulate_moments(mu: float, sigma: float) -> Tuple[float, float]:
            masses_full = rng_master.lognormal(mu, sigma, int(n_molecules / 10))
            vols_full = (np.log10(masses_full) - b) / a
            if asymmetry != 0.0:
                span = volume_range[1] - volume_range[0]
                lag_scale = 0.01 * span
                vols_full += np.sign(asymmetry) * rng_master.exponential(
                    abs(asymmetry) * lag_scale, size=vols_full.size
                )

            keep = (vols_full >= volume_range[0]) & (vols_full <= volume_range[1])
            vols_full, masses_full = vols_full[keep], masses_full[keep]
            hist, edges = np.histogram(
                vols_full,
                bins=n_points_detect,
                range=volume_range,
                weights=masses_full,
            )
            vol_axis = 0.5 * (edges[:-1] + edges[1:])
            signal = hist.astype(float)

            if smooth_sigma and smooth_sigma > 0:
                signal = gaussian_filter1d(signal, float(smooth_sigma))

            if noise_level > 0.0:
                signal += rng_master.normal(
                    0.0, noise_level * signal.max(), signal.shape
                )
            if baseline_level:
                signal += baseline_level

            raw = pd.DataFrame({cls.DEFAULT_VOLUME_COLUMN: vol_axis, "signal": signal})
            ins = cls(raw_data=raw, calibration_params=calibration_params)
            ins.autodetect_signal_boarders()
            for i, (b1, b2, p) in enumerate(ins.signal_boarders):
                m1, m2 = ins.mass_range[[b2, b1]]
                p = ins.calc_mass_params(m1, m2)
                mn, mw = p["Mn"], p["Mw"]
                return mn, mw

            return (
                1e-32,
                1e-32,
            )  # dont return zero as this will later cause a division by zero

        # --------------------------------------------------------------
        # Newton‑style parameter refinement (μ shifts Mn, σ tunes Mw/Mn)
        # --------------------------------------------------------------
        sigma2 = np.log(Mw / Mn)
        sigma = np.sqrt(sigma2)
        mu = np.log(Mn) - sigma2 / 2

        for _ in range(max_iter):
            Mn_est, Mw_est = _simulate_moments(mu, sigma)
            err_Mn = Mn_est / Mn - 1
            err_Mw = Mw_est / Mw - 1
            if max(abs(err_Mn), abs(err_Mw)) < tol:
                break
            # μ update (scales Mn & Mw identically)
            mu -= np.log1p(err_Mn)
            # σ update (targets Mw/Mn ratio)
            ratio_tgt = Mw / Mn
            ratio_est = Mw_est / Mn_est
            sigma *= np.sqrt(ratio_tgt / ratio_est)
        else:
            raise ValueError(
                f"Failed to converge after {max_iter} iterations. "
                f"Last error: Mn = {err_Mn}, Mw = {err_Mw}"
            )

        # --------------------------------------------------------------
        # Final full‑size simulation with converged parameters
        # --------------------------------------------------------------
        masses_full = rng_master.lognormal(mu, sigma, n_molecules)
        vols_full = (np.log10(masses_full) - b) / a
        if asymmetry != 0.0:
            span = volume_range[1] - volume_range[0]
            lag_scale = 0.01 * span
            vols_full += np.sign(asymmetry) * rng_master.exponential(
                abs(asymmetry) * lag_scale, size=vols_full.size
            )

        keep = (vols_full >= volume_range[0]) & (vols_full <= volume_range[1])
        vols_full, masses_full = vols_full[keep], masses_full[keep]
        hist, edges = np.histogram(
            vols_full, bins=n_points, range=volume_range, weights=masses_full
        )
        vol_axis = 0.5 * (edges[:-1] + edges[1:])
        signal = hist.astype(float)

        if smooth_sigma and smooth_sigma > 0:
            signal = gaussian_filter1d(signal, float(smooth_sigma))

        if noise_level > 0.0:
            signal += rng_master.normal(0.0, noise_level * signal.max(), signal.shape)
        if baseline_level:
            signal += baseline_level

        raw = pd.DataFrame({cls.DEFAULT_VOLUME_COLUMN: vol_axis, "signal": signal})
        return cls(raw_data=raw, calibration_params=calibration_params)
