"""
Synthetic historical-data generator for the contextual 2SSP food-truck model.

Generates one row per (area, day, food_type) with columns
    area_index | day | weekday | weather | food_type | demand

True data-generating process
----------------------------
Per-area attribute (fixed across days, not used as an OLS feature):
    pop_size_i    in {1, 2, 3}     1=small, 2=medium, 3=large

Per-day covariates (shared across areas on the same day):
    weekday_day   in {0, 1}        1=weekday, 0=weekend
    weather_day   in {0, 1}        1=sunny,   0=rain

Demand model for area i, food type t, day:
    nominal_i                   = pop_size_i * base_per_capita
    alpha_i                     = a * pop_size_i      (weather coefficient)
    beta_i                      = b * pop_size_i      (weekday coefficient)
    d_{i,t}(day)                = nominal_i
                                + alpha_i * weather_day
                                + beta_i  * weekday_day
                                + eps_{i,t},   eps_{i,t} ~ N(0, sigma^2)
    truncated at 0 from below.

The coefficients (a, b) are problem-setting parameters that are held fixed
across experiments; only eps varies when re-sampling with a different seed.
OLS per (area, food_type) should recover the intercept nominal_i and the
shared slopes alpha_i, beta_i up to noise.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class DGPTruth:
    """Ground-truth parameters of the DGP, for sanity checks and oracle runs."""
    pop_size: np.ndarray         # shape (I,)
    nominal: np.ndarray          # shape (I,)
    alpha: np.ndarray            # shape (I,), weather coefficients (shared across food types)
    beta:  np.ndarray            # shape (I,), weekday coefficients (shared across food types)
    a: float                     # scalar: alpha_i = a * pop_size_i
    b: float                     # scalar: beta_i  = b * pop_size_i
    noise_std: float
    base_per_capita: float
    # Heterogeneity knobs (see generate_historical_data docstring).
    surge_factor: float = 1.0
    heteroscedastic: bool = False
    noise_scale_factor: float = 0.0


def generate_historical_data(
    num_areas: int = 5,
    num_food_types: int = 3,
    num_days: int = 180,
    base_per_capita: float = 70.0,
    a: float = 10.0,             # alpha_i = a * pop_size_i  (weather sensitivity)
    b: float = 8.0,              # beta_i  = b * pop_size_i  (weekday sensitivity)
    noise_std: float = 15.0,
    weekday_prob: float = 5.0 / 7.0,
    sunny_prob: float = 0.6,
    demand_min: float = 0.0,
    demand_max: float = 300.0,
    pop_size: Optional[np.ndarray] = None,
    surge_factor: float = 1.0,         # NEW: see docstring; 1.0 = off
    heteroscedastic: bool = False,     # NEW: see docstring; False = off
    noise_scale_factor: float = 0.0,   # NEW: see docstring; 0.0 = off
    seed: int = 27,
    save_data: bool = False,
) -> tuple[pd.DataFrame, DGPTruth]:
    """
    Generate synthetic historical demand data for the contextual 2SSP.

    The contextual coefficients are fixed problem-setting parameters:
        alpha_i = a * pop_size_i,
        beta_i  = b * pop_size_i.
    Across seeds, only the daily covariate realizations and the noise eps
    change; the DGP's response structure (a, b, nominal) stays put.

    Parameters
    ----------
    num_areas, num_food_types : int
        Problem dimensions I and T.
    num_days : int
        Number of historical days to simulate (default 90).
    base_per_capita : float
        Multiplier on pop_size to set nominal demand (default 70).
    a, b : float
        Weather and weekday sensitivities per unit of pop_size. With the
        defaults (a=10, b=8) a sunny weekday lifts demand by about 10+8=18
        at a small area and 30+24=54 at a large area.
    noise_std : float
        Std of the additive Gaussian noise on demand.
    weekday_prob, sunny_prob : float
        Bernoulli probabilities for the daily covariates.
    pop_size : Optional[np.ndarray]
        If provided, override the random draw for this persistent area
        attribute. Must be a length-I integer array with values in {1, 2, 3}.
    surge_factor : float, default 1.0
        Multiplicative demand surge on (weather=1) AND (weekday=1) days
        (sunny weekdays). Mean demand on those days is multiplied by
        `surge_factor`; on all other days the multiplier is 1.0. Setting
        `surge_factor > 1` introduces a `weather x weekday` interaction
        that a linear OLS fit `a + b*weather + c*weekday` cannot capture
        (only 3 free parameters for 4 cells), so RF / GBM strictly
        outperform OLS in this regime. Default 1.0 reproduces the
        original additive DGP (no interaction).
    heteroscedastic : bool, default False
        If True, the residual std scales with the conditional mean:
            sigma_{d,i,t} = noise_std + noise_scale_factor * mean_d_{d,i,t}.
        Combined with `surge_factor > 1`, this produces a bimodal-mixture
        residual distribution (small noise on normal days, large noise on
        surge days) that a Gaussian-with-fixed-sigma sampler cannot match
        — the empirical-residual bootstrap (Sun's ER-SAA) does. Default
        False reproduces the original homoscedastic DGP.
    noise_scale_factor : float, default 0.0
        Slope of the heteroscedastic-noise relation; only used when
        `heteroscedastic=True`. With `noise_std=15` and
        `noise_scale_factor=0.1`, a mean of 200 gives sigma = 35 (vs 15
        on a mean-zero day).
    seed : int
        Seed for the RNG. All stochastic pieces (pop_size when not provided,
        daily covariates, noise) are derived from this single seed.
    save_data : bool
        If True, write the DataFrame to dataset/historical_data.csv (creating
        the dataset/ folder if needed). Default False.

    Returns
    -------
    df : pd.DataFrame
        Columns: area_index, day, weekday, weather, food_type, demand.
        Shape (num_days * num_areas * num_food_types, 6).
    truth : DGPTruth
        The ground-truth parameters of the DGP.
    """
    rng = np.random.default_rng(seed)
    I, T, D = num_areas, num_food_types, num_days

    # --- persistent per-area attribute -------------------------------------
    if pop_size is None:
        pop_size = rng.integers(1, 4, size=I)
    else:
        pop_size = np.asarray(pop_size, dtype=int)
        assert pop_size.shape == (I,), "pop_size must have shape (I,)"

    nominal = pop_size.astype(float) * base_per_capita    # shape (I,)
    alpha = a * pop_size.astype(float)                    # shape (I,)
    beta  = b * pop_size.astype(float)                    # shape (I,)

    # --- daily covariates --------------------------------------------------
    weekday = rng.binomial(1, weekday_prob, size=D)       # shape (D,)
    weather = rng.binomial(1, sunny_prob,   size=D)       # shape (D,)

    # --- vectorized demand construction ------------------------------------
    # mean demand has shape (D, I), broadcast to (D, I, T) via the food-type axis
    wx = weather[:, None].astype(float)                   # (D, 1)
    wd = weekday[:, None].astype(float)                   # (D, 1)

    mean_di = (
        nominal[None, :]
        + alpha[None, :] * wx
        + beta [None, :] * wd
    )                                                     # shape (D, I)

    # Multiplicative surge on (weather=1) AND (weekday=1) days. With binary
    # features the surge mask is the elementwise product wx * wd, which is
    # 1 only on sunny weekdays. surge_factor=1.0 reproduces the additive DGP.
    surge_mask = (wx * wd)                                # (D, 1)
    mean_di = mean_di * (1.0 + (surge_factor - 1.0) * surge_mask)

    mean_d = np.broadcast_to(mean_di[:, :, None], (D, I, T))

    # Heteroscedastic noise: sigma scales with the conditional mean. Bigger
    # mean -> bigger residual std. Combined with surge_factor > 1 this gives
    # a bimodal residual mixture that a fixed-sigma Gaussian misses but the
    # empirical-residual bootstrap recovers.
    if heteroscedastic:
        sigma = noise_std + noise_scale_factor * mean_d   # (D, I, T)
    else:
        sigma = noise_std                                  # scalar
    noise = rng.normal(0.0, 1.0, size=(D, I, T)) * sigma
    demand = np.clip(mean_d + noise, demand_min, demand_max)   # truncate to the DGP support

    # --- flatten to long-format DataFrame ----------------------------------
    day_idx, area_idx, ft_idx = np.meshgrid(
        np.arange(D), np.arange(I), np.arange(T), indexing="ij"
    )
    df = pd.DataFrame({
        "area_index": area_idx.ravel(),
        "day":        day_idx.ravel(),
        "weekday":    np.broadcast_to(weekday[:, None, None], (D, I, T)).ravel(),
        "weather":    np.broadcast_to(weather[:, None, None], (D, I, T)).ravel(),
        "food_type":  ft_idx.ravel(),
        "demand":     demand.ravel(),
    })

    truth = DGPTruth(
        pop_size=pop_size,
        nominal=nominal,
        alpha=alpha,
        beta=beta,
        a=a,
        b=b,
        noise_std=noise_std,
        base_per_capita=base_per_capita,
        surge_factor=surge_factor,
        heteroscedastic=heteroscedastic,
        noise_scale_factor=noise_scale_factor,
    )

    if save_data:
        out_path = Path("data") / "historical_data.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"Historical data saved to {out_path}")

    return df, truth


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df, truth = generate_historical_data(
        num_areas=5, num_food_types=3, num_days=90, seed=42, save_data=True,
    )

    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}\n")
    print("First 10 rows:")
    print(df.head(10).to_string(index=False))

    print(f"\nFixed sensitivity constants: a={truth.a}, b={truth.b}")
    print("Per-area attributes:")
    for i in range(len(truth.pop_size)):
        print(
            f"  area {i}: pop_size={truth.pop_size[i]}  "
            f"nominal={truth.nominal[i]:.0f}  "
            f"alpha={truth.alpha[i]:.0f}  beta={truth.beta[i]:.0f}"
        )

    print("\nDemand summary by area and food type (mean +/- std):")
    summary = df.groupby(["area_index", "food_type"])["demand"].agg(["mean", "std"])
    print(summary.round(2).to_string())

    # Oracle sanity check: predicted mean given observed covariate frequencies
    pwx = df["weather"].mean()
    pwd = df["weekday"].mean()
    print(
        f"\nEmpirical Pr(weather=1)={pwx:.3f}, "
        f"Pr(weekday=1)={pwd:.3f}"
    )
    print("Predicted vs empirical mean demand by area (averaged over food types):")
    for i in range(len(truth.pop_size)):
        pred = truth.nominal[i] + truth.alpha[i] * pwx + truth.beta[i] * pwd
        emp = df[df["area_index"] == i]["demand"].mean()
        print(f"  area {i}: predicted={pred:.2f}  empirical={emp:.2f}")
