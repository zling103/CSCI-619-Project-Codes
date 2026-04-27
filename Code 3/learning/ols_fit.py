"""
OLS demand-model fitter for the food-truck C-2SSP.

Reads the historical DataFrame produced by data_generation.generate_historical_data
and returns a DemandModel (from scenario_generator.py) containing the estimated
intercept, slope on weather, slope on weekday, and residual std.

Pooling mode
------------
Under the default option of the DGP, alpha and beta are shared across food types at the
same area (alpha_i = a*pop_size_i and beta_i = b*pop_size_i are scalar per i).
So the statistically correct thing is to POOL observations across food types
when fitting per area — this triples the effective sample size per regression
and tightens the estimates.

The returned DemandModel still has shape-(I, T) arrays, with the pooled
estimates broadcast across the food-type axis. Downstream code (scenario
generator, solver) doesn't know or care how the fit was done.

Unpooled mode is kept as an option for future DGPs where per-(i, t)
coefficients actually differ.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataset.scenario_generator import DemandModel


def fit_demand_model(
    df: pd.DataFrame,
    num_food_types: int,
    pool_across_food_types: bool = True,
) -> DemandModel:
    """
    Fit a linear demand model d ~ 1 + weather + weekday on historical data.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: area_index, weekday, weather, food_type, demand.
        This is exactly the schema produced by generate_historical_data.
    num_food_types : int
        T. Used to broadcast pooled estimates to shape (I, T).
    pool_across_food_types : bool
        If True (default), fit one regression per area, using all food types
        as independent observations. Correct under option A of the DGP where
        alpha_i, beta_i are shared across t.
        If False, fit a separate regression for each (area, food_type) pair.

    Returns
    -------
    DemandModel with intercept, alpha, beta, residual_std each shape (I, T).
    """
    required = {"area_index", "weekday", "weather", "food_type", "demand"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing columns: {missing}")

    I = int(df["area_index"].max()) + 1
    T = num_food_types

    intercept = np.zeros((I, T))
    alpha     = np.zeros((I, T))
    beta      = np.zeros((I, T))
    residual  = np.zeros((I, T))

    if pool_across_food_types:
        # One OLS per area, pooling all food types.
        for i in range(I):
            sub = df[df["area_index"] == i]
            coef, sigma = _ols_fit(
                weather=sub["weather"].to_numpy(dtype=float),
                weekday=sub["weekday"].to_numpy(dtype=float),
                demand =sub["demand" ].to_numpy(dtype=float),
            )
            # Broadcast pooled estimates across food types.
            intercept[i, :] = coef[0]
            alpha    [i, :] = coef[1]
            beta     [i, :] = coef[2]
            residual [i, :] = sigma
    else:
        # One OLS per (area, food_type).
        for i in range(I):
            for t in range(T):
                sub = df[(df["area_index"] == i) & (df["food_type"] == t)]
                coef, sigma = _ols_fit(
                    weather=sub["weather"].to_numpy(dtype=float),
                    weekday=sub["weekday"].to_numpy(dtype=float),
                    demand =sub["demand" ].to_numpy(dtype=float),
                )
                intercept[i, t] = coef[0]
                alpha    [i, t] = coef[1]
                beta     [i, t] = coef[2]
                residual [i, t] = sigma

    return DemandModel(
        intercept    = intercept,
        alpha        = alpha,
        beta         = beta,
        residual_std = residual,
    )


def _ols_fit(
    weather: np.ndarray,
    weekday: np.ndarray,
    demand:  np.ndarray,
) -> tuple[np.ndarray, float]:
    """
    Fit demand ~ intercept + b_weather*weather + b_weekday*weekday via OLS.

    Returns
    -------
    coef : shape (3,) array [intercept, alpha, beta]
    sigma : residual std (unbiased, divided by n - p)
    """
    n = len(demand)
    X = np.column_stack([np.ones(n), weather, weekday])   # (n, 3)
    coef, *_ = np.linalg.lstsq(X, demand, rcond=None)
    resid = demand - X @ coef
    # Unbiased residual std: divide by (n - p), p = 3.
    dof = max(n - X.shape[1], 1)
    sigma = float(np.sqrt((resid @ resid) / dof))
    return coef, sigma


# ---------------------------------------------------------------------------
# Self-test: generate historical data from a known DGP, fit the model, and
# verify recovery of the true coefficients.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from dataset.data_generation import generate_historical_data

    df, truth = generate_historical_data(
        num_areas=5, num_food_types=3, num_days=90, seed=27,
    )

    print("=== Pooled fit (1 regression per area) ===")
    model_pooled = fit_demand_model(df, num_food_types=3, pool_across_food_types=True)

    print(f"{'area':>4} {'pop':>4}  "
          f"{'nominal (true/est)':>22}  "
          f"{'alpha (true/est)':>20}  "
          f"{'beta (true/est)':>20}  "
          f"{'sigma (true/est)':>18}")
    for i in range(5):
        print(
            f"{i:>4} {truth.pop_size[i]:>4}  "
            f"{truth.nominal[i]:>10.2f} / {model_pooled.intercept[i, 0]:>8.2f}  "
            f"{truth.alpha[i]:>9.2f} / {model_pooled.alpha[i, 0]:>8.2f}  "
            f"{truth.beta [i]:>9.2f} / {model_pooled.beta [i, 0]:>8.2f}  "
            f"{truth.noise_std:>8.2f} / {model_pooled.residual_std[i, 0]:>7.2f}"
        )

    # Check: in pooled mode, estimates should be identical across food types
    # (we broadcast the scalar estimate to shape (T,)).
    print(f"\nBroadcast check: alpha[:, 0] == alpha[:, 2] for all areas? "
          f"{np.allclose(model_pooled.alpha[:, 0], model_pooled.alpha[:, 2])}")

    print("\n=== Unpooled fit (1 regression per (area, food_type)) ===")
    model_unpooled = fit_demand_model(df, num_food_types=3, pool_across_food_types=False)

    # Unpooled estimates should DIFFER slightly across food types because of
    # independent noise realizations per (i, t).
    print(f"alpha spread across food types for area 0: "
          f"{model_unpooled.alpha[0, :].round(3).tolist()}")
    print(f"alpha spread across food types for area 1: "
          f"{model_unpooled.alpha[1, :].round(3).tolist()}")

    # Pooled estimate should be the average (approximately) of the unpooled ones
    print(f"\nPooled alpha[0, 0] vs mean of unpooled alpha[0, :]: "
          f"{model_pooled.alpha[0, 0]:.3f}  vs  "
          f"{model_unpooled.alpha[0, :].mean():.3f}")