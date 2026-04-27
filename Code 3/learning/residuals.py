"""
Compute empirical residuals from a fitted demand predictor on historical data.

Used by `dataset.scenario_generator.scenarios_for_method(method='contextual_er',
training_residuals=...)` to bootstrap noise from the actual training-set
prediction errors instead of from a parametric Gaussian.

The helper is predictor-agnostic: it works for any object that exposes
`predict(weather, weekday) -> (N, I, T)`. Both `DemandModel` (OLS, in
`learning.ols_fit`) and `NonparametricDemandModel` (RF/GBM, in
`learning.erm_fit`) qualify.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_training_residuals(
    model,
    df: pd.DataFrame,
    num_food_types: int,
) -> np.ndarray:
    """
    Empirical residuals on the historical training set.

    Parameters
    ----------
    model : predictor with `.predict(weather, weekday) -> (N, I, T)`.
    df    : historical DataFrame (schema from
            `dataset.data_generation.generate_historical_data`):
            columns area_index, day, weekday, weather, food_type, demand.
    num_food_types : T, used to validate the reshape.

    Returns
    -------
    residuals : (n_days, I, T) array of `actual - predicted`.
        Each row is the (I, T) residual vector observed on one historical
        day. `scenarios_for_method(method='contextual_er')` bootstraps whole
        rows with replacement, preserving any cross-(i, t) residual
        correlation that per-cell Gaussian sampling would collapse.
    """
    required = {"area_index", "day", "weekday", "weather", "food_type", "demand"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing columns: {missing}")

    # Canonical (day, area, food_type) ordering — matches the broadcast layout
    # used by data_generation.generate_historical_data, so reshape is safe.
    df_sorted = df.sort_values(["day", "area_index", "food_type"], kind="stable")

    # Per-day context features (constant within a day across all (i, t) rows).
    day_features = (
        df_sorted.groupby("day")[["weather", "weekday"]]
                 .first()
                 .sort_index()
    )
    weather_seq = day_features["weather"].to_numpy()
    weekday_seq = day_features["weekday"].to_numpy()
    n_days = len(day_features)

    I = int(df["area_index"].max()) + 1
    T = int(num_food_types)
    expected_rows = n_days * I * T
    if len(df_sorted) != expected_rows:
        raise ValueError(
            f"DataFrame has {len(df_sorted)} rows but expected "
            f"n_days * I * T = {n_days} * {I} * {T} = {expected_rows}. "
            "Are you sure `df` covers a complete (day, area, food_type) grid?"
        )

    demand = df_sorted["demand"].to_numpy().reshape(n_days, I, T)

    d_hat = np.asarray(model.predict(weather_seq, weekday_seq), dtype=float)
    if d_hat.shape != (n_days, I, T):
        raise ValueError(
            f"model.predict returned shape {d_hat.shape}; expected ({n_days}, {I}, {T})"
        )

    return demand - d_hat


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from dataset.data_generation import generate_historical_data
    from learning.ols_fit import fit_demand_model

    df, truth = generate_historical_data(
        num_areas=3, num_food_types=3, num_days=90, seed=27,
    )
    model = fit_demand_model(df, num_food_types=3, pool_across_food_types=True)

    residuals = compute_training_residuals(model, df, num_food_types=3)
    print(f"residuals shape: {residuals.shape}  (n_days, I, T)")
    print(f"  mean   = {residuals.mean():+.4f}  (~0 if predictor is unbiased)")
    print(f"  std    = {residuals.std():.4f}    (true DGP noise = {truth.noise_std})")
    print(f"  range  = [{residuals.min():+.2f}, {residuals.max():+.2f}]")
