"""
Nonparametric demand-model fitter for the food-truck C-2SSP.

Drop-in replacement for `learning.ols_fit.fit_demand_model`: takes the same
historical DataFrame, returns an object with the same `.predict` /
`.residual_std` / `.num_areas` / `.num_food_types` interface as
`dataset.scenario_generator.DemandModel`. Downstream
(`scenarios_for_method`, the solver, etc.) doesn't know or care which
fitter produced the predictor.

Hypothesis classes available:
    'rf'  : RandomForestRegressor          (recommended default)
    'gbm' : HistGradientBoostingRegressor  (often competitive on tabular)
    'knn' : KNeighborsRegressor            (degenerate with binary features)
    'dt'  : DecisionTreeRegressor          (single tree, mostly a contrast)

Together with `learning.residuals.compute_training_residuals` and the new
`method='contextual_er'` branch in `scenarios_for_method`, this completes
the four DBL ablation cells:

    Linear (ols_fit)   x  Gaussian noise (contextual_residual)
    Linear (ols_fit)   x  Empirical residuals (contextual_er)
    Nonparametric (this file) x  Gaussian noise
    Nonparametric (this file) x  Empirical residuals
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


class NonparametricDemandModel:
    """
    Wraps a fitted sklearn-style estimator and exposes the same surface as
    `DemandModel` for the scenario generator. Specifically:
        - .num_areas, .num_food_types
        - .predict(weather, weekday) -> (N, I, T)
        - .residual_std                 (I, T) — per-(i, t) training-set noise
    """

    def __init__(
        self,
        estimator,
        num_areas: int,
        num_food_types: int,
        residual_std: np.ndarray,
        kind: str = "rf",
    ):
        self.estimator       = estimator
        self._num_areas      = int(num_areas)
        self._num_food_types = int(num_food_types)
        self.residual_std    = np.asarray(residual_std, dtype=float)
        self.kind            = kind

        if self.residual_std.shape != (self._num_areas, self._num_food_types):
            raise ValueError(
                f"residual_std must have shape "
                f"({self._num_areas}, {self._num_food_types}); "
                f"got {self.residual_std.shape}"
            )

    @property
    def num_areas(self) -> int:
        return self._num_areas

    @property
    def num_food_types(self) -> int:
        return self._num_food_types

    def predict(self, weather: np.ndarray, weekday: np.ndarray) -> np.ndarray:
        """
        Vectorized point prediction. weather/weekday are (N,); returns (N, I, T).
        """
        wx = np.asarray(weather, dtype=float).reshape(-1)
        wd = np.asarray(weekday, dtype=float).reshape(-1)
        if wx.shape != wd.shape:
            raise ValueError(
                f"weather and weekday must have the same shape; "
                f"got {wx.shape} and {wd.shape}"
            )
        X = np.column_stack([wx, wd])                                  # (N, 2)
        flat = np.asarray(self.estimator.predict(X), dtype=float)      # (N, I*T)
        return flat.reshape(-1, self._num_areas, self._num_food_types)


# ---------------------------------------------------------------------------
# Estimator factory
# ---------------------------------------------------------------------------
def _make_estimator(kind: str, random_state: int):
    """Construct a fresh, untrained multi-output estimator."""
    kind = kind.lower()
    if kind == "rf":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(
            n_estimators=200,
            random_state=random_state,
            n_jobs=-1,
        )
    if kind == "gbm":
        # HistGradientBoostingRegressor is single-output; wrap for multi-target.
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.multioutput import MultiOutputRegressor
        return MultiOutputRegressor(
            HistGradientBoostingRegressor(
                learning_rate=0.1,
                max_depth=6,
                max_iter=200,
                random_state=random_state,
            )
        )
    if kind == "knn":
        from sklearn.neighbors import KNeighborsRegressor
        return KNeighborsRegressor(n_neighbors=5)
    if kind == "dt":
        from sklearn.tree import DecisionTreeRegressor
        return DecisionTreeRegressor(random_state=random_state)
    raise ValueError(
        f"Unknown predictor kind '{kind}'. Use one of: 'rf', 'gbm', 'knn', 'dt'."
    )


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------
def fit_demand_model_nonparametric(
    df: pd.DataFrame,
    num_food_types: int,
    kind: str = "rf",
    random_state: int = 0,
) -> NonparametricDemandModel:
    """
    Fit a multi-output nonparametric demand model on historical data.

    Same data contract as `learning.ols_fit.fit_demand_model`:
        df columns: area_index, day, weekday, weather, food_type, demand.

    The target is the full (I, T) demand vector flattened across (i, t) so
    that one estimator predicts the entire demand grid in one call. RF
    handles this natively; HistGradientBoosting / KNN / DecisionTree are
    wrapped via MultiOutputRegressor where needed.

    Parameters
    ----------
    df : pd.DataFrame
    num_food_types : int
    kind : {'rf', 'gbm', 'knn', 'dt'}
    random_state : int

    Returns
    -------
    NonparametricDemandModel
    """
    required = {"area_index", "day", "weekday", "weather", "food_type", "demand"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing columns: {missing}")

    # Canonical (day, area, food_type) ordering — matches the broadcast
    # layout in dataset.data_generation, so reshape is safe.
    df_sorted = df.sort_values(["day", "area_index", "food_type"], kind="stable")

    day_features = (
        df_sorted.groupby("day")[["weather", "weekday"]]
                 .first()
                 .sort_index()
    )
    weather_seq = day_features["weather"].to_numpy(dtype=float)
    weekday_seq = day_features["weekday"].to_numpy(dtype=float)
    n_days = len(day_features)

    I = int(df["area_index"].max()) + 1
    T = int(num_food_types)
    expected_rows = n_days * I * T
    if len(df_sorted) != expected_rows:
        raise ValueError(
            f"DataFrame has {len(df_sorted)} rows but expected "
            f"n_days * I * T = {n_days} * {I} * {T} = {expected_rows}."
        )

    demand = df_sorted["demand"].to_numpy(dtype=float).reshape(n_days, I, T)

    X = np.column_stack([weather_seq, weekday_seq])                    # (n_days, 2)
    y = demand.reshape(n_days, I * T)                                  # (n_days, I*T)

    estimator = _make_estimator(kind, random_state)
    estimator.fit(X, y)

    # Per-(i, t) training residual std for the Gaussian-noise sampler.
    pred = np.asarray(estimator.predict(X), dtype=float).reshape(n_days, I, T)
    residual_std = (demand - pred).std(axis=0)                         # (I, T)
    # Avoid divide-by-zero if a cell has no residual variation (overfit tree).
    residual_std = np.where(residual_std < 1e-8, 1e-8, residual_std)

    return NonparametricDemandModel(
        estimator      = estimator,
        num_areas      = I,
        num_food_types = T,
        residual_std   = residual_std,
        kind           = kind,
    )


# ---------------------------------------------------------------------------
# Self-test: fit each available predictor on synthetic historical data and
# verify the predict / residual_std interface.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from dataset.data_generation import generate_historical_data
    from learning.residuals import compute_training_residuals

    df, truth = generate_historical_data(
        num_areas=3, num_food_types=3, num_days=90, seed=27,
    )
    print(f"Historical data: {len(df)} rows, "
          f"true noise_std = {truth.noise_std}")

    for kind in ("rf", "gbm", "knn", "dt"):
        try:
            model = fit_demand_model_nonparametric(df, num_food_types=3, kind=kind)
        except ImportError as e:
            print(f"  [{kind}] skipped (missing dep): {e}")
            continue

        # Interface check
        d_hat = model.predict(np.array([0, 1, 1]), np.array([1, 0, 1]))
        assert d_hat.shape == (3, 3, 3), d_hat.shape

        # Residual diagnostics
        resid = compute_training_residuals(model, df, num_food_types=3)
        print(
            f"  [{kind}] residual_std mean = {model.residual_std.mean():.3f} | "
            f"empirical residual std = {resid.std():.3f} | "
            f"mean = {resid.mean():+.3f}"
        )
