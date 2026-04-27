"""
Pseudo-DFL (decision-aware ERM) demand-model fitter for the food-truck C-2SSP.

Trains a linear predictor d_hat(weather, weekday) by solving a SINGLE MILP
whose loss is the realized two-stage cost when the prediction is used as
an upper bound on serving capacity and the unmet-demand penalty is computed
against the realized historical demand.

Formulation
-----------
For each historical day k = 1, ..., S with features f_k = [1, weather_k,
weekday_k] and realized demand d_k^true:

    min   (1/S) sum_k [ c^f * sum_j y_{k,j}
                       + sum_{j,t} c^a_t * w_{k,j,t}
                       + sum_{i,t}  c^u_t * u_{k,i,t} ]
    s.t.  sum_t w_{k,j,t} <= C * y_{k,j}                       for all k, j
          s_{k,i,j,t}     <= e_{i,j} * w_{k,j,t}                for all k, i, j, t
          sum_j s_{k,i,j,t} <= (H f_k)_{i,t}                    for all k, i, t   <- prediction caps service
          u_{k,i,t} >= d_k^true_{i,t} - sum_j s_{k,i,j,t}        for all k, i, t   <- unmet vs TRUTH
          y_k in {0,1}^J;  w, s, u >= 0

The predictor is parameterized as
    d_hat_{i,t}(weather, weekday) = H_int_{i,t}
                                  + H_alp_{i,t} * weather
                                  + H_bet_{i,t} * weekday
which mirrors the parameterization of `dataset.scenario_generator.DemandModel`.
The fitted (intercept, alpha, beta) are wrapped in a plain DemandModel so the
SPO predictor plugs into `scenarios_for_method` exactly like the OLS-fit
version. No downstream changes required.

Relation to true DFL
--------------------
This is a SINGLE-MILP surrogate that captures the asymmetric cost impact
(c^u >> c^a) which a pure MSE-trained DBL predictor never learns. It does
NOT enforce inner-optimality of (y_k, w_k) under the prediction d_hat_k, so
it is not a literal SPO regret minimizer. True DFL formulations for
two-stage MILPs (Chen et al. 2022 single-MILP SPO; Wu et al. 2026 BiMILP)
require either a careful KKT/strong-duality encoding of the inner
optimization or iterative bilevel decomposition algorithms — out of scope
for this project. We refer to the formulation here as "pseudo-DFL" or
"decision-aware ERM" in the writeup.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import gurobipy as gp

from helpers.parameter import Params
from dataset.scenario_generator import DemandModel
from learning.ols_fit import fit_demand_model


def fit_demand_model_spo(
    df: pd.DataFrame,
    params: Params,
    num_food_types: int,
    max_train_samples: Optional[int] = None,
    time_limit: Optional[float] = None,
    warm_start_ols: bool = True,
    verbose: bool = False,
    seed: int = 0,
) -> tuple[DemandModel, dict]:
    """
    Fit the pseudo-DFL linear demand model and return it as a DemandModel.

    Parameters
    ----------
    df : pd.DataFrame
        Historical training data; same schema as `learning.ols_fit.fit_demand_model`
        (columns: area_index, day, weekday, weather, food_type, demand).
    params : Params
        Provides cost coefficients (c^f, c^a, c^u), capacity C, and
        eligibility e — the downstream-optimization information that
        differentiates this fit from MSE-based DBL.
    num_food_types : int
        T. Used for shape validation.
    max_train_samples : int, optional
        If set and < n_days, randomly subsample to this many training days
        before building the MILP. The MILP scales as O(S * I * J * T) in
        variables; subsampling is a safety valve for large instances.
        Note: the predictor's residual_std is still computed on the FULL
        training set so the Gaussian-noise sampler remains well-calibrated.
    time_limit : float, optional
        Gurobi time limit (seconds). Defaults to params.time_limit.
    warm_start_ols : bool
        If True, warm-start H from an OLS pre-fit. Almost always worth it.
    verbose : bool
        If True, let Gurobi print its log.
    seed : int
        RNG seed for the subsample selection.

    Returns
    -------
    model : DemandModel
        Fitted predictor, drop-in compatible with `scenarios_for_method`.
    info : dict
        MILP solve diagnostics (obj, gap, runtime, solve status, sample counts,
        variable / constraint counts).
    """
    # --- Validate inputs ----------------------------------------------------
    required = {"area_index", "day", "weekday", "weather", "food_type", "demand"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing columns: {missing}")

    # --- Reshape historical data -------------------------------------------
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

    if I != params.num_areas or T != params.num_food_types:
        raise ValueError(
            f"DataFrame shape (I={I}, T={T}) does not match Params "
            f"(num_areas={params.num_areas}, num_food_types={params.num_food_types})"
        )
    expected_rows = n_days * I * T
    if len(df_sorted) != expected_rows:
        raise ValueError(
            f"DataFrame has {len(df_sorted)} rows but expected "
            f"n_days * I * T = {expected_rows}."
        )

    demand_full = df_sorted["demand"].to_numpy(dtype=float).reshape(n_days, I, T)

    # --- Optional subsampling ----------------------------------------------
    if max_train_samples is not None and max_train_samples < n_days:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(n_days, size=max_train_samples, replace=False))
        weather_train = weather_seq[idx]
        weekday_train = weekday_seq[idx]
        demand_train  = demand_full[idx]
    else:
        weather_train = weather_seq
        weekday_train = weekday_seq
        demand_train  = demand_full
    S = int(len(weather_train))
    if S == 0:
        raise ValueError("No training samples available (max_train_samples=0?).")

    # --- Problem constants --------------------------------------------------
    J   = params.num_locations
    C   = params.capacity
    c_f = params.fixed_cost
    c_a = params.acquisition_cost                     # (T,)
    c_u = params.unmet_penalty                        # (T,)
    e   = params.e                                    # (I, J)

    # --- Optional OLS warm start --------------------------------------------
    ols_warm = None
    if warm_start_ols:
        ols_warm = fit_demand_model(
            df, num_food_types=T, pool_across_food_types=True,
        )

    # --- Build MILP ---------------------------------------------------------
    model = gp.Model("SPO_pseudo_DFL")
    if not verbose:
        model.setParam("OutputFlag", 0)
    model.setParam("Threads", 8)
    if time_limit is not None:
        model.setParam("TimeLimit", float(time_limit))
    elif params.time_limit is not None:
        model.setParam("TimeLimit", float(params.time_limit))

    # Predictor parameters: H_int (intercept), H_alp (weather), H_bet (weekday).
    # Each shape (I, T). Unbounded; warm-started from OLS if requested.
    H_int = model.addVars(I, T, lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY,
                          vtype=gp.GRB.CONTINUOUS, name="H_int")
    H_alp = model.addVars(I, T, lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY,
                          vtype=gp.GRB.CONTINUOUS, name="H_alp")
    H_bet = model.addVars(I, T, lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY,
                          vtype=gp.GRB.CONTINUOUS, name="H_bet")

    if ols_warm is not None:
        for i in range(I):
            for t in range(T):
                H_int[i, t].Start = float(ols_warm.intercept[i, t])
                H_alp[i, t].Start = float(ols_warm.alpha[i, t])
                H_bet[i, t].Start = float(ols_warm.beta[i, t])

    # Per-scenario decision variables.
    y = model.addVars(S, J,           vtype=gp.GRB.BINARY,     name="y")
    w = model.addVars(S, J, T, lb=0.0, vtype=gp.GRB.CONTINUOUS, name="w")
    s = model.addVars(S, I, J, T, lb=0.0, vtype=gp.GRB.CONTINUOUS, name="s")
    u = model.addVars(S, I, T,    lb=0.0, vtype=gp.GRB.CONTINUOUS, name="u")

    # Objective: average-of-realized-cost over training samples. No revenue
    # term — we focus on the cost-asymmetry signal that DBL/MSE misses.
    inv_S = 1.0 / S
    model.setObjective(
        inv_S * gp.quicksum(c_f * y[k, j]
                            for k in range(S) for j in range(J))
        + inv_S * gp.quicksum(c_a[t] * w[k, j, t]
                              for k in range(S) for j in range(J)
                              for t in range(T))
        + inv_S * gp.quicksum(c_u[t] * u[k, i, t]
                              for k in range(S) for i in range(I)
                              for t in range(T)),
        gp.GRB.MINIMIZE,
    )

    # Constraints (per training sample k).
    for k in range(S):
        wx_k = float(weather_train[k])
        wd_k = float(weekday_train[k])

        for j in range(J):
            model.addConstr(
                gp.quicksum(w[k, j, t] for t in range(T)) <= C * y[k, j],
                name=f"capacity_{k}_{j}",
            )

        for i in range(I):
            for j in range(J):
                for t in range(T):
                    model.addConstr(
                        s[k, i, j, t] <= float(e[i, j]) * w[k, j, t],
                        name=f"service_cap_{k}_{i}_{j}_{t}",
                    )

        # Served <= predicted demand. Predicted is a linear function of H
        # (H_int + H_alp*weather + H_bet*weekday), with weather/weekday CONSTANTS
        # for this training sample, so the constraint stays linear.
        for i in range(I):
            for t in range(T):
                pred_it = H_int[i, t] + H_alp[i, t] * wx_k + H_bet[i, t] * wd_k
                model.addConstr(
                    gp.quicksum(s[k, i, j, t] for j in range(J)) <= pred_it,
                    name=f"served_le_pred_{k}_{i}_{t}",
                )

        # Unmet penalty against TRUTH (this is what makes the formulation
        # decision-aware rather than a degenerate predict-zero collapse).
        for i in range(I):
            for t in range(T):
                model.addConstr(
                    u[k, i, t] >= demand_train[k, i, t]
                                  - gp.quicksum(s[k, i, j, t] for j in range(J)),
                    name=f"unmet_truth_{k}_{i}_{t}",
                )

    # --- Solve --------------------------------------------------------------
    model.optimize()

    info = {
        "n_train_samples": S,
        "n_total_samples": int(n_days),
        "obj":             float(model.ObjVal) if model.SolCount > 0 else float("nan"),
        "gap":             float(model.MIPGap)  if model.SolCount > 0 else float("nan"),
        "runtime":         float(model.Runtime),
        "status":          int(model.Status),
        "num_vars":        int(model.NumVars),
        "num_constrs":     int(model.NumConstrs),
        "warm_started":    bool(warm_start_ols),
    }

    if model.SolCount == 0:
        raise RuntimeError(
            f"SPO MILP did not find a feasible solution (status={model.Status})."
        )

    # --- Extract H and build the predictor ----------------------------------
    intercept = np.array([[H_int[i, t].X for t in range(T)] for i in range(I)])
    alpha     = np.array([[H_alp[i, t].X for t in range(T)] for i in range(I)])
    beta      = np.array([[H_bet[i, t].X for t in range(T)] for i in range(I)])

    # Compute residual_std on the FULL training set so the Gaussian-noise
    # sampler in scenarios_for_method stays well-calibrated even when the
    # SPO MILP itself trained on a subsample.
    predictions = (
        intercept[None, :, :]
        + alpha[None, :, :] * weather_seq[:, None, None]
        + beta [None, :, :] * weekday_seq[:, None, None]
    )                                                     # (n_days, I, T)
    residuals = demand_full - predictions
    residual_std = residuals.std(axis=0)                   # (I, T)
    residual_std = np.where(residual_std < 1e-8, 1e-8, residual_std)

    return DemandModel(
        intercept    = intercept,
        alpha        = alpha,
        beta         = beta,
        residual_std = residual_std,
    ), info


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from dataset.data_generation import generate_historical_data
    from learning.residuals import compute_training_residuals

    I, J, T = 3, 6, 3
    pop_size = np.array([1, 2, 3])

    params = Params(
        num_areas         = I,
        num_locations     = J,
        num_food_types    = T,
        capacity          = 200.0,
        fixed_cost        = 50.0,
        demand_min        = 0.0,
        demand_max        = 600.0,
        # Asymmetric costs so the DFL signal is visible
        acquisition_cost  = np.array([2.0, 2.0, 2.0]),
        unmet_penalty     = np.array([20.0, 20.0, 20.0]),
        revenue           = np.array([0.0, 0.0, 0.0]),
        num_scenarios_saa = 20,
        time_limit        = 120.0,
        seed              = 7,
    )

    df, truth = generate_historical_data(
        num_areas=I, num_food_types=T, num_days=30,           # small for fast self-test
        pop_size=pop_size, seed=42,
        surge_factor=3.0, heteroscedastic=True, noise_scale_factor=0.1,
    )
    print(f"Historical data: {len(df)} rows, num_days=30")
    print(f"True demand range: [{df['demand'].min():.1f}, {df['demand'].max():.1f}]")

    print("\n=== OLS baseline (DBL) ===")
    ols_model = fit_demand_model(df, num_food_types=T, pool_across_food_types=True)
    print(f"  intercept = {ols_model.intercept}")
    print(f"  alpha     = {ols_model.alpha}")
    print(f"  beta      = {ols_model.beta}")

    print("\n=== Pseudo-DFL (single-MILP SPO surrogate) ===")
    spo_model, info = fit_demand_model_spo(
        df, params, num_food_types=T,
        max_train_samples=None, time_limit=120.0,
        warm_start_ols=True, verbose=False,
    )
    print(f"  obj         = {info['obj']:.3f}")
    print(f"  gap         = {info['gap']*100:.4f}%")
    print(f"  runtime     = {info['runtime']:.2f}s")
    print(f"  num_vars    = {info['num_vars']}")
    print(f"  num_constrs = {info['num_constrs']}")
    print(f"  intercept   = {spo_model.intercept}")
    print(f"  alpha       = {spo_model.alpha}")
    print(f"  beta        = {spo_model.beta}")

    print("\n=== Comparison: predicted demand on each (weather, weekday) cell ===")
    print(f"{'cell':<10} {'OLS pred':>30}  {'SPO pred':>30}")
    for wx, wd in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        ols_p = ols_model.predict(np.array([wx]), np.array([wd]))[0]
        spo_p = spo_model.predict(np.array([wx]), np.array([wd]))[0]
        print(f"  ({wx},{wd}) {str(ols_p.flatten().round(1)):>30}  {str(spo_p.flatten().round(1)):>30}")
    print()
    print("Expected: SPO predictions are biased upward vs OLS (especially on the")
    print("(1,1) surge cell), because under-prediction is penalized harder than")
    print("over-prediction in the loss (c^u >> c^a).")
