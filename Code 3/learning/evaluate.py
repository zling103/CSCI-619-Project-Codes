"""
In-sample evaluation harness for the six contextual-learning cells.

Pipeline (per cell):

    1. Fit a regression model on historical data:
        - OLS  (learning.ols_fit)
        - RF   (learning.erm_fit, kind='rf')
        - SPO  (learning.spo_fit, single-MILP pseudo-DFL)
    2. Compute empirical residuals (only needed for ER cells).
    3. Generate N scenario contexts (drawn from the same DGP that produced
       historical) and project to (N, I, T) demand using
       method='contextual_residual' (Gaussian) or method='contextual_er'
       (empirical-residual bootstrap).
    4. Solve the SAA MILP over those N scenarios.
    5. Record SAA objective, first-stage cost, average per-scenario unmet
       demand, and average per-scenario unmet penalty cost.

Six cells in the comparison matrix:

                  +----------------+----------------+
                  | Gaussian noise | ER bootstrap   |
    +-------------+----------------+----------------+
    | OLS (DBL)   | Linear-Gauss   | Linear-ER      |
    | RF  (DBL)   | NonP-Gauss     | NonP-ER        |
    | SPO (pDFL)  | SPO-Gauss      | SPO-ER         |
    +-------------+----------------+----------------+

All cells share the same scenario context bundle (`generate_scenarios` is
called once with a fixed seed). What differs across cells is only the
predictor + noise sampler; everything downstream is identical, so any
differences in cost are attributable to the contextual-learning choice.
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np
import pandas as pd

from helpers.parameter import Params
from dataset.data_generation import DGPTruth
from dataset.scenario_generator import generate_scenarios, scenarios_for_method
from learning.ols_fit import fit_demand_model
from learning.erm_fit import fit_demand_model_nonparametric
from learning.spo_fit import fit_demand_model_spo
from learning.residuals import compute_training_residuals
from subproblems.saa import saa
from subproblems.recourse_problem import recourse_problem


# Default cell list. Each entry is (cell_name, predictor_kind, noise_kind),
# where predictor_kind in {'ols', 'rf', 'spo'} and noise_kind in {'gauss', 'er'}.
DEFAULT_CELLS = [
    ("Linear-Gauss", "ols", "gauss"),
    ("Linear-ER",    "ols", "er"),
    ("NonP-Gauss",   "rf",  "gauss"),
    ("NonP-ER",      "rf",  "er"),
    ("SPO-Gauss",    "spo", "gauss"),
    ("SPO-ER",       "spo", "er"),
]


def run_in_sample_evaluation(
    df: pd.DataFrame,
    truth: DGPTruth,
    params: Params,
    num_food_types: int,
    cells: Optional[list] = None,
    n_scenarios: Optional[int] = None,
    nonparametric_kind: str = "rf",
    spo_max_train_samples: Optional[int] = None,
    spo_time_limit: Optional[float] = 120.0,
    spo_warm_start_ols: bool = True,
    saa_time_limit: Optional[float] = None,
    scenario_seed: int = 1001,
    residual_seed: int = 2027,
    n_oos_scenarios: Optional[int] = None,
    oos_seed: int = 9001,
    verbose: bool = True,
) -> dict:
    """
    Train each predictor once, build the shared scenario context bundle once,
    then loop over cells projecting + solving SAA.

    Parameters
    ----------
    df : pd.DataFrame
        Historical training data (schema from
        `dataset.data_generation.generate_historical_data`).
    truth : DGPTruth
        Used by `generate_scenarios` to draw the shared context bundle.
    params : Params
        Problem parameters; provides cost coefficients, eligibility, etc.
        `params.num_scenarios_saa` is the default N if `n_scenarios` not set.
    num_food_types : int
        T.
    cells : list of (name, predictor_kind, noise_kind), optional
        Subset of cells to run. Defaults to all six (DEFAULT_CELLS).
    n_scenarios : int, optional
        Number of SAA scenarios. Defaults to params.num_scenarios_saa.
    nonparametric_kind : {'rf', 'gbm', 'knn', 'dt'}
        Hypothesis class for the NonP cells.
    spo_max_train_samples, spo_time_limit, spo_warm_start_ols
        Forwarded to `fit_demand_model_spo`.
    saa_time_limit : float, optional
        Override params.time_limit for the SAA solves only.
    scenario_seed, residual_seed : int
        Seeds for context sampling and bootstrap draws. Held fixed across
        cells so all cells see the same realizations.
    n_oos_scenarios : int, optional
        If set and > 0, also evaluate each cell's (y*, w*) — the SAA
        first-stage decision — against `n_oos_scenarios` realized-demand
        scenarios drawn from the TRUE DGP (`generate_scenarios` +
        method='oracle'). This is the proper out-of-sample metric: it
        measures the realized cost of deploying the cell's decision to
        the world, instead of grading the cell against its own predicted
        scenario distribution. The same OOS scenario set is used for all
        cells (one draw with `oos_seed`), so the comparison is fair.
        Default None = skip OOS.
    oos_seed : int
        RNG seed for the OOS scenario draw. Held fixed across cells.
    verbose : bool
        If True, print a per-cell summary table at the end.

    Returns
    -------
    results : dict with keys:
        'cells'      : list of per-cell result dicts
        'predictors' : dict of fit-time diagnostics (one entry per
                       predictor that was actually trained)
        'scenarios'  : dict with the shared scenario bundle metadata
    """
    if cells is None:
        cells = DEFAULT_CELLS
    if n_scenarios is None:
        n_scenarios = params.num_scenarios_saa

    needed_predictors = set(p for _, p, _ in cells)
    needed_residuals  = set(p for _, p, n in cells if n == "er")

    # ----- Step 1: train predictors -----------------------------------------
    predictors = {}
    pred_info  = {}

    if "ols" in needed_predictors:
        if verbose: print("[1/3] Fitting OLS ...", flush=True)
        t0 = time.perf_counter()
        predictors["ols"] = fit_demand_model(
            df, num_food_types=num_food_types, pool_across_food_types=True,
        )
        pred_info["ols"] = {"runtime": time.perf_counter() - t0}

    if "rf" in needed_predictors:
        if verbose: print(f"[2/3] Fitting nonparametric ({nonparametric_kind}) ...", flush=True)
        t0 = time.perf_counter()
        predictors["rf"] = fit_demand_model_nonparametric(
            df, num_food_types=num_food_types, kind=nonparametric_kind,
        )
        pred_info["rf"] = {"runtime": time.perf_counter() - t0,
                           "kind": nonparametric_kind}

    if "spo" in needed_predictors:
        if verbose: print("[3/3] Fitting pseudo-DFL (SPO MILP) ...", flush=True)
        spo_model, spo_diag = fit_demand_model_spo(
            df, params, num_food_types=num_food_types,
            max_train_samples=spo_max_train_samples,
            time_limit=spo_time_limit,
            warm_start_ols=spo_warm_start_ols,
            verbose=False,
        )
        predictors["spo"] = spo_model
        pred_info["spo"] = spo_diag
        if verbose:
            print(f"      SPO MILP: obj={spo_diag['obj']:.2f} | "
                  f"gap={spo_diag['gap']*100:.3f}% | "
                  f"time={spo_diag['runtime']:.1f}s | "
                  f"vars={spo_diag['num_vars']}, constrs={spo_diag['num_constrs']}")

    # ----- Step 2: empirical residuals (only for ER cells) -------------------
    residuals = {}
    for kind in needed_residuals:
        residuals[kind] = compute_training_residuals(
            predictors[kind], df, num_food_types=num_food_types,
        )

    # ----- Step 3: shared scenario context bundle (in-sample) ---------------
    scen = generate_scenarios(
        truth,
        N              = n_scenarios,
        num_food_types = num_food_types,
        demand_min     = params.demand_min,
        demand_max     = params.demand_max,
        seed           = scenario_seed,
    )
    if verbose:
        print(f"\nShared scenario bundle: N={n_scenarios}, "
              f"context source = generate_scenarios(seed={scenario_seed})")

    # ----- Step 3b: out-of-sample scenarios from the TRUE DGP ---------------
    # Drawn ONCE with a fixed seed so all cells are evaluated on the same
    # ground-truth realizations. Skipped entirely if n_oos_scenarios is None.
    if n_oos_scenarios is not None and n_oos_scenarios > 0:
        oos_scen = generate_scenarios(
            truth,
            N              = n_oos_scenarios,
            num_food_types = num_food_types,
            demand_min     = params.demand_min,
            demand_max     = params.demand_max,
            seed           = oos_seed,
        )
        # method='oracle' returns d_real verbatim (no clipping shift).
        oos_demand = scenarios_for_method(
            oos_scen, method="oracle",
            demand_min=params.demand_min, demand_max=params.demand_max,
        )                                              # (n_oos, I, T)
        if verbose:
            print(f"OOS evaluation:         N_oos={n_oos_scenarios}, "
                  f"true demand source = generate_scenarios(seed={oos_seed})")
    else:
        oos_demand = None
    if verbose:
        print()

    # ----- Step 4 & 5: per-cell project + solve + record --------------------
    cell_results = []
    width = max(len(name) for name, _, _ in cells)

    if verbose:
        oos_cols = "  " + f"{'OOS cost':>13}  {'OOS unmet':>12}" if oos_demand is not None else ""
        header = (f"  {'cell':<{width}}  {'SAA obj':>13}  "
                  f"{'first-stage':>12}  {'avg unmet':>12}  "
                  f"{'avg unmet $':>12}  {'time (s)':>9}{oos_cols}")
        print(header)
        print("  " + "-" * (len(header) - 2))

    for cell_name, predictor_kind, noise_kind in cells:
        model = predictors[predictor_kind]

        # Project the shared context bundle through this cell's noise model.
        if noise_kind == "gauss":
            scenarios = scenarios_for_method(
                scen,
                method        = "contextual_residual",
                model         = model,
                demand_min    = params.demand_min,
                demand_max    = params.demand_max,
                residual_seed = residual_seed,
            )
        elif noise_kind == "er":
            scenarios = scenarios_for_method(
                scen,
                method             = "contextual_er",
                model              = model,
                training_residuals = residuals[predictor_kind],
                demand_min         = params.demand_min,
                demand_max         = params.demand_max,
                residual_seed      = residual_seed,
            )
        else:
            raise ValueError(f"Unknown noise_kind '{noise_kind}'")

        # Solve SAA.
        time_limit_used = params.time_limit
        if saa_time_limit is not None:
            old_tl = params.time_limit
            params.time_limit = saa_time_limit
        try:
            t0 = time.perf_counter()
            obj, y_val, w_val, s_val, u_val, runtime = saa(
                scenarios, params, verbose=False,
            )
            wall = time.perf_counter() - t0
        finally:
            if saa_time_limit is not None:
                params.time_limit = old_tl

        if obj is None or u_val is None:
            cell_results.append(dict(
                cell=cell_name, predictor=predictor_kind, noise=noise_kind,
                obj=float("nan"), first_stage=float("nan"),
                avg_unmet=float("nan"), avg_unmet_cost=float("nan"),
                oos_cost=float("nan"), oos_unmet=float("nan"),
                oos_unmet_cost=float("nan"), oos_recourse_cost=float("nan"),
                runtime=runtime, status="failed",
            ))
            if verbose:
                print(f"  {cell_name:<{width}}  {'FAILED':>13}")
            continue

        # First-stage cost: c^f * sum(y) + c^a · w.
        first_stage = (
            float(params.fixed_cost) * float(np.asarray(y_val).sum())
            + float((np.asarray(w_val) * params.acquisition_cost[None, :]).sum())
        )

        # Average per-scenario unmet demand and average per-scenario unmet
        # penalty cost.
        u_arr = np.asarray(u_val)               # (N, I, T)
        N = u_arr.shape[0]
        avg_unmet      = float(u_arr.sum() / N)                          # units
        avg_unmet_cost = float((u_arr * params.unmet_penalty[None, None, :]).sum() / N)

        # ----- Out-of-sample evaluation against TRUE demand ----------------
        # Take the in-sample SAA decision (y*, w*) and replay the recourse
        # against each OOS realization of true demand. This is the proper
        # deployment-cost metric: it asks "if we deployed this cell's
        # decision in the real world, what would we actually pay?"
        if oos_demand is not None:
            n_oos = oos_demand.shape[0]
            oos_q_vals       = np.full(n_oos, np.nan)
            oos_unmet_units  = np.full(n_oos, np.nan)
            oos_unmet_dollar = np.full(n_oos, np.nan)
            for n in range(n_oos):
                q_n, _, u_n, _ = recourse_problem(
                    y_val, w_val, oos_demand[n], params, time_used=0.0,
                )
                if q_n is None or u_n is None:
                    continue
                oos_q_vals[n]       = q_n
                u_n_arr             = np.asarray(u_n)
                oos_unmet_units[n]  = u_n_arr.sum()
                oos_unmet_dollar[n] = (u_n_arr * params.unmet_penalty[None, :]).sum()
            valid = ~np.isnan(oos_q_vals)
            if valid.any():
                oos_recourse_cost = float(np.nanmean(oos_q_vals))
                oos_cost          = first_stage + oos_recourse_cost
                oos_unmet         = float(np.nanmean(oos_unmet_units))
                oos_unmet_cost    = float(np.nanmean(oos_unmet_dollar))
            else:
                oos_recourse_cost = oos_cost = oos_unmet = oos_unmet_cost = float("nan")
        else:
            oos_recourse_cost = oos_cost = oos_unmet = oos_unmet_cost = float("nan")

        cell_results.append(dict(
            cell              = cell_name,
            predictor         = predictor_kind,
            noise             = noise_kind,
            obj               = float(obj),
            first_stage       = first_stage,
            avg_unmet         = avg_unmet,
            avg_unmet_cost    = avg_unmet_cost,
            oos_cost          = oos_cost,
            oos_recourse_cost = oos_recourse_cost,
            oos_unmet         = oos_unmet,
            oos_unmet_cost    = oos_unmet_cost,
            runtime           = float(runtime),
            wall              = wall,
            status            = "ok",
        ))

        if verbose:
            base = (f"  {cell_name:<{width}}  {obj:>13.4f}  "
                    f"{first_stage:>12.4f}  {avg_unmet:>12.4f}  "
                    f"{avg_unmet_cost:>12.4f}  {runtime:>9.2f}")
            if oos_demand is not None:
                base += f"  {oos_cost:>13.4f}  {oos_unmet:>12.4f}"
            print(base)

    # ----- Relative comparison vs baseline (Linear-Gauss if present) --------
    if verbose:
        baseline = next(
            (r for r in cell_results if r["cell"] == "Linear-Gauss" and r["status"] == "ok"),
            None,
        )
        if baseline is not None and len(cell_results) > 1:
            print("\n  Relative SAA obj vs Linear-Gauss baseline (in-sample):")
            for r in cell_results:
                if r["status"] != "ok":
                    continue
                rel = (r["obj"] - baseline["obj"]) / abs(baseline["obj"])
                tag = "(baseline)" if r is baseline else ""
                print(f"    {r['cell']:<{width}}  {rel*100:+7.3f}%   {tag}")

            if oos_demand is not None and not np.isnan(baseline.get("oos_cost", float("nan"))):
                print("\n  Relative OOS realized cost vs Linear-Gauss baseline:")
                for r in cell_results:
                    if r["status"] != "ok" or np.isnan(r.get("oos_cost", float("nan"))):
                        continue
                    rel = (r["oos_cost"] - baseline["oos_cost"]) / abs(baseline["oos_cost"])
                    tag = "(baseline)" if r is baseline else ""
                    print(f"    {r['cell']:<{width}}  {rel*100:+7.3f}%   {tag}")

    return {
        "cells":      cell_results,
        "predictors": pred_info,
        "scenarios":  dict(N=n_scenarios, seed=scenario_seed,
                           residual_seed=residual_seed),
        "oos":        dict(N=n_oos_scenarios, seed=oos_seed) if oos_demand is not None else None,
    }


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from dataset.data_generation import generate_historical_data

    # --- Instance --------------------------------------------------------
    seed = 27
    I, J, T = 3, 6, 3
    np.random.seed(seed)
    pop_size = np.array([1, 2, 3])

    params = Params(
        num_areas         = I,
        num_locations     = J,
        num_food_types    = T,
        capacity          = 200.0,
        fixed_cost        = 50.0,
        demand_min        = 0.0,
        demand_max        = 600.0,
        acquisition_cost  = np.array([2.0, 2.0, 2.0]),
        unmet_penalty     = np.array([20.0, 20.0, 20.0]),
        revenue           = np.array([0.0, 0.0, 0.0]),
        num_scenarios_saa = 200,
        time_limit        = 120.0,
        seed              = seed,
    )

    # --- Heterogeneous DGP (so the cells differentiate meaningfully) ----
    df, truth = generate_historical_data(
        num_areas          = I,
        num_food_types     = T,
        num_days           = 90,
        pop_size           = pop_size,
        seed               = seed,
        surge_factor       = 3.0,
        heteroscedastic    = True,
        noise_scale_factor = 0.1,
    )

    print("=" * 78)
    print(f"In-sample evaluation: I={I}, J={J}, T={T}, num_days=90, "
          f"N_SAA={params.num_scenarios_saa}")
    print(f"DGP: surge_factor=3.0, heteroscedastic=True, noise_scale_factor=0.1")
    print("=" * 78, "\n")

    results = run_in_sample_evaluation(
        df, truth, params, num_food_types=T,
        nonparametric_kind   = "rf",
        spo_max_train_samples= None,
        spo_time_limit       = 120.0,
        verbose              = True,
    )

    print("\n=== Done. ===")
