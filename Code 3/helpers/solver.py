import time
from typing import Optional, Union
import numpy as np
from helpers.parameter import Params
from helpers.functions import first_stage_cost
from learning.q_surrogate import QSurrogate
from subproblems.ccg import ccg
from subproblems.saa import saa
from subproblems.neural_ccg import neural_ccg


def solver(
    params: Params,
    scenarios: np.ndarray,
    method: str = "ccg",
    print_level: int = 1,
    tolerance: float = 1e-4,
    surrogate: Optional[Union[str, QSurrogate]] = None,
    compute_true_ub: bool = False
) -> tuple[float, np.ndarray, np.ndarray, dict]:
    """
    Unified entry point for solving the SAA of the food-truck C-2SSP under
    different methods.

    Args:
        params: Params object that carries all problem parameters and solver
            settings (sizes I, J, T; cost vectors; capacity; eligibility;
            time limits, etc.).
        scenarios: array of shape (N, I, T); the sampled demand realizations
            that define the SAA instance. Produced by
            scenarios_for_method(...) in scenario_generator.py.
        method: which method to use; one of the following three:
            - "saa":   solve the SAA directly as a single MILP (no decomposition).
            - "ccg":   classical column-and-constraint generation (CCG).
            - "n-ccg": neural CCG that uses a trained surrogate to predict
                       the recourse value Q(y, w; d). (TBD - not yet
                       implemented; raises NotImplementedError.)
        print_level: 0 to suppress per-iteration printouts, 1 to show them.
            Passed through to the underlying method where applicable.
        tolerance: relative optimality gap at which CCG (and future n-ccg)
            terminates. Ignored by "saa".

    Returns:
        obj_val, y_val, w_val, info
        - obj_val: objective value reported by the method. For "saa" this is
          the MILP optimum; for "ccg" this is the best upper bound at
          convergence, equal to the SAA optimum up to `tolerance`.
        - y_val:   (J,)    array of first-stage open/close decisions.
        - w_val:   (J, T)  array of first-stage prepositioning decisions.
        - info:    dict with method-specific diagnostics. Always contains:
              "method"     : the method string,
              "time"       : wall-clock solve time (seconds),
              "first_stage": f(y, w) for the returned (y, w).
          Method-specific keys:
              "saa":   "s", "u" (second-stage solutions at each scenario).
              "ccg":   "iterations", "LB", "UB", "gap", "S_hat", "cvg_flag".
    """
    scenarios = np.asarray(scenarios, dtype=float)
    if scenarios.ndim != 3:
        raise ValueError(f"scenarios must have shape (N, I, T); got {scenarios.shape}")
    if scenarios.shape[1] != params.num_areas or scenarios.shape[2] != params.num_food_types:
        raise ValueError(
            f"scenario shape {scenarios.shape[1:]} does not match "
            f"(num_areas, num_food_types) = ({params.num_areas}, {params.num_food_types})"
        )

    method = method.lower()

    # ------------------------------------------------------------------ SAA
    if method == "saa":
        t0 = time.perf_counter()
        obj_val, y_val, w_val, s_val, u_val = saa(
            scenarios, params, verbose=(print_level == 1)
        )
        elapsed = time.perf_counter() - t0

        if obj_val is None:
            info = {
                "method":      "saa",
                "time":        elapsed,
                "first_stage": None,
                "s":           None,
                "u":           None,
                "status":      "infeasible_or_failed",
            }
            return float("nan"), None, None, info   # type: ignore

        info = {
            "method":      "saa",
            "time":        elapsed,
            "first_stage": first_stage_cost(y_val, w_val, params),
            "s":           s_val,
            "u":           u_val,
            "status":      "ok",
        }
        if print_level == 1:
            print(
                f"[SAA] obj = {obj_val:.4f} | first_stage = {info['first_stage']:.4f}"
                f" | time = {elapsed:.2f} s"
            )
        return obj_val, y_val, w_val, info

    # ------------------------------------------------------------------ CCG
    if method == "ccg":
        t0 = time.perf_counter()
        obj_val, y_val, w_val, ccg_info = ccg(
            scenarios, params, tolerance=tolerance, print_level=print_level
        )
        elapsed = time.perf_counter() - t0

        if y_val is None:
            info = {
                "method":      "ccg",
                "time":        elapsed,
                "first_stage": None,
                "status":      "failed",
                **ccg_info,
            }
            return float("nan"), None, None, info

        info = {
            "method":      "ccg",
            "time":        elapsed,
            "first_stage": first_stage_cost(y_val, w_val, params),
            "status":      "ok",
            **ccg_info,        # adds iterations, LB, UB, gap, S_hat, cvg_flag
        }
        if print_level == 1:
            print(
                f"[CCG] obj = {obj_val:.4f} | gap = {ccg_info['gap']*100:.4f}%"
                f" | iterations = {ccg_info['iterations']}"
                f" | first_stage = {info['first_stage']:.4f}"
                f" | time = {elapsed:.2f} s"
            )
        return obj_val, y_val, w_val, info

    # ---------------------------------------------------------- Neural CCG
    if method == "n-ccg":
        # Resolve the surrogate: accept an instance, a path, or None.
        if surrogate is None:
            surr = QSurrogate(params.surrogate_path)
        elif isinstance(surrogate, str):
            surr = QSurrogate(surrogate)
        elif isinstance(surrogate, QSurrogate):
            surr = surrogate
        else:
            raise TypeError(
                "surrogate must be a QSurrogate instance, a path string, or None"
            )
 
        t0 = time.perf_counter()
        obj_val, y_val, w_val, n_info = neural_ccg(
            scenarios, params, surr,
            tolerance=tolerance,
            print_level=print_level,
            compute_true_ub=compute_true_ub,
        )
        elapsed = time.perf_counter() - t0
 
        if y_val is None:
            info = {
                "method":      "n-ccg",
                "time":        elapsed,
                "first_stage": None,
                "status":      "failed",
                **n_info,
            }
            return float("nan"), None, None, info
 
        info = {
            "method":      "n-ccg",
            "time":        elapsed,
            "first_stage": first_stage_cost(y_val, w_val, params),
            "status":      "ok",
            **n_info,
        }
        if print_level == 1:
            iters = n_info.get("iterations", "-")
            LB = n_info.get("LB", float("nan"))
            msg = (
                f"[N-CCG] obj = {obj_val:12.4f} | LB = {LB:12.4f} "
                f"| iters = {iters} | first_stage = {info['first_stage']:10.4f}"
                f" | time = {elapsed:.3f} s"
            )
            if compute_true_ub and "gap_true" in n_info:
                msg += f" | gap_true = {n_info['gap_true']*100:.4f}%"
            print(msg)
        return obj_val, y_val, w_val, info

    raise ValueError(
        f"Unknown method '{method}'. Valid options: 'saa', 'ccg', 'n-ccg'."
    )