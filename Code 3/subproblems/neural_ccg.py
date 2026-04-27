import numpy as np

from helpers.functions import first_stage_cost
from learning.q_surrogate import QSurrogate
from subproblems.ccg_master import master_problem as master
from subproblems.recourse_problem import recourse_problem


# Convergence flags
CVG_NONE       = 0
CVG_NN         = 1   # Shao eq. (14) fired
CVG_TIME_LIMIT = 2
CVG_EXHAUSTED  = 3   # all N scenarios added, nothing left


def neural_ccg(
    scenarios,
    params,
    surrogate: QSurrogate,
    tolerance: float = 1.0,
    print_level: int = 1,
    compute_true_ub: bool = False,
):
    """
    Neural Column-and-Constraint Generation (Shao et al. 2025, Algorithm 2).

    This implementation follows Shao's paper verbatim:

        selection:    i^* = argmax_{s in S \\ S^k}  NN(z^k, xi_s)

        termination (eq. 14):
            max_{s in S}     NN(z^k, xi_s)
         <= max_{s in S^k}   NN(z^k, xi_s) + eps

    Both rules require Q(y,w;d^s) >= 0, which is the case here once the
    revenue term is removed from the recourse objective. (If you re-add a
    negative-Q regime, you'll need a violation-based analog; see git
    history for the older Q - LB form.)

    `tolerance` is the absolute eps in eq. (14), in the same units as Q.
    A useful starting value is roughly 1% of a typical Q value.

    Each iteration:
        1. Solve MP; get (y^k, w^k) and theta^k.
        2. Compute q_hat = surrogate(y^k, w^k, xi_s) for ALL s in S.
        3. Check Shao eq. (14); break if satisfied.
        4. Select i^* = argmax_{remaining} q_hat; append xi_{i^*}.

    NO LB/UB tracking inside the loop. The master's theta^k is captured
    on the final iteration and returned as the reported objective. If
    `compute_true_ub` is set, we compute the true UB via LP solves ONCE
    at the end for diagnostics.

    Args:
        scenarios:       (N, I, T) array of demand realizations.
        params:          Params object.
        surrogate:       trained QSurrogate.
        tolerance:       absolute epsilon in Shao eq. (14), in Q units.
        print_level:     0 silent, 1 per-iteration printout.
        compute_true_ub: if True, solve true recourse at the final (y*, w*)
                         to compute a true UB and gap.

    Returns:
        obj_val, y_val, w_val, info
    """
    scenarios = np.asarray(scenarios, dtype=float)
    if scenarios.ndim != 3:
        raise ValueError("scenarios must have shape (N, I, T)")

    N, I, T = scenarios.shape
    if I != params.num_areas or T != params.num_food_types:
        raise ValueError(
            "scenario shape does not match params.num_areas / params.num_food_types"
        )

    if (surrogate.num_areas, surrogate.num_locations, surrogate.num_food_types) \
            != (I, params.num_locations, T):
        raise ValueError(
            f"Surrogate was trained on "
            f"(I, J, T)=({surrogate.num_areas}, {surrogate.num_locations}, "
            f"{surrogate.num_food_types}); instance is "
            f"(I, J, T)=({I}, {params.num_locations}, {T})."
        )

    S_hat     = []     # list of (I, T) arrays, passed to master
    S_hat_idx = []     # parallel list of integer indices into `scenarios`,
                       # used for fast q_hat slicing in eq. (14)
    theta = np.nan
    y_val, w_val = None, None
    time_used = 0.0
    iter = 0
    cvg_flag = CVG_NONE
    hist = []

    while cvg_flag == CVG_NONE:
        iter_time = 0.0

        # ---- 1. Master problem ----------------------------------------
        theta, y_val, w_val, _, mp_time = master(params, S_hat, scenarios, time_used)
        time_used += mp_time
        iter_time += mp_time

        if y_val is None:
            if print_level == 1:
                print("Neural CCG: master infeasible or failed; stopping.")
            cvg_flag = CVG_TIME_LIMIT
            break

        # ---- 2. Surrogate forward pass over ALL N scenarios -----------
        q_hat = surrogate.predict(y_val, w_val, scenarios)      # (N,)

        # ---- 3. Shao eq. (14) termination, BEFORE appending -----------
        # max_{s in S}    NN(y,w; xi_s)
        #     <= max_{s in S^k}  NN(y,w; xi_s) + eps
        # Both maxima are well-defined for K >= 1; at K=0 we cannot yet
        # check the criterion (no S^k to take a max over) and just go
        # add the first cut.
        in_set        = set(S_hat_idx)
        idx_remaining = [i for i in range(N) if i not in in_set]

        max_q_all = float(q_hat.max())
        if S_hat_idx:
            max_q_in  = float(q_hat[S_hat_idx].max())
            shao_gap  = max_q_all - max_q_in            # >= 0 by construction
        else:
            max_q_in  = float("-inf")
            shao_gap  = float("inf")

        iter += 1

        record = dict(
            iter      = iter,
            theta     = theta,
            max_q_all = max_q_all,
            max_q_in  = max_q_in,
            shao_gap  = shao_gap,
            S_size    = len(S_hat),
            time      = iter_time,
        )
        hist.append(record)

        if print_level == 1:
            print(
                f"iter {iter:4d} | theta = {theta:12.4f} "
                f"| max_q_S = {max_q_all:12.4f} "
                f"| max_q_Sk = {max_q_in:12.4f} "
                f"| shao_gap = {shao_gap:10.4f} "
                f"| |S_hat| = {len(S_hat):4d} "
                f"| iter_time = {iter_time:.3f}s"
            )

        if S_hat_idx and shao_gap <= tolerance:
            cvg_flag = CVG_NN
            break
        if time_used >= params.time_limit:
            if print_level == 1:
                print("Neural CCG: time limit reached.")
            cvg_flag = CVG_TIME_LIMIT
            break

        # ---- 4. Select argmax q_hat over remaining (Shao); append -----
        if not idx_remaining:
            if print_level == 1:
                print("Neural CCG: all scenarios already in S_hat; stopping.")
            cvg_flag = CVG_EXHAUSTED
            break

        i_star = max(idx_remaining, key=lambda i: q_hat[i])
        S_hat.append(scenarios[i_star])
        S_hat_idx.append(i_star)

    # ---- Post-loop diagnostics ---------------------------------------
    info = {
        "method":     "n-ccg",
        "iterations": iter,
        "theta":      theta,
        "S_hat":      S_hat,
        "time":       time_used,
        "cvg_flag":   cvg_flag,
        "hist":       hist,
    }

    if compute_true_ub and y_val is not None:
        q_true = np.zeros(N)
        for i in range(N):
            qv, _, _, sp_time = recourse_problem(
                y_val, w_val, scenarios[i], params, time_used=time_used
            )
            time_used += sp_time
            q_true[i] = qv if qv is not None else 0.0
        UB_true = first_stage_cost(y_val, w_val, params) + float(q_true.mean())
        info["time"]    = time_used
        info["UB_true"] = UB_true
        info["gap_true"] = (
            abs(UB_true - theta) / max(abs(UB_true), abs(theta), 1.0)
            if np.isfinite(theta) else np.inf
        )
        if print_level == 1:
            print(
                f"\n  post-hoc: UB_true = {UB_true:12.4f}, "
                f"gap_true = {info['gap_true']*100:.4f}%"
            )

    return theta, y_val, w_val, info
