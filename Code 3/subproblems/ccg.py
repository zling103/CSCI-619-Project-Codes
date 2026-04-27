import time
import numpy as np
import gurobipy as gp
from subproblems.ccg_master import master_problem as master
from helpers.functions import *
from subproblems.recourse_problem import recourse_problem

CVG_NONE       = 0
CVG_OPTIMAL    = 1  
CVG_TIME_LIMIT = 2
CVG_EXHAUSTED  = 3

def ccg(
        scenarios,
        params,
        tolerance = 1e-2,
        print_level = 1
):
    """
    Column-and-constraint generation (CCG) for the SAA of the food-truck C-2SSP.

    Follows Algorithm 1 in the project report:

        1. Initialize S_hat = {}, LB = -inf, UB = +inf.
        2. repeat:
             a. solve MP -> (y^k, w^k), phi^k; set LB <- phi^k.
             b. solve SP Q(y^k, w^k; d^i) for all i; compute
                UB^k = f(y^k, w^k) + (1 / N) sum_i Q^i; update UB.
             c. pick most-violating scenarios i not in S_hat and add them.
        3. until UB - LB <= tolerance.

    Args:
        scenarios: (N, I, T) array of demand realizations over demand regions
        params: Params object
        tolerance: relative optimality gap (UB - LB)/UB at which to stop
        print_level: 0 to suppress per-iteration output, 1 to print it

    Returns:
        obj_val, y_val, w_val, info
        - obj_val: best upper bound found (exact optimum of SAA at convergence)
        - y_val, w_val: incumbent first-stage solution achieving obj_val
        - info: dict with 'iterations', 'LB', 'UB', 'gap', 'S_hat', 'time',
                'cvg_flag'
    """

    scenarios = np.asarray(scenarios, dtype=float)
    if scenarios.ndim != 3:
        raise ValueError("scenarios must have shape (N, I, T)")

    N, I, T = scenarios.shape
    if I != params.num_areas or T != params.num_food_types:
        raise ValueError("scenario shape does not match params.num_areas / params.num_food_types")

    S_hat = []
    LB, UB = -np.inf, np.inf
    best_y, best_w = None, None
    iter = 0
    time_used = 0
    converged = False
    cvg_flag = 0

    while not converged:

        iter_time = 0

        # ---- Master problem ----
        LB, y_val, w_val, eta_val, mp_time = master(params, S_hat, scenarios, time_used)
        time_used += mp_time
        iter_time += mp_time

        # ---- Subproblems: evaluate true recourse at (y_val, w_val) ----
        Q_vals = np.zeros(N)
        for i in range(N):
            d = scenarios[i]
            q_val, _, _, sp_time = recourse_problem(y_val, w_val, d, params, time_used=time_used)
            time_used += sp_time
            iter_time += sp_time
            Q_vals[i] = q_val if q_val is not None else 0

        # ---- Upper bound ----
        UB_k = first_stage_cost(y_val, w_val, params) + float(np.mean(Q_vals))
        if UB_k < UB:
            UB = UB_k
            best_y, best_w = y_val, w_val


        if LB != 0:
            gap = (UB - LB) / np.abs(UB)
        else:
            gap = 1

        # ---- Select the most-violating scenario(s) ----
        # Selection: only consider scenarios NOT yet in S_hat
        remaining = [i for i in range(N) if not cut_in_list(scenarios[i], S_hat)]
        if remaining:
            i_star = max(remaining, key=lambda i: Q_vals[i])
            # i_star_prime = min(remaining, key=lambda i: Q_vals[i])
            S_hat.append(scenarios[i_star])
            # S_hat.append(scenarios[i_star_prime])
        else:
            # Exhausted — MP equivalent to full SAA. Gap should be ~0;
            # if it isn't, it's numerical residue. Report with a distinct flag.
            converged = True
            cvg_flag = 3   # "exhausted"
        # ---- Check Convergence Criteria ----
        iter += 1

        if print_level == 1 and iter % 1 == 0:
            print(f"iter {iter:>5d} | LB = {LB:>10.3f} | UB = {UB:>10.3f} | gap = {gap * 100:>8.2f}% | |S_hat| = {len(S_hat):>5d} | iter_time = {iter_time:>5.2f} s")

        print("True" if gap <= tolerance else "Not", "converged at iter", iter, "with gap", gap)

        if gap <= tolerance:
            converged = True
            cvg_flag = 1
        elif time_used >= params.time_limit:
            if print_level == 1:
                print("CCG: time limit reached.")
            converged = True
            cvg_flag = 2

    info = {
        'iterations': iter,
        'LB': LB,
        'UB': UB,
        'gap': gap, 
        'S_hat': S_hat,
        'time': time_used,
        'cvg_flag': cvg_flag # 0 - not converged; 1 - converged; 2 - Time Limit Reached
    }

    return UB, best_y, best_w, info
