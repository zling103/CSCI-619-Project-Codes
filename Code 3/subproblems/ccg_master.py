import numpy as np
import gurobipy as gp
from helpers.functions import cut_in_list, scenario_Q_lower_bound


def master_problem(params, S_hat, scenarios, time_used=0.0):
    """
    Build and solve the CCG master problem (MP) at the current iteration.

    The MP is a valid relaxation of the full SAA:

        MP: min  f(y, w) + (1/N) * sum_{i=1..N} eta_i
            s.t. eta_i >= Q(y, w; d^i)   for i in S_hat           (explicit cut)
                 eta_i >= LB_i           for i NOT in S_hat       (valid lower bound)
                 first-stage feasibility

    For scenarios in S_hat, we add full second-stage variables (s, u) and a
    linking cut eta_idx >= <recourse objective under (y, w, d^idx)>.
    For scenarios NOT in S_hat, we add a free eta variable with lower bound
    LB_i = scenario_Q_lower_bound(d^i), which is a valid global lower bound
    on Q(y, w; d^i) for any feasible (y, w). This is the piece that makes MP
    a valid relaxation when Q can be negative (as it is in the food-truck
    problem where revenue can exceed cost).

    Args:
        params: Params object.
        S_hat: list of (I, T) arrays, the scenarios currently in the MP.
        scenarios: full (N, I, T) scenario set.
        time_used: elapsed time across previous CCG iterations (seconds).

    Returns:
        obj, y_val, w_val, eta_val, runtime
    """
    c_f = params.fixed_cost
    c_a = params.acquisition_cost
    c_u = params.unmet_penalty
    c_r = params.revenue
    C   = params.capacity
    I   = params.num_areas
    J   = params.num_locations
    T   = params.num_food_types
    N   = params.num_scenarios_saa
    e   = params.e

    scenarios = np.asarray(scenarios, dtype=float)
    if scenarios.shape[0] != N:
        raise ValueError(
            f"scenarios has {scenarios.shape[0]} rows but params.num_scenarios_saa = {N}"
        )

    K = len(S_hat)

    mp = gp.Model("CCG_Master")
    mp.setParam("OutputFlag", 0)
    mp.setParam("TimeLimit", max(params.time_limit - time_used, 0))
    mp.setParam("Threads", 8)
    # mp.setParam("DualReductions", 0)
    # mp.setParam("InfUnbdInfo", 1)

    # first-stage variables
    y = mp.addVars(J, vtype=gp.GRB.BINARY, name="y")
    w = mp.addVars(J, T, lb=0.0, vtype=gp.GRB.CONTINUOUS, name="w")
    for j in range(J):
        mp.addConstr(
            gp.quicksum(w[j, t] for t in range(T)) <= C * y[j],
            name=f"capacity_link_{j}",
        )

    # etas + second-stage columns for scenarios IN S_hat
    if K > 0:
        eta = mp.addVars(K, lb=-gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name="eta")
        s = mp.addVars(K, I, J, T, lb=0.0, vtype=gp.GRB.CONTINUOUS, name="s")
        u = mp.addVars(K, I, T,    lb=0.0, vtype=gp.GRB.CONTINUOUS, name="u")

        for idx in range(K):
            d = S_hat[idx]
            mp.addConstr(
                eta[idx]
                >= gp.quicksum(c_u[t] * u[idx, i, t]
                               for i in range(I) for t in range(T))
                -  gp.quicksum(c_r[t] * s[idx, i, j, t]
                               for i in range(I)
                               for j in range(J)
                               for t in range(T)),
                name=f"ccg_link_{idx}",
            )
            for i in range(I):
                for t in range(T):
                    mp.addConstr(
                        gp.quicksum(s[idx, i, j, t] for j in range(J))
                        + u[idx, i, t] == d[i, t],
                        name=f"demand_balance_{idx}_{i}_{t}",
                    )
            for i in range(I):
                for j in range(J):
                    for t in range(T):
                        mp.addConstr(
                            s[idx, i, j, t] <= e[i, j] * w[j, t],
                            name=f"service_limit_{idx}_{i}_{j}_{t}",
                        )
    else:
        eta = None

    # free etas for scenarios NOT in S_hat: bounded below by a valid LB
    missing_idx = [n for n in range(N) if not cut_in_list(scenarios[n], S_hat)]
    eta_free = {}
    for n in missing_idx:
        # lb_n = scenario_Q_lower_bound(scenarios[n], params)
        lb_n = 0
        eta_free[n] = mp.addVar(lb=lb_n, vtype=gp.GRB.CONTINUOUS, name=f"eta_free_{n}")

    # objective: ALL N etas averaged, not just the K in S_hat
    obj = (
        gp.quicksum(c_f * y[j] for j in range(J))
        + gp.quicksum(c_a[t] * w[j, t] for j in range(J) for t in range(T))
    )
    if K > 0:
        obj += (1.0 / N) * gp.quicksum(eta[idx] for idx in range(K))
    if missing_idx:
        obj += (1.0 / N) * gp.quicksum(eta_free[n] for n in missing_idx)
    mp.setObjective(obj, gp.GRB.MINIMIZE)

    mp.optimize()

    if mp.status == gp.GRB.OPTIMAL or (
        mp.status == gp.GRB.TIME_LIMIT and mp.SolCount > 0
    ):
        y_val = np.array([y[j].X for j in range(J)])
        w_val = np.array([[w[j, t].X for t in range(T)] for j in range(J)])
        eta_val = (
            np.array([eta[idx].X for idx in range(K)])  # type: ignore
            if K > 0 else np.zeros(0)
        )
        return mp.ObjVal, y_val, w_val, eta_val, mp.Runtime

    elif mp.status == gp.GRB.INFEASIBLE:
        print("Master problem is infeasible.")
        mp.computeIIS()
        mp.write("gurobi_log/infeasible_MP.ilp")
        return 0.0, None, None, None, mp.Runtime

    return 0.0, None, None, None, mp.Runtime
