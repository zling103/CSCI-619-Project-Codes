import numpy as np
import gurobipy as gp


def saa(scenarios, params, time_used=0.0, verbose=False):
    """
    Sample-average approximation (SAA) of the food-truck C-2SSP.

    Given a sample S = {d^1, ..., d^N} with d^n of shape (I, T), we solve the
    deterministic MILP

        min_{y, w, s, u}
            sum_j c^f y_j
            + sum_{j, t} c^a_t w_{j, t}
            + (1 / N) sum_n sum_{i, t} c^u_t u^n_{i, t}
            - (1 / N) sum_n sum_{i, j, t} c^r_t s^n_{i, j, t}

        s.t.
            sum_t w_{j, t} <= C y_j                                 for j in J
            sum_j s^n_{i, j, t} + u^n_{i, t} = d^n_{i, t}           for n, i, t
            s^n_{i, j, t} <= e_{i, j} w_{j, t}                       for n, i, j, t
            y_j in {0, 1}; w_{j, t}, s^n_{i, j, t}, u^n_{i, t} >= 0

    Args:
        scenarios: array-like with shape (N, I, T); the sampled demand realizations
        params: Params object
        time_used: seconds already spent (deducted from Gurobi time limit)
        verbose: if True, let Gurobi print its log

    Returns:
        obj_val, y_val, w_val, s_val, u_val
        - y_val: (J,) array of first-stage binary decisions
        - w_val: (J, T) array of first-stage prepositioning decisions
        - s_val: (N, I, J, T) array of per-scenario served amounts
        - u_val: (N, I, T)    array of per-scenario unmet demand
        All return None if no feasible solution was found.
    """

    scenarios = np.asarray(scenarios, dtype=float)
    if scenarios.ndim != 3:
        raise ValueError("scenarios must have shape (N, I, T)")

    N, I, T = scenarios.shape
    if I != params.num_areas or T != params.num_food_types:
        raise ValueError("scenario shape does not match params.num_areas / params.num_food_types")

    J = params.num_locations
    C = params.capacity
    c_f = params.fixed_cost
    c_a = params.acquisition_cost
    c_u = params.unmet_penalty
    c_r = params.revenue
    e = params.e

    model = gp.Model("SAA")
    model.setParam("OutputFlag", 0)
    model.setParam("Threads", 8)
    model.setParam("TimeLimit", max(1e-3, params.time_limit - time_used))

    y = model.addVars(J, vtype=gp.GRB.BINARY, name="y")
    w = model.addVars(J, T, lb=0.0, vtype=gp.GRB.CONTINUOUS, name="w")
    s = model.addVars(N, I, J, T, lb=0.0, vtype=gp.GRB.CONTINUOUS, name="s")
    u = model.addVars(N, I, T, lb=0.0, vtype=gp.GRB.CONTINUOUS, name="u")

    model.setObjective(
        gp.quicksum(c_f * y[j] for j in range(J))
        + gp.quicksum(c_a[t] * w[j, t] for j in range(J) for t in range(T))
        + (1.0 / N) * gp.quicksum(
            c_u[t] * u[n, i, t]
            for n in range(N) for i in range(I) for t in range(T)
        )
        - (1.0 / N) * gp.quicksum(
            c_r[t] * s[n, i, j, t]
            for n in range(N) for i in range(I) for j in range(J) for t in range(T)
        ),
        gp.GRB.MINIMIZE,
    )

    # First-stage capacity linking
    for j in range(J):
        model.addConstr(
            gp.quicksum(w[j, t] for t in range(T)) <= C * y[j],
            name=f"truck_capacity_{j}",
        )

    # Per-scenario second-stage constraints
    for n in range(N):
        for i in range(I):
            for t in range(T):
                model.addConstr(
                    gp.quicksum(s[n, i, j, t] for j in range(J)) + u[n, i, t]
                    == scenarios[n, i, t],
                    name=f"demand_{n}_{i}_{t}",
                )
                for j in range(J):
                    model.addConstr(
                        s[n, i, j, t] <= float(e[i, j]) * w[j, t],
                        name=f"serve_cap_{n}_{i}_{j}_{t}",
                    )

    model.optimize()

    if model.status == gp.GRB.OPTIMAL or model.SolCount > 0:
        y_val = np.array([y[j].X for j in range(J)])
        w_val = np.array([[w[j, t].X for t in range(T)] for j in range(J)])
        s_val = np.array([
            [[[s[n, i, j, t].X for t in range(T)] for j in range(J)] for i in range(I)]
            for n in range(N)
        ])
        u_val = np.array([
            [[u[n, i, t].X for t in range(T)] for i in range(I)]
            for n in range(N)
        ])
        return model.ObjVal, y_val, w_val, s_val, u_val, model.Runtime

    print("SAA: no feasible solution found.")
    return None, None, None, None, None, model.Runtime
