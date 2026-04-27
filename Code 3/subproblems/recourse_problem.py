import numpy as np
import gurobipy as gp


def recourse_problem(y, w, d, params, time_used=0.0):
    """
    Recourse problem Q(y, w; d) for the food-truck C-2SSP.

    For a fixed first-stage decision (y, w) and a realized demand d, we solve

        Q(y, w; d) = min   sum_{i, t} c^u_t u_{i, t}
                          - sum_{i, j, t} c^r_t s_{i, j, t}
                    s, u
                    s.t. sum_{j in J} s_{i, j, t} + u_{i, t} = d_{i, t}
                                                        for all i in I, t in T
                         s_{i, j, t} <= e_{i, j} w_{j, t} for all i, j, t
                         s_{i, j, t} >= 0, u_{i, t} >= 0

    Note: y does not appear in the recourse constraints. The open/close decision
    y is encoded through w (with w_{j, t} = 0 whenever y_j = 0 enforced in the
    first stage via the capacity-linking constraint), and the eligibility matrix
    e selects which region each truck's inventory supplies.

    Args:
        y: (J,) array of first-stage open/close decisions (unused; kept for
           interface symmetry with the master problem)
        w: (J, T) array of first-stage prepositioning decisions
        d: (I, T) demand realization over demand regions
        params: Params object
        time_used: seconds already spent (deducted from the Gurobi time limit)

    Returns:
        obj_val, s_val, u_val, runtime
        - s_val: (I, J, T) array of served amounts per (region, location, food type)
        - u_val: (I, T)    array of unmet demand per (region, food type)
        - runtime: Gurobi solve time for this subproblem
        Returns (0, None, None, runtime) when infeasible (IIS is dumped),
        or (None, None, None, runtime) on other failures.
    """

    I = params.num_areas
    J = params.num_locations
    T = params.num_food_types
    c_u = params.unmet_penalty
    c_r = params.revenue
    e = params.e

    w = np.asarray(w, dtype=float)
    d = np.asarray(d, dtype=float)

    model = gp.Model("Recourse")
    model.setParam("OutputFlag", 0)
    model.setParam("TimeLimit", np.max([params.time_limit - time_used, 0]))
    model.setParam("Threads", 8)
    

    s = model.addVars(I, J, T, lb=0.0, vtype=gp.GRB.CONTINUOUS, name="s")
    u = model.addVars(I, T, lb=0.0, vtype=gp.GRB.CONTINUOUS, name="u")

    model.setObjective(
        gp.quicksum(c_u[t] * u[i, t] for i in range(I) for t in range(T))
        - gp.quicksum(
            c_r[t] * s[i, j, t]
            for i in range(I) for j in range(J) for t in range(T)
        ),
        gp.GRB.MINIMIZE,
    )

    for i in range(I):
        for t in range(T):
            model.addConstr(
                gp.quicksum(s[i, j, t] for j in range(J)) + u[i, t] == d[i, t],
                name=f"demand_{i}_{t}",
            )
            for j in range(J):
                model.addConstr(
                    s[i, j, t] <= float(e[i, j]) * w[j, t],
                    name=f"serve_cap_{i}_{j}_{t}",
                )

    model.optimize()

    if model.status == gp.GRB.OPTIMAL or model.SolCount > 0:
        s_val = np.array([[[s[i, j, t].X for t in range(T)] for j in range(J)] for i in range(I)])
        u_val = np.array([[u[i, t].X for t in range(T)] for i in range(I)])
        return model.ObjVal, s_val, u_val, model.Runtime

    elif model.status == gp.GRB.INFEASIBLE:
        print("Recourse problem: no feasible solution found.")
        model.computeIIS()
        model.write("gurobi_log/infeasible_sp.ilp")
        return 0, None, None, model.Runtime

    else:
        print("Recourse problem: optimization failed.")
        return None, None, None, model.Runtime
