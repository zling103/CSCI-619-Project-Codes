import numpy as np


def generate_demand_scenarios(params, N, seed=None):
    """
    Generate N i.i.d. demand scenarios uniformly on [d_min, d_max].

    Args:
        params: Params object (uses num_areas, num_food_types, demand_min, demand_max)
        N: number of scenarios
        seed: random seed for reproducibility

    Returns:
        scenarios: np.ndarray of shape (N, I, T); scenarios[n, i, t] is the demand
                   for food type t in demand region i under scenario n
    """
    rng = np.random.default_rng(seed)
    return rng.uniform(
        params.demand_min,
        params.demand_max,
        size=(N, params.num_areas, params.num_food_types),
    )


def generate_context(params, N, seed=None):
    """
    Generate N context vectors x^i of dimension params.context_dim.

    Useful as a placeholder for the contextual feature X in C-2SSP; the
    optimization routines in this project do not require it, since they only
    see the demand scenarios d(x^i).

    Args:
        params: Params object
        N: number of scenarios
        seed: random seed

    Returns:
        X: np.ndarray of shape (N, context_dim)
    """
    rng = np.random.default_rng(seed)
    return rng.normal(size=(N, params.context_dim))


def first_stage_cost(y, w, params):
    """
    Compute f(y, w) = sum_j c^f y_j + sum_{j, t} c^a_t w_{j, t}.

    Args:
        y: (J,) array of first-stage binary decisions
        w: (J, T) array of first-stage prepositioning decisions
        params: Params object

    Returns:
        float, the first-stage cost
    """
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    fixed = params.fixed_cost * y.sum()
    acquisition = (w * params.acquisition_cost[np.newaxis, :]).sum()
    return float(fixed + acquisition)


def scenario_Q_lower_bound(d, params):
    """
    Valid per-scenario lower bound on the recourse value Q(y, w; d).

    Rewriting the recourse objective using u_{i,t} = d_{i,t} - sum_j s_{i,j,t}:

        Q = sum_{i,t} c^u_t d_{i,t}  -  sum_{i,j,t} (c^u_t + c^r_t) s_{i,j,t}

    So minimizing Q is equivalent to maximizing sum (c^u+c^r) s. A valid LB on
    Q comes from relaxing the (y, w) feasibility: let
        a_{i,t} = sum_j s_{i,j,t}.
    Every feasible (y, w, s) satisfies
        0 <= a_{i,t} <= d_{i,t}                      (u >= 0)
        sum_t a_{i,t} <= |J_i| * C                   (capacity across region i)
    where J_i = {j : e_{i,j} = 1}. The second line: trucks serving region i
    have individual capacity <= C, so their total prepositioning <= |J_i| * C,
    which bounds total service to region i across all food types.

    The LB is thus
        Q >= sum_{i,t} c^u_t d_{i,t}  -  max_a sum_{i,t} (c^u+c^r)_t a_{i,t}
    with a satisfying the two relaxed bounds. The inner max decomposes per
    region into a capacity-limited knapsack solved greedily: sort food types
    by (c^u+c^r) descending and fill a_{i,t} = d_{i,t} until the capacity
    budget |J_i|*C is exhausted.

    This LB equals the old serve-all bound (-c^r * d summed) whenever
    capacity is not binding (sum_t d_{i,t} <= |J_i|*C for every i), and is
    strictly tighter whenever some region is capacity-constrained.

    Args:
        d: (I, T) demand array for a single scenario
        params: Params object

    Returns:
        float, a valid lower bound on Q(y, w; d) for any feasible (y, w)
    """
    d = np.asarray(d, dtype=float)
    c_u = params.unmet_penalty
    c_r = params.revenue
    C   = params.capacity
    e   = params.e                                # (I, J)
    I, T = d.shape
    weights = c_u + c_r                           # (T,)
    order = np.argsort(-weights)                  # food types by weight desc

    trucks_per_region = e.sum(axis=1)             # (I,)

    lb = 0.0
    for i in range(I):
        cap_i = float(trucks_per_region[i]) * C
        served_value = 0.0
        budget = cap_i
        for t in order:
            if budget <= 0.0:
                break
            a = min(d[i, t], budget)
            served_value += weights[t] * a
            budget -= a
        lb += float((c_u * d[i]).sum()) - served_value

    return float(lb)


def cut_in_list(d, cut_list):
    for d0 in cut_list:
        if np.array_equal(d, d0):
            return True
    return False