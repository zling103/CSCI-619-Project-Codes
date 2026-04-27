# Neural-Enhanced Contextual Two-Stage Stochastic Programming

Code accompanying the ISE 619 final project *"Neural-Enhanced Contextual
Two-Stage Stochastic Programming"* (Ling & Ma, 2026). The repository
implements the food-truck C-2SSP instance from the report, its
sample-average approximation (SAA) formulation, and a
column-and-constraint generation (CCG) algorithm for decomposing the SAA.

## Problem setting

A decision maker decides where to open food trucks and how much of each
food type to preposition, before uncertain demand is observed. Each
candidate location `j ∈ J` is pre-assigned to exactly one **demand
region** `i ∈ I` via a binary eligibility matrix `e_{i,j}` satisfying
`Σ_i e_{i,j} = 1`.

* `I`   — set of demand regions
* `J`   — set of potential food-truck locations (`|J| ≥ |I|`)
* `T`   — set of food types
* `C`   — capacity of one food truck
* `c^f` — fixed cost of opening a food truck
* `c^a`, `c^u`, `c^r` — per-unit acquisition cost, unmet-demand penalty, and served-demand revenue (one value per food type)
* `e_{i,j}` — `1` iff a truck at location `j` serves demand region `i`

First-stage decisions: `y_j ∈ {0, 1}` (open a truck at `j`) and
`w_{j,t} ≥ 0` (amount of food type `t` prepositioned at `j`).
Second-stage decisions after demand `d_{i,t}` is revealed:
`s_{i,j,t}` (food type `t` served from location `j` to region `i`) and
`u_{i,t}` (unmet demand for food type `t` in region `i`).

The two-stage program:

```
min  f(y, w) + E[ Q(y, w; d(x)) | X = x ]
 y,w
 s.t.  Σ_t w_{j,t} ≤ C y_j      ∀ j ∈ J
       y_j ∈ {0,1},  w_{j,t} ≥ 0
```

with `f(y, w) = Σ_j c^f y_j + Σ_{j,t} c^a_t w_{j,t}` and

```
Q(y, w; d) = min   Σ_{i,t} c^u_t u_{i,t}  −  Σ_{i,j,t} c^r_t s_{i,j,t}
              s,u
              s.t.  Σ_{j ∈ J} s_{i,j,t} + u_{i,t} = d_{i,t}   ∀ i, t
                    s_{i,j,t} ≤ e_{i,j} w_{j,t}              ∀ i, j, t
                    s, u ≥ 0.
```

## Directory layout

```
Code/
├── helpers/
│   ├── parameter.py        # Params dataclass (sets, costs, e, solver options)
│   ├── functions.py        # scenario generation, first-stage cost, Q LB, cut_in_list
│   └── solver.py           # convenience front-end (saa / ccg / n-ccg dispatch)
├── subproblems/
│   ├── recourse_problem.py # Q(y, w; d) as a Gurobi LP
│   ├── saa.py              # sample-average-approximation MILP
│   ├── ccg_master.py       # CCG master problem
│   └── ccg.py              # CCG outer loop
├── outputs/                # reserved for run artifacts
├── gurobi_log/             # infeasibility IIS dumps land here
└── README.md
```

## Requirements

* Python ≥ 3.10
* `numpy`
* `gurobipy` (Gurobi 10+ with a valid license)

Install via

```bash
pip install numpy gurobipy
```

Run scripts from the `Code/` directory so that `helpers/` and
`subproblems/` resolve as top-level packages.

## Quick start

```python
from helpers.parameter import Params
from helpers.functions import generate_demand_scenarios
from subproblems.saa import saa
from subproblems.ccg import ccg

params = Params(
    num_areas=3,          # |I|
    num_locations=6,      # |J|, must be ≥ num_areas
    num_food_types=3,     # |T|
    num_scenarios_saa=20, # N
    capacity=100.0,
    fixed_cost=50.0,
    demand_max=50.0,
    time_limit=120.0,
    seed=0,
)

# Default e partitions J evenly across I:
#   I=3, J=6  ->  J(1)={0,1}, J(2)={2,3}, J(3)={4,5}
print(params.e)

scenarios = generate_demand_scenarios(params, N=params.num_scenarios_saa, seed=1)
# shape (N, I, T)

# Solve directly
obj_saa, y_saa, w_saa, *_ = saa(scenarios, params)

# Solve with CCG
obj_ccg, y_ccg, w_ccg, info = ccg(scenarios, params, tolerance=1e-4, print_level=1)

print(f"SAA = {obj_saa:.4f}")
print(f"CCG = {obj_ccg:.4f}  (iters={info['iterations']}, time={info['time']:.1f}s)")
```

At convergence the two objectives should agree up to the tolerance.

## API reference

### `Params` (`helpers/parameter.py`)
Dataclass holding problem sizes, costs, eligibility, demand support,
and solver options. Cost vectors default to random values drawn with
`seed`. The eligibility matrix `e` has shape `(I, J)`; if not provided,
it is built by partitioning the locations into contiguous blocks of
roughly equal size across the demand regions.

### `generate_demand_scenarios(params, N, seed=None)`
Returns an `(N, I, T)` array of i.i.d. uniform demand scenarios on
`[demand_min, demand_max]`.

### `recourse_problem(y, w, d, params, time_used=0.0)`
Solves `Q(y, w; d)` as a Gurobi LP. `d` has shape `(I, T)`. Returns
`(obj, s, u, runtime)` with `s` shape `(I, J, T)` and `u` shape `(I, T)`.

### `saa(scenarios, params, time_used=0.0, verbose=False)`
Solves the full SAA MILP in one shot. `scenarios` has shape `(N, I, T)`.
Returns `(obj, y, w, s, u)` with `s` shape `(N, I, J, T)` and `u` shape
`(N, I, T)`.

### `master_problem(params, S_hat, time_used=0.0)` (`subproblems/ccg_master.py`)
Builds and solves the CCG master at the current iteration. `S_hat` is a
list of demand arrays (each `(I, T)`) already added. Returns
`(obj, y, w, eta, runtime)` where `eta` has length `|S_hat|`.

### `ccg(scenarios, params, tolerance=1e-4, print_level=1)`
CCG outer loop. At each iteration it solves the master to get
`(y^k, w^k)`, evaluates the true `Q(y^k, w^k; d^i)` for every sampled
scenario to update the UB, adds the highest-cost scenario not yet in
`S_hat`, and terminates when the relative gap `(UB - LB)/UB` falls below
`tolerance` or the time limit is hit. Returns `(obj, y, w, info)` with
`info` containing `iterations`, `LB`, `UB`, `gap`, `S_hat`, `time`, and
`cvg_flag` (`0` not converged, `1` converged, `2` time limit).
