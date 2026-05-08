# Neural-Enhanced Contextual Two-Stage Stochastic Programming

Code accompanying the ISE 619 final project *"Neural-Enhanced Contextual
Two-Stage Stochastic Programming"* (Ling & Ma, 2026). The repository
implements

1. the food-truck **C-2SSP** instance from the report,
2. its **sample-average approximation (SAA)** as a single MILP,
3. the **classical column-and-constraint generation (CCG)** algorithm for
   decomposing the SAA,
4. a **neural-CCG (n-CCG)** variant that replaces per-iteration recourse
   LPs with calls to a trained MLP Q-surrogate, following
   Shao et al. (2025), and
5. a six-cell **decision-based-learning (DBL) / decision-focused-learning
   (DFL)** ablation that crosses three demand predictors (OLS, RF/GBM,
   pseudo-DFL via a single-MILP SPO surrogate) with two noise samplers
   (Gaussian residuals, empirical-residual bootstrap).

All optimization code uses Gurobi; the surrogate uses PyTorch.

---

## 1. Mathematical formulation

### 1.1. Sets and parameters

| Symbol      | Meaning |
|-------------|---------|
| `I`         | demand regions |
| `J`         | candidate food-truck locations (`|J| ≥ |I|`) |
| `T`         | food types |
| `C`         | per-truck capacity |
| `c^f`       | fixed cost of opening a truck |
| `c^a_t`     | per-unit acquisition cost of food type `t` |
| `c^u_t`     | per-unit unmet-demand penalty for food type `t` |
| `c^r_t`     | per-unit revenue for served demand of food type `t` |
| `e_{i,j}`   | binary: `1` iff a truck at `j` serves region `i`; `Σ_i e_{i,j}=1` |
| `x`         | observable context (here: weather, weekday) |
| `d_{i,t}(x)`| stochastic demand at region `i` for food type `t`, indexed by `x` |

### 1.2. First-stage problem

```
min   f(y, w) + E[ Q(y, w; d(x)) | X = x ]
y, w
s.t.  Σ_t w_{j, t} ≤ C y_j        ∀ j ∈ J            (1b)
      y_j ∈ {0, 1},  w_{j, t} ≥ 0                    (1c)
```

with `f(y, w) = Σ_j c^f y_j + Σ_{j, t} c^a_t w_{j, t}`.

### 1.3. Recourse problem (eqs. 2a–2d)

```
Q(y, w; d) = min   Σ_{i, t} c^u_t u_{i, t}  −  Σ_{i, j, t} c^r_t s_{i, j, t}
              s, u
              s.t.  Σ_{j ∈ J} s_{i, j, t} + u_{i, t} = d_{i, t}    ∀ i, t   (2b)
                    s_{i, j, t} ≤ e_{i, j} w_{j, t}                ∀ i,j,t (2c)
                    s, u ≥ 0                                       (2d)
```

`s_{i, j, t}` is the amount of food type `t` shipped from location `j`
to region `i`; `u_{i, t}` is unmet demand for type `t` in region `i`.
Equality in (2b) prevents over-serving; the cap (2c) routes inventory
through the eligibility matrix `e`.

### 1.4. Sample-average approximation

Given `N` scenarios `S = {d^1, …, d^N}`,

```
min   f(y, w) + (1/N) Σ_n Q(y, w; d^n)        (3)
y, w
s.t.  (1b), (1c).
```

### 1.5. Classical CCG (Algorithm 1)

```
Init  k=0, S_hat=∅, LB=−∞, UB=+∞.
Repeat:
  1. Solve MP (1/N Σ η_i, η_i ≥ Q(·;d^i) for d^i ∈ S_hat) → (y^k, w^k), φ^k; LB ← φ^k.
  2. Solve SP Q(y^k, w^k; d^i) for all i ∈ {1, …, N}.
  3. UB^k = f(y^k, w^k) + (1/N) Σ_i Q^i; UB ← min(UB, UB^k).
  4. Pick the highest-cost scenario not yet in S_hat and append it.
Until (UB − LB) / |UB| ≤ ε.
```

Master-problem cleanliness: scenarios *in* `S_hat` get full second-stage
columns plus the cut `η_i ≥ Q(y, w; d^i)`; scenarios *not* in `S_hat` get
a free `η_i` bounded below by a valid `Q` lower bound, so the MP is a
genuine relaxation of the full SAA.

### 1.6. Neural CCG (Shao et al. 2025, Algorithm 2)

Replaces step 2 with a forward pass of a trained MLP `NN ≈ Q`, and uses

* selection: `i* = argmax_{s ∉ S^k} NN(y^k, w^k; d^s)`,
* termination (eq. 14): `max_s NN ≤ max_{s ∈ S^k} NN + ε`.

These rules require `Q ≥ 0`, so n-CCG is run with `c^r = 0` (revenue
disabled). With the surrogate, `N` LP solves per iteration collapse to
one batched neural-net call.

---

## 2. Repository layout

```
Code/
├── helpers/
│   ├── parameter.py        # Params dataclass (sets, costs, e, paths)
│   ├── functions.py        # demand sampling, first-stage cost, scenario LB, dedup
│   └── solver.py           # unified saa / ccg / n-ccg dispatch
├── subproblems/
│   ├── recourse_problem.py # Q(y, w; d) Gurobi LP
│   ├── saa.py              # SAA MILP
│   ├── ccg_master.py       # CCG master with full-N η + per-scenario LBs
│   ├── ccg.py              # CCG outer loop
│   └── neural_ccg.py       # Shao-style n-CCG outer loop
├── dataset/
│   ├── data_generation.py  # synthetic DGP (DGPTruth, generate_historical_data)
│   └── scenario_generator.py # ScenarioSet, DemandModel, scenarios_for_method
├── learning/
│   ├── ols_fit.py          # linear DemandModel (pooled across food types)
│   ├── erm_fit.py          # nonparametric DemandModel (RF / GBM / KNN / DT)
│   ├── spo_fit.py          # pseudo-DFL DemandModel via single-MILP SPO surrogate
│   ├── residuals.py        # training-set residual matrix for ER bootstrap
│   ├── nn_training_data.py # collect (y, w, d, Q) triples for the surrogate
│   ├── mlp_training.py     # train QSurrogateMLP (PyTorch) + save .pt blob
│   ├── q_surrogate.py      # NumPy-in/out inference wrapper
│   └── evaluate.py         # six-cell DBL/DFL in-sample + OOS harness
├── data/                   # collected training tuples and surrogate weights
├── outputs/                # figures saved by test_*.py
├── gurobi_log/             # IIS dumps for infeasibility post-mortem
├── test_in_sample.py       # six-cell in-sample comparison + figure
├── test_out_of_sample.py   # six-cell OOS-replay comparison + figures
└── README.md
```

---

## 3. Requirements

* Python ≥ 3.10
* `numpy`, `pandas`, `scikit-learn`, `matplotlib`
* `gurobipy` (Gurobi 10+ with a valid license)
* `torch` (CPU is fine; CUDA used automatically if available)

```bash
pip install numpy pandas scikit-learn matplotlib torch gurobipy
```

Run all scripts from the `Code/` directory so that `helpers/`,
`subproblems/`, `dataset/`, and `learning/` resolve as top-level packages.

---

## 4. Quick start

### 4.1. Solve a hand-built SAA instance

```python
from helpers.parameter import Params
from helpers.functions import generate_demand_scenarios
from subproblems.saa import saa
from subproblems.ccg import ccg

params = Params(
    num_areas=3, num_locations=6, num_food_types=3,
    num_scenarios_saa=20, capacity=200.0, fixed_cost=50.0,
    demand_max=300.0, time_limit=120.0, seed=0,
)
# Default e: contiguous even partition of J across I
#   I=3, J=6  ->  J(0)={0,1}, J(1)={2,3}, J(2)={4,5}
print(params.e)

scenarios = generate_demand_scenarios(params, N=params.num_scenarios_saa, seed=1)
# shape (N, I, T)

obj_saa, y_saa, w_saa, *_ = saa(scenarios, params)
obj_ccg, y_ccg, w_ccg, info = ccg(scenarios, params, tolerance=1e-2, print_level=1)

print(f"SAA = {obj_saa:.4f}")
print(f"CCG = {obj_ccg:.4f}  (iters={info['iterations']}, time={info['time']:.1f}s)")
```

### 4.2. Use the unified `solver` dispatch

```python
from helpers.solver import solver

obj, y, w, info = solver(params, scenarios, method="ccg", tolerance=1e-2)
# method ∈ {"saa", "ccg", "n-ccg"}
# n-ccg requires a trained QSurrogate; pass `surrogate=` or rely on
# params.surrogate_path.
```

### 4.3. End-to-end six-cell DBL/DFL evaluation

```python
from helpers.parameter import Params
from dataset.data_generation import generate_historical_data
from learning.evaluate import run_in_sample_evaluation

params = Params(num_areas=3, num_locations=6, num_food_types=3,
                num_scenarios_saa=200, time_limit=120.0, seed=27)
df, truth = generate_historical_data(num_areas=3, num_food_types=3,
                                     num_days=90, seed=27,
                                     surge_factor=3.0, heteroscedastic=True,
                                     noise_scale_factor=0.1)

results = run_in_sample_evaluation(
    df, truth, params, num_food_types=3,
    n_scenarios=200, n_oos_scenarios=500, verbose=True,
)
```

`results["cells"]` is a list of six dicts, one per cell in the matrix

|              | **Gauss noise**   | **ER bootstrap**   |
|:------------:|:-----------------:|:------------------:|
| **OLS**      | Linear-Gauss      | Linear-ER          |
| **RF/GBM**   | NonP-Gauss        | NonP-ER            |
| **SPO**      | SPO-Gauss         | SPO-ER             |

Each row records SAA objective, first-stage cost, average scenario unmet
demand, average unmet penalty, and (if `n_oos_scenarios > 0`) realized
out-of-sample cost replayed against the true DGP. The driver scripts
[`test_in_sample.py`](test_in_sample.py) and
[`test_out_of_sample.py`](test_out_of_sample.py) wrap this and emit the
plots in `outputs/`.

### 4.4. Train and use the Q-surrogate (neural CCG)

```python
from helpers.parameter import Params
from dataset.data_generation import generate_historical_data
from learning.nn_training_data import collect_training_data
from learning.mlp_training import train_surrogate
from learning.q_surrogate import QSurrogate
from helpers.solver import solver

params = Params(num_areas=3, num_locations=6, num_food_types=3,
                capacity=200.0, demand_max=400.0, seed=7)
_, truth = generate_historical_data(num_areas=3, num_food_types=3,
                                    num_days=90, seed=42)

# 1. Sample (y, w, d, Q) tuples and write to data/recourse_training_<size>.npz
collect_training_data(params, truth, num_samples=100_000)

# 2. Train MLP and save weights to data/q_surrogate_<size>.pt
train_surrogate(params)

# 3. Use it in n-CCG
surrogate = QSurrogate(params.surrogate_path)
obj, y, w, info = solver(params, scenarios, method="n-ccg",
                         surrogate=surrogate, compute_true_ub=True)
```

`Params` exposes `training_data_path` and `surrogate_path` properties
that key off the instance size `(I, J, T)`, so multiple instance sizes
coexist on disk without collision.

---

## 5. API reference

### `Params` (`helpers/parameter.py`)
Dataclass holding sizes, costs, eligibility, demand support, and solver
options. Cost vectors and the eligibility matrix `e` (contiguous even
partition of `J` across `I`) are auto-generated from `seed` if not
supplied. `_size_tag`, `training_data_path`, and `surrogate_path`
properties give per-instance disk paths.

### `generate_demand_scenarios(params, N, seed=None)`
Returns an `(N, I, T)` array of i.i.d. uniform demand scenarios on
`[demand_min, demand_max]`. Used for hand-built (non-contextual)
instances; contextual experiments use the DGP+projector instead.

### `recourse_problem(y, w, d, params, time_used=0.0)` (`subproblems/`)
Solves `Q(y, w; d)` as a Gurobi LP. `d` has shape `(I, T)`. Returns
`(obj, s, u, runtime)` with `s` shape `(I, J, T)` and `u` shape `(I, T)`.
On infeasibility, dumps the IIS to `gurobi_log/infeasible_sp.ilp`.

### `saa(scenarios, params, time_used=0.0, verbose=False)` (`subproblems/`)
Solves the full SAA MILP in one shot. `scenarios` has shape `(N, I, T)`.
Returns `(obj, y, w, s, u, runtime)`.

### `master_problem(params, S_hat, scenarios, time_used=0.0)` (`subproblems/ccg_master.py`)
Builds and solves the CCG master. `S_hat` is a list of `(I, T)` demand
arrays already added; `scenarios` is the full `(N, I, T)` set so the
master can attach a free `η_n` with a valid lower bound to every
scenario not yet in `S_hat`. Returns `(obj, y, w, eta, runtime)`.

### `ccg(scenarios, params, tolerance=1e-2, print_level=1)` (`subproblems/`)
Classical CCG outer loop. Selects the highest-cost remaining scenario
each iteration; stops when `(UB − LB) / |UB| ≤ tolerance`, when all
scenarios are added, or when `params.time_limit` is hit. Returns
`(obj, y, w, info)` with `info` containing `iterations`, `LB`, `UB`,
`gap`, `S_hat`, `time`, and `cvg_flag` (`0` not converged, `1` converged,
`2` time limit, `3` all scenarios exhausted).

### `neural_ccg(scenarios, params, surrogate, tolerance=1.0, …)` (`subproblems/`)
Shao-style n-CCG. Uses the surrogate to score every scenario in `O(N)`
neural calls per iteration, applies eq. (14) for termination, and
returns the master's `θ^k` as the reported objective. Pass
`compute_true_ub=True` to solve the true recourse once at the end for a
diagnostic gap.

### `solver(params, scenarios, method, …)` (`helpers/solver.py`)
Single entry point for `"saa"`, `"ccg"`, or `"n-ccg"`. Adds `time` and
`first_stage` to whichever `info` dict the underlying method returns.

### `Params._size_tag`, `training_data_path`, `surrogate_path`
Per-instance file paths so different `(I, J, T)` instances keep their
training tuples and surrogate weights in separate files.

### Dataset and projector (`dataset/`)
* `generate_historical_data(num_areas, num_food_types, num_days, …)` —
  synthetic DGP. Returns a long-format `pandas.DataFrame` and a
  `DGPTruth` carrying the true coefficients (so `build_oracle_model` can
  produce an oracle predictor for sanity-checks). Knobs:
  `surge_factor` (multiplicative weather × weekday surge),
  `heteroscedastic`, `noise_scale_factor` for residual std scaling.
* `generate_scenarios(truth, N, num_food_types, …)` — draws a
  `ScenarioSet` of `N` future days from the same DGP, including the
  realized contexts `(weather, weekday)` and the true demand `d_real`.
* `scenarios_for_method(scenarios, method, …)` — projects a
  `ScenarioSet` to the `(N, I, T)` array each method consumes:
  `"oracle"`, `"contextual"` (predictor mean), `"contextual_residual"`
  (predictor mean + Gaussian noise), `"contextual_er"` (predictor mean +
  bootstrapped empirical residuals), or `"noncontextual"` (broadcast a
  fixed mean-demand vector).

### Predictors (`learning/`)
* `fit_demand_model(df, num_food_types, …)` — OLS, pooled across food
  types by default. Returns a `DemandModel`.
* `fit_demand_model_nonparametric(df, num_food_types, kind, …)` — RF,
  GBM, KNN, or DT. Returns a `NonparametricDemandModel` with the same
  `.predict(weather, weekday)` and `.residual_std` interface.
* `fit_demand_model_spo(df, params, num_food_types, …)` — pseudo-DFL: a
  single-MILP whose loss is the realized two-stage cost when the
  prediction caps service. Returns a `DemandModel` plus a diagnostics
  dict.
* `compute_training_residuals(model, df, num_food_types)` — returns a
  `(n_days, I, T)` matrix of training-set residuals for the ER
  bootstrap.

### Q-surrogate pipeline (`learning/`)
* `collect_training_data(params, truth, num_samples, …)` — sample
  `(y, w, d)` triples across multiple `(p_open, total_frac_min)` regimes
  for broad coverage, solve the true recourse for each, and write to
  `params.training_data_path`.
* `train_surrogate(params, …)` — load tuples, standardize, train an MLP
  with Adam + early stopping, save weights and scaler stats to
  `params.surrogate_path`.
* `QSurrogate(path)` — NumPy-in/out inference wrapper. `predict(y, w,
  scenarios)` batches over the `N` axis in one forward pass.

### Six-cell evaluation harness (`learning/evaluate.py`)
`run_in_sample_evaluation(df, truth, params, num_food_types, …)` trains
each predictor once, draws one shared `ScenarioSet` so all cells see the
same realizations, projects via `scenarios_for_method` per cell, solves
SAA per cell, and (if `n_oos_scenarios > 0`) replays each cell's
`(y*, w*)` against the true DGP for an out-of-sample realized-cost
metric. Returns a results dict consumed by `test_in_sample.py` and
`test_out_of_sample.py` to produce the figures in `outputs/`.

---

## 6. Verification

Sanity checks the code is wired correctly:

1. `params.e.shape == (I, J)` and every column sums to 1.
2. `generate_demand_scenarios` returns `(N, I, T)`.
3. `saa` and `ccg` return objectives within tolerance on the same
   scenario set.
4. With `c^r = 0`, `Q ≥ 0` for every scenario and any feasible `(y, w)`,
   so the n-CCG selection / termination rules are well-defined.
5. `QSurrogate.predict_one(y, w, d)` agrees with `predict(y, w, d[None])[0]`.
6. The DGP's empirical context frequencies match the configured
   Bernoulli marginals (`weekday_prob ≈ 5/7`, `sunny_prob ≈ 0.6`).
