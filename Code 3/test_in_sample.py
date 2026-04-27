"""
End-to-end in-sample test of the six contextual-learning cells, with plots.

Run from the Code/ directory:
    python test_in_sample.py

Outputs:
    - Per-cell summary printed to stdout (also returned by the harness).
    - PNG figure saved to outputs/in_sample_comparison.png with three panels:
        (1) SAA objective per cell
        (2) Cost breakdown (first-stage + unmet penalty)
        (3) Average per-scenario unmet demand
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from helpers.parameter import Params
from dataset.data_generation import generate_historical_data
from learning.evaluate import run_in_sample_evaluation


# =============================================================================
# 1. Configuration
# =============================================================================
SEED               = 27
I, J, T            = 3, 6, 3
NUM_DAYS           = 90              # historical training days
N_SAA              = 200             # SAA scenarios used by every cell

# Heterogeneous DGP: surge on (sunny, weekday); noise scales with mean.
# Without these, all four DBL cells tie the baseline.
SURGE_FACTOR       = 3.0
HETEROSCEDASTIC    = True
NOISE_SCALE_FACTOR = 0.1

# SPO MILP knobs.
SPO_MAX_TRAIN      = None            # None = use all NUM_DAYS days
SPO_TIME_LIMIT     = 180.0
NONPARAMETRIC_KIND = "rf"            # "rf" | "gbm" | "knn" | "dt"

OUTPUT_DIR         = Path("outputs")
FIGURE_PATH        = OUTPUT_DIR / "in_sample_comparison.png"


# =============================================================================
# 2. Build the instance and historical data
# =============================================================================
np.random.seed(SEED)
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
    num_scenarios_saa = N_SAA,
    time_limit        = 120.0,
    seed              = SEED,
)

df, truth = generate_historical_data(
    num_areas          = I,
    num_food_types     = T,
    num_days           = NUM_DAYS,
    pop_size           = pop_size,
    seed               = SEED,
    surge_factor       = SURGE_FACTOR,
    heteroscedastic    = HETEROSCEDASTIC,
    noise_scale_factor = NOISE_SCALE_FACTOR,
)

print("=" * 78)
print(f"In-sample test: I={I}, J={J}, T={T}, num_days={NUM_DAYS}, N_SAA={N_SAA}")
print(f"DGP: surge_factor={SURGE_FACTOR}, heteroscedastic={HETEROSCEDASTIC}, "
      f"noise_scale_factor={NOISE_SCALE_FACTOR}")
print(f"Predictors: OLS, {NONPARAMETRIC_KIND.upper()}, SPO (pseudo-DFL)")
print("=" * 78)


# =============================================================================
# 3. Run the harness
# =============================================================================
results = run_in_sample_evaluation(
    df, truth, params, num_food_types=T,
    n_scenarios          = N_SAA,
    nonparametric_kind   = NONPARAMETRIC_KIND,
    spo_max_train_samples= SPO_MAX_TRAIN,
    spo_time_limit       = SPO_TIME_LIMIT,
    verbose              = True,
)

cells = results["cells"]


# =============================================================================
# 4. Plot
# =============================================================================
# Layout: rows = predictor (Linear / NonP / SPO), columns = noise (Gauss / ER).
# Pull values into matrices indexed by (predictor_idx, noise_idx).
predictor_kinds   = ["ols",   "rf",   "spo"]
predictor_labels  = ["Linear", "NonP", "SPO"]
noise_kinds       = ["gauss", "er"]
noise_labels      = ["Gauss", "ER"]

def _matrix(field: str) -> np.ndarray:
    M = np.full((len(predictor_kinds), len(noise_kinds)), np.nan)
    for r in cells:
        if r["status"] != "ok":
            continue
        i = predictor_kinds.index(r["predictor"])
        j = noise_kinds.index(r["noise"])
        M[i, j] = r[field]
    return M

obj_M         = _matrix("obj")
first_stage_M = _matrix("first_stage")
avg_unmet_M   = _matrix("avg_unmet")
unmet_cost_M  = _matrix("avg_unmet_cost")

x       = np.arange(len(predictor_kinds))
width   = 0.35
gauss_x = x - width / 2
er_x    = x + width / 2

GAUSS_COLOR = "#4C72B0"   # blue
ER_COLOR    = "#DD8452"   # orange (Tableau-style)

fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

# ---- Panel 1: SAA objective -------------------------------------------------
ax = axes[0]
ax.bar(gauss_x, obj_M[:, 0], width, label="Gauss", color=GAUSS_COLOR)
ax.bar(er_x,    obj_M[:, 1], width, label="ER",    color=ER_COLOR)
ax.set_xticks(x)
ax.set_xticklabels(predictor_labels)
ax.set_ylabel("SAA objective")
ax.set_title("(a) SAA cost (lower is better)")
ax.legend(title="Noise model")
ax.grid(axis="y", linestyle=":", alpha=0.5)
# Annotate bars with values
for bx, by in zip(gauss_x, obj_M[:, 0]):
    if np.isfinite(by):
        ax.annotate(f"{by:.0f}", (bx, by), ha="center", va="bottom", fontsize=8)
for bx, by in zip(er_x, obj_M[:, 1]):
    if np.isfinite(by):
        ax.annotate(f"{by:.0f}", (bx, by), ha="center", va="bottom", fontsize=8)

# ---- Panel 2: cost breakdown (stacked: first-stage + unmet penalty) ---------
ax = axes[1]
ax.bar(gauss_x, first_stage_M[:, 0], width, color=GAUSS_COLOR, label="first-stage")
ax.bar(gauss_x, unmet_cost_M[:, 0],  width, color=GAUSS_COLOR, alpha=0.45,
       bottom=first_stage_M[:, 0], label="unmet penalty")
ax.bar(er_x,    first_stage_M[:, 1], width, color=ER_COLOR)
ax.bar(er_x,    unmet_cost_M[:, 1],  width, color=ER_COLOR, alpha=0.45,
       bottom=first_stage_M[:, 1])
ax.set_xticks(x)
ax.set_xticklabels(predictor_labels)
ax.set_ylabel("Cost")
ax.set_title("(b) Cost decomposition (Gauss left, ER right)")
ax.grid(axis="y", linestyle=":", alpha=0.5)
# Custom legend (only show one Gauss + one ER + the alpha distinction)
from matplotlib.patches import Patch
legend_handles = [
    Patch(facecolor=GAUSS_COLOR, label="Gauss"),
    Patch(facecolor=ER_COLOR,    label="ER"),
    Patch(facecolor="gray",       label="first-stage"),
    Patch(facecolor="gray", alpha=0.45, label="unmet penalty"),
]
ax.legend(handles=legend_handles, loc="upper right", fontsize=8)

# ---- Panel 3: avg unmet demand (units) --------------------------------------
ax = axes[2]
ax.bar(gauss_x, avg_unmet_M[:, 0], width, label="Gauss", color=GAUSS_COLOR)
ax.bar(er_x,    avg_unmet_M[:, 1], width, label="ER",    color=ER_COLOR)
ax.set_xticks(x)
ax.set_xticklabels(predictor_labels)
ax.set_ylabel("Avg unmet demand per scenario (units)")
ax.set_title("(c) Avg unmet demand (lower is better)")
ax.legend(title="Noise model")
ax.grid(axis="y", linestyle=":", alpha=0.5)
for bx, by in zip(gauss_x, avg_unmet_M[:, 0]):
    if np.isfinite(by):
        ax.annotate(f"{by:.1f}", (bx, by), ha="center", va="bottom", fontsize=8)
for bx, by in zip(er_x, avg_unmet_M[:, 1]):
    if np.isfinite(by):
        ax.annotate(f"{by:.1f}", (bx, by), ha="center", va="bottom", fontsize=8)

plt.suptitle(
    f"In-sample comparison of contextual-learning cells "
    f"(I={I}, J={J}, T={T}, N_SAA={N_SAA}; surge={SURGE_FACTOR}, het={HETEROSCEDASTIC})",
    fontsize=11, y=1.02,
)
plt.tight_layout()

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
plt.savefig(FIGURE_PATH, dpi=150, bbox_inches="tight")
print(f"\nFigure saved to {FIGURE_PATH}")

plt.show()
