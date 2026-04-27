"""
End-to-end out-of-sample (OOS) test of the six contextual-learning cells,
with plots.

For each cell:
    1. Train predictor (OLS / RF / SPO) on historical data.
    2. Compute residuals (only for ER cells).
    3. Build in-sample scenarios from (predictor + noise model).
    4. Solve SAA to get the first-stage decision (y*, w*).
    5. Replay the recourse against N_oos draws of TRUE demand from the
       DGP. Record average realized cost and unmet.

The OOS metric — realized cost of (y*, w*) against ground-truth demand —
is the proper deployment-quality metric. The in-sample SAA objective is
NOT directly comparable across cells because each cell's SAA optimizes
against its own predicted scenario distribution.

Run from the Code/ directory:
    python test_out_of_sample.py

Outputs:
    - Per-cell summary printed to stdout (in-sample + OOS columns).
    - PNG figure saved to outputs/out_of_sample_comparison.png with three
      panels:
        (1) OOS realized cost per cell
        (2) Cost decomposition (first-stage + OOS unmet penalty)
        (3) OOS avg unmet demand
    - PNG figure saved to outputs/in_vs_oos_comparison.png with two panels
      side-by-side: in-sample SAA obj vs OOS realized cost. Visualizes the
      gap between "what SAA thinks the cost is" and "what cost actually
      gets realized in the world."
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from helpers.parameter import Params
from dataset.data_generation import generate_historical_data
from learning.evaluate import run_in_sample_evaluation


# =============================================================================
# 1. Configuration
# =============================================================================
SEED               = 27
I, J, T            = 10, 20, 3
NUM_DAYS           = 100            # historical training days
N_SAA              = 200            # SAA scenarios per cell (in-sample)
N_OOS              = 500            # ground-truth draws used for OOS evaluation

# Heterogeneous DGP — same as test_in_sample.py for direct comparability.
# SURGE_FACTOR       = 3.0
# HETEROSCEDASTIC    = True
# NOISE_SCALE_FACTOR = 0.1

SURGE_FACTOR       = 6.0
HETEROSCEDASTIC    = True
NOISE_SCALE_FACTOR = 3

SPO_MAX_TRAIN      = None
SPO_TIME_LIMIT     = 180.0
NONPARAMETRIC_KIND = "rf"

OUTPUT_DIR             = Path("outputs")
FIGURE_OOS_PATH        = OUTPUT_DIR / "out_of_sample_comparison.png"
FIGURE_IN_VS_OOS_PATH  = OUTPUT_DIR / "in_vs_oos_comparison.png"


# =============================================================================
# 2. Build the instance and historical data
# =============================================================================
np.random.seed(SEED)
pop_size = np.random.randint(1, 5, size=I)

params = Params(
    num_areas         = I,
    num_locations     = J,
    num_food_types    = T,
    capacity          = 500.0,
    fixed_cost        = 100.0,
    demand_min        = 0.0,
    demand_max        = 300.0,
    acquisition_cost  = np.array([4.0, 6.0, 2.0]),
    unmet_penalty     = np.array([10.0, 15.0, 5.0]),
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
print(f"OOS test: I={I}, J={J}, T={T}, num_days={NUM_DAYS}, "
      f"N_SAA={N_SAA}, N_OOS={N_OOS}")
print(f"DGP: surge_factor={SURGE_FACTOR}, heteroscedastic={HETEROSCEDASTIC}, "
      f"noise_scale_factor={NOISE_SCALE_FACTOR}")
print(f"Predictors: OLS, {NONPARAMETRIC_KIND.upper()}, SPO (pseudo-DFL)")
print("=" * 78)


# =============================================================================
# 3. Run the harness with OOS enabled
# =============================================================================
results = run_in_sample_evaluation(
    df, truth, params, num_food_types=T,
    n_scenarios          = N_SAA,
    nonparametric_kind   = NONPARAMETRIC_KIND,
    spo_max_train_samples= SPO_MAX_TRAIN,
    spo_time_limit       = SPO_TIME_LIMIT,
    n_oos_scenarios      = N_OOS,
    verbose              = True,
)

cells = results["cells"]


# =============================================================================
# 4. Plot — OOS three-panel comparison
# =============================================================================
predictor_kinds  = ["ols",   "rf",   "spo"]
predictor_labels = ["Linear", "NonP", "SPO"]
noise_kinds      = ["gauss", "er"]
noise_labels     = ["Gauss", "ER"]


def _matrix(field: str) -> np.ndarray:
    M = np.full((len(predictor_kinds), len(noise_kinds)), np.nan)
    for r in cells:
        if r["status"] != "ok":
            continue
        i = predictor_kinds.index(r["predictor"])
        j = noise_kinds.index(r["noise"])
        M[i, j] = r.get(field, np.nan)
    return M


obj_M             = _matrix("obj")              # in-sample SAA obj
first_stage_M     = _matrix("first_stage")
oos_cost_M        = _matrix("oos_cost")
oos_unmet_M       = _matrix("oos_unmet")
oos_unmet_cost_M  = _matrix("oos_unmet_cost")

x       = np.arange(len(predictor_kinds))
width   = 0.35
gauss_x = x - width / 2
er_x    = x + width / 2

GAUSS_COLOR = "#4C72B0"
ER_COLOR    = "#DD8452"

fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

# ---- Panel 1: OOS realized cost --------------------------------------------
ax = axes[0]
ax.bar(gauss_x, oos_cost_M[:, 0], width, label="Gauss", color=GAUSS_COLOR)
ax.bar(er_x,    oos_cost_M[:, 1], width, label="ER",    color=ER_COLOR)
ax.set_xticks(x)
ax.set_xticklabels(predictor_labels)
ax.set_ylabel("OOS realized cost")
ax.set_title("(a) OOS realized cost (lower is better)")
ax.legend(title="Noise model")
ax.grid(axis="y", linestyle=":", alpha=0.5)
for bx, by in zip(gauss_x, oos_cost_M[:, 0]):
    if np.isfinite(by):
        ax.annotate(f"{by:.0f}", (bx, by), ha="center", va="bottom", fontsize=8)
for bx, by in zip(er_x, oos_cost_M[:, 1]):
    if np.isfinite(by):
        ax.annotate(f"{by:.0f}", (bx, by), ha="center", va="bottom", fontsize=8)

# ---- Panel 2: OOS cost breakdown --------------------------------------------
ax = axes[1]
ax.bar(gauss_x, first_stage_M[:, 0], width, color=GAUSS_COLOR)
ax.bar(gauss_x, oos_unmet_cost_M[:, 0], width, color=GAUSS_COLOR, alpha=0.45,
       bottom=first_stage_M[:, 0])
ax.bar(er_x,    first_stage_M[:, 1], width, color=ER_COLOR)
ax.bar(er_x,    oos_unmet_cost_M[:, 1], width, color=ER_COLOR, alpha=0.45,
       bottom=first_stage_M[:, 1])
ax.set_xticks(x)
ax.set_xticklabels(predictor_labels)
ax.set_ylabel("Cost")
ax.set_title("(b) OOS cost decomposition (Gauss left, ER right)")
ax.grid(axis="y", linestyle=":", alpha=0.5)
ax.legend(handles=[
    Patch(facecolor=GAUSS_COLOR, label="Gauss"),
    Patch(facecolor=ER_COLOR,    label="ER"),
    Patch(facecolor="gray",       label="first-stage"),
    Patch(facecolor="gray", alpha=0.45, label="OOS unmet penalty"),
], loc="upper right", fontsize=8)

# ---- Panel 3: OOS avg unmet demand -----------------------------------------
ax = axes[2]
ax.bar(gauss_x, oos_unmet_M[:, 0], width, label="Gauss", color=GAUSS_COLOR)
ax.bar(er_x,    oos_unmet_M[:, 1], width, label="ER",    color=ER_COLOR)
ax.set_xticks(x)
ax.set_xticklabels(predictor_labels)
ax.set_ylabel("OOS avg unmet demand per scenario (units)")
ax.set_title("(c) OOS avg unmet demand (lower is better)")
ax.legend(title="Noise model")
ax.grid(axis="y", linestyle=":", alpha=0.5)
for bx, by in zip(gauss_x, oos_unmet_M[:, 0]):
    if np.isfinite(by):
        ax.annotate(f"{by:.1f}", (bx, by), ha="center", va="bottom", fontsize=8)
for bx, by in zip(er_x, oos_unmet_M[:, 1]):
    if np.isfinite(by):
        ax.annotate(f"{by:.1f}", (bx, by), ha="center", va="bottom", fontsize=8)

plt.suptitle(
    f"Out-of-sample evaluation of contextual-learning cells "
    f"(I={I}, J={J}, T={T}, N_SAA={N_SAA}, N_OOS={N_OOS}; "
    f"surge={SURGE_FACTOR}, het={HETEROSCEDASTIC})",
    fontsize=11, y=1.02,
)
plt.tight_layout()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
plt.savefig(FIGURE_OOS_PATH, dpi=150, bbox_inches="tight")
print(f"\nFigure saved to {FIGURE_OOS_PATH}")


# =============================================================================
# 5. Plot — In-sample vs OOS gap
# =============================================================================
# Side-by-side comparison: in-sample SAA obj vs OOS realized cost. Visualizes
# the "SAA grades its own homework" issue — cells whose in-sample obj looks
# good often look worse OOS, and vice versa.
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)

ax = axes2[0]
ax.bar(gauss_x, obj_M[:, 0], width, label="Gauss", color=GAUSS_COLOR)
ax.bar(er_x,    obj_M[:, 1], width, label="ER",    color=ER_COLOR)
ax.set_xticks(x)
ax.set_xticklabels(predictor_labels)
ax.set_ylabel("Cost")
ax.set_title("In-sample SAA objective\n(SAA grades its own homework)")
ax.legend(title="Noise model")
ax.grid(axis="y", linestyle=":", alpha=0.5)

ax = axes2[1]
ax.bar(gauss_x, oos_cost_M[:, 0], width, label="Gauss", color=GAUSS_COLOR)
ax.bar(er_x,    oos_cost_M[:, 1], width, label="ER",    color=ER_COLOR)
ax.set_xticks(x)
ax.set_xticklabels(predictor_labels)
ax.set_title(f"OOS realized cost\n(N_OOS={N_OOS} ground-truth draws)")
ax.legend(title="Noise model")
ax.grid(axis="y", linestyle=":", alpha=0.5)

plt.suptitle("In-sample vs OOS — same y-axis scale for direct comparison",
             fontsize=11, y=1.02)
plt.tight_layout()
plt.savefig(FIGURE_IN_VS_OOS_PATH, dpi=150, bbox_inches="tight")
print(f"Figure saved to {FIGURE_IN_VS_OOS_PATH}")

plt.show()
