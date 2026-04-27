"""
Inference-time wrapper for the trained Q-surrogate.

Exposes a clean NumPy-in/NumPy-out interface that encapsulates all the torch
machinery, so downstream code (neural_ccg.py, experiment drivers) doesn't
need to know anything about tensors, devices, or standardization.

Usage
-----
    from q_surrogate import QSurrogate

    surrogate = QSurrogate("data/q_surrogate.pt")

    # For a single (y, w) and a batch of N scenarios, get predicted Q for each.
    y = np.array([1, 0, 1, 1, 0, 1], dtype=int)       # (J,)
    w = ...                                            # (J, T)
    scenarios = ...                                    # (N, I, T)
    q_hat = surrogate.predict(y, w, scenarios)        # (N,)

    # For a single triple:
    q_hat_single = surrogate.predict_one(y, w, d)     # scalar
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from learning.mlp_training import QSurrogateMLP, pick_device


class QSurrogate:
    """
    Lightweight wrapper around a trained QSurrogateMLP.

    Responsibilities:
      - Load weights + normalization stats from disk once at construction.
      - Accept NumPy arrays in the natural problem shapes (y, w, d).
      - Internally: flatten, standardize, forward-pass, de-standardize.
      - Batch over N scenarios in one forward pass (this is the point of
        neural CCG — it replaces N LP solves with one NN call).
    """

    def __init__(self, path: str = "data/q_surrogate.pt",
                 device: Optional[torch.device] = None):
        """
        Parameters
        ----------
        path : str
            Path to the .pt file produced by train_surrogate.py.
        device : torch.device, optional
            Override the auto-detected device (CUDA if available, else CPU).
        """
        if device is None:
            device = pick_device()
        self.device = device

        blob = torch.load(path, map_location=device, weights_only=False)

        # Architecture
        self.input_dim    = int(blob["input_dim"])
        self.hidden_sizes = tuple(blob["hidden_sizes"])

        # Rebuild the model and load weights
        self.model = QSurrogateMLP(
            input_dim    = self.input_dim,
            hidden_sizes = self.hidden_sizes,
        ).to(device)
        self.model.load_state_dict(blob["state_dict"])
        self.model.eval()

        # Normalization stats (kept as NumPy — we convert on each call)
        self.input_mean  = np.asarray(blob["input_mean"],  dtype=np.float32)
        self.input_std   = np.asarray(blob["input_std"],   dtype=np.float32)
        self.target_mean = float(blob["target_mean"])
        self.target_std  = float(blob["target_std"])

        # Metadata from the training run
        self.meta = blob.get("meta", {})

    # ----------------------------------------------------------------------
    # Shape helpers
    # ----------------------------------------------------------------------
    @property
    def num_areas(self) -> int:
        return int(self.meta.get("I", 0))

    @property
    def num_locations(self) -> int:
        return int(self.meta.get("J", 0))

    @property
    def num_food_types(self) -> int:
        return int(self.meta.get("T", 0))

    def _flatten_batch(
        self,
        y: np.ndarray,         # (J,)
        w: np.ndarray,         # (J, T)
        scenarios: np.ndarray, # (N, I, T)
    ) -> np.ndarray:
        """
        Build the (N, input_dim) input matrix by broadcasting (y, w) across
        all N scenarios and concatenating with the scenario-specific d.
        """
        N = scenarios.shape[0]
        y_flat = np.asarray(y, dtype=np.float32).reshape(-1)          # (J,)
        w_flat = np.asarray(w, dtype=np.float32).reshape(-1)          # (J*T,)
        d_flat = np.asarray(scenarios, dtype=np.float32).reshape(N, -1)  # (N, I*T)

        fixed = np.concatenate([y_flat, w_flat])                      # (J + J*T,)
        fixed_b = np.broadcast_to(fixed, (N, fixed.size))             # (N, J + J*T)
        return np.concatenate([fixed_b, d_flat], axis=1)              # (N, input_dim)

    # ----------------------------------------------------------------------
    # Prediction
    # ----------------------------------------------------------------------
    def predict(
        self,
        y: np.ndarray,
        w: np.ndarray,
        scenarios: np.ndarray,
    ) -> np.ndarray:
        """
        Predict Q for one (y, w) across N scenarios.

        Parameters
        ----------
        y : (J,) array  — first-stage open/close decision.
        w : (J, T) array — first-stage prepositioning.
        scenarios : (N, I, T) array — demand realizations.

        Returns
        -------
        q_hat : (N,) float64 array — predicted Q values in ORIGINAL units.
        """
        scenarios = np.asarray(scenarios, dtype=np.float32)
        if scenarios.ndim != 3:
            raise ValueError(
                f"scenarios must have shape (N, I, T); got {scenarios.shape}"
            )

        # Build, standardize, forward, de-standardize.
        X = self._flatten_batch(y, w, scenarios)                      # (N, input_dim)
        if X.shape[1] != self.input_dim:
            raise ValueError(
                f"flattened input dim {X.shape[1]} does not match surrogate's "
                f"input_dim {self.input_dim}. Check (I, J, T) vs. the training "
                f"instance."
            )
        X_std = (X - self.input_mean) / self.input_std

        with torch.no_grad():
            X_t = torch.from_numpy(X_std).to(self.device)
            pred_std = self.model(X_t).cpu().numpy()                  # (N,)

        q_hat = pred_std * self.target_std + self.target_mean
        return q_hat.astype(np.float64)

    def predict_one(
        self,
        y: np.ndarray,
        w: np.ndarray,
        d: np.ndarray,
    ) -> float:
        """
        Predict Q for a single (y, w, d) triple. Convenience wrapper around
        predict() that accepts a single scenario of shape (I, T) and returns
        a scalar.
        """
        d = np.asarray(d, dtype=np.float32)
        if d.ndim != 2:
            raise ValueError(f"d must have shape (I, T); got {d.shape}")
        return float(self.predict(y, w, d[None, ...])[0])


# ---------------------------------------------------------------------------
# Self-test: load the surrogate and compare its predictions against the
# true recourse solver on a few random points.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from helpers.parameter import Params
    from dataset.data_generation import generate_historical_data
    from dataset.scenario_generator import generate_scenarios
    from subproblems.recourse_problem import recourse_problem as solve_recourse
    from learning.nn_training_data import sample_y, sample_w

    I, J, T = 3, 6, 3
    pop_size = np.array([1, 2, 3])
    params = Params(
        num_areas=I, num_locations=J, num_food_types=T,
        capacity=200.0, fixed_cost=50.0,
        demand_min=0.0, demand_max=400.0,
        num_scenarios_saa=20, time_limit=60.0, seed=7,
    )
    _, truth = generate_historical_data(
        num_areas=I, num_food_types=T, num_days=90,
        pop_size=pop_size, seed=42,
    )

    surrogate = QSurrogate(params.surrogate_path)
    print(f"Loaded surrogate: input_dim={surrogate.input_dim}, "
          f"hidden={surrogate.hidden_sizes}, device={surrogate.device}")
    print(f"Training meta: I={surrogate.num_areas}, J={surrogate.num_locations}, "
          f"T={surrogate.num_food_types}")

    # --- Smoke test: one (y, w) across a BATCH of N scenarios ---
    rng = np.random.default_rng(123)
    y = sample_y(J, rng, p_open=0.9)
    w = sample_w(y, T, params.capacity, rng, total_frac_min=0.5)
    scen = generate_scenarios(truth, N=20, num_food_types=T,
                              demand_min=params.demand_min,
                              demand_max=params.demand_max,
                              seed=2024)
    d_batch = scen.d_real                                     # (20, I, T)

    q_hat = surrogate.predict(y, w, d_batch)                  # (20,)

    # Solve the true recourse for each scenario for comparison.
    q_true = np.array([
        solve_recourse(y, w, d_batch[m], params)[0] for m in range(20)
    ])

    resid = q_true - q_hat
    print(f"\nBatched predict(): shape {q_hat.shape}")
    print(f"  True Q range   : [{q_true.min():9.2f}, {q_true.max():9.2f}]")
    print(f"  Predicted range: [{q_hat.min():9.2f}, {q_hat.max():9.2f}]")
    print(f"  Mean |error|   : {np.abs(resid).mean():.3f}")
    print(f"  Max  |error|   : {np.abs(resid).max():.3f}")

    # --- Does the surrogate's argmax match the true argmax? ---
    # This is the specific thing neural CCG cares about: picking the
    # highest-Q scenario, not getting every Q perfectly.
    i_true = int(np.argmax(q_true))
    i_hat  = int(np.argmax(q_hat))
    print(f"\n  Argmax agreement check:")
    print(f"    true argmax = {i_true} (Q = {q_true[i_true]:.2f})")
    print(f"    hat  argmax = {i_hat} (Q = {q_true[i_hat]:.2f}, "
          f"hat = {q_hat[i_hat]:.2f})")
    if i_true == i_hat:
        print("    -> exact agreement.")
    else:
        gap = q_true[i_true] - q_true[i_hat]
        print(f"    -> disagreement; missed true max by {gap:.2f} "
              f"({gap/abs(q_true[i_true])*100:.2f}% of true argmax).")

    # --- predict_one() round-trip check ---
    q_single = surrogate.predict_one(y, w, d_batch[0])
    assert abs(q_single - q_hat[0]) < 1e-5, "predict_one != predict[0]"
    print(f"\n  predict_one == predict()[0]?  OK  ({q_single:.4f})")