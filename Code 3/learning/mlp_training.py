"""
Train an MLP surrogate Q-hat(y, w, d) for the food-truck recourse problem.

Input  : concatenation of flattened y, w, d. Shape  (J + J*T + I*T,).
Output : scalar Q prediction.

Training pipeline
-----------------
1. Load dataset/recourse_training.npz produced by collect_training_data.py.
2. Split 80 / 10 / 10 into train / val / test.
3. Standardize inputs (zero mean, unit variance per feature) and target Q.
   Save the scaling stats so inference can de-normalize.
4. Train with MSE loss and Adam, with early stopping on validation loss.
5. Report RMSE and R^2 on the held-out test set, in ORIGINAL Q units.
6. Save to dataset/q_surrogate.pt:
       state_dict          - network weights
       input_mean/std      - per-feature input scaling
       target_mean/std     - target Q scaling
       hidden_sizes        - architecture
       meta                - I, J, T, etc. (copied from training data)

Device
------
Uses CUDA if available, else CPU. Controlled automatically.
"""
from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from learning.nn_training_data import load_training_data


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------
def pick_device() -> torch.device:
    """CUDA if available, else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class QSurrogateMLP(nn.Module):
    """Feed-forward MLP with ReLU activations. Single scalar output."""
    def __init__(self, input_dim: int, hidden_sizes: tuple[int, ...] = (128, 128, 64)):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)              # (B,)


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
def flatten_inputs(y: np.ndarray, w: np.ndarray, d: np.ndarray) -> np.ndarray:
    """
    Concatenate flattened (y, w, d) into a single (M, input_dim) array.

    Shapes:  y (M, J),  w (M, J, T),  d (M, I, T)
    Output:  (M, J + J*T + I*T), float32
    """
    M = y.shape[0]
    y_flat = y.reshape(M, -1).astype(np.float32)
    w_flat = w.reshape(M, -1).astype(np.float32)
    d_flat = d.reshape(M, -1).astype(np.float32)
    return np.concatenate([y_flat, w_flat, d_flat], axis=1)


def split_indices(
    M: int,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random train / val / test index split."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(M)
    n_train = int(train_frac * M)
    n_val   = int(val_frac * M)
    return perm[:n_train], perm[n_train:n_train + n_val], perm[n_train + n_val:]


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_surrogate(
    data_path: str = "data/recourse_training.npz",
    out_path:  str = "data/q_surrogate.pt",
    hidden_sizes: tuple[int, ...] = (128, 128, 64),
    batch_size:   int = 256,
    max_epochs:   int = 200,
    lr:           float = 1e-3,
    weight_decay: float = 1e-5,
    patience:     int = 15,           # early-stopping patience on val loss
    seed:         int = 0,
    verbose:      bool = True,
) -> dict:
    """
    Train the Q-surrogate. Returns a summary dict with test metrics.
    """
    device = pick_device()
    if verbose:
        print(f"Device: {device}")

    # --- load data -------------------------------------------------------
    data = load_training_data(data_path)
    y, w, d, Q = data["y"], data["w"], data["d"], data["Q"]
    meta = data["meta"]
    M = len(Q)

    X = flatten_inputs(y, w, d)                     # (M, input_dim)
    if verbose:
        print(f"Loaded {M} samples; input_dim = {X.shape[1]}")

    # --- split -----------------------------------------------------------
    idx_tr, idx_va, idx_te = split_indices(M, train_frac=0.8, val_frac=0.1, seed=seed)
    X_tr, X_va, X_te = X[idx_tr], X[idx_va], X[idx_te]
    Q_tr, Q_va, Q_te = Q[idx_tr], Q[idx_va], Q[idx_te]

    # --- standardize (using TRAIN stats only) ----------------------------
    input_mean = X_tr.mean(axis=0)
    input_std  = X_tr.std(axis=0)
    input_std[input_std < 1e-8] = 1.0               # avoid divide-by-zero on binary y

    target_mean = float(Q_tr.mean())
    target_std  = float(Q_tr.std())
    if target_std < 1e-8:
        target_std = 1.0

    def std_x(arr): return (arr - input_mean) / input_std
    def std_q(arr): return (arr - target_mean) / target_std

    X_tr_s = std_x(X_tr).astype(np.float32)
    X_va_s = std_x(X_va).astype(np.float32)
    X_te_s = std_x(X_te).astype(np.float32)
    Q_tr_s = std_q(Q_tr).astype(np.float32)
    Q_va_s = std_q(Q_va).astype(np.float32)
    # Keep Q_te in ORIGINAL units for final reporting.

    # --- torch dataloaders ----------------------------------------------
    def to_loader(X_s, Q_s, shuffle):
        ds = TensorDataset(torch.from_numpy(X_s), torch.from_numpy(Q_s))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    train_loader = to_loader(X_tr_s, Q_tr_s, shuffle=True)
    val_loader   = to_loader(X_va_s, Q_va_s, shuffle=False)

    # --- model / optim --------------------------------------------------
    torch.manual_seed(seed)
    model = QSurrogateMLP(input_dim=X.shape[1], hidden_sizes=hidden_sizes).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    bad_epochs = 0
    history = {"train": [], "val": []}

    t_start = perf_counter()
    for epoch in range(1, max_epochs + 1):
        model.train()
        tr_loss_sum = 0.0
        tr_count    = 0
        for xb, qb in train_loader:
            xb = xb.to(device, non_blocking=True)
            qb = qb.to(device, non_blocking=True)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, qb)
            loss.backward()
            opt.step()
            tr_loss_sum += loss.item() * xb.size(0)
            tr_count    += xb.size(0)
        tr_loss = tr_loss_sum / tr_count

        model.eval()
        val_loss_sum = 0.0
        val_count    = 0
        with torch.no_grad():
            for xb, qb in val_loader:
                xb = xb.to(device, non_blocking=True)
                qb = qb.to(device, non_blocking=True)
                pred = model(xb)
                val_loss_sum += loss_fn(pred, qb).item() * xb.size(0)
                val_count    += xb.size(0)
        val_loss = val_loss_sum / val_count

        history["train"].append(tr_loss)
        history["val"  ].append(val_loss)

        improved = val_loss < best_val - 1e-6
        if improved:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        if verbose and (epoch == 1 or epoch % 10 == 0 or bad_epochs == 0):
            flag = "*" if improved else " "
            print(
                f"  epoch {epoch:4d} | train {tr_loss:.4f} | val {val_loss:.4f} {flag} "
                f"| best val {best_val:.4f} | bad {bad_epochs}"
            )

        if bad_epochs >= patience:
            if verbose:
                print(f"  early stop at epoch {epoch} (patience {patience})")
            break

    total_time = perf_counter() - t_start
    model.load_state_dict(best_state)                # restore best

    # --- test evaluation in ORIGINAL Q units ----------------------------
    model.eval()
    with torch.no_grad():
        X_te_t   = torch.from_numpy(X_te_s).to(device)
        pred_std = model(X_te_t).cpu().numpy()
    pred_orig = pred_std * target_std + target_mean

    residuals = Q_te - pred_orig
    rmse = float(np.sqrt((residuals ** 2).mean()))
    mae  = float(np.abs(residuals).mean())
    ss_res = float((residuals ** 2).sum())
    ss_tot = float(((Q_te - Q_te.mean()) ** 2).sum())
    r2     = 1.0 - ss_res / ss_tot

    # --- save -----------------------------------------------------------
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(
        state_dict   = best_state,
        input_mean   = input_mean,
        input_std    = input_std,
        target_mean  = target_mean,
        target_std   = target_std,
        hidden_sizes = hidden_sizes,
        input_dim    = X.shape[1],
        meta         = meta,
        history      = history,
    ), out_path)

    summary = dict(
        num_train       = len(idx_tr),
        num_val         = len(idx_va),
        num_test        = len(idx_te),
        input_dim       = X.shape[1],
        hidden_sizes    = hidden_sizes,
        best_val_std    = best_val,
        epochs_trained  = len(history["train"]),
        total_time      = total_time,
        device          = str(device),
        test_rmse       = rmse,
        test_mae        = mae,
        test_r2         = r2,
        Q_test_std      = float(Q_te.std()),
        out_path        = out_path,
    )
    return summary


# ---------------------------------------------------------------------------
# Inference-time loader (used later by the QSurrogate wrapper class)
# ---------------------------------------------------------------------------
def load_surrogate(path: str = "data/q_surrogate.pt",
                   device: Optional[torch.device] = None) -> dict:
    """
    Load a saved surrogate. Returns a dict with model + scaling constants;
    the actual prediction wrapper class will consume this.
    """
    if device is None:
        device = pick_device()
    blob = torch.load(path, map_location=device, weights_only=False)
    model = QSurrogateMLP(
        input_dim    = blob["input_dim"],
        hidden_sizes = blob["hidden_sizes"],
    ).to(device)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    blob["model"]  = model
    blob["device"] = device
    return blob


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from helpers.parameter import Params

    # Match the (I, J, T) used by nn_training_data.py's __main__ so the
    # writer and reader agree on data/recourse_training_*.npz and
    # data/q_surrogate_*.pt paths.
    params = Params(
        num_areas         = 3,
        num_locations     = 6,
        num_food_types    = 3,
        capacity          = 200.0,
        fixed_cost        = 50.0,
        demand_min        = 0.0,
        demand_max        = 400.0,
        num_scenarios_saa = 20,
        time_limit        = 60.0,
        seed              = 7,
    )

    summary = train_surrogate(
        data_path    = params.training_data_path,
        out_path     = params.surrogate_path,
        hidden_sizes = (128, 128, 64),
        batch_size   = 256,
        max_epochs   = 200,
        lr           = 1e-3,
        weight_decay = 1e-5,
        patience     = 15,
        seed         = 0,
        verbose      = True,
    )

    print("\n=== Training summary ===")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k:17s}: {v:.6f}")
        else:
            print(f"  {k:17s}: {v}")