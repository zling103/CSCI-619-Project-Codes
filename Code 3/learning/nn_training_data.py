"""
Training-data collection for the Q-surrogate (neural CCG, Shao et al. 2025 style).

We sample (y, w, d) triples INDEPENDENTLY and solve the
true recourse Q(y, w; d) for each, then save to disk. Independent sampling keeps
the surrogate's training distribution decoupled from any solver's behavior, which
matches Shao et al.'s recipe and avoids any appearance of "cheating" by training
on CCG-specific intermediate states.

Sampling scheme (multi-regime for broad coverage)
-------------------------------------------------
The sample budget is split across REGIMES, where each regime is a tuple
(p_open, total_frac_min) controlling how aggressively y and w are sampled:

    p_open         : Pr(each truck is open). Low -> under-provisioned; high -> over-provisioned.
    total_frac_min : per-open-truck fill is Uniform(total_frac_min * C, C).
                     Low -> open trucks often nearly empty; high -> open trucks mostly full.

Per sample inside a given regime:
    y_j          ~  Bernoulli(p_open)
    total_j      ~  Uniform(total_frac_min * C, C)   if y_j = 1, else 0
    allocation   ~  Dirichlet(1, ..., 1)             across food types
    w_{j, :}     =  total_j * allocation             if y_j = 1, else 0
    d            ~  generate_scenarios(truth, N=1)   from the true DGP

The default regime list spans the full {under,over}-provisioned cross-product:
    p_open         in {0.1, 0.3, 0.5, 0.7, 0.9, 1.0}
    total_frac_min in {0.0, 0.3, 0.6}
This gives 18 regimes, so for num_samples=100_000 each regime contributes ~5_556
samples. Good for covering the (y, w) regions CCG's master occasionally visits.

Output format
-------------
dataset/recourse_training.npz with arrays
    y       : (M, J)     int8
    w       : (M, J, T)  float32
    d       : (M, I, T)  float32
    Q       : (M,)       float32
    regime  : (M,)       int32    regime index (row in meta['regimes'])
    meta    : object-array wrapping a dict with keys
              I, J, T, capacity, noise_std, num_samples, regimes, ...
"""
from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Optional

import numpy as np

from helpers.parameter import Params
from dataset.data_generation import generate_historical_data, DGPTruth
from subproblems.recourse_problem import recourse_problem
from dataset.scenario_generator import generate_scenarios


# ---------------------------------------------------------------------------
# Default regime grid
# ---------------------------------------------------------------------------
DEFAULT_REGIMES = [
    (p_open, total_frac_min)
    for p_open in (0.1, 0.3, 0.5, 0.7, 0.9, 1.0)
    for total_frac_min in (0.0, 0.3, 0.6)
]   # 6 x 3 = 18 regimes


def _assign_regimes(num_samples: int, num_regimes: int,
                    rng: np.random.Generator) -> np.ndarray:
    """
    Return shape-(num_samples,) int array of regime indices, distributing
    samples as evenly as possible across regimes and then shuffling the order
    so regimes are interleaved. Even distribution matters when num_samples
    is not divisible by num_regimes.
    """
    base, rem = divmod(num_samples, num_regimes)
    counts = np.full(num_regimes, base, dtype=int)
    counts[:rem] += 1                                     # first `rem` regimes get +1
    regime_idx = np.repeat(np.arange(num_regimes), counts)
    rng.shuffle(regime_idx)
    return regime_idx



def sample_y(J: int, rng: np.random.Generator, p_open: float = 0.9) -> np.ndarray:
    """
    Bernoulli(p_open) per component. Returns shape (J,), dtype int8.
    p_open default is 0.9 so that, on average, almost every truck is open.
    This matters because Q depends on PER-AREA provisioning: if both trucks
    serving an area are closed, that area's entire demand becomes unmet,
    dwarfing any effect elsewhere. With p_open=0.9, the Pr(all closed in
    an area) ~ 1% and Q can span negative values (over-provisioned regime).
    """
    return (rng.random(size=J) < p_open).astype(np.int8)


def sample_w(y: np.ndarray, T: int, capacity: float,
             rng: np.random.Generator,
             total_frac_min: float = 0.5) -> np.ndarray:
    """
    For each open truck, sample total capacity ~ U(total_frac_min*C, C) and
    allocate it across T food types via Dirichlet(1,...,1) (uniform on the
    simplex). Closed trucks get zero prepositioning.

    Default total_frac_min=0.5 means open trucks are at least half-full;
    combined with p_open=0.9 this puts the typical sample in a regime
    where Q spans zero (some positive, some negative), which is where
    neural CCG will actually query Q at test time.

    Returns shape (J, T), float32.
    """
    J = y.shape[0]
    w = np.zeros((J, T), dtype=np.float32)
    total_lo = total_frac_min * capacity
    for j in range(J):
        if y[j] == 1:
            total     = rng.uniform(total_lo, capacity)
            frac      = rng.dirichlet(np.ones(T))
            w[j, :]   = total * frac
    return w


# ---------------------------------------------------------------------------
# Main collection routine
# ---------------------------------------------------------------------------
def collect_training_data(
    params: Params,
    truth: DGPTruth,
    num_samples: int = 50_000,
    out_path: str = "data/recourse_training.npz",
    seed: int = 0,
    verbose_every: int = 2000,
    regimes: Optional[list[tuple[float, float]]] = None,
) -> dict:
    """
    Generate `num_samples` independent (y, w, d, Q) tuples across multiple
    sampling regimes, and save to `out_path`.

    Parameters
    ----------
    params : Params
    truth  : DGPTruth
        Produced by generate_historical_data; used to sample d from the DGP.
    num_samples : int
        Total budget across all regimes. Default 50_000.
    out_path : str
        Path to the .npz file. Parent directory is created if needed.
    seed : int
        Master RNG seed. (y, w) sampling uses `seed`; d sampling uses seed+1;
        regime-assignment uses seed+2.
    verbose_every : int
        Print progress every this many samples (0 to silence).
    regimes : list of (p_open, total_frac_min) tuples, optional
        The sampling regimes to draw from. If None, uses DEFAULT_REGIMES,
        which spans {0.3, 0.5, 0.7, 0.9, 1.0} x {0.0, 0.3, 0.6} = 15 regimes.
        Each regime gets roughly num_samples / len(regimes) samples.

    Returns
    -------
    summary : dict with num_samples, total_time, per-regime Q stats, etc.
    """
    I = params.num_areas
    J = params.num_locations
    T = params.num_food_types

    if regimes is None:
        regimes = DEFAULT_REGIMES
    regimes = list(regimes)
    num_regimes = len(regimes)

    rng_yw      = np.random.default_rng(seed)
    rng_regime  = np.random.default_rng(seed + 2)

    # Assign each of the M samples to a regime (shuffled, roughly-balanced).
    regime_idx = _assign_regimes(num_samples, num_regimes, rng_regime)

    # Pre-sample all d at once from the true DGP.
    scen = generate_scenarios(
        truth, N=num_samples, num_food_types=T,
        demand_min=params.demand_min, demand_max=params.demand_max,
        seed=seed + 1,
    )
    d_all = scen.d_real.astype(np.float32)

    # Allocate output buffers.
    y_buf = np.zeros((num_samples, J), dtype=np.int8)
    w_buf = np.zeros((num_samples, J, T), dtype=np.float32)
    Q_buf = np.zeros(num_samples, dtype=np.float32)
    r_buf = regime_idx.astype(np.int32)

    num_failures = 0
    t_start = perf_counter()
    solve_time_total = 0.0

    for m in range(num_samples):
        p_open, total_frac_min = regimes[r_buf[m]]
        y = sample_y(J, rng_yw, p_open=p_open)
        w = sample_w(y, T, params.capacity, rng_yw, total_frac_min=total_frac_min)
        d = d_all[m]

        q_val, _, _, rt = recourse_problem(y, w, d, params, time_used=0.0)
        solve_time_total += rt if rt is not None else 0.0

        if q_val is None:
            num_failures += 1
            q_val = 0.0

        y_buf[m] = y
        w_buf[m] = w
        Q_buf[m] = q_val

        if verbose_every and (m + 1) % verbose_every == 0:
            elapsed = perf_counter() - t_start
            rate = (m + 1) / elapsed
            eta  = (num_samples - m - 1) / rate
            print(
                f"  {m + 1:6d}/{num_samples}  "
                f"elapsed {elapsed:6.1f}s  "
                f"rate {rate:5.1f}/s  "
                f"eta {eta:6.1f}s  "
                f"failures {num_failures}"
            )

    total_time = perf_counter() - t_start

    if num_failures > 0:
        print(
            f"Warning: {num_failures}/{num_samples} recourse solves failed "
            f"(entries saved as Q=0)."
        )

    # Per-regime diagnostics
    regime_stats = []
    for r_i, (p_open, total_frac_min) in enumerate(regimes):
        mask = r_buf == r_i
        if mask.any():
            Qr = Q_buf[mask]
            regime_stats.append(dict(
                regime        = r_i,
                p_open        = p_open,
                total_frac_min= total_frac_min,
                count         = int(mask.sum()),
                Q_mean        = float(Qr.mean()),
                Q_std         = float(Qr.std()),
                Q_min         = float(Qr.min()),
                Q_max         = float(Qr.max()),
                Q_frac_neg    = float((Qr < 0).mean()),
            ))

    meta = dict(
        I              = I,
        J              = J,
        T              = T,
        capacity       = float(params.capacity),
        noise_std      = float(truth.noise_std),
        num_samples    = int(num_samples),
        num_failures   = int(num_failures),
        sampling       = "shao_style_multi_regime",
        regimes        = regimes,
        seed           = int(seed),
    )
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        y      = y_buf,
        w      = w_buf,
        d      = d_all,
        Q      = Q_buf,
        regime = r_buf,
        meta   = np.array(meta, dtype=object),
    )

    summary = dict(
        num_samples     = num_samples,
        num_regimes     = num_regimes,
        total_time      = total_time,
        mean_solve_time = solve_time_total / max(num_samples, 1),
        Q_mean          = float(Q_buf.mean()),
        Q_std           = float(Q_buf.std()),
        Q_min           = float(Q_buf.min()),
        Q_max           = float(Q_buf.max()),
        Q_frac_negative = float((Q_buf < 0).mean()),
        regime_stats    = regime_stats,
        out_path        = str(out),
    )
    return summary


def load_training_data(path: str = "data/recourse_training.npz") -> dict:
    """
    Load a .npz produced by collect_training_data.
    Returns a dict with keys 'y', 'w', 'd', 'Q', 'regime', 'meta'.
    """
    data = np.load(path, allow_pickle=True)
    out = dict(
        y    = data["y"],
        w    = data["w"],
        d    = data["d"],
        Q    = data["Q"],
        meta = data["meta"].item(),
    )
    if "regime" in data.files:
        out["regime"] = data["regime"]
    return out


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    I, J, T = 3, 6, 3
    pop_size = np.array([1, 2, 3])

    params = Params(
        num_areas         = I,
        num_locations     = J,
        num_food_types    = T,
        capacity          = 200.0,
        fixed_cost        = 50.0,
        demand_min        = 0.0,
        demand_max        = 400.0,
        num_scenarios_saa = 20,
        time_limit        = 60.0,
        seed              = 7,
    )

    _, truth = generate_historical_data(
        num_areas=I, num_food_types=T, num_days=90,
        pop_size=pop_size, seed=42,
    )

    print("Collecting recourse training data (multi-regime, 50k samples)...")
    summary = collect_training_data(
        params, truth, num_samples=50_000, seed=0,
        out_path=params.training_data_path,
    )

    print("\nOverall summary:")
    for k, v in summary.items():
        if k == "regime_stats":
            continue
        if isinstance(v, float):
            print(f"  {k:17s}: {v:.4f}")
        else:
            print(f"  {k:17s}: {v}")

    print("\nPer-regime Q statistics:")
    print(f"  {'regime':>6} {'p_open':>7} {'frac_min':>9} "
          f"{'count':>7} {'Q_mean':>10} {'Q_std':>9} "
          f"{'Q_min':>9} {'Q_max':>9} {'Q<0':>6}")
    for rs in summary["regime_stats"]:
        print(
            f"  {rs['regime']:>6d} {rs['p_open']:>7.2f} {rs['total_frac_min']:>9.2f} "
            f"{rs['count']:>7d} {rs['Q_mean']:>10.1f} {rs['Q_std']:>9.1f} "
            f"{rs['Q_min']:>9.1f} {rs['Q_max']:>9.1f} {rs['Q_frac_neg']:>6.1%}"
        )

    data = load_training_data(summary["out_path"])
    print(f"\nLoaded shapes: y={data['y'].shape}, w={data['w'].shape}, "
          f"d={data['d'].shape}, Q={data['Q'].shape}, regime={data['regime'].shape}")