"""
Scenario generator for the food-truck C-2SSP.

Design philosophy
-----------------
The scenario generator is a pure DGP sampler. It produces bundles of the
form
    (nominal, context=(weather, weekday), eps, d_real)
where:
    - `nominal` and `context` are observable to the decision maker at
      first-stage time,
    - `eps` is unobserved noise,
    - `d_real = nominal + alpha*weather + beta*weekday + eps` is the true
      realized demand, hidden from the SP at decision time but revealed
      for post-hoc evaluation.

Downstream methods (oracle / C-2SSP / non-contextual 2SSP) see different
pieces of the bundle, but they all consume the same (N, I, T) scenario
array produced by `scenarios_for_method`. That keeps saa() and ccg()
oblivious to which method is currently running.

Pipeline
--------
    truth       = data_generation.generate_historical_data(...)      # the DGP
    fit_model   = ols_fit.fit_demand_model(historical_df)            # [TBD]
    scenarios   = generate_scenarios(truth, N=..., seed=...)         # ScenarioSet
    # dispatch to a method:
    d_ctx   = scenarios_for_method(scenarios, "contextual",   model=fit_model)
    d_noctx = scenarios_for_method(scenarios, "noncontextual", mean_demand=...)
    d_orc   = scenarios_for_method(scenarios, "oracle")
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np

# DemandModel lives here alongside the projector; both are small and tightly
# coupled through the `scenarios_for_method(..., "contextual")` call.
# ---------------------------------------------------------------------------


@dataclass
class DemandModel:
    """
    Fitted linear demand model used by the C-2SSP method to turn observed
    context (weather, weekday) into a point prediction d_hat.

    The OLS fitter (ols_fit.py, to be written) produces this object from
    historical data. `build_oracle_model` below produces it from the true
    DGP parameters; both work transparently with scenarios_for_method.

    Shape convention: all arrays are (I, T).
        d_hat_{i,t}(weather, weekday) = intercept_{i,t}
                                      + alpha_{i,t}    * weather
                                      + beta_{i,t}     * weekday.

    residual_std is stored for completeness (it lets downstream code resample
    an eps if needed) but is NOT used by scenarios_for_method — the method-
    specific projection uses point predictions only.
    """
    intercept:    np.ndarray
    alpha:        np.ndarray
    beta:         np.ndarray
    residual_std: np.ndarray

    def __post_init__(self):
        self.intercept    = np.asarray(self.intercept,    dtype=float)
        self.alpha        = np.asarray(self.alpha,        dtype=float)
        self.beta         = np.asarray(self.beta,         dtype=float)
        self.residual_std = np.asarray(self.residual_std, dtype=float)
        if not (self.intercept.shape == self.alpha.shape == self.beta.shape):
            raise ValueError(
                "intercept, alpha, beta must all have shape (I, T); got "
                f"{self.intercept.shape}, {self.alpha.shape}, {self.beta.shape}"
            )

    @property
    def num_areas(self) -> int:
        return self.intercept.shape[0]

    @property
    def num_food_types(self) -> int:
        return self.intercept.shape[1]

    def predict(self, weather: np.ndarray, weekday: np.ndarray) -> np.ndarray:
        """
        Vectorized point prediction. weather/weekday are (N,); returns (N, I, T).
        """
        wx = np.asarray(weather, dtype=float)[:, None, None]
        wd = np.asarray(weekday, dtype=float)[:, None, None]
        return (
            self.intercept[None, :, :]
            + self.alpha[None, :, :] * wx
            + self.beta [None, :, :] * wd
        )


@dataclass
class ScenarioSet:
    """
    Bundle of N scenarios. All arrays are numpy; shapes noted below.

    Fields
    ------
    weather : (N,)       realized weather context, 0/1
    weekday : (N,)       realized weekday context, 0/1
    eps     : (N, I, T)  realized noise, hidden from the SP
    d_real  : (N, I, T)  realized demand = nominal + alpha*wx + beta*wd + eps,
                         clipped to [demand_min, demand_max]
    nominal : (I,)       persistent baseline demand per area, observable
    """
    weather: np.ndarray
    weekday: np.ndarray
    eps:     np.ndarray
    d_real:  np.ndarray
    nominal: np.ndarray

    @property
    def N(self) -> int:
        return self.weather.shape[0]

    @property
    def num_areas(self) -> int:
        return self.d_real.shape[1]

    @property
    def num_food_types(self) -> int:
        return self.d_real.shape[2]


# ---------------------------------------------------------------------------
# Helper 1: sample from the DGP
# ---------------------------------------------------------------------------
def generate_scenarios(
    truth,                                    # DGPTruth instance
    N: int,
    num_food_types: int,
    weekday_prob: float = 5.0 / 7.0,
    sunny_prob: float = 0.6,
    demand_min: float = 0.0,
    demand_max: float = 300.0,
    weather: Optional[np.ndarray] = None,
    weekday: Optional[np.ndarray] = None,
    seed: Optional[int] = 227,
) -> ScenarioSet:
    """
    Draw N scenarios from the true DGP.

    Mirrors `dataset.data_generation.generate_historical_data` exactly,
    including the optional surge / heteroscedasticity recorded on `truth`.
    For each scenario n = 1, ..., N:
        1. Sample (weather^n, weekday^n) from Bernoulli marginals, unless
           overridden via the `weather` / `weekday` arguments.
        2. Compute mean_d^n = (nominal + alpha*weather^n + beta*weekday^n)
                              * (1 + (surge_factor - 1) * weather^n*weekday^n).
        3. Sample eps^n ~ N(0, sigma^n) where sigma^n = noise_std (homoscedastic)
           or sigma^n = noise_std + noise_scale_factor * mean_d^n (heteroscedastic).
        4. Set d_real^n = mean_d^n + eps^n, clipped to [demand_min, demand_max].
    With `surge_factor=1.0` and `heteroscedastic=False` (the DGPTruth defaults)
    this reduces to the original additive-Gaussian DGP.

    Parameters
    ----------
    truth : DGPTruth
        Ground-truth DGP parameters produced by `generate_historical_data`.
    N : int
        Number of scenarios.
    num_food_types : int
        T. Passed explicitly because DGPTruth stores per-area (length-I)
        coefficients that are shared across food types under option A.
    weekday_prob, sunny_prob : float
        Bernoulli marginals for context sampling.
    demand_min, demand_max : float
        Clip bounds (must match the Params used by the solver to avoid
        feasibility mismatches with complete recourse).
    weather, weekday : (N,) arrays, optional
        Pre-specified context sequences. When provided, used verbatim and
        the corresponding random draws are skipped. This is the right way
        to compare methods on EXACTLY the same scenarios.
    seed : int, optional
        RNG seed; affects only the pieces not passed in explicitly.

    Returns
    -------
    ScenarioSet
    """
    rng = np.random.default_rng(seed)
    I = len(truth.nominal)
    T = num_food_types

    # --- context realizations ---------------------------------------------
    if weather is None:
        weather = rng.binomial(1, sunny_prob, size=N)
    else:
        weather = np.asarray(weather, dtype=int)
        if weather.shape != (N,):
            raise ValueError(f"weather must have shape ({N},), got {weather.shape}")

    if weekday is None:
        weekday = rng.binomial(1, weekday_prob, size=N)
    else:
        weekday = np.asarray(weekday, dtype=int)
        if weekday.shape != (N,):
            raise ValueError(f"weekday must have shape ({N},), got {weekday.shape}")

    # --- realized demand under the TRUE DGP --------------------------------
    # Mirror the structure of dataset.data_generation.generate_historical_data
    # exactly, including the optional surge and heteroscedasticity. Reading
    # the heterogeneity knobs off `truth` keeps deployment scenarios aligned
    # with whatever DGP produced the historical data the predictor was fit on.
    wx = weather[:, None, None].astype(float)                    # (N, 1, 1)
    wd = weekday[:, None, None].astype(float)                    # (N, 1, 1)
    nominal_b = truth.nominal[None, :, None]                     # (1, I, 1)
    alpha_b   = truth.alpha  [None, :, None]                     # (1, I, 1)
    beta_b    = truth.beta   [None, :, None]                     # (1, I, 1)

    mean_d = nominal_b + alpha_b * wx + beta_b * wd               # (N, I, 1)
    surge_factor       = getattr(truth, "surge_factor", 1.0)
    heteroscedastic    = getattr(truth, "heteroscedastic", False)
    noise_scale_factor = getattr(truth, "noise_scale_factor", 0.0)

    surge_mask = wx * wd                                          # (N, 1, 1)
    mean_d = mean_d * (1.0 + (surge_factor - 1.0) * surge_mask)
    mean_d = np.broadcast_to(mean_d, (N, I, T))

    if heteroscedastic:
        sigma = truth.noise_std + noise_scale_factor * mean_d     # (N, I, T)
    else:
        sigma = truth.noise_std                                    # scalar
    eps = rng.normal(0.0, 1.0, size=(N, I, T)) * sigma

    d_real = mean_d + eps                                          # (N, I, T)
    d_real = np.clip(d_real, demand_min, demand_max)

    return ScenarioSet(
        weather=weather,
        weekday=weekday,
        eps=eps,
        d_real=d_real,
        nominal=np.asarray(truth.nominal, dtype=float),
    )


# ---------------------------------------------------------------------------
# Helper 2: project a ScenarioSet into the (N, I, T) array the solver consumes
# ---------------------------------------------------------------------------
def scenarios_for_method(
    scenarios: ScenarioSet,
    method: str,
    model=None,
    mean_demand: Optional[np.ndarray] = None,
    training_residuals: Optional[np.ndarray] = None,
    demand_min: float = 0.0,
    demand_max: float = 300.0,
    residual_seed: Optional[int] = None,
) -> np.ndarray:
    """
    Project a ScenarioSet to the (N, I, T) demand array each method solves with.

    method="oracle"
        Uses d_real directly — perfect information about realized demand.
        Gives the unattainable lower bound on cost.

    method="contextual"
        Requires `model` (a DemandModel). For each n, applies the model to
        (weather^n, weekday^n) to produce d_hat^n = model.predict(...).
        This is the *conditional-mean* C-2SSP method. Note that with
        binary (weather, weekday), the image of this map has at most four
        distinct points, so the (N, I, T) array typically contains many
        duplicate rows.

    method="contextual_residual"
        Like "contextual", but adds a fresh Gaussian residual on top of
        each prediction:
            d^n = model.predict(weather^n, weekday^n)
                  + eps^n,    eps^n ~ N(0, model.residual_std^2)
        where model.residual_std is the OLS-estimated noise scale (NOT
        the true DGP noise; the method does not have access to it).
        This produces N distinct scenarios while still respecting the
        contextual structure, and is the standard "predict the
        conditional distribution, then optimize" CSO setup. Use
        `residual_seed` to fix the residual draws across runs.

    method="contextual_er"
        Empirical-residual bootstrap (Sun et al. / Kannan-Bayraksan-
        Luedtke ER-SAA). Same as "contextual_residual" but the noise is
        sampled with replacement from a precomputed array of training
        residuals instead of drawn from a Gaussian:
            d^n = model.predict(weather^n, weekday^n) + eps_{k(n)}
        where k(n) ~ Uniform{0, ..., n_train - 1} and eps_k is the
        k-th training residual (the (I, T) vector observed on day k).
        Bootstrapping whole (I, T) vectors preserves any cross-(i, t)
        residual correlation that Gaussian sampling collapses. Requires
        `training_residuals` of shape (n_train, I, T).

    method="noncontextual"
        Requires `mean_demand`, a shape-(I, T) array interpreted as the
        method's belief about demand. Broadcast to (N, I, T). Typically
        either:
            - the unconditional historical mean d_bar_{i,t}, or
            - the nominal vector broadcast across food types.
        The method ignores (weather, weekday) entirely.

    All projections are clipped to [demand_min, demand_max] for consistency
    with the two-stage problem's complete-recourse assumption.

    Parameters
    ----------
    scenarios : ScenarioSet
    method : {"oracle", "contextual", "contextual_residual",
              "contextual_er", "noncontextual"}
    model : predictor with .predict(weather, weekday) -> (N, I, T);
        required if method in {"contextual", "contextual_residual",
        "contextual_er"}. For "contextual_residual" must also expose
        `.residual_std`. Linear and nonparametric variants both work.
    mean_demand : (I, T) array, required if method == "noncontextual"
    training_residuals : (n_train, I, T) array, required if
        method == "contextual_er". Typically produced by
        `learning.residuals.compute_training_residuals(model, df, T)`.
    demand_min, demand_max : float
    residual_seed : int, optional
        RNG seed for the residual draws under method="contextual_residual"
        or "contextual_er"; ignored otherwise. None falls back to system
        entropy.

    Returns
    -------
    d : (N, I, T) array, ready to pass to saa() or ccg().
    """
    N = scenarios.N
    I = scenarios.num_areas
    T = scenarios.num_food_types

    if method == "oracle":
        d = scenarios.d_real                                   # already clipped

    elif method == "contextual":
        if model is None:
            raise ValueError("method='contextual' requires a DemandModel via `model=`")
        if (model.num_areas, model.num_food_types) != (I, T):
            raise ValueError(
                f"DemandModel shape ({model.num_areas}, {model.num_food_types}) "
                f"does not match scenarios ({I}, {T})"
            )
        d = model.predict(scenarios.weather, scenarios.weekday)  # (N, I, T)

    elif method == "contextual_residual":
        if model is None:
            raise ValueError(
                "method='contextual_residual' requires a DemandModel via `model=`"
            )
        if (model.num_areas, model.num_food_types) != (I, T):
            raise ValueError(
                f"DemandModel shape ({model.num_areas}, {model.num_food_types}) "
                f"does not match scenarios ({I}, {T})"
            )
        d_hat = model.predict(scenarios.weather, scenarios.weekday)  # (N, I, T)
        # Broadcast residual_std to (I, T); per-(i, t) noise scale.
        sigma = np.broadcast_to(np.asarray(model.residual_std, dtype=float),
                                (I, T))
        rng = np.random.default_rng(residual_seed)
        eps = rng.normal(0.0, 1.0, size=(N, I, T)) * sigma[None, :, :]
        d = d_hat + eps

    elif method == "contextual_er":
        if model is None:
            raise ValueError(
                "method='contextual_er' requires a predictor via `model=`"
            )
        if (model.num_areas, model.num_food_types) != (I, T):
            raise ValueError(
                f"predictor shape ({model.num_areas}, {model.num_food_types}) "
                f"does not match scenarios ({I}, {T})"
            )
        if training_residuals is None:
            raise ValueError(
                "method='contextual_er' requires `training_residuals` "
                "of shape (n_train, I, T); compute via "
                "learning.residuals.compute_training_residuals(...)"
            )
        residuals = np.asarray(training_residuals, dtype=float)
        if residuals.ndim != 3 or residuals.shape[1:] != (I, T):
            raise ValueError(
                f"training_residuals must have shape (n_train, {I}, {T}); "
                f"got {residuals.shape}"
            )
        n_train = residuals.shape[0]
        d_hat = model.predict(scenarios.weather, scenarios.weekday)  # (N, I, T)
        rng = np.random.default_rng(residual_seed)
        idx = rng.integers(0, n_train, size=N)
        d = d_hat + residuals[idx]                                   # (N, I, T)

    elif method == "noncontextual":
        if mean_demand is None:
            raise ValueError(
                "method='noncontextual' requires a (I, T) `mean_demand` array"
            )
        mean_demand = np.asarray(mean_demand, dtype=float)
        if mean_demand.shape != (I, T):
            raise ValueError(
                f"mean_demand must have shape ({I}, {T}); got {mean_demand.shape}"
            )
        d = np.broadcast_to(mean_demand[None, :, :], (N, I, T)).copy()

    else:
        raise ValueError(
            f"Unknown method '{method}'. Use 'oracle', 'contextual', "
            "'contextual_residual', 'contextual_er', or 'noncontextual'."
        )

    return np.clip(d, demand_min, demand_max)


# ---------------------------------------------------------------------------
# Helper 3: build an oracle DemandModel directly from DGPTruth
# (handy as a correctness baseline before OLS is wired up)
# ---------------------------------------------------------------------------
def build_oracle_model(truth, num_food_types: int) -> DemandModel:
    """
    Build a DemandModel from the true DGP parameters. Useful for:
        1. Sanity-testing the pipeline without the OLS fitter in place.
        2. As a ceiling for the C-2SSP method: any OLS-based DemandModel
           should do no better than this one.
    """
    I = len(truth.nominal)
    T = num_food_types
    return DemandModel(
        intercept    = np.tile(truth.nominal[:, None], (1, T)),
        alpha        = np.tile(truth.alpha  [:, None], (1, T)),
        beta         = np.tile(truth.beta   [:, None], (1, T)),
        residual_std = np.full((I, T), truth.noise_std),
    )


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Mock a DGPTruth so this file is runnable on its own.
    @dataclass
    class _MockTruth:
        nominal:   np.ndarray
        alpha:     np.ndarray
        beta:      np.ndarray
        noise_std: float

    pop_size = np.array([1, 3, 2, 2, 2])
    truth = _MockTruth(
        nominal   = pop_size * 70.0,
        alpha     = pop_size * 10.0,
        beta     = pop_size * 8.0,
        noise_std = 10.0,
    )
    T = 2
    N = 500

    # --- Helper 1: sample ScenarioSet --------------------------------------
    scen = generate_scenarios(truth, N=N, num_food_types=T, seed=0)
    print(f"ScenarioSet: N={scen.N}, I={scen.num_areas}, T={scen.num_food_types}")
    print(f"  empirical Pr(sunny)   = {scen.weather.mean():.3f}  (nominal 0.600)")
    print(f"  empirical Pr(weekday) = {scen.weekday.mean():.3f}  (nominal 0.714)")
    print(f"  eps std               = {scen.eps.std():.3f}        "
          f"(true {truth.noise_std:.2f})")
    print(f"  d_real range          = [{scen.d_real.min():.2f}, "
          f"{scen.d_real.max():.2f}]\n")

    # --- Helper 3 + Helper 2: oracle projection ----------------------------
    oracle_model = build_oracle_model(truth, num_food_types=T)
    d_oracle    = scenarios_for_method(scen, "oracle")
    d_ctx_orc   = scenarios_for_method(scen, "contextual", model=oracle_model)

    # d_oracle uses realized demand; d_ctx_orc uses the oracle model's mean
    # prediction, i.e. d_real - eps (up to clipping). Their difference should
    # be approximately eps in distribution.
    diff = d_oracle - d_ctx_orc
    print("Sanity: oracle model's prediction error should match true noise")
    print(f"  (d_oracle - d_contextual_oracle).std  = {diff.std():.3f}")
    print(f"  (d_oracle - d_contextual_oracle).mean = {diff.mean():+.3f}\n")

    # --- Non-contextual projection -----------------------------------------
    # Build a mean-demand vector: unconditional historical mean, approximated
    # here by nominal + alpha*E[weather] + beta*E[weekday] per (i, t).
    I = len(truth.nominal)
    mean_per_area = (
        truth.nominal
        + truth.alpha * 0.6         # sunny_prob
        + truth.beta * (5.0 / 7.0)  # weekday_prob
    )
    mean_demand = np.tile(mean_per_area[:, None], (1, T))       # (I, T)
    d_noctx = scenarios_for_method(scen, "noncontextual", mean_demand=mean_demand)

    print("Non-contextual projection: all N scenarios should be identical")
    print(f"  std of d_noncontextual across N = {d_noctx.std(axis=0).max():.4e} "
          f"(should be 0)")
    print(f"  mean of d_noncontextual         = {d_noctx.mean():.2f}")
    print(f"  mean of d_real                  = {scen.d_real.mean():.2f}  "
          f"(should agree on average)\n")

    # --- Method output shapes ---------------------------------------------
    print("Output shapes (all should be (N, I, T)):")
    print(f"  oracle:         {d_oracle.shape}")
    print(f"  contextual:     {d_ctx_orc.shape}")
    print(f"  noncontextual:  {d_noctx.shape}")
