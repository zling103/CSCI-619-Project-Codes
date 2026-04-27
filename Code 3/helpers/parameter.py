from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class Params:
    """
    Parameter container for the contextual two-stage stochastic programming
    (C-2SSP) food-truck problem.

    Sets:
        num_areas (I):  number of demand regions
        num_locations (J):     number of potential food-truck locations (J >= I)
        num_food_types (T):    number of food types

    Cost / capacity parameters:
        capacity (C):           capacity of a food truck
        fixed_cost (c^f):       fixed cost of opening a food truck
        acquisition_cost (c^a): per-unit cost of acquiring food type t, shape (T,)
        unmet_penalty (c^u):    per-unit penalty for unmet demand, shape (T,)
        revenue (c^r):          per-unit revenue for served demand, shape (T,)

    Eligibility:
        e:                      binary matrix of shape (I, J); e[i, j] = 1 iff
                                a truck at location j serves demand region i.
                                Each column sums to 1 (one region per truck).
                                Default: contiguous even partition of J across I.

    Demand support (d in [d_min, d_max] ensures complete recourse):
        demand_min (d^min): lower bound on demand
        demand_max (d^max): upper bound on demand

    Context (for contextual SAA; optional):
        context_dim: dimensionality of the context vector x

    SAA / solver options:
        num_scenarios_saa:  N, the number of sampled scenarios used by SAA/CCG
        time_limit:         Gurobi time limit (seconds)
        seed:               random seed used to build the default cost vectors
    """

    # Core sets
    num_areas: int = 3
    num_locations: int = 6
    num_food_types: int = 3

    # Scalar cost / capacity
    capacity: float = 100.0
    fixed_cost: float = 50.0

    # Cost vectors (one entry per food type); auto-generated if not supplied
    acquisition_cost: Optional[np.ndarray] = None
    unmet_penalty: Optional[np.ndarray] = None
    revenue: Optional[np.ndarray] = None

    # Eligibility matrix e (I, J); auto-generated if not supplied
    e: Optional[np.ndarray] = None

    # Demand support
    demand_min: float = 0.0
    demand_max: float = 300.0

    # Context (optional)
    context_dim: int = 5

    # SAA / solver
    num_scenarios_saa: int = 50
    time_limit: float = 7200.0
    seed: int = 27

    def __post_init__(self):
        if self.num_locations < self.num_areas:
            raise ValueError("num_locations must be >= num_areas (each region needs at least one eligible truck location)")

        rng = np.random.default_rng(self.seed)
        if self.acquisition_cost is None:
            self.acquisition_cost = rng.uniform(1.0, 3.0, size=self.num_food_types)
        if self.unmet_penalty is None:
            self.unmet_penalty = rng.uniform(5.0, 10.0, size=self.num_food_types)
        if self.revenue is None:
            self.revenue = rng.uniform(4.0, 8.0, size=self.num_food_types)

        self.acquisition_cost = np.asarray(self.acquisition_cost, dtype=float)
        self.unmet_penalty = np.asarray(self.unmet_penalty, dtype=float)
        self.revenue = np.asarray(self.revenue, dtype=float)

        if self.acquisition_cost.shape != (self.num_food_types,):
            raise ValueError("acquisition_cost must have shape (num_food_types,)")
        if self.unmet_penalty.shape != (self.num_food_types,):
            raise ValueError("unmet_penalty must have shape (num_food_types,)")
        if self.revenue.shape != (self.num_food_types,):
            raise ValueError("revenue must have shape (num_food_types,)")

        # Default eligibility e (I, J): contiguous even partition of J into I groups
        if self.e is None:
            groups = np.array_split(np.arange(self.num_locations), self.num_areas)
            e = np.zeros((self.num_areas, self.num_locations), dtype=int)
            for i, js in enumerate(groups):
                e[i, js] = 1
            self.e = e

        self.e = np.asarray(self.e, dtype=int)
        if self.e.shape != (self.num_areas, self.num_locations):
            raise ValueError("e must have shape (num_areas, num_locations)")
        if not np.all(self.e.sum(axis=0) == 1):
            raise ValueError("each column of e must sum to 1 (each truck serves exactly one region)")

    @property
    def _size_tag(self) -> str:
        return f"{self.num_areas}I_{self.num_locations}J_{self.num_food_types}T"

    @property
    def training_data_path(self) -> str:
        """Path to the (y, w, d, Q) training tuples for this instance size."""
        return f"data/recourse_training_{self._size_tag}.npz"

    @property
    def surrogate_path(self) -> str:
        """Path to the trained Q-surrogate weights for this instance size."""
        return f"data/q_surrogate_{self._size_tag}.pt"
