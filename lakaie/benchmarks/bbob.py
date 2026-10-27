"""COCO BBOB functions f1-f24 (exploratory Path A extension, experiments/MPHBS_BBOB_ADDENDUM.md).

Uses the official COCO implementation (`coco-experiment`, module `cocoex`) through `cocoex.BareProblem`, which
exposes the optimum value of each (function, dimension, instance). The callable returns the error f(x) - f*.
"""
from __future__ import annotations

import numpy as np

BOUNDS = (-5.0, 5.0)
N_INSTANCES = 15


def instance_for_run(run: int) -> int:
    """Run r (1-based) uses BBOB instance ((r - 1) mod 15) + 1 (addendum)."""
    return (int(run) - 1) % N_INSTANCES + 1


class BBOBProblem:
    """Callable returning the error f(x) - f* of BBOB function `func_id` in dimension `dim`, instance `instance`."""

    def __init__(self, func_id: int, dim: int, instance: int):
        import cocoex          # imported here so the rest of the package does not depend on COCO
        if not 1 <= func_id <= 24:
            raise ValueError(f"BBOB function id must be 1..24, got {func_id}")
        self.func_id, self.dim, self.instance = int(func_id), int(dim), int(instance)
        self._p = cocoex.BareProblem("bbob", self.func_id, self.dim, self.instance)
        self.optimum = float(self._p.best_value())
        self.lb = np.full(self.dim, BOUNDS[0])
        self.ub = np.full(self.dim, BOUNDS[1])
        self.name = f"f{self.func_id}_i{self.instance}"
        self.benchmark = f"BBOB-D{self.dim}"

    def __call__(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        return np.fromiter((self._p(x) for x in X), dtype=np.float64, count=X.shape[0]) - self.optimum
