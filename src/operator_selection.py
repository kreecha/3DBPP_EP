# Add this to AlnsExtremePoint.py (or create a new file, e.g., operator_selection.py)
from typing import List, Tuple
import numpy as np
from numpy.random import RandomState


class RouletteWheel:
    """
    Roulette wheel selection scheme for ALNS operator selection.
    Updates operator weights based on four outcomes: new global best, better than current,
    accepted, or rejected, using scores and a decay parameter.
    Adapted from Wouda's ALNS framework.

    Parameters
    ----------
    scores
        List of four non-negative floats for outcomes: [new global best, better than current, accepted, rejected]
    decay
        Operator decay parameter (theta in [0, 1]) for weight updates
    num_destroy
        Number of destroy operators
    num_repair
        Number of repair operators
    """

    def __init__(self, scores: List[float], decay: float, num_destroy: int, num_repair: int):
        if any(score < 0 for score in scores):
            raise ValueError("Negative scores are not understood.")
        if len(scores) < 4:
            raise ValueError(f"Expected four scores, found {len(scores)}")
        if not (0 <= decay <= 1):
            raise ValueError("decay outside [0, 1] not understood.")

        self._scores = scores[:4]  # Use first four scores
        self._decay = decay
        self._d_weights = np.ones(num_destroy, dtype=float)
        self._r_weights = np.ones(num_repair, dtype=float)

    @property
    def destroy_weights(self) -> np.ndarray:
        return self._d_weights

    @property
    def scores(self) -> List[float]:
        return self._scores

    @property
    def repair_weights(self) -> np.ndarray:
        return self._r_weights

    def select_operators(self, rng: RandomState) -> Tuple[int, int]:
        """
        Selects a destroy and repair operator pair based on normalized weights.

        Parameters
        ----------
        rng
            Random number generator (RandomState)

        Returns
        -------
        Tuple of (d_idx, r_idx) for destroy and repair operator indices
        """

        def select(weights):
            if np.sum(weights) == 0:
                return rng.randint(len(weights))
            probs = weights / np.sum(weights)
            return rng.choice(len(weights), p=probs)

        d_idx = select(self._d_weights)
        r_idx = select(self._r_weights)
        return d_idx, r_idx

    def update(self, d_idx: int, r_idx: int, outcome: int):
        """
        Updates weights for the selected destroy and repair operators based on the outcome.

        Parameters
        ----------
        d_idx
            Index of the destroy operator
        r_idx
            Index of the repair operator
        outcome
            Outcome index: 0 (new global best), 1 (better than current), 2 (accepted), 3 (rejected)
        """
        self._d_weights[d_idx] = self._decay * self._d_weights[d_idx] + (1 - self._decay) * self._scores[outcome]
        self._r_weights[r_idx] = self._decay * self._r_weights[r_idx] + (1 - self._decay) * self._scores[outcome]