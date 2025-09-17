"""


@author: Kreecha Puphaiboon

MIT License

Copyright (c) 2025 Kreecha Puphaiboon

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""
import math
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


class MultiArmedBandit:
    """
    Multi-Armed Bandit selection using UCB1 algorithm.
    Drop-in replacement for RouletteWheel with same interface.
    """

    def __init__(self, scores: List[float], decay: float, num_destroy: int, num_repair: int):
        # Keep same parameters for compatibility (though MAB doesn't use scores/decay)
        self._scores = scores[:4] if len(scores) >= 4 else [100.0, 50.0, 20.0, 5.0]
        self._decay = decay

        # MAB-specific data structures
        self._d_counts = np.zeros(num_destroy, dtype=int)
        self._r_counts = np.zeros(num_repair, dtype=int)
        self._d_values = np.zeros(num_destroy, dtype=float)
        self._r_values = np.zeros(num_repair, dtype=float)
        self._total_count = 0

        # Expose weights for compatibility (computed from MAB values)
        self._d_weights = np.ones(num_destroy, dtype=float)
        self._r_weights = np.ones(num_repair, dtype=float)

    @property
    def destroy_weights(self) -> np.ndarray:
        """Compatibility property - returns UCB values as weights"""
        return self._compute_ucb_values(self._d_counts, self._d_values)

    @property
    def repair_weights(self) -> np.ndarray:
        """Compatibility property - returns UCB values as weights"""
        return self._compute_ucb_values(self._r_counts, self._r_values)

    @property
    def scores(self) -> List[float]:
        """Compatibility property"""
        return self._scores

    def _compute_ucb_values(self, counts: np.ndarray, values: np.ndarray) -> np.ndarray:
        """Compute UCB1 values for arm selection"""
        if self._total_count == 0:
            return np.ones_like(counts, dtype=float)

        ucb_values = np.zeros_like(counts, dtype=float)
        for i in range(len(counts)):
            if counts[i] == 0:
                ucb_values[i] = float('inf')  # Ensure unplayed arms are selected
            else:
                confidence = math.sqrt((2 * math.log(self._total_count)) / counts[i])
                ucb_values[i] = values[i] + confidence

        return ucb_values

    def select_operators(self, rng: RandomState) -> Tuple[int, int]:
        """
        Select destroy and repair operators using UCB1 algorithm.
        Same interface as RouletteWheel.
        """

        def select_arm(counts, values):
            ucb_values = self._compute_ucb_values(counts, values)
            # If there are infinite values (unplayed arms), select randomly among them
            inf_indices = np.where(np.isinf(ucb_values))[0]
            if len(inf_indices) > 0:
                return rng.choice(inf_indices)
            else:
                return np.argmax(ucb_values)

        d_idx = select_arm(self._d_counts, self._d_values)
        r_idx = select_arm(self._r_counts, self._r_values)

        return d_idx, r_idx

    def update(self, d_idx: int, r_idx: int, outcome: int):
        """
        Update MAB based on outcome.
        Same interface as RouletteWheel.

        Parameters:
        - outcome: 0 (new global best), 1 (better than current), 2 (accepted), 3 (rejected)
        """
        # Convert outcome to reward (0-3 scale to 0-1 scale)
        reward = self._scores[outcome] / max(self._scores)

        # Update destroy operator
        self._d_counts[d_idx] += 1
        n_d = self._d_counts[d_idx]
        old_value_d = self._d_values[d_idx]
        self._d_values[d_idx] = ((n_d - 1) * old_value_d + reward) / n_d

        # Update repair operator
        self._r_counts[r_idx] += 1
        n_r = self._r_counts[r_idx]
        old_value_r = self._r_values[r_idx]
        self._r_values[r_idx] = ((n_r - 1) * old_value_r + reward) / n_r

        self._total_count += 1

        # Update weight properties for compatibility
        self._d_weights = self.destroy_weights
        self._r_weights = self.repair_weights