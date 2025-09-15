# -*- coding: utf-8 -*-
"""
Adaptive ALNS Destroy and Repair Operators for 3D Bin Packing
Features dynamic learning from problem size and recent performance
Adapted for 3D Bin Packing Problem using Extreme Point Heuristics


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

from abc import ABC
from collections import deque
from typing import List

import numpy as np

# Import validation framework
from src.classes.ValidateOperatorManager import ValidatedDestroyOperator
from src.classes.solution import Solution
from src.common import Item


class PerformanceTracker:
    """Tracks operator performance and adapts parameters"""

    def __init__(self, window_size=50, learning_rate=0.1):
        self.window_size = window_size
        self.learning_rate = learning_rate
        self.success_history = deque(maxlen=window_size)
        self.improvement_history = deque(maxlen=window_size)
        self.usage_count = 0

    def record_performance(self, success: bool, improvement: float = 0.0):
        """Record the performance of this operator"""
        self.success_history.append(success)
        self.improvement_history.append(improvement)
        self.usage_count += 1

    def get_success_rate(self) -> float:
        """Get recent success rate"""
        if not self.success_history:
            return 0.5  # Default neutral rate
        return sum(self.success_history) / len(self.success_history)

    def get_avg_improvement(self) -> float:
        """Get average improvement score"""
        if not self.improvement_history:
            return 0.0
        return sum(self.improvement_history) / len(self.improvement_history)

    def should_increase_intensity(self) -> bool:
        """Determine if operator should be more aggressive"""
        success_rate = self.get_success_rate()
        return success_rate < 0.3  # Low success rate

    def should_decrease_intensity(self) -> bool:
        """Determine if operator should be less aggressive"""
        success_rate = self.get_success_rate()
        return success_rate > 0.7  # High success rate


class AdaptiveDestroyBase(ValidatedDestroyOperator, ABC):
    """Base class for adaptive destroy operators"""

    def __init__(self, name: str, base_min_percent=0.05, base_max_percent=0.25):
        super().__init__(name)
        self.base_min_percent = base_min_percent
        self.base_max_percent = base_max_percent
        self.performance_tracker = PerformanceTracker()
        self.current_min_percent = base_min_percent
        self.current_max_percent = base_max_percent

    def adapt_parameters(self, total_items: int, solution_quality: float = None):
        """Adapt removal parameters based on performance and problem state"""
        # Adapt based on performance
        if self.performance_tracker.should_increase_intensity():
            # Increase removal intensity for poor performance
            self.current_min_percent = min(self.base_min_percent * 1.2, 0.4)
            self.current_max_percent = min(self.base_max_percent * 1.3, 0.6)
        elif self.performance_tracker.should_decrease_intensity():
            # Decrease intensity for good performance
            self.current_min_percent = max(self.base_min_percent * 0.8, 0.02)
            self.current_max_percent = max(self.base_max_percent * 0.9, 0.1)
        else:
            # Gradually return to base parameters
            self.current_min_percent = (self.current_min_percent + self.base_min_percent) / 2
            self.current_max_percent = (self.current_max_percent + self.base_max_percent) / 2

        # Calculate actual removal amounts
        min_remove = max(1, int(total_items * self.current_min_percent))
        max_remove = max(min_remove + 1, int(total_items * self.current_max_percent))

        return min_remove, max_remove


class AdaptiveUtilizationBasedDestroy(AdaptiveDestroyBase):
    """
    Destroys items from bins with poor utilization to encourage better packing.
    Adapts removal intensity based on recent performance.
    """

    def __init__(self):
        super().__init__("Adaptive Utilization Based Destroy",
                         base_min_percent=0.08, base_max_percent=0.25)
        self.utilization_threshold = 0.6  # Target bins with < 60% utilization

    def _destroy_implementation(self, solution: Solution, rnd_state: np.random.RandomState) -> List[Item]:
        """Remove items from poorly utilized bins"""
        if not solution.bins:
            return []

        # Count total items
        total_items = len(solution.get_all_items())
        if total_items == 0:
            return []

        min_remove, max_remove = self.adapt_parameters(total_items)

        # Calculate bin utilizations and find poorly utilized bins
        utilizations = solution.get_bin_utilizations()
        poor_bins = []

        for i, util in enumerate(utilizations):
            if util < self.utilization_threshold and solution.bins[i]:
                poor_bins.append((i, util))

        # Sort by utilization (worst first)
        poor_bins.sort(key=lambda x: x[1])

        removed_items = []
        target_remove = rnd_state.randint(min_remove, max_remove + 1)

        # Remove items from poorest bins first
        for bin_idx, util in poor_bins:
            if len(removed_items) >= target_remove:
                break

            bin_items = [pi.item for pi in solution.bins[bin_idx]]
            if not bin_items:
                continue

            # Remove random items from this poor bin
            items_to_remove = min(len(bin_items),
                                  target_remove - len(removed_items),
                                  max(1, len(bin_items) // 2))  # Don't empty bins completely

            selected_items = rnd_state.choice(bin_items, size=items_to_remove, replace=False)
            removed_items.extend(selected_items)

        # If still need more items, remove from random bins
        while len(removed_items) < target_remove:
            remaining_items = solution.get_all_items()
            if not remaining_items or len(removed_items) >= len(remaining_items):
                break

            item = rnd_state.choice(remaining_items)
            if item not in removed_items:
                removed_items.append(item)

        solution.remove_items(removed_items)
        # print(f"{self.name}: Removed {len(removed_items)} items from {len(poor_bins)} poorly utilized bins")
        return removed_items


class AdaptiveLargeItemDestroy(AdaptiveDestroyBase):
    """
    Destroys large items that might be blocking better arrangements.
    Focuses on items that consume significant space but may be poorly placed.
    """

    def __init__(self):
        super().__init__("Adaptive Large Item Destroy",
                         base_min_percent=0.06, base_max_percent=0.20)
        self.size_percentile = 70  # Target items in top 30% by volume

    def _destroy_implementation(self, solution: Solution, rnd_state: np.random.RandomState) -> List[Item]:
        """Remove large items that might be constraining the solution"""
        if not solution.bins:
            return []

        # Collect all items with their volumes and bin info
        all_items_info = []
        for bin_idx, bin_items in enumerate(solution.bins):
            for placed_item in bin_items:
                volume = placed_item.item.volume
                all_items_info.append((placed_item.item, volume, bin_idx))

        if not all_items_info:
            return []

        total_items = len(all_items_info)
        min_remove, max_remove = self.adapt_parameters(total_items)

        # Find volume threshold for large items
        volumes = [volume for _, volume, _ in all_items_info]
        volume_threshold = np.percentile(volumes, self.size_percentile)

        # Identify large items
        large_items = [(item, bin_idx) for item, volume, bin_idx in all_items_info
                       if volume >= volume_threshold]

        if not large_items:
            # Fallback: remove random items if no large items
            all_items = [item for item, _, _ in all_items_info]
            target_remove = rnd_state.randint(min_remove, max_remove + 1)
            items_to_remove = rnd_state.choice(all_items,
                                               size=min(target_remove, len(all_items)),
                                               replace=False).tolist()
            solution.remove_items(items_to_remove)
            return items_to_remove

        # Prioritize large items in underutilized bins
        def item_priority(item_bin_pair):
            item, bin_idx = item_bin_pair
            utilizations = solution.get_bin_utilizations()
            utilization = utilizations[bin_idx] if bin_idx < len(utilizations) else 1.0
            volume = item.volume
            # Higher priority for large items in poorly utilized bins
            return volume * (1.0 / (utilization + 0.1))  # Avoid division by zero

        large_items.sort(key=item_priority, reverse=True)

        # Remove items
        target_remove = rnd_state.randint(min_remove, max_remove + 1)
        items_to_remove = [item for item, _ in large_items[:target_remove]]

        solution.remove_items(items_to_remove)
        # print(f"{self.name}: Removed {len(items_to_remove)} large items (threshold: {volume_threshold:.2f})")
        return items_to_remove


class AdaptiveWorstBinDestroy(AdaptiveDestroyBase):
    """
    Destroys items from the worst-performing bins based on utilization and efficiency.
    Completely empties selected bad bins to allow for complete reorganization.
    """

    def __init__(self):
        super().__init__("Adaptive Worst Bin Destroy",
                         base_min_percent=0.15, base_max_percent=0.40)
        self.min_bins_to_target = 1
        self.max_bins_ratio = 0.3  # Target at most 30% of bins

    def _destroy_implementation(self, solution: Solution, rnd_state: np.random.RandomState) -> List[Item]:
        """Remove all items from the worst-performing bins"""
        if not solution.bins:
            return []

        # Evaluate bin performance
        utilizations = solution.get_bin_utilizations()
        bin_scores = []

        for i, (bin_items, util) in enumerate(zip(solution.bins, utilizations)):
            if not bin_items:
                continue

            item_count = len(bin_items)

            # Score combines low utilization with sparse packing
            # Lower score = worse bin (higher priority for removal)
            score = util * (1.0 + item_count / 10.0)  # Favor removing sparsely packed bins
            bin_scores.append((i, score, util))

        if not bin_scores:
            return []

        # Sort by score (worst first)
        bin_scores.sort(key=lambda x: x[1])

        total_items = len(solution.get_all_items())
        min_remove, max_remove = self.adapt_parameters(total_items)

        # Determine how many bins to target
        max_bins_to_target = max(1, int(len(bin_scores) * self.max_bins_ratio))

        removed_items = []
        bins_emptied = 0

        for bin_idx, score, utilization in bin_scores:
            if bins_emptied >= max_bins_to_target or len(removed_items) >= max_remove:
                break

            # Empty this bin completely
            bin_items = [pi.item for pi in solution.bins[bin_idx]]
            removed_items.extend(bin_items)
            bins_emptied += 1

            # Stop if we've removed enough items
            if len(removed_items) >= min_remove:
                break

        solution.remove_items(removed_items)
        # print(f"{self.name}: Emptied {bins_emptied} worst bins, removed {len(removed_items)} items")
        return removed_items


def get_adaptive_destroy_operators():
    """
    Factory function to get all adaptive destroy operators
    Returns: List of destroy operators
    """

    destroy_operators = [
        AdaptiveUtilizationBasedDestroy(),
        AdaptiveLargeItemDestroy(),
        AdaptiveWorstBinDestroy()
    ]

    return destroy_operators


if __name__ == "__main__":

    destroy_ops = get_adaptive_destroy_operators()

    print("Adaptive Destroy Operators:")
    for op in destroy_ops:
        print(f"  - {op.name}")
