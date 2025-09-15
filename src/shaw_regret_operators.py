# -*- coding: utf-8 -*-
"""
Adaptive Shaw, Regret, and K-Regret Operators for 3D Bin Packing ALNS

These are the foundational ALNS operators from:
- Shaw (1998): Original LNS with relatedness-based removal
- Ropke & Pisinger (2006): K-regret insertion with foresight

Adapted for 3D Bin Packing Problem using Extreme Point Heuristics
Now with adaptive percentage-based removal and performance tracking


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

import numpy as np
import math
import random
from typing import List, Tuple, Dict
from dataclasses import dataclass

# Import shared classes
from src.common import Item, Bin, PlacedItem, ExtremePoint, SortingRule, MeritFunction
from src.ExtremePointHeuristic import ExtremePointBinPacking3D
from src.classes.ValidateOperatorManager import ValidatedDestroyOperator, ValidatedRepairOperator
from src.classes.solution import Solution


# =============================================================================
# ADAPTIVE SHAW DESTROY OPERATORS (Original LNS Operators)
# =============================================================================

class AdaptiveShawDestroy(ValidatedDestroyOperator):
    """
    Adaptive Shaw's original relatedness-based removal operator (Shaw, 1998)
    Removes items that are similar/related according to multiple criteria
    Now with percentage-based adaptive removal
    """

    def __init__(self, name: str = 'AdaptiveShawDestroy',
                 min_rate: float = 0.05,
                 max_rate: float = 0.25,
                 randomization: float = 0.3):

        super().__init__(name)  # Initialize validation attributes
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.randomization = randomization  # Controls randomness in selection
        self.recent_improvements = []  # Track recent performance for adaptation

    def _destroy_implementation(self, solution: Solution, rnd_state: np.random.RandomState) -> List[Item]:
        """Remove related items based on Shaw's relatedness measure with adaptive removal"""
        all_items = solution.get_all_items()
        if len(all_items) < 1:  # No items to remove
            return []

        # Calculate adaptive removal rate based on recent performance
        if len(self.recent_improvements) >= 5:
            avg_improvement = np.mean(self.recent_improvements[-5:])
            removal_rate = self.max_rate if avg_improvement > 0 else self.min_rate
        else:
            removal_rate = (self.min_rate + self.max_rate) / 2

        # Calculate number of items to remove based on percentage
        num_to_remove = max(1, int(len(all_items) * removal_rate))
        num_to_remove = min(num_to_remove, len(all_items))

        # Select seed item randomly
        seed_item = rnd_state.choice(all_items)

        # Calculate relatedness for all other items, excluding seed
        items_to_remove_set = {seed_item}  # Track selected items to avoid duplicates
        items_to_remove = [seed_item]  # Initialize with seed item

        # Calculate relatedness scores for remaining items
        relatedness_scores = []
        for item in all_items:
            if item not in items_to_remove_set:
                relatedness = self._calculate_relatedness(seed_item, item, solution)
                relatedness_scores.append((relatedness, item))

        if not relatedness_scores:
            solution.remove_items(items_to_remove)
            return items_to_remove

        # Sort by relatedness (most related first)
        relatedness_scores.sort(key=lambda x: -x[0])

        # Select additional items to remove up to num_to_remove
        candidates = [(score, item) for score, item in relatedness_scores]
        while len(items_to_remove) < num_to_remove and candidates:
            if rnd_state.random() < (1 - self.randomization):
                # Select most related item
                selected_item = candidates[0][1]
                items_to_remove.append(selected_item)
                items_to_remove_set.add(selected_item)
                candidates.pop(0)  # Remove selected item from candidates
            else:
                # Random selection from top candidates
                top_k = min(len(candidates), 5)  # Consider top 5 candidates
                if top_k == 0:
                    break
                selected_idx = rnd_state.randint(0, top_k)
                selected_item = candidates[selected_idx][1]
                items_to_remove.append(selected_item)
                items_to_remove_set.add(selected_item)
                candidates.pop(selected_idx)  # Remove selected item from candidates

        solution.remove_items(items_to_remove)
        return items_to_remove

    def update_performance(self, improvement: float):
        """Update performance tracking for adaptation"""
        self.recent_improvements.append(improvement)
        # Keep only recent history
        if len(self.recent_improvements) > 10:
            self.recent_improvements = self.recent_improvements[-10:]

    def _calculate_relatedness(self, item1: Item, item2: Item, solution: Solution) -> float:
        """Calculate relatedness between two items using Shaw's criteria"""
        raise NotImplementedError("Subclasses must implement relatedness calculation")


class AdaptiveShawSizeDestroy(AdaptiveShawDestroy):
    """Adaptive Shaw destroy based on item size similarity"""

    def __init__(self, min_rate: float = 0.06, max_rate: float = 0.20):
        super().__init__("Adaptive Shaw Size Destroy", min_rate, max_rate)

    def _calculate_relatedness(self, item1: Item, item2: Item, solution: Solution) -> float:
        """Relatedness based on size similarity"""
        # Volume similarity
        vol_diff = abs(item1.volume - item2.volume) / max(item1.volume, item2.volume)
        vol_similarity = 1.0 - vol_diff

        # Dimensional similarity
        dim1 = [item1.width, item1.depth, item1.height]
        dim2 = [item2.width, item2.depth, item2.height]

        dim_similarity = 0
        for d1, d2 in zip(dim1, dim2):
            dim_diff = abs(d1 - d2) / max(d1, d2)
            dim_similarity += (1.0 - dim_diff)
        dim_similarity /= 3

        # Combine similarities
        return 0.6 * vol_similarity + 0.4 * dim_similarity


class AdaptiveShawSpatialDestroy(AdaptiveShawDestroy):
    """Adaptive Shaw destroy based on spatial proximity"""

    def __init__(self, min_rate: float = 0.08, max_rate: float = 0.22):
        super().__init__("Adaptive Shaw Spatial Destroy", min_rate, max_rate)

    def _calculate_relatedness(self, item1: Item, item2: Item, solution: Solution) -> float:
        """Relatedness based on spatial proximity"""
        # Find positions of items in solution
        pos1 = self._find_item_position(item1, solution)
        pos2 = self._find_item_position(item2, solution)

        if pos1 is None or pos2 is None:
            return 0.0

        # Same bin bonus
        same_bin = (pos1[0] == pos2[0])

        if same_bin:
            # Calculate spatial distance
            center1 = (pos1[1] + item1.width / 2, pos1[2] + item1.depth / 2, pos1[3] + item1.height / 2)
            center2 = (pos2[1] + item2.width / 2, pos2[2] + item2.depth / 2, pos2[3] + item2.height / 2)

            distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(center1, center2)))
            max_distance = math.sqrt(3 * (100 ** 2))  # Assuming 100x100x100 bins

            spatial_similarity = 1.0 - (distance / max_distance)
            return 0.7 + 0.3 * spatial_similarity  # Bonus for same bin
        else:
            return 0.3  # Low relatedness for different bins

    def _find_item_position(self, item: Item, solution: Solution) -> Tuple[int, float, float, float]:
        """Find the position of an item in the solution"""
        for bin_idx, bin_items in enumerate(solution.bins):
            for placed_item in bin_items:
                if placed_item.item == item:
                    return (bin_idx, placed_item.x, placed_item.y, placed_item.z)
        return None


class AdaptiveShawTimeDestroy(AdaptiveShawDestroy):
    """Adaptive Shaw destroy based on packing time/order similarity"""

    def __init__(self, min_rate: float = 0.07, max_rate: float = 0.18):
        super().__init__("Adaptive Shaw Time Destroy", min_rate, max_rate)

    def _calculate_relatedness(self, item1: Item, item2: Item, solution) -> float:
        """Relatedness based on packing order"""
        pos1 = self._find_item_packing_order(item1, solution)
        pos2 = self._find_item_packing_order(item2, solution)

        if pos1 is None or pos2 is None:
            return 0.0

        bin1, order1 = pos1
        bin2, order2 = pos2

        if bin1 == bin2:
            # Same bin - high relatedness if packed close in time
            order_diff = abs(order1 - order2)
            max_diff = len(solution.bins[bin1])
            return 1.0 - (order_diff / max_diff)
        else:
            # Different bins - moderate relatedness
            return 0.2

    def _find_item_packing_order(self, item: Item, solution) -> Tuple[int, int]:
        """Find the packing order of an item"""
        for bin_idx, bin_items in enumerate(solution.bins):
            for order, placed_item in enumerate(bin_items):
                if placed_item.item == item:
                    return (bin_idx, order)
        return None


class AdaptiveShawHybridDestroy(AdaptiveShawDestroy):
    """Adaptive Shaw destroy combining size and spatial proximity"""

    def __init__(self, min_rate: float = 0.09, max_rate: float = 0.24):
        super().__init__("Adaptive Shaw Hybrid Destroy", min_rate, max_rate)

    def _calculate_relatedness(self, item1: Item, item2: Item, solution: Solution) -> float:
        """Combine size and spatial relatedness"""
        size_destroy = AdaptiveShawSizeDestroy()
        spatial_destroy = AdaptiveShawSpatialDestroy()

        size_relatedness = size_destroy._calculate_relatedness(item1, item2, solution)
        spatial_relatedness = spatial_destroy._calculate_relatedness(item1, item2, solution)

        return 0.5 * size_relatedness + 0.5 * spatial_relatedness


# =============================================================================
# ADAPTIVE REGRET REPAIR OPERATORS
# =============================================================================

class AdaptiveRegretRepair(ValidatedRepairOperator):
    """Base class for adaptive regret-based repair operators (Ropke & Pisinger 2006)"""

    def __init__(self, name: str, k: int = None):
        super().__init__(name)  # Initialize validation attributes
        self.k = k  # Number of insertion options to consider for regret
        self.recent_improvements = []  # Track recent performance
        self.success_rate = 0.0

    def _repair_implementation(self, solution: Solution, items: List[Item], bin_template: Bin,
                               rnd_state: np.random.RandomState) -> bool:
        """Repair by selecting items with highest regret value"""
        if not items:
            return True

        try:
            solver = ExtremePointBinPacking3D(bin_template)
            solver.bins = solution.bins
            solver.extreme_points = solution.extreme_points

            remaining_items = items.copy()

            while remaining_items:
                best_item = None
                best_position = None
                best_regret = -float('inf')

                # Evaluate regret for each item
                for item in remaining_items:
                    regret = self._calculate_regret(item, solver, bin_template)
                    if regret > best_regret:
                        best_regret = regret
                        best_item = item
                        best_position = self._find_best_position(item, solver, bin_template)

                if best_item is None or best_position is None:
                    # Fallback: place remaining items in new bins
                    for item in remaining_items:
                        self._place_in_new_bin(item, solver)
                    break

                # Insert best item at best position
                success = self._place_item_safely(best_item, best_position, solver)
                if not success:
                    self._place_in_new_bin(best_item, solver)

                remaining_items.remove(best_item)

            solution.bins = solver.bins
            solution.extreme_points = solver.extreme_points
            return True

        except Exception:
            return False

    def _find_best_position(self, item: Item, solver: ExtremePointBinPacking3D,
                            bin_template: Bin) -> Tuple[str, int, ExtremePoint, float]:
        """Find best insertion position for an item"""
        best_cost = float('inf')
        best_position = ("new", -1, None, float('inf'))

        # Try existing bins
        for bin_idx in range(len(solver.bins)):
            try:
                ep, cost = solver._find_best_ep_bfd(item, bin_idx, MeritFunction.RESIDUAL_SPACE)
                if ep is not None and hasattr(ep, 'x') and cost < best_cost:
                    best_cost = cost
                    best_position = ("existing", bin_idx, ep, cost)
            except Exception:
                continue

        return best_position

    def _place_item_safely(self, item: Item, position: Tuple, solver: ExtremePointBinPacking3D) -> bool:
        """Place item safely with error handling"""
        try:
            placement_type, bin_idx, ep, cost = position

            if placement_type == "new":
                self._place_in_new_bin(item, solver)
                return True
            else:
                if ep is None or not hasattr(ep, 'x'):
                    return False

                placed_item = PlacedItem(item, ep.x, ep.y, ep.z)
                solver.bins[bin_idx].append(placed_item)
                solver.extreme_points[bin_idx].remove(ep)
                solver.update_extreme_points(item, ep, bin_idx)
                return True

        except Exception:
            return False

    def _place_in_new_bin(self, item: Item, solver: ExtremePointBinPacking3D):
        """Safely place item in new bin"""
        try:
            bin_idx = solver.add_new_bin()
            ep = solver.extreme_points[bin_idx][0]
            placed_item = PlacedItem(item, ep.x, ep.y, ep.z)
            solver.bins[bin_idx].append(placed_item)
            solver.extreme_points[bin_idx].remove(ep)
            solver.update_extreme_points(item, ep, bin_idx)
        except Exception:
            pass  # Silent fail for safety

    def update_performance(self, improvement: float):
        """Update performance tracking for adaptation"""
        self.recent_improvements.append(improvement)
        # Keep only recent history
        if len(self.recent_improvements) > 10:
            self.recent_improvements = self.recent_improvements[-10:]

        # Update success rate
        if self.recent_improvements:
            self.success_rate = len([x for x in self.recent_improvements if x > 0]) / len(self.recent_improvements)

    def _calculate_new_bin_cost(self, item: Item, bin_template: Bin) -> float:
        """Cost of opening a new bin"""
        return 1000.0 - (item.volume / bin_template.volume)

    def _calculate_regret(self, item: Item, solver: ExtremePointBinPacking3D,
                          bin_template: Bin) -> float:
        """Calculate regret for inserting an item"""
        raise NotImplementedError("Subclasses must implement regret calculation")


class AdaptiveKRegretRepair(AdaptiveRegretRepair):
    """Adaptive K-regret insertion (Ropke & Pisinger 2006)"""

    def __init__(self, k: int = 3):
        super().__init__(f"Adaptive {k}-Regret Repair", k)

    def _calculate_regret(self, item: Item, solver: ExtremePointBinPacking3D,
                          bin_template: Bin) -> float:
        """Calculate k-regret: difference between k-th best and best insertion cost"""
        insertion_costs = []

        # Find insertion costs for existing bins
        for bin_idx in range(len(solver.bins)):
            try:
                ep, cost = solver._find_best_ep_bfd(item, bin_idx, MeritFunction.RESIDUAL_SPACE)
                if ep is not None:
                    insertion_costs.append(cost)
            except Exception:
                continue

        # Add new bin cost
        insertion_costs.append(self._calculate_new_bin_cost(item, bin_template))

        # Sort costs and calculate regret
        insertion_costs.sort()

        # Adaptive k based on performance
        effective_k = self.k
        if self.recent_improvements and np.mean(self.recent_improvements[-5:]) > 0:
            effective_k = min(self.k + 1, len(insertion_costs))  # More aggressive when performing well

        if len(insertion_costs) >= effective_k:
            return insertion_costs[effective_k - 1] - insertion_costs[0]  # k-th best - best
        elif len(insertion_costs) >= 2:
            return insertion_costs[1] - insertion_costs[0]  # 2-regret as fallback
        else:
            return 0.0  # Only one option


class AdaptiveMaxRegretRepair(AdaptiveRegretRepair):
    """Adaptive Maximum regret insertion - considers worst-case scenario"""

    def __init__(self):
        super().__init__("Adaptive Max Regret Repair", k=None)

    def _calculate_regret(self, item: Item, solver: ExtremePointBinPacking3D,
                          bin_template: Bin) -> float:
        """Calculate maximum regret: difference between worst and best insertion cost"""
        insertion_costs = []

        # Find insertion costs for existing bins
        for bin_idx in range(len(solver.bins)):
            try:
                ep, cost = solver._find_best_ep_bfd(item, bin_idx, MeritFunction.RESIDUAL_SPACE)
                if ep is not None:
                    insertion_costs.append(cost)
            except Exception:
                continue

        # Add new bin cost
        insertion_costs.append(self._calculate_new_bin_cost(item, bin_template))

        if len(insertion_costs) >= 2:
            regret = max(insertion_costs) - min(insertion_costs)

            # Adaptive scaling based on recent performance
            if self.recent_improvements and np.mean(self.recent_improvements[-3:]) < 0:
                regret *= 0.8  # More conservative when not performing well

            return regret
        else:
            return 0.0


class AdaptiveWeightedRegretRepair(AdaptiveRegretRepair):
    """Adaptive Weighted regret considering multiple factors"""

    def __init__(self, k: int = 3, cost_weight: float = 0.7, util_weight: float = 0.3):
        super().__init__(f"Adaptive Weighted {k}-Regret Repair", k)
        self.cost_weight = cost_weight
        self.util_weight = util_weight

    def _calculate_regret(self, item: Item, solver: ExtremePointBinPacking3D,
                          bin_template: Bin) -> float:
        """Calculate weighted regret combining cost and utilization factors"""
        insertion_options = []

        # Analyze insertion options
        for bin_idx in range(len(solver.bins)):
            try:
                ep, cost = solver._find_best_ep_bfd(item, bin_idx, MeritFunction.RESIDUAL_SPACE)
                if ep is not None:
                    # Calculate utilization after insertion
                    current_volume = sum(pi.item.volume for pi in solver.bins[bin_idx])
                    util_after = (current_volume + item.volume) / bin_template.volume

                    # Adaptive weighting based on performance
                    adaptive_cost_weight = self.cost_weight
                    adaptive_util_weight = self.util_weight

                    if self.recent_improvements and np.mean(self.recent_improvements[-3:]) > 0:
                        # Emphasize utilization more when performing well
                        adaptive_util_weight *= 1.2
                        adaptive_cost_weight *= 0.9

                    weighted_cost = adaptive_cost_weight * cost - adaptive_util_weight * util_after
                    insertion_options.append(weighted_cost)
            except Exception:
                continue

        # Add new bin option
        new_bin_util = item.volume / bin_template.volume
        new_bin_weighted_cost = (self.cost_weight * self._calculate_new_bin_cost(item, bin_template) -
                                 self.util_weight * new_bin_util)
        insertion_options.append(new_bin_weighted_cost)

        # Calculate regret
        insertion_options.sort()

        if len(insertion_options) >= self.k:
            return insertion_options[self.k - 1] - insertion_options[0]
        elif len(insertion_options) >= 2:
            return insertion_options[1] - insertion_options[0]
        else:
            return 0.0


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def get_adaptive_shaw_operators() -> List[ValidatedDestroyOperator]:
    """Get all adaptive Shaw destroy operators"""
    return [
        AdaptiveShawSizeDestroy(min_rate=0.06, max_rate=0.20),
        AdaptiveShawSpatialDestroy(min_rate=0.08, max_rate=0.22),
        AdaptiveShawTimeDestroy(min_rate=0.07, max_rate=0.18),
        AdaptiveShawHybridDestroy(min_rate=0.09, max_rate=0.24)
    ]


def get_adaptive_regret_operators() -> List[ValidatedRepairOperator]:
    """Get all adaptive regret repair operators"""
    return [
        AdaptiveKRegretRepair(k=2),
        AdaptiveKRegretRepair(k=3),
        AdaptiveKRegretRepair(k=4),
        AdaptiveMaxRegretRepair(),
        AdaptiveWeightedRegretRepair(k=3, cost_weight=0.7, util_weight=0.3)
    ]


def get_adaptive_shaw_regret_operators() -> Tuple[List[ValidatedDestroyOperator], List[ValidatedRepairOperator]]:
    """Get both adaptive Shaw destroy and regret repair operators"""
    return get_adaptive_shaw_operators(), get_adaptive_regret_operators()


# =============================================================================
# DEMONSTRATION FUNCTION
# =============================================================================

def demonstrate_adaptive_operators():
    """Demonstrate adaptive Shaw and regret operators"""
    print("Adaptive Shaw and Regret Operators for 3D Bin Packing ALNS")
    print("=" * 60)

    shaw_ops, regret_ops = get_adaptive_shaw_regret_operators()

    print("\nADAPTIVE SHAW DESTROY OPERATORS:")
    for i, op in enumerate(shaw_ops, 1):
        print(f"{i}. {op.name} (min_rate: {op.min_rate:.2f}, max_rate: {op.max_rate:.2f})")

    print("\nADAPTIVE REGRET REPAIR OPERATORS:")
    for i, op in enumerate(regret_ops, 1):
        print(f"{i}. {op.name}")

    print("\n" + "=" * 60)
    print("Key Improvements:")
    print("- Percentage-based adaptive removal (5-25% of items)")
    print("- Performance tracking with recent improvement history")
    print("- Adaptive parameters based on operator success")
    print("- Enhanced error handling and safety measures")
    print("- Different percentage ranges per operator type")
    print("=" * 60)


if __name__ == "__main__":
    demonstrate_adaptive_operators()