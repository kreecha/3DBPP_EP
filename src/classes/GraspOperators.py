# -*- coding: utf-8 -*-
"""
Top 10 ALNS Operators for 3D Bin Packing
Based on the network meta-analysis from Voigt (2025) - "A review and ranking of operators
in adaptive large neighborhood search for vehicle routing problems"

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

import time

import numpy as np
import random
import copy
import math
from typing import List, Tuple, Set
from dataclasses import dataclass

from src.classes.ValidateOperatorManager import ValidatedDestroyOperator, ValidatedRepairOperator
# Import shared classes (assuming these are available from   existing code)
from src.common import Item, Bin, PlacedItem, ExtremePoint, SortingRule, MeritFunction, BenchmarkGenerator
from src.ExtremePointHeuristic import ExtremePointBinPacking3D
# from src.AlnsExtremePoint_Cluade import RobustGreedyRepair, SafeBestFitRepair, RegretRepair
from src.classes.ObjectionFunction import CompleteImprovedObjectives
from src.destroy.destroy_operators import RandomDestroy, WorstBinDestroy, LargeItemDestroy
from src.operator_selection import RouletteWheel

# GRASP-Inspired ALNS Operators for 3D Bin Packing
# Based on GRASP MSC Pages 50-51

import numpy as np
import random
import copy
import math
from typing import List, Tuple, Set, Dict
from dataclasses import dataclass

from src.common import Item, Bin, PlacedItem, ExtremePoint, SortingRule, MeritFunction
from src.ExtremePointHeuristic import ExtremePointBinPacking3D
from src.classes.solution import Solution
from src.repair.repair_operators import RobustGreedyRepair


# =============================================================================
# GRASP-INSPIRED DESTROY OPERATORS
# =============================================================================

class LeastOccupiedBinDestroy(ValidatedDestroyOperator):
    """Remove items from the least occupied bins (utilization < threshold)"""

    def __init__(self, threshold: float = 0.4, max_bins_to_remove: int = 2):
        super().__init__("Least Occupied Bin Destroy")
        self.threshold = threshold
        self.max_bins_to_remove = max_bins_to_remove

    def _destroy_implementation(self, solution: Solution, rnd_state: np.random.RandomState) -> List[Item]:
        if not solution.bins:
            return []

        # Calculate utilizations and find least occupied bins
        utilizations = solution.get_bin_utilizations()
        least_occupied_bins = []

        for bin_idx, util in enumerate(utilizations):
            if util < self.threshold and solution.bins[bin_idx]:
                least_occupied_bins.append((bin_idx, util))

        if not least_occupied_bins:
            # If no bins below threshold, take the least occupied one
            min_util_idx = min(range(len(utilizations)), key=lambda i: utilizations[i])
            least_occupied_bins = [(min_util_idx, utilizations[min_util_idx])]

        # Sort by utilization (lowest first) and select bins to remove
        least_occupied_bins.sort(key=lambda x: x[1])
        bins_to_remove = least_occupied_bins[:self.max_bins_to_remove]

        items_to_remove = []
        for bin_idx, _ in bins_to_remove:
            items_to_remove.extend([pi.item for pi in solution.bins[bin_idx]])

        solution.remove_items(items_to_remove)
        return items_to_remove


class SelectiveRemovalDestroy(ValidatedDestroyOperator):
    """Selectively remove a percentage of items from the least occupied bin"""

    def __init__(self, removal_percentage: float = 0.2):
        super().__init__("Selective Removal Destroy")
        self.removal_percentage = removal_percentage

    def _destroy_implementation(self, solution: Solution, rnd_state: np.random.RandomState) -> List[Item]:
        if not solution.bins:
            return []

        # Find least occupied bin
        utilizations = solution.get_bin_utilizations()
        least_occupied_idx = min(range(len(utilizations)), key=lambda i: utilizations[i])

        bin_items = [pi.item for pi in solution.bins[least_occupied_idx]]
        if not bin_items:
            return []

        # Remove a percentage of items from this bin
        num_to_remove = max(1, int(len(bin_items) * self.removal_percentage))
        items_to_remove = rnd_state.choice(bin_items, size=num_to_remove, replace=False).tolist()

        solution.remove_items(items_to_remove)
        return items_to_remove


class SplittingBinDestroy(ValidatedDestroyOperator):
    """Split bins based on spatial dimensions and remove items from one side"""

    def __init__(self, split_threshold: float = 0.5):
        super().__init__("Splitting Bin Destroy")
        self.split_threshold = split_threshold

    def _destroy_implementation(self, solution: Solution, rnd_state: np.random.RandomState) -> List[Item]:
        if not solution.bins:
            return []

        # Select a non-empty bin randomly
        non_empty_bins = [i for i, bin_items in enumerate(solution.bins) if bin_items]
        if not non_empty_bins:
            return []

        bin_idx = rnd_state.choice(non_empty_bins)
        bin_items = solution.bins[bin_idx]

        # Select random splitting axis (0=width/x, 1=depth/y, 2=height/z)
        axis = rnd_state.randint(0, 3)
        axis_names = ['width', 'depth', 'height']

        # Find items that occupy more than threshold along selected axis
        items_to_remove = []
        bin_dimension = getattr(solution.bin_template, axis_names[axis])
        split_point = bin_dimension * self.split_threshold

        for placed_item in bin_items:
            item_start = [placed_item.x, placed_item.y, placed_item.z][axis]
            item_size = [placed_item.item.width, placed_item.item.depth, placed_item.item.height][axis]
            item_end = item_start + item_size

            # Remove items that extend beyond the split point
            if item_end > split_point:
                items_to_remove.append(placed_item.item)

        if not items_to_remove:
            # Fallback: remove largest items
            sorted_items = sorted([pi.item for pi in bin_items], key=lambda x: -x.volume)
            items_to_remove = sorted_items[:max(1, len(sorted_items) // 3)]

        solution.remove_items(items_to_remove)
        return items_to_remove


class UnderutilizedBinDestroy(ValidatedDestroyOperator):
    """Remove items from bins with utilization below 50% for compaction"""

    def __init__(self, utilization_threshold: float = 0.5):
        super().__init__("Underutilized Bin Destroy")
        self.utilization_threshold = utilization_threshold

    def _destroy_implementation(self, solution: Solution, rnd_state: np.random.RandomState) -> List[Item]:
        if not solution.bins:
            return []

        # Find all underutilized bins
        utilizations = solution.get_bin_utilizations()
        underutilized_bins = []

        for bin_idx, util in enumerate(utilizations):
            if util < self.utilization_threshold and solution.bins[bin_idx]:
                underutilized_bins.append(bin_idx)

        if not underutilized_bins:
            return []

        # Remove items from underutilized bins
        items_to_remove = []
        for bin_idx in underutilized_bins:
            items_to_remove.extend([pi.item for pi in solution.bins[bin_idx]])

        solution.remove_items(items_to_remove)
        return items_to_remove


# =============================================================================
# GRASP-INSPIRED REPAIR OPERATORS
# =============================================================================

class CompactingRepair(ValidatedRepairOperator):
    """Repair by compacting items into higher-utilization bins"""

    def __init__(self, high_util_threshold: float = 0.7):
        super().__init__("Compacting Repair")
        self.high_util_threshold = high_util_threshold

    def _repair_implementation(self, solution: Solution, items: List[Item],
                               bin_template: Bin, rnd_state: np.random.RandomState) -> bool:
        if not items:
            return True

        solver = ExtremePointBinPacking3D(bin_template)
        solver.bins = solution.bins
        solver.extreme_points = solution.extreme_points

        # Sort bins by utilization (highest first) to prefer high-utilization bins
        utilizations = []
        for bin_idx, bin_items in enumerate(solver.bins):
            used_volume = sum(pi.item.volume for pi in bin_items)
            util = used_volume / bin_template.volume
            utilizations.append((bin_idx, util))

        utilizations.sort(key=lambda x: -x[1])  # Sort by utilization descending

        # Sort items by volume (largest first for better compaction)
        sorted_items = sorted(items, key=lambda x: -x.volume)

        for item in sorted_items:
            placed = False

            # Try high-utilization bins first
            for bin_idx, util in utilizations:
                if util >= self.high_util_threshold:
                    ep, cost = solver._find_best_ep_bfd(item, bin_idx, MeritFunction.RESIDUAL_SPACE)
                    if ep is not None:
                        placed_item = PlacedItem(item, ep.x, ep.y, ep.z)
                        solver.bins[bin_idx].append(placed_item)
                        solver.extreme_points[bin_idx].remove(ep)
                        solver.update_extreme_points(item, ep, bin_idx)
                        placed = True
                        break

            # If not placed in high-util bins, try all bins
            if not placed:
                for bin_idx, _ in utilizations:
                    ep, cost = solver._find_best_ep_bfd(item, bin_idx, MeritFunction.RESIDUAL_SPACE)
                    if ep is not None:
                        placed_item = PlacedItem(item, ep.x, ep.y, ep.z)
                        solver.bins[bin_idx].append(placed_item)
                        solver.extreme_points[bin_idx].remove(ep)
                        solver.update_extreme_points(item, ep, bin_idx)
                        placed = True
                        break

            # Create new bin if necessary
            if not placed:
                bin_idx = solver.add_new_bin()
                ep = solver.extreme_points[bin_idx][0]
                placed_item = PlacedItem(item, ep.x, ep.y, ep.z)
                solver.bins[bin_idx].append(placed_item)
                solver.extreme_points[bin_idx].remove(ep)
                solver.update_extreme_points(item, ep, bin_idx)

        solution.bins = solver.bins
        solution.extreme_points = solver.extreme_points
        return True


class PairwiseMergingRepair(ValidatedRepairOperator):
    """Repair by attempting to merge items from two bins into one"""

    def __init__(self):
        super().__init__("Pairwise Merging Repair")

    def _repair_implementation(self, solution: Solution, items: List[Item],
                               bin_template: Bin, rnd_state: np.random.RandomState) -> bool:
        if not items:
            return True

        solver = ExtremePointBinPacking3D(bin_template)
        solver.bins = solution.bins
        solver.extreme_points = solution.extreme_points

        # First, try regular insertion
        remaining_items = []
        for item in items:
            placed = False

            # Try existing bins
            for bin_idx in range(len(solver.bins)):
                ep, cost = solver._find_best_ep_bfd(item, bin_idx, MeritFunction.RESIDUAL_SPACE)
                if ep is not None:
                    placed_item = PlacedItem(item, ep.x, ep.y, ep.z)
                    solver.bins[bin_idx].append(placed_item)
                    solver.extreme_points[bin_idx].remove(ep)
                    solver.update_extreme_points(item, ep, bin_idx)
                    placed = True
                    break

            if not placed:
                remaining_items.append(item)

        # For remaining items, try pairwise merging approach
        for item in remaining_items:
            merged = False

            # Try to find two bins that could be merged to accommodate this item
            if len(solver.bins) >= 2:
                merged = self._attempt_pairwise_merge(item, solver, bin_template)

            # If merging didn't work, create new bin
            if not merged:
                bin_idx = solver.add_new_bin()
                ep = solver.extreme_points[bin_idx][0]
                placed_item = PlacedItem(item, ep.x, ep.y, ep.z)
                solver.bins[bin_idx].append(placed_item)
                solver.extreme_points[bin_idx].remove(ep)
                solver.update_extreme_points(item, ep, bin_idx)

        solution.bins = solver.bins
        solution.extreme_points = solver.extreme_points
        return True

    def _attempt_pairwise_merge(self, new_item: Item, solver: ExtremePointBinPacking3D,
                                bin_template: Bin) -> bool:
        """Try to merge two bins to accommodate the new item"""

        # Find pairs of bins with low utilization that might merge
        utilizations = []
        for bin_idx, bin_items in enumerate(solver.bins):
            used_volume = sum(pi.item.volume for pi in bin_items)
            util = used_volume / bin_template.volume
            utilizations.append((bin_idx, util, used_volume))

        # Try pairs of bins with combined volume that fits in one bin
        for i in range(len(utilizations)):
            for j in range(i + 1, len(utilizations)):
                bin1_idx, util1, vol1 = utilizations[i]
                bin2_idx, util2, vol2 = utilizations[j]

                combined_volume = vol1 + vol2 + new_item.volume
                if combined_volume <= bin_template.volume * 0.95:  # Leave some margin

                    # Try to merge bins
                    if self._try_merge_bins(bin1_idx, bin2_idx, new_item, solver, bin_template):
                        return True

        return False

    def _try_merge_bins(self, bin1_idx: int, bin2_idx: int, new_item: Item,
                        solver: ExtremePointBinPacking3D, bin_template: Bin) -> bool:
        """Try to merge two specific bins plus new item into one bin"""

        # Collect all items from both bins
        items_to_merge = []
        items_to_merge.extend([pi.item for pi in solver.bins[bin1_idx]])
        items_to_merge.extend([pi.item for pi in solver.bins[bin2_idx]])
        items_to_merge.append(new_item)

        # Create temporary solver to test merge
        temp_solver = ExtremePointBinPacking3D(bin_template)
        temp_solver.add_new_bin()

        # Try to pack all items in one bin
        all_fit = True
        for item in sorted(items_to_merge, key=lambda x: -x.volume):
            ep = temp_solver._find_best_ep_ffd(item, 0)
            if ep is not None:
                placed_item = PlacedItem(item, ep.x, ep.y, ep.z)
                temp_solver.bins[0].append(placed_item)
                temp_solver.extreme_points[0].remove(ep)
                temp_solver.update_extreme_points(item, ep, 0)
            else:
                all_fit = False
                break

        if all_fit:
            # Merge successful - update original solver
            # Clear the two original bins
            solver.bins[bin1_idx] = []
            solver.bins[bin2_idx] = []
            solver.extreme_points[bin1_idx] = [ExtremePoint(0, 0, 0,
                                                            bin_template.width,
                                                            bin_template.depth,
                                                            bin_template.height)]
            solver.extreme_points[bin2_idx] = [ExtremePoint(0, 0, 0,
                                                            bin_template.width,
                                                            bin_template.depth,
                                                            bin_template.height)]

            # Copy merged result to first bin
            solver.bins[bin1_idx] = temp_solver.bins[0]
            solver.extreme_points[bin1_idx] = temp_solver.extreme_points[0]

            return True

        return False


class BalancedUtilizationRepair(ValidatedRepairOperator):
    """Repair by balancing utilization across bins"""

    def __init__(self):
        super().__init__("Balanced Utilization Repair")

    def _repair_implementation(self, solution: Solution, items: List[Item],
                               bin_template: Bin, rnd_state: np.random.RandomState) -> bool:
        if not items:
            return True

        solver = ExtremePointBinPacking3D(bin_template)
        solver.bins = solution.bins
        solver.extreme_points = solution.extreme_points

        for item in items:
            # Find bin with lowest utilization that can fit the item
            best_bin_idx = None
            best_ep = None
            lowest_utilization = float('inf')

            for bin_idx in range(len(solver.bins)):
                ep, cost = solver._find_best_ep_bfd(item, bin_idx, MeritFunction.RESIDUAL_SPACE)
                if ep is not None:
                    # Calculate current utilization
                    current_volume = sum(pi.item.volume for pi in solver.bins[bin_idx])
                    utilization = current_volume / bin_template.volume

                    if utilization < lowest_utilization:
                        lowest_utilization = utilization
                        best_bin_idx = bin_idx
                        best_ep = ep

            # Place item in best bin or create new bin
            if best_ep is not None:
                placed_item = PlacedItem(item, best_ep.x, best_ep.y, best_ep.z)
                solver.bins[best_bin_idx].append(placed_item)
                solver.extreme_points[best_bin_idx].remove(best_ep)
                solver.update_extreme_points(item, best_ep, best_bin_idx)
            else:
                bin_idx = solver.add_new_bin()
                ep = solver.extreme_points[bin_idx][0]
                placed_item = PlacedItem(item, ep.x, ep.y, ep.z)
                solver.bins[bin_idx].append(placed_item)
                solver.extreme_points[bin_idx].remove(ep)
                solver.update_extreme_points(item, ep, bin_idx)

        solution.bins = solver.bins
        solution.extreme_points = solver.extreme_points
        return True


# =============================================================================
# OPERATOR FACTORY FUNCTIONS
# =============================================================================

def get_grasp_destroy_operators() -> List[ValidatedDestroyOperator]:
    """Get all GRASP-inspired destroy operators"""
    return [
        LeastOccupiedBinDestroy(threshold=0.4, max_bins_to_remove=2),
        SelectiveRemovalDestroy(removal_percentage=0.2),
        SplittingBinDestroy(split_threshold=0.5),
        UnderutilizedBinDestroy(utilization_threshold=0.5),
    ]


def get_grasp_repair_operators() -> List[ValidatedRepairOperator]:
    """Get all GRASP-inspired repair operators"""
    return [
        CompactingRepair(high_util_threshold=0.7),
        PairwiseMergingRepair(),
        BalancedUtilizationRepair(),
    ]


def get_all_grasp_operators() -> Tuple[List[ValidatedDestroyOperator], List[ValidatedRepairOperator]]:
    """Get both GRASP destroy and repair operators"""
    return get_grasp_destroy_operators(), get_grasp_repair_operators()

