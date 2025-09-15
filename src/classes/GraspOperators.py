# -*- coding: utf-8 -*-
"""
Top 10 ALNS Operators for 3D Bin Packing
Based on the network meta-analysis from Voigt (2025) - "A review and ranking of operators
in adaptive large neighborhood search for vehicle routing problems"

Adapted for 3D Bin Packing Problem using Extreme Point Heuristics

@author: Kreecha Puphaiboon
MIT License
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
from src.AlnsExtremePoint_Cluade import RobustGreedyRepair, SafeBestFitRepair, RegretRepair
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


# =============================================================================
# DEMONSTRATION FUNCTION
# =============================================================================

def demonstrate_grasp_operators():
    """Demonstrate GRASP-inspired operators"""
    print("=== GRASP-INSPIRED OPERATORS DEMONSTRATION ===\n")

    # Create test problem
    from src.common import BenchmarkGenerator
    items, bin_template = BenchmarkGenerator.generate_martello_instance(5, 30, random_seed=42)

    # Create initial solution
    solver = ExtremePointBinPacking3D(bin_template)
    solver.c_epbfd(items)
    initial_solution = Solution(
        bins=copy.deepcopy(solver.bins),
        extreme_points=copy.deepcopy(solver.extreme_points),
        bin_template=bin_template
    )

    print("Initial Solution:")
    stats = initial_solution.get_solution_statistics()
    print(f"  Bins: {stats['num_bins']}")
    print(f"  Average utilization: {stats['avg_utilization']:.3f}")
    print(f"  Utilization range: [{stats['min_utilization']:.3f}, {stats['max_utilization']:.3f}]")
    print(f"  Individual utilizations: {[f'{u:.3f}' for u in initial_solution.get_bin_utilizations()]}")
    print()

    rnd_state = np.random.RandomState(42)

    # Test GRASP destroy operators
    print("--- Testing GRASP Destroy Operators ---")
    destroy_ops = get_grasp_destroy_operators()

    for destroy_op in destroy_ops:
        test_solution = initial_solution.copy()
        items_before = test_solution.get_items_count()

        removed_items = destroy_op.destroy(test_solution, rnd_state)
        items_after = test_solution.get_items_count()

        print(f"{destroy_op.name}:")
        print(f"  Items before: {items_before}")
        print(f"  Removed: {len(removed_items)} items")
        print(f"  Items after: {items_after}")
        print(f"  Remaining bins: {test_solution.num_bins}")
        print()

    # Test GRASP repair operators
    print("--- Testing GRASP Repair Operators ---")
    repair_ops = get_grasp_repair_operators()

    # Use output from LeastOccupiedBinDestroy for repair testing
    test_solution = initial_solution.copy()
    removed_items = destroy_ops[0].destroy(test_solution, rnd_state)

    print(f"Starting repair test with {len(removed_items)} items to reinsert")
    print(f"Partial solution: {test_solution.num_bins} bins, {test_solution.get_items_count()} items")
    print()

    for repair_op in repair_ops:
        repair_test_solution = test_solution.copy()
        items_before_repair = repair_test_solution.get_items_count()

        success = repair_op.repair(repair_test_solution, removed_items, bin_template, rnd_state)
        items_after_repair = repair_test_solution.get_items_count()

        print(f"{repair_op.name}:")
        print(f"  Repair success: {success}")
        print(f"  Items before: {items_before_repair}")
        print(f"  Items after: {items_after_repair}")
        print(f"  Final bins: {repair_test_solution.num_bins}")
        if repair_test_solution.num_bins > 0:
            final_utils = repair_test_solution.get_bin_utilizations()
            print(f"  Final utilizations: {[f'{u:.3f}' for u in final_utils]}")
        print()


# Integration of GRASP operators into   existing ALNS

class Enhanced_ALNS_3D_BinPacking:
    """Your ALNS enhanced with GRASP-inspired operators"""

    def __init__(self, bin_template: Bin,
                 max_iterations: int = 1000,
                 max_time: float = 60.0,
                 random_seed: int = 42,
                 objective_type: str = 'hybrid',
                 use_grasp_operators: bool = True):

        self.bin_template = bin_template
        self.max_iterations = max_iterations
        self.max_time = max_time
        self.objective_type = objective_type
        self.use_grasp_operators = use_grasp_operators

        # Initialize operators with GRASP additions
        self._initialize_operators()

        # Initialize RouletteWheel with updated operator counts
        self.operator_selector = RouletteWheel(
            scores=[100.0, 50.0, 20.0, 5.0],
            decay=0.99,
            num_destroy=len(self.destroy_operators),
            num_repair=len(self.repair_operators)
        )

        self.rnd_state = np.random.RandomState(random_seed)

        # Convergence tracking
        self.convergence_data = {
            'iterations': [],
            'current_bins': [],
            'best_bins': [],
            'acceptance_rate': [],
            'current_util': [],
            'best_util': [],
            'current_obj': [],
            'best_obj': []
        }

    def _initialize_operators(self):
        """Initialize operators with GRASP additions"""

        # Base destroy operators   
        self.destroy_operators = [
            RandomDestroy(min_remove=5, max_remove=15),
            WorstBinDestroy(num_bins_to_target=1),
            LargeItemDestroy(percentage=0.2),
        ]

        # Base repair operators   
        self.repair_operators = [
            RobustGreedyRepair(),
            SafeBestFitRepair(),
            RegretRepair(k=2),
            # SimpleRegretRepair(k=3),
        ]

        # Add GRASP operators if enabled
        if self.use_grasp_operators:
            grasp_destroy, grasp_repair = get_all_grasp_operators()
            self.destroy_operators.extend(grasp_destroy)
            self.repair_operators.extend(grasp_repair)

        print(f"Initialized {len(self.destroy_operators)} destroy operators "
              f"({4 if self.use_grasp_operators else 0} GRASP-inspired)")
        print(f"Initialized {len(self.repair_operators)} repair operators "
              f"({3 if self.use_grasp_operators else 0} GRASP-inspired)")

    def solve(self, items: List[Item]) -> Solution:
        """Enhanced solve method with GRASP operators"""
        print(f"Starting Enhanced ALNS with {len(items)} items...")

        # Create initial solution using C-EPBFD
        initial_solver = ExtremePointBinPacking3D(self.bin_template)
        initial_solver.c_epbfd(items)
        current_solution = Solution(
            bins=copy.deepcopy(initial_solver.bins),
            extreme_points=copy.deepcopy(initial_solver.extreme_points),
            bin_template=self.bin_template
        )

        best_solution = current_solution.copy()

        # Use improved objective functions
        best_obj = self._calculate_objective(best_solution)
        current_obj = best_obj

        start_time = time.time()
        iteration = 0
        temperature = 200.0  # Improved initial temperature
        acceptance_count = 0

        print(f"Initial solution: {best_solution.num_bins} bins, "
              f"{best_solution.average_utilization:.3f} util, obj={best_obj:.2f}")

        while iteration < self.max_iterations and (time.time() - start_time) < self.max_time:
            # Select operators
            destroy_idx, repair_idx = self.operator_selector.select_operators(self.rnd_state)
            destroy_op = self.destroy_operators[destroy_idx]
            repair_op = self.repair_operators[repair_idx]

            # Create new solution
            new_solution = current_solution.copy()
            removed_items = destroy_op.destroy(new_solution, self.rnd_state)

            if not removed_items:
                iteration += 1
                continue

            repair_success = repair_op.repair(new_solution, removed_items,
                                              self.bin_template, self.rnd_state)

            if not repair_success:
                iteration += 1
                continue

            # Evaluate new solution with improved objective
            new_obj = self._calculate_objective(new_solution)

            outcome = 3  # Default: rejected

            if new_obj < best_obj - 1e-6:
                outcome = 0  # New global best
                best_solution = new_solution.copy()
                best_obj = new_obj
                current_solution = new_solution.copy()
                current_obj = new_obj
                acceptance_count += 1

                print(f"Iteration {iteration}: NEW BEST - {best_solution.num_bins} bins, "
                      f"util={best_solution.average_utilization:.3f}, obj={best_obj:.2f} "
                      f"[{destroy_op.name} + {repair_op.name}]")

            elif new_obj < current_obj - 1e-6:
                outcome = 1  # Better than current
                current_solution = new_solution.copy()
                current_obj = new_obj
                acceptance_count += 1

            elif self._accept_solution_improved(current_obj, new_obj, temperature):
                outcome = 2  # Accepted
                current_solution = new_solution.copy()
                current_obj = new_obj
                acceptance_count += 1

            # Update operator weights
            self.operator_selector.update(destroy_idx, repair_idx, outcome)
            destroy_op.usage_count += 1
            repair_op.usage_count += 1

            # Track convergence data
            self.convergence_data['iterations'].append(iteration)
            self.convergence_data['current_bins'].append(current_solution.num_bins)
            self.convergence_data['current_util'].append(current_solution.average_utilization)
            self.convergence_data['best_bins'].append(best_solution.num_bins)
            self.convergence_data['best_util'].append(best_solution.average_utilization)
            self.convergence_data['current_obj'].append(current_obj)
            self.convergence_data['best_obj'].append(best_obj)

            # Calculate acceptance rate
            if iteration > 0:
                recent_window = min(100, iteration + 1)
                recent_acceptances = sum(1 for i in range(max(0, iteration - recent_window + 1), iteration + 1)
                                         if i < len(self.convergence_data['iterations']) and
                                         (i == 0 or self.convergence_data['current_obj'][i] !=
                                          self.convergence_data['current_obj'][i - 1]))
                self.convergence_data['acceptance_rate'].append(recent_acceptances / recent_window)
            else:
                self.convergence_data['acceptance_rate'].append(0.0)

            # Update temperature (improved cooling)
            temperature = max(1.0, temperature * 0.9995)
            iteration += 1

        elapsed_time = time.time() - start_time
        print(f"\nCompleted {iteration} iterations in {elapsed_time:.1f}s")
        print(f"Final best: {best_solution.num_bins} bins, "
              f"util={best_solution.average_utilization:.3f}, obj={best_obj:.2f}")
        print(f"Total acceptances: {acceptance_count}/{iteration} ({100 * acceptance_count / iteration:.1f}%)")

        return best_solution

    def _calculate_objective(self, solution: Solution) -> float:
        """Calculate objective using improved functions"""
        if self.objective_type == 'hybrid':
            return CompleteImprovedObjectives.hybrid_objective(solution)
        elif self.objective_type == 'volume_waste':
            return CompleteImprovedObjectives.volume_waste_objective(solution)
        elif self.objective_type == 'balanced':
            return CompleteImprovedObjectives.balanced_objective(solution)
        elif self.objective_type == 'lexicographic':
            return CompleteImprovedObjectives.lexicographic_objective(solution)
        else:
            return CompleteImprovedObjectives.hybrid_objective(solution)  # Default

    def _accept_solution_improved(self, current_obj: float, new_obj: float, temperature: float) -> bool:
        """Improved acceptance criterion with better parameters"""
        if new_obj <= current_obj:
            return True

        delta = new_obj - current_obj
        # Improved acceptance probability calculation
        acceptance_prob = math.exp(-delta / temperature)

        # Add some randomness to avoid getting stuck
        if self.rnd_state.random() < 0.1:  # 10% random acceptance
            acceptance_prob = min(1.0, acceptance_prob * 2.0)

        return self.rnd_state.random() < acceptance_prob

    def print_operator_statistics(self):
        """Print detailed operator performance statistics"""
        print("\n=== OPERATOR PERFORMANCE STATISTICS ===")

        print("\nDestroy Operators:")
        print("Operator Name                    | Usage | Weight | Avg Score | Type")
        print("-" * 70)

        for idx, op in enumerate(self.destroy_operators):
            usage = op.usage_count
            weight = self.operator_selector.destroy_weights[idx]
            avg_score = weight / usage if usage > 0 else 0
            op_type = "GRASP" if any(isinstance(op, cls) for cls in [
                LeastOccupiedBinDestroy, SelectiveRemovalDestroy,
                SplittingBinDestroy, UnderutilizedBinDestroy
            ]) else "Base"

            print(f"{op.name:30s} | {usage:5d} | {weight:6.1f} | {avg_score:9.2f} | {op_type}")

        print("\nRepair Operators:")
        print("Operator Name                    | Usage | Weight | Avg Score | Type")
        print("-" * 70)

        for idx, op in enumerate(self.repair_operators):
            usage = op.usage_count
            weight = self.operator_selector.repair_weights[idx]
            avg_score = weight / usage if usage > 0 else 0
            op_type = "GRASP" if any(isinstance(op, cls) for cls in [
                CompactingRepair, PairwiseMergingRepair, BalancedUtilizationRepair
            ]) else "Base"

            print(f"{op.name:30s} | {usage:5d} | {weight:6.1f} | {avg_score:9.2f} | {op_type}")

        # GRASP operator summary
        grasp_destroy_usage = sum(op.usage_count for op in self.destroy_operators
                                  if any(isinstance(op, cls) for cls in [
            LeastOccupiedBinDestroy, SelectiveRemovalDestroy,
            SplittingBinDestroy, UnderutilizedBinDestroy
        ]))
        total_destroy_usage = sum(op.usage_count for op in self.destroy_operators)

        grasp_repair_usage = sum(op.usage_count for op in self.repair_operators
                                 if any(isinstance(op, cls) for cls in [
            CompactingRepair, PairwiseMergingRepair, BalancedUtilizationRepair
        ]))
        total_repair_usage = sum(op.usage_count for op in self.repair_operators)

        print(f"\nGRASP Operator Usage Summary:")
        print(f"  GRASP Destroy: {grasp_destroy_usage}/{total_destroy_usage} "
              f"({100 * grasp_destroy_usage / total_destroy_usage:.1f}%)")
        print(f"  GRASP Repair:  {grasp_repair_usage}/{total_repair_usage} "
              f"({100 * grasp_repair_usage / total_repair_usage:.1f}%)")


# Comparison function to test GRASP vs non-GRASP
def compare_grasp_vs_baseline(random_seed: int = 42):
    """Compare ALNS performance with and without GRASP operators"""
    print("=== GRASP vs BASELINE OPERATORS COMPARISON ===\n")

    test_instances = [
        (1, 20),  # Class 1, 40 items
        (5, 50),  # Class 5, 40 items
        (8, 100),  # Class 8, 50 items
    ]

    results = []

    for class_type, n_items in test_instances:
        print(f"Testing Class {class_type}, {n_items} items...")
        items, bin_template = BenchmarkGenerator.generate_martello_instance(
            class_type, n_items, random_seed=random_seed)

        # Test baseline ALNS (without GRASP)
        print("  Running Baseline ALNS...")
        baseline_alns = Enhanced_ALNS_3D_BinPacking(
            bin_template=bin_template,
            max_iterations=3000,
            max_time=60.0,
            random_seed=random_seed,
            objective_type='hybrid',
            use_grasp_operators=False  # Disable GRASP
        )
        baseline_solution = baseline_alns.solve(items)

        # Test GRASP-enhanced ALNS
        print("  Running GRASP-Enhanced ALNS...")
        grasp_alns = Enhanced_ALNS_3D_BinPacking(
            bin_template=bin_template,
            max_iterations=3000,
            max_time=60.0,
            random_seed=random_seed,
            objective_type='hybrid',
            use_grasp_operators=True  # Enable GRASP
        )
        grasp_solution = grasp_alns.solve(items)

        # Calculate improvements
        bin_improvement = baseline_solution.num_bins - grasp_solution.num_bins
        util_improvement = grasp_solution.average_utilization - baseline_solution.average_utilization

        result = {
            'instance': f"Class {class_type} ({n_items} items)",
            'baseline_bins': baseline_solution.num_bins,
            'baseline_util': baseline_solution.average_utilization,
            'grasp_bins': grasp_solution.num_bins,
            'grasp_util': grasp_solution.average_utilization,
            'bin_improvement': bin_improvement,
            'util_improvement': util_improvement,
            'baseline_obj': CompleteImprovedObjectives.hybrid_objective(baseline_solution),
            'grasp_obj': CompleteImprovedObjectives.hybrid_objective(grasp_solution)
        }

        results.append(result)
        print(f"  Baseline: {result['baseline_bins']} bins, {result['baseline_util']:.3f} util")
        print(f"  GRASP:    {result['grasp_bins']} bins, {result['grasp_util']:.3f} util")
        print(f"  Improvement: {bin_improvement:+d} bins, {util_improvement:+.3f} util")
        print()

    # Summary table
    print("=== COMPARISON SUMMARY ===")
    print("Instance              | Baseline | GRASP    | Improvement | Util Baseline | Util GRASP | Util Δ")
    print("-" * 90)

    total_baseline_bins = 0
    total_grasp_bins = 0

    for result in results:
        total_baseline_bins += result['baseline_bins']
        total_grasp_bins += result['grasp_bins']

        print(f"{result['instance']:20s} | {result['baseline_bins']:8d} | {result['grasp_bins']:8d} | "
              f"{result['bin_improvement']:+10d} | {result['baseline_util']:13.3f} | "
              f"{result['grasp_util']:10.3f} | {result['util_improvement']:+.3f}")


    total_improvement = total_baseline_bins - total_grasp_bins
    improvement_pct = (total_improvement / total_baseline_bins) * 100 if total_baseline_bins > 0 else 0

    print("-" * 90)
    print(f"{'TOTAL':20s} | {total_baseline_bins:8d} | {total_grasp_bins:8d} | "
          f"{total_improvement:+10d} | {'':13s} | {'':10s} | {improvement_pct:+.1f}%")

    return results


# Usage example with   existing code
def integrate_with_existing_alns():
    """Show how to integrate GRASP operators with   existing ALNS"""
    print("=== INTEGRATION WITH EXISTING ALNS ===\n")

    # Create test problem
    items, bin_template = BenchmarkGenerator.generate_martello_instance(5, 30, random_seed=42)

    # Option 1: Use the enhanced ALNS directly
    print("Option 1: Enhanced ALNS with GRASP operators")
    enhanced_alns = Enhanced_ALNS_3D_BinPacking(
        bin_template=bin_template,
        max_iterations=2000,
        random_seed=42,
        use_grasp_operators=True
    )

    solution1 = enhanced_alns.solve(items)
    enhanced_alns.print_operator_statistics()

    print("\n" + "=" * 60)

    # Option 2: Add GRASP operators to   existing ALNS
    print("Option 2: Add GRASP operators to existing ALNS")
    print("Add these lines to   existing ALNS __init__ method:")
    print("""
    # In   ALNS __init__ method, add:
    grasp_destroy, grasp_repair = get_all_grasp_operators()
    self.destroy_operators.extend(grasp_destroy)
    self.repair_operators.extend(grasp_repair)

    # Update   RouletteWheel with new operator counts:
    self.operator_selector = RouletteWheel(
        scores=[100.0, 50.0, 20.0, 5.0],
        decay=0.99,
        num_destroy=len(self.destroy_operators),  # Updated count
        num_repair=len(self.repair_operators)     # Updated count
    )
    """)


if __name__ == "__main__":
    # Run demonstrations
    print("Running GRASP operator demonstrations...\n")

    # Test individual operators
    demonstrate_grasp_operators()

    print("\n" + "=" * 80)

    # Compare GRASP vs baseline
    compare_grasp_vs_baseline(random_seed=42)

    print("\n" + "=" * 80)

    # Show integration examples
    integrate_with_existing_alns()

# if __name__ == "__main__":
#    demonstrate_grasp_operators()

