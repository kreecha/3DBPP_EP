import time

import numpy as np
import random
import copy
import math
from typing import List, Tuple, Set
from dataclasses import dataclass

from src.common import Item, Bin, PlacedItem, ExtremePoint, SortingRule, MeritFunction
from src.ExtremePointHeuristic import ExtremePointBinPacking3D
from src.validated_operators_base import ValidatedDestroyOperator, ValidatedRepairOperator
from src.classes.solution import Solution


# =============================================================================
# ADAPTIVE PERCENTAGE-BASED BASE CLASSES
# =============================================================================

class AdaptiveDestroyOperator(ValidatedDestroyOperator):
    """Base class for destroy operators with percentage-based removal"""

    def __init__(self, name: str, min_removal_percent: float = 0.05, max_removal_percent: float = 0.25):
        super().__init__(name)
        self.min_removal_percent = min_removal_percent
        self.max_removal_percent = max_removal_percent

    def destroy(self, solution: Solution, rnd_state: np.random.RandomState) -> List[Item]:
        """Override destroy to calculate adaptive removal amounts"""
        # Calculate total items in solution
        total_items = solution.get_items_count()

        if total_items == 0:
            return []

        # Calculate adaptive min/max based on percentages
        min_remove = max(1, int(total_items * self.min_removal_percent))
        max_remove = max(min_remove + 1, int(total_items * self.max_removal_percent))

        # Ensure we don't try to remove more items than exist
        max_remove = min(max_remove, total_items)

        # Store calculated values for use in _destroy_implementation
        self.calculated_min_remove = min_remove
        self.calculated_max_remove = max_remove

        # Call parent's destroy method which will call _destroy_implementation
        return super().destroy(solution, rnd_state)


class AdaptiveBinDestroyOperator(ValidatedDestroyOperator):
    """Base class for bin-focused destroy operators with percentage-based targeting"""

    def __init__(self, name: str, min_bin_percent: float = 0.10, max_bin_percent: float = 0.30):
        super().__init__(name)
        self.min_bin_percent = min_bin_percent
        self.max_bin_percent = max_bin_percent

    def destroy(self, solution: Solution, rnd_state: np.random.RandomState) -> List[Item]:
        """Override destroy to calculate adaptive bin targeting"""
        total_bins = solution.num_bins

        if total_bins == 0:
            return []

        # Calculate adaptive bin targeting based on percentages
        min_bins = max(1, int(total_bins * self.min_bin_percent))
        max_bins = max(min_bins, int(total_bins * self.max_bin_percent))

        # Ensure we don't target more bins than exist
        max_bins = min(max_bins, total_bins)

        # Store calculated values
        self.calculated_min_bins = min_bins
        self.calculated_max_bins = max_bins

        return super().destroy(solution, rnd_state)


# =============================================================================
# ADAPTIVE DESTROY OPERATORS
# =============================================================================

class AdaptiveRandomDestroy(AdaptiveDestroyOperator):
    """Randomly remove percentage of items from the solution"""

    def __init__(self, min_removal_percent: float = 0.08, max_removal_percent: float = 0.20):
        super().__init__("Adaptive Random Destroy", min_removal_percent, max_removal_percent)

    def _destroy_implementation(self, solution: Solution, rnd_state: np.random.RandomState) -> List[Item]:
        all_items = solution.get_all_items()
        if not all_items:
            return []

        # Use calculated removal amounts from parent class
        num_to_remove = rnd_state.randint(self.calculated_min_remove, self.calculated_max_remove + 1)
        num_to_remove = min(num_to_remove, len(all_items))

        items_to_remove = rnd_state.choice(all_items, size=num_to_remove, replace=False).tolist()
        solution.remove_items(items_to_remove)
        return items_to_remove


class AdaptiveWorstBinDestroy(AdaptiveBinDestroyOperator):
    """Remove items from worst utilization bins based on percentage"""

    def __init__(self, min_bin_percent: float = 0.15, max_bin_percent: float = 0.40):
        super().__init__("Adaptive Worst Bin Destroy", min_bin_percent, max_bin_percent)

    def _destroy_implementation(self, solution: Solution, rnd_state: np.random.RandomState) -> List[Item]:
        if not solution.bins:
            return []

        # Calculate utilizations and find worst bins
        utilizations = []
        for bin_idx, bin_items in enumerate(solution.bins):
            if bin_items:  # Only consider non-empty bins
                used_volume = sum(item.item.volume for item in bin_items)
                util = used_volume / solution.bin_template.volume
                utilizations.append((bin_idx, util))

        if not utilizations:
            return []

        # Sort by utilization (worst first)
        utilizations.sort(key=lambda x: x[1])

        # Use calculated bin targeting from parent class
        num_bins_to_target = rnd_state.randint(self.calculated_min_bins, self.calculated_max_bins + 1)
        num_bins_to_target = min(num_bins_to_target, len(utilizations))

        worst_bins = [bin_idx for bin_idx, _ in utilizations[:num_bins_to_target]]

        # Remove all items from targeted bins
        items_to_remove = []
        for bin_idx in worst_bins:
            items_to_remove.extend([item.item for item in solution.bins[bin_idx]])

        solution.remove_items(items_to_remove)
        return items_to_remove


class AdaptiveLargeItemDestroy(AdaptiveDestroyOperator):
    """Remove largest items based on percentage"""

    def __init__(self, min_removal_percent: float = 0.10, max_removal_percent: float = 0.30):
        super().__init__("Adaptive Large Item Destroy", min_removal_percent, max_removal_percent)

    def _destroy_implementation(self, solution: Solution, rnd_state: np.random.RandomState) -> List[Item]:
        all_items = solution.get_all_items()
        if not all_items:
            return []

        # Sort items by volume (largest first)
        sorted_items = sorted(all_items, key=lambda x: -x.volume)

        # Use calculated removal amounts from parent class
        num_to_remove = rnd_state.randint(self.calculated_min_remove, self.calculated_max_remove + 1)
        num_to_remove = min(num_to_remove, len(sorted_items))

        items_to_remove = sorted_items[:num_to_remove]
        solution.remove_items(items_to_remove)
        return items_to_remove


class AdaptiveClusterDestroy(AdaptiveDestroyOperator):
    """Remove spatially clustered items based on percentage"""

    def __init__(self, min_removal_percent: float = 0.06, max_removal_percent: float = 0.18):
        super().__init__("Adaptive Cluster Destroy", min_removal_percent, max_removal_percent)

    def _destroy_implementation(self, solution: Solution, rnd_state: np.random.RandomState) -> List[Item]:
        if not solution.bins:
            return []

        # Find bin with most items for clustering
        best_bin_idx = max(range(len(solution.bins)),
                           key=lambda i: len(solution.bins[i]) if solution.bins[i] else 0)

        bin_items = solution.bins[best_bin_idx]
        if len(bin_items) < 2:
            # Fallback to random destroy
            return AdaptiveRandomDestroy()._destroy_implementation(solution, rnd_state)

        # Select seed item
        seed_item = rnd_state.choice(bin_items)

        # Find spatially close items
        clustered_items = [seed_item]
        remaining_items = [item for item in bin_items if item != seed_item]

        for item in remaining_items:
            distance = self._spatial_distance(seed_item, item)
            cluster_threshold = self._get_cluster_threshold(seed_item, solution.bin_template)
            if distance < cluster_threshold:
                clustered_items.append(item)

        # Use calculated removal amounts from parent class
        target_removal = rnd_state.randint(self.calculated_min_remove, self.calculated_max_remove + 1)
        num_to_remove = min(target_removal, len(clustered_items))

        items_to_remove = [item.item for item in clustered_items[:num_to_remove]]
        solution.remove_items(items_to_remove)
        return items_to_remove

    def _spatial_distance(self, item1: PlacedItem, item2: PlacedItem) -> float:
        """Calculate spatial distance between items"""
        c1 = (item1.x + item1.item.width / 2, item1.y + item1.item.depth / 2, item1.z + item1.item.height / 2)
        c2 = (item2.x + item2.item.width / 2, item2.y + item2.item.depth / 2, item2.z + item2.item.height / 2)
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(c1, c2)))

    def _get_cluster_threshold(self, item: PlacedItem, bin_template: Bin) -> float:
        """Dynamic clustering threshold based on item and bin size"""
        item_diagonal = math.sqrt(item.item.width ** 2 + item.item.depth ** 2 + item.item.height ** 2)
        return item_diagonal * 1.5


class AdaptiveUnderutilizedBinDestroy(AdaptiveBinDestroyOperator):
    """Remove items from underutilized bins based on percentage"""

    def __init__(self, utilization_threshold: float = 0.5, min_bin_percent: float = 0.20,
                 max_bin_percent: float = 0.50):
        super().__init__("Adaptive Underutilized Bin Destroy", min_bin_percent, max_bin_percent)
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
            # Fallback: target lowest utilization bins
            worst_bin_idx = min(range(len(utilizations)), key=lambda i: utilizations[i])
            underutilized_bins = [worst_bin_idx]

        # Use calculated bin targeting, but limit to available underutilized bins
        num_bins_to_target = min(
            rnd_state.randint(self.calculated_min_bins, self.calculated_max_bins + 1),
            len(underutilized_bins)
        )

        target_bins = rnd_state.choice(underutilized_bins, size=num_bins_to_target, replace=False)

        # Remove items from selected bins
        items_to_remove = []
        for bin_idx in target_bins:
            items_to_remove.extend([pi.item for pi in solution.bins[bin_idx]])

        solution.remove_items(items_to_remove)
        return items_to_remove


# =============================================================================
# ADAPTIVE ALNS CLASS WITH PERCENTAGE-BASED OPERATORS
# =============================================================================

class AdaptivePercentageALNS:
    """ALNS with percentage-based adaptive operators"""

    def __init__(self, bin_template: Bin,
                 max_iterations: int = 1000,
                 max_time: float = 60.0,
                 random_seed: int = 42,
                 objective_type: str = 'hybrid'):

        self.bin_template = bin_template
        self.max_iterations = max_iterations
        self.max_time = max_time
        self.objective_type = objective_type

        # Initialize adaptive percentage-based operators
        self.destroy_operators = [
            AdaptiveRandomDestroy(min_removal_percent=0.08, max_removal_percent=0.20),
            AdaptiveWorstBinDestroy(min_bin_percent=0.25, max_bin_percent=0.40),
            AdaptiveLargeItemDestroy(min_removal_percent=0.10, max_removal_percent=0.30),
            AdaptiveClusterDestroy(min_removal_percent=0.06, max_removal_percent=0.18),
            AdaptiveUnderutilizedBinDestroy(utilization_threshold=0.5, min_bin_percent=0.20, max_bin_percent=0.50),
        ]

        # Keep existing repair operators (they don't need percentage adaptation as much)
        from src.repair.repair_operators import GreedyRepair, RandomRepair, BestFitRepair
        self.repair_operators = [
            GreedyRepair(),
            RandomRepair(),
            BestFitRepair(),
        ]

        # Initialize operator selection
        from src.operator_selection import RouletteWheel
        self.operator_selector = RouletteWheel(
            scores=[26.0, 15.0, 9.0, 0],
            decay=0.8,
            num_destroy=len(self.destroy_operators),
            num_repair=len(self.repair_operators)
        )

        self.rnd_state = np.random.RandomState(random_seed)

        # Track adaptive behavior
        self.adaptive_stats = {
            'removal_amounts': [],  # Track actual removal amounts per iteration
            'problem_sizes': [],  # Track problem sizes over time
            'operator_usage': {}  # Track operator usage with problem size context
        }

    def solve(self, items: List[Item]) -> Solution:
        """Solve using adaptive percentage-based operators"""
        print(f"Starting Adaptive Percentage ALNS with {len(items)} items...")

        # Create initial solution
        initial_solver = ExtremePointBinPacking3D(self.bin_template)
        initial_solver.c_epbfd(items)

        current_solution = Solution(
            bins=copy.deepcopy(initial_solver.bins),
            extreme_points=copy.deepcopy(initial_solver.extreme_points),
            bin_template=self.bin_template
        )

        best_solution = current_solution.copy()
        best_obj = self._calculate_objective(best_solution)
        current_obj = best_obj

        print(f"Initial: {best_solution.num_bins} bins, {best_solution.average_utilization:.3f} util")
        print(f"Initial removal ranges: {self._get_removal_ranges(current_solution)}")

        start_time = time.time()
        iteration = 0
        temperature = 100.0

        while iteration < self.max_iterations and (time.time() - start_time) < self.max_time:
            # Track problem size
            current_items = current_solution.get_items_count()
            self.adaptive_stats['problem_sizes'].append(current_items)

            # Select operators
            destroy_idx, repair_idx = self.operator_selector.select_operators(self.rnd_state)
            destroy_op = self.destroy_operators[destroy_idx]
            repair_op = self.repair_operators[repair_idx]

            # Track operator usage with context
            if destroy_op.name not in self.adaptive_stats['operator_usage']:
                self.adaptive_stats['operator_usage'][destroy_op.name] = []
            self.adaptive_stats['operator_usage'][destroy_op.name].append(current_items)

            # Apply operators
            new_solution = current_solution.copy()
            removed_items = destroy_op.destroy(new_solution, self.rnd_state)

            # Track actual removal amount
            self.adaptive_stats['removal_amounts'].append({
                'iteration': iteration,
                'total_items': current_items,
                'removed': len(removed_items),
                'removal_percent': len(removed_items) / current_items if current_items > 0 else 0,
                'operator': destroy_op.name
            })

            if not removed_items:
                iteration += 1
                continue

            repair_success = repair_op.repair(new_solution, removed_items, self.bin_template, self.rnd_state)

            if not repair_success:
                iteration += 1
                continue

            # Evaluate and accept
            new_obj = self._calculate_objective(new_solution)

            if new_obj < best_obj - 1.0:
                best_solution = new_solution.copy()
                best_obj = new_obj
                current_solution = new_solution.copy()
                current_obj = new_obj
                print(f"Iter {iteration}: NEW BEST {best_solution.num_bins} bins, "
                      f"util={best_solution.average_utilization:.3f}, "
                      f"removed {len(removed_items)}/{current_items} items ({100 * len(removed_items) / current_items:.1f}%)")

            elif new_obj < current_obj - 1e-6 or self._accept_solution(current_obj, new_obj, temperature):
                current_solution = new_solution.copy()
                current_obj = new_obj

            # Update operator weights
            outcome = 0 if new_obj < best_obj else (1 if new_obj < current_obj else 3)
            self.operator_selector.update(destroy_idx, repair_idx, outcome)

            temperature *= 0.9995
            iteration += 1

        elapsed_time = time.time() - start_time
        print(f"\nCompleted {iteration} iterations in {elapsed_time:.1f}s")
        print(f"Final: {best_solution.num_bins} bins, util={best_solution.average_utilization:.3f}")

        return best_solution

    def _calculate_objective(self, solution: Solution) -> float:
        """Calculate objective function"""
        if self.objective_type == 'hybrid':
            utilizations = solution.get_bin_utilizations()
            return (solution.num_bins * 1000.0 +
                    (1.0 - solution.average_utilization) * 50.0 +
                    np.var(utilizations) * 20.0)
        else:
            return solution.num_bins    # * 1000.0 + (1.0 - solution.average_utilization) * 10.0

    def _accept_solution(self, current_obj: float, new_obj: float, temperature: float) -> bool:
        """Simulated annealing acceptance"""
        if new_obj <= current_obj:
            return True
        delta = new_obj - current_obj
        acceptance_prob = math.exp(-delta / temperature)
        return self.rnd_state.random() < acceptance_prob

    def _get_removal_ranges(self, solution: Solution) -> str:
        """Get current removal ranges for all operators"""
        total_items = solution.get_items_count()
        total_bins = solution.num_bins

        ranges = []
        for op in self.destroy_operators:
            if isinstance(op, AdaptiveDestroyOperator):
                min_remove = max(1, int(total_items * op.min_removal_percent))
                max_remove = max(min_remove + 1, int(total_items * op.max_removal_percent))
                ranges.append(f"{op.name}: {min_remove}-{max_remove}")
            elif isinstance(op, AdaptiveBinDestroyOperator):
                min_bins = max(1, int(total_bins * op.min_bin_percent))
                max_bins = max(min_bins, int(total_bins * op.max_bin_percent))
                ranges.append(f"{op.name}: {min_bins}-{max_bins} bins")

        return "; ".join(ranges)

    def print_adaptive_statistics(self):
        """Print statistics about adaptive behavior"""
        print("\n=== ADAPTIVE PERCENTAGE STATISTICS ===")

        if not self.adaptive_stats['removal_amounts']:
            print("No statistics available")
            return

        # Overall removal statistics
        removal_data = self.adaptive_stats['removal_amounts']
        total_removals = len(removal_data)
        avg_removal_percent = np.mean([r['removal_percent'] for r in removal_data])

        print(f"Total operations: {total_removals}")
        print(f"Average removal percentage: {avg_removal_percent:.2%}")

        # Per-operator statistics
        print("\nPer-Operator Removal Statistics:")
        print("Operator                     | Usage | Avg % Removed | Avg Items Removed")
        print("-" * 75)

        operator_stats = {}
        for data in removal_data:
            op_name = data['operator']
            if op_name not in operator_stats:
                operator_stats[op_name] = []
            operator_stats[op_name].append(data)

        for op_name, op_data in operator_stats.items():
            usage = len(op_data)
            avg_percent = np.mean([d['removal_percent'] for d in op_data])
            avg_items = np.mean([d['removed'] for d in op_data])
            print(f"{op_name:28s} | {usage:5d} | {avg_percent:12.1%} | {avg_items:16.1f}")

        # Problem size evolution
        sizes = self.adaptive_stats['problem_sizes']
        if sizes:
            print(f"\nProblem Size Evolution:")
            print(f"  Initial items: {sizes[0] if sizes else 0}")
            print(f"  Final items: {sizes[-1] if sizes else 0}")
            print(f"  Average items: {np.mean(sizes):.1f}")


# =============================================================================
# DEMONSTRATION FUNCTION
# =============================================================================

def demonstrate_adaptive_percentage():
    """Demonstrate adaptive percentage-based operators"""
    print("=== ADAPTIVE PERCENTAGE OPERATORS DEMONSTRATION ===\n")

    from src.common import BenchmarkGenerator
    items, bin_template = BenchmarkGenerator.generate_martello_instance(5, 40, random_seed=42)

    print(f"Test instance: {len(items)} items")
    print(f"Bin size: {bin_template.width}x{bin_template.depth}x{bin_template.height}\n")

    # Run adaptive ALNS
    adaptive_alns = AdaptivePercentageALNS(
        bin_template=bin_template,
        max_iterations=2000,
        max_time=60.0,
        random_seed=42
    )

    solution = adaptive_alns.solve(items)

    # Print statistics
    adaptive_alns.print_adaptive_statistics()

    return adaptive_alns, solution


if __name__ == "__main__":
    demonstrate_adaptive_percentage()