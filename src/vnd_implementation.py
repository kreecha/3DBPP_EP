
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 9 08:26:17 2025

AlnsVnD.py

Function VND(start_solution, neighborhoods):
  k = 1  // Start with the first neighborhood
  current_solution = start_solution

  While (k <= number of neighborhoods):
    // Find the best solution in the current neighborhood
    neighbor_solution = find_best_neighbor(current_solution, neighborhoods[k])

    // Check for improvement
    If (Cost(neighbor_solution) < Cost(current_solution)):
      current_solution = neighbor_solution
      k = 1 // Restart the search from the first neighborhood
    Else:
      k = k + 1 // Move to the next neighborhood

  Return current_solution


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
import random
import copy
import math
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

from src.common import Item, Bin, PlacedItem, ExtremePoint, SortingRule, MeritFunction
from src.ExtremePointHeuristic import ExtremePointBinPacking3D
from src.classes.solution import Solution


# =============================================================================
# VND NEIGHBORHOOD STRUCTURES
# =============================================================================

class VNDNeighborhood:
    """Base class for VND neighborhood structures"""

    def __init__(self, name: str):
        self.name = name
        self.usage_count = 0
        self.improvement_count = 0

    def search(self, solution: Solution, objective_func) -> Tuple[Solution, bool]:
        """Search neighborhood for improvements"""
        raise NotImplementedError("Subclasses must implement search method")

    def get_success_rate(self) -> float:
        """Get improvement success rate"""
        return self.improvement_count / self.usage_count if self.usage_count > 0 else 0.0


class ItemSwapNeighborhood(VNDNeighborhood):
    """Swap items between different bins"""

    def __init__(self, max_swaps: int = 10):
        super().__init__("Item Swap")
        self.max_swaps = max_swaps

    def search(self, solution: Solution, objective_func) -> Tuple[Solution, bool]:
        """Try swapping items between bins"""
        self.usage_count += 1
        current_obj = objective_func(solution)
        best_solution = solution.copy()
        best_obj = current_obj
        improved = False

        all_items_with_bins = []
        for bin_idx, bin_items in enumerate(solution.bins):
            for placed_item in bin_items:
                all_items_with_bins.append((placed_item.item, bin_idx))

        if len(all_items_with_bins) < 2:
            return solution, False

        attempts = 0
        while attempts < self.max_swaps and len(all_items_with_bins) >= 2:
            # Select two random items from different bins
            idx1, idx2 = random.sample(range(len(all_items_with_bins)), 2)
            item1, bin1 = all_items_with_bins[idx1]
            item2, bin2 = all_items_with_bins[idx2]

            if bin1 == bin2:
                attempts += 1
                continue

            # Try swapping
            test_solution = solution.copy()
            if self._try_swap_items(test_solution, item1, bin1, item2, bin2):
                test_obj = objective_func(test_solution)
                if test_obj < best_obj:
                    best_solution = test_solution
                    best_obj = test_obj
                    improved = True

            attempts += 1

        if improved:
            self.improvement_count += 1

        return best_solution, improved

    def _try_swap_items(self, solution: Solution, item1: Item, bin1: int,
                        item2: Item, bin2: int) -> bool:
        """Try to swap two items between bins"""
        try:
            # Remove both items
            solution.remove_items([item1, item2])

            # Try to place item1 in bin2 and item2 in bin1
            solver = ExtremePointBinPacking3D(solution.bin_template)
            solver.bins = solution.bins
            solver.extreme_points = solution.extreme_points

            # Place item1 in original bin2 location
            if bin2 < len(solver.bins):
                ep1 = solver._find_best_ep_ffd(item1, bin2)
                if ep1 is None:
                    return False
            else:
                return False

            # Place item2 in original bin1 location
            if bin1 < len(solver.bins):
                ep2 = solver._find_best_ep_ffd(item2, bin1)
                if ep2 is None:
                    return False
            else:
                return False

            # Actually place the items
            placed_item1 = PlacedItem(item1, ep1.x, ep1.y, ep1.z)
            solver.bins[bin2].append(placed_item1)
            solver.extreme_points[bin2].remove(ep1)
            solver.update_extreme_points(item1, ep1, bin2)

            placed_item2 = PlacedItem(item2, ep2.x, ep2.y, ep2.z)
            solver.bins[bin1].append(placed_item2)
            solver.extreme_points[bin1].remove(ep2)
            solver.update_extreme_points(item2, ep2, bin1)

            solution.bins = solver.bins
            solution.extreme_points = solver.extreme_points
            return True

        except Exception:
            return False


class ItemRelocateNeighborhood(VNDNeighborhood):
    """Relocate single items to different bins"""

    def __init__(self, max_relocations: int = 15):
        super().__init__("Item Relocate")
        self.max_relocations = max_relocations

    def search(self, solution: Solution, objective_func) -> Tuple[Solution, bool]:
        """Try relocating items to different bins"""
        self.usage_count += 1
        current_obj = objective_func(solution)
        best_solution = solution.copy()
        best_obj = current_obj
        improved = False

        all_items_with_bins = []
        for bin_idx, bin_items in enumerate(solution.bins):
            for placed_item in bin_items:
                all_items_with_bins.append((placed_item.item, bin_idx))

        attempts = 0
        while attempts < self.max_relocations and all_items_with_bins:
            # Select random item
            item, source_bin = random.choice(all_items_with_bins)

            # Try relocating to different bins
            test_solution = solution.copy()
            if self._try_relocate_item(test_solution, item, source_bin):
                test_obj = objective_func(test_solution)
                if test_obj < best_obj:
                    best_solution = test_solution
                    best_obj = test_obj
                    improved = True

            attempts += 1

        if improved:
            self.improvement_count += 1

        return best_solution, improved

    def _try_relocate_item(self, solution: Solution, item: Item, source_bin: int) -> bool:
        """Try to relocate item to a better bin"""
        try:
            # Remove item
            solution.remove_items([item])

            solver = ExtremePointBinPacking3D(solution.bin_template)
            solver.bins = solution.bins
            solver.extreme_points = solution.extreme_points

            best_bin = None
            best_ep = None
            best_cost = float('inf')

            # Try all bins except source bin
            for bin_idx in range(len(solver.bins)):
                if bin_idx != source_bin:
                    ep, cost = solver._find_best_ep_bfd(item, bin_idx, MeritFunction.RESIDUAL_SPACE)
                    if ep is not None and cost < best_cost:
                        best_cost = cost
                        best_bin = bin_idx
                        best_ep = ep

            # Place in best position found
            if best_ep is not None:
                placed_item = PlacedItem(item, best_ep.x, best_ep.y, best_ep.z)
                solver.bins[best_bin].append(placed_item)
                solver.extreme_points[best_bin].remove(best_ep)
                solver.update_extreme_points(item, best_ep, best_bin)

                solution.bins = solver.bins
                solution.extreme_points = solver.extreme_points
                return True

            return False

        except Exception:
            return False


class BinCompactionNeighborhood(VNDNeighborhood):
    """Compact underutilized bins"""

    def __init__(self, utilization_threshold: float = 0.6):
        super().__init__("Bin Compaction")
        self.utilization_threshold = utilization_threshold

    def search(self, solution: Solution, objective_func) -> Tuple[Solution, bool]:
        """Try compacting underutilized bins"""
        self.usage_count += 1
        current_obj = objective_func(solution)

        # Find underutilized bins
        utilizations = solution.get_bin_utilizations()
        underutil_bins = [i for i, util in enumerate(utilizations)
                          if util < self.utilization_threshold and solution.bins[i]]

        if not underutil_bins:
            return solution, False

        best_solution = solution.copy()
        best_obj = current_obj
        improved = False

        # Try compacting each underutilized bin
        for bin_idx in underutil_bins:
            test_solution = solution.copy()
            if self._try_compact_bin(test_solution, bin_idx):
                test_obj = objective_func(test_solution)
                if test_obj < best_obj:
                    best_solution = test_solution
                    best_obj = test_obj
                    improved = True

        if improved:
            self.improvement_count += 1

        return best_solution, improved

    def _try_compact_bin(self, solution: Solution, bin_idx: int) -> bool:
        """Try to compact a specific bin by relocating its items"""
        try:
            if bin_idx >= len(solution.bins) or not solution.bins[bin_idx]:
                return False

            # Get items from this bin
            items_to_relocate = [pi.item for pi in solution.bins[bin_idx]]

            # Remove items from this bin
            solution.remove_items(items_to_relocate)

            solver = ExtremePointBinPacking3D(solution.bin_template)
            solver.bins = solution.bins
            solver.extreme_points = solution.extreme_points

            # Try to place all items in other bins
            all_placed = True
            for item in items_to_relocate:
                placed = False
                best_bin = None
                best_ep = None
                best_cost = float('inf')

                # Try existing bins (excluding the original bin which should be empty now)
                for target_bin in range(len(solver.bins)):
                    ep, cost = solver._find_best_ep_bfd(item, target_bin, MeritFunction.RESIDUAL_SPACE)
                    if ep is not None and cost < best_cost:
                        best_cost = cost
                        best_bin = target_bin
                        best_ep = ep

                if best_ep is not None:
                    placed_item = PlacedItem(item, best_ep.x, best_ep.y, best_ep.z)
                    solver.bins[best_bin].append(placed_item)
                    solver.extreme_points[best_bin].remove(best_ep)
                    solver.update_extreme_points(item, best_ep, best_bin)
                    placed = True

                if not placed:
                    all_placed = False
                    break

            if all_placed:
                solution.bins = solver.bins
                solution.extreme_points = solver.extreme_points
                # Clean up empty bins
                solution.compact_bins()
                return True

            return False

        except Exception:
            return False


class PairwiseBinMergeNeighborhood(VNDNeighborhood):
    """Try to merge pairs of bins"""

    def __init__(self, max_attempts: int = 10):
        super().__init__("Pairwise Bin Merge")
        self.max_attempts = max_attempts

    def search(self, solution: Solution, objective_func) -> Tuple[Solution, bool]:
        """Try merging pairs of bins"""
        self.usage_count += 1

        if len(solution.bins) < 2:
            return solution, False

        current_obj = objective_func(solution)
        best_solution = solution.copy()
        best_obj = current_obj
        improved = False

        # Try merging pairs of bins
        attempts = 0
        while attempts < self.max_attempts and len(solution.bins) >= 2:
            # Select two random bins
            bin1, bin2 = random.sample(range(len(solution.bins)), 2)

            test_solution = solution.copy()
            if self._try_merge_bins(test_solution, bin1, bin2):
                test_obj = objective_func(test_solution)
                if test_obj < best_obj:
                    best_solution = test_solution
                    best_obj = test_obj
                    improved = True

            attempts += 1

        if improved:
            self.improvement_count += 1

        return best_solution, improved

    def _try_merge_bins(self, solution: Solution, bin1: int, bin2: int) -> bool:
        """Try to merge two specific bins"""
        try:
            if bin1 >= len(solution.bins) or bin2 >= len(solution.bins):
                return False

            # Get all items from both bins
            items1 = [pi.item for pi in solution.bins[bin1]]
            items2 = [pi.item for pi in solution.bins[bin2]]
            all_items = items1 + items2

            if not all_items:
                return False

            # Check if combined volume can fit in one bin
            total_volume = sum(item.volume for item in all_items)
            if total_volume > solution.bin_template.volume * 0.95:  # Leave margin
                return False

            # Remove all items from both bins
            solution.remove_items(all_items)

            # Try to pack all items in one bin
            solver = ExtremePointBinPacking3D(solution.bin_template)
            solver.bins = solution.bins
            solver.extreme_points = solution.extreme_points

            # Add new bin for merged items
            merge_bin_idx = solver.add_new_bin()

            # Try to place all items
            for item in sorted(all_items, key=lambda x: -x.volume):
                ep = solver._find_best_ep_ffd(item, merge_bin_idx)
                if ep is not None:
                    placed_item = PlacedItem(item, ep.x, ep.y, ep.z)
                    solver.bins[merge_bin_idx].append(placed_item)
                    solver.extreme_points[merge_bin_idx].remove(ep)
                    solver.update_extreme_points(item, ep, merge_bin_idx)
                else:
                    # Merge failed
                    return False

            solution.bins = solver.bins
            solution.extreme_points = solver.extreme_points
            solution.compact_bins()
            return True

        except Exception:
            return False


# =============================================================================
# VARIABLE NEIGHBORHOOD DESCENT (VND) ENGINE
# =============================================================================

class VariableNeighborhoodDescent:
    """Variable Neighborhood Descent for 3D Bin Packing"""

    def __init__(self, neighborhoods: List[VNDNeighborhood], max_time_per_search: float = 5.0):
        self.neighborhoods = neighborhoods
        self.max_time_per_search = max_time_per_search
        self.total_improvements = 0
        self.total_searches = 0

    def search(self, solution: Solution, objective_func, max_iterations: int = 100) -> Solution:
        """Apply VND to improve the solution"""
        import time

        start_time = time.time()
        current_solution = solution.copy()
        iteration = 0

        while iteration < max_iterations and (time.time() - start_time) < self.max_time_per_search:
            improved_this_iteration = False

            # Apply each neighborhood in sequence
            for neighborhood in self.neighborhoods:
                if time.time() - start_time >= self.max_time_per_search:
                    break

                new_solution, improved = neighborhood.search(current_solution, objective_func)

                if improved:
                    current_solution = new_solution
                    improved_this_iteration = True
                    self.total_improvements += 1
                    # Restart from first neighborhood when improvement found
                    break

            if not improved_this_iteration:
                # No improvement in any neighborhood, terminate VND
                break

            iteration += 1
            self.total_searches += 1

        return current_solution

    def get_statistics(self) -> Dict:
        """Get VND performance statistics"""
        stats = {
            'total_searches': self.total_searches,
            'total_improvements': self.total_improvements,
            'improvement_rate': self.total_improvements / max(1, self.total_searches),
            'neighborhoods': {}
        }

        for neighborhood in self.neighborhoods:
            stats['neighborhoods'][neighborhood.name] = {
                'usage_count': neighborhood.usage_count,
                'improvement_count': neighborhood.improvement_count,
                'success_rate': neighborhood.get_success_rate()
            }

        return stats

    def print_statistics(self):
        """Print VND performance statistics"""
        stats = self.get_statistics()

        print("\n=== VND PERFORMANCE STATISTICS ===")
        print(f"Total VND searches: {stats['total_searches']}")
        print(f"Total improvements: {stats['total_improvements']}")
        print(f"Overall improvement rate: {stats['improvement_rate']:.3f}")
        print()

        print("Neighborhood Performance:")
        print("Neighborhood Name        | Usage | Improvements | Success Rate")
        print("-" * 60)

        for name, data in stats['neighborhoods'].items():
            print(f"{name:23s} | {data['usage_count']:5d} | {data['improvement_count']:12d} | "
                  f"{data['success_rate']:11.3f}")


# =============================================================================
# DEFAULT VND CONFIGURATIONS
# =============================================================================

def get_default_vnd_neighborhoods() -> List[VNDNeighborhood]:
    """Get default set of VND neighborhoods for 3D bin packing"""
    return [
        ItemRelocateNeighborhood(max_relocations=10),
        ItemSwapNeighborhood(max_swaps=8),
        BinCompactionNeighborhood(utilization_threshold=0.6),
        PairwiseBinMergeNeighborhood(max_attempts=5),
    ]


def create_vnd_engine(max_time_per_search: float = 3.0) -> VariableNeighborhoodDescent:
    """Create a VND engine with default neighborhoods"""
    neighborhoods = get_default_vnd_neighborhoods()
    return VariableNeighborhoodDescent(neighborhoods, max_time_per_search)


# =============================================================================
# DEMONSTRATION FUNCTION
# =============================================================================

def demonstrate_vnd():
    """Demonstrate VND functionality"""
    print("=== VARIABLE NEIGHBORHOOD DESCENT DEMONSTRATION ===\n")

    # Create test problem
    from common import BenchmarkGenerator
    items, bin_template = BenchmarkGenerator.generate_martello_instance(5, 25, random_seed=42)

    # Create initial solution
    solver = ExtremePointBinPacking3D(bin_template)
    solver.c_epbfd(items)

    initial_solution = Solution(
        bins=copy.deepcopy(solver.bins),
        extreme_points=copy.deepcopy(solver.extreme_points),
        bin_template=bin_template
    )

    print("Initial Solution:")
    print(f"  Bins: {initial_solution.num_bins}")
    print(f"  Average utilization: {initial_solution.average_utilization:.3f}")
    print(f"  Items: {initial_solution.get_items_count()}")

    # Define objective function
    def objective_func(solution):
        from src.classes.ObjectionFunction import CompleteImprovedObjectives
        return CompleteImprovedObjectives.hybrid_objective(solution)

    initial_obj = objective_func(initial_solution)
    print(f"  Initial objective: {initial_obj:.2f}")
    print()

    # Apply VND
    print("Applying VND...")
    vnd = create_vnd_engine(max_time_per_search=10.0)
    improved_solution = vnd.search(initial_solution, objective_func, max_iterations=50)

    final_obj = objective_func(improved_solution)

    print("VND Results:")
    print(f"  Final bins: {improved_solution.num_bins}")
    print(f"  Final utilization: {improved_solution.average_utilization:.3f}")
    print(f"  Final objective: {final_obj:.2f}")
    print(f"  Improvement: {initial_obj - final_obj:.2f}")
    print(f"  Bin reduction: {initial_solution.num_bins - improved_solution.num_bins}")
    print()

    # Print VND statistics
    vnd.print_statistics()


if __name__ == "__main__":
    demonstrate_vnd()