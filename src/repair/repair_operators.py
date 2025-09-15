# -*- coding: utf-8 -*-
"""
ALNS Operators for 3D Bin Packing
Based on the network meta-analysis from Voigt (2025) - "A review and ranking of operators
in adaptive large neighborhood search for vehicle routing problems"

Adapted for 3D Bin Packing Problem using Extreme Point Heuristics

@author: Kreecha Puphaiboon
MIT License
"""

import copy
from typing import List, Tuple, Optional

import numpy as np

from src.ExtremePointHeuristic import ExtremePointBinPacking3D
from src.classes.ValidateOperatorManager import ValidatedRepairOperator
from src.classes.solution import Solution
from src.common import Item, Bin, PlacedItem, ExtremePoint, SortingRule, MeritFunction


class RepairOperator(ValidatedRepairOperator):
    """Base class for repair operators"""

    def __init__(self, name: str):
        super().__init__(name)

    def _repair_implementation(self, solution, items_to_repair: List[Item],
                               bin_template: Bin, rnd_state: np.random.RandomState) -> bool:
        """
        Default implementation that calls the old repair() method if it exists,
        otherwise raises NotImplementedError for subclasses to implement.
        """
        # Check if subclass has implemented the old repair() method
        if hasattr(self, '_old_repair') and callable(getattr(self, '_old_repair')):
            return self._old_repair(solution, items_to_repair, bin_template, rnd_state)
        else:
            # Subclass should implement _repair_implementation directly
            raise NotImplementedError(f"{self.__class__.__name__} must implement _repair_implementation()")

    def repair(self, solution, items_to_repair: List[Item], bin_template: Bin,
               rnd_state: np.random.RandomState) -> bool:
        """
        This method is now handled by the parent ValidatedRepairOperator
        If a subclass overrides this method, it will bypass validation.
        Subclasses should implement _repair_implementation() instead.
        """
        return super().repair(solution, items_to_repair, bin_template, rnd_state)

class GreedyRepair(RepairOperator):
    """Insert items greedily using EP-BFD"""
    def __init__(self):
        super().__init__("Greedy Repair")

    def _repair_implementation(self, solution: Solution, items: List[Item], bin_template: Bin,
              rnd_state: np.random.RandomState) -> bool:
        solver = ExtremePointBinPacking3D(bin_template)
        solver.bins = solution.bins
        solver.extreme_points = solution.extreme_points

        sorted_items = solver.sort_items(items, SortingRule.VOLUME_HEIGHT)
        for item in sorted_items:
            placed = False
            for bin_idx in range(len(solver.bins)):
                ep, merit = solver._find_best_ep_bfd(item, bin_idx, MeritFunction.RESIDUAL_SPACE)
                if ep is not None:
                    placed_item = PlacedItem(item, ep.x, ep.y, ep.z)
                    solver.bins[bin_idx].append(placed_item)
                    solver.extreme_points[bin_idx].remove(ep)
                    solver.update_extreme_points(item, ep, bin_idx)
                    placed = True
                    break

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

class RandomRepair(RepairOperator):
    """Insert items in random order at feasible positions"""
    def __init__(self):
        super().__init__("Random Repair")

    def _repair_implementation(self, solution: Solution, items: List[Item], bin_template: Bin,
              rnd_state: np.random.RandomState) -> bool:
        solver = ExtremePointBinPacking3D(bin_template)
        solver.bins = solution.bins
        solver.extreme_points = solution.extreme_points

        items_to_place = items.copy()
        rnd_state.shuffle(items_to_place)

        for item in items_to_place:
            placed = False
            bin_indices = list(range(len(solver.bins)))
            rnd_state.shuffle(bin_indices)

            for bin_idx in bin_indices:
                ep = solver._find_best_ep_ffd(item, bin_idx)
                if ep is not None:
                    placed_item = PlacedItem(item, ep.x, ep.y, ep.z)
                    solver.bins[bin_idx].append(placed_item)
                    solver.extreme_points[bin_idx].remove(ep)
                    solver.update_extreme_points(item, ep, bin_idx)
                    placed = True
                    break

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

class BestFitRepair(RepairOperator):
    """Insert items using best-fit strategy across all bins"""
    def __init__(self):
        super().__init__("Best Fit Repair")

    def _repair_implementation(self, solution: Solution, items: List[Item], bin_template: Bin,
              rnd_state: np.random.RandomState) -> bool:
        solver = ExtremePointBinPacking3D(bin_template)
        solver.bins = solution.bins
        solver.extreme_points = solution.extreme_points

        sorted_items = solver.sort_items(items, SortingRule.VOLUME_HEIGHT)
        for item in sorted_items:
            best_bin = None
            best_ep = None
            best_merit = float('inf')

            for bin_idx in range(len(solver.bins)):
                ep, merit = solver._find_best_ep_bfd(item, bin_idx, MeritFunction.RESIDUAL_SPACE)
                if ep is not None and merit < best_merit:
                    best_merit = merit
                    best_bin = bin_idx
                    best_ep = ep

            if best_ep is not None:
                placed_item = PlacedItem(item, best_ep.x, best_ep.y, best_ep.z)
                solver.bins[best_bin].append(placed_item)
                solver.extreme_points[best_bin].remove(best_ep)
                solver.update_extreme_points(item, best_ep, best_bin)
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
# TOP 10 REPAIR OPERATORS (Based on Voigt 2025 Rankings)
# =============================================================================

class RegretRepair(ValidatedRepairOperator):
    """Safe regret-based repair with proper error handling"""

    def __init__(self, k: int = 3):
        super().__init__(f"Safe {k}-Regret Repair")
        self.k = k

    def _repair_implementation(self, solution: Solution, items: List[Item],
                               bin_template: Bin, rnd_state: np.random.RandomState) -> bool:
        if not items:
            return True

        try:
            solver = ExtremePointBinPacking3D(bin_template)
            solver.bins = copy.deepcopy(solution.bins)
            solver.extreme_points = copy.deepcopy(solution.extreme_points)

            remaining_items = items.copy()

            while remaining_items:
                best_item = None
                best_placement = None
                best_regret = -float('inf')

                for item in remaining_items:
                    placements = self._get_safe_placements(item, solver, bin_template)
                    regret = self._calculate_regret(placements)

                    if regret > best_regret:
                        best_regret = regret
                        best_item = item
                        best_placement = placements[0] if placements else None

                if best_item is None or best_placement is None:
                    # Fallback: place remaining items in new bins
                    for item in remaining_items:
                        self._place_in_new_bin(item, solver)
                    break

                # Place the best item
                success = self._place_item_safely(best_item, best_placement, solver)
                if not success:
                    # Fallback to new bin
                    self._place_in_new_bin(best_item, solver)

                remaining_items.remove(best_item)

            solution.bins = solver.bins
            solution.extreme_points = solver.extreme_points
            return True

        except Exception as e:
            return False

    def _get_safe_placements(self, item: Item, solver: ExtremePointBinPacking3D,
                            bin_template: Bin) -> List[Tuple[float, str, int, Optional[ExtremePoint]]]:
        """Get safe placement options with proper validation"""
        placements = []

        # Try existing bins
        for bin_idx in range(len(solver.bins)):
            try:
                ep, cost = solver._find_best_ep_bfd(item, bin_idx, MeritFunction.RESIDUAL_SPACE)
                if ep is not None and hasattr(ep, 'x') and hasattr(ep, 'y') and hasattr(ep, 'z'):
                    placements.append((cost, "existing", bin_idx, ep))
            except Exception:
                continue

        # Add new bin option
        new_bin_cost = 1000.0 - (item.volume / bin_template.volume)
        placements.append((new_bin_cost, "new", -1, None))

        return sorted(placements, key=lambda x: x[0])

    def _calculate_regret(self, placements: List[Tuple]) -> float:
        """Calculate regret value safely"""
        if len(placements) < 2:
            return 0.0

        costs = [p[0] for p in placements]
        if len(costs) >= self.k:
            return costs[self.k - 1] - costs[0]
        else:
            return costs[1] - costs[0] if len(costs) >= 2 else 0.0

    def _place_item_safely(self, item: Item, placement: Tuple,
                          solver: ExtremePointBinPacking3D) -> bool:
        """Place item with comprehensive error checking"""
        try:
            cost, placement_type, bin_idx, ep = placement

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


class RobustGreedyRepair(ValidatedRepairOperator):
    """Robust greedy repair with extensive error checking"""

    def __init__(self):
        super().__init__("Robust Greedy Repair")

    def _repair_implementation(self, solution: Solution, items: List[Item],
                               bin_template: Bin, rnd_state: np.random.RandomState) -> bool:
        if not items:
            return True

        try:
            solver = ExtremePointBinPacking3D(bin_template)
            solver.bins = copy.deepcopy(solution.bins)  # Use deepcopy to avoid reference issues
            solver.extreme_points = copy.deepcopy(solution.extreme_points)

            # Sort items by volume (largest first for better packing)
            sorted_items = sorted(items, key=lambda x: -x.volume)

            for item in sorted_items:
                placed = False
                best_bin_idx = None
                best_ep = None
                best_cost = float('inf')

                # Try existing bins
                for bin_idx in range(len(solver.bins)):
                    try:
                        ep, cost = solver._find_best_ep_bfd(item, bin_idx, MeritFunction.RESIDUAL_SPACE)
                        if ep is not None and hasattr(ep, 'x') and cost < best_cost:
                            best_cost = cost
                            best_bin_idx = bin_idx
                            best_ep = ep
                    except Exception as e:
                        # Skip this bin if there's an error
                        continue

                # Place item in best position
                if best_ep is not None and best_bin_idx is not None:
                    try:
                        placed_item = PlacedItem(item, best_ep.x, best_ep.y, best_ep.z)
                        solver.bins[best_bin_idx].append(placed_item)
                        solver.extreme_points[best_bin_idx].remove(best_ep)
                        solver.update_extreme_points(item, best_ep, best_bin_idx)
                        placed = True
                    except Exception as e:
                        # If placement fails, fall back to new bin
                        placed = False

                # Create new bin if needed
                if not placed:
                    try:
                        bin_idx = solver.add_new_bin()
                        ep = solver.extreme_points[bin_idx][0]
                        placed_item = PlacedItem(item, ep.x, ep.y, ep.z)
                        solver.bins[bin_idx].append(placed_item)
                        solver.extreme_points[bin_idx].remove(ep)
                        solver.update_extreme_points(item, ep, bin_idx)
                    except Exception as e:
                        # Ultimate fallback - return failure
                        return False

            # Update solution only if successful
            solution.bins = solver.bins
            solution.extreme_points = solver.extreme_points
            return True

        except Exception as e:
            # If anything goes wrong, return failure
            return False


class BestCostRepair(ValidatedRepairOperator):
    """Safe best cost repair with proper error handling"""

    def __init__(self):
        super().__init__("Safe Best Cost Repair")

    def _repair_implementation(self, solution: Solution, items: List[Item],
                               bin_template: Bin, rnd_state: np.random.RandomState) -> bool:
        if not items:
            return True

        try:
            solver = ExtremePointBinPacking3D(bin_template)
            solver.bins = copy.deepcopy(solution.bins)
            solver.extreme_points = copy.deepcopy(solution.extreme_points)

            sorted_items = sorted(items, key=lambda x: -x.volume)

            for item in sorted_items:
                best_placement = self._find_best_placement(item, solver, bin_template)

                if not self._place_item_safely(item, best_placement, solver):
                    # Fallback to new bin
                    self._place_in_new_bin(item, solver)

            solution.bins = solver.bins
            solution.extreme_points = solver.extreme_points
            return True

        except Exception:
            return False

    def _find_best_placement(self, item: Item, solver: ExtremePointBinPacking3D,
                             bin_template: Bin) -> Tuple[float, str, int, Optional[ExtremePoint]]:
        """Find best placement with error handling"""
        best_cost = float('inf')
        best_placement = (float('inf'), "new", -1, None)

        for bin_idx in range(len(solver.bins)):
            try:
                ep, cost = solver._find_best_ep_bfd(item, bin_idx, MeritFunction.RESIDUAL_SPACE)
                if ep is not None and hasattr(ep, 'x') and cost < best_cost:
                    best_cost = cost
                    best_placement = (cost, "existing", bin_idx, ep)
            except Exception:
                continue

        return best_placement

    def _place_item_safely(self, item: Item, placement: Tuple,
                           solver: ExtremePointBinPacking3D) -> bool:
        """Place item safely with validation"""
        try:
            cost, placement_type, bin_idx, ep = placement

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
            pass


class ClosestRouteRepair(ValidatedRepairOperator):
    """Safe closest route repair with proper error handling"""

    def __init__(self):
        super().__init__("Safe Closest Route Repair")

    def _repair_implementation(self, solution: Solution, items: List[Item],
                               bin_template: Bin, rnd_state: np.random.RandomState) -> bool:
        if not items:
            return True

        try:
            solver = ExtremePointBinPacking3D(bin_template)
            solver.bins = copy.deepcopy(solution.bins)
            solver.extreme_points = copy.deepcopy(solution.extreme_points)

            sorted_items = sorted(items, key=lambda x: -x.volume)

            for item in sorted_items:
                best_placement = self._find_closest_fit(item, solver, bin_template)

                if not self._place_item_safely(item, best_placement, solver):
                    self._place_in_new_bin(item, solver)

            solution.bins = solver.bins
            solution.extreme_points = solver.extreme_points
            return True

        except Exception:
            return False

    def _find_closest_fit(self, item: Item, solver: ExtremePointBinPacking3D,
                          bin_template: Bin) -> Tuple[float, str, int, Optional[ExtremePoint]]:
        """Find closest fitting bin"""
        best_fit_score = float('inf')
        best_placement = (float('inf'), "new", -1, None)

        for bin_idx in range(len(solver.bins)):
            try:
                ep, cost = solver._find_best_ep_bfd(item, bin_idx, MeritFunction.RESIDUAL_SPACE)
                if ep is not None and hasattr(ep, 'x'):
                    current_volume = sum(pi.item.volume for pi in solver.bins[bin_idx])
                    remaining_space = bin_template.volume - current_volume - item.volume

                    if remaining_space >= 0 and remaining_space < best_fit_score:
                        best_fit_score = remaining_space
                        best_placement = (cost, "existing", bin_idx, ep)
            except Exception:
                continue

        return best_placement

    def _place_item_safely(self, item: Item, placement: Tuple,
                           solver: ExtremePointBinPacking3D) -> bool:
        """Place item safely"""
        try:
            cost, placement_type, bin_idx, ep = placement

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
            pass


class BalancedRepair(RepairOperator):
    """Balanced insertion to maintain even bin utilization"""

    def __init__(self):
        super().__init__("Balanced Repair")

    def _repair_implementation(self, solution, items: List[Item], bin_template: Bin,
               rnd_state: np.random.RandomState) -> bool:
        """Insert items to balance utilization across bins"""
        solver = ExtremePointBinPacking3D(bin_template)
        solver.bins = solution.bins
        solver.extreme_points = solution.extreme_points

        sorted_items = sorted(items, key=lambda x: -x.volume)

        for item in sorted_items:
            # Find bin with lowest current utilization that can fit the item
            best_bin = None
            best_ep = None
            lowest_utilization = float('inf')

            for bin_idx in range(len(solver.bins)):
                ep = solver._find_best_ep_ffd(item, bin_idx)
                if ep is not None:
                    current_volume = sum(pi.item.volume for pi in solver.bins[bin_idx])
                    utilization = current_volume / bin_template.volume

                    if utilization < lowest_utilization:
                        lowest_utilization = utilization
                        best_bin = bin_idx
                        best_ep = ep

            # Insert at best position or create new bin
            if best_ep is not None:
                placed_item = PlacedItem(item, best_ep.x, best_ep.y, best_ep.z)
                solver.bins[best_bin].append(placed_item)
                solver.extreme_points[best_bin].remove(best_ep)
                solver.update_extreme_points(item, best_ep, best_bin)
            else:
                bin_idx = solver.add_new_bin()
                ep = solver.extreme_points[bin_idx][0]
                placed_item = PlacedItem(item, ep.x, ep.y, ep.z)
                solver.bins[bin_idx].append(placed_item)
                solver.extreme_points[bin_idx].remove(ep)
                solver.update_extreme_points(item, ep, bin_idx)

        solution.bins = solver.bins
        solution.extreme_points = solution.extreme_points
        return True


class TimeBasedRepair(RepairOperator):
    """Insert items considering timing/ordering constraints"""

    def __init__(self):
        super().__init__("Time Based Repair")

    def _repair_implementation(self, solution, items: List[Item], bin_template: Bin,
               rnd_state: np.random.RandomState) -> bool:
        """Insert items considering their original order/timing"""
        solver = ExtremePointBinPacking3D(bin_template)
        solver.bins = solution.bins
        solver.extreme_points = solution.extreme_points

        # Maintain original item ordering for time-based insertion
        for item in items:  # Don't sort - preserve order
            best_bin = None
            best_ep = None
            best_timing_score = float('inf')

            for bin_idx in range(len(solver.bins)):
                ep = solver._find_best_ep_ffd(item, bin_idx)
                if ep is not None:
                    # Prefer bins with fewer items (earlier in packing sequence)
                    timing_score = len(solver.bins[bin_idx])
                    if timing_score < best_timing_score:
                        best_timing_score = timing_score
                        best_bin = bin_idx
                        best_ep = ep

            if best_ep is not None:
                placed_item = PlacedItem(item, best_ep.x, best_ep.y, best_ep.z)
                solver.bins[best_bin].append(placed_item)
                solver.extreme_points[best_bin].remove(best_ep)
                solver.update_extreme_points(item, best_ep, best_bin)
            else:
                bin_idx = solver.add_new_bin()
                ep = solver.extreme_points[bin_idx][0]
                placed_item = PlacedItem(item, ep.x, ep.y, ep.z)
                solver.bins[bin_idx].append(placed_item)
                solver.extreme_points[bin_idx].remove(ep)
                solver.update_extreme_points(item, ep, bin_idx)

        solution.bins = solver.bins
        solution.extreme_points = solution.extreme_points
        return True


class FillGapRepair(ValidatedRepairOperator):
    """Safe fill gap repair with proper error handling"""

    def __init__(self):
        super().__init__("Safe Fill Gap Repair")

    def _repair_implementation(self, solution: Solution, items: List[Item],
                               bin_template: Bin, rnd_state: np.random.RandomState) -> bool:
        if not items:
            return True

        try:
            solver = ExtremePointBinPacking3D(bin_template)
            solver.bins = copy.deepcopy(solution.bins)
            solver.extreme_points = copy.deepcopy(solution.extreme_points)

            # Sort items by volume (smallest first to fill gaps better)
            sorted_items = sorted(items, key=lambda x: x.volume)

            for item in sorted_items:
                best_placement = self._find_best_gap_fill(item, solver, bin_template)

                if not self._place_item_safely(item, best_placement, solver):
                    self._place_in_new_bin(item, solver)

            solution.bins = solver.bins
            solution.extreme_points = solver.extreme_points
            return True

        except Exception:
            return False

    def _find_best_gap_fill(self, item: Item, solver: ExtremePointBinPacking3D,
                            bin_template: Bin) -> Tuple[float, str, int, Optional[ExtremePoint]]:
        """Find best gap-filling placement"""
        best_gap_fill = 0
        best_placement = (float('inf'), "new", -1, None)

        for bin_idx in range(len(solver.bins)):
            try:
                ep, cost = solver._find_best_ep_bfd(item, bin_idx, MeritFunction.RESIDUAL_SPACE)
                if ep is not None and hasattr(ep, 'x'):
                    current_volume = sum(pi.item.volume for pi in solver.bins[bin_idx])
                    remaining_space = bin_template.volume - current_volume
                    gap_fill = item.volume / remaining_space if remaining_space > 0 else 0

                    if gap_fill > best_gap_fill:
                        best_gap_fill = gap_fill
                        best_placement = (cost, "existing", bin_idx, ep)
            except Exception:
                continue

        return best_placement

    def _place_item_safely(self, item: Item, placement: Tuple,
                           solver: ExtremePointBinPacking3D) -> bool:
        """Place item safely"""
        try:
            cost, placement_type, bin_idx, ep = placement

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
            pass


class LoosestRouteRepair(RepairOperator):
    """Insert into bins with most available space"""

    def __init__(self):
        super().__init__("Loosest Route Repair")

    def _repair_implementation(self, solution, items: List[Item], bin_template: Bin,
               rnd_state: np.random.RandomState) -> bool:
        """Insert items into bins with most available space"""
        solver = ExtremePointBinPacking3D(bin_template)
        solver.bins = solution.bins
        solver.extreme_points = solution.extreme_points

        sorted_items = sorted(items, key=lambda x: -x.volume)

        for item in sorted_items:
            best_bin = None
            best_ep = None
            most_space = -1

            for bin_idx in range(len(solver.bins)):
                ep = solver._find_best_ep_ffd(item, bin_idx)
                if ep is not None:
                    current_volume = sum(pi.item.volume for pi in solver.bins[bin_idx])
                    available_space = bin_template.volume - current_volume

                    if available_space > most_space:
                        most_space = available_space
                        best_bin = bin_idx
                        best_ep = ep

            if best_ep is not None:
                placed_item = PlacedItem(item, best_ep.x, best_ep.y, best_ep.z)
                solver.bins[best_bin].append(placed_item)
                solver.extreme_points[best_bin].remove(best_ep)
                solver.update_extreme_points(item, best_ep, best_bin)
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


class SolomonRepair(ValidatedRepairOperator):
    """Safe Solomon repair with proper error handling"""

    def __init__(self, alpha: float = 0.5, beta: float = 0.5):
        super().__init__("Safe Solomon Repair")
        self.alpha = alpha
        self.beta = beta

    def _repair_implementation(self, solution: Solution, items: List[Item],
                               bin_template: Bin, rnd_state: np.random.RandomState) -> bool:
        if not items:
            return True

        try:
            solver = ExtremePointBinPacking3D(bin_template)
            solver.bins = copy.deepcopy(solution.bins)
            solver.extreme_points = copy.deepcopy(solution.extreme_points)

            remaining_items = items.copy()

            while remaining_items:
                best_selection = self._find_best_solomon_insertion(remaining_items, solver, bin_template)

                if best_selection is None:
                    # Place remaining items in new bins
                    for item in remaining_items:
                        self._place_in_new_bin(item, solver)
                    break

                item, placement = best_selection
                if not self._place_item_safely(item, placement, solver):
                    self._place_in_new_bin(item, solver)

                remaining_items.remove(item)

            solution.bins = solver.bins
            solution.extreme_points = solver.extreme_points
            return True

        except Exception:
            return False

    def _find_best_solomon_insertion(self, items: List[Item], solver: ExtremePointBinPacking3D,
                                     bin_template: Bin) -> Optional[Tuple[Item, Tuple]]:
        """Find best item-placement combination using Solomon criteria"""
        best_score = float('inf')
        best_selection = None

        for item in items:
            for bin_idx in range(len(solver.bins)):
                try:
                    ep, cost = solver._find_best_ep_bfd(item, bin_idx, MeritFunction.RESIDUAL_SPACE)
                    if ep is not None and hasattr(ep, 'x'):
                        cost_component = cost
                        urgency_component = item.volume / bin_template.volume
                        solomon_score = self.alpha * cost_component - self.beta * urgency_component

                        if solomon_score < best_score:
                            best_score = solomon_score
                            best_selection = (item, (cost, "existing", bin_idx, ep))
                except Exception:
                    continue

        return best_selection

    def _place_item_safely(self, item: Item, placement: Tuple,
                           solver: ExtremePointBinPacking3D) -> bool:
        """Place item safely"""
        try:
            cost, placement_type, bin_idx, ep = placement

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
            pass


class SmartRegretRepair(ValidatedRepairOperator):
    """Regret repair with better position evaluation"""

    def __init__(self, k: int = 3):
        super().__init__(f"Smart {k}-Regret Repair")
        self.k = k

    def _repair_implementation(self, solution: Solution, items: List[Item],
                               bin_template: Bin, rnd_state: np.random.RandomState) -> bool:
        if not items:
            return True

        solver = ExtremePointBinPacking3D(bin_template)
        solver.bins = solution.bins
        solver.extreme_points = solution.extreme_points

        remaining_items = items.copy()

        while remaining_items:
            best_item = None
            best_position = None
            best_regret = -float('inf')

            for item in remaining_items:
                insertion_costs = self._get_insertion_costs(item, solver, bin_template)
                regret = self._calculate_smart_regret(insertion_costs)

                if regret > best_regret:
                    best_regret = regret
                    best_item = item
                    best_position = insertion_costs[0] if insertion_costs else None

            if best_item is None or best_position is None:
                break

            # Insert item
            if best_position[0] == -1:  # New bin
                bin_idx = solver.add_new_bin()
                ep = solver.extreme_points[bin_idx][0]
            else:
                bin_idx, ep = best_position[0], best_position[1]

            placed_item = PlacedItem(best_item, ep.x, ep.y, ep.z)
            solver.bins[bin_idx].append(placed_item)
            solver.extreme_points[bin_idx].remove(ep)
            solver.update_extreme_points(best_item, ep, bin_idx)

            remaining_items.remove(best_item)

        solution.bins = solver.bins
        solution.extreme_points = solver.extreme_points
        return True

    def _get_insertion_costs(self, item: Item, solver: ExtremePointBinPacking3D,
                             bin_template: Bin) -> List[Tuple[float, int, ExtremePoint]]:
        """Get sorted insertion costs for an item"""
        costs = []

        # Existing bins
        for bin_idx in range(len(solver.bins)):
            ep, cost = solver._find_best_ep_bfd(item, bin_idx, MeritFunction.RESIDUAL_SPACE)
            if ep is not None:
                # Enhanced cost considering bin utilization
                current_volume = sum(pi.item.volume for pi in solver.bins[bin_idx])
                util_after = (current_volume + item.volume) / bin_template.volume
                adjusted_cost = cost - util_after * 10  # Prefer higher utilization
                costs.append((adjusted_cost, bin_idx, ep))

        # New bin option
        new_bin_cost = 1000.0 - (item.volume / bin_template.volume)
        costs.append((new_bin_cost, -1, None))

        return sorted(costs, key=lambda x: x[0])

    def _calculate_smart_regret(self, insertion_costs: List[Tuple]) -> float:
        """Calculate regret with smart weighting"""
        if len(insertion_costs) < 2:
            return 0.0

        # Weight regret by position quality
        best_cost = insertion_costs[0][0]

        if len(insertion_costs) >= self.k:
            kth_cost = insertion_costs[self.k - 1][0]
        else:
            kth_cost = insertion_costs[-1][0]

        base_regret = kth_cost - best_cost

        # Bonus for items that have very few good insertion options
        scarcity_bonus = 100.0 / len(insertion_costs) if len(insertion_costs) < 5 else 0

        return base_regret + scarcity_bonus