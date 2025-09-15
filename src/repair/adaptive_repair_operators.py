# -*- coding: utf-8 -*-
"""
Adaptive ALNS Repair Operators for 3D Bin Packing
Features intelligent insertion strategies based on research findings
Compatible with ValidatedRepairOperator framework

Adapted for 3D Bin Packing Problem using Extreme Point Heuristics

@author: Kreecha Puphaiboon
MIT License
"""


import random
import numpy as np
from typing import List, Tuple, Dict, Any
from collections import deque

# Import validation framework
from src.classes.ValidateOperatorManager import ValidatedRepairOperator
from src.classes.solution import Solution
from src.common import Item, Bin, PlacedItem, ExtremePoint


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


class AdaptiveRepairBase(ValidatedRepairOperator):
    """Base class for adaptive repair operators"""

    def __init__(self, name: str):
        super().__init__(name)
        self.performance_tracker = PerformanceTracker()

    def record_performance(self, success: bool, improvement: float = 0.0):
        """Record performance for learning"""
        self.performance_tracker.record_performance(success, improvement)



class AdaptiveRegretInsertRepair(AdaptiveRepairBase):
    """
    Regret-based insertion that considers the cost of not placing an item
    in its best position. Shows excellent performance in research.
    """

    def __init__(self, regret_degree=3):
        super().__init__("Adaptive Regret Insert Repair")
        self.regret_degree = regret_degree  # Consider top N positions

    def _repair_implementation(self, solution: Solution, items_to_repair: List[Item],
                               bin_template: Bin, rnd_state: np.random.RandomState) -> bool:
        """Insert items using regret-based strategy"""
        if not items_to_repair:
            return True

        # Work on copies to avoid modifying original solution until success
        import copy
        working_bins = copy.deepcopy(solution.bins)
        working_extreme_points = copy.deepcopy(solution.extreme_points) if hasattr(solution, 'extreme_points') else []

        # Sort items by volume (large first) for better initial placement
        items_sorted = sorted(items_to_repair, key=lambda x: x.volume, reverse=True)
        uninserted_items = []

        for item in items_sorted:
            best_positions = self._find_best_positions(working_bins, working_extreme_points, item, bin_template)

            if not best_positions:
                uninserted_items.append(item)
                continue

            # Calculate regret: difference between best and second-best positions
            if len(best_positions) >= 2:
                regret = best_positions[1][1] - best_positions[0][1]  # cost difference
                position_to_use = best_positions[0]  # Always use best for now
            else:
                position_to_use = best_positions[0]

            # Insert item at best position
            bin_idx, cost, position, is_new_bin = position_to_use
            if self._insert_item_at_position(working_bins, working_extreme_points, item, bin_idx, position, is_new_bin, bin_template):
                continue
            else:
                uninserted_items.append(item)

        success = len(uninserted_items) == 0

        # Only update original solution if completely successful
        if success:
            solution.bins = working_bins
            if hasattr(solution, 'extreme_points'):
                solution.extreme_points = working_extreme_points

        if uninserted_items:
            # print(f"{self.name}: Failed to insert {len(uninserted_items)} items: {[item.id for item in uninserted_items]}")
            # for item in uninserted_items:
            #     print(f"Uninserted item {item.id}: w={item.width}, d={item.depth}, h={item.height}, volume={item.volume}")
            pass

        return success

    def _try_insert_in_working_bin(self, working_bins: List[List[PlacedItem]], working_extreme_points: List[List[ExtremePoint]],
                                   item: Item, bin_idx: int, bin_template: Bin) -> bool:
        """Try to insert item in working bin at first available position"""
        if bin_idx >= len(working_extreme_points) or not working_extreme_points[bin_idx]:
            # print(f"No extreme points available for bin {bin_idx} for item {item.id}")
            return False

        for ep in working_extreme_points[bin_idx]:
            if self._can_fit_at_working_position(working_bins, bin_idx, item, (ep.x, ep.y, ep.z), bin_template):
                try:
                    x, y, z = ep.x, ep.y, ep.z
                    placed_item = PlacedItem(item, x, y, z)
                    working_bins[bin_idx].append(placed_item)
                    self._update_extreme_points_working(working_bins, working_extreme_points, bin_idx, placed_item, bin_template)
                    return True
                except Exception as e:
                    # print(f"Error inserting item {item.id} at position ({ep.x}, {ep.y}, {ep.z}) in bin {bin_idx}: {e}")
                    continue
        return False

    def _can_fit_at_working_position(self, working_bins: List[List[PlacedItem]], bin_idx: int, item: Item,
                                     position: Tuple, bin_template: Bin) -> bool:
        """Check if item can fit at position in working bin"""
        x, y, z = position
        epsilon = 1e-9
        if (x + item.width > bin_template.width + epsilon or
                y + item.depth > bin_template.depth + epsilon or
                z + item.height > bin_template.height + epsilon):
            # print(f"Item {item.id} cannot fit at {position} in bin {bin_idx}: exceeds bin dimensions "
            #       f"(w={item.width}, d={item.depth}, h={item.height}, bin={bin_template.width}x{bin_template.depth}x{bin_template.height})")
            #
            return False

        for placed_item in working_bins[bin_idx]:
            if self._items_overlap(item, position, placed_item):
                # print(f"Item {item.id} overlaps with {placed_item.item.id} at {position} in bin {bin_idx}")
                return False

        return True

    def _try_new_working_bin(self, working_bins: List[List[PlacedItem]], working_extreme_points: List[List[ExtremePoint]],
                             item: Item, bin_template: Bin) -> bool:
        """Try to create a new working bin for the item"""
        try:
            new_bin_items = []
            placed_item = PlacedItem(item, 0, 0, 0)
            new_bin_items.append(placed_item)
            working_bins.append(new_bin_items)

            if working_extreme_points is not None:
                new_extreme_points = [
                    ExtremePoint(item.width, 0, 0),
                    ExtremePoint(0, item.depth, 0),
                    ExtremePoint(0, 0, item.height)
                ]
                working_extreme_points.append(new_extreme_points)

            return True
        except Exception as e:
            print(f"Error creating new bin in working copy for item {item.id}: {e}")
            return False

    def _update_extreme_points_working(self, working_bins: List[List[PlacedItem]], working_extreme_points: List[List[ExtremePoint]],
                                       bin_idx: int, placed_item: PlacedItem, bin_template: Bin):
        """Update extreme points after item insertion in working copy"""
        x, y, z = placed_item.x, placed_item.y, placed_item.z
        w, d, h = placed_item.item.width, placed_item.item.depth, placed_item.item.height

        new_points = [
            (x + w, y, z),
            (x, y + d, z),
            (x, y, z + h),
            (x + w, y + d, z),
            (x + w, y, z + h),
            (x, y + d, z + h),
            (x + w, y + d, z + h)
        ]

        for point in new_points:
            if (point[0] <= bin_template.width + 1e-9 and
                    point[1] <= bin_template.depth + 1e-9 and
                    point[2] <= bin_template.height + 1e-9):
                is_valid = True
                for pi in working_bins[bin_idx]:
                    if (pi.x < point[0] < pi.x + pi.item.width and
                        pi.y < point[1] < pi.y + pi.item.depth and
                        pi.z < point[2] < pi.z + pi.item.height):
                        is_valid = False
                        break
                if is_valid:
                    working_extreme_points[bin_idx].append(ExtremePoint(point[0], point[1], point[2]))
                else:
                    # print(f"Invalid extreme point {point} for bin {bin_idx}: inside existing item")
                    pass
            else:
                # print(f"Invalid extreme point {point} for bin {bin_idx}: exceeds bin dimensions")
                pass

    def _try_insert_in_bin(self, working_bins: List[List[PlacedItem]], working_extreme_points: List[List[ExtremePoint]],
                           item: Item, bin_idx: int, bin_template: Bin) -> bool:
        """Try to insert item in bin at first available position"""
        if bin_idx >= len(working_extreme_points) or not working_extreme_points[bin_idx]:
            return False

        for ep in working_extreme_points[bin_idx]:
            if self._can_fit_at_position(working_bins, bin_idx, item, (ep.x, ep.y, ep.z), bin_template):
                try:
                    x, y, z = ep.x, ep.y, ep.z
                    placed_item = PlacedItem(item, x, y, z)
                    working_bins[bin_idx].append(placed_item)
                    return True
                except:
                    continue
        return False

    def _can_fit_at_position(self, working_bins: List[List[PlacedItem]], bin_idx: int, item: Item,
                             position: Tuple, bin_template: Bin) -> bool:
        """Check if item can fit at position"""
        x, y, z = position
        epsilon = 1e-9
        if (x + item.width > bin_template.width + epsilon or
                y + item.depth > bin_template.depth + epsilon or
                z + item.height > bin_template.height + epsilon):
            # print(f"Item {item.id} cannot fit at {position} in bin {bin_idx}: exceeds bin dimensions "
            #       f"(w={item.width}, d={item.depth}, h={item.height}, bin={bin_template.width}x{bin_template.depth}x{bin_template.height})")
            return False

        # Check overlap with existing items
        for placed_item in working_bins[bin_idx]:
            if self._items_overlap(item, position, placed_item):
                # print(f"Item {item.id} overlaps with {placed_item.item.id} at {position} in bin {bin_idx}")
                return False

        return True

    def _items_overlap(self, item: Item, position: Tuple, placed_item: PlacedItem) -> bool:
        """Check if items overlap"""
        x1, y1, z1 = position
        x2, y2, z2 = placed_item.x, placed_item.y, placed_item.z

        return not (x1 + item.width <= x2 or x2 + placed_item.item.width <= x1 or
                    y1 + item.depth <= y2 or y2 + placed_item.item.depth <= y1 or
                    z1 + item.height <= z2 or z2 + placed_item.item.height <= z1)

    def _find_best_positions(self, working_bins: List[List[PlacedItem]], working_extreme_points: List[List[ExtremePoint]],
                             item: Item, bin_template: Bin) -> List[Tuple]:
        """Find best insertion positions with their costs"""
        positions = []

        # Check existing bins
        for bin_idx, bin_items in enumerate(working_bins):
            if not working_extreme_points or bin_idx >= len(working_extreme_points):
                continue

            for ep in working_extreme_points[bin_idx]:
                if self._can_fit_at_position(working_bins, bin_idx, item, (ep.x, ep.y, ep.z), bin_template):
                    cost = self._calculate_insertion_cost(working_bins, bin_idx, item, (ep.x, ep.y, ep.z), bin_template)
                    positions.append((bin_idx, cost, (ep.x, ep.y, ep.z), False))

        # Consider new bin
        if bin_template:
            cost = self._calculate_new_bin_cost(item)
            positions.append((-1, cost, (0, 0, 0), True))  # -1 indicates new bin

        # Sort by cost and return top candidates
        positions.sort(key=lambda x: x[1])
        return positions[:self.regret_degree]

    def _calculate_insertion_cost(self, working_bins: List[List[PlacedItem]], bin_idx: int, item: Item,
                                  position: Tuple, bin_template: Bin) -> float:
        """Calculate the cost of inserting item at position"""
        # Simple cost: remaining space after insertion
        used_volume = sum(pi.item.volume for pi in working_bins[bin_idx]) if bin_idx >= 0 else 0
        remaining_space = bin_template.volume - used_volume
        item_volume = item.volume
        return 1.0 / (remaining_space - item_volume + 1)  # Lower remaining space = higher cost

    def _calculate_new_bin_cost(self, item: Item) -> float:
        """Cost of opening a new bin"""
        return 100.0  # High cost to discourage new bins

    def _insert_item_at_position(self, working_bins: List[List[PlacedItem]], working_extreme_points: List[List[ExtremePoint]],
                                 item: Item, bin_idx: int, position: Tuple, is_new_bin: bool, bin_template: Bin) -> bool:
        """Insert item at specific position"""
        try:
            if is_new_bin:
                # Create new bin
                new_bin_items = []
                new_extreme_points = [ExtremePoint(0, 0, 0)]  # Initial point at origin
                working_bins.append(new_bin_items)
                if working_extreme_points is not None:
                    working_extreme_points.append(new_extreme_points)
                bin_idx = len(working_bins) - 1

            # Add item to bin
            x, y, z = position
            placed_item = PlacedItem(item, x, y, z)
            working_bins[bin_idx].append(placed_item)

            # Update extreme points if available
            if working_extreme_points and bin_idx < len(working_extreme_points):
                self._update_extreme_points_working(working_bins, working_extreme_points, bin_idx, placed_item, bin_template)

            return True
        except Exception as e:
            print(f"Error inserting item {item.id} at position {position} in bin {bin_idx}: {e}")
            return False


class AdaptiveFirstFitDecreasingRepair(AdaptiveRepairBase):
    """
    First-fit decreasing that sorts items by size and places in first available position.
    Simple but effective baseline repair operator.
    """

    def __init__(self):
        super().__init__("Adaptive First Fit Decreasing Repair")

    def _repair_implementation(self, solution: Solution, items_to_repair: List[Item],
                               bin_template: Bin, rnd_state: np.random.RandomState) -> bool:
        """Insert items using first-fit decreasing strategy"""
        if not items_to_repair:
            return True

        # Work on copies to avoid modifying original solution until success
        import copy
        working_bins = copy.deepcopy(solution.bins)
        working_extreme_points = copy.deepcopy(solution.extreme_points) if hasattr(solution, 'extreme_points') else []

        # Sort items by volume (decreasing)
        items_sorted = sorted(items_to_repair, key=lambda x: x.volume, reverse=True)

        uninserted_items = []

        for item in items_sorted:
            placed = False

            # Try existing bins first
            for bin_idx in range(len(working_bins)):
                if self._try_insert_in_working_bin(working_bins, working_extreme_points, item, bin_idx,
                                                   bin_template):
                    placed = True
                    break

            # Try new bin if not placed
            if not placed and self._try_new_working_bin(working_bins, working_extreme_points, item, bin_template):
                placed = True

            if not placed:
                uninserted_items.append(item)

        success = len(uninserted_items) == 0

        # Only update original solution if completely successful
        if success:
            solution.bins = working_bins
            if hasattr(solution, 'extreme_points'):
                solution.extreme_points = working_extreme_points

        if uninserted_items:
            # print(f"{self.name}: Failed to insert {len(uninserted_items)} items: {[item.id for item in uninserted_items]}")
            # for item in uninserted_items:
            #     print(f"Uninserted item {item.id}: w={item.width}, d={item.depth}, h={item.height}, volume={item.volume}")
            pass

        return success

    def _try_insert_in_working_bin(self, working_bins: List[List[PlacedItem]], working_extreme_points: List[List[ExtremePoint]],
                                   item: Item, bin_idx: int, bin_template: Bin) -> bool:
        """Try to insert item in working bin at first available position"""
        if bin_idx >= len(working_extreme_points) or not working_extreme_points[bin_idx]:
            # print(f"No extreme points available for bin {bin_idx} for item {item.id}")
            return False

        for ep in working_extreme_points[bin_idx]:
            if self._can_fit_at_working_position(working_bins, bin_idx, item, (ep.x, ep.y, ep.z), bin_template):
                try:
                    x, y, z = ep.x, ep.y, ep.z
                    placed_item = PlacedItem(item, x, y, z)
                    working_bins[bin_idx].append(placed_item)
                    self._update_extreme_points_working(working_bins, working_extreme_points, bin_idx, placed_item, bin_template)
                    return True
                except Exception as e:
                    print(f"Error inserting item {item.id} at position ({ep.x}, {ep.y}, {ep.z}) in bin {bin_idx}: {e}")
                    continue
        return False

    def _can_fit_at_working_position(self, working_bins: List[List[PlacedItem]], bin_idx: int, item: Item,
                                     position: Tuple, bin_template: Bin) -> bool:
        """Check if item can fit at position in working bin"""
        x, y, z = position
        epsilon = 1e-9
        if (x + item.width > bin_template.width + epsilon or
                y + item.depth > bin_template.depth + epsilon or
                z + item.height > bin_template.height + epsilon):
            # print(f"Item {item.id} cannot fit at {position} in bin {bin_idx}: exceeds bin dimensions "
            #       f"(w={item.width}, d={item.depth}, h={item.height}, bin={bin_template.width}x{bin_template.depth}x{bin_template.height})")
            return False

        for placed_item in working_bins[bin_idx]:
            if self._items_overlap(item, position, placed_item):
                # print(f"Item {item.id} overlaps with {placed_item.item.id} at {position} in bin {bin_idx}")
                return False

        return True

    def _try_new_working_bin(self, working_bins: List[List[PlacedItem]], working_extreme_points: List[List[ExtremePoint]],
                             item: Item, bin_template: Bin) -> bool:
        """Try to create a new working bin for the item"""
        try:
            new_bin_items = []
            placed_item = PlacedItem(item, 0, 0, 0)
            new_bin_items.append(placed_item)
            working_bins.append(new_bin_items)

            if working_extreme_points is not None:
                new_extreme_points = [
                    ExtremePoint(item.width, 0, 0),
                    ExtremePoint(0, item.depth, 0),
                    ExtremePoint(0, 0, item.height)
                ]
                working_extreme_points.append(new_extreme_points)

            return True
        except Exception as e:
            print(f"Error creating new bin in working copy for item {item.id}: {e}")
            return False

    def _update_extreme_points_working(self, working_bins: List[List[PlacedItem]], working_extreme_points: List[List[ExtremePoint]],
                                       bin_idx: int, placed_item: PlacedItem, bin_template: Bin):
        """Update extreme points after item insertion in working copy"""
        x, y, z = placed_item.x, placed_item.y, placed_item.z
        w, d, h = placed_item.item.width, placed_item.item.depth, placed_item.item.height

        new_points = [
            (x + w, y, z),
            (x, y + d, z),
            (x, y, z + h),
            (x + w, y + d, z),
            (x + w, y, z + h),
            (x, y + d, z + h),
            (x + w, y + d, z + h)
        ]

        for point in new_points:
            if (point[0] <= bin_template.width + 1e-9 and
                    point[1] <= bin_template.depth + 1e-9 and
                    point[2] <= bin_template.height + 1e-9):
                is_valid = True
                for pi in working_bins[bin_idx]:
                    if (pi.x < point[0] < pi.x + pi.item.width and
                        pi.y < point[1] < pi.y + pi.item.depth and
                        pi.z < point[2] < pi.z + pi.item.height):
                        is_valid = False
                        break
                if is_valid:
                    working_extreme_points[bin_idx].append(ExtremePoint(point[0], point[1], point[2]))
                else:
                    # print(f"Invalid extreme point {point} for bin {bin_idx}: inside existing item")
                    pass
            else:
                # print(f"Invalid extreme point {point} for bin {bin_idx}: exceeds bin dimensions")
                pass

    def _try_insert_in_bin(self, solution: Solution, item: Item, bin_idx: int) -> bool:
        """Try to insert item in bin at first available position"""
        if not hasattr(solution, 'extreme_points') or bin_idx >= len(solution.extreme_points):
            return False

        for ep in solution.extreme_points[bin_idx]:
            if self._can_fit_at_position(solution, bin_idx, item, (ep.x, ep.y, ep.z)):
                try:
                    x, y, z = ep.x, ep.y, ep.z
                    placed_item = PlacedItem(item, x, y, z)
                    solution.bins[bin_idx].append(placed_item)
                    return True
                except:
                    continue
        return False

    def _can_fit_at_position(self, solution: Solution, bin_idx: int, item: Item, position: Tuple) -> bool:
        """Check if item can fit at position"""
        x, y, z = position
        if (x + item.width > solution.bin_template.width or
                y + item.depth > solution.bin_template.depth or
                z + item.height > solution.bin_template.height):
            return False

        # Check overlap with existing items
        for placed_item in solution.bins[bin_idx]:
            if self._items_overlap(item, position, placed_item):
                return False

        return True

    def _items_overlap(self, item: Item, position: Tuple, placed_item: PlacedItem) -> bool:
        """Check if items overlap"""
        x1, y1, z1 = position
        x2, y2, z2 = placed_item.x, placed_item.y, placed_item.z

        return not (x1 + item.width <= x2 or x2 + placed_item.item.width <= x1 or
                    y1 + item.depth <= y2 or y2 + placed_item.item.depth <= y1 or
                    z1 + item.height <= z2 or z2 + placed_item.item.height <= z1)

    def _try_new_bin(self, solution: Solution, item: Item, bin_template: Bin) -> bool:
        """Try to create a new bin for the item"""
        try:
            # Create new bin
            new_bin_items = []
            placed_item = PlacedItem(item, 0, 0, 0)
            new_bin_items.append(placed_item)
            solution.bins.append(new_bin_items)

            # Add extreme points if solution supports it
            if hasattr(solution, 'extreme_points'):
                new_extreme_points = [ExtremePoint((item.width, 0, 0)),
                                      ExtremePoint((0, item.depth, 0)),
                                      ExtremePoint((0, 0, item.height))]
                solution.extreme_points.append(new_extreme_points)

            return True
        except:
            return False

class AdaptiveBestFitRepair(AdaptiveRepairBase):
    """
    Best-fit insertion that places items in the tightest available space.
    Good performance and simple to implement.
    """

    def __init__(self):
        super().__init__("Adaptive Best Fit Repair")

    def _repair_implementation(self, solution: Solution, items_to_repair: List[Item],
                               bin_template: Bin, rnd_state: np.random.RandomState) -> bool:
        """Insert items using best-fit strategy"""
        if not items_to_repair:
            return True

        # Work on copies to avoid modifying original solution until success
        import copy
        working_bins = copy.deepcopy(solution.bins)
        working_extreme_points = copy.deepcopy(solution.extreme_points) if hasattr(solution, 'extreme_points') else []

        # Sort items by volume (large first)
        items_sorted = sorted(items_to_repair, key=lambda x: x.volume, reverse=True)
        uninserted_items = []

        for item in items_sorted:
            if not self._try_insert_item_working(working_bins, working_extreme_points, item, bin_template):
                uninserted_items.append(item)

        success = len(uninserted_items) == 0

        # Only update original solution if completely successful
        if success:
            solution.bins = working_bins
            if hasattr(solution, 'extreme_points'):
                solution.extreme_points = working_extreme_points

        if uninserted_items:
            # print(f"{self.name}: Failed to insert {len(uninserted_items)} items: {[item.id for item in uninserted_items]}")
            pass

        return success

    def _try_insert_item_working(self, working_bins: List[List[PlacedItem]],
                                 working_extreme_points: List[List[ExtremePoint]],
                                 item: Item, bin_template: Bin) -> bool:
        """Try to insert item using best-fit strategy in working copy"""
        best_fit = self._find_best_fit_working(working_bins, working_extreme_points, item, bin_template)

        if best_fit:
            bin_idx, position = best_fit
            return self._insert_at_position_working(working_bins, working_extreme_points, item, bin_idx, position,
                                                    bin_template)

        # Try creating new bin
        return self._try_new_bin_working(working_bins, working_extreme_points, item, bin_template)

    def _find_best_fit_working(self, working_bins: List[List[PlacedItem]],
                               working_extreme_points: List[List[ExtremePoint]],
                               item: Item, bin_template: Bin) -> Tuple:
        """Find the tightest fit for the item in working copy"""
        best_fit = None
        best_waste = float('inf')

        for bin_idx, bin_items in enumerate(working_bins):
            if not working_extreme_points or bin_idx >= len(working_extreme_points):
                continue

            for ep in working_extreme_points[bin_idx]:
                if self._can_fit_at_position_working(working_bins, bin_idx, item, (ep.x, ep.y, ep.z), bin_template):
                    waste = self._calculate_waste_working(working_bins, bin_idx, item, bin_template)
                    if waste < best_waste:
                        best_waste = waste
                        best_fit = (bin_idx, (ep.x, ep.y, ep.z))

        return best_fit

    def _can_fit_at_position_working(self, working_bins: List[List[PlacedItem]], bin_idx: int, item: Item,
                                     position: Tuple, bin_template: Bin) -> bool:
        """Check if item can fit at position in working copy"""
        x, y, z = position
        epsilon = 1e-9
        if (x + item.width > bin_template.width + epsilon or
                y + item.depth > bin_template.depth + epsilon or
                z + item.height > bin_template.height + epsilon):
            # print(f"Item {item.id} cannot fit at {position} in bin {bin_idx}: exceeds bin dimensions "
            #      f"(w={item.width}, d={item.depth}, h={item.height}, bin={bin_template.width}x{bin_template.depth}x{bin_template.height})")
            return False

        for placed_item in working_bins[bin_idx]:
            if self._items_overlap(item, position, placed_item):
                # print(f"Item {item.id} overlaps with {placed_item.item.id} at {position} in bin {bin_idx}")
                return False

        return True

    def _calculate_waste_working(self, working_bins: List[List[PlacedItem]], bin_idx: int, item: Item,
                                 bin_template: Bin) -> float:
        """Calculate wasted space for this insertion in working copy"""
        used_volume = sum(pi.item.volume for pi in working_bins[bin_idx])
        remaining_space = bin_template.volume - used_volume
        return remaining_space - item.volume

    def _insert_at_position_working(self, working_bins: List[List[PlacedItem]],
                                    working_extreme_points: List[List[ExtremePoint]],
                                    item: Item, bin_idx: int, position: Tuple, bin_template: Bin) -> bool:
        """Insert item at position in working copy"""
        try:
            x, y, z = position
            placed_item = PlacedItem(item, x, y, z)
            working_bins[bin_idx].append(placed_item)
            if working_extreme_points and bin_idx < len(working_extreme_points):
                self._update_extreme_points_working(working_bins, working_extreme_points, bin_idx, placed_item,
                                                    bin_template)
            return True
        except Exception as e:
            print(f"Error inserting item in working copy: {e}")
            return False

    def _try_new_bin_working(self, working_bins: List[List[PlacedItem]],
                             working_extreme_points: List[List[ExtremePoint]],
                             item: Item, bin_template: Bin) -> bool:
        """Try to create a new bin for the item in working copy"""
        try:
            new_bin_items = []
            placed_item = PlacedItem(item, 0, 0, 0)
            new_bin_items.append(placed_item)
            working_bins.append(new_bin_items)

            if working_extreme_points is not None:
                new_extreme_points = [
                    ExtremePoint(item.width, 0, 0),
                    ExtremePoint(0, item.depth, 0),
                    ExtremePoint(0, 0, item.height)
                ]
                working_extreme_points.append(new_extreme_points)

            return True
        except Exception as e:
            print(f"Error creating new bin in working copy for item {item.id}: {e}")
            return False

    def _update_extreme_points_working(self, working_bins: List[List[PlacedItem]],
                                       working_extreme_points: List[List[ExtremePoint]],
                                       bin_idx: int, placed_item: PlacedItem, bin_template: Bin):
        """Update extreme points after item insertion in working copy"""
        x, y, z = placed_item.x, placed_item.y, placed_item.z
        w, d, h = placed_item.item.width, placed_item.item.depth, placed_item.item.height

        new_points = [
            (x + w, y, z),
            (x, y + d, z),
            (x, y, z + h),
            (x + w, y + d, z),
            (x + w, y, z + h),
            (x, y + d, z + h),
            (x + w, y + d, z + h)
        ]

        for point in new_points:
            if (point[0] <= bin_template.width and
                    point[1] <= bin_template.depth and
                    point[2] <= bin_template.height):
                working_extreme_points[bin_idx].append(ExtremePoint(point[0], point[1], point[2]))

    def _try_insert_item(self, solution: Solution, item: Item, bin_template: Bin) -> bool:
        """Try to insert item using best-fit strategy"""
        best_fit = self._find_best_fit(solution, item)

        if best_fit:
            bin_idx, position = best_fit
            return self._insert_at_position(solution, item, bin_idx, position)

        # Try creating new bin
        return self._try_new_bin(solution, item, bin_template)

    def _find_best_fit(self, solution: Solution, item: Item) -> Tuple:
        """Find the tightest fit for the item"""
        best_fit = None
        best_waste = float('inf')

        for bin_idx, bin_items in enumerate(solution.bins):
            if not hasattr(solution, 'extreme_points') or bin_idx >= len(solution.extreme_points):
                continue

            for ep in solution.extreme_points[bin_idx]:
                if self._can_fit_at_position(solution, bin_idx, item, (ep.x, ep.y, ep.z)):
                # if self._can_fit_at_position(solution, bin_idx, item, ep.position):
                    # Calculate waste (unused space)
                    waste = self._calculate_waste(solution, bin_idx, item)
                    if waste < best_waste:
                        best_waste = waste
                        best_fit = (bin_idx,(ep.x, ep.y, ep.z))

        return best_fit

    def _can_fit_at_position(self, solution: Solution, bin_idx: int, item: Item, position: Tuple) -> bool:
        """Check if item can fit at position"""
        x, y, z = position
        if (x + item.width > solution.bin_template.width or
                y + item.depth > solution.bin_template.depth or
                z + item.height > solution.bin_template.height):
            return False

        # Check overlap with existing items
        for placed_item in solution.bins[bin_idx]:
            if self._items_overlap(item, position, placed_item):
                return False

        return True

    def _items_overlap(self, item: Item, position: Tuple, placed_item: PlacedItem) -> bool:
        x1, y1, z1 = position
        x2, y2, z2 = placed_item.x, placed_item.y, placed_item.z
        epsilon = 1e-9
        return not (x1 + item.width <= x2 + epsilon or
                    x2 + placed_item.item.width <= x1 + epsilon or
                    y1 + item.depth <= y2 + epsilon or
                    y2 + placed_item.item.depth <= y1 + epsilon or
                    z1 + item.height <= z2 + epsilon or
                    z2 + placed_item.item.height <= z1 + epsilon)

    def _calculate_waste(self, solution: Solution, bin_idx: int, item: Item) -> float:
        """Calculate wasted space for this insertion"""
        used_volume = sum(pi.item.volume for pi in solution.bins[bin_idx])
        remaining_space = solution.bin_template.volume - used_volume
        return remaining_space - item.volume

    def _insert_at_position(self, solution: Solution, item: Item, bin_idx: int, position: Tuple) -> bool:
        """Insert item at position"""
        try:
            x, y, z = position
            placed_item = PlacedItem(item, x, y, z)
            solution.bins[bin_idx].append(placed_item)
            return True
        except:
            return False

    def _try_new_bin(self, solution: Solution, item: Item, bin_template: Bin) -> bool:
        """Try to create a new bin for the item"""
        try:
            # Create new bin
            new_bin_items = []
            placed_item = PlacedItem(item, 0, 0, 0)
            new_bin_items.append(placed_item)
            solution.bins.append(new_bin_items)

            # Add extreme points if solution supports it
            if hasattr(solution, 'extreme_points'):
                new_extreme_points = [ExtremePoint((item.width, 0, 0)),
                                      ExtremePoint((0, item.depth, 0)),
                                      ExtremePoint((0, 0, item.height))]
                solution.extreme_points.append(new_extreme_points)

            return True
        except:
            return False


def get_adaptive_repair_operators():
    """
    Factory function to get all adaptive repair operators
    Returns: List of repair operators
    """

    repair_operators = [
        AdaptiveRegretInsertRepair(regret_degree=3),
        AdaptiveBestFitRepair(),
        AdaptiveFirstFitDecreasingRepair()
    ]

    return repair_operators



if __name__ == "__main__":
    # Example usage
    repair_ops = get_adaptive_repair_operators()

    print("Adaptive Repair Operators:")
    for op in repair_ops:
        print(f"  - {op.name}")

    print("\nFeatures:")
    print("  - Regret-based insertion (research-proven best performance)")
    print("  - Best-fit and first-fit strategies")
    print("  - Performance learning and adaptation")
    print("  - Extends ValidatedRepairOperator for item conservation")
    print("  - Implements _repair_implementation() method")