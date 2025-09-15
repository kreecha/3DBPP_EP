import numpy as np
import random
import copy
import math
from typing import List, Tuple, Set
from dataclasses import dataclass

# Import shared classes (assuming these are available from your existing code)
from src.common import Item, Bin, PlacedItem, ExtremePoint, SortingRule, MeritFunction
from src.ExtremePointHeuristic import ExtremePointBinPacking3D
from src.classes.ValidateOperatorManager import ValidatedDestroyOperator, ValidatedRepairOperator


@dataclass
class Solution:
    """Represents a complete packing solution"""
    bins: List[List[PlacedItem]]
    extreme_points: List[List[ExtremePoint]]
    bin_template: Bin

    def copy(self):
        """Create a deep copy of the solution"""
        return Solution(
            bins=copy.deepcopy(self.bins),
            extreme_points=copy.deepcopy(self.extreme_points),
            bin_template=self.bin_template
        )

    @property
    def num_bins(self) -> int:
        """Number of bins used in the solution"""
        return len(self.bins)

    @property
    def average_utilization(self) -> float:
        """Average bin utilization"""
        utilizations = self.get_bin_utilizations()
        return np.mean(utilizations) if utilizations else 0.0

    def get_bin_utilizations(self) -> List[float]:
        """Get list of individual bin utilizations (FIXED VERSION)"""
        utilizations = []
        for bin_items in self.bins:
            used_volume = sum(item.item.volume for item in bin_items)
            utilization = used_volume / self.bin_template.volume
            utilizations.append(utilization)
        return utilizations

    def get_total_utilization(self) -> float:
        """Get sum of all bin utilizations (if you need the old behavior)"""
        utilizations = self.get_bin_utilizations()
        return np.sum(utilizations) if utilizations else 0.0

    def get_all_items(self) -> List[Item]:
        """Get all items in the solution"""
        return [placed_item.item for bin_items in self.bins for placed_item in bin_items]

    def get_items_count(self) -> int:
        """Get total number of items in solution"""
        return len(self.get_all_items())

    def get_solution_statistics(self) -> dict:
        """Get comprehensive solution statistics"""
        utilizations = self.get_bin_utilizations()

        if not utilizations:
            return {
                'num_bins': 0,
                'total_items': 0,
                'avg_utilization': 0.0,
                'min_utilization': 0.0,
                'max_utilization': 0.0,
                'utilization_variance': 0.0,
                'utilization_std': 0.0,
                'total_volume_used': 0.0,
                'total_volume_wasted': 0.0
            }

        total_volume_capacity = len(self.bins) * self.bin_template.volume
        total_volume_used = sum(
            sum(pi.item.volume for pi in bin_items)
            for bin_items in self.bins
        )

        return {
            'num_bins': self.num_bins,
            'total_items': self.get_items_count(),
            'avg_utilization': np.mean(utilizations),
            'min_utilization': np.min(utilizations),
            'max_utilization': np.max(utilizations),
            'utilization_variance': np.var(utilizations),
            'utilization_std': np.std(utilizations),
            'total_volume_used': total_volume_used,
            'total_volume_wasted': total_volume_capacity - total_volume_used
        }

    def is_solution_valid(self) -> bool:
        """Check if the solution is valid (no overlaps, items fit in bins)"""
        for bin_idx, bin_items in enumerate(self.bins):
            # Check each item fits in bin
            for placed_item in bin_items:
                if (placed_item.x + placed_item.item.width > self.bin_template.width + 1e-9 or
                        placed_item.y + placed_item.item.depth > self.bin_template.depth + 1e-9 or
                        placed_item.z + placed_item.item.height > self.bin_template.height + 1e-9):
                    return False

            # Check no overlaps between items in same bin
            for i, item1 in enumerate(bin_items):
                for j, item2 in enumerate(bin_items[i + 1:], i + 1):
                    if self._items_overlap(item1, item2):
                        return False
        return True

    def _items_overlap(self, item1: PlacedItem, item2: PlacedItem) -> bool:
        """Check if two placed items overlap"""
        return not (
                item1.x + item1.item.width <= item2.x + 1e-9 or
                item2.x + item2.item.width <= item1.x + 1e-9 or
                item1.y + item1.item.depth <= item2.y + 1e-9 or
                item2.y + item2.item.depth <= item1.y + 1e-9 or
                item1.z + item1.item.height <= item2.z + 1e-9 or
                item2.z + item2.item.height <= item1.z + 1e-9
        )

    def remove_items(self, items_to_remove: List[Item]):
        """Remove specified items from the solution and repack remaining items"""
        items_to_remove_set = set(items_to_remove)  # For efficient lookup

        for bin_idx in range(len(self.bins) - 1, -1, -1):  # Iterate backwards to safely modify list
            bin_items = self.bins[bin_idx]
            # Collect items to keep
            items_to_keep = [pi for pi in bin_items if pi.item not in items_to_remove_set]

            if len(items_to_keep) < len(bin_items):  # Only re-pack if items were removed
                self.bins[bin_idx] = []  # Clear the bin
                self.extreme_points[bin_idx] = [ExtremePoint(0, 0, 0,
                                                             self.bin_template.width,
                                                             self.bin_template.depth,
                                                             self.bin_template.height)]

                if items_to_keep:  # Re-pack only if there are items to keep
                    solver = ExtremePointBinPacking3D(self.bin_template)
                    solver.bins = [[]]
                    solver.extreme_points = [self.extreme_points[bin_idx]]

                    for pi in items_to_keep:
                        item = pi.item
                        ep = solver._find_best_ep_ffd(item, 0)
                        if ep:
                            solver.bins[0].append(PlacedItem(item, ep.x, ep.y, ep.z))
                            solver.extreme_points[0].remove(ep)
                            solver.update_extreme_points(item, ep, 0)
                        else:
                            # Instead of removing, add to a new bin to avoid loss
                            new_bin_idx = solver.add_new_bin()
                            ep = solver.extreme_points[new_bin_idx][0]
                            solver.bins[new_bin_idx].append(PlacedItem(item, ep.x, ep.y, ep.z))
                            solver.extreme_points[new_bin_idx].remove(ep)
                            solver.update_extreme_points(item, ep, new_bin_idx)

                    # Update bin and extreme points
                    self.bins[bin_idx] = solver.bins[0]
                    self.extreme_points[bin_idx] = solver.extreme_points[0]

                    # Append any new bins created
                    if len(solver.bins) > 1:
                        self.bins.extend(solver.bins[1:])
                        self.extreme_points.extend(solver.extreme_points[1:])

            if not self.bins[bin_idx]:  # Remove empty bins
                self.bins.pop(bin_idx)
                self.extreme_points.pop(bin_idx)

    def compact_bins(self):
        """Remove empty bins and consolidate sparse bins if possible"""
        # Remove empty bins
        non_empty_indices = [i for i, bin_items in enumerate(self.bins) if bin_items]
        self.bins = [self.bins[i] for i in non_empty_indices]
        self.extreme_points = [self.extreme_points[i] for i in non_empty_indices]

        # TODO: Could add logic here to try consolidating very sparse bins
        # This would involve checking if items from multiple sparse bins can fit into fewer bins

    def print_solution_summary(self):
        """Print a summary of the solution"""
        stats = self.get_solution_statistics()
        print(f"Solution Summary:")
        print(f"  Bins used: {stats['num_bins']}")
        print(f"  Total items: {stats['total_items']}")
        print(f"  Average utilization: {stats['avg_utilization']:.3f}")
        print(f"  Utilization range: [{stats['min_utilization']:.3f}, {stats['max_utilization']:.3f}]")
        print(f"  Utilization variance: {stats['utilization_variance']:.6f}")
        print(f"  Total volume wasted: {stats['total_volume_wasted']:.1f}")
        print(f"  Solution valid: {self.is_solution_valid()}")


@dataclass
class Old_Solution:
    """Represents a complete packing solution"""
    bins: List[List[PlacedItem]]
    extreme_points: List[List[ExtremePoint]]
    bin_template: Bin

    def copy(self):
        """Create a deep copy of the solution"""
        return Solution(
            bins=copy.deepcopy(self.bins),
            extreme_points=copy.deepcopy(self.extreme_points),
            bin_template=self.bin_template
        )

    @property
    def num_bins(self) -> int:
        """Number of bins used in the solution"""
        return len(self.bins)

    @property
    def average_utilization(self) -> float:
        """Average bin utilization"""
        utilizations = []
        for bin_items in self.bins:
            used_volume = sum(item.item.volume for item in bin_items)
            utilizations.append(used_volume / self.bin_template.volume)
        return np.mean(utilizations) if utilizations else 0.0

    def get_bin_utilizations(self) -> List[float]:  # Correct return type
        utilizations = []
        for bin_items in self.bins:
            used_volume = sum(item.item.volume for item in bin_items)
            utilization = used_volume / self.bin_template.volume
            utilizations.append(utilization)
        return utilizations  # Returns list of individual utilizations


    def get_all_items(self) -> List[Item]:
        """Get all items in the solution"""
        return [placed_item.item for bin_items in self.bins for placed_item in bin_items]

    def get_items_count(self) -> int:
        """Get total number of items in solution"""
        return len(self.get_all_items())

    def remove_items(self, items_to_remove: List[Item]):
        items_to_remove_set = set(items_to_remove)  # For efficient lookup
        for bin_idx in range(len(self.bins) - 1, -1, -1):  # Iterate backwards to safely modify list
            bin_items = self.bins[bin_idx]
            # Collect items to keep
            items_to_keep = [pi for pi in bin_items if pi.item not in items_to_remove_set]
            if len(items_to_keep) < len(bin_items):  # Only re-pack if items were removed
                self.bins[bin_idx] = []  # Clear the bin
                self.extreme_points[bin_idx] = [ExtremePoint(0, 0, 0,
                                                             self.bin_template.width,
                                                             self.bin_template.depth,
                                                             self.bin_template.height)]
                if items_to_keep:  # Re-pack only if there are items to keep
                    solver = ExtremePointBinPacking3D(self.bin_template)
                    solver.bins = [[]]
                    solver.extreme_points = [self.extreme_points[bin_idx]]
                    for pi in items_to_keep:
                        item = pi.item
                        ep = solver._find_best_ep_ffd(item, 0)
                        if ep:
                            solver.bins[0].append(PlacedItem(item, ep.x, ep.y, ep.z))
                            solver.extreme_points[0].remove(ep)
                            solver.update_extreme_points(item, ep, 0)
                        else:
                            # Instead of removing, add to a new bin to avoid loss
                            new_bin_idx = solver.add_new_bin()
                            ep = solver.extreme_points[new_bin_idx][0]
                            solver.bins[new_bin_idx].append(PlacedItem(item, ep.x, ep.y, ep.z))
                            solver.extreme_points[new_bin_idx].remove(ep)
                            solver.update_extreme_points(item, ep, new_bin_idx)
                    # Update bin and extreme points
                    self.bins[bin_idx] = solver.bins[0]
                    self.extreme_points[bin_idx] = solver.extreme_points[0]
                    # Append any new bins created
                    if len(solver.bins) > 1:
                        self.bins.extend(solver.bins[1:])
                        self.extreme_points.extend(solver.extreme_points[1:])
            if not self.bins[bin_idx]:  # Remove empty bins
                self.bins.pop(bin_idx)
                self.extreme_points.pop(bin_idx)
