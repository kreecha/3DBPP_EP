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

import numpy as np
import random
import copy
import math
from typing import List, Tuple, Set
from dataclasses import dataclass


from src.common import Item, Bin, PlacedItem, ExtremePoint, SortingRule, MeritFunction
from src.ExtremePointHeuristic import ExtremePointBinPacking3D

from src.classes.ValidateOperatorManager import ValidatedDestroyOperator, ValidatedRepairOperator
from src.classes.solution import Solution

class DestroyOperator(ValidatedDestroyOperator):
    """Base class for destroy operators"""

    def __init__(self, name: str):
        super().__init__(name)

    def _destroy_implementation(self, solution, rnd_state: np.random.RandomState) -> List[Item]:
        """
        Default implementation that calls the old destroy() method if it exists,
        otherwise raises NotImplementedError for subclasses to implement.
        """
        # Check if subclass has implemented the old destroy() method
        if hasattr(self, '_old_destroy') and callable(getattr(self, '_old_destroy')):
            return self._old_destroy(solution, rnd_state)
        else:
            # Subclass should implement _destroy_implementation directly
            raise NotImplementedError(f"{self.__class__.__name__} must implement _destroy_implementation()")

    def destroy(self, solution, rnd_state: np.random.RandomState) -> List[Item]:
        """
        This method is now handled by the parent ValidatedDestroyOperator
        If a subclass overrides this method, it will bypass validation.
        Subclasses should implement _destroy_implementation() instead.
        """
        return super().destroy(solution, rnd_state)


class RandomDestroy(DestroyOperator):
    """Randomly remove items from the solution"""
    def __init__(self, min_remove: int, max_remove: int):
        super().__init__("Random Destroy")
        self.min_remove = min_remove
        self.max_remove = max_remove

    def _destroy_implementation(self, solution: Solution, rnd_state: np.random.RandomState) -> List[Item]:
        all_items = solution.get_all_items()
        num_to_remove = rnd_state.randint(self.min_remove, self.max_remove + 1)
        num_to_remove = min(num_to_remove, len(all_items))
        items_to_remove = rnd_state.choice(all_items, size=num_to_remove, replace=False).tolist()
        solution.remove_items(items_to_remove)
        return items_to_remove


class WorstBinDestroy(DestroyOperator):
    """Remove items from the bin with lowest utilization"""
    def __init__(self, num_bins_to_target: int):
        super().__init__("Worst Bin Destroy")
        self.num_bins_to_target = num_bins_to_target

    def _destroy_implementation(self, solution: Solution, rnd_state: np.random.RandomState) -> List[Item]:
        utilizations = []
        for bin_items in solution.bins:
            used_volume = sum(item.item.volume for item in bin_items)
            utilizations.append(used_volume / solution.bin_template.volume)

        if not utilizations:
            return []

        worst_bins = np.argsort(utilizations)[:self.num_bins_to_target]
        items_to_remove = []
        for bin_idx in worst_bins:
            items_to_remove.extend([item.item for item in solution.bins[bin_idx]])

        solution.remove_items(items_to_remove)
        return items_to_remove


class LargeItemDestroy(DestroyOperator):
    """Remove largest items based on volume"""
    def __init__(self, percentage: float):
        super().__init__("Large Item Destroy")
        self.percentage = percentage

    def _destroy_implementation(self, solution: Solution, rnd_state: np.random.RandomState) -> List[Item]:
        all_items = solution.get_all_items()
        num_to_remove = int(len(all_items) * self.percentage)
        if num_to_remove == 0:
            return []

        sorted_items = sorted(all_items, key=lambda x: -x.volume)
        items_to_remove = sorted_items[:num_to_remove]
        solution.remove_items(items_to_remove)
        return items_to_remove


class ClusterDestroy(DestroyOperator):
    def __init__(self, cluster_size: int):
        super().__init__("Cluster Destroy")
        self.cluster_size = cluster_size

    def _destroy_implementation(self, solution: Solution, rnd_state: np.random.RandomState) -> List[Item]:
        all_items = []
        for bin_idx, bin_items in enumerate(solution.bins):
            for item in bin_items:
                all_items.append((item, bin_idx))

        if not all_items:
            return []

        # Select a random anchor item using indices
        anchor_idx = rnd_state.choice(len(all_items))  # Choose an index
        anchor_item, anchor_bin = all_items[anchor_idx]

        # Find items in the same bin that are spatially close
        distances = []
        for item, bin_idx in all_items:
            if bin_idx == anchor_bin and item != anchor_item:
                distance = self._calculate_distance(anchor_item, item)
                distances.append((distance, item))

        if not distances:
            return [anchor_item.item]

        distances.sort()
        items_to_remove = [anchor_item.item]
        num_to_remove = min(self.cluster_size, len(distances))
        for i in range(num_to_remove):
            items_to_remove.append(distances[i][1].item)

        solution.remove_items(items_to_remove)
        return items_to_remove

    def _calculate_distance(self, item1: PlacedItem, item2: PlacedItem) -> float:
        """Calculate distance between item centers"""
        c1 = (item1.x + item1.item.width / 2, item1.y + item1.item.depth / 2, item1.z + item1.item.height / 2)
        c2 = (item2.x + item2.item.width / 2, item2.y + item2.item.depth / 2, item2.z + item2.item.height / 2)
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(c1, c2)))

# =============================================================================
# TOP 10 DESTROY OPERATORS (Based on Voigt 2025 Rankings)
# =============================================================================

class WorstCostDestroy(DestroyOperator):
    """R16: Worst cost customers - Top ranked destroy operator"""

    def __init__(self, min_remove: int = 3, max_remove: int = 8):
        super().__init__("Worst Cost Destroy")
        self.min_remove = min_remove
        self.max_remove = max_remove

    def _destroy_implementation(self, solution, rnd_state: np.random.RandomState) -> List[Item]:
        """Remove items that contribute most to the objective (worst utilization impact)"""
        if not solution.bins:
            return []

        item_costs = []

        # Calculate cost contribution for each item (how much removing it improves utilization)
        for bin_idx, bin_items in enumerate(solution.bins):
            if not bin_items:
                continue

            bin_volume = solution.bin_template.volume
            current_utilization = sum(item.item.volume for item in bin_items) / bin_volume

            for placed_item in bin_items:
                # Cost = negative impact on utilization when removed
                without_item_util = (sum(pi.item.volume for pi in bin_items if pi != placed_item)) / bin_volume
                cost = current_utilization - without_item_util  # Higher means worse to remove
                item_costs.append((cost, placed_item.item, bin_idx))

        if not item_costs:
            return []

        # Sort by cost (highest first - worst items to remove)
        item_costs.sort(key=lambda x: -x[0])

        num_to_remove = min(rnd_state.randint(self.min_remove, self.max_remove + 1), len(item_costs))
        items_to_remove = [item for _, item, _ in item_costs[:num_to_remove]]

        solution.remove_items(items_to_remove)
        return items_to_remove


class APosterioriScoreDestroy(DestroyOperator):
    """R13: A posteriori score related customers - 2nd ranked destroy operator"""

    def __init__(self, min_remove: int = 3, max_remove: int = 8):
        super().__init__("A Posteriori Score Destroy")
        self.min_remove = min_remove
        self.max_remove = max_remove

    def _destroy_implementation(self, solution, rnd_state: np.random.RandomState) -> List[Item]:
        """Remove items based on their position quality score after packing"""
        if not solution.bins:
            return []

        item_scores = []

        for bin_idx, bin_items in enumerate(solution.bins):
            if not bin_items:
                continue

            for placed_item in bin_items:
                # Score based on position efficiency (corner preference + space utilization)
                corner_score = self._calculate_corner_score(placed_item, solution.bin_template)
                space_score = self._calculate_space_utilization_score(placed_item, bin_items, solution.bin_template)

                total_score = corner_score + space_score
                item_scores.append((total_score, placed_item.item))

        if not item_scores:
            return []

        # Remove items with lowest scores (poorly positioned)
        item_scores.sort(key=lambda x: x[0])

        num_to_remove = min(rnd_state.randint(self.min_remove, self.max_remove + 1), len(item_scores))
        items_to_remove = [item for _, item in item_scores[:num_to_remove]]

        solution.remove_items(items_to_remove)
        return items_to_remove

    def _calculate_corner_score(self, placed_item: PlacedItem, bin_template: Bin) -> float:
        """Higher score for items closer to corners"""
        x_score = min(placed_item.x, bin_template.width - placed_item.x - placed_item.item.width)
        y_score = min(placed_item.y, bin_template.depth - placed_item.y - placed_item.item.depth)
        z_score = min(placed_item.z, bin_template.height - placed_item.z - placed_item.item.height)
        return -(x_score + y_score + z_score)  # Negative because we want corner preference

    def _calculate_space_utilization_score(self, placed_item: PlacedItem, bin_items: List[PlacedItem],
                                           bin_template: Bin) -> float:
        """Score based on how well item utilizes space"""
        item_volume = placed_item.item.volume
        bin_volume = bin_template.volume
        return item_volume / bin_volume


class RandomSequenceDestroy(DestroyOperator):
    """R9: All customers from randomly selected sequence - Top sequence-based operator"""

    def __init__(self, min_length: int = 2, max_length: int = 6):
        super().__init__("Random Sequence Destroy")
        self.min_length = min_length
        self.max_length = max_length

    def _destroy_implementation(self, solution, rnd_state: np.random.RandomState) -> List[Item]:
        """Remove a sequence of adjacent items from a random bin"""
        if not solution.bins:
            return []

        non_empty_bins = [i for i, bin_items in enumerate(solution.bins) if bin_items]
        if not non_empty_bins:
            return []

        # Select random bin
        bin_idx = rnd_state.choice(non_empty_bins)
        bin_items = solution.bins[bin_idx]

        if len(bin_items) < 2:
            return []

        # Select sequence length and starting position
        seq_length = min(rnd_state.randint(self.min_length, self.max_length + 1), len(bin_items))
        start_pos = rnd_state.randint(0, len(bin_items) - seq_length + 1)

        # Remove sequence
        items_to_remove = [bin_items[i].item for i in range(start_pos, start_pos + seq_length)]

        solution.remove_items(items_to_remove)
        return items_to_remove


class RandomRouteDestroy(DestroyOperator):
    """R18/R19: All customers from random route - High performing route-based operator"""

    def __init__(self, max_bins: int = 2):
        super().__init__("Random Route Destroy")
        self.max_bins = max_bins

    def _destroy_implementation(self, solution, rnd_state: np.random.RandomState) -> List[Item]:
        """Remove all items from one or more randomly selected bins"""
        if not solution.bins:
            return []

        non_empty_bins = [i for i, bin_items in enumerate(solution.bins) if bin_items]
        if not non_empty_bins:
            return []

        # Select random number of bins to empty
        num_bins = min(rnd_state.randint(1, self.max_bins + 1), len(non_empty_bins))
        selected_bins = rnd_state.choice(non_empty_bins, size=num_bins, replace=False)

        items_to_remove = []
        for bin_idx in selected_bins:
            items_to_remove.extend([item.item for item in solution.bins[bin_idx]])

        solution.remove_items(items_to_remove)
        return items_to_remove


class APrioriRelatedDestroy(DestroyOperator):
    """R6: A priori distance related customers to seed customer"""

    def __init__(self, min_remove: int = 3, max_remove: int = 8):
        super().__init__("A Priori Related Destroy")
        self.min_remove = min_remove
        self.max_remove = max_remove

    def _destroy_implementation(self, solution, rnd_state: np.random.RandomState) -> List[Item]:
        """Remove items that are similar in size/characteristics to a seed item"""
        all_items = solution.get_all_items()
        if len(all_items) < 2:
            return []

        # Select seed item randomly
        seed_item = rnd_state.choice(all_items)

        # Calculate similarity scores
        item_similarities = []
        for item in all_items:
            if item != seed_item:
                similarity = self._calculate_similarity(seed_item, item)
                item_similarities.append((similarity, item))

        if not item_similarities:
            return []

        # Sort by similarity (highest first)
        item_similarities.sort(key=lambda x: -x[0])

        num_to_remove = min(rnd_state.randint(self.min_remove, self.max_remove + 1), len(item_similarities))
        items_to_remove = [seed_item] + [item for _, item in item_similarities[:num_to_remove - 1]]

        solution.remove_items(items_to_remove)
        return items_to_remove

    def _calculate_similarity(self, item1: Item, item2: Item) -> float:
        """Calculate similarity based on dimensions and volume"""
        vol_diff = abs(item1.volume - item2.volume) / max(item1.volume, item2.volume)
        dim_diff = (abs(item1.width - item2.width) + abs(item1.depth - item2.depth) +
                    abs(item1.height - item2.height)) / (item1.width + item1.depth + item1.height)

        return 1.0 - (0.5 * vol_diff + 0.5 * dim_diff)


class ClusterBasedDestroy(DestroyOperator):
    """R10: Customers from Kruskal clusters - Spatial clustering approach"""

    def __init__(self, min_remove: int = 3, max_remove: int = 8):
        super().__init__("Cluster Based Destroy")
        self.min_remove = min_remove
        self.max_remove = max_remove

    def _destroy_implementation(self, solution, rnd_state: np.random.RandomState) -> List[Item]:
        """Remove spatially clustered items from the same bin"""
        if not solution.bins:
            return []

        # Find bin with most items for clustering
        best_bin_idx = max(range(len(solution.bins)),
                           key=lambda i: len(solution.bins[i]) if solution.bins[i] else 0)

        bin_items = solution.bins[best_bin_idx]
        if len(bin_items) < 2:
            return self._fallback_random_destroy(solution, rnd_state)

        # Select seed item
        seed_item = rnd_state.choice(bin_items)

        # Find spatially close items
        clustered_items = [seed_item]
        remaining_items = [item for item in bin_items if item != seed_item]

        for item in remaining_items:
            distance = self._spatial_distance(seed_item, item)
            if distance < self._get_cluster_threshold(seed_item, solution.bin_template):
                clustered_items.append(item)

            if len(clustered_items) >= self.max_remove:
                break

        num_to_remove = min(len(clustered_items),
                            rnd_state.randint(self.min_remove, self.max_remove + 1))
        items_to_remove = [item.item for item in clustered_items[:num_to_remove]]

        solution.remove_items(items_to_remove)
        return items_to_remove

    def _spatial_distance(self, item1: PlacedItem, item2: PlacedItem) -> float:
        """Calculate spatial distance between items"""
        center1 = (item1.x + item1.item.width / 2, item1.y + item1.item.depth / 2, item1.z + item1.item.height / 2)
        center2 = (item2.x + item2.item.width / 2, item2.y + item2.item.depth / 2, item2.z + item2.item.height / 2)

        return math.sqrt(sum((a - b) ** 2 for a, b in zip(center1, center2)))

    def _get_cluster_threshold(self, item: PlacedItem, bin_template: Bin) -> float:
        """Dynamic clustering threshold based on item and bin size"""
        item_diagonal = math.sqrt(item.item.width ** 2 + item.item.depth ** 2 + item.item.height ** 2)
        return item_diagonal * 1.5

    def _fallback_random_destroy(self, solution, rnd_state):
        """Fallback to random destroy if clustering fails"""
        all_items = solution.get_all_items()
        if not all_items:
            return []
        num_to_remove = min(rnd_state.randint(self.min_remove, self.max_remove + 1), len(all_items))
        items_to_remove = rnd_state.choice(all_items, size=num_to_remove, replace=False).tolist()
        solution.remove_items(items_to_remove)
        return items_to_remove


class WorstDistanceDestroy(DestroyOperator):
    """R14: Worst distance cost customers"""

    def __init__(self, min_remove: int = 3, max_remove: int = 8):
        super().__init__("Worst Distance Destroy")
        self.min_remove = min_remove
        self.max_remove = max_remove

    def _destroy_implementation(self, solution, rnd_state: np.random.RandomState) -> List[Item]:
        """Remove items that are poorly positioned distance-wise"""
        all_placed_items = []
        for bin_items in solution.bins:
            all_placed_items.extend(bin_items)

        if len(all_placed_items) < 2:
            return []

        item_distance_costs = []

        for placed_item in all_placed_items:
            # Calculate average distance to other items (poor positioning indicator)
            distances = []
            for other_item in all_placed_items:
                if other_item != placed_item:
                    dist = self._calculate_distance(placed_item, other_item)
                    distances.append(dist)

            avg_distance = np.mean(distances) if distances else 0
            item_distance_costs.append((avg_distance, placed_item.item))

        # Sort by distance cost (highest first - worst positioned)
        item_distance_costs.sort(key=lambda x: -x[0])

        num_to_remove = min(rnd_state.randint(self.min_remove, self.max_remove + 1), len(item_distance_costs))
        items_to_remove = [item for _, item in item_distance_costs[:num_to_remove]]

        solution.remove_items(items_to_remove)
        return items_to_remove

    def _calculate_distance(self, item1: PlacedItem, item2: PlacedItem) -> float:
        """Calculate distance between item centers"""
        c1 = (item1.x + item1.item.width / 2, item1.y + item1.item.depth / 2, item1.z + item1.item.height / 2)
        c2 = (item2.x + item2.item.width / 2, item2.y + item2.item.depth / 2, item2.z + item2.item.height / 2)
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(c1, c2)))


class RandomDestroy(DestroyOperator):
    """R1: Random customers - Classic diversification operator"""

    def __init__(self, min_remove: int = 3, max_remove: int = 8):
        super().__init__("Random Destroy")
        self.min_remove = min_remove
        self.max_remove = max_remove

    def _destroy_implementation(self, solution, rnd_state: np.random.RandomState) -> List[Item]:
        """Remove random items for diversification"""
        all_items = solution.get_all_items()
        if not all_items:
            return []

        num_to_remove = min(rnd_state.randint(self.min_remove, self.max_remove + 1), len(all_items))
        items_to_remove = rnd_state.choice(all_items, size=num_to_remove, replace=False).tolist()

        solution.remove_items(items_to_remove)
        return items_to_remove


class StartTimeRelatedDestroy(DestroyOperator):
    """R11: Start time related customers to seed customer - Temporal-based"""

    def __init__(self, min_remove: int = 3, max_remove: int = 8):
        super().__init__("Start Time Related Destroy")
        self.min_remove = min_remove
        self.max_remove = max_remove

    def _destroy_implementation(self, solution, rnd_state: np.random.RandomState) -> List[Item]:
        """Remove items based on their packing order/timing"""
        all_placed_items = []
        for bin_idx, bin_items in enumerate(solution.bins):
            for pos, placed_item in enumerate(bin_items):
                all_placed_items.append((bin_idx, pos, placed_item))

        if len(all_placed_items) < 2:
            return []

        # Select seed item
        seed_bin, seed_pos, seed_item = rnd_state.choice(all_placed_items)

        # Find items with similar "timing" (position in packing sequence)
        related_items = [seed_item]

        for bin_idx, pos, placed_item in all_placed_items:
            if placed_item != seed_item:
                # Items are related if they're in similar positions in their bins
                position_similarity = abs(pos - seed_pos)
                if position_similarity <= 2:  # Within 2 positions
                    related_items.append(placed_item)

        num_to_remove = min(rnd_state.randint(self.min_remove, self.max_remove + 1), len(related_items))
        items_to_remove = [item.item for item in related_items[:num_to_remove]]

        solution.remove_items(items_to_remove)
        return items_to_remove


class DemandRelatedDestroy(DestroyOperator):
    """R7: Demand related customers to seed customer - Size-based clustering"""

    def __init__(self, min_remove: int = 3, max_remove: int = 8):
        super().__init__("Demand Related Destroy")
        self.min_remove = min_remove
        self.max_remove = max_remove

    def _destroy_implementation(self, solution, rnd_state: np.random.RandomState) -> List[Item]:
        """Remove items with similar 'demand' (volume) to seed item"""
        all_items = solution.get_all_items()
        if len(all_items) < 2:
            return []

        # Select seed item
        seed_item = rnd_state.choice(all_items)

        # Find items with similar volume (demand)
        volume_threshold = seed_item.volume * 0.3  # 30% tolerance
        related_items = [seed_item]

        for item in all_items:
            if item != seed_item:
                volume_diff = abs(item.volume - seed_item.volume)
                if volume_diff <= volume_threshold:
                    related_items.append(item)

        num_to_remove = min(rnd_state.randint(self.min_remove, self.max_remove + 1), len(related_items))
        items_to_remove = related_items[:num_to_remove]

        solution.remove_items(items_to_remove)
        return items_to_remove


class FullBinDestroy(ValidatedDestroyOperator):
    def __init__(self, num_bins_to_target: int = 2):
        super().__init__("Full Bin Destroy")
        self.num_bins_to_target = num_bins_to_target

    def _destroy_implementation(self, solution: Solution, rnd_state: np.random.RandomState) -> List[Item]:
        if not solution.bins:
            return []
        num_bins = min(self.num_bins_to_target, len(solution.bins))
        bin_indices = rnd_state.choice(len(solution.bins), size=num_bins, replace=False)
        items_to_remove = []
        for idx in sorted(bin_indices, reverse=True):
            items_to_remove.extend([pi.item for pi in solution.bins[idx]])
            solution.bins.pop(idx)
            solution.extreme_points.pop(idx)
        return items_to_remove


class AdaptiveRandomDestroy(ValidatedDestroyOperator):
    """Random destroy with adaptive removal percentage"""

    def __init__(self, min_rate: float = 0.10, max_rate: float = 0.30):
        super().__init__("Adaptive Random Destroy")
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.recent_improvements = []

    def _destroy_implementation(self, solution: Solution, rnd_state: np.random.RandomState) -> List[Item]:
        all_items = solution.get_all_items()
        if not all_items:
            return []

        # Adaptive removal rate based on recent performance
        if len(self.recent_improvements) >= 5:
            avg_improvement = np.mean(self.recent_improvements[-5:])
            removal_rate = self.max_rate if avg_improvement > 0 else self.min_rate
        else:
            removal_rate = (self.min_rate + self.max_rate) / 2

        num_to_remove = max(1, int(len(all_items) * removal_rate))
        items_to_remove = rnd_state.choice(all_items, size=num_to_remove, replace=False).tolist()

        solution.remove_items(items_to_remove)
        return items_to_remove


class UtilizationBasedDestroy(ValidatedDestroyOperator):
    """Remove items from poorly utilized bins"""

    def __init__(self, target_threshold: float = 0.6):
        super().__init__("Utilization Based Destroy")
        self.target_threshold = target_threshold

    def _destroy_implementation(self, solution: Solution, rnd_state: np.random.RandomState) -> List[Item]:
        if not solution.bins:
            return []

        # Find bins with poor utilization
        utilizations = solution.get_bin_utilizations()
        poor_bins = [i for i, util in enumerate(utilizations)
                     if util < self.target_threshold and solution.bins[i]]

        if not poor_bins:
            # Fall back to worst bins
            poor_bins = [np.argmin(utilizations)]

        items_to_remove = []
        for bin_idx in poor_bins:
            # Remove some items from this bin (not all)
            bin_items = [pi.item for pi in solution.bins[bin_idx]]
            num_to_remove = max(1, len(bin_items) // 2)
            selected_items = rnd_state.choice(bin_items, size=num_to_remove, replace=False)
            items_to_remove.extend(selected_items)

        solution.remove_items(items_to_remove)
        return items_to_remove