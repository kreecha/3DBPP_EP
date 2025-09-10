# -*- coding: utf-8 -*-
"""
Created on Tue Sep 9 08:26:17 2025

common.py - Shared classes and methods for 3D Bin Packing implementations

This module contains data classes, enumerations, and the core BinPacking3D class
used by both ExtremePointHeuristic and ALNS implementations.

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
from typing import List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


@dataclass
class Item:
    """Represents a 3D item with dimensions and optional identifier"""
    width: float  # w - x dimension
    depth: float  # d - y dimension
    height: float  # h - z dimension
    id: Optional[int] = None

    @property
    def volume(self) -> float:
        return self.width * self.depth * self.height

    @property
    def base_area(self) -> float:
        return self.width * self.depth

    def __repr__(self):
        return f"Item(id={self.id}, w={self.width}, d={self.depth}, h={self.height})"


@dataclass
class Bin:
    """Represents a 3D bin container"""
    width: float  # W - x dimension
    depth: float  # D - y dimension
    height: float  # H - z dimension

    @property
    def volume(self) -> float:
        return self.width * self.depth * self.height


@dataclass
class PlacedItem:
    """Represents an item placed in a specific position within a bin"""
    item: Item
    x: float  # left position
    y: float  # back position
    z: float  # bottom position

    @property
    def right(self) -> float:
        return self.x + self.item.width

    @property
    def front(self) -> float:
        return self.y + self.item.depth

    @property
    def top(self) -> float:
        return self.z + self.item.height


@dataclass
class ExtremePoint:
    """Represents an extreme point where items can be placed"""
    x: float
    y: float
    z: float
    residual_space_x: float = 0.0
    residual_space_y: float = 0.0
    residual_space_z: float = 0.0

    def __repr__(self):
        return f"EP({self.x}, {self.y}, {self.z})"


class SortingRule(Enum):
    """Available sorting rules for items"""
    VOLUME_HEIGHT = "volume_height"
    HEIGHT_VOLUME = "height_volume"
    AREA_HEIGHT = "area_height"
    HEIGHT_AREA = "height_area"
    CLUSTERED_AREA_HEIGHT = "clustered_area_height"
    CLUSTERED_HEIGHT_AREA = "clustered_height_area"


class MeritFunction(Enum):
    """Merit functions for EP-BFD heuristic"""
    FREE_VOLUME = "free_volume"
    MINIMIZE_PACKING = "minimize_packing"
    LEVEL_PACKING = "level_packing"
    RESIDUAL_SPACE = "residual_space"


class BinPacking3D:
    """3D Bin Packing solver using Extreme Point-based heuristics"""

    def __init__(self, bin_template: Bin):
        self.bin_template = bin_template
        self.bins: List[List[PlacedItem]] = []
        self.extreme_points: List[List[ExtremePoint]] = []

    def clear_solution(self):
        """Clear current solution"""
        self.bins.clear()
        self.extreme_points.clear()

    def add_new_bin(self) -> int:
        """Add a new empty bin and return its index"""
        self.bins.append([])
        initial_ep = ExtremePoint(0, 0, 0,
                                 self.bin_template.width,
                                 self.bin_template.depth,
                                 self.bin_template.height)
        self.extreme_points.append([initial_ep])
        return len(self.bins) - 1

    def can_place_item(self, item: Item, ep: ExtremePoint, bin_idx: int) -> bool:
        """Check if item can be placed at extreme point without overlapping"""
        if (ep.x + item.width > self.bin_template.width + 1e-9 or
                ep.y + item.depth > self.bin_template.depth + 1e-9 or
                ep.z + item.height > self.bin_template.height + 1e-9):
            return False

        for placed_item in self.bins[bin_idx]:
            if self._items_overlap(item, ep, placed_item):
                return False

        return True

    def _items_overlap(self, item: Item, ep: ExtremePoint, placed_item: PlacedItem) -> bool:
        """Check if two items overlap in 3D space"""
        return not (ep.x + item.width <= placed_item.x + 1e-9 or
                    placed_item.right <= ep.x + 1e-9 or
                    ep.y + item.depth <= placed_item.y + 1e-9 or
                    placed_item.front <= ep.y + 1e-9 or
                    ep.z + item.height <= placed_item.z + 1e-9 or
                    placed_item.top <= ep.z + 1e-9)

    def update_extreme_points(self, item: Item, ep: ExtremePoint, bin_idx: int):
        """Update extreme points after placing an item (Algorithm 1 from paper)"""
        eps = self.extreme_points[bin_idx]
        new_eps = []

        projections = [
            (ep.x + item.width, ep.y, ep.z),  # YZ direction
            (ep.x, ep.y + item.depth, ep.z),  # XZ direction
            (ep.x, ep.y, ep.z + item.height)  # XY direction
        ]

        for proj_x, proj_y, proj_z in projections:
            if proj_x < self.bin_template.width:
                new_eps.append(self._project_point(proj_x, ep.y, ep.z, bin_idx))
            if proj_y < self.bin_template.depth:
                new_eps.append(self._project_point(ep.x, proj_y, ep.z, bin_idx))
            if proj_z < self.bin_template.height:
                new_eps.append(self._project_point(ep.x, ep.y, proj_z, bin_idx))

        if ep.x + item.width < self.bin_template.width and ep.y + item.depth < self.bin_template.depth:
            new_eps.append(self._project_point(ep.x + item.width, ep.y + item.depth, ep.z, bin_idx))
        if ep.x + item.width < self.bin_template.width and ep.z + item.height < self.bin_template.height:
            new_eps.append(self._project_point(ep.x + item.width, ep.y, ep.z + item.height, bin_idx))
        if ep.y + item.depth < self.bin_template.depth and ep.z + item.height < self.bin_template.height:
            new_eps.append(self._project_point(ep.x, ep.y + item.depth, ep.z + item.height, bin_idx))

        for new_ep in new_eps:
            if new_ep and self._is_valid_extreme_point(new_ep, bin_idx):
                eps.append(new_ep)

        self._clean_extreme_points(bin_idx)
        self._update_residual_spaces(bin_idx)

    def _project_point(self, x: float, y: float, z: float, bin_idx: int) -> Optional[ExtremePoint]:
        """Project a point to valid position considering existing items"""
        if (x <= self.bin_template.width + 1e-9 and
                y <= self.bin_template.depth + 1e-9 and
                z <= self.bin_template.height + 1e-9):
            return ExtremePoint(x, y, z)
        return None

    def _is_valid_extreme_point(self, ep: ExtremePoint, bin_idx: int) -> bool:
        """Check if extreme point is valid (not inside any item)"""
        for placed_item in self.bins[bin_idx]:
            if (placed_item.x < ep.x < placed_item.right and
                    placed_item.y < ep.y < placed_item.front and
                    placed_item.z < ep.z < placed_item.top):
                return False
        return True

    def _clean_extreme_points(self, bin_idx: int):
        """Remove duplicate and dominated extreme points"""
        eps = self.extreme_points[bin_idx]

        unique_eps = []
        for ep in eps:
            is_duplicate = False
            for existing_ep in unique_eps:
                if (abs(ep.x - existing_ep.x) < 1e-9 and
                        abs(ep.y - existing_ep.y) < 1e-9 and
                        abs(ep.z - existing_ep.z) < 1e-9):
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_eps.append(ep)

        unique_eps.sort(key=lambda ep: (ep.z, ep.y, ep.x))

        self.extreme_points[bin_idx] = unique_eps

    def _update_residual_spaces(self, bin_idx: int):
        """Update residual spaces for all extreme points (Algorithm 2 from paper)"""
        for ep in self.extreme_points[bin_idx]:
            ep.residual_space_x = self.bin_template.width - ep.x
            ep.residual_space_y = self.bin_template.depth - ep.y
            ep.residual_space_z = self.bin_template.height - ep.z

            for placed_item in self.bins[bin_idx]:
                if ep.z <= placed_item.z < ep.z + ep.residual_space_z:
                    if ep.x <= placed_item.x and placed_item.x < ep.x + ep.residual_space_x:
                        ep.residual_space_x = min(ep.residual_space_x, placed_item.x - ep.x)
                    if ep.y <= placed_item.y and placed_item.y < ep.y + ep.residual_space_y:
                        ep.residual_space_y = min(ep.residual_space_y, placed_item.y - ep.y)

                if ep.z <= placed_item.z:
                    ep.residual_space_z = min(ep.residual_space_z, placed_item.z - ep.z)

    def sort_items(self, items: List[Item], rule: SortingRule, cluster_param: int = 10) -> List[Item]:
        """Sort items according to specified rule"""
        sorted_items = items.copy()

        if rule == SortingRule.VOLUME_HEIGHT:
            sorted_items.sort(key=lambda item: (-item.volume, -item.height))
        elif rule == SortingRule.HEIGHT_VOLUME:
            sorted_items.sort(key=lambda item: (-item.height, -item.volume))
        elif rule == SortingRule.AREA_HEIGHT:
            sorted_items.sort(key=lambda item: (-item.base_area, -item.height))
        elif rule == SortingRule.HEIGHT_AREA:
            sorted_items.sort(key=lambda item: (-item.height, -item.base_area))
        elif rule == SortingRule.CLUSTERED_AREA_HEIGHT:
            sorted_items = self._clustered_sort(items, 'area', cluster_param)
        elif rule == SortingRule.CLUSTERED_HEIGHT_AREA:
            sorted_items = self._clustered_sort(items, 'height', cluster_param)

        return sorted_items

    def _clustered_sort(self, items: List[Item], primary: str, cluster_param: int) -> List[Item]:
        """Implement clustered sorting as described in paper"""
        if primary == 'area':
            bin_area = self.bin_template.width * self.bin_template.depth
            clusters = {}

            for item in items:
                cluster_id = int(item.base_area * cluster_param / bin_area)
                if cluster_id not in clusters:
                    clusters[cluster_id] = []
                clusters[cluster_id].append(item)

            sorted_items = []
            for cluster_id in sorted(clusters.keys(), reverse=True):
                cluster_items = sorted(clusters[cluster_id], key=lambda x: -x.height)
                sorted_items.extend(cluster_items)

        else:  # height clustering
            bin_height = self.bin_template.height
            clusters = {}

            for item in items:
                cluster_id = int(item.height * cluster_param / bin_height)
                if cluster_id not in clusters:
                    clusters[cluster_id] = []
                clusters[cluster_id].append(item)

            sorted_items = []
            for cluster_id in sorted(clusters.keys(), reverse=True):
                cluster_items = sorted(clusters[cluster_id], key=lambda x: -x.base_area)
                sorted_items.extend(cluster_items)

        return sorted_items

    def ep_ffd(self, items: List[Item], sorting_rule: SortingRule = SortingRule.VOLUME_HEIGHT) -> int:
        """Extreme Point First Fit Decreasing heuristic"""
        self.clear_solution()
        sorted_items = self.sort_items(items, sorting_rule)

        for item in sorted_items:
            placed = False

            for bin_idx in range(len(self.bins)):
                best_ep = self._find_best_ep_ffd(item, bin_idx)
                if best_ep is not None:
                    placed_item = PlacedItem(item, best_ep.x, best_ep.y, best_ep.z)
                    self.bins[bin_idx].append(placed_item)

                    self.extreme_points[bin_idx].remove(best_ep)

                    self.update_extreme_points(item, best_ep, bin_idx)

                    placed = True
                    break

            if not placed:
                bin_idx = self.add_new_bin()
                ep = self.extreme_points[bin_idx][0]

                placed_item = PlacedItem(item, ep.x, ep.y, ep.z)
                self.bins[bin_idx].append(placed_item)

                self.extreme_points[bin_idx].remove(ep)

                self.update_extreme_points(item, ep, bin_idx)

        return len(self.bins)

    def _find_best_ep_ffd(self, item: Item, bin_idx: int) -> Optional[ExtremePoint]:
        """Find the first valid extreme point for an item in a bin"""
        for ep in self.extreme_points[bin_idx]:
            if self.can_place_item(item, ep, bin_idx):
                return ep
        return None

    def ep_bfd(self, items: List[Item], sorting_rule: SortingRule = SortingRule.VOLUME_HEIGHT,
               merit_function: MeritFunction = MeritFunction.RESIDUAL_SPACE) -> int:
        """Extreme Point Best Fit Decreasing heuristic"""
        self.clear_solution()
        sorted_items = self.sort_items(items, sorting_rule)

        for item in sorted_items:
            best_bin = None
            best_ep = None
            best_merit = float('inf')

            for bin_idx in range(len(self.bins)):
                ep, merit = self._find_best_ep_bfd(item, bin_idx, merit_function)
                if ep is not None and merit < best_merit:
                    best_merit = merit
                    best_bin = bin_idx
                    best_ep = ep

            if best_ep is not None:
                placed_item = PlacedItem(item, best_ep.x, best_ep.y, best_ep.z)
                self.bins[best_bin].append(placed_item)

                self.extreme_points[best_bin].remove(best_ep)

                self.update_extreme_points(item, best_ep, best_bin)
            else:
                bin_idx = self.add_new_bin()
                ep = self.extreme_points[bin_idx][0]

                placed_item = PlacedItem(item, ep.x, ep.y, ep.z)
                self.bins[bin_idx].append(placed_item)

                self.extreme_points[bin_idx].remove(ep)

                self.update_extreme_points(item, ep, bin_idx)

        return len(self.bins)

    def _find_best_ep_bfd(self, item: Item, bin_idx: int,
                          merit_function: MeritFunction) -> Tuple[Optional[ExtremePoint], float]:
        """Find the best extreme point for an item based on merit function"""
        best_ep = None
        best_merit = float('inf')

        for ep in self.extreme_points[bin_idx]:
            if self.can_place_item(item, ep, bin_idx):
                merit = self._calculate_merit(item, ep, bin_idx, merit_function)
                if merit < best_merit:
                    best_merit = merit
                    best_ep = ep

        return best_ep, best_merit

    def _calculate_merit(self, item: Item, ep: ExtremePoint, bin_idx: int,
                         merit_function: MeritFunction) -> float:
        """Calculate merit score for placing an item at an extreme point"""
        if merit_function == MeritFunction.FREE_VOLUME:
            used_volume = sum(placed_item.item.volume for placed_item in self.bins[bin_idx])
            return (self.bin_template.volume - used_volume - item.volume) / self.bin_template.volume
        elif merit_function == MeritFunction.MINIMIZE_PACKING:
            return ep.z + item.height
        elif merit_function == MeritFunction.LEVEL_PACKING:
            return abs(ep.z - item.height)
        elif merit_function == MeritFunction.RESIDUAL_SPACE:
            return -(ep.residual_space_x * ep.residual_space_y * ep.residual_space_z)
        return float('inf')

class BenchmarkGenerator:
    """Generate 3D bin packing instances following Martello et al. format"""

    @staticmethod
    def generate_martello_instance(class_type: int, n_items: int, bin_size: int = 100) -> Tuple[List[Item], Bin]:
        """Generate instance following Martello et al. (2000) classification"""
        np.random.seed()  # Use current time

        items = []
        bin_template = Bin(bin_size, bin_size, bin_size)

        if class_type == 1:
            # Type 1 majority: wj ∈ [1, W/2], hj ∈ [2H/3, H], dj ∈ [2D/3, D]
            for i in range(n_items):
                if np.random.random() < 0.6:  # Type 1 - 60%
                    w = np.random.uniform(1, bin_size / 2)
                    h = np.random.uniform(2 * bin_size / 3, bin_size)
                    d = np.random.uniform(2 * bin_size / 3, bin_size)
                else:  # Other types - 10% each
                    type_choice = np.random.choice([2, 3, 4, 5])
                    w, h, d = BenchmarkGenerator._generate_item_by_type(type_choice, bin_size)

                items.append(Item(w, d, h, i))

        elif class_type == 4:
            # Type 4 majority: wj ∈ [W/2, W], hj ∈ [H/2, H], dj ∈ [D/2, D]
            for i in range(n_items):
                if np.random.random() < 0.6:  # Type 4 - 60%
                    w = np.random.uniform(bin_size / 2, bin_size)
                    h = np.random.uniform(bin_size / 2, bin_size)
                    d = np.random.uniform(bin_size / 2, bin_size)
                else:  # Other types - 10% each
                    type_choice = np.random.choice([1, 2, 3, 5])
                    w, h, d = BenchmarkGenerator._generate_item_by_type(type_choice, bin_size)

                items.append(Item(w, d, h, i))

        elif class_type == 5:
            # Type 5 majority: wj ∈ [1, W/2], hj ∈ [1, H/2], dj ∈ [1, D/2]
            for i in range(n_items):
                if np.random.random() < 0.6:  # Type 5 - 60%
                    w = np.random.uniform(1, bin_size / 2)
                    h = np.random.uniform(1, bin_size / 2)
                    d = np.random.uniform(1, bin_size / 2)
                else:  # Other types - 10% each
                    type_choice = np.random.choice([1, 2, 3, 4])
                    w, h, d = BenchmarkGenerator._generate_item_by_type(type_choice, bin_size)

                items.append(Item(w, d, h, i))

        elif class_type == 6:
            # Class 6: wj, hj, dj ∈ [1, 10] and W = H = D = 10
            bin_template = Bin(10, 10, 10)
            for i in range(n_items):
                w = np.random.uniform(1, 10)
                h = np.random.uniform(1, 10)
                d = np.random.uniform(1, 10)
                items.append(Item(w, d, h, i))

        elif class_type == 7:
            # Class 7: wj, hj, dj ∈ [1, 35] and W = H = D = 40
            bin_template = Bin(40, 40, 40)
            for i in range(n_items):
                w = np.random.uniform(1, 35)
                h = np.random.uniform(1, 35)
                d = np.random.uniform(1, 35)
                items.append(Item(w, d, h, i))

        elif class_type == 8:
            # Class 8: wj, hj, dj ∈ [1, 100] and W = H = D = 100
            for i in range(n_items):
                w = np.random.uniform(1, 100)
                h = np.random.uniform(1, 100)
                d = np.random.uniform(1, 100)
                items.append(Item(w, d, h, i))

        return items, bin_template

    @staticmethod
    def _generate_item_by_type(item_type: int, bin_size: int) -> Tuple[float, float, float]:
        """Generate item dimensions by type"""
        if item_type == 1:
            w = np.random.uniform(1, bin_size / 2)
            h = np.random.uniform(2 * bin_size / 3, bin_size)
            d = np.random.uniform(2 * bin_size / 3, bin_size)
        elif item_type == 2:
            w = np.random.uniform(2 * bin_size / 3, bin_size)
            h = np.random.uniform(1, bin_size / 2)
            d = np.random.uniform(2 * bin_size / 3, bin_size)
        elif item_type == 3:
            w = np.random.uniform(2 * bin_size / 3, bin_size)
            h = np.random.uniform(2 * bin_size / 3, bin_size)
            d = np.random.uniform(1, bin_size / 2)
        elif item_type == 4:
            w = np.random.uniform(bin_size / 2, bin_size)
            h = np.random.uniform(bin_size / 2, bin_size)
            d = np.random.uniform(bin_size / 2, bin_size)
        elif item_type == 5:
            w = np.random.uniform(1, bin_size / 2)
            h = np.random.uniform(1, bin_size / 2)
            d = np.random.uniform(1, bin_size / 2)

        return w, h, d
