# -*- coding: utf-8 -*-
"""
Created on Tue Sep 9 08:26:17 2025

ExtremePointHeuristic.py - based on Extreme Point Heuristics from Crainic, Perboli, and Tadei (2008)

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
import time
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.art3d as art3d
from typing import List, Dict

# Import shared classes and methods
from src.common import Item, Bin, PlacedItem, ExtremePoint, SortingRule, MeritFunction, BinPacking3D, BenchmarkGenerator


class ExtremePointBinPacking3D(BinPacking3D):
    """Extend BinPacking3D with additional heuristic methods and utilities"""

    def get_bin_utilization(self) -> List[float]:
        """Calculate utilization for each bin"""
        utilizations = []
        for bin_items in self.bins:
            used_volume = sum(item.item.volume for item in bin_items)
            utilizations.append(used_volume / self.bin_template.volume)
        return utilizations

    def print_solution_summary(self):
        """Print summary of current solution"""
        print(f"Number of bins used: {len(self.bins)}")
        utilizations = self.get_bin_utilization()
        if utilizations:
            print(f"Average bin utilization: {np.mean(utilizations):.3f}")
        for bin_idx, bin_items in enumerate(self.bins):
            print(f"Bin {bin_idx}: {len(bin_items)} items")
            for item in bin_items:
                print(f"  {item.item} at ({item.x}, {item.y}, {item.z})")

    def c_epbfd(self, items: List[Item]) -> int:
        """Composite Extreme Point Best Fit Decreasing heuristic"""
        self.clear_solution()
        best_solution = None
        best_num_bins = float('inf')

        # Try different sorting rules
        sorting_rules = [
            SortingRule.VOLUME_HEIGHT,
            SortingRule.HEIGHT_VOLUME,
            SortingRule.AREA_HEIGHT,
            SortingRule.CLUSTERED_AREA_HEIGHT
        ]

        merit_functions = [
            MeritFunction.FREE_VOLUME,
            MeritFunction.RESIDUAL_SPACE
        ]

        for sort_rule in sorting_rules:
            for merit_func in merit_functions:
                # Create a temporary solver to avoid modifying self
                temp_solver = ExtremePointBinPacking3D(self.bin_template)
                num_bins = temp_solver.ep_bfd(items, sort_rule, merit_func)

                if num_bins < best_num_bins:
                    best_num_bins = num_bins
                    best_solution = copy.deepcopy(temp_solver.bins)
                    best_eps = copy.deepcopy(temp_solver.extreme_points)

        if best_solution is not None:
            self.bins = best_solution
            self.extreme_points = best_eps

        return len(self.bins)

    def is_solution_feasible(self) -> bool:
        """Check if the current complete solution is feasible"""
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
                    if self._items_overlap(item1.item,
                                           ExtremePoint(item1.x, item1.y, item1.z),
                                           item2):
                        return False
        return True

def visualize_packing_solution(solver: ExtremePointBinPacking3D, bin_idx: int = 0, title: str = "3D Bin Packing Solution"):
    """Visualize a single bin's packing solution"""
    if bin_idx >= len(solver.bins):
        print(f"Bin {bin_idx} doesn't exist")
        return

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Draw bin boundaries
    bin_template = solver.bin_template

    # Draw bin edges
    edges = [
        [(0, 0, 0), (bin_template.width, 0, 0)],
        [(0, 0, 0), (0, bin_template.depth, 0)],
        [(0, 0, 0), (0, 0, bin_template.height)],
        [(bin_template.width, 0, 0), (bin_template.width, bin_template.depth, 0)],
        [(bin_template.width, 0, 0), (bin_template.width, 0, bin_template.height)],
        [(0, bin_template.depth, 0), (bin_template.width, bin_template.depth, 0)],
        [(0, bin_template.depth, 0), (0, bin_template.depth, bin_template.height)],
        [(0, 0, bin_template.height), (bin_template.width, 0, bin_template.height)],
        [(0, 0, bin_template.height), (0, bin_template.depth, bin_template.height)],
        [(bin_template.width, bin_template.depth, 0), (bin_template.width, bin_template.depth, bin_template.height)],
        [(bin_template.width, 0, bin_template.height), (bin_template.width, bin_template.depth, bin_template.height)],
        [(0, bin_template.depth, bin_template.height), (bin_template.width, bin_template.depth, bin_template.height)]
    ]

    for edge in edges:
        points = np.array(edge)
        ax.plot3D(points[:, 0], points[:, 1], points[:, 2], 'k-', alpha=0.3)

    # Draw items
    colors = plt.cm.Set3(np.linspace(0, 1, len(solver.bins[bin_idx])))

    for i, placed_item in enumerate(solver.bins[bin_idx]):
        item = placed_item.item
        x, y, z = placed_item.x, placed_item.y, placed_item.z

        # Create the six faces of the rectangular item
        vertices = [
            [x, y, z],
            [x + item.width, y, z],
            [x + item.width, y + item.depth, z],
            [x, y + item.depth, z],
            [x, y, z + item.height],
            [x + item.width, y, z + item.height],
            [x + item.width, y + item.depth, z + item.height],
            [x, y + item.depth, z + item.height]
        ]

        vertices = np.array(vertices)

        # Define the faces using vertex indices
        faces = [
            [vertices[0], vertices[1], vertices[5], vertices[4]],  # front
            [vertices[7], vertices[6], vertices[2], vertices[3]],  # back
            [vertices[0], vertices[4], vertices[7], vertices[3]],  # left
            [vertices[1], vertices[2], vertices[6], vertices[5]],  # right
            [vertices[0], vertices[3], vertices[2], vertices[1]],  # bottom
            [vertices[4], vertices[5], vertices[6], vertices[7]]  # top
        ]

        # Add faces to the plot using Poly3DCollection
        poly = art3d.Poly3DCollection(faces, alpha=0.7, facecolors=colors[i], linewidths=0.5, edgecolors='k')
        ax.add_collection3d(poly)

        # Add item label at center
        center_x = x + item.width / 2
        center_y = y + item.depth / 2
        center_z = z + item.height / 2
        ax.text(center_x, center_y, center_z, f'{item.id}', fontsize=8)

    # Draw extreme points
    if bin_idx < len(solver.extreme_points):
        for ep in solver.extreme_points[bin_idx]:
            ax.scatter([ep.x], [ep.y], [ep.z], c='red', s=50, marker='o', alpha=0.8)

    ax.set_xlabel('Width (X)')
    ax.set_ylabel('Depth (Y)')
    ax.set_zlabel('Height (Z)')
    ax.set_title(f'{title} - Bin {bin_idx}')

    # Set equal aspect ratio
    max_range = max(bin_template.width, bin_template.depth, bin_template.height)
    ax.set_xlim([0, max_range])
    ax.set_ylim([0, max_range])
    ax.set_zlim([0, max_range])

    plt.tight_layout()
    plt.show()


def demonstrate_algorithm():
    """Demonstrate the extreme point algorithm with a small example"""
    print("=== Demonstration: Small 3D Bin Packing Example ===\n")

    # Create a small example
    bin_template = Bin(10, 10, 10)  # 10x10x10 bin
    items = [
        Item(4, 4, 3, 0),  # Item 0: 4x4x3
        Item(3, 3, 2, 1),  # Item 1: 3x3x2
        Item(2, 2, 4, 2),  # Item 2: 2x2x4
        Item(5, 2, 2, 3),  # Item 3: 5x2x2
        Item(3, 5, 1, 4),  # Item 4: 3x5x1
    ]

    print("Items to pack:")
    for item in items:
        print(f"  {item} (volume: {item.volume})")

    print(f"\nBin template: {bin_template.width}x{bin_template.depth}x{bin_template.height}")
    print(f"Bin volume: {bin_template.volume}")
    print(f"Total item volume: {sum(item.volume for item in items)}")

    # Test different algorithms
    solver = ExtremePointBinPacking3D(bin_template)

    print("\n=== Testing EP-FFD with Volume-Height sorting ===")
    num_bins_ffd = solver.ep_ffd(items, SortingRule.VOLUME_HEIGHT)
    print(f"EP-FFD result: {num_bins_ffd} bins")
    print(f"solution is {solver.is_solution_feasible()}")

    solver.print_solution_summary()

    print("\n=== Testing EP-BFD with Residual Space merit ===")
    num_bins_bfd = solver.ep_bfd(items, SortingRule.VOLUME_HEIGHT, MeritFunction.RESIDUAL_SPACE)
    print(f"EP-BFD result: {num_bins_bfd} bins")
    solver.print_solution_summary()

    print("\n=== Testing C-EPBFD Composite heuristic ===")
    num_bins_composite = solver.c_epbfd(items)
    print(f"C-EPBFD result: {num_bins_composite} bins")
    solver.print_solution_summary()

    # Visualize the best solution
    print(f"\nVisualizing solution with {len(solver.bins)} bins...")
    for i in range(min(2, len(solver.bins))):  # Show first 2 bins
        visualize_packing_solution(solver, i, f"C-EPBFD Solution")

    return solver, items


def compare_with_simple_heuristics(items: List[Item], bin_template: Bin):
    """Compare with simple heuristics for validation"""
    print("\n=== Comparison with Simple Heuristics ===")

    # Simple First Fit Decreasing (volume-based)
    def simple_ffd(items_list, bin_template):
        # Sort by volume (decreasing)
        sorted_items = sorted(items_list, key=lambda x: -x.volume)
        bins = []

        for item in sorted_items:
            placed = False
            for bin_items in bins:
                # Simple volume check (not considering 3D constraints)
                used_volume = sum(placed_item.volume for placed_item in bin_items)
                if used_volume + item.volume <= bin_template.volume:
                    bin_items.append(item)
                    placed = True
                    break

            if not placed:
                bins.append([item])

        return len(bins)

    # Simple First Fit Decreasing with basic 3D constraint
    def simple_3d_ffd(items_list, bin_template):
        sorted_items = sorted(items_list, key=lambda x: -x.volume)
        bins = []  # Each bin will store PlacedItem objects

        for item in sorted_items:
            placed = False
            for bin_items in bins:
                # Try to place at origin if possible (very simple)
                if len(bin_items) == 0:
                    if (item.width <= bin_template.width and
                            item.depth <= bin_template.depth and
                            item.height <= bin_template.height):
                        bin_items.append(PlacedItem(item, 0, 0, 0))
                        placed = True
                        break
                # For simplicity, just check if there's "room" (crude approximation)
                elif len(bin_items) < 3:  # Max 3 items per bin (arbitrary limit)
                    bin_items.append(PlacedItem(item, 0, 0, 0))
                    placed = True
                    break

            if not placed:
                bins.append([PlacedItem(item, 0, 0, 0)])

        return len(bins)

    # Run comparisons
    volume_ffd_bins = simple_ffd(items, bin_template)
    simple_3d_bins = simple_3d_ffd(items, bin_template)

    # Run our EP algorithms
    solver = ExtremePointBinPacking3D(bin_template)
    ep_ffd_bins = solver.ep_ffd(items, SortingRule.VOLUME_HEIGHT)

    solver_bfd = ExtremePointBinPacking3D(bin_template)
    ep_bfd_bins = solver_bfd.ep_bfd(items, SortingRule.VOLUME_HEIGHT, MeritFunction.RESIDUAL_SPACE)

    solver_composite = ExtremePointBinPacking3D(bin_template)
    c_epbfd_bins = solver_composite.c_epbfd(items)

    print(f"Simple Volume FFD:     {volume_ffd_bins} bins")
    print(f"Simple 3D FFD:         {simple_3d_bins} bins")
    print(f"EP-FFD:               {ep_ffd_bins} bins")
    print(f"EP-BFD:               {ep_bfd_bins} bins")
    print(f"C-EPBFD:              {c_epbfd_bins} bins")

    # Calculate theoretical lower bound (volume-based)
    total_volume = sum(item.volume for item in items)
    theoretical_lb = int(np.ceil(total_volume / bin_template.volume))
    print(f"Theoretical Lower Bound: {theoretical_lb} bins")

    return {
        'volume_ffd': volume_ffd_bins,
        'simple_3d': simple_3d_bins,
        'ep_ffd': ep_ffd_bins,
        'ep_bfd': ep_bfd_bins,
        'c_epbfd': c_epbfd_bins,
        'lower_bound': theoretical_lb
    }


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)

    print("3D Bin Packing with Extreme Point-Based Heuristics")
    print("Implementation based on Crainic, Perboli, and Tadei (2008)")
    print("=" * 60)

    # 1. Demonstrate with small example
    solver, demo_items = demonstrate_algorithm()

    # 2. Compare with simple heuristics
    bin_template = Bin(10, 10, 10)
    comparison_results = compare_with_simple_heuristics(demo_items, bin_template)

    # 3. Run benchmark comparison (commented out to avoid long execution)
    print("\n" + "=" * 60)
    print("Running limited benchmark comparison...")

    # Generate a few test instances
    test_instances = [
        (1, 20),  # Class 1, 20 items
        (5, 20),  # Class 5, 20 items
        (8, 30),  # Class 8, 30 items
    ]

    benchmark_results = []
    for class_type, n_items in test_instances:
        print(f"\nTesting Class {class_type}, {n_items} items...")
        items, bin_template = BenchmarkGenerator.generate_martello_instance(class_type, n_items)

        solver = ExtremePointBinPacking3D(bin_template)

        # Test key methods
        start_time = time.time()
        ep_ffd_bins = solver.ep_ffd(items, SortingRule.VOLUME_HEIGHT)
        ep_ffd_time = time.time() - start_time
        ep_ffd_util = np.mean(solver.get_bin_utilization())

        start_time = time.time()
        ep_bfd_bins = solver.ep_bfd(items, SortingRule.VOLUME_HEIGHT, MeritFunction.RESIDUAL_SPACE)
        ep_bfd_time = time.time() - start_time
        ep_bfd_util = np.mean(solver.get_bin_utilization())

        start_time = time.time()
        c_epbfd_bins = solver.c_epbfd(items)
        c_epbfd_time = time.time() - start_time
        c_epbfd_util = np.mean(solver.get_bin_utilization())

        print(f"  EP-FFD:   {ep_ffd_bins} bins, {ep_ffd_util:.3f} util, {ep_ffd_time:.4f}s")
        print(f"  EP-BFD:   {ep_bfd_bins} bins, {ep_bfd_util:.3f} util, {ep_bfd_time:.4f}s")
        print(f"  C-EPBFD:  {c_epbfd_bins} bins, {c_epbfd_util:.3f} util, {c_epbfd_time:.4f}s")

        benchmark_results.append({
            'class': class_type,
            'items': n_items,
            'ep_ffd': ep_ffd_bins,
            'ep_bfd': ep_bfd_bins,
            'c_epbfd': c_epbfd_bins
        })

    print("\n" + "=" * 60)
    print("IMPLEMENTATION SUMMARY")
    print("=" * 60)
    print("✓ Extreme Point concept implementation")
    print("✓ EP-FFD heuristic with multiple sorting rules")
    print("✓ EP-BFD heuristic with different merit functions")
    print("✓ C-EPBFD composite heuristic")
    print("✓ Martello benchmark instance generator")
    print("✓ 3D visualization capabilities")
    print("✓ Performance comparison framework")
    print("\nKey Features:")
    print("- Polynomial time complexity O(n³) as proven in paper")
    print("- Support for all sorting rules from paper")
    print("- Residual space merit function implementation")
    print("- Clustered sorting with parameter tuning")
    print("- Comprehensive benchmarking against simple heuristics")