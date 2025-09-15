# -*- coding: utf-8 -*-
"""
Created on Tue Sep 9 08:26:17 2025

AlnsExtremePoint.py - Adaptive Large Neighborhood Search for 3D Bin Packing

Based on Extreme Point Heuristics from Crainic, Perboli, and Tadei (2008)


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

from datetime import timedelta

import numpy as np
import random
import copy
import time
import math
from datetime import timedelta, datetime
#
from numpy.random import RandomState
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
from collections import defaultdict

from src.ExtremePointHeuristic import ExtremePointBinPacking3D
from src.classes.solution import Solution
from src.common import BenchmarkGenerator


class ImprovedObjectives:
    """Better objective functions for 3D bin packing ALNS"""

    @staticmethod
    def lexicographic_objective(solution: Solution, penalty: float = 1000.0) -> float:
        """Lexicographic: minimize bins first, then maximize utilization"""
        num_bins = solution.num_bins
        avg_util = solution.average_utilization
        # Strong preference for fewer bins
        return num_bins * penalty + (1.0 - avg_util) * 10

    @staticmethod
    def volume_waste_objective(solution: Solution, penalty: float = 1000.0) -> float:
        """Focus on minimizing wasted volume"""
        num_bins = solution.num_bins
        bin_volume = solution.bin_template.volume
        total_item_volume = sum(
            sum(pi.item.volume for pi in bin_items)
            for bin_items in solution.bins
        )
        wasted_volume = num_bins * bin_volume - total_item_volume
        return wasted_volume  # Direct waste minimization

    @staticmethod
    def hybrid_objective(solution: Solution) -> float:
        """Hybrid approach balancing bins and space efficiency"""
        num_bins = solution.num_bins
        avg_util = solution.average_utilization

        # Calculate utilization variance (penalty for uneven bins)
        utilizations = solution.get_bin_utilizations()
        util_variance = np.var(utilizations) if len(utilizations) > 1 else 0

        # Multi-component objective
        return (
                num_bins * 1000.0 +  # Primary: minimize bins
                (1.0 - avg_util) * 50.0 +  # Secondary: maximize utilization
                util_variance * 20.0  # Tertiary: balanced bins
        )

    @staticmethod
    def primal_integral_objective(solution: Solution, time_elapsed: float) -> float:
        """Time-aware objective function"""
        num_bins = solution.num_bins
        # Penalize solutions that take longer to find
        return num_bins * 1000.0 + time_elapsed * 0.1


class CompleteImprovedObjectives:
    """Complete set of improved objective functions for 3D bin packing ALNS"""

    @staticmethod
    def lexicographic_objective(solution: Solution, penalty: float = 1000.0) -> float:
        """Lexicographic: minimize bins first, then maximize utilization"""
        num_bins = solution.num_bins
        avg_util = solution.average_utilization
        # Strong preference for fewer bins
        return num_bins * penalty + (1.0 - avg_util) * 10

    @staticmethod
    def bin_objective(solution: Solution, penalty: float = 1000.0) -> float:
        """Lexicographic: minimize bins first, then maximize utilization"""
        num_bins = solution.num_bins
        return num_bins

    @staticmethod
    def volume_waste_objective(solution: Solution) -> float:
        """Focus on minimizing wasted volume"""
        stats = solution.get_solution_statistics()
        return stats['total_volume_wasted']

    @staticmethod
    def hybrid_objective(solution: Solution) -> float:
        """Hybrid approach balancing bins and space efficiency"""
        num_bins = solution.num_bins
        avg_util = solution.average_utilization

        # Calculate utilization variance (penalty for uneven bins)
        utilizations = solution.get_bin_utilizations()  # Returns List[float]
        util_variance = np.var(utilizations) if len(utilizations) > 1 else 0

        # Multi-component objective
        return (
                num_bins * 1000.0 +  # Primary: minimize bins
                (1.0 - avg_util) * 50.0 +  # Secondary: maximize utilization
                util_variance * 20.0  # Tertiary: balanced bins
        )

    @staticmethod
    def balanced_objective(solution: Solution) -> float:
        """Focus on balanced utilization across bins"""
        utilizations = solution.get_bin_utilizations()
        if not utilizations:
            return float('inf')

        num_bins = len(utilizations)
        avg_util = np.mean(utilizations)
        util_std = np.std(utilizations)

        # Minimize bins, maximize average utilization, minimize standard deviation
        return num_bins * 1000.0 + (1.0 - avg_util) * 100.0 + util_std * 50.0

    @staticmethod
    def balanced_objective_v2(solution: Solution) -> float:
        """Alternative balanced objective with different weighting"""
        utilizations = solution.get_bin_utilizations()
        if not utilizations:
            return float('inf')

        num_bins = len(utilizations)
        avg_util = np.mean(utilizations)
        min_util = np.min(utilizations)
        max_util = np.max(utilizations)
        util_range = max_util - min_util  # Difference between best and worst bin

        # Minimize bins, maximize min utilization, minimize range
        return (
                num_bins * 1000.0 +  # Primary: minimize bins
                (1.0 - min_util) * 200.0 +  # Secondary: maximize worst bin utilization
                util_range * 100.0  # Tertiary: minimize utilization spread
        )

    @staticmethod
    def balanced_objective_v3(solution: Solution) -> float:
        """Gini coefficient based balanced objective"""
        utilizations = solution.get_bin_utilizations()
        if not utilizations:
            return float('inf')

        num_bins = len(utilizations)
        avg_util = np.mean(utilizations)

        # Calculate Gini coefficient for utilization inequality
        gini = CompleteImprovedObjectives._calculate_gini_coefficient(utilizations)

        return (
                num_bins * 1000.0 +  # Primary: minimize bins
                (1.0 - avg_util) * 100.0 +  # Secondary: maximize utilization
                gini * 100.0  # Tertiary: minimize inequality (Gini)
        )

    @staticmethod
    def _calculate_gini_coefficient(values: List[float]) -> float:
        """Calculate Gini coefficient for measuring inequality"""
        if len(values) <= 1:
            return 0.0

        # Sort values
        sorted_values = sorted(values)
        n = len(sorted_values)

        # Calculate Gini coefficient
        numerator = 0.0
        for i, value in enumerate(sorted_values):
            numerator += (2 * (i + 1) - n - 1) * value

        denominator = n * sum(sorted_values)

        return numerator / denominator if denominator > 0 else 0.0

    @staticmethod
    def efficiency_objective(solution: Solution) -> float:
        """Focus on packing efficiency (items per bin)"""
        if not solution.bins:
            return float('inf')

        num_bins = solution.num_bins
        total_items = solution.get_items_count()
        avg_items_per_bin = total_items / num_bins

        # Maximize items per bin, minimize bins
        return num_bins * 1000.0 - avg_items_per_bin * 10.0

    @staticmethod
    def primal_integral_objective(solution: Solution, time_elapsed: float) -> float:
        """Time-aware objective function"""
        num_bins = solution.num_bins
        # Penalize solutions that take longer to find
        return num_bins * 1000.0 + time_elapsed * 0.1

    @staticmethod
    def weighted_utilization_objective(solution: Solution, weight_high_util: float = 2.0) -> float:
        """Objective that gives extra reward for high-utilization bins"""
        utilizations = solution.get_bin_utilizations()
        if not utilizations:
            return float('inf')

        num_bins = len(utilizations)

        # Calculate weighted utilization score
        total_weighted_util = 0.0
        for util in utilizations:
            if util > 0.8:  # High utilization bins get extra credit
                total_weighted_util += util * weight_high_util
            else:
                total_weighted_util += util

        avg_weighted_util = total_weighted_util / num_bins

        return num_bins * 1000.0 + (1.0 - avg_weighted_util) * 100.0


# Comprehensive objective comparison function
def compare_all_objectives(solution: Solution) -> dict:
    """Compare a solution using all available objective functions"""
    objectives = {
        'lexicographic': CompleteImprovedObjectives.lexicographic_objective(solution),
        'volume_waste': CompleteImprovedObjectives.volume_waste_objective(solution),
        'hybrid': CompleteImprovedObjectives.hybrid_objective(solution),
        'balanced': CompleteImprovedObjectives.balanced_objective(solution),
        'balanced_v2': CompleteImprovedObjectives.balanced_objective_v2(solution),
        'balanced_v3': CompleteImprovedObjectives.balanced_objective_v3(solution),
        'efficiency': CompleteImprovedObjectives.efficiency_objective(solution),
        'weighted_util': CompleteImprovedObjectives.weighted_utilization_objective(solution)
    }

    # Add solution statistics
    stats = solution.get_solution_statistics()

    return {
        'objectives': objectives,
        'statistics': stats,
        'utilizations': solution.get_bin_utilizations()
    }


# Example usage demonstrating all balanced objectives
def demonstrate_balanced_objectives():
    """Demonstrate different balanced objective functions"""
    print("=== BALANCED OBJECTIVE FUNCTIONS DEMONSTRATION ===\n")

    # Create a test solution with uneven utilization
    items, bin_template = BenchmarkGenerator.generate_martello_instance(5, 30, random_seed=42)

    solver = ExtremePointBinPacking3D(bin_template)
    solver.c_epbfd(items)
    solution = Solution(
        bins=copy.deepcopy(solver.bins),
        extreme_points=copy.deepcopy(solver.extreme_points),
        bin_template=bin_template
    )

    # Compare all objective functions
    comparison = compare_all_objectives(solution)

    print("Solution Statistics:")
    stats = comparison['statistics']
    utils = comparison['utilizations']

    print(f"  Bins: {stats['num_bins']}")
    print(f"  Average utilization: {stats['avg_utilization']:.3f}")
    print(f"  Min utilization: {stats['min_utilization']:.3f}")
    print(f"  Max utilization: {stats['max_utilization']:.3f}")
    print(f"  Utilization std: {stats['utilization_std']:.3f}")
    print(f"  Individual utilizations: {[f'{u:.3f}' for u in utils]}")
    print()

    print("Objective Function Values:")
    objectives = comparison['objectives']
    for obj_name, obj_value in objectives.items():
        print(f"  {obj_name:15s}: {obj_value:8.2f}")

    print()
    print("Balanced Objective Comparison:")
    print(f"  Standard balanced:     {objectives['balanced']:.2f}")
    print(f"  Range-based balanced:  {objectives['balanced_v2']:.2f}")
    print(f"  Gini-based balanced:   {objectives['balanced_v3']:.2f}")

    # Show which balanced objective is most sensitive to utilization imbalance
    print(f"\nInterpretation:")
    print(f"  Standard deviation: {stats['utilization_std']:.4f}")
    print(f"  Utilization range: {stats['max_utilization'] - stats['min_utilization']:.4f}")

    gini = CompleteImprovedObjectives._calculate_gini_coefficient(utils)
    print(f"  Gini coefficient: {gini:.4f}")


if __name__ == "__main__":
    demonstrate_balanced_objectives()