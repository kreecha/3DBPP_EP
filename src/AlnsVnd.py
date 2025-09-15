# -*- coding: utf-8 -*-
"""
Created on Tue Sep 9 08:26:17 2025

AlnsVnD.py - Adaptive Large Neighborhood Search (ALNS) and Variable Neighborhood Descent (VND) are both metaheuristics
used to solve complex combinatorial optimization problems, such as the Vehicle Routing Problem.This file purpose to
solve 3D Bin Packing.

Extreme Point Heuristics, Adaptive Large Neighborhood Search and Variable Neighborhood Descent for 3D Bin Packing

Based on:
 1) Extreme Point Heuristics from Crainic, Perboli, and Tadei (2008)
 2) Pisinger, D., & Ropke, S. (2007). A general heuristic for vehicle routing problems.
 Computers & Operations Research, 34(8), 2403-2435.
 3) VND Hemmelmayr, V. C., Cordeau, J. F., & Crainic, T. G. (2012). An adaptive large neighborhood search heuristic
 for the multi-echelon vehicle routing problem.

@author: Kreecha Puphaiboon

MIT License

Copyright (c) 2025 Kreecha Puphaiboon

"""

from datetime import timedelta

import numpy as np
import random
import copy
import math
import time
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

from src.common import Item, Bin, PlacedItem, ExtremePoint, SortingRule, MeritFunction, BenchmarkGenerator
from src.ExtremePointHeuristic import ExtremePointBinPacking3D
from src.classes.solution import Solution
from src.AlnsGraspOperators import get_all_grasp_operators

from src.BinVisualizer import BinPackingVisualizer
from src.destroy.destroy_operators import AdaptiveRandomDestroy, UtilizationBasedDestroy
from src.repair.repair_operators import RegretRepair, SmartRegretRepair, RobustGreedyRepair
from src.shaw_regret_operators import AdaptiveShawDestroy, AdaptiveShawSizeDestroy
from src.vnd_implementation import VariableNeighborhoodDescent, create_vnd_engine

# TODO: Refactor Operators into destroy and repair
# TODO: Test on specific instances: filenames = ['thpack1.txt', 'thpack2.txt']  see main_benchmark_driver.py

class HybridALNS_VND:
    """Hybrid ALNS-VND for 3D Bin Packing following the provided pseudocode"""

    def __init__(self,
                 bin_template: Bin,
                 destroy_operators: List,
                 repair_operators: List,
                 max_iterations: int = 2000,
                 max_time: float = 120.0,
                 vnd_frequency: float = 0.3,  # Apply VND to 30% of solutions
                 vnd_max_time: float = 5.0,  # Max 5 seconds per VND call
                 objective_type: str = 'hybrid',
                 random_seed: int = 42):

        self.bin_template = bin_template
        self.destroy_operators = destroy_operators
        self.repair_operators = repair_operators
        self.max_iterations = max_iterations
        self.max_time = max_time
        self.vnd_frequency = vnd_frequency
        self.objective_type = objective_type

        # Initialize VND component
        self.vnd = create_vnd_engine(max_time_per_search=vnd_max_time)

        # Initialize ALNS components
        from src.operator_selection import RouletteWheel
        self.operator_selector = RouletteWheel(
            scores=[100.0, 50.0, 20.0, 5.0],
            # scores=[100.0, 50.0, 20.0, 1.0],  # POOR
            decay=0.80,
            # decay=0.99,
            num_destroy=len(self.destroy_operators),
            num_repair=len(self.repair_operators)
        )

        self.rnd_state = np.random.RandomState(random_seed)

        # Statistics tracking
        self.stats = {
            'iterations': 0,
            'vnd_applications': 0,
            'vnd_improvements': 0,
            'alns_improvements': 0,
            'total_improvements': 0,
            'best_obj_history': [],
            'vnd_time_total': 0.0,
            'alns_time_total': 0.0
        }

        # Convergence tracking
        self.convergence_data = {
            'iterations': [],
            'best_obj': [],
            'current_obj': [],
            'vnd_applied': [],
            'improvement_type': []  # 'alns', 'vnd', or 'none'
        }

    def solve(self, items: List[Item]) -> Solution:
        """Main hybrid ALNS-VND algorithm following the pseudocode"""
        print(f"Starting Hybrid ALNS-VND with {len(items)} items...")

        # Initialize solution
        initial_solution = self._create_initial_solution(items)
        current_solution = initial_solution.copy()
        best_solution = initial_solution.copy()

        # Calculate initial objectives
        current_obj = self._calculate_objective(current_solution)
        best_obj = current_obj

        print(f"Initial solution: {best_solution.num_bins} bins, "
              f"util={best_solution.average_utilization:.3f}, obj={best_obj:.2f}")

        # Initialize algorithm state
        start_time = time.time()
        iteration = 0
        temperature = self._calculate_initial_temperature()
        no_improvement_count = 0

        # Main ALNS-VND loop
        while iteration < self.max_iterations and (time.time() - start_time) < self.max_time:
            alns_start_time = time.time()

            # 1. ALNS operator selection (Adaptive step)
            destroy_idx, repair_idx = self.operator_selector.select_operators(self.rnd_state)
            destroy_operator = self.destroy_operators[destroy_idx]
            repair_operator = self.repair_operators[repair_idx]

            # 2. Ruin-and-recreate
            destroyed_solution = current_solution.copy()
            removed_items = destroy_operator.destroy(destroyed_solution, self.rnd_state)

            if not removed_items:
                iteration += 1
                continue

            repair_success = repair_operator.repair(destroyed_solution, removed_items,
                                                    self.bin_template, self.rnd_state)

            if not repair_success:
                iteration += 1
                continue

            repaired_solution = destroyed_solution
            alns_time = time.time() - alns_start_time
            self.stats['alns_time_total'] += alns_time

            # 3. Apply VND for local intensification (conditionally)
            vnd_applied = False
            candidate_solution = repaired_solution

            if self._should_apply_vnd(iteration, repaired_solution):
                vnd_start_time = time.time()
                vnd_solution = self.vnd.search(repaired_solution, self._calculate_objective)
                vnd_time = time.time() - vnd_start_time

                self.stats['vnd_time_total'] += vnd_time
                self.stats['vnd_applications'] += 1
                vnd_applied = True

                # Check if VND improved the solution
                repaired_obj = self._calculate_objective(repaired_solution)
                vnd_obj = self._calculate_objective(vnd_solution)

                if vnd_obj < repaired_obj - 1e-6:
                    candidate_solution = vnd_solution
                    self.stats['vnd_improvements'] += 1
                else:
                    candidate_solution = repaired_solution

            # 4. ALNS acceptance criterion
            candidate_obj = self._calculate_objective(candidate_solution)
            improvement_type = 'none'

            if self._accept_solution(candidate_solution, current_solution, temperature):
                current_solution = candidate_solution
                current_obj = candidate_obj

                # Update best solution if improved
                if candidate_obj < best_obj - 1:
                    best_solution = candidate_solution.copy()
                    best_obj = candidate_obj
                    improvement_type = 'vnd' if vnd_applied and vnd_obj < repaired_obj - 1e-6 else 'alns'
                    self.stats['total_improvements'] += 1
                    no_improvement_count = 0

                    print(f"Iteration {iteration}: NEW BEST - {best_solution.num_bins} bins, "
                          f"util={best_solution.average_utilization:.3f}, obj={best_obj:.2f} "
                          f"[{destroy_operator.name} + {repair_operator.name}"
                          f"{' + VND' if vnd_applied else ''}]")
                else:
                    improvement_type = 'accepted'
            else:
                no_improvement_count += 1

            # 5. Update ALNS operator weights
            outcome = self._determine_outcome(candidate_obj, current_obj, best_obj)
            self.operator_selector.update(destroy_idx, repair_idx, outcome)
            destroy_operator.usage_count += 1
            repair_operator.usage_count += 1

            # Track convergence data
            self._update_convergence_data(iteration, current_obj, best_obj,
                                          vnd_applied, improvement_type)

            # Update temperature and iteration counter
            temperature = max(1.0, temperature * 0.9995)
            iteration += 1

            # Diversification if stuck
            if no_improvement_count > 200:
                temperature = self._calculate_initial_temperature() * 0.5
                no_improvement_count = 0

        # Final statistics
        elapsed_time = time.time() - start_time
        self.stats['iterations'] = iteration

        print(f"\nHybrid ALNS-VND completed:")
        print(f"  Total time: {elapsed_time:.1f}s")
        print(
            f"  ALNS time: {self.stats['alns_time_total']:.1f}s ({100 * self.stats['alns_time_total'] / elapsed_time:.1f}%)")
        print(
            f"  VND time:  {self.stats['vnd_time_total']:.1f}s ({100 * self.stats['vnd_time_total'] / elapsed_time:.1f}%)")
        print(
            f"  Final: {best_solution.num_bins} bins, util={best_solution.average_utilization:.3f}, obj={best_obj:.2f}")
        print(
            f"  VND applications: {self.stats['vnd_applications']}/{iteration} ({100 * self.stats['vnd_applications'] / iteration:.1f}%)")
        print(
            f"  VND improvements: {self.stats['vnd_improvements']}/{self.stats['vnd_applications']} ({100 * self.stats['vnd_improvements'] / max(1, self.stats['vnd_applications']):.1f}%)")

        return best_solution

    def _create_initial_solution(self, items: List[Item]) -> Solution:
        """Create initial solution using C-EPBFD"""
        solver = ExtremePointBinPacking3D(self.bin_template)
        solver.c_epbfd(items)
        return Solution(
            bins=copy.deepcopy(solver.bins),
            extreme_points=copy.deepcopy(solver.extreme_points),
            bin_template=self.bin_template
        )

    def _calculate_objective(self, solution: Solution) -> float:
        """Calculate objective using selected function"""
        # Use CompleteImprovedObjectives from previous implementations
        if self.objective_type == 'hybrid':
            return solution.num_bins * 1000.0 + (1.0 - solution.average_utilization) * 50.0 + np.var(
                solution.get_bin_utilizations()) * 20.0
        elif self.objective_type == 'volume_waste':
            stats = solution.get_solution_statistics()
            return stats['total_volume_wasted']
        elif self.objective_type == 'lexicographic':
            return solution.num_bins * 1000.0 + (1.0 - solution.average_utilization) * 10.0
        else:
            # Default hybrid
            return solution.num_bins * 1000.0 + (1.0 - solution.average_utilization) * 50.0

    def _calculate_initial_temperature(self) -> float:
        """Calculate appropriate initial temperature"""
        return 100.0  # Reasonable default for bin packing objectives

    def _should_apply_vnd(self, iteration: int, solution: Solution) -> bool:
        """Decide whether to apply VND to this solution"""
        # Apply VND with specified frequency, but more often early in the search
        base_probability = self.vnd_frequency

        # Higher probability for better solutions
        if solution.num_bins <= 3:
            base_probability *= 1.5
        elif solution.average_utilization > 0.8:
            base_probability *= 1.2

        # Higher probability early in search
        if iteration < 500:
            base_probability *= 1.3

        # Randomized application
        return self.rnd_state.random() < base_probability

    def _accept_solution(self, candidate_solution: Solution, current_solution: Solution,
                         temperature: float) -> bool:
        """ALNS acceptance criterion"""
        candidate_obj = self._calculate_objective(candidate_solution)
        current_obj = self._calculate_objective(current_solution)

        if candidate_obj <= current_obj:
            return True

        # Simulated annealing acceptance
        delta = candidate_obj - current_obj
        acceptance_prob = math.exp(-delta / temperature)
        return self.rnd_state.random() < acceptance_prob

    def _determine_outcome(self, candidate_obj: float, current_obj: float, best_obj: float) -> int:
        """Determine outcome for operator weight update"""
        if candidate_obj < best_obj - 1e-6:
            return 0  # New global best
        elif candidate_obj < current_obj - 1e-6:
            return 1  # Better than current
        elif candidate_obj <= current_obj + 1e-6:
            return 2  # Accepted (same quality)
        else:
            return 3  # Rejected

    def _update_convergence_data(self, iteration: int, current_obj: float, best_obj: float,
                                 vnd_applied: bool, improvement_type: str):
        """Update convergence tracking data"""
        self.convergence_data['iterations'].append(iteration)
        self.convergence_data['current_obj'].append(current_obj)
        self.convergence_data['best_obj'].append(best_obj)
        self.convergence_data['vnd_applied'].append(vnd_applied)
        self.convergence_data['improvement_type'].append(improvement_type)

    def print_detailed_statistics(self):
        """Print comprehensive algorithm statistics"""
        print("\n=== HYBRID ALNS-VND DETAILED STATISTICS ===")

        # General statistics
        print(f"\nGeneral Performance:")
        print(f"  Total iterations: {self.stats['iterations']}")
        print(f"  Total improvements: {self.stats['total_improvements']}")
        print(f"  VND applications: {self.stats['vnd_applications']}")
        print(f"  VND improvements: {self.stats['vnd_improvements']}")
        print(
            f"  VND success rate: {100 * self.stats['vnd_improvements'] / max(1, self.stats['vnd_applications']):.1f}%")

        # Time breakdown
        total_time = self.stats['alns_time_total'] + self.stats['vnd_time_total']
        print(f"\nTime Breakdown:")
        print(f"  Total time: {total_time:.1f}s")
        print(
            f"  ALNS time: {self.stats['alns_time_total']:.1f}s ({100 * self.stats['alns_time_total'] / total_time:.1f}%)")
        print(
            f"  VND time: {self.stats['vnd_time_total']:.1f}s ({100 * self.stats['vnd_time_total'] / total_time:.1f}%)")
        print(
            f"  Avg VND time: {self.stats['vnd_time_total'] / max(1, self.stats['vnd_applications']):.2f}s per application")

        # Operator performance
        print(f"\nOperator Performance:")
        print("Destroy Operators:")
        for idx, op in enumerate(self.destroy_operators):
            weight = self.operator_selector.destroy_weights[idx]
            usage = op.usage_count
            print(f"  {op.name:<25}: weight={weight:6.1f}, usage={usage:4d}")

        print("Repair Operators:")
        for idx, op in enumerate(self.repair_operators):
            weight = self.operator_selector.repair_weights[idx]
            usage = op.usage_count
            print(f"  {op.name:<25}: weight={weight:6.1f}, usage={usage:4d}")

        # VND neighborhood performance
        print(f"\nVND Neighborhood Performance:")
        self.vnd.print_statistics()

    def get_convergence_data(self) -> Dict:
        """Get convergence data for plotting"""
        return self.convergence_data


# =============================================================================
# FACTORY FUNCTION FOR HYBRID ALNS-VND
# =============================================================================

def create_hybrid_alns_vnd(bin_template: Bin,
                           use_grasp_operators: bool = True,
                           max_iterations: int = 2000,
                           max_time: float = 120.0,
                           vnd_frequency: float = 0.3,
                           random_seed: int = 42) -> HybridALNS_VND:
    """Create a hybrid ALNS-VND instance with specified configuration"""

    destroy_operators = [
        # RandomDestroy(min_remove=4, max_remove=5),
        # WorstBinDestroy(num_bins_to_target=1),
        # LargeItemDestroy(percentage=0.2),
        AdaptiveRandomDestroy(),    # min_rate: float = 0.10, max_rate: float = 0.30
        UtilizationBasedDestroy(),
        # AdaptiveShawSizeDestroy(),
    ]

    repair_operators = [
        RobustGreedyRepair(),
        # SafeBestFitRepair(),
        # RobustRegretRepair(k=1),
        RegretRepair(k=2),
    ]

    # Add GRASP operators if requested
    if use_grasp_operators:
        grasp_destroy, grasp_repair = get_all_grasp_operators()
        destroy_operators.extend(grasp_destroy)
        repair_operators.extend(grasp_repair)

    return HybridALNS_VND(
        bin_template=bin_template,
        destroy_operators=destroy_operators,
        repair_operators=repair_operators,
        max_iterations=max_iterations,
        max_time=max_time,
        vnd_frequency=vnd_frequency,
        random_seed=random_seed
    )


# =============================================================================
# COMPARISON AND DEMONSTRATION FUNCTIONS
# =============================================================================

def compare_alns_vs_hybrid(random_seed: int = 42, _is_plot=False):
    """Compare pure ALNS vs Hybrid ALNS-VND"""
    print("=== ALNS vs HYBRID ALNS-VND COMPARISON ===\n")

    test_instances = [
        (1, 30),  # Class 1, 30 items
        (5, 30),  # Class 5, 30 items
        (8, 40),  # Class 8, 40 items
    ]

    results = []

    MAX_ITERATION = 2000

    for class_type, n_items in test_instances:
        print(f"Testing Class {class_type}, {n_items} items...")

        # Generate instance

        items, bin_template = BenchmarkGenerator.generate_martello_instance(
            class_type, n_items, random_seed=random_seed)

        # Test EP Composite
        solver_composite = ExtremePointBinPacking3D(bin_template)
        c_epbfd_bins = solver_composite.c_epbfd(items)
        c_epbfd_util = np.mean(solver_composite.get_bin_utilization())
        print(f"  C-EPBFD:  {c_epbfd_bins} bins, {c_epbfd_util:.3f} util")

        # Test pure ALNS
        print("  Running pure ALNS...")
        pure_alns = create_hybrid_alns_vnd(
            bin_template,
            use_grasp_operators=True,
            max_iterations=MAX_ITERATION,
            max_time=60.0,
            vnd_frequency=0.0,  # No VND
            random_seed=random_seed
        )

        pure_solution = pure_alns.solve(items)

        # Test Hybrid ALNS-VND
        print("  Running Hybrid ALNS-VND...")
        hybrid_alns = create_hybrid_alns_vnd(
            bin_template,
            use_grasp_operators=True,
            max_iterations=MAX_ITERATION,
            max_time=60.0,
            vnd_frequency=0.3,  # 30% VND
            random_seed=random_seed
        )

        hybrid_solution = hybrid_alns.solve(items)
        items_packed = hybrid_solution.get_items_count()
        items_lost = len(items) - items_packed
        if items_lost > 0:
            print(f"⚠️  WARNING: {items_lost} items were lost during ALNS_VND!")
            # raise Error


        # Record results
        result = {
            'instance': f"Class {class_type} ({n_items} items)",
            'ep_bins': c_epbfd_bins,
            'ep_util': c_epbfd_util,
            'pure_bins': pure_solution.num_bins,
            'pure_util': pure_solution.average_utilization,
            'hybrid_bins': hybrid_solution.num_bins,
            'hybrid_util': hybrid_solution.average_utilization,
            'items_packed': items_packed,
            'items_lost': items_lost,
            'bin_improvement': pure_solution.num_bins - hybrid_solution.num_bins,
            'util_improvement': hybrid_solution.average_utilization - pure_solution.average_utilization,
            'vnd_applications': hybrid_alns.stats['vnd_applications'],
            'vnd_improvements': hybrid_alns.stats['vnd_improvements']
        }

        results.append(result)
        print(f"  EP:  {result['ep_bins']} bins, {result['ep_util']:.3f} util")
        print(f"  Pure ALNS:  {result['pure_bins']} bins, {result['pure_util']:.3f} util")
        print(f"  Hybrid:     {result['hybrid_bins']} bins, {result['hybrid_util']:.3f} util")
        print(f"  VND stats:  {result['vnd_improvements']}/{result['vnd_applications']} improvements")
        print()

    # Summary table
    print("=== COMPARISON SUMMARY ===")
    print("Instance              |      EP | Pure ALNS | Hybrid   | Bin Δ | Util Δ  | VND Apps")
    print("-" * 75)

    for result in results:
        print(f"{result['instance']:<20} | {result['ep_bins']:8d} |{result['pure_bins']:8d} | {result['hybrid_bins']:8d} | "
              f"{result['bin_improvement']:+5d} | {result['util_improvement']:+.3f} | {result['vnd_applications']:7d}")

    if _is_plot:

        # Create visualizer
        visualizer = BinPackingVisualizer(hybrid_solution)

        print("2. All bins overview...")
        visualizer.plot_all_bins_3d(max_bins=hybrid_solution.num_bins)

    return results


def demonstrate_hybrid_alns_vnd():
    """Demonstrate the Hybrid ALNS-VND algorithm"""
    print("=== HYBRID ALNS-VND DEMONSTRATION ===\n")

    # Create test problem
    from common import BenchmarkGenerator
    items, bin_template = BenchmarkGenerator.generate_martello_instance(5, 35, random_seed=42)

    print(f"Test instance: {len(items)} items")
    print(f"Bin size: {bin_template.width}x{bin_template.depth}x{bin_template.height}")

    # Run Hybrid ALNS-VND
    hybrid = create_hybrid_alns_vnd(
        bin_template=bin_template,
        use_grasp_operators=False,
        max_iterations=3000,
        max_time=90.0,
        vnd_frequency=0.4,  # Apply VND to 40% of solutions
        random_seed=42
    )

    solution = hybrid.solve(items)

    # Print detailed results
    hybrid.print_detailed_statistics()

    return hybrid, solution


if __name__ == "__main__":
    start_time = time.time()
    # Run
    print("Running Hybrid ALNS-VND...\n")

    print("\n" + "=" * 80)

    # Compare EP, ALNS vs Hybrid
    compare_alns_vs_hybrid(random_seed=42, _is_plot=False)

    end_time = time.time()
    end = end_time - start_time
    print(f"==" * 60)
    print('Finished performing everything, time elapsed {}'.format(str(timedelta(seconds=end))))
    print(f"==" * 60)