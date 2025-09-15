"""
from multiple_runs_analyzer import run_multiple_trials_analysis

# Run with default settings (10 runs per instance)
results = run_multiple_trials_analysis(n_runs=10)

# Or run basic version without pandas dependency
from multiple_runs_analyzer import run_basic_multiple_trials
run_basic_multiple_trials(n_runs=5)


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
import pandas as pd
from datetime import timedelta
import time
from typing import List, Dict, Tuple
import copy
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

from src.common import Item, Bin, BenchmarkGenerator
from src.ExtremePointHeuristic import ExtremePointBinPacking3D
from src.AlnsVnd import create_hybrid_alns_vnd

# Try to import pandas, use fallback if not available
try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("Warning: pandas not available. Using basic statistics only.")

# Try to import scipy for statistical tests
try:
    from scipy import stats as scipy_stats

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def run_single_trial(args):
    """Wrapper function for multiprocessing to run a single trial"""
    items, bin_template, class_type, n_items, run, seed = args
    analyzer = MultipleRunsAnalyzer._run_single_trial_static(items, bin_template, class_type, n_items, run, seed)
    return analyzer


class MultipleRunsAnalyzer:
    """Statistical analysis for comparing EP, ALNS, and ALNS-VND algorithms"""

    def __init__(self, n_runs: int = 10, max_iterations: int = 2000, max_time: float = 60.0):
        self.n_runs = n_runs
        self.max_iterations = max_iterations
        self.max_time = max_time
        self.results = []

    @staticmethod
    def _run_single_trial_static(items: List[Item], bin_template: Bin,
                                 class_type: int, n_items: int, run: int, seed: int) -> Dict:
        """Static method to run a single trial, suitable for multiprocessing"""
        result = {
            'class_type': class_type,
            'n_items': n_items,
            'run': run,
            'seed': seed,
            'total_volume': sum(item.volume for item in items),
            'bin_volume': bin_template.volume
        }

        # 1. Extreme Point (C-EPBFD)
        start_time = time.time()
        ep_solver = ExtremePointBinPacking3D(bin_template)
        ep_bins = ep_solver.c_epbfd(copy.deepcopy(items))
        ep_util = np.mean(ep_solver.get_bin_utilization()) if ep_bins > 0 else 0.0
        ep_time = time.time() - start_time

        result.update({
            'ep_bins': ep_bins,
            'ep_utilization': ep_util,
            'ep_time': ep_time
        })

        # 2. Pure ALNS (VND frequency = 0)
        start_time = time.time()
        try:
            pure_alns = create_hybrid_alns_vnd(
                bin_template,
                use_grasp_operators=True,
                max_iterations=2000,
                max_time=60.0,
                vnd_frequency=0.0,  # No VND
                random_seed=seed + 1
            )

            alns_solution = pure_alns.solve(copy.deepcopy(items))
            alns_time = time.time() - start_time

            result.update({
                'alns_bins': alns_solution.num_bins,
                'alns_utilization': alns_solution.average_utilization,
                'alns_time': alns_time,
                'alns_iterations': pure_alns.stats['iterations']
            })

        except ImportError:
            print(f"    Warning (Run {run}): ALNS not available, skipping... (Class {class_type}, {n_items} items)")
            result.update({
                'alns_bins': ep_bins,  # Fallback to EP result
                'alns_utilization': ep_util,
                'alns_time': 0.0,
                'alns_iterations': 0
            })

        # 3. Hybrid ALNS-VND
        start_time = time.time()
        try:
            hybrid_alns = create_hybrid_alns_vnd(
                bin_template,
                use_grasp_operators=True,
                max_iterations=2000,
                max_time=60.0,
                vnd_frequency=0.3,  # 30% VND
                random_seed=seed + 2
            )

            hybrid_solution = hybrid_alns.solve(copy.deepcopy(items))
            hybrid_time = time.time() - start_time

            # Check for lost items
            items_packed = hybrid_solution.get_items_count()
            items_lost = len(items) - items_packed

            result.update({
                'hybrid_bins': hybrid_solution.num_bins,
                'hybrid_utilization': hybrid_solution.average_utilization,
                'hybrid_time': hybrid_time,
                'hybrid_iterations': hybrid_alns.stats['iterations'],
                'vnd_applications': hybrid_alns.stats['vnd_applications'],
                'vnd_improvements': hybrid_alns.stats['vnd_improvements'],
                'items_lost': items_lost
            })

        except ImportError:
            print(
                f"    Warning (Run {run}): Hybrid ALNS-VND not available, skipping... (Class {class_type}, {n_items} items)")
            result.update({
                'hybrid_bins': ep_bins,  # Fallback to EP result
                'hybrid_utilization': ep_util,
                'hybrid_time': 0.0,
                'hybrid_iterations': 0,
                'vnd_applications': 0,
                'vnd_improvements': 0,
                'items_lost': 0
            })

        # Calculate improvements
        result['alns_vs_ep_bins'] = result['ep_bins'] - result['alns_bins']
        result['hybrid_vs_ep_bins'] = result['ep_bins'] - result['hybrid_bins']
        result['hybrid_vs_alns_bins'] = result['alns_bins'] - result['hybrid_bins']

        result['alns_vs_ep_util'] = result['ep_utilization'] - result['alns_utilization']
        result['hybrid_vs_ep_util'] = result['ep_utilization'] - result['hybrid_utilization']
        result['hybrid_vs_alns_util'] = result['alns_utilization'] - result['hybrid_utilization']

        return result

    def run_comprehensive_comparison(self, test_instances: List[Tuple[int, int]] = None):
        """Run comprehensive comparison across multiple instances and runs"""

        if test_instances is None:
            test_instances = [
                (1, 30),  # Class 1, 30 items
                (5, 30),  # Class 5, 30 items
                (8, 40),  # Class 8, 40 items

                (1, 60),  # Class 1, 60 items
                (5, 60),  # Class 5, 60 items
                (8, 80),  # Class 8, 80 items

                (1, 90),  # Class 1, 90 items
                (5, 90),  # Class 5, 90 items
                (8, 160),  # Class 8, 160 items

                (1, 500),  # Class 1, 90 items
                (5, 500),  # Class 5, 90 items
                (8, 1000),  # Class 8, 160 items
            ]

        print(f"=== COMPREHENSIVE STATISTICAL COMPARISON ===")
        print(f"Running {self.n_runs} trials for each instance...")
        print(f"Test instances: {len(test_instances)}")
        print(f"Total experiments: {len(test_instances) * self.n_runs}")
        print()

        all_results = []

        # Use multiprocessing to run trials in parallel
        with Pool(processes=cpu_count()) as pool:
            for instance_idx, (class_type, n_items) in enumerate(test_instances):
                print(
                    f"Processing Instance {instance_idx + 1}/{len(test_instances)}: Class {class_type}, {n_items} items")

                # Prepare arguments for each trial
                trial_args = []
                for run in range(self.n_runs):
                    base_seed = 42 + run * 1000 + class_type * 100 + n_items
                    items, bin_template = BenchmarkGenerator.generate_martello_instance(
                        class_type, n_items, random_seed=base_seed
                    )
                    trial_args.append((items, bin_template, class_type, n_items, run, base_seed))

                # Run trials in parallel
                instance_results = list(tqdm(pool.imap(run_single_trial, trial_args),
                                             total=self.n_runs,
                                             desc=f"Class {class_type} ({n_items} items)"))

                all_results.extend(instance_results)

                # Print instance summary
                self._print_instance_summary(instance_results, class_type, n_items)
                print()

        # Overall analysis
        self._print_overall_analysis(all_results)
        self._create_summary_tables(all_results)

        return all_results

    def _print_instance_summary(self, instance_results: List[Dict], class_type: int, n_items: int):
        """Print summary for a single instance type"""
        ep_bins = [r['ep_bins'] for r in instance_results]
        alns_bins = [r['alns_bins'] for r in instance_results]
        hybrid_bins = [r['hybrid_bins'] for r in instance_results]

        print(f"  Instance Summary (Class {class_type}, {n_items} items):")
        print(f"    ExtremePoint: {np.mean(ep_bins):.1f} ± {np.std(ep_bins):.1f} bins")
        print(f"    Pure ALNS:    {np.mean(alns_bins):.1f} ± {np.std(alns_bins):.1f} bins")
        print(f"    Hybrid ALNS-VND: {np.mean(hybrid_bins):.1f} ± {np.std(hybrid_bins):.1f} bins")

        # Check for lost items
        lost_items = [r['items_lost'] for r in instance_results]
        if any(lost_items):
            print(f"    Warning: {sum(lost_items)} items lost across {sum(1 for x in lost_items if x > 0)} runs")

    def _print_overall_analysis(self, all_results: List[Dict]):
        """Print overall statistical analysis"""
        # Bin count improvements
        alns_vs_ep_bins = np.mean([r['alns_vs_ep_bins'] for r in all_results])
        hybrid_vs_ep_bins = np.mean([r['hybrid_vs_ep_bins'] for r in all_results])
        hybrid_vs_alns_bins = np.mean([r['hybrid_vs_alns_bins'] for r in all_results])

        alns_vs_ep_util = np.mean([r['alns_vs_ep_util'] for r in all_results])
        hybrid_vs_ep_util = np.mean([r['hybrid_vs_ep_util'] for r in all_results])
        hybrid_vs_alns_util = np.mean([r['hybrid_vs_alns_util'] for r in all_results])

        print(f"\nAverage Improvements:")
        print(f"  ALNS vs EP: {alns_vs_ep_bins:+.2f} bins, {alns_vs_ep_util:+.3f} util")
        print(f"  Hybrid vs EP: {hybrid_vs_ep_bins:+.2f} bins, {hybrid_vs_ep_util:+.3f} util")
        print(f"  Hybrid vs ALNS: {hybrid_vs_alns_bins:+.2f} bins, {hybrid_vs_alns_util:+.3f} util")

        # Time analysis
        ep_time_avg = np.mean([r['ep_time'] for r in all_results])
        alns_time_avg = np.mean([r['alns_time'] for r in all_results])
        hybrid_time_avg = np.mean([r['hybrid_time'] for r in all_results])

        print(f"\nAverage Runtime (seconds):")
        print(f"  ExtremePoint: {ep_time_avg:.2f}s")
        print(f"  Pure ALNS: {alns_time_avg:.1f}s")
        print(f"  Hybrid ALNS-VND: {hybrid_time_avg:.1f}s")

        # VND effectiveness
        total_vnd_apps = sum(r['vnd_applications'] for r in all_results)
        total_vnd_imps = sum(r['vnd_improvements'] for r in all_results)
        vnd_success_rate = (total_vnd_imps / total_vnd_apps * 100) if total_vnd_apps > 0 else 0

        print(f"\nVND Effectiveness:")
        print(f"  Applications: {total_vnd_apps} total")
        print(f"  Improvements: {total_vnd_imps} total")
        print(f"  Success rate: {vnd_success_rate:.1f}%")

    def _create_summary_tables(self, all_results: List[Dict]):
        """Create detailed summary tables with ANOVA and post-hoc analysis"""
        print("\n=== DETAILED SUMMARY BY INSTANCE TYPE ===")

        # Group by instance type
        instance_groups = {}
        for result in all_results:
            key = (result['class_type'], result['n_items'])
            if key not in instance_groups:
                instance_groups[key] = []
            instance_groups[key].append(result)

        # Create summary
        print(
            f"{'Instance':<20} | {'Runs':<4} | {'EP Bins':<12} | {'ALNS Bins':<12} | {'Hybrid Bins':<12} | {'Best':<8} | {'ANOVA p-value':<12} | {'Tukey HSD':<20}")
        print("-" * 95)

        for (class_type, n_items), group in instance_groups.items():
            instance_name = f"Class {class_type} ({n_items})"

            ep_bins = [r['ep_bins'] for r in group]
            alns_bins = [r['alns_bins'] for r in group]
            hybrid_bins = [r['hybrid_bins'] for r in group]

            ep_str = f"{np.mean(ep_bins):.1f}±{np.std(ep_bins):.1f}"
            alns_str = f"{np.mean(alns_bins):.1f}±{np.std(alns_bins):.1f}"
            hybrid_str = f"{np.mean(hybrid_bins):.1f}±{np.std(hybrid_bins):.1f}"

            # Determine best
            avg_bins = {
                'EP': np.mean(ep_bins),
                'ALNS': np.mean(alns_bins),
                'Hybrid': np.mean(hybrid_bins)
            }

            best_algo = min(avg_bins.keys(), key=lambda k: avg_bins[k])

            # Perform ANOVA
            if HAS_SCIPY:
                f_stat, p_value = scipy_stats.f_oneway(ep_bins, alns_bins, hybrid_bins)
                anova_result = f"{p_value:.4f}"
            else:
                anova_result = "N/A (no scipy)"

            # Perform Tukey HSD if ANOVA is significant and scipy is available
            tukey_result = "N/A"
            if HAS_SCIPY and p_value < 0.05:
                from statsmodels.stats.multicomp import MultiComparison
                data = np.concatenate([ep_bins, alns_bins, hybrid_bins])
                groups = ['EP'] * len(ep_bins) + ['ALNS'] * len(alns_bins) + ['Hybrid'] * len(hybrid_bins)
                mc = MultiComparison(data, groups)
                tukey_result = mc.tukeyhsd().summary().as_text()

            print(
                f"{instance_name:<20} | {len(group):<4} | {ep_str:<12} | {alns_str:<12} | {hybrid_str:<12} | {best_algo:<8} | "
                f"{anova_result:<12} | {tukey_result[:20]:<20}")


def run_multiple_trials_analysis(n_runs: int = 10):
    """Main function to run the comprehensive analysis"""
    print(f"Starting comprehensive analysis with {n_runs} runs per instance...")

    start_time = time.time()

    # Create analyzer
    analyzer = MultipleRunsAnalyzer(
        n_runs=n_runs,
        max_iterations=2000,
        max_time=60.0
    )

    # Define test instances (you can modify this list)
    test_instances = [
        (1, 30),  # Class 1, 30 items
        (5, 30),  # Class 5, 30 items
        (8, 40),  # Class 8, 40 items

        (1, 60),  # Class 1, 60 items
        (5, 60),  # Class 5, 60 items
        (8, 80),  # Class 8, 80 items

        (1, 90),  # Class 1, 90 items
        (5, 90),  # Class 5, 90 items
        (8, 160),  # Class 8, 160 items

        (1, 500),  # Class 1, 90 items
        (5, 500),  # Class 5, 90 items
        (8, 1000),  # Class 8, 160 items
    ]

    # Run comprehensive comparison
    results = analyzer.run_comprehensive_comparison(test_instances)

    # Save results if pandas available
    if HAS_PANDAS:
        try:
            df = pd.DataFrame(results)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"alns_vnd_comparison_{timestamp}.csv"
            df.to_csv(filename, index=False)
            print(f"\nResults saved to: {filename}")
        except Exception as e:
            print(f"Could not save to CSV: {e}")
    else:
        print("\nPandas not available - results not saved to CSV")

    elapsed_time = time.time() - start_time
    print(f"\nTotal analysis time: {str(timedelta(seconds=elapsed_time))}")

    return results


# Simplified version without pandas dependency
def run_basic_multiple_trials(n_runs: int = 10):
    """Basic multiple trials without pandas dependency"""

    test_instances = [(1, 30), (5, 30), (8, 40)]

    print(f"=== BASIC MULTIPLE TRIALS ANALYSIS ({n_runs} runs) ===")

    for class_type, n_items in test_instances:
        print(f"\nTesting Class {class_type}, {n_items} items:")

        ep_bins_list = []
        alns_bins_list = []
        hybrid_bins_list = []

        for run in range(n_runs):
            print(f"  Run {run + 1}/{n_runs}...", end=" ")

            # Generate instance
            seed = 42 + run * 1000 + class_type * 100 + n_items
            items, bin_template = BenchmarkGenerator.generate_martello_instance(
                class_type, n_items, random_seed=seed
            )

            # EP
            ep_solver = ExtremePointBinPacking3D(bin_template)
            ep_bins = ep_solver.c_epbfd(copy.deepcopy(items))
            ep_bins_list.append(ep_bins)

            # Try ALNS algorithms
            try:
                from AlnsVnd import create_hybrid_alns_vnd

                # Pure ALNS
                pure_alns = create_hybrid_alns_vnd(
                    bin_template, vnd_frequency=0.0, max_time=30.0, random_seed=seed + 1
                )
                alns_solution = pure_alns.solve(copy.deepcopy(items))
                alns_bins_list.append(alns_solution.num_bins)

                # Hybrid
                hybrid_alns = create_hybrid_alns_vnd(
                    bin_template, vnd_frequency=0.3, max_time=30.0, random_seed=seed + 2
                )
                hybrid_solution = hybrid_alns.solve(copy.deepcopy(items))
                hybrid_bins_list.append(hybrid_solution.num_bins)

                print("✓")

            except ImportError:
                alns_bins_list.append(ep_bins)
                hybrid_bins_list.append(ep_bins)
                print("(ALNS unavailable)")

        # Print results
        print(f"  Results:")
        print(f"    EP:     {np.mean(ep_bins_list):.1f} ± {np.std(ep_bins_list):.1f} bins")
        print(f"    ALNS:   {np.mean(alns_bins_list):.1f} ± {np.std(alns_bins_list):.1f} bins")
        print(f"    Hybrid: {np.mean(hybrid_bins_list):.1f} ± {np.std(hybrid_bins_list):.1f} bins")


if __name__ == "__main__":
    # Run basic analysis
    # run_basic_multiple_trials(n_runs=5)
    run_multiple_trials_analysis(n_runs=10)