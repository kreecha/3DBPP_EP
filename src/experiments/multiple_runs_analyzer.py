
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
# Assuming your existing imports
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


class MultipleRunsAnalyzer:
    """Statistical analysis for comparing EP, ALNS, and ALNS-VND algorithms"""

    def __init__(self, n_runs: int = 10, max_iterations: int = 2000, max_time: float = 60.0):
        self.n_runs = n_runs
        self.max_iterations = max_iterations
        self.max_time = max_time
        self.results = []

    def run_comprehensive_comparison(self, test_instances: List[Tuple[int, int]] = None):
        """Run comprehensive comparison across multiple instances and runs"""

        if test_instances is None:
            test_instances = [
                (1, 30),  # Class 1, 30 items
                (5, 30),  # Class 5, 30 items
                (8, 40),  # Class 8, 40 items
                (1, 50),  # Class 1, 50 items (larger)
                (5, 50),  # Class 5, 50 items (larger)
            ]

        print(f"=== COMPREHENSIVE STATISTICAL COMPARISON ===")
        print(f"Running {self.n_runs} trials for each instance...")
        print(f"Test instances: {len(test_instances)}")
        print(f"Total experiments: {len(test_instances) * self.n_runs}")
        print()

        all_results = []

        for instance_idx, (class_type, n_items) in enumerate(test_instances):
        # for instance_idx, (class_type, n_items) in tqdm(enumerate(test_instances), total=len(test_instances), desc="Processing"):
        # for run in tqdm(range(self.n_runs), desc=f"Class {class_type} ({n_items} items)", leave=False):
            print(f"Testing Instance {instance_idx + 1}/{len(test_instances)}: "
                  f"Class {class_type}, {n_items} items")

            instance_results = self._run_instance_trials(class_type, n_items)
            all_results.extend(instance_results)

            # Print instance summary
            self._print_instance_summary(instance_results, class_type, n_items)
            print()

        # Overall analysis
        self._print_overall_analysis(all_results)
        self._create_summary_tables(all_results)

        return all_results

    def _run_instance_trials(self, class_type: int, n_items: int) -> List[Dict]:
        """Run multiple trials for a single instance type"""
        instance_results = []

        for run in tqdm(range(self.n_runs), desc=f"Class {class_type} ({n_items} items)", leave=False):
        # for run in range(self.n_runs):
            # Generate instance with unique seed for each run
            base_seed = 42 + run * 1000 + class_type * 100 + n_items
            items, bin_template = BenchmarkGenerator.generate_martello_instance(
                class_type, n_items, random_seed=base_seed
            )

            # Run all three algorithms
            result = self._run_single_trial(items, bin_template, class_type, n_items, run, base_seed)
            instance_results.append(result)

        return instance_results

    def _run_single_trial(self, items: List[Item], bin_template: Bin,
                          class_type: int, n_items: int, run: int, seed: int) -> Dict:
        """Run a single trial comparing all three algorithms"""

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
                max_iterations=self.max_iterations,
                max_time=self.max_time,
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
            print("    Warning: ALNS not available, skipping...")
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
                max_iterations=self.max_iterations,
                max_time=self.max_time,
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
                'items_packed': items_packed,
                'items_lost': items_lost
            })

        except ImportError:
            print("    Warning: Hybrid ALNS-VND not available, skipping...")
            result.update({
                'hybrid_bins': ep_bins,  # Fallback to EP result
                'hybrid_utilization': ep_util,
                'hybrid_time': 0.0,
                'hybrid_iterations': 0,
                'vnd_applications': 0,
                'vnd_improvements': 0,
                'items_packed': len(items),
                'items_lost': 0
            })

        # Calculate improvements
        result.update({
            'alns_vs_ep_bins': result['ep_bins'] - result['alns_bins'],
            'hybrid_vs_ep_bins': result['ep_bins'] - result['hybrid_bins'],
            'hybrid_vs_alns_bins': result['alns_bins'] - result['hybrid_bins'],
            'alns_vs_ep_util': result['alns_utilization'] - result['ep_utilization'],
            'hybrid_vs_ep_util': result['hybrid_utilization'] - result['ep_utilization'],
            'hybrid_vs_alns_util': result['hybrid_utilization'] - result['alns_utilization']
        })

        return result

    def _print_instance_summary(self, results: List[Dict], class_type: int, n_items: int):
        """Print summary statistics for an instance"""
        print(f"  Results for Class {class_type}, {n_items} items ({len(results)} runs):")

        # Calculate basic statistics
        stats = {}
        for algo in ['ep', 'alns', 'hybrid']:
            bins_values = [r[f'{algo}_bins'] for r in results]
            util_values = [r[f'{algo}_utilization'] for r in results]
            time_values = [r[f'{algo}_time'] for r in results]

            stats[algo] = {
                'bins_mean': np.mean(bins_values),
                'bins_std': np.std(bins_values),
                'bins_min': np.min(bins_values),
                'bins_max': np.max(bins_values),
                'util_mean': np.mean(util_values),
                'util_std': np.std(util_values),
                'time_mean': np.mean(time_values)
            }

        # Print table
        print(f"    Algorithm | Bins (μ±σ)      | Min-Max | Util (μ±σ)     | Time (s)")
        print(f"    ----------|----------------|---------|---------------|--------")

        for algo in ['ep', 'alns', 'hybrid']:
            s = stats[algo]
            algo_name = {'ep': 'EP', 'alns': 'ALNS', 'hybrid': 'Hybrid'}[algo]
            print(f"    {algo_name:<9} | {s['bins_mean']:5.1f}±{s['bins_std']:4.1f}   | "
                  f"{s['bins_min']:2.0f}-{s['bins_max']:2.0f}  | "
                  f"{s['util_mean']:.3f}±{s['util_std']:.3f} | {s['time_mean']:6.1f}")

        # Best algorithm for this instance
        best_bins = min(stats['ep']['bins_mean'], stats['alns']['bins_mean'], stats['hybrid']['bins_mean'])
        best_algo = [k for k, v in stats.items() if abs(v['bins_mean'] - best_bins) < 0.01][0]
        best_name = {'ep': 'ExtremePoint', 'alns': 'Pure ALNS', 'hybrid': 'Hybrid ALNS-VND'}[best_algo]

        print(f"    Best performer: {best_name} ({best_bins:.1f} bins average)")

        # Statistical significance tests if scipy available
        if HAS_SCIPY and len(results) >= 5:
            try:
                hybrid_bins = [r['hybrid_bins'] for r in results]
                alns_bins = [r['alns_bins'] for r in results]
                ep_bins = [r['ep_bins'] for r in results]

                # Wilcoxon signed-rank test for paired comparisons
                if not np.array_equal(hybrid_bins, alns_bins):
                    stat, p_val = scipy_stats.wilcoxon(hybrid_bins, alns_bins, alternative='less')
                    if p_val < 0.05:
                        print(f"    Hybrid significantly better than ALNS (p={p_val:.3f})")

                if not np.array_equal(hybrid_bins, ep_bins):
                    stat, p_val = scipy_stats.wilcoxon(hybrid_bins, ep_bins, alternative='less')
                    if p_val < 0.05:
                        print(f"    Hybrid significantly better than EP (p={p_val:.3f})")

            except Exception:
                pass  # Skip if test cannot be performed

    def _print_overall_analysis(self, all_results: List[Dict]):
        """Print overall analysis across all instances"""
        print("=== OVERALL ANALYSIS ACROSS ALL INSTANCES ===")

        # Count wins by algorithm
        wins = {'ep': 0, 'alns': 0, 'hybrid': 0}

        for result in all_results:
            min_bins = min(result['ep_bins'], result['alns_bins'], result['hybrid_bins'])

            if result['ep_bins'] == min_bins:
                wins['ep'] += 1
            if result['alns_bins'] == min_bins:
                wins['alns'] += 1
            if result['hybrid_bins'] == min_bins:
                wins['hybrid'] += 1

        total_runs = len(all_results)
        print(f"Algorithm Performance Summary ({total_runs} total runs):")
        print(f"  ExtremePoint wins: {wins['ep']} ({100 * wins['ep'] / total_runs:.1f}%)")
        print(f"  Pure ALNS wins: {wins['alns']} ({100 * wins['alns'] / total_runs:.1f}%)")
        print(f"  Hybrid ALNS-VND wins: {wins['hybrid']} ({100 * wins['hybrid'] / total_runs:.1f}%)")

        # Average improvements
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
        """Create detailed summary tables"""
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
            f"{'Instance':<20} | {'Runs':<4} | {'EP Bins':<12} | {'ALNS Bins':<12} | {'Hybrid Bins':<12} | {'Best':<8}")
        print("-" * 85)

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

            print(
                f"{instance_name:<20} | {len(group):<4} | {ep_str:<12} | {alns_str:<12} | {hybrid_str:<12} | {best_algo:<8}")


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