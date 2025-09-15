# -*- coding: utf-8 -*-
"""
Created on Tue Sep 9 08:26:17 2025



@author: Kreecha Puphaiboon

MIT License

Copyright (c) 2025 Kreecha Puphaiboon

"""

import numpy as np
from typing import List, Dict, Set, Tuple
from abc import ABC, abstractmethod
import traceback

# Import shared classes
from src.common import Item, Bin, PlacedItem, ExtremePoint


class ValidationError(Exception):
    """Custom exception for operator validation failures"""

    def __init__(self, message: str, operator_name: str, details: Dict = None):
        super().__init__(message)
        self.operator_name = operator_name
        self.details = details or {}


class ValidatedDestroyOperator(ABC):
    """
    Base class for destroy operators with automatic item conservation validation
    """

    def __init__(self, name: str):
        self.name = name
        self.weight = 1.0
        self.score = 0
        self.usage_count = 0
        self.validation_enabled = True
        self.debug_mode = True

        # Statistics
        self.total_calls = 0
        self.successful_calls = 0
        self.validation_failures = 0

    def destroy(self, solution, rnd_state: np.random.RandomState) -> List[Item]:
        """
        Public interface with validation wrapper
        """
        self.total_calls += 1

        if not self.validation_enabled:
            return self._destroy_implementation(solution, rnd_state)

        # Pre-operation validation
        initial_state = self._capture_solution_state(solution)

        try:
            # Execute the actual destroy operation
            removed_items = self._destroy_implementation(solution, rnd_state)

            # Post-operation validation
            self._validate_destroy_operation(initial_state, solution, removed_items)

            self.successful_calls += 1
            return removed_items

        except ValidationError as e:
            self.validation_failures += 1
            self._handle_validation_error(e, initial_state, solution)
            raise
        except Exception as e:
            self.validation_failures += 1
            error_msg = f"Unexpected error in {self.name}: {str(e)}"
            if self.debug_mode:
                error_msg += f"\nTraceback: {traceback.format_exc()}"
            raise ValidationError(error_msg, self.name)

    @abstractmethod
    def _destroy_implementation(self, solution, rnd_state: np.random.RandomState) -> List[Item]:
        """
        Subclasses implement their specific destroy logic here
        """
        pass

    def _capture_solution_state(self, solution) -> Dict:
        """Capture solution state for validation"""
        return {
            'total_items_before': len(solution.get_all_items()),
            'items_before': set(solution.get_all_items()),
            'bins_before': len(solution.bins),
            'bin_contents_before': [len(bin_items) for bin_items in solution.bins]
        }

    def _validate_destroy_operation(self, initial_state: Dict, solution, removed_items: List[Item]):
        """Validate that destroy operation maintains item conservation"""

        # Current solution state
        current_items = set(solution.get_all_items())
        removed_items_set = set(removed_items)

        # Expected state after removal
        expected_items = initial_state['items_before'] - removed_items_set

        # Validation checks
        errors = []

        # Check 1: No items should disappear beyond those explicitly removed
        if current_items != expected_items:
            missing_items = expected_items - current_items
            extra_items = current_items - expected_items

            if missing_items:
                errors.append(f"Items lost (not in removed list): {len(missing_items)} items")
            if extra_items:
                errors.append(f"Unexpected items appeared: {len(extra_items)} items")

        # Check 2: Removed items should not still be in solution
        still_present = current_items & removed_items_set
        if still_present:
            errors.append(f"Removed items still in solution: {len(still_present)} items")

        # Check 3: Removed items list should not have duplicates
        if len(removed_items) != len(removed_items_set):
            errors.append(f"Duplicate items in removed list: {len(removed_items)} vs {len(removed_items_set)}")

        # Check 4: All removed items should have been in initial solution
        not_originally_present = removed_items_set - initial_state['items_before']
        if not_originally_present:
            errors.append(f"Removed items not originally in solution: {len(not_originally_present)} items")

        # Check 5: Solution structure integrity
        total_in_bins = sum(len(bin_items) for bin_items in solution.bins)
        if total_in_bins != len(current_items):
            errors.append(
                f"Bin structure inconsistency: {total_in_bins} in bins vs {len(current_items)} in get_all_items()")

        if errors:
            details = {
                'initial_items': initial_state['total_items_before'],
                'current_items': len(current_items),
                'removed_items': len(removed_items),
                'expected_remaining': len(expected_items),
                'actual_remaining': len(current_items),
                'errors': errors
            }

            error_msg = f"Destroy validation failed for {self.name}:\n" + "\n".join(f"  - {error}" for error in errors)
            raise ValidationError(error_msg, self.name, details)

    def _handle_validation_error(self, error: ValidationError, initial_state: Dict, solution):
        """Handle validation errors with debugging information"""
        if self.debug_mode:
            print(f"\n{'=' * 60}")
            print(f"VALIDATION ERROR in {self.name}")
            print(f"{'=' * 60}")
            print(f"Error: {error}")
            print(f"Initial items: {initial_state['total_items_before']}")
            print(f"Current items: {len(solution.get_all_items())}")
            print(f"Bins before: {initial_state['bins_before']}")
            print(f"Bins after: {len(solution.bins)}")
            if error.details:
                print("Details:", error.details)
            print(f"{'=' * 60}")

    def get_statistics(self) -> Dict:
        """Get operator statistics"""
        success_rate = (self.successful_calls / self.total_calls * 100) if self.total_calls > 0 else 0
        return {
            'name': self.name,
            'total_calls': self.total_calls,
            'successful_calls': self.successful_calls,
            'validation_failures': self.validation_failures,
            'success_rate': success_rate,
            'weight': self.weight,
            'score': self.score,
            'usage_count': self.usage_count
        }


class ValidatedRepairOperator(ABC):
    """
    Base class for repair operators with automatic item conservation validation
    """

    def __init__(self, name: str):
        self.name = name
        self.weight = 1.0
        self.score = 0
        self.usage_count = 0
        self.validation_enabled = True
        self.debug_mode = True

        # Statistics
        self.total_calls = 0
        self.successful_calls = 0
        self.validation_failures = 0

    def repair(self, solution, items_to_repair: List[Item], bin_template: Bin,
               rnd_state: np.random.RandomState) -> bool:
        """
        Public interface with validation wrapper
        """
        self.total_calls += 1

        if not self.validation_enabled:
            return self._repair_implementation(solution, items_to_repair, bin_template, rnd_state)

        # Pre-operation validation
        initial_state = self._capture_repair_state(solution, items_to_repair)

        try:
            # Execute the actual repair operation
            success = self._repair_implementation(solution, items_to_repair, bin_template, rnd_state)

            # Post-operation validation
            self._validate_repair_operation(initial_state, solution, items_to_repair, success)

            self.successful_calls += 1
            return success

        except ValidationError as e:
            self.validation_failures += 1
            self._handle_validation_error(e, initial_state, solution, items_to_repair)
            raise
        except Exception as e:
            self.validation_failures += 1
            error_msg = f"Unexpected error in {self.name}: {str(e)}"
            if self.debug_mode:
                error_msg += f"\nTraceback: {traceback.format_exc()}"
            raise ValidationError(error_msg, self.name)

    @abstractmethod
    def _repair_implementation(self, solution, items_to_repair: List[Item], bin_template: Bin,
                               rnd_state: np.random.RandomState) -> bool:
        """
        Subclasses implement their specific repair logic here
        """
        pass

    def _capture_repair_state(self, solution, items_to_repair: List[Item]) -> Dict:
        """Capture solution state before repair"""
        return {
            'items_before': set(solution.get_all_items()),
            'items_to_repair': set(items_to_repair),
            'total_items_before': len(solution.get_all_items()),
            'bins_before': len(solution.bins),
            'expected_total_after': len(solution.get_all_items()) + len(items_to_repair)
        }

    def _validate_repair_operation(self, initial_state: Dict, solution, items_to_repair: List[Item], success: bool):
        """Validate that repair operation maintains item conservation"""

        current_items = set(solution.get_all_items())
        items_to_repair_set = set(items_to_repair)

        errors = []

        if success:
            # If repair was successful, all items should be in solution
            expected_items = initial_state['items_before'] | items_to_repair_set

            # Check 1: All items should be present
            if current_items != expected_items:
                missing_items = expected_items - current_items
                extra_items = current_items - expected_items

                if missing_items:
                    errors.append(f"Items missing after repair: {len(missing_items)} items")
                if extra_items:
                    errors.append(f"Unexpected extra items: {len(extra_items)} items")

            # Check 2: Total count should be correct
            expected_total = initial_state['expected_total_after']
            actual_total = len(current_items)
            if actual_total != expected_total:
                errors.append(f"Item count mismatch: expected {expected_total}, got {actual_total}")

        else:
            # If repair failed, original items should still be there, repair items should not be added
            if current_items != initial_state['items_before']:
                errors.append("Solution modified despite repair failure")

        # Check 3: Solution structure integrity
        total_in_bins = sum(len(bin_items) for bin_items in solution.bins)
        if total_in_bins != len(current_items):
            errors.append(
                f"Bin structure inconsistency: {total_in_bins} in bins vs {len(current_items)} in get_all_items()")

        # Check 4: Items to repair should not have duplicates
        if len(items_to_repair) != len(items_to_repair_set):
            errors.append(f"Duplicate items in repair list: {len(items_to_repair)} vs {len(items_to_repair_set)}")

        if errors:
            details = {
                'success': success,
                'initial_items': initial_state['total_items_before'],
                'items_to_repair': len(items_to_repair),
                'expected_total': initial_state['expected_total_after'],
                'actual_total': len(current_items),
                'errors': errors
            }

            error_msg = f"Repair validation failed for {self.name}:\n" + "\n".join(f"  - {error}" for error in errors)
            raise ValidationError(error_msg, self.name, details)

    def _handle_validation_error(self, error: ValidationError, initial_state: Dict, solution,
                                 items_to_repair: List[Item]):
        """Handle validation errors with debugging information"""
        if self.debug_mode:
            print(f"\n{'=' * 60}")
            print(f"VALIDATION ERROR in {self.name}")
            print(f"{'=' * 60}")
            print(f"Error: {error}")
            print(f"Initial items: {initial_state['total_items_before']}")
            print(f"Items to repair: {len(items_to_repair)}")
            print(f"Expected total: {initial_state['expected_total_after']}")
            print(f"Actual total: {len(solution.get_all_items())}")
            if error.details:
                print("Details:", error.details)
            print(f"{'=' * 60}")

    def get_statistics(self) -> Dict:
        """Get operator statistics"""
        success_rate = (self.successful_calls / self.total_calls * 100) if self.total_calls > 0 else 0
        return {
            'name': self.name,
            'total_calls': self.total_calls,
            'successful_calls': self.successful_calls,
            'validation_failures': self.validation_failures,
            'success_rate': success_rate,
            'weight': self.weight,
            'score': self.score,
            'usage_count': self.usage_count
        }


class OperatorValidationManager:
    """
    Manager class for operator validation settings and statistics
    """

    def __init__(self):
        self.destroy_operators: List[ValidatedDestroyOperator] = []
        self.repair_operators: List[ValidatedRepairOperator] = []
        self.global_validation_enabled = True
        self.global_debug_mode = True

    def register_operators(self, destroy_ops: List[ValidatedDestroyOperator],
                           repair_ops: List[ValidatedRepairOperator]):
        """Register operators for validation management"""
        self.destroy_operators = destroy_ops
        self.repair_operators = repair_ops

        # Apply global settings
        self._apply_global_settings()

    def enable_validation(self, enabled: bool = True):
        """Enable/disable validation for all operators"""
        self.global_validation_enabled = enabled
        self._apply_global_settings()

    def enable_debug_mode(self, enabled: bool = True):
        """Enable/disable debug mode for all operators"""
        self.global_debug_mode = enabled
        self._apply_global_settings()

    def _apply_global_settings(self):
        """Apply global settings to all operators"""
        for op in self.destroy_operators + self.repair_operators:
            op.validation_enabled = self.global_validation_enabled
            op.debug_mode = self.global_debug_mode

    def get_validation_report(self) -> Dict:
        """Generate comprehensive validation report"""
        report = {
            'destroy_operators': [op.get_statistics() for op in self.destroy_operators],
            'repair_operators': [op.get_statistics() for op in self.repair_operators],
            'summary': {
                'total_operators': len(self.destroy_operators) + len(self.repair_operators),
                'validation_enabled': self.global_validation_enabled,
                'debug_mode': self.global_debug_mode
            }
        }

        # Calculate summary statistics
        all_ops = self.destroy_operators + self.repair_operators
        total_calls = sum(op.total_calls for op in all_ops)
        total_failures = sum(op.validation_failures for op in all_ops)

        if total_calls > 0:
            report['summary']['overall_failure_rate'] = (total_failures / total_calls * 100)
            report['summary']['total_calls'] = total_calls
            report['summary']['total_failures'] = total_failures

        return report

    def print_validation_report(self):
        """Print formatted validation report"""
        report = self.get_validation_report()

        print("\n" + "=" * 80)
        print("OPERATOR VALIDATION REPORT")
        print("=" * 80)

        print(f"Validation Enabled: {report['summary']['validation_enabled']}")
        print(f"Debug Mode: {report['summary']['debug_mode']}")
        print(f"Total Operators: {report['summary']['total_operators']}")

        if 'total_calls' in report['summary']:
            print(f"Total Calls: {report['summary']['total_calls']}")
            print(f"Total Failures: {report['summary']['total_failures']}")
            print(f"Overall Failure Rate: {report['summary']['overall_failure_rate']:.2f}%")

        print("\nDESTROY OPERATORS:")
        print("-" * 50)
        for op_stats in report['destroy_operators']:
            print(f"{op_stats['name']:<25} | Calls: {op_stats['total_calls']:>4} | "
                  f"Failures: {op_stats['validation_failures']:>2} | "
                  f"Success: {op_stats['success_rate']:>6.1f}%")

        print("\nREPAIR OPERATORS:")
        print("-" * 50)
        for op_stats in report['repair_operators']:
            print(f"{op_stats['name']:<25} | Calls: {op_stats['total_calls']:>4} | "
                  f"Failures: {op_stats['validation_failures']:>2} | "
                  f"Success: {op_stats['success_rate']:>6.1f}%")

        print("=" * 80)

if __name__ == "__main__":
    print("Validated ALNS Operators Base Classes")
    print("=" * 50)
    print("Features:")
    print("- Automatic item conservation validation")
    print("- Comprehensive error reporting")
    print("- Statistical tracking")
    print("- Debug mode support")
    print("- Fail-fast error detection")
    print("\nTo use: Inherit from ValidatedDestroyOperator or ValidatedRepairOperator")
    print("and implement _destroy_implementation() or _repair_implementation()")