#!/usr/bin/env python3
"""
Script to run all tests from the main NTQR package.

This script verifies that our fork can properly run and pass all tests
from the main package, ensuring compatibility.
"""

import os
import sys
import unittest
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from fork.src.utils.test_helpers import get_all_test_modules, import_and_run_test

def main():
    """
    Run all tests from the main package.
    """
    print("Running all tests from the main NTQR package...")
    
    # Get all test modules
    test_modules = get_all_test_modules()
    
    if not test_modules:
        print("No test modules found in the main package.")
        return 1
    
    print(f"Found {len(test_modules)} test modules.")
    
    # Run each test module
    failed_modules = []
    for module_path in test_modules:
        print(f"\nRunning tests from {module_path}...")
        try:
            # Import the module to check if it's importable
            __import__(module_path)
            
            # Run tests from the module
            import_and_run_test(module_path)
        except (ImportError, ModuleNotFoundError) as e:
            print(f"Error importing module {module_path}: {e}")
            failed_modules.append((module_path, f"Import error: {e}"))
        except Exception as e:
            print(f"Error running tests from module {module_path}: {e}")
            failed_modules.append((module_path, f"Test error: {e}"))
    
    # Print summary
    print("\n" + "="*80)
    print("Test Run Summary")
    print("="*80)
    print(f"Total test modules: {len(test_modules)}")
    print(f"Failed modules: {len(failed_modules)}")
    
    if failed_modules:
        print("\nFailed modules:")
        for module_path, error in failed_modules:
            print(f"  - {module_path}: {error}")
        return 1
    
    print("\nAll tests passed!")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 