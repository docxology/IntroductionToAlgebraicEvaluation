"""
Test helper utilities for running and adapting tests from the main NTQR package.
"""

import importlib
import inspect
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, Union

import pytest

from fork.src.utils.compatibility import ensure_main_package_importable


def get_test_classes_from_module(module_path: str) -> List[Type]:
    """
    Get all test classes from a module in the main package.
    
    Parameters
    ----------
    module_path : str
        The import path of the module (e.g., 'python.tests.r2.test_evaluators')
        
    Returns
    -------
    List[Type]
        List of test classes found in the module
    """
    ensure_main_package_importable()
    
    try:
        module = importlib.import_module(module_path)
        
        # Get all classes that start with "Test"
        test_classes = []
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and name.startswith("Test"):
                test_classes.append(obj)
                
        return test_classes
    
    except ImportError:
        pytest.skip(f"Could not import module {module_path} from main package")
        return []


def adapt_test_class(test_class: Type, **modifications: Dict[str, Any]) -> Type:
    """
    Create an adapted version of a test class from the main package.
    
    Parameters
    ----------
    test_class : Type
        The original test class to adapt
    modifications : Dict[str, Any]
        Dictionary of attribute names and their new values to modify in the class
        
    Returns
    -------
    Type
        A new test class based on the original with the specified modifications
    """
    # Create a new class that inherits from the original
    adapted_class_name = f"Adapted{test_class.__name__}"
    adapted_class = type(adapted_class_name, (test_class,), {})
    
    # Apply modifications
    for attr_name, new_value in modifications.items():
        setattr(adapted_class, attr_name, new_value)
    
    return adapted_class


def import_and_run_test(test_module_path: str, test_name: Optional[str] = None) -> None:
    """
    Import a test module from the main package and run specified tests.
    
    Parameters
    ----------
    test_module_path : str
        The import path of the test module (e.g., 'python.tests.r2.test_evaluators')
    test_name : Optional[str]
        Specific test name to run, or None to run all tests in the module
    """
    ensure_main_package_importable()
    
    # Construct the pytest arguments
    args = ["-v", test_module_path]
    if test_name:
        args.append(f"{test_module_path}::{test_name}")
    
    # Run the tests
    pytest.main(args)


def get_all_test_modules() -> List[str]:
    """
    Get a list of all test modules in the main package.
    
    Returns
    -------
    List[str]
        List of import paths for all test modules
    """
    ensure_main_package_importable()
    
    # Base path for the main package tests
    base_path = Path(__file__).parent.parent.parent.parent / "python" / "tests"
    
    # Find all Python files that start with "test_"
    test_modules = []
    for root, _, files in os.walk(base_path):
        for file in files:
            if file.startswith("test_") and file.endswith(".py"):
                # Convert file path to module path
                rel_path = os.path.relpath(os.path.join(root, file), base_path.parent)
                module_path = rel_path.replace(os.sep, ".").replace(".py", "")
                test_modules.append(module_path)
    
    return test_modules 