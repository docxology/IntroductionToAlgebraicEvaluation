"""
Compatibility utilities for working with the main NTQR package.
"""

import importlib
import sys
import warnings
from pathlib import Path
from typing import Any, Callable, Type, TypeVar

# Type variable for class types
T = TypeVar('T')

def ensure_main_package_importable():
    """
    Ensure the main NTQR package is importable by adding parent directory to sys.path if needed.
    """
    parent_dir = Path(__file__).parent.parent.parent.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    
    try:
        import python.src.ntqr
        return True
    except ImportError:
        warnings.warn("Could not import main NTQR package. Some functionality may not work correctly.")
        return False

def ensure_compatible(cls_or_func: T) -> T:
    """
    Ensures a class or function from the main package is compatible with our fork.
    
    Parameters
    ----------
    cls_or_func : Type or Callable
        The class or function to check for compatibility
        
    Returns
    -------
    Type or Callable
        The original class or function, or a compatible wrapper if needed
    """
    # Currently just returns the original class or function
    # In future versions, this could add compatibility wrappers based on version checks
    
    ensure_main_package_importable()
    return cls_or_func

def get_main_package_version() -> str:
    """
    Get the version of the main NTQR package.
    
    Returns
    -------
    str
        The version string, or "unknown" if not found
    """
    try:
        from python.src.ntqr import __version__
        return __version__
    except ImportError:
        return "unknown"

def check_compatibility() -> bool:
    """
    Check if the main package version is compatible with our fork.
    
    Returns
    -------
    bool
        True if compatible, False otherwise
    """
    from fork.src import __main_package_version__
    
    main_version = get_main_package_version()
    
    if main_version == "unknown":
        warnings.warn("Could not determine main package version. Assuming compatibility.")
        return True
    
    if main_version != __main_package_version__:
        warnings.warn(
            f"Fork was developed against NTQR version {__main_package_version__}, "
            f"but version {main_version} is detected. Some features may not work correctly."
        )
        return False
    
    return True 