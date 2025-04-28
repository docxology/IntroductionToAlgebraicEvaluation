"""
NTQR Fork - Extensions to the NTQR algebraic evaluation package.

This fork extends the NTQR package with additional functionality while
maintaining strict isolation from the original codebase.
"""

__version__ = "0.1.0"
__main_package_version__ = "0.5.1"  # Version of NTQR this fork is compatible with

# Import version detection utility
import importlib.metadata
import importlib.util
import sys
from pathlib import Path

# Check if the main package is available
try:
    if importlib.util.find_spec("ntqr") is not None:
        # Installed as a package
        main_version = importlib.metadata.version("ntqr")
    else:
        # Using local version
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from python.src.ntqr import __version__ as main_version
    
    # Check version compatibility
    if main_version != __main_package_version__:
        print(f"Warning: This fork was developed against NTQR version {__main_package_version__}, "
              f"but version {main_version} is installed.")
except (ImportError, ModuleNotFoundError):
    print("Warning: Could not detect NTQR package version. Some functionality may not work as expected.")
    main_version = None

# Feature detection
try:
    from python.src.ntqr.alarms import LabelsSafetySpecification
    HAS_SAFETY_SPEC = True
except ImportError:
    HAS_SAFETY_SPEC = False

try:
    from python.src.ntqr.r3 import evaluators as r3_evaluators
    HAS_R3_SUPPORT = True
except ImportError:
    HAS_R3_SUPPORT = False

# Package exports
from . import core
from . import extensions
from . import utils 