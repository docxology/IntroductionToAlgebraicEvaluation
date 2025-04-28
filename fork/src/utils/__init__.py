"""
Utility functions for the NTQR fork.

This module contains utility functions, compatibility helpers, and other support
code for working with the NTQR package.
"""

from .compatibility import (
    ensure_compatible,
    ensure_main_package_importable,
    get_main_package_version,
    check_compatibility,
)

from .test_helpers import (
    get_test_classes_from_module,
    adapt_test_class,
    import_and_run_test,
    get_all_test_modules,
)

# Utils module exports will go here as they are implemented 