# Installation Guide

This guide will help you install and set up the NTQR fork for development or usage.

## Prerequisites

Before installing, ensure you have the following:

- Python 3.8 or higher
- Git
- pip (Python package manager)

## Installation Options

### Option 1: Development Installation

For active development on the fork, use the development installation approach:

```bash
# Clone the repository
git clone https://github.com/yourusername/IntroductionToAlgebraicEvaluation.git
cd IntroductionToAlgebraicEvaluation

# Install main NTQR package in development mode
pip install -e python/

# Install fork dependencies
pip install -r fork/requirements.txt

# Install development tools
pip install pytest pytest-cov black isort mypy
```

This installation method creates an editable install, which means changes to the source code will be immediately reflected without needing to reinstall.

### Option 2: User Installation

For using the fork without development:

```bash
# Clone the repository
git clone https://github.com/yourusername/IntroductionToAlgebraicEvaluation.git
cd IntroductionToAlgebraicEvaluation

# Install main NTQR package
pip install ./python/

# Install fork package
pip install ./fork/
```

## Verifying Installation

Verify your installation by running:

```bash
# Run tests
cd IntroductionToAlgebraicEvaluation
pytest fork/tests/

# Run a simple example
python -c "from fork.src.core import version; print(f'NTQR Fork installed successfully. Version: {version.__version__}')"
```

## Dependencies

The fork has the following primary dependencies:

- numpy (≥1.20.0)
- scipy (≥1.7.0)
- sympy (≥1.8)
- matplotlib (≥3.4.0)
- pandas (≥1.3.0)

Additional development dependencies:

- pytest (≥6.2.5)
- pytest-cov (≥2.12.0)
- black (≥21.5b2)
- isort (≥5.9.0)
- mypy (≥0.812)

## Troubleshooting

### Common Installation Issues

1. **Python Version Compatibility**

   If you encounter errors related to Python version compatibility, ensure you're using Python 3.8 or higher:
   
   ```bash
   python --version
   ```

2. **Dependency Conflicts**

   If you encounter dependency conflicts, consider using a virtual environment:
   
   ```bash
   # Create a virtual environment
   python -m venv ntqr_env
   
   # Activate the virtual environment
   # On Windows:
   ntqr_env\Scripts\activate
   # On Unix or MacOS:
   source ntqr_env/bin/activate
   
   # Then proceed with installation
   ```

3. **Import Errors After Installation**

   If you encounter import errors after installation, ensure your PYTHONPATH includes both the main package and the fork:
   
   ```bash
   # From the repository root
   export PYTHONPATH=$PYTHONPATH:$(pwd)
   ```

## Getting Help

If you encounter issues not covered here, please:

1. Check the existing issues in the repository
2. Consult the documentation for the main NTQR package
3. Open a new issue with detailed information about your problem

## Next Steps

After installation:

- See the [Quickstart Guide](operational/quickstart_guide.md) for basic usage examples
- Review the [Core Concepts](analytical/core_concepts.md) to understand the theoretical foundations
- Check the [Implementation Guide](operational/implementation_guide.md) for development guidance 