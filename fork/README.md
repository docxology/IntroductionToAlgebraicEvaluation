# NTQR Fork - Algebraic Evaluation Tools

This fork directory contains complementary work based on the main NTQR package for algebraic evaluation, while maintaining isolation from the original code. Our work strictly remains within this `fork/` folder.

## Structure

The fork is organized as follows:

- `fork/` - Root of our experimental work
  - `src/` - Custom extensions and implementations 
  - `tests/` - Test cases for our fork's functionality
  - `docs/` - Documentation and analysis
    - `analytical/` - Analytical findings and theoretical work
    - `operational/` - Operational guides and considerations

## Main Package Structure (`../python/`)

The main NTQR package that we're building upon has the following structure:

- **Core Modules**:
  - `ntqr.evaluations` - Core evaluation algorithms
  - `ntqr.alarms` - Logical alarms implementation
  - `ntqr.raxioms` - Axioms for different response models
  - `ntqr.plots` - Visualization tools
  - `ntqr.statistics` - Statistical analysis utilities
  - `ntqr.testsketches` - Test sketch implementations

- **Response Models**:
  - `ntqr.r2` - Binary classifier/responder (2-class) implementations
  - `ntqr.r3` - 3-class classifier implementations

- **Tests**:
  - `tests/r2` - Tests for binary classifiers
  - `tests/r3` - Tests for 3-class classifiers

## Usage

To run tests and functions from the main package within our fork:

```bash
# Run from the project root directory
cd IntroductionToAlgebraicEvaluation
python -m python.tests.r2.test_file

# Or import in your code
from python.src.ntqr import evaluations, alarms
```

## Contributing

All work should be done exclusively within the `fork/` directory. Do not modify files outside this directory unless explicitly instructed to do so.

## License

This work is governed by the same license as the main repository. 