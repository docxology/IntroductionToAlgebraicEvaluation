# Quickstart Guide

This guide provides a quick introduction to using our fork of the NTQR package, with practical examples to get you started.

## Basic Usage

### 1. Import Modules

To begin using the NTQR fork, import the necessary modules:

```python
# Import core functionality
from fork.src.core import evaluators, axioms

# Import R2 (binary classification) specific modules
from fork.src.core.r2 import r2_evaluators, r2_axioms

# Import utilities
from fork.src.utils import data_processing, visualization
```

### 2. Load or Generate Data

You can either load existing data or generate synthetic data for testing:

```python
# Example 1: Load data from CSV
import pandas as pd
import numpy as np

# Load classifier responses (0s and 1s)
responses_df = pd.read_csv('path/to/classifier_responses.csv')

# Convert to numpy array format expected by NTQR
# Format: counts of agreement patterns between classifiers
agreement_counts = data_processing.convert_responses_to_agreement_counts(responses_df)

# Example 2: Generate synthetic data
synthetic_data = data_processing.generate_synthetic_responses(
    n_samples=1000,       # Number of examples 
    n_classifiers=3,      # Number of classifiers
    accuracy_range=(0.6, 0.9),  # Range of accuracies
    error_correlation=0.2  # Correlation between errors
)

# Convert to agreement counts
synthetic_counts = data_processing.convert_responses_to_agreement_counts(synthetic_data)
```

### 3. Evaluate Classifiers

Use the NTQR evaluation methods to evaluate classifiers:

```python
# Create an evaluator for binary classification with three classifiers
evaluator = r2_evaluators.EnhancedTrioEvaluation()

# Evaluate the classifiers
results = evaluator.evaluate(synthetic_counts)

# Print the results
print(f"Classifier 1 accuracy: {results.accuracies[0]:.4f}")
print(f"Classifier 2 accuracy: {results.accuracies[1]:.4f}")
print(f"Classifier 3 accuracy: {results.accuracies[2]:.4f}")
print(f"Error correlation: {results.error_correlation:.4f}")
```

### 4. Visualize Results

Visualize evaluation results using the visualization utilities:

```python
# Create a plot showing the evaluation space for binary classifiers
visualization.plot_evaluation_space(
    results,
    title="Evaluation Results for Binary Classifiers",
    show_constraints=True
)

# Plot error correlation matrix
visualization.plot_error_correlation_matrix(results.error_correlation_matrix)
```

## Common Use Cases

### Evaluating a New Classifier Against Reference Classifiers

```python
# Assuming we have responses from two reference classifiers and one new classifier
reference_1 = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])  # First reference classifier
reference_2 = np.array([0, 1, 0, 1, 0, 0, 1, 1, 0, 1])  # Second reference classifier
new_clf = np.array([0, 1, 1, 1, 0, 0, 0, 1, 0, 0])      # New classifier

# Combine responses
responses = np.column_stack([reference_1, reference_2, new_clf])

# Convert to agreement counts
agreement_counts = data_processing.convert_responses_to_agreement_counts(responses)

# Evaluate
evaluator = r2_evaluators.EnhancedTrioEvaluation()
results = evaluator.evaluate(agreement_counts)

# Check if the new classifier performs well
if results.accuracies[2] > 0.7:
    print("The new classifier performs well!")
else:
    print("The new classifier needs improvement.")
```

### Safety Checking with Alarms

```python
# Import alarm module
from fork.src.core import alarms

# Create a safety specification
safety_spec = alarms.EnhancedSafetySpecification(
    min_accuracy=0.7,
    max_error_correlation=0.3
)

# Check if evaluation meets safety requirements
is_safe = safety_spec.check(results)

if is_safe:
    print("Evaluation results meet safety requirements.")
else:
    print("Evaluation results do not meet safety requirements!")
    print(safety_spec.get_violations(results))
```

### Extending to Ternary Classification (R=3)

```python
# Import R3 modules
from fork.src.core.r3 import r3_evaluators, r3_axioms

# Generate synthetic ternary data (0, 1, 2 responses)
ternary_data = data_processing.generate_synthetic_responses(
    n_samples=1000,
    n_classifiers=3,
    n_classes=3,  # R=3 for ternary classification
    accuracy_range=(0.6, 0.9)
)

# Convert to agreement counts
ternary_counts = data_processing.convert_responses_to_agreement_counts(
    ternary_data, 
    n_classes=3
)

# Create an evaluator for ternary classification
r3_evaluator = r3_evaluators.TernaryTrioEvaluation()

# Evaluate
r3_results = r3_evaluator.evaluate(ternary_counts)

# Visualize results
visualization.plot_r3_evaluation_space(r3_results)
```

## Next Steps

Now that you're familiar with the basics, you can:

- Explore the [Implementation Guide](implementation_guide.md) for advanced usage
- Check out the [Example Gallery](examples.md) for more detailed examples
- Read [Core Concepts](../analytical/core_concepts.md) to understand the theory
- Learn about [Integration with the Main Package](integration_with_main_package.md)

## Troubleshooting

If you encounter issues:

- Check that your data is formatted correctly
- Ensure you're using the appropriate R-class evaluators (R2 for binary, R3 for ternary)
- Look at our [Testing Framework](testing_framework.md) for debugging assistance
- Use data validation utilities: `data_processing.validate_input_data(data)` 