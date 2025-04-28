# Visualization Utilities

This directory contains utility modules for the NTQR fork, including advanced visualization tools.

## Visualization Module

The `visualization.py` module provides enhanced visualization capabilities for algebraic evaluation results. These visualizations help illustrate key concepts and make the evaluation results more intuitive to understand.

### Key Features

- **Enhanced Evaluation Space**: Improved visualization of the evaluation space with constraint intersections and confidence regions
- **Error Correlation Matrix**: Heatmap visualization of error correlations between classifiers
- **Agreement Patterns**: Bar chart visualization of classifier agreement patterns
- **Accuracy Comparison**: Comparative visualization of true vs. estimated accuracies
- **3D Evaluation Space**: Three-dimensional visualization for ternary (R=3) classification

## Usage Examples

### Basic Usage

```python
from fork.src.utils.visualization import enhanced_evaluation_space

# Plot the evaluation space for a result
fig = enhanced_evaluation_space(
    evaluation_result,
    title="My Evaluation Space",
    highlight_constraints=True,
    show_confidence=True
)

# Save the figure
fig.savefig("evaluation_space.png", dpi=300)
```

### Error Correlation Matrix

```python
from fork.src.utils.visualization import plot_error_correlation_matrix

# Plot error correlation matrix
fig = plot_error_correlation_matrix(
    error_correlation_matrix,
    classifier_names=["Classifier A", "Classifier B", "Classifier C"],
    title="Error Correlation Matrix"
)
```

### Agreement Patterns

```python
from fork.src.utils.visualization import plot_agreement_patterns

# Plot agreement patterns
fig = plot_agreement_patterns(
    agreement_counts,
    pattern_labels=["000", "001", "010", "011", "100", "101", "110", "111"],
    title="Agreement Patterns"
)
```

### Accuracy Comparison

```python
from fork.src.utils.visualization import plot_accuracy_comparison

# Plot accuracy comparison
fig = plot_accuracy_comparison(
    classifier_names=["Random Forest", "Logistic Regression", "Neural Network"],
    true_accuracies=[0.85, 0.80, 0.78],
    estimated_accuracies=[0.83, 0.79, 0.80],
    title="True vs. Estimated Accuracies"
)
```

### 3D Evaluation Space

```python
from fork.src.utils.visualization import plot_3d_evaluation_space

# Plot 3D evaluation space for ternary classification
fig = plot_3d_evaluation_space(
    evaluation_result,
    title="3D Evaluation Space",
    show_constraints=True
)
```

## Demo Script

A comprehensive demonstration of all visualization capabilities is available in the examples directory:

```
fork/src/examples/enhanced_visualization_demo.py
```

Run this script to generate example visualizations and save them to the output directory.

## Other Utilities

- `test_helpers.py`: Helper functions for testing
- `compatibility.py`: Compatibility utilities for integration with the main NTQR package 