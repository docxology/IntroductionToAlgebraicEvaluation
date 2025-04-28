# Example Gallery

This document provides a collection of detailed examples for using our NTQR fork in various scenarios. Each example includes complete code that you can adapt for your specific needs.

## Table of Contents

1. [Binary Classification (R=2) Examples](#binary-classification-r2-examples)
   - [Basic Trio Evaluation](#basic-trio-evaluation)
   - [Error Correlation Analysis](#error-correlation-analysis)
   - [Ensemble Optimization](#ensemble-optimization)
   
2. [Ternary Classification (R=3) Examples](#ternary-classification-r3-examples)
   - [Basic Ternary Evaluation](#basic-ternary-evaluation)
   - [Multi-Class Error Analysis](#multi-class-error-analysis)
   
3. [Advanced Use Cases](#advanced-use-cases)
   - [Integration with Machine Learning Pipelines](#integration-with-machine-learning-pipelines)
   - [Scaling to Large Datasets](#scaling-to-large-datasets)
   - [Custom Axiom Development](#custom-axiom-development)

## Binary Classification (R=2) Examples

### Basic Trio Evaluation

This example demonstrates how to evaluate three binary classifiers without ground truth:

```python
import numpy as np
from fork.src.core.r2 import r2_evaluators
from fork.src.utils import data_processing, visualization

# Create synthetic data for three classifiers
# Each classifier produces binary labels (0 or 1) for 1000 examples
np.random.seed(42)
true_labels = np.random.randint(0, 2, size=1000)

# Simulate classifier responses with different accuracy levels
clf1_acc = 0.8  # 80% accuracy
clf2_acc = 0.75  # 75% accuracy
clf3_acc = 0.7   # 70% accuracy

# Generate classifier responses based on true labels and target accuracies
def generate_responses(true_labels, accuracy):
    responses = true_labels.copy()
    flip_indices = np.random.choice(
        np.arange(len(true_labels)),
        size=int(len(true_labels) * (1 - accuracy)),
        replace=False
    )
    responses[flip_indices] = 1 - responses[flip_indices]
    return responses

clf1_responses = generate_responses(true_labels, clf1_acc)
clf2_responses = generate_responses(true_labels, clf2_acc)
clf3_responses = generate_responses(true_labels, clf3_acc)

# Combine all responses
all_responses = np.column_stack([clf1_responses, clf2_responses, clf3_responses])

# Convert to agreement counts (format needed for evaluation)
agreement_counts = data_processing.convert_responses_to_agreement_counts(all_responses)

# Print agreement patterns
print("Agreement patterns:")
for pattern, count in zip(data_processing.get_agreement_patterns(2, 3), agreement_counts.flatten()):
    print(f"Pattern {pattern}: {count} occurrences")

# Create an evaluator for trio evaluation (three classifiers)
evaluator = r2_evaluators.EnhancedTrioEvaluation()

# Evaluate the classifiers
results = evaluator.evaluate(agreement_counts)

# Print results
print("\nEvaluation results:")
print(f"Classifier 1 accuracy: {results.accuracies[0]:.4f} (true: {clf1_acc:.4f})")
print(f"Classifier 2 accuracy: {results.accuracies[1]:.4f} (true: {clf2_acc:.4f})")
print(f"Classifier 3 accuracy: {results.accuracies[2]:.4f} (true: {clf3_acc:.4f})")

# Visualize the results
visualization.plot_evaluation_space(
    results,
    title="Trio Evaluation Results",
    show_constraints=True,
    save_path="trio_evaluation_results.png"
)
```

### Error Correlation Analysis

This example demonstrates how to analyze error correlations between classifiers:

```python
import numpy as np
import matplotlib.pyplot as plt
from fork.src.core.r2 import r2_evaluators
from fork.src.utils import data_processing, visualization

# Generate synthetic data with different error correlation patterns
def generate_correlated_responses(n_samples, accuracies, error_correlation):
    """Generate classifier responses with controlled error correlation."""
    n_classifiers = len(accuracies)
    
    # Generate true labels
    true_labels = np.random.randint(0, 2, size=n_samples)
    
    # Initialize responses
    responses = np.zeros((n_samples, n_classifiers), dtype=int)
    
    # Generate errors with correlation
    errors = np.random.normal(0, 1, (n_samples, 1))  # Common error component
    for i in range(n_classifiers):
        # Individual error component
        indiv_error = np.random.normal(0, 1, (n_samples, 1))
        
        # Combine common and individual error components
        combined_error = error_correlation * errors + np.sqrt(1 - error_correlation**2) * indiv_error
        
        # Determine which examples to flip based on accuracy
        error_threshold = np.percentile(combined_error, accuracies[i] * 100)
        flip_mask = combined_error > error_threshold
        
        # Generate responses
        responses[:, i] = true_labels.copy()
        responses[flip_mask.flatten(), i] = 1 - true_labels[flip_mask.flatten()]
    
    return responses

# Generate datasets with different error correlations
n_samples = 2000
accuracies = [0.8, 0.75, 0.7]  # Same accuracies for all datasets
correlations = [0.0, 0.3, 0.6, 0.9]  # Different error correlations
datasets = []
true_correlations = []

for corr in correlations:
    responses = generate_correlated_responses(n_samples, accuracies, corr)
    datasets.append(responses)
    
    # Calculate true error correlation
    errors = np.zeros((n_samples, len(accuracies)), dtype=int)
    for i in range(len(accuracies)):
        # For each classifier, determine where it made errors
        true_labels = np.argmax(np.bincount(responses[:, i]))  # Majority vote as proxy for truth
        errors[:, i] = (responses[:, i] != true_labels).astype(int)
    
    # Calculate pairwise error correlations
    true_corr_matrix = np.corrcoef(errors.T)
    true_correlations.append(np.mean([true_corr_matrix[0, 1], true_corr_matrix[0, 2], true_corr_matrix[1, 2]]))

# Evaluate each dataset
fig, axes = plt.subplots(1, len(correlations), figsize=(20, 5))
estimated_correlations = []

for i, (responses, corr) in enumerate(zip(datasets, correlations)):
    # Convert to agreement counts
    agreement_counts = data_processing.convert_responses_to_agreement_counts(responses)
    
    # Evaluate
    evaluator = r2_evaluators.EnhancedTrioEvaluation()
    results = evaluator.evaluate(agreement_counts)
    
    # Store estimated correlation
    if hasattr(results, 'error_correlation'):
        estimated_correlations.append(results.error_correlation)
    else:
        # If the evaluator doesn't provide error correlation directly,
        # calculate from the error correlation matrix
        est_corr = np.mean([
            results.error_correlation_matrix[0, 1],
            results.error_correlation_matrix[0, 2], 
            results.error_correlation_matrix[1, 2]
        ])
        estimated_correlations.append(est_corr)
    
    # Plot results
    ax = axes[i]
    visualization.plot_error_correlation_matrix(
        results.error_correlation_matrix,
        ax=ax,
        title=f"Target Correlation: {corr:.1f}"
    )

# Plot comparison of true vs. estimated correlations
plt.figure(figsize=(8, 6))
plt.plot(correlations, true_correlations, 'o-', label='True Correlation')
plt.plot(correlations, estimated_correlations, 's-', label='Estimated Correlation')
plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
plt.xlabel('Target Error Correlation')
plt.ylabel('Measured Error Correlation')
plt.title('NTQR Correlation Estimation Performance')
plt.legend()
plt.grid(True)
plt.savefig('correlation_estimation_performance.png')
plt.show()

print("\nComparison of target vs. estimated error correlations:")
for target, true, est in zip(correlations, true_correlations, estimated_correlations):
    print(f"Target: {target:.2f}, True: {true:.4f}, Estimated: {est:.4f}")
```

### Ensemble Optimization

This example shows how to use evaluation results to optimize an ensemble classifier:

```python
import numpy as np
from fork.src.core.r2 import r2_evaluators
from fork.src.utils import data_processing, ensemble_optimization

# Generate synthetic data
np.random.seed(42)
n_samples = 1000
true_labels = np.random.randint(0, 2, size=n_samples)

# Generate responses for 5 classifiers with different accuracies
n_classifiers = 5
accuracies = [0.85, 0.8, 0.75, 0.7, 0.65]
responses = np.zeros((n_samples, n_classifiers), dtype=int)

for i in range(n_classifiers):
    responses[:, i] = true_labels.copy()
    flip_indices = np.random.choice(
        np.arange(n_samples), 
        size=int(n_samples * (1 - accuracies[i])),
        replace=False
    )
    responses[flip_indices, i] = 1 - responses[flip_indices, i]

# Split data into evaluation and test sets
eval_size = 600
eval_responses = responses[:eval_size, :]
test_responses = responses[eval_size:, :]
test_true_labels = true_labels[eval_size:]

# Evaluate all possible trios of classifiers
trios = [(i, j, k) for i in range(n_classifiers) for j in range(i+1, n_classifiers) 
         for k in range(j+1, n_classifiers)]

trio_results = []
for trio in trios:
    # Extract responses for this trio
    trio_responses = eval_responses[:, trio]
    
    # Convert to agreement counts
    agreement_counts = data_processing.convert_responses_to_agreement_counts(trio_responses)
    
    # Evaluate
    evaluator = r2_evaluators.EnhancedTrioEvaluation()
    results = evaluator.evaluate(agreement_counts)
    
    # Store results
    trio_results.append((trio, results))

# Create different ensemble strategies
ensemble_strategies = {
    "majority_vote": ensemble_optimization.majority_vote_ensemble,
    "weighted_vote": ensemble_optimization.weighted_vote_ensemble,
    "optimal_subset": ensemble_optimization.optimal_subset_ensemble
}

# Evaluate each strategy on test data
strategy_performance = {}

for name, strategy_fn in ensemble_strategies.items():
    # Create ensemble using this strategy
    ensemble_fn = strategy_fn(trio_results, responses.shape[1])
    
    # Apply ensemble to test data
    ensemble_predictions = ensemble_fn(test_responses)
    
    # Calculate accuracy
    accuracy = np.mean(ensemble_predictions == test_true_labels)
    strategy_performance[name] = accuracy

# Print results
print("Ensemble Strategy Performance:")
for name, accuracy in strategy_performance.items():
    print(f"{name}: {accuracy:.4f}")

# Compare to individual classifier performance
print("\nIndividual Classifier Performance:")
for i in range(n_classifiers):
    accuracy = np.mean(test_responses[:, i] == test_true_labels)
    print(f"Classifier {i+1}: {accuracy:.4f} (true: {accuracies[i]:.4f})")

# Plot comparison
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.bar(
    np.arange(len(strategy_performance)), 
    list(strategy_performance.values()),
    tick_label=list(strategy_performance.keys())
)
plt.axhline(y=max(accuracies), color='r', linestyle='--', 
            label=f'Best Individual Classifier ({max(accuracies):.4f})')
plt.ylabel('Accuracy')
plt.title('Ensemble Strategy Performance Comparison')
plt.legend()
plt.grid(True, axis='y')
plt.savefig('ensemble_comparison.png')
plt.show()
```

## Ternary Classification (R=3) Examples

### Basic Ternary Evaluation

This example demonstrates evaluation for ternary classifiers (R=3):

```python
import numpy as np
from fork.src.core.r3 import r3_evaluators
from fork.src.utils import data_processing, visualization

# Generate synthetic ternary data
np.random.seed(42)
n_samples = 1000
n_classes = 3  # Ternary classification (R=3)
n_classifiers = 3

# Generate true labels with balanced classes
true_labels = np.random.randint(0, n_classes, size=n_samples)

# Generate classifier responses with different accuracies
accuracies = [0.75, 0.7, 0.65]
responses = np.zeros((n_samples, n_classifiers), dtype=int)

for i in range(n_classifiers):
    responses[:, i] = true_labels.copy()
    
    # Select examples to change
    n_to_change = int(n_samples * (1 - accuracies[i]))
    change_indices = np.random.choice(np.arange(n_samples), size=n_to_change, replace=False)
    
    # For each example to change, select a different class
    for idx in change_indices:
        current_label = responses[idx, i]
        new_label = np.random.choice([c for c in range(n_classes) if c != current_label])
        responses[idx, i] = new_label

# Convert to agreement counts for R=3
agreement_counts = data_processing.convert_responses_to_agreement_counts(
    responses, n_classes=n_classes
)

# Create a ternary evaluator
evaluator = r3_evaluators.TernaryTrioEvaluation()

# Evaluate the classifiers
results = evaluator.evaluate(agreement_counts)

# Print results
print("Ternary Classification Results:")
for i in range(n_classifiers):
    print(f"Classifier {i+1}:")
    for c in range(n_classes):
        print(f"  Class {c} accuracy: {results.class_accuracies[i][c]:.4f}")
    print(f"  Overall accuracy: {results.accuracies[i]:.4f} (true: {accuracies[i]:.4f})")

# Visualize results
visualization.plot_r3_evaluation_space(
    results,
    title="Ternary Classification Evaluation",
    save_path="ternary_evaluation.png"
)
```

## Advanced Use Cases

### Integration with Machine Learning Pipelines

This example shows how to integrate NTQR evaluation with a scikit-learn pipeline:

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from fork.src.core.r2 import r2_evaluators
from fork.src.utils import data_processing, visualization

# Load dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train three different classifiers
classifiers = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "Neural Network": MLPClassifier(random_state=42, max_iter=1000)
}

# Train each classifier
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)
    print(f"{name}: Train acc = {train_acc:.4f}, Test acc = {test_acc:.4f}")

# Get predictions from each classifier
predictions = np.column_stack([
    clf.predict(X_test) for name, clf in classifiers.items()
])

# Convert to agreement counts
agreement_counts = data_processing.convert_responses_to_agreement_counts(predictions)

# Evaluate without ground truth
evaluator = r2_evaluators.EnhancedTrioEvaluation()
results = evaluator.evaluate(agreement_counts)

# Print NTQR evaluation results
print("\nNTQR Evaluation Results (without ground truth):")
for i, name in enumerate(classifiers.keys()):
    print(f"{name}: Estimated accuracy = {results.accuracies[i]:.4f}")

# Compare with ground truth
true_accuracies = [
    np.mean(predictions[:, i] == y_test) for i in range(len(classifiers))
]

print("\nComparison with ground truth:")
for i, name in enumerate(classifiers.keys()):
    print(f"{name}: True acc = {true_accuracies[i]:.4f}, " +
          f"NTQR est = {results.accuracies[i]:.4f}, " +
          f"Diff = {results.accuracies[i] - true_accuracies[i]:.4f}")

# Visualize results
visualization.plot_accuracy_comparison(
    list(classifiers.keys()),
    true_accuracies,
    results.accuracies,
    title="NTQR vs. Ground Truth Accuracy",
    save_path="ntqr_vs_ground_truth.png"
)
```

### Custom Axiom Development

This example demonstrates how to develop and test a custom axiom:

```python
import numpy as np
import sympy as sp
from fork.src.core.r2 import r2_axioms
from fork.src.utils import data_processing, axiom_testing

# Define a custom axiom for binary classification
class CustomPairwiseCorrelationAxiom(r2_axioms.PairwiseAxiom):
    """
    A custom axiom that encodes a different correlation constraint between
    pairs of classifiers.
    """
    
    def __init__(self, classifier_indices=None):
        super().__init__(classifier_indices)
        self.name = "Custom Pairwise Correlation Axiom"
    
    def generate_constraint_equation(self, p_vars, data):
        """
        Generate a custom constraint equation based on pairwise correlation.
        
        Parameters:
        -----------
        p_vars : list of sympy.Symbol
            Symbolic variables representing classifier accuracies
        data : numpy.ndarray
            Agreement counts data
            
        Returns:
        --------
        sympy.Expr
            The constraint equation
        """
        i, j = self.classifier_indices
        
        # Extract variables for classifiers i and j
        p_i0, p_i1 = p_vars[2*i], p_vars[2*i+1]
        p_j0, p_j1 = p_vars[2*j], p_vars[2*j+1]
        
        # Custom constraint based on weighted correlation
        constraint = p_i0 * p_j0 + p_i1 * p_j1 - 0.5 * (p_i0 + p_i1 + p_j0 + p_j1)
        
        return constraint
    
    def compute_constraints(self, p_vars, data):
        """
        Compute constraints as equations.
        
        Parameters:
        -----------
        p_vars : list of sympy.Symbol
            Symbolic variables representing classifier accuracies
        data : numpy.ndarray
            Agreement counts data
            
        Returns:
        --------
        list of sympy.Eq
            Constraint equations
        """
        constraint_expr = self.generate_constraint_equation(p_vars, data)
        return [sp.Eq(constraint_expr, 0)]

# Test the custom axiom with synthetic data
def test_custom_axiom():
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    true_labels = np.random.randint(0, 2, size=n_samples)
    
    # Generate classifier responses
    accuracies = [0.8, 0.75, 0.7]
    responses = np.zeros((n_samples, len(accuracies)), dtype=int)
    
    for i, acc in enumerate(accuracies):
        responses[:, i] = true_labels.copy()
        flip_indices = np.random.choice(
            np.arange(n_samples),
            size=int(n_samples * (1 - acc)),
            replace=False
        )
        responses[flip_indices, i] = 1 - responses[flip_indices, i]
    
    # Convert to agreement counts
    agreement_counts = data_processing.convert_responses_to_agreement_counts(responses)
    
    # Create symbolic variables
    p_vars = []
    for i in range(len(accuracies)):
        p_vars.extend([sp.Symbol(f'p_{i}0'), sp.Symbol(f'p_{i}1')])
    
    # Test standard axiom
    standard_axiom = r2_axioms.PairwiseAxiom([0, 1])
    standard_constraints = standard_axiom.compute_constraints(p_vars, agreement_counts)
    
    # Test custom axiom
    custom_axiom = CustomPairwiseCorrelationAxiom([0, 1])
    custom_constraints = custom_axiom.compute_constraints(p_vars, agreement_counts)
    
    # Print results
    print("Standard PairwiseAxiom Constraint:")
    print(standard_constraints[0])
    
    print("\nCustom Axiom Constraint:")
    print(custom_constraints[0])
    
    # Verify constraints with known accuracies
    values = {
        p_vars[0]: accuracies[0],  # p_00
        p_vars[1]: accuracies[0],  # p_01
        p_vars[2]: accuracies[1],  # p_10
        p_vars[3]: accuracies[1],  # p_11
    }
    
    standard_residual = abs(float(standard_constraints[0].lhs.subs(values)))
    custom_residual = abs(float(custom_constraints[0].lhs.subs(values)))
    
    print(f"\nStandard axiom residual with true accuracies: {standard_residual:.6f}")
    print(f"Custom axiom residual with true accuracies: {custom_residual:.6f}")
    
    # Test if the custom axiom can be used in an evaluator
    axiom_testing.verify_axiom_compatibility(
        custom_axiom,
        agreement_counts,
        expected_accuracies=accuracies[:2]
    )

# Run the test
test_custom_axiom()
```

## Resources

- [Source code for these examples](https://github.com/yourusername/IntroductionToAlgebraicEvaluation/tree/main/fork/examples)
- [Implementation Guide](implementation_guide.md) for more details on extending these examples
- [Core Concepts](../analytical/core_concepts.md) for theoretical background 