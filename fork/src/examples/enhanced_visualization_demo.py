#!/usr/bin/env python3
"""
Enhanced Visualization Demo

This script demonstrates the enhanced visualization capabilities
of the NTQR fork.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import from main package
try:
    from python.src.ntqr.r2.evaluators import (
        SupervisedEvaluation,
        ErrorIndependentEvaluation,
        MajorityVotingEvaluation
    )
    from python.src.ntqr.r2.datasets import synthetic_binary_dataset
except ImportError:
    print("Warning: Could not import from main NTQR package. Using mock data instead.")
    SupervisedEvaluation = None
    ErrorIndependentEvaluation = None
    MajorityVotingEvaluation = None
    synthetic_binary_dataset = None

# Import our visualization tools
from fork.src.utils.visualization import (
    enhanced_evaluation_space,
    plot_error_correlation_matrix,
    plot_agreement_patterns,
    plot_accuracy_comparison,
    plot_3d_evaluation_space
)


def create_output_directory():
    """Create output directory for saving plots."""
    output_dir = Path(__file__).parent.parent.parent / "output" / "visualization_demo"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def generate_mock_data():
    """Generate mock data for demonstration."""
    # Generate synthetic class accuracies for 3 classifiers
    np.random.seed(42)
    
    # Create mock evaluation result
    class MockResult:
        def __init__(self):
            self.accuracies = [0.82, 0.75, 0.68]
            self.accuracy_p0 = 0.8
            self.accuracy_p1 = 0.7
            self.constraint_intersection = (0.75, 0.65)
            self.bootstrap_samples = np.random.normal(
                loc=[0.8, 0.7], scale=[0.05, 0.05], size=(100, 2))
            self.error_correlation_matrix = np.array([
                [1.0, 0.3, 0.5],
                [0.3, 1.0, 0.2],
                [0.5, 0.2, 1.0]
            ])
            self.class_accuracies = [
                [0.8, 0.7, 0.6],  # Classifier 1
                [0.75, 0.7, 0.65],  # Classifier 2
                [0.7, 0.65, 0.6]  # Classifier 3
            ]
    
    # Generate agreement pattern counts for 3 classifiers
    # Total 2^3 = 8 patterns: 000, 001, ..., 111
    agreement_counts = np.array([120, 30, 40, 20, 25, 35, 60, 80])
    
    # Generate classifier names, true and estimated accuracies
    classifier_names = ["Random Forest", "Logistic Regression", "Neural Network"]
    true_accuracies = [0.85, 0.80, 0.78]
    estimated_accuracies = [0.83, 0.79, 0.80]
    
    return MockResult(), agreement_counts, classifier_names, true_accuracies, estimated_accuracies


def generate_real_data():
    """Generate real data using the main NTQR package."""
    if synthetic_binary_dataset is None:
        print("Error: Main NTQR package not available.")
        return None, None, None, None, None
    
    # Generate synthetic dataset
    accuracies = [0.82, 0.75, 0.68]
    dataset = synthetic_binary_dataset(
        n_examples=1000,
        accuracies=accuracies,
        error_correlation=0.2,
        class_balance=0.5,
        seed=42
    )
    
    # Get ground truth
    ground_truth = dataset["ground_truth"]
    
    # Get classifier responses
    responses = np.column_stack([
        dataset["classifier1"],
        dataset["classifier2"],
        dataset["classifier3"]
    ])
    
    # Perform supervised evaluation (with ground truth)
    supervised_eval = SupervisedEvaluation()
    supervised_result = supervised_eval.evaluate(responses, ground_truth)
    
    # Create label vote counts for error-independent evaluation
    vote_counts = np.zeros((2, 2, 2))  # Counts for each voting pattern
    for i in range(len(responses)):
        idx = tuple(responses[i])
        vote_counts[idx] += 1
    
    # Perform error-independent evaluation (without ground truth)
    ei_eval = ErrorIndependentEvaluation()
    ei_result = ei_eval.evaluate(vote_counts)
    
    # Return results
    return (supervised_result, vote_counts, 
            ["Classifier 1", "Classifier 2", "Classifier 3"],
            [supervised_result.accuracy_clfs[i] for i in range(3)],
            [ei_result.accuracy_clfs[i] for i in range(3)])


def demonstrate_enhanced_evaluation_space(result, output_dir):
    """Demonstrate the enhanced evaluation space plot."""
    print("\nDemonstrating enhanced evaluation space...")
    
    # Create plot with default settings
    fig1 = enhanced_evaluation_space(
        result,
        title="Enhanced Evaluation Space",
        save_path=str(output_dir / "enhanced_evaluation_space.png")
    )
    
    # Create plot with confidence regions
    fig2 = enhanced_evaluation_space(
        result,
        title="Evaluation Space with Confidence Regions",
        highlight_constraints=True,
        show_confidence=True,
        confidence_level=0.95,
        save_path=str(output_dir / "evaluation_space_with_confidence.png")
    )
    
    print(f"  - Created evaluation space plots in {output_dir}")
    return fig1, fig2


def demonstrate_error_correlation_matrix(result, output_dir, classifier_names):
    """Demonstrate the error correlation matrix plot."""
    print("\nDemonstrating error correlation matrix...")
    
    # Create plot
    fig = plot_error_correlation_matrix(
        result.error_correlation_matrix,
        classifier_names=classifier_names,
        title="Error Correlation Matrix",
        save_path=str(output_dir / "error_correlation_matrix.png")
    )
    
    print(f"  - Created error correlation matrix plot in {output_dir}")
    return fig


def demonstrate_agreement_patterns(agreement_counts, output_dir):
    """Demonstrate the agreement patterns plot."""
    print("\nDemonstrating agreement patterns...")
    
    # Generate pattern labels
    pattern_labels = ["000", "001", "010", "011", "100", "101", "110", "111"]
    
    # Create plot
    fig = plot_agreement_patterns(
        agreement_counts,
        pattern_labels=pattern_labels,
        title="Classifier Agreement Patterns",
        save_path=str(output_dir / "agreement_patterns.png")
    )
    
    print(f"  - Created agreement patterns plot in {output_dir}")
    return fig


def demonstrate_accuracy_comparison(classifier_names, true_accuracies, 
                                   estimated_accuracies, output_dir):
    """Demonstrate the accuracy comparison plot."""
    print("\nDemonstrating accuracy comparison...")
    
    # Create plot
    fig = plot_accuracy_comparison(
        classifier_names,
        true_accuracies,
        estimated_accuracies,
        title="True vs. Estimated Accuracies",
        include_difference=True,
        save_path=str(output_dir / "accuracy_comparison.png")
    )
    
    print(f"  - Created accuracy comparison plot in {output_dir}")
    return fig


def demonstrate_3d_evaluation_space(result, output_dir):
    """Demonstrate the 3D evaluation space plot."""
    print("\nDemonstrating 3D evaluation space...")
    
    # Create plot
    fig = plot_3d_evaluation_space(
        result,
        title="3D Evaluation Space for Ternary Classification",
        show_constraints=True,
        save_path=str(output_dir / "3d_evaluation_space.png")
    )
    
    print(f"  - Created 3D evaluation space plot in {output_dir}")
    return fig


def main():
    """Run the visualization demonstrations."""
    print("NTQR Fork - Enhanced Visualization Demonstration")
    print("="*50)
    
    # Create output directory
    output_dir = create_output_directory()
    print(f"Output directory: {output_dir}")
    
    # Try to generate real data first
    result, agreement_counts, classifier_names, true_accuracies, estimated_accuracies = (
        generate_real_data()
    )
    
    # Fall back to mock data if needed
    if result is None:
        print("Using mock data for demonstration.")
        result, agreement_counts, classifier_names, true_accuracies, estimated_accuracies = (
            generate_mock_data()
        )
    
    # Run demonstrations
    demonstrate_enhanced_evaluation_space(result, output_dir)
    demonstrate_error_correlation_matrix(result, output_dir, classifier_names)
    demonstrate_agreement_patterns(agreement_counts, output_dir)
    demonstrate_accuracy_comparison(
        classifier_names, true_accuracies, estimated_accuracies, output_dir)
    
    # Re-enable 3D visualization with error handling
    print("\nDemonstrating 3D evaluation space...")
    try:
        demonstrate_3d_evaluation_space(result, output_dir)
        print(f"  - Created 3D evaluation space plot in {output_dir}")
    except Exception as e:
        print(f"  - Failed to create 3D plot: {str(e)}")
        print("  - This is likely due to matplotlib 3D support not being available.")
        print("  - The visualization module will create a fallback image instead.")
    
    print("\nAll demonstrations complete!")
    print(f"Output saved to: {output_dir}")
    
    # Show plots interactively if running in interactive mode
    if sys.flags.interactive:
        plt.show()


if __name__ == "__main__":
    main() 