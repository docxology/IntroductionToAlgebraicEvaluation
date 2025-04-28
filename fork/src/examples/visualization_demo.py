#!/usr/bin/env python
"""
Demonstration of enhanced visualization tools from the NTQR fork.

This script showcases the visualization tools developed in our fork
by applying them to some example evaluation scenarios.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path if needed
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Try direct imports first
try:
    from python.src.ntqr.r2.evaluators import (
        SupervisedEvaluation,
        ErrorIndependentEvaluation,
        MajorityVotingEvaluation,
    )
    from python.src.ntqr.r2.datasketches import TrioVoteCounts, TrioLabelVoteCounts
except ImportError:
    print("Warning: Could not import from python.src.ntqr directly.")
    print("The demo may not work correctly.")
    sys.exit(1)

# Import our visualizations
try:
    from fork.src.extensions.visualizations import (
        plot_comparative_evaluation,
        plot_trio_agreement_matrix,
        plot_evaluation_confidence,
    )
except ImportError:
    # If the above import fails, try a relative import
    print("Warning: Using relative imports for visualization tools.")
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from extensions.visualizations import (
        plot_comparative_evaluation,
        plot_trio_agreement_matrix,
        plot_evaluation_confidence,
    )


def create_example_data():
    """
    Create example data for demonstration.
    
    Returns
    -------
    tuple
        Tuple containing example data objects
    """
    # Create binary classification data for supervised evaluation
    binary_data_good = np.array([
        [80, 20],  # 80 true positives, 20 false negatives
        [10, 90]   # 10 false positives, 90 true negatives
    ])
    
    binary_data_fair = np.array([
        [70, 30],  # 70 true positives, 30 false negatives
        [25, 75]   # 25 false positives, 75 true negatives
    ])
    
    binary_data_poor = np.array([
        [60, 40],  # 60 true positives, 40 false negatives
        [40, 60]   # 40 false positives, 60 true negatives
    ])
    
    # Create trio data for error-independent evaluation
    # This data represents a case where:
    # - All classifiers are reasonably accurate (75-80%)
    # - Errors are mostly independent
    # - Majority vote is generally correct
    trio_votes = np.array([
        20,  # n_000: all voted 0
        5,   # n_001: 1 and 2 voted 0, 3 voted 1
        5,   # n_010: 1 and 3 voted 0, 2 voted 1
        3,   # n_011: 1 voted 0, 2 and 3 voted 1
        5,   # n_100: 2 and 3 voted 0, 1 voted 1
        3,   # n_101: 2 voted 0, 1 and 3 voted 1
        3,   # n_110: 3 voted 0, 1 and 2 voted 1
        26,  # n_111: all voted 1
    ])
    
    # Create trio counts object
    trio_counts = TrioVoteCounts(trio_votes)
    
    # Create label vote counts for supervised evaluation
    # For binary classification with known ground truth
    label_vote_counts = np.array([
        [18, 3, 3, 1, 3, 1, 1, 0],  # patterns for true label 0
        [2, 2, 2, 2, 2, 2, 2, 26],  # patterns for true label 1
    ])
    
    # Create TrioLabelVoteCounts object
    trio_label_counts = TrioLabelVoteCounts(label_vote_counts)
    
    return (
        binary_data_good, 
        binary_data_fair, 
        binary_data_poor, 
        trio_votes, 
        trio_counts, 
        trio_label_counts
    )


def demo_comparative_evaluation(
    binary_data_good, 
    binary_data_fair,
    binary_data_poor,
    output_dir
):
    """
    Demonstrate the comparative evaluation plot.
    
    Parameters
    ----------
    binary_data_good, binary_data_fair, binary_data_poor : np.ndarray
        Binary classification data with different quality levels
    output_dir : str
        Directory to save output files
    """
    print("Demonstrating comparative evaluation plot...")
    
    # Create evaluator
    evaluator = SupervisedEvaluation()
    
    # Evaluate each dataset
    result_good = evaluator.evaluate(binary_data_good)
    result_fair = evaluator.evaluate(binary_data_fair)
    result_poor = evaluator.evaluate(binary_data_poor)
    
    # Create evaluations list
    evaluations = [
        ("Good Classifier", result_good),
        ("Fair Classifier", result_fair),
        ("Poor Classifier", result_poor),
    ]
    
    # Create plot
    fig = plot_comparative_evaluation(
        evaluations=evaluations,
        title="Comparison of Classifier Performance",
        save_path=os.path.join(output_dir, "comparative_evaluation.png")
    )
    
    print(f"  - Created comparative evaluation plot")
    print(f"  - Saved to {os.path.join(output_dir, 'comparative_evaluation.png')}")
    
    # Display metrics
    print("  - Metrics:")
    for name, result in evaluations:
        print(f"    {name}:")
        print(f"      Accuracy on Label 0: {result.accuracy_p0:.3f}")
        print(f"      Accuracy on Label 1: {result.accuracy_p1:.3f}")
        print(f"      Overall Accuracy: {result.accuracy:.3f}")
    
    plt.close(fig)


def demo_trio_agreement_matrix(trio_votes, output_dir):
    """
    Demonstrate the trio agreement matrix plot.
    
    Parameters
    ----------
    trio_votes : np.ndarray
        Array of vote counts with 8 elements [n_000, n_001, ..., n_111]
    output_dir : str
        Directory to save output files
    """
    print("\nDemonstrating trio agreement matrix plot...")
    
    # Create plot
    fig = plot_trio_agreement_matrix(
        trio_data=trio_votes,
        classifier_names=["LLM Model A", "LLM Model B", "Human Expert"],
        title="Agreement Patterns in LLM Evaluation",
        save_path=os.path.join(output_dir, "trio_agreement_matrix.png")
    )
    
    print(f"  - Created trio agreement matrix plot")
    print(f"  - Saved to {os.path.join(output_dir, 'trio_agreement_matrix.png')}")
    
    # Display analytics about the agreement patterns
    print("  - Agreement analytics:")
    total_items = np.sum(trio_votes)
    unanimous_agreement = trio_votes[0] + trio_votes[7]
    majority_agreement = (
        trio_votes[0] + trio_votes[1] + trio_votes[2] + trio_votes[4] +  # majority 0
        trio_votes[3] + trio_votes[5] + trio_votes[6] + trio_votes[7]    # majority 1
    )
    print(f"    Total test items: {total_items}")
    print(f"    Unanimous agreement: {unanimous_agreement} items ({unanimous_agreement/total_items:.1%})")
    print(f"    Majority agreement: {majority_agreement} items ({majority_agreement/total_items:.1%})")
    
    # Patterns where human disagrees with both LLMs
    human_disagrees_both = trio_votes[1] + trio_votes[6]
    print(f"    Human disagrees with both LLMs: {human_disagrees_both} items ({human_disagrees_both/total_items:.1%})")
    
    plt.close(fig)


def demo_evaluation_confidence(trio_label_counts, output_dir):
    """
    Demonstrate the evaluation confidence plot.
    
    Parameters
    ----------
    trio_label_counts : TrioLabelVoteCounts
        Label vote counts for a trio of classifiers
    output_dir : str
        Directory to save output files
    """
    print("\nDemonstrating evaluation confidence plot...")
    
    # Create evaluators
    ei_evaluator = ErrorIndependentEvaluation()
    mv_evaluator = MajorityVotingEvaluation()
    
    # Evaluate using error-independent evaluator
    ei_result = ei_evaluator.evaluate(trio_label_counts)
    
    # Evaluate using majority voting
    mv_result = mv_evaluator.evaluate(trio_label_counts)
    
    # Create plots for both evaluation results
    fig1 = plot_evaluation_confidence(
        evaluation_result=ei_result.classifier_accuracies[0],
        title="Confidence in Error-Independent Evaluation (Classifier 1)",
        save_path=os.path.join(output_dir, "ei_confidence.png")
    )
    
    fig2 = plot_evaluation_confidence(
        evaluation_result=mv_result,
        title="Confidence in Majority Voting Evaluation",
        save_path=os.path.join(output_dir, "mv_confidence.png")
    )
    
    print(f"  - Created evaluation confidence plots")
    print(f"  - Saved to {os.path.join(output_dir, 'ei_confidence.png')}")
    print(f"  - Saved to {os.path.join(output_dir, 'mv_confidence.png')}")
    
    # Display metrics
    print("  - Error-Independent Evaluation Metrics:")
    for i, result in enumerate(ei_result.classifier_accuracies):
        print(f"    Classifier {i+1}:")
        print(f"      Accuracy on Label 0: {result.accuracy_p0:.3f}")
        print(f"      Accuracy on Label 1: {result.accuracy_p1:.3f}")
        print(f"      Overall Accuracy: {result.accuracy:.3f}")
    
    print("  - Majority Voting Evaluation Metrics:")
    print(f"    Accuracy on Label 0: {mv_result.accuracy_p0:.3f}")
    print(f"    Accuracy on Label 1: {mv_result.accuracy_p1:.3f}")
    print(f"    Overall Accuracy: {mv_result.accuracy:.3f}")
    
    plt.close(fig1)
    plt.close(fig2)


def main():
    """
    Run the visualization demonstrations.
    """
    print("=" * 80)
    print("NTQR Fork - Visualization Demonstration")
    print("=" * 80)
    
    # Create directory for output files
    output_dir = Path(project_root) / "fork" / "output"
    output_dir.mkdir(exist_ok=True)
    print(f"Output files will be saved to: {output_dir}")
    
    # Create example data
    (
        binary_data_good, 
        binary_data_fair, 
        binary_data_poor, 
        trio_votes, 
        trio_counts, 
        trio_label_counts
    ) = create_example_data()
    
    # Run demonstrations
    demo_comparative_evaluation(
        binary_data_good, 
        binary_data_fair,
        binary_data_poor,
        output_dir
    )
    
    demo_trio_agreement_matrix(trio_votes, output_dir)
    
    demo_evaluation_confidence(trio_label_counts, output_dir)
    
    print("\nDemonstration complete.")
    print(f"All output files have been saved to: {output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 