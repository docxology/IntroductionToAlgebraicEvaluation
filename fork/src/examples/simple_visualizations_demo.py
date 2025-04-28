#!/usr/bin/env python3
"""
Simple demonstration of all visualization techniques for the NTQR package.

This script showcases the basic visualization capabilities of the NTQR package
and the enhanced visualizations from our fork, saving all outputs to a specified
output directory.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path if needed
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import from main package
from python.src.ntqr.plots import plot_evaluation_space

# Import our extensions
from fork.src.extensions.visualizations import (
    plot_comparative_evaluation,
    plot_trio_agreement_matrix,
    plot_evaluation_confidence,
)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default=None,
        help="Directory to save output files (default: fork/output)"
    )
    parser.add_argument(
        "--dpi", 
        type=int,
        default=300,
        help="DPI for saved images (default: 300)"
    )
    parser.add_argument(
        "--seed", 
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    return parser.parse_args()


def setup_output_directory(args):
    """Set up the output directory."""
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = project_root / "fork" / "output"
    
    output_dir.mkdir(exist_ok=True)
    print(f"Output will be saved to: {output_dir}")
    return output_dir


def generate_synthetic_data(seed=42):
    """Generate synthetic data for visualizations."""
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Generate predictions for three classifiers
    n_samples = 200
    ground_truth = np.random.randint(0, 2, size=n_samples)
    
    # Generate predictions with specified accuracy
    def generate_predictions(true_labels, accuracy):
        predictions = np.copy(true_labels)
        # Flip some predictions according to accuracy
        flip_mask = np.random.random(len(true_labels)) > accuracy
        predictions[flip_mask] = 1 - predictions[flip_mask]
        return predictions
    
    # Create classifiers with different accuracies
    classifier_a = generate_predictions(ground_truth, 0.85)
    classifier_b = generate_predictions(ground_truth, 0.75)
    classifier_c = generate_predictions(ground_truth, 0.80)
    
    # Create evaluation results for each classifier
    class EvaluationResult:
        def __init__(self, accuracy_p0, accuracy_p1):
            self.accuracy_p0 = accuracy_p0
            self.accuracy_p1 = accuracy_p1
            self.accuracy = (accuracy_p0 + accuracy_p1) / 2
    
    # Calculate accuracies for each classifier and label
    def calculate_accuracies(predictions, true_labels):
        # For label 0
        mask_0 = (true_labels == 0)
        accuracy_0 = np.mean(predictions[mask_0] == true_labels[mask_0]) if np.any(mask_0) else 0
        
        # For label 1
        mask_1 = (true_labels == 1)
        accuracy_1 = np.mean(predictions[mask_1] == true_labels[mask_1]) if np.any(mask_1) else 0
        
        return accuracy_0, accuracy_1
    
    # Calculate actual accuracies
    acc_a0, acc_a1 = calculate_accuracies(classifier_a, ground_truth)
    acc_b0, acc_b1 = calculate_accuracies(classifier_b, ground_truth)
    acc_c0, acc_c1 = calculate_accuracies(classifier_c, ground_truth)
    
    # Create evaluation results
    eval_a = EvaluationResult(acc_a0, acc_a1)
    eval_b = EvaluationResult(acc_b0, acc_b1)
    eval_c = EvaluationResult(acc_c0, acc_c1)
    
    # Create trio vote patterns (8 possible patterns: 000, 001, ..., 111)
    vote_patterns = np.zeros(8)
    
    # Count occurrences of each pattern
    for i in range(n_samples):
        # Convert to pattern index (000 = 0, 001 = 1, etc.)
        pattern_idx = classifier_a[i] * 4 + classifier_b[i] * 2 + classifier_c[i]
        vote_patterns[pattern_idx] += 1
    
    return {
        'ground_truth': ground_truth,
        'classifiers': {
            'A': classifier_a,
            'B': classifier_b,
            'C': classifier_c
        },
        'evaluations': {
            'A': eval_a,
            'B': eval_b,
            'C': eval_c
        },
        'vote_patterns': vote_patterns
    }


def create_visualizations(data, output_dir, args):
    """Create all visualizations."""
    print("Creating visualizations...")
    
    # 1. Basic evaluation space
    fig = plot_evaluation_space(title="NTQR Evaluation Space")
    fig.savefig(output_dir / "evaluation_space.png", dpi=args.dpi, bbox_inches='tight')
    print(f"  - Saved evaluation space to {output_dir / 'evaluation_space.png'}")
    plt.close(fig)
    
    # 2. Comparative evaluation
    evaluations = [
        (f"Classifier {name}", eval_result) 
        for name, eval_result in data['evaluations'].items()
    ]
    
    fig = plot_comparative_evaluation(
        evaluations=evaluations,
        title="Comparison of Classifier Performance",
        colors=['#3498db', '#2ecc71', '#9b59b6']  # Custom colors
    )
    fig.savefig(output_dir / "comparative_evaluation.png", dpi=args.dpi, bbox_inches='tight')
    print(f"  - Saved comparative evaluation to {output_dir / 'comparative_evaluation.png'}")
    plt.close(fig)
    
    # 3. Trio agreement matrix
    fig = plot_trio_agreement_matrix(
        trio_data=data['vote_patterns'],
        classifier_names=["Classifier A", "Classifier B", "Classifier C"],
        title="Agreement Patterns Among Classifiers"
    )
    fig.savefig(output_dir / "agreement_matrix.png", dpi=args.dpi, bbox_inches='tight')
    print(f"  - Saved agreement matrix to {output_dir / 'agreement_matrix.png'}")
    plt.close(fig)
    
    # 4. Evaluation confidence plots
    for name, eval_result in data['evaluations'].items():
        fig = plot_evaluation_confidence(
            evaluation_result=eval_result,
            title=f"Confidence in Evaluation (Classifier {name})",
            confidence_range=(0.9, 0.99)
        )
        fig.savefig(output_dir / f"confidence_classifier_{name}.png", dpi=args.dpi, bbox_inches='tight')
        print(f"  - Saved confidence plot for Classifier {name}")
        plt.close(fig)
    
    # 5. Classifier points in evaluation space
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot the unit square (valid accuracy range)
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, linestyle='-', 
                              linewidth=2, color='black'))
    
    # Add diagonal line (representing p0 + p1 = 1)
    ax.plot([0, 1], [1, 0], 'r--', label='p₀ + p₁ = 1')
    
    # Add points for each classifier
    colors = ['blue', 'green', 'purple']
    for i, (name, eval_result) in enumerate(data['evaluations'].items()):
        ax.scatter([eval_result.accuracy_p0], [eval_result.accuracy_p1], 
                   color=colors[i], s=100, marker='o', label=f'Classifier {name}')
    
    # Add labels and title
    ax.set_xlabel('Accuracy on Label 0 (p₀)')
    ax.set_ylabel('Accuracy on Label 1 (p₁)')
    ax.set_title('Classifier Performance in Evaluation Space')
    
    # Set limits with some padding
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    
    # Add grid and legend
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    
    fig.savefig(output_dir / "classifier_evaluation_space.png", dpi=args.dpi, bbox_inches='tight')
    print(f"  - Saved classifier evaluation space to {output_dir / 'classifier_evaluation_space.png'}")
    plt.close(fig)
    
    # 6. Create a voting distribution plot
    fig, ax = plt.subplots(figsize=(10, 6))
    pattern_labels = ['000', '001', '010', '011', '100', '101', '110', '111']
    ax.bar(pattern_labels, data['vote_patterns'])
    ax.set_title('Distribution of Voting Patterns')
    ax.set_xlabel('Voting Pattern (A,B,C)')
    ax.set_ylabel('Count')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add color coded annotations for majority vote
    for i, count in enumerate(data['vote_patterns']):
        majority = '0' if pattern_labels[i].count('1') < 2 else '1'
        color = 'blue' if majority == '0' else 'red'
        if count > 0:  # Only add text if there's a visible bar
            ax.text(i, count + 1, f'Maj: {majority}', ha='center', color=color)
    
    fig.savefig(output_dir / "voting_patterns.png", dpi=args.dpi, bbox_inches='tight')
    print(f"  - Saved voting patterns to {output_dir / 'voting_patterns.png'}")
    plt.close(fig)
    
    # 7. Create a pairwise agreement heatmap
    pairwise_agreement = np.zeros((3, 3))
    classifier_values = list(data['classifiers'].values())
    
    for i in range(3):
        for j in range(3):
            pairwise_agreement[i, j] = np.mean(classifier_values[i] == classifier_values[j])
    
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(pairwise_agreement, cmap='Reds', vmin=0.5, vmax=1)
    
    # Add labels
    classifier_names = ['A', 'B', 'C']
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(classifier_names)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(classifier_names)
    ax.set_title('Pairwise Agreement Between Classifiers')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Agreement Ratio')
    
    # Add text annotations
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f'{pairwise_agreement[i, j]:.2f}', 
                   ha='center', va='center', 
                   color='white' if pairwise_agreement[i, j] > 0.75 else 'black')
    
    fig.savefig(output_dir / "pairwise_agreement_heatmap.png", dpi=args.dpi, bbox_inches='tight')
    print(f"  - Saved pairwise agreement to {output_dir / 'pairwise_agreement_heatmap.png'}")
    plt.close(fig)
    
    # 8. Create a summary report
    report_path = output_dir / "evaluation_summary.txt"
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("NTQR Evaluation Summary Report\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("1. Individual Classifier Performance\n")
        f.write("-" * 40 + "\n")
        for name, result in data['evaluations'].items():
            f.write(f"Classifier {name}:\n")
            f.write(f"  Accuracy on Class 0: {result.accuracy_p0:.4f}\n")
            f.write(f"  Accuracy on Class 1: {result.accuracy_p1:.4f}\n")
            f.write(f"  Overall Accuracy:    {result.accuracy:.4f}\n\n")
        
        f.write("2. Agreement Statistics\n")
        f.write("-" * 40 + "\n")
        f.write(f"Pairwise Agreement:\n")
        for i, name1 in enumerate(['A', 'B', 'C']):
            for j, name2 in enumerate(['A', 'B', 'C']):
                if i < j:  # Only print each pair once
                    f.write(f"  {name1}-{name2}: {pairwise_agreement[i, j]:.4f}\n")
        
        # Calculate unanimous agreement
        classifiers = list(data['classifiers'].values())
        unanimous_agreement = np.mean((classifiers[0] == classifiers[1]) & 
                                     (classifiers[1] == classifiers[2]))
        f.write(f"Unanimous Agreement: {unanimous_agreement:.4f}\n\n")
        
        f.write("3. Voting Pattern Distribution\n")
        f.write("-" * 40 + "\n")
        pattern_labels = ['000', '001', '010', '011', '100', '101', '110', '111']
        for pattern, count in zip(pattern_labels, data['vote_patterns']):
            f.write(f"Pattern {pattern}: {int(count)} votes\n")
        
        f.write("\nReport generated by NTQR fork visualization tools\n")
    
    print(f"  - Saved summary report to {report_path}")


def main():
    """Main function."""
    # Parse arguments
    args = parse_arguments()
    
    # Set up output directory
    output_dir = setup_output_directory(args)
    
    # Generate synthetic data
    data = generate_synthetic_data(seed=args.seed)
    
    print("=" * 80)
    print("NTQR Visualization Demonstration")
    print("=" * 80)
    
    # Create visualizations
    create_visualizations(data, output_dir, args)
    
    print("\n" + "=" * 80)
    print("Demonstration complete!")
    print(f"All outputs saved to: {output_dir}")
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 