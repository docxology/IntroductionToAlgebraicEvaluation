#!/usr/bin/env python3
"""
Comprehensive demonstration of all visualization techniques in the NTQR package.

This script showcases both the basic visualization tools from the main package
and the enhanced visualization tools from our fork, saving all outputs to the
specified output directory.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Dict, Any

# Add project root to path if needed
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import compatibility utils
from fork.src.utils.compatibility import ensure_main_package_importable, check_compatibility

# Ensure main package is importable
if not ensure_main_package_importable():
    print("ERROR: Could not import the main NTQR package.")
    print("Please make sure it is installed or available in the project structure.")
    sys.exit(1)

# Check compatibility with main package
if not check_compatibility():
    print("WARNING: Version mismatch with main NTQR package.")
    print("Some visualizations may not work as expected.")

# Import from main package
try:
    from python.src.ntqr.plots import plot_evaluation_space as main_plot_evaluation_space
    from python.src.ntqr.r2.evaluators import (
        SupervisedEvaluation,
        ErrorIndependentEvaluation,
        MajorityVotingEvaluation,
    )
    from python.src.ntqr.r2.datasketches import TrioVoteCounts, TrioLabelVoteCounts
except ImportError as e:
    print(f"ERROR: Failed to import from main package: {e}")
    sys.exit(1)

# Import our enhanced visualizations
try:
    from fork.src.extensions.visualizations import (
        plot_comparative_evaluation,
        plot_trio_agreement_matrix,
        plot_evaluation_confidence,
    )
except ImportError as e:
    print(f"ERROR: Failed to import from fork extensions: {e}")
    sys.exit(1)


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
        "--show-plots", 
        action="store_true",
        help="Show plots interactively (default: False)"
    )
    parser.add_argument(
        "--seed", 
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    return parser.parse_args()


def setup_output_directory(args) -> Path:
    """Set up the output directory."""
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = project_root / "fork" / "output"
    
    output_dir.mkdir(exist_ok=True)
    print(f"Output will be saved to: {output_dir}")
    return output_dir


def save_figure(fig, filename, output_dir, dpi=300, show=False):
    """Save a figure to disk and optionally display it."""
    # Save the figure
    save_path = output_dir / filename
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f"Saved: {save_path}")
    
    # Show if requested
    if show:
        plt.show()
    else:
        plt.close(fig)


def generate_synthetic_data(seed=42):
    """Generate synthetic data for demos."""
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Generate ground truth (0 or 1)
    n_samples = 200
    ground_truth = np.random.randint(0, 2, size=n_samples)
    
    # Generate predictions with specified accuracy
    def generate_predictions(true_labels, accuracy):
        predictions = np.copy(true_labels)
        # Flip some predictions according to accuracy
        flip_mask = np.random.random(len(true_labels)) > accuracy
        predictions[flip_mask] = 1 - predictions[flip_mask]
        return predictions
    
    # Generate predictions for three classifiers
    pred_a = generate_predictions(ground_truth, 0.85)
    pred_b = generate_predictions(ground_truth, 0.75)
    pred_c = generate_predictions(ground_truth, 0.80)
    
    # Create confusion matrices for supervised evaluation
    def create_confusion_matrix(true_labels, predictions):
        # Format:
        # [
        #   [TP, FN],  # Row for positive examples (true label = 1)
        #   [FP, TN]   # Row for negative examples (true label = 0)
        # ]
        tp = np.sum((true_labels == 1) & (predictions == 1))
        fn = np.sum((true_labels == 1) & (predictions == 0))
        fp = np.sum((true_labels == 0) & (predictions == 1))
        tn = np.sum((true_labels == 0) & (predictions == 0))
        return np.array([
            [tp, fn],
            [fp, tn]
        ])
    
    binary_data_a = create_confusion_matrix(ground_truth, pred_a)
    binary_data_b = create_confusion_matrix(ground_truth, pred_b)
    binary_data_c = create_confusion_matrix(ground_truth, pred_c)
    
    # Create trio vote patterns
    # Create an 8-element array for vote patterns [000, 001, 010, 011, 100, 101, 110, 111]
    trio_votes = np.zeros(8)
    
    for i in range(n_samples):
        # Convert predictions to a pattern index (0-7)
        pattern_idx = pred_a[i] * 4 + pred_b[i] * 2 + pred_c[i]
        trio_votes[pattern_idx] += 1
    
    # Create trio counts
    trio_counts = TrioVoteCounts(trio_votes)
    
    # Create label vote counts
    # Initialize 2×8 array (rows for true labels 0,1; columns for 8 patterns)
    label_vote_counts = np.zeros((2, 8))
    
    for i in range(n_samples):
        true_label = ground_truth[i]
        pattern_idx = pred_a[i] * 4 + pred_b[i] * 2 + pred_c[i]
        label_vote_counts[true_label, pattern_idx] += 1
    
    # Create trio label counts
    trio_label_counts = TrioLabelVoteCounts(label_vote_counts)
    
    return {
        'ground_truth': ground_truth,
        'predictions': {
            'A': pred_a,
            'B': pred_b,
            'C': pred_c
        },
        'binary_data': {
            'A': binary_data_a,
            'B': binary_data_b,
            'C': binary_data_c
        },
        'trio_votes': trio_votes,
        'trio_counts': trio_counts,
        'label_vote_counts': label_vote_counts,
        'trio_label_counts': trio_label_counts
    }


def demo_main_package_basic_plots(data, output_dir, args):
    """Demonstrate basic visualization from the main package."""
    print("\n=== Basic Plots from Main Package ===")
    
    # Create figure for evaluation space
    fig = main_plot_evaluation_space(title="NTQR Evaluation Space")
    save_figure(fig, "main_evaluation_space.png", output_dir, dpi=args.dpi, show=args.show_plots)
    
    # Add classifier points to evaluation space
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Create evaluator
    evaluator = SupervisedEvaluation()
    
    # Evaluate each classifier
    results = {}
    for name, binary_data in data['binary_data'].items():
        results[name] = evaluator.evaluate(binary_data)
    
    # Plot the unit square (valid accuracy range)
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, linestyle='-', 
                              linewidth=2, color='black'))
    
    # Add diagonal line (representing p0 + p1 = 1)
    ax.plot([0, 1], [1, 0], 'r--', label='p₀ + p₁ = 1')
    
    # Add points for each classifier
    colors = ['blue', 'green', 'purple']
    for i, (name, result) in enumerate(results.items()):
        ax.scatter([result.accuracy_p0], [result.accuracy_p1], color=colors[i], s=100, 
                  marker='o', label=f'Classifier {name}')
    
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
    
    save_figure(fig, "classifier_evaluation_space.png", output_dir, dpi=args.dpi, show=args.show_plots)
    
    # Create a voting distribution plot
    fig, ax = plt.subplots(figsize=(10, 6))
    pattern_labels = ['000', '001', '010', '011', '100', '101', '110', '111']
    ax.bar(pattern_labels, data['trio_votes'])
    ax.set_title('Distribution of Voting Patterns')
    ax.set_xlabel('Voting Pattern (A,B,C)')
    ax.set_ylabel('Count')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add color coded annotations for majority vote
    for i, count in enumerate(data['trio_votes']):
        majority = '0' if pattern_labels[i].count('1') < 2 else '1'
        color = 'blue' if majority == '0' else 'red'
        ax.text(i, count + 1, f'Maj: {majority}', ha='center', color=color)
    
    save_figure(fig, "voting_patterns.png", output_dir, dpi=args.dpi, show=args.show_plots)


def demo_enhanced_visualizations(data, output_dir, args):
    """Demonstrate enhanced visualizations from the fork."""
    print("\n=== Enhanced Visualizations from Fork ===")
    
    # Create evaluator
    evaluator = SupervisedEvaluation()
    
    # Evaluate each classifier
    results = {}
    for name, binary_data in data['binary_data'].items():
        results[name] = evaluator.evaluate(binary_data)
    
    # 1. Comparative evaluation plot
    evaluations = [(f"Classifier {name}", result) for name, result in results.items()]
    
    fig = plot_comparative_evaluation(
        evaluations=evaluations,
        title="Comparison of Classifier Performance",
        colors=['#3498db', '#2ecc71', '#9b59b6'],  # Custom colors
    )
    save_figure(fig, "comparative_evaluation.png", output_dir, dpi=args.dpi, show=args.show_plots)
    
    # 2. Trio agreement matrix
    fig = plot_trio_agreement_matrix(
        trio_data=data['trio_votes'],
        classifier_names=["Classifier A", "Classifier B", "Classifier C"],
        title="Agreement Patterns Among Classifiers",
        cmap='viridis',
    )
    save_figure(fig, "agreement_matrix.png", output_dir, dpi=args.dpi, show=args.show_plots)
    
    # 3. Error-independent evaluation
    ei_evaluator = ErrorIndependentEvaluation()
    ei_result = ei_evaluator.evaluate(data['trio_label_counts'])
    
    # 4. Majority voting evaluation
    mv_evaluator = MajorityVotingEvaluation()
    mv_result = mv_evaluator.evaluate(data['trio_label_counts'])
    
    # 5. Evaluation confidence plots
    for i, classifier_result in enumerate(ei_result.classifier_accuracies):
        fig = plot_evaluation_confidence(
            evaluation_result=classifier_result,
            title=f"Confidence in Error-Independent Evaluation (Classifier {chr(65+i)})",
            confidence_range=(0.85, 0.99),
        )
        save_figure(fig, f"confidence_classifier_{chr(65+i)}.png", output_dir, dpi=args.dpi, show=args.show_plots)
    
    # Majority voting confidence
    fig = plot_evaluation_confidence(
        evaluation_result=mv_result,
        title="Confidence in Majority Voting Evaluation",
        confidence_range=(0.9, 0.99),
    )
    save_figure(fig, "confidence_majority_voting.png", output_dir, dpi=args.dpi, show=args.show_plots)


def create_animated_evaluation_space(data, output_dir, args):
    """Create an animated visualization of the evaluation space."""
    try:
        from matplotlib.animation import FuncAnimation
        print("\n=== Creating Animated Evaluation Space ===")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Initial setup
        # Plot the unit square (valid accuracy range)
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, linestyle='-', 
                                  linewidth=2, color='black'))
        
        # Add diagonal line (representing p0 + p1 = 1)
        ax.plot([0, 1], [1, 0], 'r--', label='p₀ + p₁ = 1')
        
        # Add reference points
        ax.scatter([0.5], [0.5], color='red', s=80, marker='x', label='Random Classifier')
        ax.scatter([1], [1], color='green', s=80, marker='x', label='Perfect Classifier')
        
        # Set labels and title
        ax.set_xlabel('Accuracy on Label 0 (p₀)')
        ax.set_ylabel('Accuracy on Label 1 (p₁)')
        ax.set_title('Animated Evaluation Space')
        
        # Set limits with some padding
        ax.set_xlim([-0.05, 1.05])
        ax.set_ylim([-0.05, 1.05])
        
        # Add grid and legend
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='upper right')
        
        # Generate a sequence of points for the animation
        num_frames = 60
        
        # Define starting points (p0, p1) for three classifiers
        start_points = [
            (0.5, 0.5),  # Start at random classifier
            (0.5, 0.5),  # Start at random classifier
            (0.5, 0.5),  # Start at random classifier
        ]
        
        # Define end points from our data
        evaluator = SupervisedEvaluation()
        end_points = []
        for name, binary_data in data['binary_data'].items():
            result = evaluator.evaluate(binary_data)
            end_points.append((result.accuracy_p0, result.accuracy_p1))
        
        # Create scatter plot elements
        scatter_plots = []
        colors = ['blue', 'green', 'purple']
        for i, (start, color) in enumerate(zip(start_points, colors)):
            scatter = ax.scatter([start[0]], [start[1]], color=color, s=100, 
                               marker='o', label=f'Classifier {chr(65+i)}')
            scatter_plots.append(scatter)
        
        # Create animation function
        def update(frame):
            for i, (scatter, start, end) in enumerate(zip(scatter_plots, start_points, end_points)):
                # Interpolate between start and end points
                t = frame / (num_frames - 1)  # Normalize to [0, 1]
                p0 = start[0] + t * (end[0] - start[0])
                p1 = start[1] + t * (end[1] - start[1])
                
                # Update position
                scatter.set_offsets([(p0, p1)])
            
            return scatter_plots
            
        # Create animation
        animation = FuncAnimation(fig, update, frames=num_frames, interval=100, blit=True)
        
        # Save animation
        try:
            animation.save(output_dir / "animated_evaluation_space.gif", writer='pillow', fps=10, dpi=args.dpi)
            print(f"Saved: {output_dir / 'animated_evaluation_space.gif'}")
        except Exception as e:
            print(f"Warning: Could not save animation as GIF: {e}")
            print("Saving as MP4 instead...")
            try:
                animation.save(output_dir / "animated_evaluation_space.mp4", writer='ffmpeg', fps=10, dpi=args.dpi)
                print(f"Saved: {output_dir / 'animated_evaluation_space.mp4'}")
            except Exception as e:
                print(f"Error: Could not save animation: {e}")
        
        plt.close(fig)
        
    except ImportError:
        print("Warning: matplotlib.animation not available. Skipping animation.")


def create_heatmap_visualizations(data, output_dir, args):
    """Create heatmap visualizations of classifier performance."""
    print("\n=== Creating Heatmap Visualizations ===")
    
    # Calculate accuracy per label for all classifiers
    accuracy_data = np.zeros((3, 2))  # 3 classifiers, 2 class labels
    
    for idx, (name, preds) in enumerate(data['predictions'].items()):
        for label in [0, 1]:
            mask = data['ground_truth'] == label
            if np.sum(mask) > 0:  # Avoid division by zero
                accuracy_data[idx, label] = np.mean(preds[mask] == label)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(accuracy_data, cmap='viridis', vmin=0, vmax=1)
    
    # Add labels
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Class 0', 'Class 1'])
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(['Classifier A', 'Classifier B', 'Classifier C'])
    ax.set_title('Classifier Accuracy by Class')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Accuracy')
    
    # Add text annotations
    for i in range(3):
        for j in range(2):
            ax.text(j, i, f'{accuracy_data[i, j]:.2f}', 
                   ha='center', va='center', color='white')
    
    # Add overall accuracy values
    overall_acc = [
        np.mean(data['predictions']['A'] == data['ground_truth']),
        np.mean(data['predictions']['B'] == data['ground_truth']),
        np.mean(data['predictions']['C'] == data['ground_truth'])
    ]
    
    # Add text annotations for overall accuracy
    for i, acc in enumerate(overall_acc):
        ax.text(-0.3, i, f'Overall: {acc:.2f}', ha='right', va='center')
    
    save_figure(fig, "accuracy_heatmap.png", output_dir, dpi=args.dpi, show=args.show_plots)
    
    # Create pairwise agreement heatmap
    pairwise_agreement = np.zeros((3, 3))
    classifiers = list(data['predictions'].values())
    
    for i in range(3):
        for j in range(3):
            pairwise_agreement[i, j] = np.mean(classifiers[i] == classifiers[j])
    
    # Create heatmap
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
    
    save_figure(fig, "pairwise_agreement_heatmap.png", output_dir, dpi=args.dpi, show=args.show_plots)


def create_summary_report(data, output_dir):
    """Create a summary report with key metrics."""
    print("\n=== Creating Summary Report ===")
    
    evaluator = SupervisedEvaluation()
    ei_evaluator = ErrorIndependentEvaluation()
    mv_evaluator = MajorityVotingEvaluation()
    
    # Evaluate each classifier
    results = {}
    for name, binary_data in data['binary_data'].items():
        results[name] = evaluator.evaluate(binary_data)
    
    # Error-independent evaluation
    ei_result = ei_evaluator.evaluate(data['trio_label_counts'])
    
    # Majority voting evaluation
    mv_result = mv_evaluator.evaluate(data['trio_label_counts'])
    
    # Calculate agreement statistics
    classifiers = list(data['predictions'].values())
    pairwise_agreement = {
        'AB': np.mean(classifiers[0] == classifiers[1]),
        'AC': np.mean(classifiers[0] == classifiers[2]),
        'BC': np.mean(classifiers[1] == classifiers[2])
    }
    unanimous_agreement = np.mean((classifiers[0] == classifiers[1]) & 
                                 (classifiers[1] == classifiers[2]))
    
    # Generate report
    report_path = output_dir / "evaluation_summary.txt"
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("NTQR Evaluation Summary Report\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("1. Individual Classifier Performance\n")
        f.write("-" * 40 + "\n")
        for name, result in results.items():
            f.write(f"Classifier {name}:\n")
            f.write(f"  Accuracy on Class 0: {result.accuracy_p0:.4f}\n")
            f.write(f"  Accuracy on Class 1: {result.accuracy_p1:.4f}\n")
            f.write(f"  Overall Accuracy:    {result.accuracy:.4f}\n\n")
        
        f.write("2. Agreement Statistics\n")
        f.write("-" * 40 + "\n")
        f.write(f"Pairwise Agreement:\n")
        f.write(f"  A-B: {pairwise_agreement['AB']:.4f}\n")
        f.write(f"  A-C: {pairwise_agreement['AC']:.4f}\n")
        f.write(f"  B-C: {pairwise_agreement['BC']:.4f}\n")
        f.write(f"Unanimous Agreement: {unanimous_agreement:.4f}\n\n")
        
        f.write("3. Error-Independent Evaluation\n")
        f.write("-" * 40 + "\n")
        for i, result in enumerate(ei_result.classifier_accuracies):
            f.write(f"Classifier {chr(65+i)}:\n")
            f.write(f"  Accuracy on Class 0: {result.accuracy_p0:.4f}\n")
            f.write(f"  Accuracy on Class 1: {result.accuracy_p1:.4f}\n")
            f.write(f"  Overall Accuracy:    {result.accuracy:.4f}\n\n")
        
        f.write("4. Majority Voting Evaluation\n")
        f.write("-" * 40 + "\n")
        f.write(f"Accuracy on Class 0: {mv_result.accuracy_p0:.4f}\n")
        f.write(f"Accuracy on Class 1: {mv_result.accuracy_p1:.4f}\n")
        f.write(f"Overall Accuracy:    {mv_result.accuracy:.4f}\n\n")
        
        f.write("5. Voting Pattern Distribution\n")
        f.write("-" * 40 + "\n")
        pattern_labels = ['000', '001', '010', '011', '100', '101', '110', '111']
        for pattern, count in zip(pattern_labels, data['trio_votes']):
            f.write(f"Pattern {pattern}: {int(count)} votes\n")
        
        f.write("\nReport generated by NTQR fork visualization tools\n")
    
    print(f"Saved: {report_path}")


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
    
    # Run demos
    demo_main_package_basic_plots(data, output_dir, args)
    demo_enhanced_visualizations(data, output_dir, args)
    create_animated_evaluation_space(data, output_dir, args)
    create_heatmap_visualizations(data, output_dir, args)
    create_summary_report(data, output_dir)
    
    print("\n" + "=" * 80)
    print("Demonstration complete!")
    print(f"All outputs saved to: {output_dir}")
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 