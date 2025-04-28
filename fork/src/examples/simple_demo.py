#!/usr/bin/env python3
"""
Simple demonstration of algebraic evaluation concepts.

This script demonstrates the basic concepts of algebraic evaluation
with simple, self-contained examples that don't rely on the NTQR package.
"""

import numpy as np
import matplotlib.pyplot as plt


def generate_synthetic_data(n_samples=100, accuracy_a=0.8, accuracy_b=0.7, accuracy_c=0.75):
    """
    Generate synthetic data for three classifiers (A, B, C) evaluating binary classification.
    
    Parameters
    ----------
    n_samples : int
        Number of samples to generate
    accuracy_a, accuracy_b, accuracy_c : float
        Accuracy of each classifier
        
    Returns
    -------
    tuple
        Ground truth, predictions from A, B, C
    """
    # Generate ground truth (0 or 1)
    ground_truth = np.random.randint(0, 2, size=n_samples)
    
    # Generate predictions with specified accuracy
    def generate_predictions(true_labels, accuracy):
        predictions = np.copy(true_labels)
        # Flip some predictions according to accuracy
        flip_mask = np.random.random(len(true_labels)) > accuracy
        predictions[flip_mask] = 1 - predictions[flip_mask]
        return predictions
    
    predictions_a = generate_predictions(ground_truth, accuracy_a)
    predictions_b = generate_predictions(ground_truth, accuracy_b)
    predictions_c = generate_predictions(ground_truth, accuracy_c)
    
    return ground_truth, predictions_a, predictions_b, predictions_c


def calculate_agreement_statistics(pred_a, pred_b, pred_c):
    """
    Calculate agreement statistics for three classifiers.
    
    Parameters
    ----------
    pred_a, pred_b, pred_c : np.ndarray
        Predictions from classifiers A, B, C
        
    Returns
    -------
    dict
        Agreement statistics
    """
    n_samples = len(pred_a)
    
    # Calculate pairwise agreement
    agreement_ab = np.mean(pred_a == pred_b)
    agreement_ac = np.mean(pred_a == pred_c)
    agreement_bc = np.mean(pred_b == pred_c)
    
    # Calculate unanimous agreement (all agree)
    unanimous_agreement = np.mean((pred_a == pred_b) & (pred_b == pred_c))
    
    # Calculate majority voting accuracy
    majority_votes = (pred_a + pred_b + pred_c > 1.5).astype(int)
    
    return {
        'pairwise_agreement': {
            'AB': agreement_ab,
            'AC': agreement_ac,
            'BC': agreement_bc
        },
        'unanimous_agreement': unanimous_agreement,
        'majority_votes': majority_votes
    }


def plot_evaluation_space():
    """
    Plot the evaluation space for binary classifiers.
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot the unit square (valid accuracy range)
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, linestyle='-', 
                              linewidth=2, color='black'))
    
    # Add diagonal line (representing p0 + p1 = 1)
    ax.plot([0, 1], [1, 0], 'r--', label='p₀ + p₁ = 1')
    
    # Add reference points
    # Random classifier (0.5, 0.5)
    ax.scatter([0.5], [0.5], color='red', s=100, 
              marker='o', label='Random Classifier')
    
    # Perfect classifier (1, 1)
    ax.scatter([1], [1], color='green', s=100, 
              marker='o', label='Perfect Classifier')
    
    # Biased classifier examples
    ax.scatter([0.9], [0.6], color='blue', s=100, 
              marker='o', label='Good Classifier')
    ax.scatter([0.3], [0.8], color='purple', s=100, 
              marker='o', label='Biased Classifier')
    
    # Add labels and title
    ax.set_xlabel('Accuracy on Label 0 (p₀)')
    ax.set_ylabel('Accuracy on Label 1 (p₁)')
    ax.set_title('Evaluation Space for Binary Classifiers')
    
    # Set limits with some padding
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    
    # Add grid and legend
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    
    # Add regions with labels
    ax.text(0.25, 0.25, 'Poor', ha='center', fontsize=12)
    ax.text(0.75, 0.25, 'Biased to 0', ha='center', fontsize=12)
    ax.text(0.25, 0.75, 'Biased to 1', ha='center', fontsize=12)
    ax.text(0.75, 0.75, 'Good', ha='center', fontsize=12)
    
    return fig


def plot_trio_agreement_patterns(ground_truth, pred_a, pred_b, pred_c):
    """
    Plot agreement patterns among three classifiers.
    
    Parameters
    ----------
    ground_truth : np.ndarray
        Ground truth labels
    pred_a, pred_b, pred_c : np.ndarray
        Predictions from classifiers A, B, C
        
    Returns
    -------
    fig
        Matplotlib figure
    """
    # Create figure
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    
    # Create voting patterns array (8 possible patterns)
    patterns = np.zeros(8)
    
    # Count occurrences of each pattern
    for i in range(len(ground_truth)):
        # Convert the three predictions to a binary index
        pattern_idx = int(pred_a[i]) * 4 + int(pred_b[i]) * 2 + int(pred_c[i])
        patterns[pattern_idx] += 1
    
    # Plot 1: Voting pattern counts
    pattern_labels = ['000', '001', '010', '011', '100', '101', '110', '111']
    axs[0].bar(pattern_labels, patterns)
    axs[0].set_title('Voting Pattern Distribution')
    axs[0].set_xlabel('Voting Pattern (A,B,C)')
    axs[0].set_ylabel('Count')
    axs[0].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add color coded annotations for majority
    for i, p in enumerate(patterns):
        majority = '0' if pattern_labels[i].count('1') < 2 else '1'
        color = 'blue' if majority == '0' else 'red'
        axs[0].text(i, p + 1, f'Maj: {majority}', ha='center', color=color)
    
    # Plot 2: Agreement heatmap
    accuracy_data = np.zeros((3, 2))  # 3 classifiers, 2 class labels
    
    # Calculate accuracy for each classifier and class
    for i, preds in enumerate([pred_a, pred_b, pred_c]):
        for label in [0, 1]:
            mask = ground_truth == label
            if np.sum(mask) > 0:  # Avoid division by zero
                accuracy_data[i, label] = np.mean(preds[mask] == label)
    
    # Plot heatmap
    im = axs[1].imshow(accuracy_data, cmap='viridis', vmin=0, vmax=1)
    axs[1].set_xticks([0, 1])
    axs[1].set_xticklabels(['Class 0', 'Class 1'])
    axs[1].set_yticks([0, 1, 2])
    axs[1].set_yticklabels(['Classifier A', 'Classifier B', 'Classifier C'])
    axs[1].set_title('Classifier Accuracy by Class')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=axs[1])
    cbar.set_label('Accuracy')
    
    # Add text annotations
    for i in range(3):
        for j in range(2):
            axs[1].text(j, i, f'{accuracy_data[i, j]:.2f}', 
                       ha='center', va='center', color='white')
    
    # Add overall accuracy values
    overall_acc = [
        np.mean(pred_a == ground_truth),
        np.mean(pred_b == ground_truth),
        np.mean(pred_c == ground_truth)
    ]
    
    # Add text above the heatmap
    for i, acc in enumerate(overall_acc):
        axs[1].text(-0.3, i, f'Overall: {acc:.2f}', ha='right', va='center')
    
    # Adjust layout
    plt.tight_layout()
    
    return fig


def main():
    """
    Run the demonstration.
    """
    # Create output directory
    import os
    from pathlib import Path
    
    output_dir = Path(__file__).parent.parent.parent / "output"
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("Simple Algebraic Evaluation Demonstration")
    print("=" * 80)
    print(f"Output will be saved to: {output_dir}")
    
    # Generate synthetic data
    np.random.seed(42)  # For reproducibility
    ground_truth, pred_a, pred_b, pred_c = generate_synthetic_data(
        n_samples=200, 
        accuracy_a=0.85, 
        accuracy_b=0.75, 
        accuracy_c=0.80
    )
    
    # Calculate agreement statistics
    stats = calculate_agreement_statistics(pred_a, pred_b, pred_c)
    
    # Display statistics
    print("\nAgreement Statistics:")
    print(f"Pairwise Agreement:")
    print(f"  A-B: {stats['pairwise_agreement']['AB']:.2f}")
    print(f"  A-C: {stats['pairwise_agreement']['AC']:.2f}")
    print(f"  B-C: {stats['pairwise_agreement']['BC']:.2f}")
    print(f"Unanimous Agreement: {stats['unanimous_agreement']:.2f}")
    
    # Calculate actual accuracies
    acc_a = np.mean(pred_a == ground_truth)
    acc_b = np.mean(pred_b == ground_truth)
    acc_c = np.mean(pred_c == ground_truth)
    acc_majority = np.mean(stats['majority_votes'] == ground_truth)
    
    print("\nActual Accuracies:")
    print(f"  Classifier A: {acc_a:.2f}")
    print(f"  Classifier B: {acc_b:.2f}")
    print(f"  Classifier C: {acc_c:.2f}")
    print(f"  Majority Vote: {acc_majority:.2f}")
    
    # Plot evaluation space
    fig1 = plot_evaluation_space()
    fig1.savefig(output_dir / "evaluation_space.png", dpi=300, bbox_inches='tight')
    print(f"\nSaved evaluation space plot to: {output_dir / 'evaluation_space.png'}")
    
    # Plot trio agreement patterns
    fig2 = plot_trio_agreement_patterns(ground_truth, pred_a, pred_b, pred_c)
    fig2.savefig(output_dir / "trio_agreement.png", dpi=300, bbox_inches='tight')
    print(f"Saved trio agreement plot to: {output_dir / 'trio_agreement.png'}")
    
    # Close figures
    plt.close(fig1)
    plt.close(fig2)
    
    print("\nDemonstration complete.")
    
    return 0


if __name__ == "__main__":
    main() 