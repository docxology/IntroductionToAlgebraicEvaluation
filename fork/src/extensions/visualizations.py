"""
Enhanced visualization tools for algebraic evaluation.

This module extends the plotting capabilities of the main NTQR package with
additional visualization tools.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from typing import List, Optional, Tuple, Union, Dict, Any

# Add project root to path if needed
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Try direct imports first
try:
    from python.src.ntqr.plots import plot_evaluation_space
    from python.src.ntqr.r2.evaluators import (
        SupervisedEvaluation,
        ErrorIndependentEvaluation, 
        MajorityVotingEvaluation,
    )
except ImportError:
    # If direct imports fail, use relative imports with adjusted path
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
    
    # Now try the imports again
    from python.src.ntqr.plots import plot_evaluation_space
    from python.src.ntqr.r2.evaluators import (
        SupervisedEvaluation,
        ErrorIndependentEvaluation,
        MajorityVotingEvaluation,
    )


def plot_comparative_evaluation(
    evaluations: List[Tuple[str, Any]],
    figsize: Tuple[int, int] = (10, 6),
    colors: Optional[List[str]] = None,
    title: str = "Comparative Evaluation",
    show_legend: bool = True,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot a comparative visualization of multiple evaluation results.
    
    Parameters
    ----------
    evaluations : List[Tuple[str, Any]]
        List of (name, evaluation_result) tuples to compare
        Each evaluation_result should have accuracy_p0 and accuracy_p1 attributes
    figsize : Tuple[int, int], optional
        Figure size, by default (10, 6)
    colors : Optional[List[str]], optional
        List of colors for each evaluation, by default None (uses default colormap)
    title : str, optional
        Plot title, by default "Comparative Evaluation"
    show_legend : bool, optional
        Whether to show legend, by default True
    save_path : Optional[str], optional
        Path to save figure, by default None (doesn't save)
        
    Returns
    -------
    plt.Figure
        The generated figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Default colors if not provided
    if colors is None:
        cm = plt.cm.get_cmap('viridis', len(evaluations))
        colors = [cm(i) for i in range(len(evaluations))]
    
    # Plot individual bars for p0 and p1 accuracies
    bar_width = 0.35
    x = np.arange(2)  # 2 metrics: p0 and p1
    
    for i, (name, eval_result) in enumerate(evaluations):
        offset = (i - len(evaluations)/2 + 0.5) * bar_width
        ax.bar(x + offset, [eval_result.accuracy_p0, eval_result.accuracy_p1], 
               width=bar_width, label=name, color=colors[i], alpha=0.7)
    
    # Add labels and title
    ax.set_xticks(x)
    ax.set_xticklabels(['Accuracy on Label 0', 'Accuracy on Label 1'])
    ax.set_ylabel('Accuracy')
    ax.set_title(title)
    
    # Add a horizontal line at y=0.5 (random guessing)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random Guessing')
    
    # Add a horizontal line at y=1.0 (perfect accuracy)
    ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Perfect Accuracy')
    
    # Set y-axis limits with some padding
    ax.set_ylim([0, 1.05])
    
    # Add legend if requested
    if show_legend:
        ax.legend()
    
    # Add grid for better readability
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_trio_agreement_matrix(
    trio_data: np.ndarray,
    classifier_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = 'Blues',
    title: str = "Trio Agreement Matrix",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot a matrix visualization of agreement patterns among a trio of classifiers.
    
    Parameters
    ----------
    trio_data : np.ndarray
        Array of vote counts with 8 elements [n_000, n_001, ..., n_111]
    classifier_names : Optional[List[str]], optional
        Names of the three classifiers, by default None (uses "Classifier 1/2/3")
    figsize : Tuple[int, int], optional
        Figure size, by default (10, 8)
    cmap : str, optional
        Colormap for the matrix, by default 'Blues'
    title : str, optional
        Plot title, by default "Trio Agreement Matrix"
    save_path : Optional[str], optional
        Path to save figure, by default None (doesn't save)
        
    Returns
    -------
    plt.Figure
        The generated figure
    """
    # Default classifier names
    if classifier_names is None:
        classifier_names = [f"Classifier {i+1}" for i in range(3)]
    
    # Create figure
    fig, axs = plt.subplots(2, 2, figsize=figsize)
    
    # Reshape trio data to visualize patterns
    # For each classifier, show how it agrees with combinations of the others
    
    # Classifier 1 vs (2,3)
    # (0 = all patterns where C1 voted 0, 1 = all patterns where C1 voted 1)
    c1_data = np.zeros((2, 4))
    c1_data[0, 0] = trio_data[0]  # 000
    c1_data[0, 1] = trio_data[1]  # 001
    c1_data[0, 2] = trio_data[2]  # 010
    c1_data[0, 3] = trio_data[3]  # 011
    c1_data[1, 0] = trio_data[4]  # 100
    c1_data[1, 1] = trio_data[5]  # 101
    c1_data[1, 2] = trio_data[6]  # 110
    c1_data[1, 3] = trio_data[7]  # 111
    
    # Classifier 2 vs (1,3)
    c2_data = np.zeros((2, 4))
    c2_data[0, 0] = trio_data[0]  # 000
    c2_data[0, 1] = trio_data[1]  # 001
    c2_data[0, 2] = trio_data[4]  # 100
    c2_data[0, 3] = trio_data[5]  # 101
    c2_data[1, 0] = trio_data[2]  # 010
    c2_data[1, 1] = trio_data[3]  # 011
    c2_data[1, 2] = trio_data[6]  # 110
    c2_data[1, 3] = trio_data[7]  # 111
    
    # Classifier 3 vs (1,2)
    c3_data = np.zeros((2, 4))
    c3_data[0, 0] = trio_data[0]  # 000
    c3_data[0, 1] = trio_data[2]  # 010
    c3_data[0, 2] = trio_data[4]  # 100
    c3_data[0, 3] = trio_data[6]  # 110
    c3_data[1, 0] = trio_data[1]  # 001
    c3_data[1, 1] = trio_data[3]  # 011
    c3_data[1, 2] = trio_data[5]  # 101
    c3_data[1, 3] = trio_data[7]  # 111
    
    # Overall agreement matrix
    # This shows counts for each voting pattern
    agreement_matrix = np.zeros((2, 4))
    for i in range(8):
        # Convert index to binary representation
        binary = format(i, '03b')
        # Count of 1s in the pattern (0, 1, 2, or 3)
        count_ones = binary.count('1')
        # Add to matrix
        agreement_matrix[0 if count_ones < 2 else 1, count_ones] += trio_data[i]
    
    # Plot matrices
    matrices = [c1_data, c2_data, c3_data, agreement_matrix]
    titles = [
        f"{classifier_names[0]} vs. ({classifier_names[1]}, {classifier_names[2]})",
        f"{classifier_names[1]} vs. ({classifier_names[0]}, {classifier_names[2]})",
        f"{classifier_names[2]} vs. ({classifier_names[0]}, {classifier_names[1]})",
        "Overall Agreement"
    ]
    
    for i, (ax, mat, subtitle) in enumerate(zip(axs.flat, matrices, titles)):
        im = ax.imshow(mat, cmap=cmap)
        ax.set_title(subtitle)
        
        # Add text annotations
        for row in range(mat.shape[0]):
            for col in range(mat.shape[1]):
                text_color = 'white' if mat[row, col] > np.max(mat) / 2 else 'black'
                ax.text(col, row, f'{int(mat[row, col])}', 
                        ha='center', va='center', color=text_color)
        
        # Set custom labels for each subplot
        if i < 3:  # Classifier vs others matrices
            ax.set_xticks(range(4))
            ax.set_xticklabels(['00', '01', '10', '11'])
            ax.set_yticks([0, 1])
            ax.set_yticklabels(['0', '1'])
            ax.set_xlabel(f"Votes from other classifiers")
            ax.set_ylabel(f"Vote from {classifier_names[i]}")
        else:  # Agreement matrix
            ax.set_xticks(range(4))
            ax.set_xticklabels(['0', '1', '2', '3'])
            ax.set_yticks([0, 1])
            ax.set_yticklabels(['Majority 0', 'Majority 1'])
            ax.set_xlabel("Number of classifiers voting 1")
            ax.set_ylabel("Majority vote")
    
    # Add colorbar
    fig.colorbar(im, ax=axs.ravel().tolist())
    
    # Set overall title
    fig.suptitle(title, fontsize=16)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_evaluation_confidence(
    evaluation_result: Any,
    confidence_range: Tuple[float, float] = (0.9, 0.99),
    num_samples: int = 1000,
    figsize: Tuple[int, int] = (10, 6),
    title: str = "Evaluation Confidence",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot a visualization of evaluation confidence based on Monte Carlo sampling.
    
    Parameters
    ----------
    evaluation_result : Any
        Evaluation result to analyze with accuracy_p0 and accuracy_p1 attributes
    confidence_range : Tuple[float, float], optional
        Range of confidence levels to visualize, by default (0.9, 0.99)
    num_samples : int, optional
        Number of Monte Carlo samples, by default 1000
    figsize : Tuple[int, int], optional
        Figure size, by default (10, 6)
    title : str, optional
        Plot title, by default "Evaluation Confidence"
    save_path : Optional[str], optional
        Path to save figure, by default None (doesn't save)
        
    Returns
    -------
    plt.Figure
        The generated figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get the point estimates
    p0 = evaluation_result.accuracy_p0
    p1 = evaluation_result.accuracy_p1
    
    # Generate random samples around the evaluation result
    # This is a simplified simulation of uncertainty
    # In practice, this would be based on the statistical properties of the evaluation method
    np.random.seed(42)  # For reproducibility
    
    # Standard deviations decrease with sample size
    # Here we use arbitrary small values for demonstration
    std_p0 = 0.05
    std_p1 = 0.05
    
    # Generate samples
    p0_samples = np.random.normal(p0, std_p0, num_samples)
    p1_samples = np.random.normal(p1, std_p1, num_samples)
    
    # Clip to valid range [0, 1]
    p0_samples = np.clip(p0_samples, 0, 1)
    p1_samples = np.clip(p1_samples, 0, 1)
    
    # Calculate confidence ellipses
    from matplotlib.patches import Ellipse
    import matplotlib.transforms as transforms
    
    def confidence_ellipse(x, y, ax, n_std=3.0, **kwargs):
        """
        Create a plot of the covariance confidence ellipse of *x* and *y*.
        """
        cov = np.cov(x, y)
        pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
        
        # Using a special case to obtain the eigenvalues of this
        # two-dimensional dataset.
        ell_radius_x = np.sqrt(1 + pearson)
        ell_radius_y = np.sqrt(1 - pearson)
        ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, **kwargs)
        
        # Calculating the standard deviation of x from the matrix
        scale_x = np.sqrt(cov[0, 0]) * n_std
        mean_x = np.mean(x)
        
        # Calculating the standard deviation of y from the matrix
        scale_y = np.sqrt(cov[1, 1]) * n_std
        mean_y = np.mean(y)
        
        transf = transforms.Affine2D() \
            .rotate_deg(45) \
            .scale(scale_x, scale_y) \
            .translate(mean_x, mean_y)
        
        ellipse.set_transform(transf + ax.transData)
        return ax.add_patch(ellipse)
    
    # Plot the samples as a scatter plot with transparency
    ax.scatter(p0_samples, p1_samples, alpha=0.1, s=5, color='blue')
    
    # Plot confidence ellipses for different confidence levels
    confidence_levels = np.linspace(confidence_range[0], confidence_range[1], 5)
    for i, conf in enumerate(confidence_levels):
        # Convert confidence level to number of standard deviations
        n_std = np.sqrt(2 * np.log(1 / (1 - conf)))
        alpha = 0.2 + 0.6 * i / len(confidence_levels)
        confidence_ellipse(p0_samples, p1_samples, ax, n_std=n_std, 
                          edgecolor='red', facecolor='none', alpha=alpha, 
                          label=f'{conf:.0%} Confidence')
    
    # Plot the point estimate
    ax.scatter([p0], [p1], color='red', s=100, marker='x', label='Point Estimate')
    
    # Add diagonal line (p0 + p1 = 1)
    ax.plot([0, 1], [1, 0], 'k--', alpha=0.3, label='p₀ + p₁ = 1')
    
    # Add reference lines for random and perfect classifiers
    ax.plot([0.5, 0.5], [0, 1], 'b--', alpha=0.3)
    ax.plot([0, 1], [0.5, 0.5], 'b--', alpha=0.3, label='Random Guessing')
    ax.plot([1, 1], [0, 1], 'g--', alpha=0.3)
    ax.plot([0, 1], [1, 1], 'g--', alpha=0.3, label='Perfect Classifier')
    
    # Set axis limits, labels, and title
    ax.set_xlim([0, 1.05])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel('Accuracy on Label 0 (p₀)')
    ax.set_ylabel('Accuracy on Label 1 (p₁)')
    ax.set_title(title)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add legend
    ax.legend(loc='upper right')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig 