"""
Advanced visualization utilities for the NTQR fork.

This module provides enhanced visualization tools for algebraic evaluation results,
with an emphasis on intuitive visual representations of complex algebraic constraints
and evaluation results.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches
from typing import List, Optional, Tuple, Union, Dict, Any, Callable
from pathlib import Path
import sys

# Try to import from main package
try:
    from python.src.ntqr.plots import plot_evaluation_space as base_plot_evaluation_space
except ImportError:
    # Fallback implementation if main package is not available
    def base_plot_evaluation_space(*args, **kwargs):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.plot([0, 1], [1, 0], 'k--', label='p₀ + p₁ = 1')
        ax.set_xlabel('p₀ (Accuracy on class 0)')
        ax.set_ylabel('p₁ (Accuracy on class 1)')
        ax.grid(True, alpha=0.3)
        return fig


def enhanced_evaluation_space(
    results: Any,
    title: str = "Evaluation Space",
    highlight_constraints: bool = True,
    show_confidence: bool = True,
    confidence_level: float = 0.95,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create an enhanced visualization of the evaluation space with constraints.
    
    Parameters
    ----------
    results : Any
        Evaluation results object with accuracy attributes
    title : str, optional
        Plot title, by default "Evaluation Space"
    highlight_constraints : bool, optional
        Whether to highlight constraint intersections, by default True
    show_confidence : bool, optional
        Whether to show confidence regions, by default True
    confidence_level : float, optional
        Confidence level for regions (0-1), by default 0.95
    figsize : Tuple[int, int], optional
        Figure size, by default (10, 8)
    save_path : Optional[str], optional
        Path to save the figure, by default None (doesn't save)
        
    Returns
    -------
    plt.Figure
        The generated figure
    """
    # Create base evaluation space plot
    fig = base_plot_evaluation_space()
    
    # Get the axes from the figure
    ax = fig.axes[0]
    
    # Update figure size
    fig.set_size_inches(figsize)
    
    # Add title with larger font size for better visibility
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add a grid for better readability
    ax.grid(True, linestyle='--', alpha=0.7, which='both')
    
    # Plot each classifier's accuracy point with larger markers for visibility
    if hasattr(results, 'accuracies'):
        for i, acc in enumerate(results.accuracies):
            if hasattr(results, 'accuracy_p0') and hasattr(results, 'accuracy_p1'):
                p0 = results.accuracy_p0
                p1 = results.accuracy_p1
                ax.scatter(p0, p1, s=150, color=f'C{i}', edgecolor='black', linewidth=1, 
                          label=f'Classifier {i+1}', zorder=10)
            else:
                # Assume balanced accuracy
                ax.scatter(acc, acc, s=150, color=f'C{i}', edgecolor='black', linewidth=1,
                          label=f'Classifier {i+1}', zorder=10)
    
    # Increase axis label size
    ax.set_xlabel('p₀ (Accuracy on class 0)', fontsize=12)
    ax.set_ylabel('p₁ (Accuracy on class 1)', fontsize=12)
    
    # Increase tick label size
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    # Highlight the constraint intersection if requested
    if highlight_constraints and hasattr(results, 'constraint_intersection'):
        # Draw a circle around the constraint intersection
        circle = plt.Circle(results.constraint_intersection, 0.05, 
                           fill=False, edgecolor='red', linestyle='--', 
                           linewidth=2, zorder=5)
        ax.add_patch(circle)
        
        # Label the intersection
        ax.annotate('Constraint\nIntersection', 
                   xy=results.constraint_intersection,
                   xytext=(results.constraint_intersection[0]+0.15, 
                          results.constraint_intersection[1]+0.15),
                   arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, 
                                  headwidth=8),
                   fontsize=11, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1, alpha=0.8))
    
    # Add confidence regions if requested
    if show_confidence and hasattr(results, 'bootstrap_samples'):
        # Draw confidence ellipse from bootstrap samples
        from matplotlib.patches import Ellipse
        import matplotlib.transforms as transforms
        
        # Calculate covariance of bootstrap samples
        cov = np.cov(results.bootstrap_samples, rowvar=False)
        pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
        
        # Calculate ellipse parameters
        ell_radius_x = np.sqrt(1 + pearson)
        ell_radius_y = np.sqrt(1 - pearson)
        ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                         facecolor='purple', alpha=0.2, edgecolor='purple', linewidth=2)
        
        # Scale and rotate ellipse
        scale_x = np.sqrt(cov[0, 0]) * 2  # 2 standard deviations
        scale_y = np.sqrt(cov[1, 1]) * 2
        
        mean_x = np.mean(results.bootstrap_samples[:, 0])
        mean_y = np.mean(results.bootstrap_samples[:, 1])
        
        transform = transforms.Affine2D() \
            .rotate_deg(45) \
            .scale(scale_x, scale_y) \
            .translate(mean_x, mean_y)
            
        ellipse.set_transform(transform + ax.transData)
        ax.add_patch(ellipse)
        
        # Label the confidence region
        ax.text(mean_x + scale_x/2, mean_y, 
               f"{confidence_level:.0%} Confidence",
               fontsize=11, fontweight='bold', color='purple', ha='center',
               bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="purple", alpha=0.8))
    
    # Add a legend with larger font and better positioning
    legend = ax.legend(loc='upper right', fontsize=11, framealpha=0.9, edgecolor='black')
    legend.get_frame().set_linewidth(1.0)
    
    # Set a tight layout for better use of space
    fig.tight_layout()
    
    # Save the figure if requested
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_error_correlation_matrix(
    error_correlation_matrix: np.ndarray,
    classifier_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (8, 6),
    cmap: str = 'coolwarm',
    title: str = "Error Correlation Matrix",
    show_values: bool = True,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot a heatmap of error correlations between classifiers.
    
    Parameters
    ----------
    error_correlation_matrix : np.ndarray
        Square matrix of error correlations between classifiers
    classifier_names : Optional[List[str]], optional
        Names for each classifier, by default None (uses "Classifier N")
    figsize : Tuple[int, int], optional
        Figure size, by default (8, 6)
    cmap : str, optional
        Colormap for heatmap, by default 'coolwarm'
    title : str, optional
        Plot title, by default "Error Correlation Matrix"
    show_values : bool, optional
        Whether to show correlation values in cells, by default True
    save_path : Optional[str], optional
        Path to save the figure, by default None (doesn't save)
        
    Returns
    -------
    plt.Figure
        The generated figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get number of classifiers
    n_classifiers = error_correlation_matrix.shape[0]
    
    # Default classifier names if not provided
    if classifier_names is None:
        classifier_names = [f"Classifier {i+1}" for i in range(n_classifiers)]
    
    # Create heatmap
    im = ax.imshow(error_correlation_matrix, cmap=cmap, vmin=-1, vmax=1)
    
    # Add colorbar with better formatting
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Error Correlation', fontsize=12, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)
    
    # Add cell values if requested with enhanced visibility
    if show_values:
        for i in range(n_classifiers):
            for j in range(n_classifiers):
                color = 'white' if abs(error_correlation_matrix[i, j]) > 0.5 else 'black'
                ax.text(j, i, f'{error_correlation_matrix[i, j]:.2f}', 
                       ha='center', va='center', color=color, fontsize=11, fontweight='bold')
    
    # Set ticks and labels
    ax.set_xticks(np.arange(n_classifiers))
    ax.set_yticks(np.arange(n_classifiers))
    ax.set_xticklabels(classifier_names, fontsize=10)
    ax.set_yticklabels(classifier_names, fontsize=10)
    
    # Rotate x tick labels and set alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add title with enhanced visibility
    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
    
    # Add grid lines with enhanced visibility
    ax.set_xticks(np.arange(n_classifiers+1)-.5, minor=True)
    ax.set_yticks(np.arange(n_classifiers+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=2)
    
    # Add a descriptive subtitle
    plt.figtext(0.5, 0.01, 
               "Values range from -1 (perfect negative correlation) to 1 (perfect positive correlation)", 
               ha="center", fontsize=10, fontstyle='italic')
    
    # Adjust layout
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    # Save the figure if requested
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_agreement_patterns(
    agreement_counts: np.ndarray,
    pattern_labels: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 6),
    colors: Optional[List[str]] = None,
    title: str = "Agreement Patterns",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot a visualization of agreement pattern counts.
    
    Parameters
    ----------
    agreement_counts : np.ndarray
        Array of counts for each agreement pattern
    pattern_labels : Optional[List[str]], optional
        Labels for each pattern, by default None (generates binary pattern strings)
    figsize : Tuple[int, int], optional
        Figure size, by default (10, 6)
    colors : Optional[List[str]], optional
        Colors for each bar, by default None (uses default colormap)
    title : str, optional
        Plot title, by default "Agreement Patterns"
    save_path : Optional[str], optional
        Path to save the figure, by default None (doesn't save)
        
    Returns
    -------
    plt.Figure
        The generated figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Flatten the agreement counts if needed
    if agreement_counts.ndim > 1:
        agreement_counts = agreement_counts.flatten()
    
    # Get number of patterns
    n_patterns = len(agreement_counts)
    
    # Generate pattern labels if not provided
    if pattern_labels is None:
        # Determine number of classifiers from pattern count
        n_classifiers = int(np.log2(n_patterns))
        pattern_labels = [format(i, f'0{n_classifiers}b') for i in range(n_patterns)]
    
    # Generate colors if not provided - updated to use colormaps directly
    if colors is None:
        try:
            # For matplotlib 3.7+ use newer API
            import matplotlib as mpl
            cmap = mpl.colormaps['viridis']
            colors = [cmap(i/n_patterns) for i in range(n_patterns)]
        except (AttributeError, ImportError):
            # Fallback for older matplotlib
            cmap = plt.cm.get_cmap('viridis', n_patterns)
            colors = [cmap(i) for i in range(n_patterns)]
    
    # Plot bars with black edges for contrast
    bars = ax.bar(pattern_labels, agreement_counts, color=colors, 
                 edgecolor='black', linewidth=1, alpha=0.8)
    
    # Add count labels on top of bars with enhanced visibility
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
               f'{int(height)}', ha='center', va='bottom', 
               fontsize=10, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.7))
    
    # Add labels with enhanced visibility
    ax.set_xlabel('Agreement Pattern (Binary Representation)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    
    # Add grid for y-axis with better visibility
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Add pattern explanation text
    pattern_text = ""
    for i, label in enumerate(pattern_labels[:3]):
        pattern_text += f"Pattern {label}: Classifier responses where 0=negative, 1=positive\n" if i == 0 else f"Pattern {label}\n"
    pattern_text += "..."
    
    fig.text(0.02, 0.02, pattern_text, fontsize=9, 
              bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="orange", alpha=0.7))
    
    # Adjust layout
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    
    # Save the figure if requested
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_accuracy_comparison(
    classifier_names: List[str],
    true_accuracies: List[float],
    estimated_accuracies: List[float],
    figsize: Tuple[int, int] = (10, 6),
    title: str = "Accuracy Comparison",
    include_difference: bool = True,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot a comparison of true vs. estimated accuracies.
    
    Parameters
    ----------
    classifier_names : List[str]
        Names of classifiers to compare
    true_accuracies : List[float]
        List of true accuracy values
    estimated_accuracies : List[float]
        List of estimated accuracy values
    figsize : Tuple[int, int], optional
        Figure size, by default (10, 6)
    title : str, optional
        Plot title, by default "Accuracy Comparison"
    include_difference : bool, optional
        Whether to include a difference subplot, by default True
    save_path : Optional[str], optional
        Path to save the figure, by default None (doesn't save)
        
    Returns
    -------
    plt.Figure
        The generated figure
    """
    # Create figure with appropriate number of subplots
    if include_difference:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, 
                                      gridspec_kw={'height_ratios': [3, 1]})
    else:
        fig, ax1 = plt.subplots(figsize=figsize)
    
    # Calculate x positions
    x = np.arange(len(classifier_names))
    width = 0.35
    
    # Plot bars with enhanced colors and edge for visibility
    bars1 = ax1.bar(x - width/2, true_accuracies, width, label='True Accuracy', 
                  color='royalblue', edgecolor='black', linewidth=1, alpha=0.8)
    bars2 = ax1.bar(x + width/2, estimated_accuracies, width, 
                  label='Estimated Accuracy', color='darkorange', 
                  edgecolor='black', linewidth=1, alpha=0.8)
    
    # Add labels with enhanced visibility
    ax1.set_xlabel('Classifier', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(classifier_names, fontsize=10)
    
    # Add enhanced legend
    legend = ax1.legend(fontsize=11, loc='best', 
                     framealpha=0.9, edgecolor='black')
    legend.get_frame().set_linewidth(1.0)
    
    # Add value labels with enhanced visibility
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="royalblue", alpha=0.7))
               
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="darkorange", alpha=0.7))
    
    # Set y limits with some padding
    ax1.set_ylim(0, max(max(true_accuracies), max(estimated_accuracies)) + 0.1)
    
    # Add grid for better readability
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    ax1.set_axisbelow(True)  # Put grid behind bars
    
    # Add difference plot if requested
    if include_difference:
        differences = [est - true for true, est in 
                      zip(true_accuracies, estimated_accuracies)]
        
        # Use color gradient for differences
        diff_colors = ['green' if d >= 0 else 'red' for d in differences]
        bars3 = ax2.bar(x, differences, width, 
                      color=diff_colors, edgecolor='black', linewidth=1, alpha=0.8)
        
        # Add value labels with enhanced visibility
        for bar in bars3:
            height = bar.get_height()
            y_pos = height + 0.005 if height >= 0 else height - 0.015
            ax2.text(bar.get_x() + bar.get_width()/2., y_pos,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", alpha=0.7))
        
        # Add labels with enhanced visibility
        ax2.set_xlabel('Classifier', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Difference', fontsize=12, fontweight='bold')
        ax2.set_title('Estimation Error (Estimated - True)', fontsize=12, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(classifier_names, fontsize=10)
        
        # Add a zero line with enhanced visibility
        ax2.axhline(y=0, color='k', linestyle='-', alpha=0.5, linewidth=1.5)
        
        # Add explanatory text
        ax2.text(0.02, 0.9, 'Positive values: Overestimation', 
                transform=ax2.transAxes, color='green', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="green", alpha=0.7))
        ax2.text(0.02, 0.1, 'Negative values: Underestimation', 
                transform=ax2.transAxes, color='red', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="red", alpha=0.7))
        
        # Add grid with enhanced visibility
        ax2.grid(axis='y', linestyle='--', alpha=0.5)
        ax2.set_axisbelow(True)  # Put grid behind bars
        
        # Set symmetric y limits
        max_diff = max(abs(min(differences)), abs(max(differences)))
        ax2.set_ylim(-max_diff - 0.02, max_diff + 0.02)
    
    # Adjust layout
    fig.tight_layout()
    
    # Save the figure if requested
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_3d_evaluation_space(
    evaluation_result: Any,
    figsize: Tuple[int, int] = (10, 8),
    title: str = "3D Evaluation Space",
    show_constraints: bool = True,
    alpha: float = 0.7,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Create a 3D visualization of the evaluation space for ternary classification.
    
    Parameters
    ----------
    evaluation_result : Any
        Evaluation results object with class accuracies
    figsize : Tuple[int, int], optional
        Figure size, by default (10, 8)
    title : str, optional
        Plot title, by default "3D Evaluation Space"
    show_constraints : bool, optional
        Whether to show constraint surfaces, by default True
    alpha : float, optional
        Transparency for surfaces, by default 0.7
    save_path : Optional[str], optional
        Path to save the figure, by default None (doesn't save)
        
    Returns
    -------
    plt.Figure
        The generated figure
    """
    # Create figure
    fig = plt.figure(figsize=figsize)
    
    try:
        # Try to create 3D axes
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - import for side effects
        ax = fig.add_subplot(111, projection='3d')
        
        # Check if we have class accuracies in the expected format
        if hasattr(evaluation_result, 'class_accuracies'):
            # Extract class accuracies for each classifier
            for i, class_accs in enumerate(evaluation_result.class_accuracies):
                # Plot point in 3D space with larger marker and black edge for visibility
                ax.scatter(class_accs[0], class_accs[1], class_accs[2], 
                          s=150, color=f'C{i}', label=f'Classifier {i+1}', 
                          zorder=10, edgecolor='black', linewidth=1)
        
        # Show constraint surfaces if requested
        if show_constraints:
            # Create a meshgrid for visualization
            p0, p1 = np.mgrid[0:1:20j, 0:1:20j]
            
            # For each type of constraint, create a surface
            # Example: Single classifier axiom
            # p0 + p1 + p2 = 1 + (R-1)/R → p2 = 1 + (R-1)/R - p0 - p1
            # For R=3: p2 = 4/3 - p0 - p1
            p2 = 4/3 - p0 - p1
            mask = (p2 >= 0) & (p2 <= 1)
            
            # Make sure we're retaining the 2D structure
            p0_masked = p0.copy()
            p1_masked = p1.copy()
            p2_masked = p2.copy()
            
            # Set values outside the mask to NaN to avoid plotting them
            p0_masked[~mask] = np.nan
            p1_masked[~mask] = np.nan
            p2_masked[~mask] = np.nan
            
            # Plot the surface with enhanced colors and transparency
            surface = ax.plot_surface(p0_masked, p1_masked, p2_masked, alpha=alpha, 
                                     color='skyblue', label='Single Classifier Axiom',
                                     edgecolor='darkblue', linewidth=0.5)
            
            # Make the surface clickable in the legend
            surface._facecolors2d = surface._facecolor3d
            surface._edgecolors2d = surface._edgecolor3d
        
        # Add labels and title with enhanced visibility
        ax.set_xlabel('p₀ (Accuracy on class 0)', fontsize=12, fontweight='bold', labelpad=10)
        ax.set_ylabel('p₁ (Accuracy on class 1)', fontsize=12, fontweight='bold', labelpad=10)
        ax.set_zlabel('p₂ (Accuracy on class 2)', fontsize=12, fontweight='bold', labelpad=10)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        # Set axis limits with some padding
        ax.set_xlim(0, 1.1)
        ax.set_ylim(0, 1.1)
        ax.set_zlim(0, 1.1)
        
        # Increase tick label size
        ax.tick_params(axis='both', which='major', labelsize=10)
        
        # Add a grid for better readability
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # Add a legend with enhanced visibility
        legend = ax.legend(loc='upper right', fontsize=11, 
                        framealpha=0.9, edgecolor='black')
        if legend:
            legend.get_frame().set_linewidth(1.0)
        
        # Add explanatory text about the constraint surface
        fig.text(0.02, 0.02, 
                "Constraint surface: p₀ + p₁ + p₂ = 4/3\nPoints on surface satisfy the single classifier axiom", 
                fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="orange", alpha=0.7))
        
        # Set a good viewing angle
        ax.view_init(elev=30, azim=45)
        
    except Exception as e:
        # Fall back to a 2D representation with error message
        ax = fig.add_subplot(111)
        
        # Show error message
        ax.text(0.5, 0.5, f"3D visualization not available:\n{str(e)}", 
               ha='center', va='center', fontsize=12, transform=ax.transAxes,
               bbox=dict(boxstyle="round,pad=1", fc="mistyrose", ec="red", alpha=0.8))
        
        # Add title
        ax.set_title("3D Visualization Error", fontsize=14, fontweight='bold', color='red')
        
        # Hide axes
        ax.axis('off')
    
    # Adjust layout
    fig.tight_layout()
    
    # Save the figure if requested
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig 