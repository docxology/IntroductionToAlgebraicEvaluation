"""
Tests for the enhanced visualization tools.
"""

import os
import tempfile
import pytest
import numpy as np
import matplotlib.pyplot as plt

from fork.src.utils.compatibility import ensure_main_package_importable

# Ensure main package is importable
ensure_main_package_importable()

# Import from main package
from python.src.ntqr.r2.evaluators import SupervisedEvaluation
from python.src.ntqr.r2.datasketches import TrioVoteCounts, TrioLabelVoteCounts

# Import visualizations module
from fork.src.extensions.visualizations import (
    plot_comparative_evaluation,
    plot_trio_agreement_matrix,
    plot_evaluation_confidence,
)


class TestVisualizations:
    """
    Tests for the enhanced visualization tools.
    """
    
    def setup_method(self):
        """
        Set up test data for visualization tests.
        """
        # Create sample evaluation results
        self.eval_result1 = SupervisedEvaluation.EvaluationResult(
            accuracy_p0=0.8,
            accuracy_p1=0.7,
            accuracy=0.75
        )
        
        self.eval_result2 = SupervisedEvaluation.EvaluationResult(
            accuracy_p0=0.6,
            accuracy_p1=0.9,
            accuracy=0.75
        )
        
        # Create sample trio vote counts
        self.trio_votes = np.array([
            10,  # n_000: all voted 0
            3,   # n_001: 1 and 2 voted 0, 3 voted 1
            3,   # n_010: 1 and 3 voted 0, 2 voted 1
            2,   # n_011: 1 voted 0, 2 and 3 voted 1
            3,   # n_100: 2 and 3 voted 0, 1 voted 1
            2,   # n_101: 2 voted 0, 1 and 3 voted 1
            2,   # n_110: 3 voted 0, 1 and 2 voted 1
            15,  # n_111: all voted 1
        ])
        
        # Create temporary directory for saving plots
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """
        Clean up after tests.
        """
        # Close all plots
        plt.close('all')
    
    def test_plot_comparative_evaluation(self):
        """
        Test the comparative evaluation plot.
        """
        # Create evaluations list
        evaluations = [
            ("Classifier 1", self.eval_result1),
            ("Classifier 2", self.eval_result2),
        ]
        
        # Create plot
        fig = plot_comparative_evaluation(
            evaluations=evaluations,
            title="Test Comparative Evaluation"
        )
        
        # Check that figure was created
        assert isinstance(fig, plt.Figure)
        
        # Check that the figure has the expected elements
        ax = fig.axes[0]
        assert len(ax.patches) == 4  # 2 bars for each classifier
        assert ax.get_title() == "Test Comparative Evaluation"
        
        # Test saving to file
        save_path = os.path.join(self.temp_dir, "comparative_eval.png")
        fig = plot_comparative_evaluation(
            evaluations=evaluations,
            save_path=save_path
        )
        assert os.path.exists(save_path)
    
    def test_plot_trio_agreement_matrix(self):
        """
        Test the trio agreement matrix plot.
        """
        # Create plot
        fig = plot_trio_agreement_matrix(
            trio_data=self.trio_votes,
            classifier_names=["Model A", "Model B", "Model C"],
            title="Test Trio Agreement"
        )
        
        # Check that figure was created
        assert isinstance(fig, plt.Figure)
        
        # Check that the figure has the expected elements
        assert len(fig.axes) == 4  # 4 subplots
        assert fig._suptitle.get_text() == "Test Trio Agreement"
        
        # Test saving to file
        save_path = os.path.join(self.temp_dir, "trio_agreement.png")
        fig = plot_trio_agreement_matrix(
            trio_data=self.trio_votes,
            save_path=save_path
        )
        assert os.path.exists(save_path)
    
    def test_plot_evaluation_confidence(self):
        """
        Test the evaluation confidence plot.
        """
        # Create plot
        fig = plot_evaluation_confidence(
            evaluation_result=self.eval_result1,
            title="Test Evaluation Confidence"
        )
        
        # Check that figure was created
        assert isinstance(fig, plt.Figure)
        
        # Check that the figure has the expected elements
        ax = fig.axes[0]
        assert ax.get_title() == "Test Evaluation Confidence"
        
        # Check that point estimate is plotted
        scatter_collections = [c for c in ax.collections if isinstance(c, plt.matplotlib.collections.PathCollection)]
        assert len(scatter_collections) >= 1
        
        # Test saving to file
        save_path = os.path.join(self.temp_dir, "eval_confidence.png")
        fig = plot_evaluation_confidence(
            evaluation_result=self.eval_result1,
            save_path=save_path
        )
        assert os.path.exists(save_path) 