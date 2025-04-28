"""
Unit tests for the visualization module.
"""

import unittest
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import the module to test
from fork.src.utils.visualization import (
    enhanced_evaluation_space,
    plot_error_correlation_matrix,
    plot_agreement_patterns,
    plot_accuracy_comparison,
    plot_3d_evaluation_space
)


class TestVisualization(unittest.TestCase):
    """Test the visualization module."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for saving plots if needed
        self.test_output_dir = Path(__file__).parent / "test_output"
        os.makedirs(self.test_output_dir, exist_ok=True)
        
        # Sample data for testing
        self.sample_error_matrix = np.array([
            [1.0, 0.3, 0.5],
            [0.3, 1.0, 0.2],
            [0.5, 0.2, 1.0]
        ])
        
        self.sample_agreement_counts = np.array([100, 20, 30, 10, 15, 25, 40, 60])
        
        # Create a mock evaluation result object
        class MockEvaluationResult:
            def __init__(self):
                self.accuracies = [0.8, 0.75, 0.7]
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
        
        self.mock_result = MockEvaluationResult()
        
        # Sample data for accuracy comparison
        self.classifier_names = ["Random Forest", "Logistic Regression", "Neural Network"]
        self.true_accuracies = [0.85, 0.80, 0.78]
        self.estimated_accuracies = [0.83, 0.79, 0.80]
    
    def tearDown(self):
        """Clean up after tests."""
        # Close all open figures to prevent memory leaks
        plt.close('all')
    
    def test_enhanced_evaluation_space(self):
        """Test the enhanced_evaluation_space function."""
        # Test with default parameters
        fig = enhanced_evaluation_space(self.mock_result)
        self.assertIsInstance(fig, plt.Figure)
        
        # Test with all options enabled
        fig = enhanced_evaluation_space(
            self.mock_result,
            title="Test Evaluation Space",
            highlight_constraints=True,
            show_confidence=True,
            confidence_level=0.95,
            figsize=(8, 6),
            save_path=str(self.test_output_dir / "test_evaluation_space.png")
        )
        self.assertIsInstance(fig, plt.Figure)
        
        # Verify file was created
        self.assertTrue((self.test_output_dir / "test_evaluation_space.png").exists())
    
    def test_plot_error_correlation_matrix(self):
        """Test the plot_error_correlation_matrix function."""
        # Test with default parameters
        fig = plot_error_correlation_matrix(self.sample_error_matrix)
        self.assertIsInstance(fig, plt.Figure)
        
        # Test with custom classifier names
        fig = plot_error_correlation_matrix(
            self.sample_error_matrix,
            classifier_names=["RF", "LR", "NN"],
            title="Test Error Correlation",
            show_values=True,
            save_path=str(self.test_output_dir / "test_error_correlation.png")
        )
        self.assertIsInstance(fig, plt.Figure)
        
        # Verify file was created
        self.assertTrue((self.test_output_dir / "test_error_correlation.png").exists())
    
    def test_plot_agreement_patterns(self):
        """Test the plot_agreement_patterns function."""
        # Test with default parameters
        fig = plot_agreement_patterns(self.sample_agreement_counts)
        self.assertIsInstance(fig, plt.Figure)
        
        # Test with custom pattern labels
        custom_labels = ["000", "001", "010", "011", "100", "101", "110", "111"]
        fig = plot_agreement_patterns(
            self.sample_agreement_counts,
            pattern_labels=custom_labels,
            title="Test Agreement Patterns",
            save_path=str(self.test_output_dir / "test_agreement_patterns.png")
        )
        self.assertIsInstance(fig, plt.Figure)
        
        # Verify file was created
        self.assertTrue((self.test_output_dir / "test_agreement_patterns.png").exists())
    
    def test_plot_accuracy_comparison(self):
        """Test the plot_accuracy_comparison function."""
        # Test with default parameters
        fig = plot_accuracy_comparison(
            self.classifier_names,
            self.true_accuracies,
            self.estimated_accuracies
        )
        self.assertIsInstance(fig, plt.Figure)
        
        # Test without difference subplot
        fig = plot_accuracy_comparison(
            self.classifier_names,
            self.true_accuracies,
            self.estimated_accuracies,
            include_difference=False,
            title="Test Accuracy Comparison",
            save_path=str(self.test_output_dir / "test_accuracy_comparison.png")
        )
        self.assertIsInstance(fig, plt.Figure)
        
        # Verify file was created
        self.assertTrue((self.test_output_dir / "test_accuracy_comparison.png").exists())
    
    def test_plot_3d_evaluation_space(self):
        """Test the plot_3d_evaluation_space function."""
        try:
            # Test with default parameters
            fig = plot_3d_evaluation_space(self.mock_result)
            self.assertIsInstance(fig, plt.Figure)
            
            # Test with constraints disabled
            fig = plot_3d_evaluation_space(
                self.mock_result,
                show_constraints=False,
                title="Test 3D Evaluation Space",
                save_path=str(self.test_output_dir / "test_3d_evaluation_space.png")
            )
            self.assertIsInstance(fig, plt.Figure)
            
            # Verify file was created
            self.assertTrue((self.test_output_dir / "test_3d_evaluation_space.png").exists())
        except ValueError as e:
            # Skip test if matplotlib 3D support is not available
            if "Unknown projection '3d'" in str(e):
                print("Skipping 3D visualization test due to matplotlib compatibility issues.")
            else:
                raise


if __name__ == '__main__':
    unittest.main() 