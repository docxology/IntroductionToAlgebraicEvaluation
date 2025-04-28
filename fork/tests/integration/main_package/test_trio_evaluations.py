"""
Tests for trio evaluation functionality from the main NTQR package.
"""

import pytest
import numpy as np

from fork.src.utils.compatibility import ensure_main_package_importable

# Ensure main package is importable
ensure_main_package_importable()

# Import from main package
from python.src.ntqr.r2.evaluators import (
    ErrorIndependentEvaluation,
    MajorityVotingEvaluation,
)
from python.src.ntqr.r2.datasketches import TrioLabelVoteCounts, TrioVoteCounts

class TestTrioEvaluations:
    """
    Tests for trio evaluation functionality from the main package.
    """
    
    def setup_method(self):
        """
        Set up test data for trio evaluations.
        """
        # Create trio data for binary classification
        # Format: counts of how three classifiers voted on test items
        # For binary classification, there are 2^3 = 8 possible voting patterns
        # [n_000, n_001, n_010, n_011, n_100, n_101, n_110, n_111]
        # Where n_abc means: classifier1 voted a, classifier2 voted b, classifier3 voted c
        
        # Create data with clear majority patterns
        # This data represents a case where:
        # - All classifiers are reasonably accurate (75-80%)
        # - Errors are mostly independent
        # - Majority vote is generally correct
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
        
        # Create TrioVoteCounts object
        self.trio_counts = TrioVoteCounts(self.trio_votes)
        
        # Create label vote counts
        # For binary classification with known ground truth
        # First row: votes on items with true label 0
        # Second row: votes on items with true label 1
        self.label_vote_counts = np.array([
            [10, 2, 2, 1, 2, 1, 0, 0],  # patterns for true label 0
            [0, 1, 1, 1, 1, 1, 2, 15],  # patterns for true label 1
        ])
        
        # Create TrioLabelVoteCounts object
        self.trio_label_counts = TrioLabelVoteCounts(self.label_vote_counts)
        
        # Create evaluators
        self.error_independent_evaluator = ErrorIndependentEvaluation()
        self.majority_voting_evaluator = MajorityVotingEvaluation()
    
    def test_trio_counts_properties(self):
        """
        Test basic properties of TrioVoteCounts.
        """
        # Check total number of test items
        assert self.trio_counts.n == np.sum(self.trio_votes)
        
        # Check majority vote counts
        # Patterns where majority voted 0: n_000, n_001, n_010, n_100
        majority_0 = np.sum(self.trio_votes[[0, 1, 2, 4]])
        # Patterns where majority voted 1: n_011, n_101, n_110, n_111
        majority_1 = np.sum(self.trio_votes[[3, 5, 6, 7]])
        
        assert self.trio_counts.majority_vote_counts[0] == majority_0
        assert self.trio_counts.majority_vote_counts[1] == majority_1
    
    def test_trio_label_counts_properties(self):
        """
        Test basic properties of TrioLabelVoteCounts.
        """
        # Check total number of items with each label
        assert self.trio_label_counts.n_by_label[0] == np.sum(self.label_vote_counts[0])
        assert self.trio_label_counts.n_by_label[1] == np.sum(self.label_vote_counts[1])
        
        # Check overall number of test items
        assert self.trio_label_counts.n == np.sum(self.label_vote_counts)
    
    def test_error_independent_evaluation(self):
        """
        Test error-independent evaluation with trio data.
        """
        # Evaluate using error-independent evaluator
        result = self.error_independent_evaluator.evaluate(self.trio_label_counts)
        
        # Check that accuracies are reasonable
        # For the data we constructed, all classifiers should be ~75-80% accurate
        for classifier_idx in range(3):
            assert 0.7 <= result.classifier_accuracies[classifier_idx].accuracy_p0 <= 0.9
            assert 0.7 <= result.classifier_accuracies[classifier_idx].accuracy_p1 <= 0.9
        
        # Detailed check for first classifier
        # Calculate expected accuracy on label 0
        # Patterns where classifier 1 voted 0: n_000, n_001, n_010, n_011
        correct_0_votes = np.sum(self.label_vote_counts[0, [0, 1, 2, 3]])
        total_0 = np.sum(self.label_vote_counts[0])
        expected_p0 = correct_0_votes / total_0 if total_0 > 0 else 0
        
        # Calculate expected accuracy on label 1
        # Patterns where classifier 1 voted 1: n_100, n_101, n_110, n_111
        correct_1_votes = np.sum(self.label_vote_counts[1, [4, 5, 6, 7]])
        total_1 = np.sum(self.label_vote_counts[1])
        expected_p1 = correct_1_votes / total_1 if total_1 > 0 else 0
        
        # Allow some tolerance since error-independent evaluation is an approximation
        assert np.isclose(result.classifier_accuracies[0].accuracy_p0, expected_p0, atol=0.1)
        assert np.isclose(result.classifier_accuracies[0].accuracy_p1, expected_p1, atol=0.1)
    
    def test_majority_voting_evaluation(self):
        """
        Test majority voting evaluation with trio data.
        """
        # Evaluate using majority voting evaluator
        result = self.majority_voting_evaluator.evaluate(self.trio_label_counts)
        
        # Calculate expected accuracy of majority voting on label 0
        # Patterns where majority voted 0: n_000, n_001, n_010, n_100
        correct_0_votes = np.sum(self.label_vote_counts[0, [0, 1, 2, 4]])
        total_0 = np.sum(self.label_vote_counts[0])
        expected_p0 = correct_0_votes / total_0 if total_0 > 0 else 0
        
        # Calculate expected accuracy of majority voting on label 1
        # Patterns where majority voted 1: n_011, n_101, n_110, n_111
        correct_1_votes = np.sum(self.label_vote_counts[1, [3, 5, 6, 7]])
        total_1 = np.sum(self.label_vote_counts[1])
        expected_p1 = correct_1_votes / total_1 if total_1 > 0 else 0
        
        # Check that accuracies match expected values
        assert np.isclose(result.accuracy_p0, expected_p0)
        assert np.isclose(result.accuracy_p1, expected_p1) 