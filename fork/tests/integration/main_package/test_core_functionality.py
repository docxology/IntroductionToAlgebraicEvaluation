"""
Tests to verify core functionality from the main NTQR package.
"""

import pytest
import numpy as np

from fork.src.utils.compatibility import ensure_main_package_importable

# Ensure main package is importable
ensure_main_package_importable()

# Import from main package
from python.src.ntqr import evaluations, alarms
from python.src.ntqr.r2.evaluators import (
    SupervisedEvaluation,
    ErrorIndependentEvaluation,
    MajorityVotingEvaluation,
)

class TestCoreEvaluations:
    """
    Tests for core evaluation functionality from the main package.
    """
    
    def setup_method(self):
        """
        Set up test data.
        """
        # Create simple test data for binary classification
        # Format:
        # [
        #   [TP, FN],  # Row for positive examples
        #   [FP, TN]   # Row for negative examples
        # ]
        self.binary_data = np.array([
            [80, 20],  # 80 true positives, 20 false negatives
            [30, 70]   # 30 false positives, 70 true negatives
        ])
        
        # Create supervised evaluator (uses ground truth for evaluation)
        self.supervised_evaluator = SupervisedEvaluation()
        
        # Create error independent evaluator (for a trio of classifiers)
        self.error_independent_evaluator = ErrorIndependentEvaluation()
        
        # Create majority voting evaluator
        self.majority_voting_evaluator = MajorityVotingEvaluation()
    
    def test_supervised_evaluation(self):
        """
        Test that supervised evaluation works properly.
        """
        # Evaluate with supervised evaluator
        result = self.supervised_evaluator.evaluate(self.binary_data)
        
        # Check that accuracies are correct
        expected_p1 = 80 / (80 + 20)  # TP / (TP + FN)
        expected_p0 = 70 / (70 + 30)  # TN / (TN + FP)
        
        assert np.isclose(result.accuracy_p1, expected_p1)
        assert np.isclose(result.accuracy_p0, expected_p0)
        
        # Check overall accuracy
        expected_accuracy = (80 + 70) / (80 + 20 + 30 + 70)
        assert np.isclose(result.accuracy, expected_accuracy)
    
    def test_alarms_functionality(self):
        """
        Test that alarms functionality works properly.
        """
        # Create a safety specification
        safety_spec = alarms.LabelsSafetySpecification(p1_min=0.7, p0_min=0.7)
        
        # Create a validator for the safety specification
        validator = alarms.SingleClassifierSafetyValidator(safety_spec)
        
        # Create an evaluation that satisfies the safety spec
        safe_result = SupervisedEvaluation.EvaluationResult(
            accuracy_p1=0.8,
            accuracy_p0=0.75,
            accuracy=0.775
        )
        
        # Create an evaluation that violates the safety spec
        unsafe_result = SupervisedEvaluation.EvaluationResult(
            accuracy_p1=0.8,
            accuracy_p0=0.65,  # Below the p0_min threshold
            accuracy=0.725
        )
        
        # Check validation results
        assert validator.is_valid(safe_result)
        assert not validator.is_valid(unsafe_result)
    
    @pytest.mark.skip(reason="Requires trio data which we'll implement in a separate test")
    def test_error_independent_evaluation(self):
        """
        Test that error-independent evaluation works properly.
        """
        # This will be implemented in a separate test with trio data
        pass
    
    @pytest.mark.skip(reason="Requires trio data which we'll implement in a separate test")
    def test_majority_voting_evaluation(self):
        """
        Test that majority voting evaluation works properly.
        """
        # This will be implemented in a separate test with trio data
        pass 