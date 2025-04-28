"""Module for convenience plotting functions related to NTQR algorithms."""

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import itertools
from collections.abc import Sequence
from types import ModuleType

import python.src.ntqr.r2.raxioms
import python.src.ntqr.r3.raxioms
import python.src.ntqr.r2.evaluations
import python.src.ntqr.r3.evaluations
import python.src.ntqr.alarms


def plot_evaluation_space(title="Evaluation Space for Binary Classifiers", figsize=(8, 8)):
    """
    Plot the evaluation space for binary classifiers.
    
    Parameters
    ----------
    title : str, optional
        Plot title, by default "Evaluation Space for Binary Classifiers"
    figsize : tuple, optional
        Figure size, by default (8, 8)
        
    Returns
    -------
    matplotlib.figure.Figure
        The generated figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
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
    
    # Add labels and title
    ax.set_xlabel('Accuracy on Label 0 (p₀)')
    ax.set_ylabel('Accuracy on Label 1 (p₁)')
    ax.set_title(title)
    
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


def plot_pair_logical_alarm_at_qs(
    labels: Sequence[str],
    classifiers: Sequence[int | str],
    qs: Sequence[int],
    responses: Sequence[Sequence[int]],
    plt: ModuleType = plt,
):
    """Plot pair possible evaluations given qs and responses.

    Parameters
    ----------
    labels : Sequence[str]
        The labels to be used.
    classifiers : Sequence[str | int]
        Classifier identifiers, could be integers or strings: 1 or 'i', eg.
    qs: Sequence[int]
        Number of label questions
    responses : Sequence[Sequence[int]]
        Number of label responses by each classifier.
    plt: Handle into the pyplot module.

    Returns
    -------
    None:
        Plots a square of possible evaluations and safety specification
        satisfying evaluations for each of the labels.
    """

    num_labels = len(labels)
    assert len(classifiers) == 2
    assert len(classifiers) == len(responses)
    assert sum(responses[0]) == sum(responses[1])
    Q = sum(responses[0])

    # Select the right axiom and evaluation classes
    if num_labels == 2:
        axioms_cls = python.src.ntqr.r2.raxioms.SingleClassifierAxioms
        evals_cls = python.src.ntqr.r2.evaluations.SingleClassifierEvaluations
    elif num_labels == 3:
        axioms_cls = python.src.ntqr.r3.raxioms.SingleClassifierAxioms
        evals_cls = python.src.ntqr.r3.evaluations.SingleClassifierEvaluations

    classifier_axioms = [
        axioms_cls(labels, classifier) for classifier in classifiers
    ]
    classifier_vars = zip(
        *[
            [
                axioms.responses_by_label[label]["correct"][label]
                for label in labels
            ]
            for axioms in classifier_axioms
        ]
    )

    classifier_evals = [evals_cls(Q, axioms) for axioms in classifier_axioms]
    # print(classifier_evals)
    # print(responses[0])
    # print(classifier_evals[0].errors_at_qs(qs, responses[0]))

    classifier_evals_by_label = zip(
        *[
            zip(
                *list(
                    classifier_by_label_evals
                    for classifier_by_label_evals in eval.correct_at_qs(
                        qs, classifier_responses
                    )
                ),
                strict=True,
            )
            for eval, classifier_responses in zip(classifier_evals, responses)
        ]
    )

    safety_specification = python.src.ntqr.alarms.LabelsSafetySpecification(
        [2] * num_labels
    )
    label_safety_evals = safety_specification.pair_safe_evaluations_at_qs(qs)

    # Start building plot
    fig, axs = plt.subplots(1, num_labels)
    for ql, ax, pair_label_values, safety_evals, label_vars in zip(
        qs, axs, classifier_evals_by_label, label_safety_evals, classifier_vars
    ):
        if ql > 0:
            _plot_pair_label_evals(
                ax, ql, pair_label_values, safety_evals, label_vars
            )
        else:
            ax.axis("off")

    fig.tight_layout(pad=2.0)

    plt.show()


def plot_pair_label_evals(ql, pair_label_evals, safety_evals, vars, plt=plt):
    x, y = zip(*pair_label_evals)

    plt.rcParams["text.usetex"] = True

    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    ax.set_xlabel("${}$".format(vars[0].name))
    ax.set_ylabel("${}$".format(vars[1].name), rotation="horizontal")
    ax.scatter(x, y, color="blue", label="evals")

    default_marker_size = plt.rcParams["lines.markersize"] ** 2

    xs, ys = zip(*safety_evals)
    ax.scatter(
        xs,
        ys,
        color="green",
        label="safety",
        marker="+",
        alpha=1.0,
        s=(default_marker_size + 90),
    )

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    ax.set_xlim(0, ql)
    ax.set_ylim(0, ql)

    plt.show()


def _plot_pair_label_evals(
    ax, ql, pair_label_evals, safety_evals, vars, plt=plt
):

    # Here is where we do the product
    x, y = zip(*itertools.product(*pair_label_evals))

    plt.rcParams["text.usetex"] = True

    ax.set_aspect("equal")
    ax.set_xlabel("${}$".format(vars[0]))
    ax.set_ylabel("${}$".format(vars[1]), rotation="horizontal")
    ax.scatter(x, y, color="blue", label="evals")

    default_marker_size = plt.rcParams["lines.markersize"] ** 2

    xs, ys = zip(*safety_evals)
    ax.scatter(
        xs,
        ys,
        color="green",
        label="safety",
        marker="+",
        alpha=1.0,
        s=(default_marker_size + 3 * default_marker_size / 2),
    )

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    extra_bit = 1 / 4
    ax.set_xlim(-extra_bit, ql + extra_bit)
    ax.set_ylim(-extra_bit, ql + extra_bit)

    return
