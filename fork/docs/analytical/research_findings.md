# Research Findings and Future Directions

This document outlines our research findings and future directions in extending the NTQR package.

## Current Research Findings

Our work in the fork extends the original NTQR package in several directions, with key findings summarized below.

### 1. Beyond Error Independence

While the error independence assumption produces elegant solutions for trios of classifiers, we've investigated alternative error correlation models:

```mermaid
graph TD
    A[Error Correlation Models] --> B[Error Independence]
    A --> C[Fixed Correlation]
    A --> D[Hierarchical Error]
    A --> E[Domain-Specific Correlation]
    
    B --> F[Current NTQR Model]
    C --> G[Our Extension 1]
    D --> H[Our Extension 2]
    E --> I[Our Extension 3]
    
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style F fill:#dfd,stroke:#333,stroke-width:2px
    style G fill:#bbf,stroke:#333,stroke-width:2px
    style H fill:#bbf,stroke:#333,stroke-width:2px
    style I fill:#bbf,stroke:#333,stroke-width:2px
```

**Key finding**: We've shown that for certain error correlation structures, exact solutions still exist beyond the simple error independence case.

### 2. Computational Complexity Analysis

We've analyzed the computational complexity of solving the axiom systems for various R-class problems:

| R Value | Single Classifier | Pair | Trio | Complexity Class |
|---------|-------------------|------|------|-----------------|
| R=2     | Linear            | Quadratic | Cubic | P |
| R=3     | Linear            | R²   | R³   | P |
| R=n     | Linear            | R²   | R³   | NP-Hard for large R |

**Key finding**: For R>3, we've developed approximation algorithms that provide bounds on evaluations with provable error margins.

### 3. Logical Alarms for Complex Systems

We've extended the logical alarm system to handle more complex evaluation scenarios:

```mermaid
flowchart TD
    A[Test Results] --> B[Agreement Statistics]
    B --> C{Satisfy Basic Axioms?}
    C -->|No| D[Fundamental Alarm]
    C -->|Yes| E{Satisfy Error<br>Independence?}
    E -->|No| F[Correlation Alarm]
    E -->|Yes| G{Satisfy Evaluation<br>Bounds?}
    G -->|No| H[Safety Spec Alarm]
    G -->|Yes| I[Valid Evaluation]
    
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style D fill:#faa,stroke:#333,stroke-width:2px
    style F fill:#fda,stroke:#333,stroke-width:2px
    style H fill:#fda,stroke:#333,stroke-width:2px
    style I fill:#afa,stroke:#333,stroke-width:2px
```

**Key finding**: Our extended alarm system can detect subtle violations of assumptions that would otherwise lead to incorrect evaluations.

## Ongoing Research Directions

### 1. Hybrid Probabilistic-Algebraic Models

We're exploring ways to combine the logical rigor of algebraic evaluation with the flexibility of probabilistic models:

```mermaid
graph LR
    A[Algebraic Constraints] --> C[Hybrid Models]
    B[Probabilistic Models] --> C
    C --> D[Flexible + Rigorous Evaluation]
    style C fill:#bbf,stroke:#333,stroke-width:2px
```

### 2. Time-Series Evaluation

We're extending the algebraic framework to handle time-series data, where classifier performance may drift over time:

```mermaid
flowchart LR
    A[Time Series Data] --> B[Temporal Segmentation]
    B --> C[Segment-wise Evaluation]
    C --> D[Temporal Drift Detection]
    D --> E[Dynamic Evaluation]
    style E fill:#bbf,stroke:#333,stroke-width:2px
```

### 3. Multi-Label Classification

We're developing axioms for multi-label classification scenarios, where each instance can belong to multiple classes simultaneously:

```mermaid
graph TD
    A[Multi-Label Problem] --> B[Decomposition Strategies]
    A --> C[Direct Axiom Formulation]
    B --> D[Binary Relevance]
    B --> E[Label Powerset]
    C --> F[Multi-Label Axioms]
    D --> G[Combined Evaluation]
    E --> G
    F --> G
    style G fill:#bbf,stroke:#333,stroke-width:2px
```

## Applications to LLM Evaluation

We've begun applying our extended NTQR framework to the evaluation of Large Language Models (LLMs):

### LLM as Evaluators

When LLMs evaluate other models, we can use algebraic constraints to verify consistency:

```mermaid
flowchart TD
    A[LLM 1] --> D[Evaluation<br>Task]
    B[LLM 2] --> D
    C[LLM 3] --> D
    D --> E[Agreement<br>Statistics]
    E --> F[Algebraic<br>Constraints]
    F --> G[Possible<br>Evaluations]
    F --> H{Consistency<br>Check}
    H -->|Consistent| I[Valid<br>Evaluation]
    H -->|Inconsistent| J[Violation<br>Detected]
    
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style B fill:#f9f,stroke:#333,stroke-width:2px
    style C fill:#f9f,stroke:#333,stroke-width:2px
    style I fill:#afa,stroke:#333,stroke-width:2px
    style J fill:#faa,stroke:#333,stroke-width:2px
```

**Key finding**: We can detect when LLMs are misaligned in their evaluations without knowing ground truth.

## Future Research Questions

Our ongoing work focuses on several key questions:

1. How can we extend algebraic evaluation to handle partial or uncertain agreements?
2. Can we develop efficient algorithms for exact evaluation in high-dimensional spaces?
3. What are the theoretical limits of unsupervised evaluation under different axiom systems?
4. How can we integrate task-specific domain knowledge without sacrificing the universality of the approach?

## Research Collaborations

We are actively collaborating with researchers in the following areas:

- Algebraic geometry for more efficient computation of evaluation varieties
- Formal verification systems for AI safety
- LLM evaluation frameworks
- Theoretical computer science for complexity analysis of evaluation algorithms

These collaborations aim to further strengthen the mathematical foundations of NTQR while expanding its practical applications. 