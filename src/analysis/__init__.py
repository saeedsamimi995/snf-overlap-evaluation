"""Analysis and evaluation scripts for addressing reviewer concerns."""

from .baseline_comparison import compare_baseline_vs_bigclam
from .benchmark import benchmark_pipeline
from .parameter_sensitivity import (
    analyze_variance_threshold_sensitivity,
    analyze_similarity_threshold_sensitivity
)

__all__ = [
    'compare_baseline_vs_bigclam',
    'benchmark_pipeline',
    'analyze_variance_threshold_sensitivity',
    'analyze_similarity_threshold_sensitivity'
]

