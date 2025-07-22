"""
Visualization module for polymer_gc.

This module provides reusable visualization functions for training analysis,
model evaluation, and data exploration in the polymer_gc package.
"""

from .plots import (
    create_parity_plot,
    create_error_distribution_plot,
    create_tsne_embeddings_plot,
    generate_training_report,
)
from .utils import (
    create_full_evaluation_suite,
    calculate_comprehensive_metrics,
    prepare_data_stats,
    prepare_training_stats,
)
from .kfold_analysis import (
    create_kfold_ensemble_predictions,
    create_kfold_visualization_suite,
)

__all__ = [
    "create_parity_plot",
    "create_error_distribution_plot", 
    "create_tsne_embeddings_plot",
    "generate_training_report",
    "create_full_evaluation_suite",
    "calculate_comprehensive_metrics",
    "prepare_data_stats",
    "prepare_training_stats",
    "create_kfold_ensemble_predictions",
    "create_kfold_visualization_suite",
]