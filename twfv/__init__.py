"""
Subject-specific time-warped B-spline factor variance model.
Exports building blocks and a simple EM-like runner.
"""

from .bspline import bspline_basis, make_open_uniform_knots
from .warp import build_time_warp_from_proxy, update_time_warp
from .variance import build_a2_from_C_and_warp
from .factor import factor_E_step, factor_log_likelihood
from .runner import run_em_like
from .simulate import (
    simulate_factor_data_bspline,
    make_open_uniform_knots_np,
    bspline_basis_np,
    build_time_warp_from_proxy_np,
    missing_prob_from_factor_variance,
    sample_mask_from_varf,
    split_cal_test_from_train_and_pmiss,
)
from .cp import (
    CPMethod,
    NaiveAbsoluteResidualCP,
    CPTDRSplitConformalCP,
    WeightedSplitConformalCP,
    LocalizedSplitConformalCP,
    LocalizedWeightedSplitConformalCP,
    weighted_split_conformal_prediction,
    localized_weighted_split_conformal_prediction,
    kernel,
    kernel_sampler,
    unflatten,
)
from .woodbury import PatternGroup, build_pattern_groups, factor_E_step_woodbury, factor_log_likelihood_woodbury
from .metrics import binned_interval_stats_by_quantiles, conditional_interval_report
from .plot import plot_intervals_on_series_from_q_sigma
from .plot import plot_compare_conditional_reports
from .diagnostics import cal_test_scaled_residual_diagnostics, resample_test_idx_and_collect_scaled_cp_artifacts,plot_em_history_grid
__all__ = [
    "bspline_basis",
    "make_open_uniform_knots",
    "build_time_warp_from_proxy",
    "update_time_warp",
    "build_a2_from_C_and_warp",
    "factor_E_step",
    "factor_log_likelihood",
    "PatternGroup",
    "build_pattern_groups",
    "factor_E_step_woodbury",
    "factor_log_likelihood_woodbury",
    "binned_interval_stats_by_quantiles",
    "conditional_interval_report",
    "plot_intervals_on_series_from_q_sigma",
    "plot_compare_conditional_reports",
    "run_em_like",
    "simulate_factor_data_bspline",
    "make_open_uniform_knots_np",
    "bspline_basis_np",
    "build_time_warp_from_proxy_np",
    "missing_prob_from_factor_variance",
    "sample_mask_from_varf",
    "split_cal_test_from_train_and_pmiss",
    "CPMethod",
    "NaiveAbsoluteResidualCP",
    "CPTDRSplitConformalCP",
    "WeightedSplitConformalCP",
    "LocalizedSplitConformalCP",
    "LocalizedWeightedSplitConformalCP",
    "weighted_split_conformal_prediction",
    "localized_weighted_split_conformal_prediction",
    "kernel",
    "kernel_sampler",
    "unflatten",
    "cal_test_scaled_residual_diagnostics",
    "resample_test_idx_and_collect_scaled_cp_artifacts",
    "plot_em_history_grid",
]

