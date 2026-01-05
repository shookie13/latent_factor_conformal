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
)
from .cp import (
    CPMethod,
    NaiveAbsoluteResidualCP,
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
    "run_em_like",
    "simulate_factor_data_bspline",
    "make_open_uniform_knots_np",
    "bspline_basis_np",
    "build_time_warp_from_proxy_np",
    "missing_prob_from_factor_variance",
    "sample_mask_from_varf",
    "CPMethod",
    "NaiveAbsoluteResidualCP",
    "WeightedSplitConformalCP",
    "LocalizedSplitConformalCP",
    "LocalizedWeightedSplitConformalCP",
    "weighted_split_conformal_prediction",
    "localized_weighted_split_conformal_prediction",
    "kernel",
    "kernel_sampler",
    "unflatten",
]

