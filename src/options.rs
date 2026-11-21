// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! Optimization options, mirroring tinyopt's `Options` (dense subset).

use std::sync::Arc;

use crate::backend::LinearSolverMethod;

/// Which solver to use.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Solver {
    /// Levenberg-Marquardt (default for NLLS problems).
    #[default]
    LevenbergMarquardt,
    /// Gauss-Newton.
    GaussNewton,
    /// Gradient descent.
    GradientDescent,
    /// Non-linear conjugate gradient.
    ConjugateGradient,
    /// Dog-leg trust region.
    DogLeg,
    /// BFGS.
    Bfgs,
    /// L-BFGS.
    Lbfgs,
}

/// Line search method.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum LineSearchMethod {
    /// No line search (direct solver step or trust region).
    #[default]
    None,
    /// Backtracking satisfying the Armijo condition.
    Backtracking,
    /// Armijo condition with safeguarded quadratic interpolation.
    Wolfe,
    /// Derivative-free golden-section search along the step direction.
    GoldenSection,
}

/// Linear solver configuration.
#[derive(Clone, Debug)]
pub struct LinearSolverOptions {
    /// Dense factorization to use (LDLT is the default, as in C++).
    pub method: LinearSolverMethod,
    /// Relative threshold for the SVD pseudo-inverse step.
    pub svd_relative_threshold: f64,
    /// Initial shift applied when a factorization needs regularization.
    pub initial_shift: f64,
}

impl Default for LinearSolverOptions {
    fn default() -> Self {
        Self {
            method: LinearSolverMethod::Ldlt,
            svd_relative_threshold: 0.0,
            initial_shift: 1e-3,
        }
    }
}

/// Line search options.
#[derive(Clone, Debug)]
pub struct LineSearchOptions {
    pub method: LineSearchMethod,
    /// Armijo sufficient decrease parameter.
    pub c1: f32,
    /// Backtracking contraction factor in `(0, 1)`.
    pub contraction: f32,
    /// Initial step-size multiplier.
    pub initial_step: f32,
    /// Minimum step-size multiplier before terminating the search.
    pub min_step: f32,
    /// Maximum line search evaluations per step.
    pub max_iters: u8,
}

impl Default for LineSearchOptions {
    fn default() -> Self {
        Self {
            method: LineSearchMethod::None,
            c1: 1e-4,
            contraction: 0.5,
            initial_step: 1.0,
            min_step: 1e-12,
            max_iters: 20,
        }
    }
}

/// Export options.
#[derive(Clone, Debug)]
pub struct Export {
    /// Save the final covariance matrix in the summary.
    pub final_cov: bool,
    /// Save per-iteration errors, step norms, and success flags.
    pub save_history: bool,
    /// Measure the optimization duration.
    pub measure_time: bool,
}

impl Default for Export {
    fn default() -> Self {
        Self {
            final_cov: false,
            save_history: true,
            measure_time: true,
        }
    }
}

/// Optimization behaviour options.
#[derive(Clone, Debug, Default)]
pub struct Optimization {
    /// Compute the step quality for adaptive damping.
    pub use_step_quality: bool,
    /// Use relative error decrease as step quality.
    pub use_step_quality_approx: bool,
    /// Gradient clipping to `[-v, +v]`, disabled when 0.
    pub grad_clipping: f32,
    /// Minimum ratio of fixed parameters to trigger the permutation solve.
    pub min_fixed_ratio: f32,
}

impl Optimization {
    /// Default matching the C++ library.
    pub fn new() -> Self {
        Self {
            use_step_quality: true,
            use_step_quality_approx: false,
            grad_clipping: 0.0,
            min_fixed_ratio: 0.15,
        }
    }
}

/// Hessian properties.
#[derive(Clone, Debug)]
pub struct HessianOptions {
    /// Whether the Hessian is fully filled (vs. upper triangular only).
    pub h_is_full: bool,
    /// Check that Hessian diagonals are not all below this threshold (0 disables).
    pub check_min_h_diag: f32,
}

impl Default for HessianOptions {
    fn default() -> Self {
        Self {
            h_is_full: true,
            check_min_h_diag: 0.0,
        }
    }
}

/// Cost scaling options.
#[derive(Clone, Debug)]
pub struct CostScaling {
    /// Use the squared norm instead of the norm.
    pub use_squared_norm: bool,
    /// Rescale the cost by 0.5.
    pub downscale_by_2: bool,
    /// Normalize the final error by the number of residuals.
    pub normalize: bool,
}

impl Default for CostScaling {
    fn default() -> Self {
        Self {
            use_squared_norm: true,
            downscale_by_2: false,
            normalize: false,
        }
    }
}

/// Callback receiving `(err, dx_norm2, grad_norm2)`; return true to stop.
pub type StopCallback = Arc<dyn Fn(f64, f64, f64) -> bool + Send + Sync>;
/// Callback receiving `(err, dx, grad)`; return true to stop.
pub type StopCallback2 = Arc<dyn Fn(f64, &[f32], &[f32]) -> bool + Send + Sync>;
/// Callback receiving the applied step and whether it was rolled back.
pub type StepCallback = Arc<dyn Fn(&[f32], bool) -> bool + Send + Sync>;

/// Stop criteria.
#[derive(Clone, Default)]
pub struct StopCriteria {
    /// Maximum number of successful outer iterations.
    pub max_iters: u16,
    /// Minimum error/cost.
    pub min_error: f32,
    /// Minimum relative error decrease.
    pub min_rerr_dec: f32,
    /// Minimum squared step norm.
    pub min_step_norm2: f32,
    /// Minimum relative squared step norm (0 disables).
    pub min_step_rel_norm2: f32,
    /// Minimum squared gradient norm.
    pub min_grad_norm2: f32,
    /// Minimum gradient infinity norm (0 disables).
    pub min_grad_max: f32,
    /// Overall maximum failures to decrease the error (0 disables).
    pub max_total_failures: u8,
    /// Maximum consecutive failures to decrease the error.
    pub max_consec_failures: u8,
    /// Maximum optimization duration in milliseconds (0 disables).
    pub max_duration_ms: f64,

    /// Called with `(err, dx_norm2, grad_norm2)` each iteration; return true to stop.
    pub stop_callback: Option<StopCallback>,
    /// Called with `(err, dx, grad)` each iteration; return true to stop.
    pub stop_callback2: Option<StopCallback2>,
    /// Called after every parameter update with the step and whether it was rolled back.
    pub step_callback: Option<StepCallback>,
}

impl core::fmt::Debug for StopCriteria {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("StopCriteria")
            .field("max_iters", &self.max_iters)
            .field("min_error", &self.min_error)
            .field("min_rerr_dec", &self.min_rerr_dec)
            .field("min_step_norm2", &self.min_step_norm2)
            .field("min_step_rel_norm2", &self.min_step_rel_norm2)
            .field("min_grad_norm2", &self.min_grad_norm2)
            .field("min_grad_max", &self.min_grad_max)
            .field("max_total_failures", &self.max_total_failures)
            .field("max_consec_failures", &self.max_consec_failures)
            .field("max_duration_ms", &self.max_duration_ms)
            .finish()
    }
}

impl StopCriteria {
    /// Default matching the C++ library.
    pub fn new() -> Self {
        Self {
            max_iters: 50,
            min_error: 1e-12,
            min_rerr_dec: 1e-6,
            min_step_norm2: 1e-14,
            min_step_rel_norm2: 1e-16,
            min_grad_norm2: 1e-18,
            min_grad_max: 1e-10,
            max_total_failures: 0,
            max_consec_failures: 5,
            max_duration_ms: 0.0,
            stop_callback: None,
            stop_callback2: None,
            step_callback: None,
        }
    }
}

/// Logging options.
#[derive(Clone, Debug)]
pub struct LogOptions {
    pub enable: bool,
    pub print_emoji: bool,
    pub print_x: bool,
    pub print_dx: bool,
    pub print_inliers: bool,
    pub print_t: bool,
    pub print_failure: bool,
}

impl Default for LogOptions {
    fn default() -> Self {
        Self {
            enable: true,
            print_emoji: true,
            print_x: false,
            print_dx: false,
            print_inliers: false,
            print_t: true,
            print_failure: false,
        }
    }
}

/// Levenberg-Marquardt options.
#[derive(Clone, Debug)]
pub struct LmOptions {
    /// Scale the normal equations using the clamped Hessian diagonal.
    pub jacobi_scaling: bool,
    /// Initial damping factor (0 behaves like Gauss-Newton).
    pub damping_init: f32,
    /// Min and max damping values.
    pub damping_range: [f32; 2],
    /// Damping scale for good steps.
    pub good_factor: f32,
    /// Damping scale for bad steps.
    pub bad_factor: f32,
}

impl Default for LmOptions {
    fn default() -> Self {
        Self {
            jacobi_scaling: true,
            damping_init: 1e-4,
            damping_range: [1e-9, 1e9],
            good_factor: 1.0 / 3.0,
            bad_factor: 2.0,
        }
    }
}

/// Gradient descent options.
#[derive(Clone, Debug)]
pub struct GdOptions {
    pub lr: f32,
}

impl Default for GdOptions {
    fn default() -> Self {
        Self { lr: 1e-3 }
    }
}

/// Conjugate gradient options.
#[derive(Clone, Debug)]
pub struct CgOptions {
    pub step_size: f32,
    pub step_reduction: f32,
}

impl Default for CgOptions {
    fn default() -> Self {
        Self {
            step_size: 0.25,
            step_reduction: 0.5,
        }
    }
}

/// Dog-leg options.
#[derive(Clone, Debug)]
pub struct DlOptions {
    pub radius_init: f32,
    pub radius_max: f32,
    pub shrink_factor: f32,
    pub expand_factor: f32,
}

impl Default for DlOptions {
    fn default() -> Self {
        Self {
            radius_init: 1.0,
            radius_max: 1e6,
            shrink_factor: 0.25,
            expand_factor: 2.0,
        }
    }
}

/// BFGS / L-BFGS options.
#[derive(Clone, Debug)]
pub struct QuasiNewtonOptions {
    pub step_size: f32,
    pub step_reduction: f32,
    pub step_growth: f32,
    pub max_step_size: f32,
    pub curvature_threshold: f32,
    /// Number of stored corrections (L-BFGS only).
    pub history_size: u8,
}

impl Default for QuasiNewtonOptions {
    fn default() -> Self {
        Self {
            step_size: 1.0,
            step_reduction: 0.5,
            step_growth: 1.5,
            max_step_size: 1.0,
            curvature_threshold: 1e-8,
            history_size: 8,
        }
    }
}

/// All optimization options.
#[derive(Clone, Debug)]
pub struct Options {
    /// Which solver to use.
    pub solver_type: Solver,
    /// Linear solver configuration.
    pub linear: LinearSolverOptions,
    /// Line search options.
    pub line_search: LineSearchOptions,
    /// Export options.
    pub exp: Export,
    /// Optimization behaviour.
    pub opt: Optimization,
    /// Hessian properties.
    pub hessian: HessianOptions,
    /// Cost scaling.
    pub cost: CostScaling,
    /// Stop criteria.
    pub stop: StopCriteria,
    /// Logging options.
    pub log: LogOptions,
    /// LM options.
    pub lm: LmOptions,
    /// GD options.
    pub gd: GdOptions,
    /// CG options.
    pub cg: CgOptions,
    /// Dog-leg options.
    pub dl: DlOptions,
    /// BFGS options.
    pub bfgs: QuasiNewtonOptions,
    /// L-BFGS options.
    pub lbfgs: QuasiNewtonOptions,
}

impl Default for Options {
    fn default() -> Self {
        Self::new(Solver::LevenbergMarquardt)
    }
}

impl Options {
    /// Options with the given solver and all C++ defaults.
    pub fn new(solver: Solver) -> Self {
        Self {
            solver_type: solver,
            linear: LinearSolverOptions::default(),
            line_search: LineSearchOptions::default(),
            exp: Export::default(),
            opt: Optimization::new(),
            hessian: HessianOptions::default(),
            cost: CostScaling::default(),
            stop: StopCriteria::new(),
            log: LogOptions::default(),
            lm: LmOptions::default(),
            gd: GdOptions::default(),
            cg: CgOptions::default(),
            dl: DlOptions::default(),
            bfgs: QuasiNewtonOptions::default(),
            lbfgs: QuasiNewtonOptions::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_cpp() {
        let o = Options::default();
        assert_eq!(o.solver_type, Solver::LevenbergMarquardt);
        assert_eq!(o.stop.max_iters, 50);
        assert_eq!(o.stop.min_error, 1e-12);
        assert_eq!(o.stop.min_rerr_dec, 1e-6);
        assert_eq!(o.stop.min_step_norm2, 1e-14);
        assert_eq!(o.stop.min_step_rel_norm2, 1e-16);
        assert_eq!(o.stop.min_grad_norm2, 1e-18);
        assert_eq!(o.stop.min_grad_max, 1e-10);
        assert_eq!(o.stop.max_consec_failures, 5);
        assert_eq!(o.lm.damping_init, 1e-4);
        assert!(o.lm.jacobi_scaling);
        assert_eq!(o.lm.damping_range, [1e-9, 1e9]);
        assert_eq!(o.gd.lr, 1e-3);
        assert_eq!(o.cg.step_size, 0.25);
        assert_eq!(o.dl.radius_init, 1.0);
        assert_eq!(o.bfgs.history_size, 8);
        assert_eq!(o.lbfgs.history_size, 8);
        assert!(o.exp.save_history);
        assert!(!o.exp.final_cov);
    }

    #[test]
    fn solver_selection() {
        let o = Options::new(Solver::GaussNewton);
        assert_eq!(o.solver_type, Solver::GaussNewton);
    }
}
