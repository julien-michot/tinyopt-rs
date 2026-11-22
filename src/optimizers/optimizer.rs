// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! Shared optimizer machinery: the accumulation function contract and the
//! common trust-region / line-search loop, mirroring tinyopt's
//! `OptimizerCore::OptimizeAcc`.
//!
//! The loop is generic over the Hessian storage (`HessianStorage`), so the same
//! code drives dense, sparse and block Hessians.

use num_traits::{FromPrimitive, One, ToPrimitive, Zero};
use std::time::Instant;

use crate::backend::{DenseVector, Dim, Matrix, Vector, zeros_vector};
use crate::cost::Cost;
use crate::covariance::{CovarianceOptions, FinalCovariance};
use crate::hessian::HessianStorage;
use crate::math::{all_finite, max_abs};
use crate::optimizers::fixed::FixedSet;
use crate::options::{LineSearchMethod, Options};
use crate::params::{Bounds, Params};
use crate::real::Real;
use crate::stop_reasons::StopReason;
use crate::summary::Summary;

/// User-provided cost + derivative accumulation function.
///
/// Mirrors the C++ accumulation callable `(x, grad, H) -> Cost`: the optimizer
/// requests only the outputs it needs, so the same function can be used for
/// evaluation (`None`, `None`), first-order steps, or second-order steps.
///
/// The Hessian parameter defaults to the dense backend matrix, so existing
/// closures compile unchanged; sparse and block accumulations name their
/// storage type explicitly.
pub trait AccumulationFn<S: Real, D: Dim, H: HessianStorage<S> = Matrix<S, D>> {
    /// Evaluate the cost at `x` and, when requested, the gradient and Hessian.
    fn eval(&self, x: &[S], grad: Option<&mut Vector<S, D>>, h: Option<&mut H>) -> Cost;

    /// Whether this accumulator uses numerical differentiation.
    fn uses_numerical_diff(&self) -> bool {
        false
    }
}

impl<S, D, H, F> AccumulationFn<S, D, H> for F
where
    S: Real,
    D: Dim,
    H: HessianStorage<S>,
    F: Fn(&[S], Option<&mut Vector<S, D>>, Option<&mut H>) -> Cost,
{
    fn eval(&self, x: &[S], grad: Option<&mut Vector<S, D>>, h: Option<&mut H>) -> Cost {
        self(x, grad, h)
    }
}

/// A computed step together with its predicted error reduction.
pub(crate) type Step<S, D> = (Vector<S, D>, f64);

/// Algorithm-specific step logic (LM damping, GN, trust region, ...).
pub(crate) trait Algorithm<P: Params, H: HessianStorage<P::Scalar>> {
    /// Initialize per-problem state.
    fn reset(&mut self, dims: usize);

    /// Whether the optimizer needs the Hessian approximation.
    fn needs_hessian(&self) -> bool {
        true
    }

    /// Number of reduced (head) parameters for the Schur solve.
    fn set_num_reduced(&mut self, _num_reduced: usize) {}

    /// Hessian prototype used to shape dynamic block layouts.
    fn hessian_prototype(&self) -> Option<&H> {
        None
    }

    /// Called after each accumulation, with the newly built gradient.
    fn on_build(&mut self, _grad: &Vector<P::Scalar, P::Dim>) {}

    /// Precondition the normal equations before solving (Jacobi scaling, damping).
    fn precondition(&mut self, grad: &mut Vector<P::Scalar, P::Dim>, h: &mut H);

    /// Solve the preconditioned normal equations.
    fn solve(&mut self, grad: &Vector<P::Scalar, P::Dim>, h: &H)
    -> Option<Step<P::Scalar, P::Dim>>;

    /// Called with the final step (after an optional line search).
    fn update_step(&mut self, _step: &Vector<P::Scalar, P::Dim>) {}

    /// Called after an accepted step.
    fn on_good_step(&mut self, quality: f64);

    /// Called after a rejected step.
    fn on_bad_step(&mut self);

    /// Fixed-parameter set.
    fn fixed(&self) -> &FixedSet;

    /// Record fixed indices coming from the parameter trait (if the explicit
    /// set was empty), so covariance export can zero their variances.
    fn set_fixed_indices(&mut self, _indices: &[usize]) {}

    /// Short solver state appended to the per-iteration log line.
    fn state_string(&self) -> String {
        String::new()
    }

    /// Store the final (un-damped) system for covariance export.
    fn save_final_system(&mut self, _h: &H, _cost: Cost) {}

    /// Covariance approximation from the final system, if available.
    fn covariance(&self, _options: &CovarianceOptions) -> Option<FinalCovariance> {
        None
    }
}

/// Apply the C++ `Options::Cost` scaling to an evaluated cost.
fn normalize_cost(cost: &mut Cost, scaling: &crate::options::CostScaling) {
    if !scaling.use_squared_norm {
        cost.cost = cost.cost.max(0.0).sqrt();
    }
    if scaling.downscale_by_2 {
        cost.cost *= 0.5;
    }
    if scaling.normalize && cost.num_residuals > 0 {
        cost.cost /= cost.num_residuals as f64;
    }
}

fn to_f64<S: Real>(v: S) -> f64 {
    v.to_f64().expect("scalar fits in f64")
}

fn notify_step(options: &Options, dx: &[f32], is_rollback: bool) -> bool {
    match &options.stop.step_callback {
        Some(cb) => cb(dx, is_rollback),
        None => false,
    }
}

/// Run the optimization loop for a given algorithm.
pub(crate) fn run<P, F, A, H>(x: &mut P, f: F, options: &Options, algo: &mut A) -> Summary
where
    P: Params,
    H: HessianStorage<P::Scalar>,
    F: AccumulationFn<P::Scalar, P::Dim, H>,
    A: Algorithm<P, H>,
{
    let mut sum = Summary::default();
    let start = Instant::now();
    let dims = P::dims(x);

    if dims == 0 {
        sum.stop_reason = StopReason::Skipped;
        return sum;
    }

    let mut grad = zeros_vector::<P::Scalar, P::Dim>(dims);
    let mut h = match algo.hessian_prototype() {
        Some(prototype) => H::zeros_like(prototype, dims),
        None => H::zeros(dims),
    };
    algo.reset(dims);

    // Reduced (Schur) block: the C++ contract requires it at the head.
    let num_reduced = match x.reduced() {
        Some((0, count)) => count.min(dims),
        Some((start, _)) => {
            log::warn!("reduced parameters must start at 0 (got {start}); disabling Schur");
            0
        }
        None => 0,
    };
    algo.set_num_reduced(num_reduced);

    // Fixed parameters: explicit `fix()` indices take precedence over the
    // `Params::fixed()` trait method, matching the C++ API (with a warning).
    let explicit_fixed = algo.fixed().indices().to_vec();
    let trait_fixed = x.fixed().unwrap_or_default();
    if !explicit_fixed.is_empty() && !trait_fixed.is_empty() {
        log::warn!("both fix() and Params::fixed() are set; using the explicit indices");
    }
    let fixed = if !explicit_fixed.is_empty() {
        explicit_fixed
    } else {
        trait_fixed
    };
    algo.set_fixed_indices(&fixed);

    let bounds = x.bounds().filter(|b| b.dims() == dims);
    if let Some(bounds) = &bounds {
        project_params(x, bounds);
    }

    let mut cost = f.eval(P::as_slice(x), None, None);
    normalize_cost(&mut cost, &options.cost);
    if cost.num_residuals == 0 {
        sum.stop_reason = StopReason::Skipped;
        return sum;
    }
    if !cost.cost.is_finite() {
        sum.stop_reason = StopReason::SystemHasNanOrInf;
        return sum;
    }
    sum.num_residuals = cost.num_residuals as u16;
    sum.final_cost = cost;
    sum.num_diff_used = f.uses_numerical_diff();

    let max_iters = options.stop.max_iters as usize;
    let mut reason = StopReason::None;
    let mut h_diag = zeros_vector::<P::Scalar, P::Dim>(dims);
    let mut active: Vec<usize> = Vec::new();

    for _iter in 0..max_iters {
        let iter = sum.num_iters;
        // --- Build the linear system at the current parameters ---------------
        DenseVector::set_zero(&mut grad);
        h.set_zero();
        let mut build = if algo.needs_hessian() {
            f.eval(P::as_slice(x), Some(&mut grad), Some(&mut h))
        } else {
            f.eval(P::as_slice(x), Some(&mut grad), None)
        };
        normalize_cost(&mut build, &options.cost);

        if build.num_residuals == 0 {
            reason = StopReason::Skipped;
            break;
        }
        if !build.cost.is_finite()
            || !all_finite(DenseVector::as_slice(&grad))
            || !h.is_all_finite()
        {
            reason = StopReason::SystemHasNanOrInf;
            break;
        }
        // Optional gradient clipping (element-wise, zero disables).
        if options.opt.grad_clipping > 0.0 {
            let limit = P::Scalar::from_f64(options.opt.grad_clipping as f64).expect("fits");
            for v in DenseVector::as_mut_slice(&mut grad) {
                if *v > limit {
                    *v = limit;
                } else if *v < -limit {
                    *v = -limit;
                }
            }
        }
        // Reject systems whose Hessian diagonal is numerically zero.
        if options.hessian.check_min_h_diag > 0.0 && algo.needs_hessian() {
            h.diagonal_into(DenseVector::as_mut_slice(&mut h_diag));
            let threshold =
                P::Scalar::from_f64(options.hessian.check_min_h_diag as f64).expect("fits");
            if DenseVector::as_slice(&h_diag)
                .iter()
                .any(|d| num_traits::Float::abs(*d) < threshold)
            {
                if options.log.enable {
                    log::debug!("#{}: Hessian diagonal below {threshold:?}", iter);
                }
                reason = StopReason::SolverFailed;
                break;
            }
        }
        sum.num_residuals = build.num_residuals as u16;
        sum.final_cost = build;
        cost = build;
        algo.on_build(&grad);

        // Mask fixed parameters and variables blocked by an active bound: zero
        // gradient, zero row/column, unit diagonal.
        let masked: &[usize] = match &bounds {
            Some(bounds) => {
                active.clear();
                active.extend_from_slice(&fixed);
                let xs = P::as_slice(x);
                for (index, value) in xs.iter().enumerate().take(dims.min(bounds.dims())) {
                    let gradient = grad.at(index);
                    let blocked = (gradient > P::Scalar::zero() && bounds.at_lower(index, *value))
                        || (gradient < P::Scalar::zero() && bounds.at_upper(index, *value));
                    if blocked {
                        active.push(index);
                    }
                }
                &active
            }
            None => &fixed,
        };
        for &idx in masked {
            if idx < dims {
                grad.set_at(idx, P::Scalar::zero());
            }
        }
        h.apply_fixed_indices(masked);

        // --- First-order stopping criteria ----------------------------------
        let grad_norm2 = to_f64(DenseVector::norm2(&grad));
        let grad_max = to_f64(max_abs(DenseVector::as_slice(&grad)));
        if options.stop.min_grad_max > 0.0 && grad_max < options.stop.min_grad_max as f64 {
            reason = StopReason::MinGradNorm;
            break;
        }
        if options.stop.min_grad_norm2 > 0.0 && grad_norm2 < options.stop.min_grad_norm2 as f64 {
            reason = StopReason::MinGradNorm;
            break;
        }
        if options.stop.min_error > 0.0 && build.cost < options.stop.min_error as f64 {
            reason = StopReason::MinError;
            break;
        }

        // --- Solve for the step ---------------------------------------------
        algo.precondition(&mut grad, &mut h);
        let Some((mut dx, predicted)) = algo.solve(&grad, &h) else {
            if options.log.print_failure {
                log::error!(
                    "❌ #{}: failed to solve the linear system\n  solver: {}\n  grad = {:?}\n  H = {:?}",
                    iter,
                    algo.state_string(),
                    grad,
                    h
                );
            }
            reason = StopReason::SolverFailed;
            break;
        };
        if !all_finite(DenseVector::as_slice(&dx)) {
            if options.log.print_failure {
                log::error!(
                    "❌ #{}: non-finite step\n  solver: {}\n  dx = {:?}\n  grad = {:?}",
                    iter,
                    algo.state_string(),
                    dx,
                    grad
                );
            }
            reason = StopReason::SystemHasNanOrInf;
            break;
        }

        if options.line_search.method != LineSearchMethod::None {
            dx = line_search(x, &dx, build.cost, &grad, &f, options);
        }
        if let Some(bounds) = &bounds {
            project_step::<P, P::Dim>(x, bounds, &mut dx);
        }
        algo.update_step(&dx);

        let dx_norm2 = to_f64(DenseVector::norm2(&dx));
        let x_norm2 = to_f64(P::norm2(x));
        let rel_norm2 = dx_norm2 / (x_norm2 + f64::EPSILON);
        sum.final_step_rel_norm2 = rel_norm2;

        if options.stop.min_step_norm2 > 0.0 && dx_norm2 < options.stop.min_step_norm2 as f64 {
            reason = StopReason::MinDeltaNorm;
            break;
        }
        if options.stop.min_step_rel_norm2 > 0.0
            && rel_norm2 < options.stop.min_step_rel_norm2 as f64
        {
            reason = StopReason::MinRelDeltaNorm;
            break;
        }

        // --- Trial step and acceptance --------------------------------------
        let needs_f32 =
            options.stop.step_callback.is_some() || options.stop.stop_callback2.is_some();
        let dx_f32: Vec<f32> = if needs_f32 {
            DenseVector::as_slice(&dx)
                .iter()
                .map(|v| v.to_f32().expect("scalar fits in f32"))
                .collect()
        } else {
            Vec::new()
        };

        P::plus_eq(x, DenseVector::as_slice(&dx));
        if notify_step(options, &dx_f32, false) {
            reason = StopReason::UserStopped;
        }

        let mut trial = f.eval(P::as_slice(x), None, None);
        normalize_cost(&mut trial, &options.cost);
        let accepted = trial.cost.is_finite() && trial.cost < build.cost;

        if accepted {
            let derr = build.cost - trial.cost;
            let quality = if options.opt.use_step_quality {
                if predicted > 0.0 {
                    derr / predicted
                } else {
                    0.0
                }
            } else if options.opt.use_step_quality_approx {
                if build.cost > f64::EPSILON {
                    derr / build.cost
                } else {
                    0.0
                }
            } else {
                1.0
            };
            algo.on_good_step(quality);

            let rerr = if build.cost > f64::EPSILON {
                derr / build.cost
            } else {
                0.0
            };
            sum.final_rerr_dec = rerr;
            sum.final_cost = trial;
            sum.num_iters = sum.num_iters.saturating_add(1);
            sum.num_consec_failures = 0;
            if options.exp.save_history {
                sum.hist.add(trial.cost, dx_norm2, true);
            }
            if reason == StopReason::None
                && options.stop.min_rerr_dec > 0.0
                && _iter > 0
                && rerr < options.stop.min_rerr_dec as f64
            {
                reason = StopReason::MinRelError;
            }
        } else {
            let mut neg_dx = dx.clone();
            DenseVector::scale(&mut neg_dx, -P::Scalar::one());
            let neg_f32: Vec<f32> = if needs_f32 {
                dx_f32.iter().map(|v| -v).collect()
            } else {
                Vec::new()
            };
            P::plus_eq(x, DenseVector::as_slice(&neg_dx));
            if notify_step(options, &neg_f32, true) {
                reason = StopReason::UserStopped;
            }
            algo.on_bad_step();

            sum.num_failures = sum.num_failures.saturating_add(1);
            sum.num_consec_failures = sum.num_consec_failures.saturating_add(1);
            if options.exp.save_history {
                sum.hist.add(build.cost, dx_norm2, false);
            }
            if options.stop.max_consec_failures > 0
                && sum.num_consec_failures >= options.stop.max_consec_failures
            {
                reason = StopReason::MaxConsecNoDecr;
            } else if options.stop.max_total_failures > 0
                && sum.num_failures >= options.stop.max_total_failures
            {
                reason = StopReason::MaxNoDecr;
            }
        }

        // --- Per-iteration log (one line, C++ format) -----------------------
        if options.log.enable {
            let state = algo.state_string();
            log_iteration(
                options,
                &IterationLog {
                    iter,
                    num_reduced,
                    accepted,
                    prev_cost: build.cost,
                    trial_cost: trial.cost,
                    num_residuals: build.num_residuals as u16,
                    inlier_ratio: trial.inlier_ratio,
                    dx_norm2,
                    grad_norm2,
                    dx: DenseVector::as_slice(&dx),
                    x: P::as_slice(x),
                    elapsed_ms: start.elapsed().as_secs_f64() * 1e3,
                    state: &state,
                },
            );
        }

        // --- User and time based stopping criteria --------------------------
        if reason == StopReason::None {
            if let Some(cb) = &options.stop.stop_callback {
                if cb(cost.cost, dx_norm2, grad_norm2) {
                    reason = StopReason::UserStopped;
                }
            }
            if reason == StopReason::None {
                if let Some(cb) = &options.stop.stop_callback2 {
                    if needs_f32 {
                        let grad_f32: Vec<f32> = DenseVector::as_slice(&grad)
                            .iter()
                            .map(|v| v.to_f32().expect("scalar fits in f32"))
                            .collect();
                        if cb(cost.cost, &dx_f32, &grad_f32) {
                            reason = StopReason::UserStopped;
                        }
                    }
                }
            }
        }

        if reason == StopReason::None && options.stop.max_duration_ms > 0.0 {
            let elapsed = start.elapsed().as_secs_f64() * 1e3;
            if elapsed > options.stop.max_duration_ms {
                reason = StopReason::TimedOut;
            }
        }

        if reason != StopReason::None {
            break;
        }
    }

    sum.duration_ms = start.elapsed().as_secs_f64() as f32 * 1e3;
    sum.stop_reason = if reason == StopReason::None && sum.num_iters as usize >= max_iters {
        StopReason::MaxIters
    } else {
        reason
    };

    // Store the final system so that covariance is available to optimizer
    // objects even when no export was requested.
    if algo.needs_hessian() {
        let mut final_h = match algo.hessian_prototype() {
            Some(prototype) => H::zeros_like(prototype, dims),
            None => H::zeros(dims),
        };
        let _ = f.eval(P::as_slice(x), None, Some(&mut final_h));
        final_h.apply_fixed_indices(&fixed);
        algo.save_final_system(&final_h, sum.final_cost);
    } else {
        algo.save_final_system(&H::zeros(dims), sum.final_cost);
    }

    if options.exp.final_cov {
        if let Some(cov) = algo.covariance(&options.exp.cov_options) {
            sum.final_cov = Some(cov);
        }
    }

    if options.log.enable && sum.stop_reason != StopReason::None {
        log::info!(
            "{}",
            sum.stop_reason
                .description(Some(options), sum.final_cost.cost)
        );
    }
    sum
}

/// Solve `H dx = -grad` with the given Hessian storage.
pub(crate) fn neg_solve<S: Real, D: Dim, H: HessianStorage<S>>(
    options: &Options,
    grad: &Vector<S, D>,
    h: &H,
    num_reduced: usize,
    cache: &mut H::SolverCache,
    fixed: &FixedSet,
) -> Option<Vector<S, D>> {
    let mut neg_grad = grad.clone();
    DenseVector::scale(&mut neg_grad, -S::one());
    let mut out = zeros_vector::<S, D>(grad.len());

    // Fixed-parameter permutation optimization (dense dynamic Hessians): solve
    // the free sub-system when enough parameters are fixed.
    let n = grad.len();
    if options.opt.min_fixed_ratio > 0.0 && !fixed.is_empty() && n > 0 && num_reduced == 0 {
        let ratio = fixed.len() as f32 / n as f32;
        if ratio >= options.opt.min_fixed_ratio {
            let free: Vec<usize> = (0..n).filter(|i| !fixed.contains(*i)).collect();
            if h.solve_free_into(
                &options.linear,
                DenseVector::as_slice(&neg_grad),
                &free,
                DenseVector::as_mut_slice(&mut out),
                cache,
            ) {
                return Some(out);
            }
        }
    }

    let ok = h.solve_into(
        &options.linear,
        DenseVector::as_slice(&neg_grad),
        num_reduced,
        DenseVector::as_mut_slice(&mut out),
        cache,
    );
    if ok { Some(out) } else { None }
}

/// Scale a step by `alpha`.
fn scaled_step<S: Real, D: Dim>(alpha: f64, dx: &Vector<S, D>) -> Vector<S, D> {
    let mut out = dx.clone();
    let a = S::from_f64(alpha).expect("step multiplier fits");
    DenseVector::scale(&mut out, a);
    out
}

/// Project the parameters into their bounds (no-op when already feasible).
fn project_params<P: Params>(x: &mut P, bounds: &Bounds<P::Scalar>) {
    let xs = P::as_slice(x);
    let mut delta: Option<Vec<P::Scalar>> = None;
    for (index, value) in xs.iter().enumerate().take(bounds.dims()) {
        let clamped = bounds.clamp(index, *value);
        if clamped != *value {
            let delta = delta.get_or_insert_with(|| vec![P::Scalar::zero(); xs.len()]);
            delta[index] = clamped - *value;
        }
    }
    if let Some(delta) = delta {
        P::plus_eq(x, &delta);
    }
}

/// Project a step so that `x + dx` stays inside the bounds.
fn project_step<P: Params, D: Dim>(
    x: &P,
    bounds: &Bounds<P::Scalar>,
    dx: &mut Vector<P::Scalar, D>,
) {
    let xs = P::as_slice(x);
    let values = DenseVector::as_mut_slice(dx);
    for (index, value) in values.iter_mut().enumerate().take(bounds.dims()) {
        let clamped = bounds.clamp(index, xs[index] + *value);
        *value = clamped - xs[index];
    }
}

/// Cost at `x + alpha * dx`, leaving `x` unchanged.
fn trial_cost<P, F, H>(
    x: &mut P,
    alpha: f64,
    dx: &Vector<P::Scalar, P::Dim>,
    f: &F,
    scaling: &crate::options::CostScaling,
) -> f64
where
    P: Params,
    H: HessianStorage<P::Scalar>,
    F: AccumulationFn<P::Scalar, P::Dim, H>,
{
    let step = scaled_step::<P::Scalar, P::Dim>(alpha, dx);
    P::plus_eq(x, DenseVector::as_slice(&step));
    let mut cost = f.eval(P::as_slice(x), None, None);
    normalize_cost(&mut cost, scaling);
    let cost = cost.cost;
    let mut back = step;
    DenseVector::scale(&mut back, -P::Scalar::one());
    P::plus_eq(x, DenseVector::as_slice(&back));
    cost
}

/// Backtracking / Wolfe / golden-section search along `dx`.
#[allow(clippy::too_many_arguments)]
fn line_search<P, F, H>(
    x: &mut P,
    dx: &Vector<P::Scalar, P::Dim>,
    f0: f64,
    grad: &Vector<P::Scalar, P::Dim>,
    f: &F,
    options: &Options,
) -> Vector<P::Scalar, P::Dim>
where
    P: Params,
    H: HessianStorage<P::Scalar>,
    F: AccumulationFn<P::Scalar, P::Dim, H>,
{
    let directional = to_f64(DenseVector::dot(grad, dx));
    if directional >= 0.0 || !directional.is_finite() {
        return dx.clone();
    }

    let ls = &options.line_search;
    let c1 = ls.c1 as f64;
    let min_step = ls.min_step as f64;
    let mut alpha = ls.initial_step as f64;
    let mut best_alpha = alpha;

    match ls.method {
        LineSearchMethod::None => return dx.clone(),
        LineSearchMethod::Backtracking => {
            for _ in 0..ls.max_iters {
                let f_trial = trial_cost::<P, F, H>(x, alpha, dx, f, &options.cost);
                if f_trial.is_finite() && f_trial <= f0 + c1 * alpha * directional {
                    best_alpha = alpha;
                    break;
                }
                alpha *= ls.contraction as f64;
                if alpha < min_step {
                    best_alpha = alpha;
                    break;
                }
            }
        }
        LineSearchMethod::Wolfe => {
            for _ in 0..ls.max_iters {
                let f_trial = trial_cost::<P, F, H>(x, alpha, dx, f, &options.cost);
                if f_trial.is_finite() && f_trial <= f0 + c1 * alpha * directional {
                    best_alpha = alpha;
                    break;
                }
                let mut next_alpha = alpha * ls.contraction as f64;
                let denom = 2.0 * (f_trial - f0 - alpha * directional);
                if f_trial.is_finite() && denom > 0.0 {
                    let quad_alpha = -directional * alpha * alpha / denom;
                    next_alpha = quad_alpha.clamp(0.1 * alpha, 0.5 * alpha);
                }
                alpha = next_alpha;
                if alpha < min_step {
                    best_alpha = alpha;
                    break;
                }
            }
        }
        LineSearchMethod::GoldenSection => {
            const INVPHI: f64 = 0.618_033_988_749_895;
            const INVPHI2: f64 = 0.381_966_011_250_105;
            let mut a = 0.0;
            let mut h = ls.initial_step as f64;
            if h > min_step {
                let mut c = a + INVPHI2 * h;
                let mut d = a + INVPHI * h;
                let mut fc = trial_cost::<P, F, H>(x, c, dx, f, &options.cost);
                let mut fd = trial_cost::<P, F, H>(x, d, dx, f, &options.cost);
                for _ in 0..ls.max_iters {
                    if fc < fd {
                        d = c;
                        fd = fc;
                        h *= INVPHI;
                        c = a + INVPHI2 * h;
                        fc = trial_cost::<P, F, H>(x, c, dx, f, &options.cost);
                    } else {
                        a = c;
                        c = d;
                        fc = fd;
                        h *= INVPHI;
                        d = a + INVPHI * h;
                        fd = trial_cost::<P, F, H>(x, d, dx, f, &options.cost);
                    }
                    if h < min_step {
                        break;
                    }
                }
                best_alpha = if fc < fd { c } else { d };
            }
        }
    }

    scaled_step::<P::Scalar, P::Dim>(best_alpha, dx)
}

/// One iteration of the per-iteration log line.
struct IterationLog<'a, S> {
    iter: u16,
    num_reduced: usize,
    accepted: bool,
    prev_cost: f64,
    trial_cost: f64,
    num_residuals: u16,
    inlier_ratio: f32,
    dx_norm2: f64,
    grad_norm2: f64,
    dx: &'a [S],
    x: &'a [S],
    elapsed_ms: f64,
    state: &'a str,
}

/// Emit the one-line iteration log, mirroring the C++ `TINYOPT_LOG` format.
fn log_iteration<S: Real>(options: &Options, it: &IterationLog<'_, S>) {
    use std::fmt::Write;

    let log = &options.log;
    let mut line = String::with_capacity(160);
    if log.print_emoji {
        line.push_str(if !it.accepted {
            "❌"
        } else if it.iter == 0 {
            "ℹ️"
        } else {
            "✅"
        });
    }
    let _ = write!(line, "#{} ", it.iter);
    if log.print_x {
        line.push_str("x:[");
        for (i, v) in it.x.iter().enumerate() {
            if i > 0 {
                line.push_str(", ");
            }
            let _ = write!(line, "{:.5}", to_f64(*v));
        }
        line.push_str("] ");
    } else {
        let _ = write!(line, "x:ℝ^{} ", it.x.len());
    }
    if it.num_reduced > 0 {
        let _ = write!(line, "r:ℝ^{} ", it.num_reduced);
    }

    let derr = it.trial_cost - it.prev_cost;
    let rerr = if it.prev_cost > f64::EPSILON {
        -derr / it.prev_cost
    } else {
        -1.0
    };
    let _ = write!(
        line,
        "{}:{:.4e} n:{} d{}:{:+.2e} r{}:{:+.1e} ",
        log.e, it.prev_cost, it.num_residuals, log.e, derr, log.e, rerr
    );
    let _ = write!(line, "|δx|:{:.2e} ", it.dx_norm2.sqrt());
    if log.print_dx {
        line.push_str("δx:[");
        for (i, v) in it.dx.iter().enumerate() {
            if i > 0 {
                line.push_str(", ");
            }
            let _ = write!(line, "{:+.4}", to_f64(*v));
        }
        line.push_str("] ");
    }
    if options.stop.min_grad_norm2 > 0.0 || options.stop.min_grad_max > 0.0 {
        let _ = write!(line, "|∇|:{:.2e} ", it.grad_norm2.sqrt());
    }
    line.push_str(it.state);
    if log.print_inliers {
        let _ = write!(line, "in:{:.2}% ", it.inlier_ratio * 100.0);
    }
    if log.print_t {
        let _ = write!(line, "τ:{:.2} ", it.elapsed_ms);
    }
    log::info!("{}", line.trim_end());
}
