// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! Box-constrained optimization (active-set projection).

use approx::assert_relative_eq;
use tinyopt::backend::{DenseMatrix, DenseVector, Dynamic, Matrix, Vector};
use tinyopt::{Bounds, Cost, Options, Params, StopReason, optimize};

type V = Vector<f64, Dynamic>;
type M = Matrix<f64, Dynamic>;

#[derive(Clone, Debug)]
struct Bounded {
    values: Vec<f64>,
    lower: Vec<f64>,
    upper: Vec<f64>,
    fixed: Vec<usize>,
    reduced: Option<(usize, usize)>,
}

impl Bounded {
    fn new(values: &[f64], lower: &[f64], upper: &[f64]) -> Self {
        Self {
            values: values.to_vec(),
            lower: lower.to_vec(),
            upper: upper.to_vec(),
            fixed: Vec::new(),
            reduced: None,
        }
    }

    fn fixed(mut self, fixed: &[usize]) -> Self {
        self.fixed = fixed.to_vec();
        self
    }

    fn reduced(mut self, reduced: Option<(usize, usize)>) -> Self {
        self.reduced = reduced;
        self
    }
}

impl Params for Bounded {
    type Scalar = f64;
    type Dim = Dynamic;

    fn dims(&self) -> usize {
        self.values.len()
    }

    fn as_slice(&self) -> &[f64] {
        &self.values
    }

    fn plus_eq(&mut self, delta: &[f64]) {
        for (value, d) in self.values.iter_mut().zip(delta) {
            *value += d;
        }
    }

    fn norm2(&self) -> f64 {
        self.values.iter().map(|v| v * v).sum()
    }

    fn bounds(&self) -> Option<Bounds<f64>> {
        Some(Bounds::from_slices(&self.lower, &self.upper))
    }

    fn fixed(&self) -> Option<Vec<usize>> {
        (!self.fixed.is_empty()).then(|| self.fixed.clone())
    }

    fn reduced(&self) -> Option<(usize, usize)> {
        self.reduced
    }
}

/// Quadratic `0.5 Σ (x_i - target_i)²` with identity Hessian.
fn quadratic(target: Vec<f64>) -> impl Fn(&[f64], Option<&mut V>, Option<&mut M>) -> Cost {
    move |x: &[f64], grad: Option<&mut V>, h: Option<&mut M>| {
        let mut cost = 0.0;
        if let Some(g) = grad {
            for i in 0..x.len() {
                let r = x[i] - target[i];
                cost += 0.5 * r * r;
                g.set_at(i, r);
            }
        } else {
            for i in 0..x.len() {
                cost += 0.5 * (x[i] - target[i]).powi(2);
            }
        }
        if let Some(h) = h {
            for i in 0..x.len() {
                h.set_elem(i, i, 1.0);
            }
        }
        Cost::with_num_residuals(cost, x.len() as i32)
    }
}

fn options() -> Options {
    let mut options = Options::default();
    options.stop.max_iters = 100;
    options.stop.min_grad_max = 1e-9;
    options.log.enable = false;
    options
}

#[test]
fn optimum_clamped_to_an_active_bound() {
    let mut x = Bounded::new(&[0.0], &[0.0], &[1.0]);
    let summary = optimize(&mut x, quadratic(vec![2.0]), &options());
    assert!(summary.succeeded(), "{:?}", summary.stop_reason);
    assert_relative_eq!(x.values[0], 1.0, epsilon = 1e-12);
    assert_eq!(summary.stop_reason, StopReason::MinGradNorm);
}

#[test]
fn each_coordinate_hits_its_own_bound() {
    let mut x = Bounded::new(&[0.0, 0.0], &[0.0, -3.0], &[3.0, 0.0]);
    let summary = optimize(&mut x, quadratic(vec![5.0, -5.0]), &options());
    assert!(summary.succeeded(), "{:?}", summary.stop_reason);
    assert_relative_eq!(x.values[0], 3.0, epsilon = 1e-12);
    assert_relative_eq!(x.values[1], -3.0, epsilon = 1e-12);
}

#[test]
fn only_the_constrained_coordinates_are_blocked() {
    // x0 is pushed to its upper bound, x1 converges to its interior optimum.
    let mut x = Bounded::new(&[0.0, 0.0], &[0.0, 0.0], &[1.5, 10.0]);
    let summary = optimize(&mut x, quadratic(vec![4.0, 1.0]), &options());
    assert!(summary.succeeded(), "{:?}", summary.stop_reason);
    assert_relative_eq!(x.values[0], 1.5, epsilon = 1e-12);
    assert_relative_eq!(x.values[1], 1.0, epsilon = 1e-6);
}

#[test]
fn an_infeasible_start_is_projected() {
    let mut x = Bounded::new(&[-10.0, 42.0], &[0.0, -1.0], &[1.0, 1.0]);
    let summary = optimize(&mut x, quadratic(vec![0.5, 0.5]), &options());
    assert!(summary.succeeded(), "{:?}", summary.stop_reason);
    assert_relative_eq!(x.values[0], 0.5, epsilon = 1e-6);
    assert_relative_eq!(x.values[1], 0.5, epsilon = 1e-6);
}

#[test]
fn unbounded_coordinates_stay_unconstrained() {
    let mut x = Bounded::new(&[0.0], &[f64::NEG_INFINITY], &[f64::INFINITY]);
    let summary = optimize(&mut x, quadratic(vec![7.0]), &options());
    assert!(summary.succeeded(), "{:?}", summary.stop_reason);
    assert_relative_eq!(x.values[0], 7.0, epsilon = 1e-6);
}

#[test]
fn projected_gradient_vanishes_at_the_solution() {
    // Cost (x - 2)² on [0, 1]: at x = 1 the raw gradient is -2, but the
    // projected gradient (used by the stopping test) is zero.
    let mut x = Bounded::new(&[0.0], &[0.0], &[1.0]);
    let summary = optimize(&mut x, quadratic(vec![2.0]), &options());
    assert!(summary.succeeded());
    // The stopping test only sees the projected gradient, so the solver stops
    // at the bound instead of chasing the unconstrained optimum.
    assert_eq!(summary.stop_reason, StopReason::MinGradNorm);
    assert_relative_eq!(x.values[0], 1.0, epsilon = 1e-12);
}

#[test]
fn bounds_helpers_behave() {
    let bounds = Bounds::from_slices(&[0.0, -1.0], &[1.0, 1.0]);
    assert_relative_eq!(bounds.clamp(0, 4.0), 1.0, epsilon = 0.0);
    assert_relative_eq!(bounds.clamp(1, -4.0), -1.0, epsilon = 0.0);
    assert!(bounds.at_lower(0, 0.0));
    assert!(bounds.at_upper(0, 1.0));
    assert!(!bounds.at_lower(0, 0.5));
    let unbounded = Bounds::<f64>::unbounded(2);
    assert!(unbounded.lower[0].is_infinite() && unbounded.upper[0].is_infinite());
}

#[test]
#[should_panic(expected = "lower bound exceeds upper bound")]
fn invalid_bounds_panic() {
    let _ = Bounds::from_slices(&[1.0], &[0.0]);
}

// --- Bounds across Hessian storages -----------------------------------------

use tinyopt::block_sparse::BlockSparseMat;
use tinyopt::block_tridiag::BlockTridiagMat;
use tinyopt::sparse::SparseMat;

fn quadratic_bounds(target: &[f64], lower: &[f64], upper: &[f64]) -> Bounded {
    Bounded::new(&vec![0.0; target.len()], lower, upper)
}

/// Sparse Hessian (`optimize_sparse`).
#[test]
fn bounds_with_a_sparse_hessian() {
    let n = 20;
    let target = vec![1.0; n];
    let mut x = quadratic_bounds(&target, &[0.0; 20], &[10.0; 20]);
    x.lower[0] = 0.0;
    x.upper[0] = 0.5;
    let acc = |xv: &[f64], mut grad: Option<&mut V>, mut h: Option<&mut SparseMat>| {
        let mut cost = 0.0;
        for (i, value) in xv.iter().enumerate().take(n) {
            let r = value - 1.0;
            cost += 0.5 * r * r;
            if let Some(g) = grad.as_mut() {
                g.set_at(i, g.at(i) + r);
            }
            if let Some(h) = h.as_mut() {
                *h.coeff_ref(i, i) += 1.0;
            }
        }
        Cost::with_num_residuals(cost, n as i32)
    };
    let summary = tinyopt::optimize_sparse(&mut x, acc, &options());
    assert!(summary.succeeded(), "{:?}", summary.stop_reason);
    assert_relative_eq!(x.values[0], 0.5, epsilon = 1e-6);
    for value in &x.values[1..] {
        assert_relative_eq!(*value, 1.0, epsilon = 1e-5);
    }
}

/// Block-tridiagonal Hessian (`optimize_tridiag`), with a fixed index above the
/// bound to exercise the sorted active set.
#[test]
fn bounds_with_a_block_tridiagonal_hessian() {
    const B: usize = 2;
    const N: usize = 8;
    let dims = N * B;
    let mut x = Bounded::new(&vec![0.0; dims], &vec![0.0; dims], &vec![10.0; dims]).fixed(&[10]);
    x.upper[3] = 0.25;
    let acc =
        |xv: &[f64], mut grad: Option<&mut V>, mut h: Option<&mut BlockTridiagMat<B, f64>>| {
            let mut cost = 0.0;
            for (i, value) in xv.iter().enumerate().take(dims) {
                let r = value - 1.0;
                cost += 0.5 * r * r;
                if let Some(g) = grad.as_mut() {
                    g.set_at(i, g.at(i) + r);
                }
                if let Some(h) = h.as_mut() {
                    let block = i / B;
                    let k = i % B;
                    let mut e = [0.0; B];
                    e[k] = 1.0;
                    h.diagonal(block).add_outer_product_self(1.0, &e);
                }
            }
            Cost::with_num_residuals(cost, dims as i32)
        };
    let summary = tinyopt::optimize_tridiag::<B, _, _>(&mut x, acc, &options());
    assert!(summary.succeeded(), "{:?}", summary.stop_reason);
    assert_relative_eq!(x.values[3], 0.25, epsilon = 1e-6);
    assert_relative_eq!(x.values[10], 0.0, epsilon = 1e-12);
    for (i, value) in x.values.iter().enumerate() {
        if i != 3 && i != 10 {
            assert_relative_eq!(*value, 1.0, epsilon = 1e-5);
        }
    }
}

/// Block-sparse Hessian with a Schur-reduced head block.
#[test]
fn bounds_with_a_block_sparse_schur() {
    tinyopt::block_sparse_types!(BaBounds, [6, 3]);
    const CAMS: usize = 2;
    const POINTS: usize = 3;
    let dims = CAMS * 6 + POINTS * 3;
    let mut x = Bounded::new(&vec![0.0; dims], &vec![0.0; dims], &vec![10.0; dims])
        .reduced(Some((0, CAMS * 6)));
    x.upper[0] = 0.5;
    let acc = |xv: &[f64],
               mut grad: Option<&mut V>,
               mut h: Option<&mut BlockSparseMat<BaBounds, f64>>| {
        let mut cost = 0.0;
        for (i, value) in xv.iter().enumerate().take(dims) {
            let r = value - 1.0;
            cost += 0.5 * r * r;
            if let Some(g) = grad.as_mut() {
                g.set_at(i, g.at(i) + r);
            }
        }
        if let Some(h) = h.as_mut() {
            for cam in 0..CAMS {
                for k in 0..6 {
                    let mut e = [0.0; 6];
                    e[k] = 1.0;
                    h.block_mut::<6, 6>(cam, cam)
                        .add_outer_product_self(1.0, &e);
                }
            }
            for point in 0..POINTS {
                for k in 0..3 {
                    let mut e = [0.0; 3];
                    e[k] = 1.0;
                    h.block_mut::<3, 3>(point, point)
                        .add_outer_product_self(1.0, &e);
                }
            }
            // Couple each camera to each point so the Schur path is exercised.
            for cam in 0..CAMS {
                for point in 0..POINTS {
                    let mut block = h.block_mut::<6, 3>(cam, point);
                    for i in 0..6 {
                        block.set_at(i, 0, 0.01);
                    }
                }
            }
        }
        Cost::with_num_residuals(cost, dims as i32)
    };
    let summary = tinyopt::optimize_block_sparse::<BaBounds, _, _>(&mut x, acc, &options());
    assert!(summary.succeeded(), "{:?}", summary.stop_reason);
    assert_relative_eq!(x.values[0], 0.5, epsilon = 1e-5);
}

/// Sparse Hessian with a Schur-reduced head block.
#[test]
fn bounds_with_a_sparse_schur() {
    let n = 24;
    let mut x = Bounded::new(&vec![0.0; n], &vec![0.0; n], &vec![10.0; n]).reduced(Some((0, 8)));
    x.upper[2] = 0.5;
    let acc = |xv: &[f64], mut grad: Option<&mut V>, mut h: Option<&mut SparseMat>| {
        let mut cost = 0.0;
        for (i, value) in xv.iter().enumerate().take(n) {
            let r = value - 1.0;
            cost += 0.5 * r * r;
            if let Some(g) = grad.as_mut() {
                g.set_at(i, g.at(i) + r);
            }
            if let Some(h) = h.as_mut() {
                *h.coeff_ref(i, i) += 1.0;
            }
        }
        // Head-tail couplings (Schur complement is non-trivial).
        for i in 0..8 {
            for j in 8..n {
                if (i + j) % 3 == 0
                    && let Some(h) = h.as_mut()
                {
                    *h.coeff_ref(i, j) += 0.05;
                    *h.coeff_ref(j, i) += 0.05;
                }
            }
        }
        Cost::with_num_residuals(cost, n as i32)
    };
    let summary = tinyopt::optimize_sparse(&mut x, acc, &options());
    assert!(summary.succeeded(), "{:?}", summary.stop_reason);
    assert_relative_eq!(x.values[2], 0.5, epsilon = 1e-5);
    assert!(x.values[0] <= 10.0 && x.values[2] <= 0.5 + 1e-12);
}
