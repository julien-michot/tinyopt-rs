// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0
#![cfg(feature = "autodiff")]

//! Robust residual blocks for the autodiff path, mirroring the C++
//! `Residuals(losses::Huber(res, delta), ...)` tuple API.

use approx::assert_relative_eq;
use tinyopt::backend::DenseVector;
use tinyopt::diff::{DualNum, ResidualBlock, RobustResidualsFn, robust_accumulation};
use tinyopt::{
    AccumulationFn, Dynamic, Options, Static, optimize_residuals, optimize_robust_residuals,
};

/// Mean estimation where each observation is its own residual block.
struct RobustMean {
    points: Vec<f64>,
    delta: f64,
}

impl RobustResidualsFn for RobustMean {
    type Scalar = f64;

    fn eval<T: DualNum<Primitive = f64>>(&self, x: &[T]) -> Vec<ResidualBlock<T>> {
        self.points
            .iter()
            .map(|p| ResidualBlock::huber(vec![x[0].clone() - *p], self.delta))
            .collect()
    }
}

/// Plain L2 version of the same problem.
struct PlainMean {
    points: Vec<f64>,
}

impl tinyopt::diff::ResidualsFn for PlainMean {
    type Scalar = f64;

    fn eval<T: DualNum<Primitive = f64>>(&self, x: &[T]) -> Vec<T> {
        self.points.iter().map(|p| x[0].clone() - *p).collect()
    }
}

#[test]
fn huber_downweights_outliers() {
    // Five inliers at 0 and one outlier at 100.
    let points = vec![0.0, 0.0, 0.0, 0.0, 0.0, 100.0];
    let f = RobustMean { points, delta: 1.0 };

    let mut x = 0.0_f64;
    let summary = optimize_robust_residuals(&mut x, &f, &Options::default());
    assert!(summary.succeeded(), "{:?}", summary.stop_reason);

    // Equilibrium: 5 inlier gradients (5x) balance the saturated Huber
    // influence of the outlier (delta = 1) -> x = 0.2.
    assert_relative_eq!(x, 0.2, epsilon = 1e-6);
    assert_eq!(summary.num_residuals, 6);
    // Five of six residuals are inliers.
    assert_relative_eq!(summary.final_cost.inlier_ratio, 5.0 / 6.0, epsilon = 1e-6);
}

#[test]
fn plain_l2_is_pulled_by_the_outlier() {
    let points = vec![0.0, 0.0, 0.0, 0.0, 0.0, 100.0];
    let f = PlainMean { points };

    let mut x = 0.0_f64;
    let summary = optimize_residuals(&mut x, &f, &Options::default());
    assert!(summary.succeeded(), "{:?}", summary.stop_reason);
    assert_relative_eq!(x, 100.0 / 6.0, epsilon = 1e-6);
    assert_relative_eq!(summary.final_cost.inlier_ratio, 1.0);
}

#[test]
fn robust_blocks_match_plain_when_all_inliers() {
    let points = vec![0.2, -0.1, 0.05];
    let robust = RobustMean {
        points: points.clone(),
        delta: 10.0,
    };
    let plain = PlainMean { points };

    let mut xr = 0.0_f64;
    let mut xp = 0.0_f64;
    let _ = optimize_robust_residuals(&mut xr, &robust, &Options::default());
    let _ = optimize_residuals(&mut xp, &plain, &Options::default());
    assert_relative_eq!(xr, xp, epsilon = 1e-6);
}

#[test]
fn robust_accumulator_matches_manual_weighting() {
    // One block with a large residual: grad = (delta / ||r||) * Jᵀ r.
    struct One;
    impl RobustResidualsFn for One {
        type Scalar = f64;
        fn eval<T: DualNum<Primitive = f64>>(&self, x: &[T]) -> Vec<ResidualBlock<T>> {
            vec![ResidualBlock::huber(vec![x[0].clone(), x[1].clone()], 1.0)]
        }
    }

    type V2 = tinyopt::backend::Vector<f64, Static<2>>;
    let acc = robust_accumulation(&One);
    let x = [3.0_f64, 4.0];
    let mut g = <V2 as tinyopt::backend::DenseVector<f64>>::zeros(2);
    let cost = AccumulationFn::<f64, Static<2>>::eval(&acc, &x, Some(&mut g), None);

    // ||x|| = 5, weight = 1/5, J = I -> grad = x / 5.
    assert_relative_eq!(g.at(0), 3.0 / 5.0, epsilon = 1e-12);
    assert_relative_eq!(g.at(1), 4.0 / 5.0, epsilon = 1e-12);
    assert_relative_eq!(cost.cost, 2.0 * 5.0 - 1.0, epsilon = 1e-12);
    assert_eq!(cost.num_inliers(), 0);
}

#[test]
fn robust_accumulator_dynamic() {
    type Vx = tinyopt::backend::Vector<f64, Dynamic>;
    let points = vec![0.0, 0.0, 0.0, 50.0];
    let f = RobustMean { points, delta: 1.0 };
    let acc = robust_accumulation(&f);
    let x = [0.5];
    let mut g = <Vx as tinyopt::backend::DenseVector<f64>>::zeros(1);
    let cost = AccumulationFn::<f64, Dynamic>::eval(&acc, &x, Some(&mut g), None);
    assert!(cost.is_valid());
    assert!(g.at(0).is_finite());
}
