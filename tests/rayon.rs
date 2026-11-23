#![cfg(feature = "rayon")]

use tinyopt::backend::{DenseMatrix, DenseVector, Matrix, Static, Vector};
use tinyopt::rayon::prelude::*;
use tinyopt::{Cost, Options, Params, optimize};

#[derive(Clone, Debug)]
struct Scalar([f64; 1]);

impl Params for Scalar {
    type Scalar = f64;
    type Dim = Static<1>;

    fn dims(&self) -> usize {
        1
    }

    fn as_slice(&self) -> &[f64] {
        &self.0
    }

    fn plus_eq(&mut self, delta: &[f64]) {
        self.0[0] += delta[0];
    }

    fn norm2(&self) -> f64 {
        self.0[0] * self.0[0]
    }
}

#[test]
fn rayon_accumulation_matches_finite_differences_and_single_thread() {
    let targets: Vec<_> = (0..256).map(|i| (i as f64 - 100.0) / 8.0).collect();
    let acc = |x: &[f64],
               grad: Option<&mut Vector<f64, Static<1>>>,
               h: Option<&mut Matrix<f64, Static<1>>>| {
        let (cost, derivative, curvature) = targets
            .par_iter()
            .map(|target| {
                let residual = x[0] - target;
                (0.5 * residual * residual, residual, 1.0)
            })
            .reduce(
                || (0.0, 0.0, 0.0),
                |left, right| (left.0 + right.0, left.1 + right.1, left.2 + right.2),
            );
        if let Some(grad) = grad {
            grad.set_at(0, derivative);
        }
        if let Some(h) = h {
            h.set_elem(0, 0, curvature);
        }
        Cost::with_num_residuals(cost, targets.len() as i32)
    };

    let run = |threads| {
        let pool = tinyopt::rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("Rayon pool");
        pool.install(|| {
            let x = [2.0];
            let mut grad = <Vector<f64, Static<1>> as DenseVector<f64>>::zeros(1);
            let mut h = <Matrix<f64, Static<1>> as DenseMatrix<f64>>::zeros(1, 1);
            let cost = acc(&x, Some(&mut grad), Some(&mut h));
            let epsilon = 1e-3;
            let plus = acc(&[x[0] + epsilon], None, None).cost;
            let minus = acc(&[x[0] - epsilon], None, None).cost;
            let gradient_fd = (plus - minus) / (2.0 * epsilon);
            let curvature_fd = (plus - 2.0 * cost.cost + minus) / (epsilon * epsilon);
            assert!((grad.at(0) - gradient_fd).abs() < 1e-6);
            assert!((h.elem(0, 0) - curvature_fd).abs() < 1e-4);

            let mut params = Scalar([20.0]);
            let summary = optimize(&mut params, acc, &Options::default());
            assert!(summary.converged(), "{:?}", summary.stop_reason);
            params.0[0]
        })
    };

    let single_thread = run(1);
    let parallel = run(4);
    let expected = targets.iter().sum::<f64>() / targets.len() as f64;
    assert!(
        (single_thread - expected).abs() < 1e-6,
        "single-thread result {single_thread} differs from {expected}"
    );
    assert!(
        (parallel - single_thread).abs() < 1e-8,
        "parallel result {parallel} differs from {single_thread}"
    );
}
