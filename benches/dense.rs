// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! Dense Levenberg-Marquardt benchmarks, mirroring
//! `benchmarks/tinyopt/dense.cpp` and `densef.cpp` plus the 1D/2D/6D
//! profiling workloads.
//!
//! Run with `cargo bench` (or `pixi run cargo bench`).

use criterion::{Criterion, criterion_group, criterion_main};

use tinyopt_rs::Real;
#[cfg(feature = "nalgebra")]
use tinyopt_rs::Static;
use tinyopt_rs::backend::{
    DenseMatrix, DenseVector, Dim, Dynamic, Matrix, Vector, zeros_matrix, zeros_vector,
};
use tinyopt_rs::optimizers::lm;
use tinyopt_rs::{Cost, Options};

/// Benchmark options mirroring `benchmarks/common/options.h`.
fn options() -> Options {
    let mut options = Options::default();
    options.log.enable = false;
    options.exp.save_history = false;
    options.exp.measure_time = false;
    options.stop.max_iters = 100;
    options.stop.min_error = 1e-14;
    options.stop.min_rerr_dec = 1e-6;
    options.stop.min_step_norm2 = 1e-16;
    options.stop.min_step_rel_norm2 = 0.0;
    options
}

fn accumulate<S: Real, D: Dim>(
    r: &[S],
    jac: &[S],
    n: usize,
    half_cost: bool,
    mut grad: Option<&mut Vector<S, D>>,
    mut h: Option<&mut Matrix<S, D>>,
) -> Cost {
    let mut cost = S::zero();
    for v in r.iter() {
        cost += *v * *v;
    }
    if half_cost {
        cost *= S::from_f64(0.5).expect("0.5 fits");
    }

    if let Some(g) = grad.as_mut() {
        DenseVector::set_zero(*g);
        for i in 0..n {
            let mut acc = S::zero();
            for k in 0..n {
                acc += jac[k * n + i] * r[k];
            }
            g.set_at(i, acc);
        }
    }
    if let Some(hm) = h.as_mut() {
        DenseMatrix::set_zero(*hm);
        for k in 0..n {
            DenseMatrix::add_outer_product_self(*hm, S::one(), &jac[k * n..(k + 1) * n]);
        }
    }
    Cost::with_num_residuals(cost.to_f64().expect("scalar fits"), n as i32)
}

/// 1D dense-math problem: r0 = x0² - 2.
fn math_1d<S: Real, D: Dim>(
    x: &[S],
    grad: Option<&mut Vector<S, D>>,
    h: Option<&mut Matrix<S, D>>,
) -> Cost {
    let two = S::from_f64(2.0).expect("2 fits");
    let r = [x[0] * x[0] - two];
    let jac = [two * x[0]];
    accumulate::<S, D>(&r, &jac, 1, false, grad, h)
}

/// 2D dense-math problem.
fn math_2d<S: Real, D: Dim>(
    x: &[S],
    grad: Option<&mut Vector<S, D>>,
    h: Option<&mut Matrix<S, D>>,
) -> Cost {
    let two = S::from_f64(2.0).expect("2 fits");
    let (x0, x1) = (x[0], x[1]);
    let r = [x0 * x0 + x1 - two, x0 + x1 * x1 - two];
    let jac = [two * x0, S::one(), S::one(), two * x1];
    accumulate::<S, D>(&r, &jac, 2, false, grad, h)
}

/// 3D dense-math problem.
fn math_3d<S: Real, D: Dim>(
    x: &[S],
    grad: Option<&mut Vector<S, D>>,
    h: Option<&mut Matrix<S, D>>,
) -> Cost {
    let two = S::from_f64(2.0).expect("2 fits");
    let three = S::from_f64(3.0).expect("3 fits");
    let (x0, x1, x2) = (x[0], x[1], x[2]);
    let r = [
        x0 * x0 + x1 + x2 - three,
        x0 + x1 * x1 + x2 - three,
        x0 + x1 + x2 * x2 - three,
    ];
    let jac = [
        two * x0,
        S::one(),
        S::one(),
        S::one(),
        two * x1,
        S::one(),
        S::one(),
        S::one(),
        two * x2,
    ];
    accumulate::<S, D>(&r, &jac, 3, false, grad, h)
}

/// Prior problem: r = x - target with cost 0.5 rᵀr and J = I.
fn prior<S: Real, D: Dim>(
    target: &[S],
    x: &[S],
    mut grad: Option<&mut Vector<S, D>>,
    mut h: Option<&mut Matrix<S, D>>,
) -> Cost {
    let n = x.len();
    let mut cost = S::zero();
    if let Some(g) = grad.as_mut() {
        DenseVector::set_zero(*g);
    }
    if let Some(hm) = h.as_mut() {
        DenseMatrix::set_zero(*hm);
    }
    for i in 0..n {
        let r = x[i] - target[i];
        cost += r * r;
        if let Some(g) = grad.as_mut() {
            g.set_at(i, r);
        }
        if let Some(hm) = h.as_mut() {
            hm.set_elem(i, i, S::one());
        }
    }
    Cost::with_num_residuals(
        (cost * S::from_f64(0.5).expect("0.5 fits"))
            .to_f64()
            .expect("scalar fits"),
        n as i32,
    )
}

/// 6D cyclic NLLS: r_i = x_i² + x_{i+1} - 2.
fn cyclic_6d<S: Real, D: Dim>(
    x: &[S],
    grad: Option<&mut Vector<S, D>>,
    h: Option<&mut Matrix<S, D>>,
) -> Cost {
    let two = S::from_f64(2.0).expect("2 fits");
    let mut r = [S::zero(); 6];
    let mut jac = [S::zero(); 36];
    for i in 0..6 {
        let next = (i + 1) % 6;
        r[i] = x[i] * x[i] + x[next] - two;
        jac[i * 6 + i] = two * x[i];
        jac[i * 6 + next] = S::one();
    }
    accumulate::<S, D>(&r, &jac, 6, false, grad, h)
}

fn math_initial<S: Real, D: Dim>(n: usize) -> Vector<S, D> {
    let mut x = zeros_vector::<S, D>(n);
    let value = if n == 1 {
        S::one()
    } else {
        S::from_f64(1.5).expect("1.5 fits")
    };
    for i in 0..n {
        x.set_at(i, value);
    }
    x
}

fn run_benchmark<S, D, F>(c: &mut Criterion, label: &str, n: usize, acc: F)
where
    S: Real + 'static,
    D: Dim,
    Vector<S, D>: tinyopt_rs::Params<Scalar = S, Dim = D>,
    F: Fn(&[S], Option<&mut Vector<S, D>>, Option<&mut Matrix<S, D>>) -> Cost + Copy,
{
    let mut optimizer = lm::Optimizer::<Vector<S, D>>::new(options());
    c.bench_function(label, |b| {
        b.iter(|| {
            let mut x = math_initial::<S, D>(n);
            let summary = optimizer.optimize(&mut x, acc);
            std::hint::black_box(summary.final_cost.cost)
        })
    });
}

fn dense_f64(c: &mut Criterion) {
    #[cfg(feature = "nalgebra")]
    {
        run_benchmark::<f64, Static<1>, _>(c, "dense_f64/1D static", 1, |x, g, h| {
            math_1d::<f64, Static<1>>(x, g, h)
        });
        run_benchmark::<f64, Static<2>, _>(c, "dense_f64/2D static", 2, |x, g, h| {
            math_2d::<f64, Static<2>>(x, g, h)
        });
        run_benchmark::<f64, Static<3>, _>(c, "dense_f64/3D static", 3, |x, g, h| {
            math_3d::<f64, Static<3>>(x, g, h)
        });
    }

    run_benchmark::<f64, Dynamic, _>(c, "dense_f64/1D dynamic", 1, |x, g, h| {
        math_1d::<f64, Dynamic>(x, g, h)
    });
    run_benchmark::<f64, Dynamic, _>(c, "dense_f64/2D dynamic", 2, |x, g, h| {
        math_2d::<f64, Dynamic>(x, g, h)
    });
    run_benchmark::<f64, Dynamic, _>(c, "dense_f64/3D dynamic", 3, |x, g, h| {
        math_3d::<f64, Dynamic>(x, g, h)
    });
}

fn dense_f32(c: &mut Criterion) {
    #[cfg(not(feature = "nalgebra"))]
    let _ = c;
    #[cfg(feature = "nalgebra")]
    {
        run_benchmark::<f32, Static<1>, _>(c, "dense_f32/1D static", 1, |x, g, h| {
            math_1d::<f32, Static<1>>(x, g, h)
        });
        run_benchmark::<f32, Static<2>, _>(c, "dense_f32/2D static", 2, |x, g, h| {
            math_2d::<f32, Static<2>>(x, g, h)
        });
        run_benchmark::<f32, Static<3>, _>(c, "dense_f32/3D static", 3, |x, g, h| {
            math_3d::<f32, Static<3>>(x, g, h)
        });
    }
}

fn prior_problems(c: &mut Criterion) {
    for n in [6usize, 12, 33, 50] {
        let target: Vec<f64> = (0..n).map(|i| 0.25 * (i + 1) as f64).collect();
        let initial: Vec<f64> = (0..n).map(|i| 2.0 + 0.1 * i as f64).collect();
        let label = format!("prior_f64/{n}D dynamic");
        let mut optimizer = lm::Optimizer::<Vector<f64, Dynamic>>::new(options());
        c.bench_function(&label, |b| {
            b.iter(|| {
                let mut x = zeros_vector::<f64, Dynamic>(n);
                for (i, v) in initial.iter().enumerate() {
                    x.set_at(i, *v);
                }
                let acc = |x: &[f64],
                           g: Option<&mut Vector<f64, Dynamic>>,
                           h: Option<&mut Matrix<f64, Dynamic>>| {
                    prior::<f64, Dynamic>(&target, x, g, h)
                };
                let summary = optimizer.optimize(&mut x, acc);
                std::hint::black_box(summary.final_cost.cost)
            })
        });
    }
}

fn profiling(c: &mut Criterion) {
    run_benchmark::<f64, Dynamic, _>(c, "profiling/1D sqrt2", 1, |x, g, h| {
        math_1d::<f64, Dynamic>(x, g, h)
    });
    run_benchmark::<f64, Dynamic, _>(c, "profiling/2D dense math", 2, |x, g, h| {
        math_2d::<f64, Dynamic>(x, g, h)
    });
    run_benchmark::<f64, Dynamic, _>(c, "profiling/6D cyclic", 6, cyclic_6d::<f64, Dynamic>);
    let _ = zeros_matrix::<f64, Dynamic>(0, 0);
}

criterion_group!(benches, dense_f64, dense_f32, prior_problems, profiling);
criterion_main!(benches);
