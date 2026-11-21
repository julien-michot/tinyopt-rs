// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! Sparse and block-structured benchmarks, mirroring
//! `benchmarks/tinyopt/{sparse,bundle_adjustment}.cpp`.
//!
//! Run with `cargo bench --bench sparse`.

use criterion::{Criterion, criterion_group, criterion_main};

use tinyopt::backend::{DenseVector, Dynamic, Vector};
use tinyopt::block_sparse::BlockSparseMat;
use tinyopt::block_sparse_types;
use tinyopt::block_tridiag::BlockTridiagMat;
use tinyopt::optimizers::lm;
use tinyopt::sparse::SparseMat;
use tinyopt::{Cost, Options, Ordering, Params, optimize_tridiag};

type Vx = Vector<f64, Dynamic>;

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

fn sparse_chain_acc(
    _dims: usize,
) -> impl Fn(&[f64], Option<&mut Vx>, Option<&mut SparseMat>) -> Cost {
    |x: &[f64], grad: Option<&mut Vx>, h: Option<&mut SparseMat>| -> Cost {
        let n = x.len();
        let mut cost = 0.0;
        let mut g = grad;
        let mut h = h;
        for (i, xi) in x.iter().enumerate() {
            let r = xi - 1.0;
            cost += r * r;
            if let Some(g) = g.as_mut() {
                g.set_at(i, g.at(i) + r);
            }
            if let Some(h) = h.as_mut() {
                *h.coeff_ref(i, i) += 1.0;
            }
        }
        for i in 0..n.saturating_sub(1) {
            let r = x[i + 1] - x[i];
            cost += r * r;
            if let Some(g) = g.as_mut() {
                g.set_at(i, g.at(i) - r);
                g.set_at(i + 1, g.at(i + 1) + r);
            }
            if let Some(h) = h.as_mut() {
                *h.coeff_ref(i, i) += 1.0;
                *h.coeff_ref(i + 1, i + 1) += 1.0;
                *h.coeff_ref(i, i + 1) -= 1.0;
                *h.coeff_ref(i + 1, i) -= 1.0;
            }
        }
        Cost::with_num_residuals(cost, (2 * n) as i32)
    }
}

fn sparse_chains(c: &mut Criterion) {
    for dims in [10usize, 100, 1000] {
        let acc = sparse_chain_acc(dims);
        let mut opt = lm::Optimizer::<Vec<f64>, SparseMat>::new(options());
        c.bench_function(&format!("sparse_chain/{dims}D"), |b| {
            b.iter(|| {
                let mut x = vec![0.0; dims];
                let summary = opt.optimize(&mut x, &acc);
                std::hint::black_box(summary.final_cost.cost)
            })
        });
    }
}

/// Ordering overhead on a banded pattern (a chain needs no reordering).
fn sparse_chain_orderings(c: &mut Criterion) {
    for dims in [10usize, 100, 1000] {
        let acc = sparse_chain_acc(dims);
        for (name, ordering) in [("natural", Ordering::Natural), ("amd", Ordering::Amd)] {
            let mut opts = options();
            opts.linear.ordering = ordering;
            let mut opt = lm::Optimizer::<Vec<f64>, SparseMat>::new(opts);
            c.bench_function(&format!("sparse_chain_ord/{dims}D/{name}"), |b| {
                b.iter(|| {
                    let mut x = vec![0.0; dims];
                    let summary = opt.optimize(&mut x, &acc);
                    std::hint::black_box(summary.final_cost.cost)
                })
            });
        }
    }
}

fn tridiag_chain(c: &mut Criterion) {
    const N: usize = 50;
    const B: usize = 6;
    let acc = |x: &[f64], grad: Option<&mut Vx>, h: Option<&mut BlockTridiagMat<B, f64>>| -> Cost {
        let mut cost = 0.0;
        let mut g = grad;
        let mut h = h;
        for i in 0..N {
            for k in 0..B {
                let idx = i * B + k;
                let r = x[idx] - 1.0;
                cost += r * r;
                if let Some(g) = g.as_mut() {
                    g.set_at(idx, g.at(idx) + r);
                }
                if let Some(h) = h.as_mut() {
                    let mut e = [0.0; B];
                    e[k] = 1.0;
                    h.diagonal(i).add_outer_product_self(1.0, &e);
                }
            }
        }
        for i in 0..N - 1 {
            for k in 0..B {
                let r = x[(i + 1) * B + k] - x[i * B + k];
                cost += r * r;
                if let Some(g) = g.as_mut() {
                    g.set_at(i * B + k, g.at(i * B + k) - r);
                    g.set_at((i + 1) * B + k, g.at((i + 1) * B + k) + r);
                }
                if let Some(h) = h.as_mut() {
                    let mut e = [0.0; B];
                    e[k] = 1.0;
                    h.diagonal(i).add_outer_product_self(1.0, &e);
                    h.diagonal(i + 1).add_outer_product_self(1.0, &e);
                    h.off_diagonal(i).add_outer_product(-1.0, &e, &e);
                }
            }
        }
        Cost::with_num_residuals(cost, (2 * N * B) as i32)
    };

    c.bench_function("block_tridiag/50x6", |b| {
        b.iter(|| {
            let mut x = vec![0.0; N * B];
            let summary = optimize_tridiag::<B, _, _>(&mut x, acc, &options());
            std::hint::black_box(summary.final_cost.cost)
        })
    });
}

// --- Small bundle adjustment ------------------------------------------------

block_sparse_types!(BaTypes, [6, 3]);

const NCAM: usize = 2;
const NPOINT: usize = 30;
const CAM: usize = 6;
const POINT: usize = 3;
const DIMS: usize = NCAM * CAM + NPOINT * POINT;

#[derive(Clone, Debug)]
struct BaParams {
    values: Vec<f64>,
}

impl Params for BaParams {
    type Scalar = f64;
    type Dim = Dynamic;

    fn dims(&self) -> usize {
        DIMS
    }

    fn as_slice(&self) -> &[f64] {
        &self.values
    }

    fn plus_eq(&mut self, delta: &[f64]) {
        for (v, d) in self.values.iter_mut().zip(delta) {
            *v += *d;
        }
    }

    fn norm2(&self) -> f64 {
        self.values.iter().map(|v| v * v).sum()
    }

    fn reduced(&self) -> Option<(usize, usize)> {
        Some((0, NCAM * CAM))
    }
}

fn ba_acc(x: &[f64], grad: Option<&mut Vx>, h: Option<&mut BlockSparseMat<BaTypes, f64>>) -> Cost {
    let mut cost = 0.0;
    let mut g = grad;
    let mut h = h;

    // Priors on all parameters.
    for (i, xi) in x.iter().enumerate() {
        let r = xi - 1.0;
        cost += r * r;
        if let Some(g) = g.as_mut() {
            g.set_at(i, g.at(i) + r);
        }
    }
    if let Some(h) = h.as_mut() {
        let e6 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let e3 = [1.0, 0.0, 0.0];
        for i in 0..NCAM {
            h.block_mut::<6, 6>(i, i).add_outer_product_self(1.0, &e6);
        }
        for j in 0..NPOINT {
            h.block_mut::<3, 3>(j, j).add_outer_product_self(1.0, &e3);
        }
    }

    // Camera-point couplings on the first coordinate.
    for i in 0..NCAM {
        let cam = i * CAM;
        for j in 0..NPOINT {
            let pt = NCAM * CAM + j * POINT;
            let r = x[cam] + x[pt] - 2.0;
            cost += r * r;
            if let Some(g) = g.as_mut() {
                g.set_at(cam, g.at(cam) + r);
                g.set_at(pt, g.at(pt) + r);
            }
            if let Some(h) = h.as_mut() {
                let e6 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
                let e3 = [1.0, 0.0, 0.0];
                h.block_mut::<6, 6>(i, i).add_outer_product_self(1.0, &e6);
                h.block_mut::<3, 3>(j, j).add_outer_product_self(1.0, &e3);
                h.block_mut::<6, 3>(i, j).add_outer_product(1.0, &e6, &e3);
            }
        }
    }

    Cost::with_num_residuals(cost, (DIMS + NCAM * NPOINT) as i32)
}

fn block_sparse_ba(c: &mut Criterion) {
    let mut opt = lm::Optimizer::<BaParams, BlockSparseMat<BaTypes, f64>>::new(options());
    opt.init_with(BlockSparseMat::with_counts(&[NCAM, NPOINT]));
    c.bench_function("block_sparse_ba/2c_30p", |b| {
        b.iter(|| {
            let mut x = BaParams {
                values: vec![0.0; DIMS],
            };
            let summary = opt.optimize(&mut x, ba_acc);
            std::hint::black_box(summary.final_cost.cost)
        })
    });
}

/// Symbolic + numeric LDLᵀ on a star graph whose natural ordering is dense.
fn sparse_ordering(c: &mut Criterion) {
    use tinyopt::Ordering;
    use tinyopt::sparse::{SimplicialLdl, SparseMatrix};

    for n in [200usize, 800] {
        let mut triplets = Vec::new();
        for i in 0..n {
            triplets.push((i, i, n as f64));
            if i > 0 {
                triplets.push((i, 0, 1.0));
                triplets.push((0, i, 1.0));
            }
        }
        let lower = SparseMatrix::<f64>::from_triplets(n, n, &triplets).to_lower();
        for (name, ordering) in [("natural", Ordering::Natural), ("amd", Ordering::Amd)] {
            c.bench_function(&format!("ordering/{n}D/{name}"), |b| {
                b.iter(|| {
                    let mut ldlt = SimplicialLdl::<f64>::default();
                    assert!(ldlt.analyze_and_factorize_with(&lower, ordering));
                    std::hint::black_box(ldlt.fill_in())
                })
            });
        }
    }

    // A banded (tridiagonal) pattern: AMD has no fill to remove, so it should
    // stay close to the natural ordering instead of dominating the solve.
    let n = 1000usize;
    let mut triplets = Vec::new();
    for i in 0..n {
        triplets.push((i, i, 4.0));
        if i + 1 < n {
            triplets.push((i + 1, i, -1.0));
            triplets.push((i, i + 1, -1.0));
        }
    }
    let lower = SparseMatrix::<f64>::from_triplets(n, n, &triplets).to_lower();
    for (name, ordering) in [("natural", Ordering::Natural), ("amd", Ordering::Amd)] {
        c.bench_function(&format!("ordering_chain/{n}D/{name}"), |b| {
            b.iter(|| {
                let mut ldlt = SimplicialLdl::<f64>::default();
                assert!(ldlt.analyze_and_factorize_with(&lower, ordering));
                std::hint::black_box(ldlt.fill_in())
            })
        });
    }
}

criterion_group!(
    benches,
    sparse_chains,
    sparse_chain_orderings,
    tridiag_chain,
    block_sparse_ba,
    sparse_ordering
);
criterion_main!(benches);
