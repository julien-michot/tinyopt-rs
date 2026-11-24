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
use tinyopt::optimizers::{lm, trcg};
use tinyopt::sparse::SparseMat;
use tinyopt::{Cost, Options, Ordering, Params, Solver, optimize_tridiag};

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

/// CHOLMOD vs the built-in simplicial LDLᵀ on a 2-D 5-point Laplacian grid.
///
/// `cold` includes the symbolic analysis; `warm` reuses it and only
/// refactorizes the values, which is what the optimizer does per iteration.
#[cfg(feature = "suitesparse")]
fn sparse_solver_cholmod(c: &mut Criterion) {
    use tinyopt::backend::LinearSolverMethod;
    use tinyopt::options::LinearSolverOptions;
    use tinyopt::sparse::{SparseLinearSolvers, SparseMatrix};

    let mut group = c.benchmark_group("sparse_solve");
    group.sample_size(30);
    for (label, n) in [("grid20", 20usize), ("grid40", 40), ("grid80", 80)] {
        let dims = n * n;
        let mut triplets = Vec::new();
        for i in 0..n {
            for j in 0..n {
                let k = i * n + j;
                triplets.push((k, k, 4.0));
                if i + 1 < n {
                    triplets.push((k, k + n, -1.0));
                    triplets.push((k + n, k, -1.0));
                }
                if j + 1 < n {
                    triplets.push((k, k + 1, -1.0));
                    triplets.push((k + 1, k, -1.0));
                }
            }
        }
        let a = SparseMatrix::<f64>::from_triplets(dims, dims, &triplets);
        let b: Vec<f64> = (0..dims).map(|i| ((i as f64 * 7.0) % 13.0) - 6.0).collect();
        for (name, method) in [
            ("ldlt", LinearSolverMethod::Ldlt),
            ("cholmod", LinearSolverMethod::SuiteSparse),
        ] {
            let options = LinearSolverOptions {
                method,
                ..Default::default()
            };
            let mut cold = options.clone();
            group.bench_function(format!("{label}/{name}/cold"), |bench| {
                bench.iter(|| {
                    let mut solver = SparseLinearSolvers::<f64>::new(cold.clone());
                    assert!(solver.analyze(&a, 0));
                    assert!(solver.compute(&a));
                    std::hint::black_box(solver.solve(&a, &b).unwrap())
                })
            });

            let mut solver = SparseLinearSolvers::<f64>::new(options);
            assert!(solver.analyze(&a, 0));
            group.bench_function(format!("{label}/{name}/warm"), |bench| {
                bench.iter(|| {
                    assert!(solver.compute(&a));
                    std::hint::black_box(solver.solve(&a, &b).unwrap())
                })
            });
            cold.method = method;
        }
    }
    group.finish();
}

/// End-to-end sparse optimization with the built-in LDLᵀ vs CHOLMOD.
#[cfg(feature = "suitesparse")]
fn sparse_optimizer_cholmod(c: &mut Criterion) {
    use tinyopt::backend::LinearSolverMethod;

    fn grid_acc(n: usize) -> impl Fn(&[f64], Option<&mut Vx>, Option<&mut SparseMat>) -> Cost {
        move |x: &[f64], grad: Option<&mut Vx>, h: Option<&mut SparseMat>| -> Cost {
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
            let mut edges = 0usize;
            for i in 0..n {
                for j in 0..n {
                    let k = i * n + j;
                    if j + 1 < n {
                        let r = x[k + 1] - x[k];
                        cost += r * r;
                        edges += 1;
                        if let Some(g) = g.as_mut() {
                            g.set_at(k, g.at(k) - r);
                            g.set_at(k + 1, g.at(k + 1) + r);
                        }
                        if let Some(h) = h.as_mut() {
                            *h.coeff_ref(k, k) += 1.0;
                            *h.coeff_ref(k + 1, k + 1) += 1.0;
                            *h.coeff_ref(k, k + 1) -= 1.0;
                            *h.coeff_ref(k + 1, k) -= 1.0;
                        }
                    }
                    if i + 1 < n {
                        let r = x[k + n] - x[k];
                        cost += r * r;
                        edges += 1;
                        if let Some(g) = g.as_mut() {
                            g.set_at(k, g.at(k) - r);
                            g.set_at(k + n, g.at(k + n) + r);
                        }
                        if let Some(h) = h.as_mut() {
                            *h.coeff_ref(k, k) += 1.0;
                            *h.coeff_ref(k + n, k + n) += 1.0;
                            *h.coeff_ref(k, k + n) -= 1.0;
                            *h.coeff_ref(k + n, k) -= 1.0;
                        }
                    }
                }
            }
            Cost::with_num_residuals(cost, (x.len() + edges) as i32)
        }
    }

    let mut group = c.benchmark_group("sparse_optimizer");
    group.sample_size(10);
    for (label, dims) in [("chain1000", 1000usize)] {
        let acc = sparse_chain_acc(dims);
        for (name, method) in [
            ("ldlt", LinearSolverMethod::Ldlt),
            ("cholmod", LinearSolverMethod::SuiteSparse),
        ] {
            let mut opts = options();
            opts.linear.method = method;
            let mut opt = lm::Optimizer::<Vec<f64>, SparseMat>::new(opts);
            group.bench_function(format!("{label}/{name}"), |b| {
                b.iter(|| {
                    let mut x = vec![0.0; dims];
                    let summary = opt.optimize(&mut x, &acc);
                    std::hint::black_box(summary.final_cost.cost)
                })
            });
        }
    }
    for (label, n) in [("grid20", 20usize), ("grid40", 40)] {
        let dims = n * n;
        let acc = grid_acc(n);
        for (name, method) in [
            ("ldlt", LinearSolverMethod::Ldlt),
            ("cholmod", LinearSolverMethod::SuiteSparse),
        ] {
            let mut opts = options();
            opts.linear.method = method;
            let mut opt = lm::Optimizer::<Vec<f64>, SparseMat>::new(opts);
            group.bench_function(format!("{label}/{name}"), |b| {
                b.iter(|| {
                    let mut x = vec![0.0; dims];
                    let summary = opt.optimize(&mut x, &acc);
                    std::hint::black_box(summary.final_cost.cost)
                })
            });
        }
    }
    group.finish();
}

/// Same grids but with a 60-parameter Schur head block (bundle-adjustment
/// shape): CHOLMOD factorizes the eliminated block, as in the C++ dispatch.
#[cfg(feature = "suitesparse")]
fn sparse_solve_schur(c: &mut Criterion) {
    use tinyopt::backend::LinearSolverMethod;
    use tinyopt::options::LinearSolverOptions;
    use tinyopt::sparse::{SparseLinearSolvers, SparseMatrix};

    let mut group = c.benchmark_group("sparse_schur");
    group.sample_size(30);
    for (label, n) in [("grid20", 20usize), ("grid40", 40)] {
        let dims = n * n;
        let mut triplets = Vec::new();
        for i in 0..n {
            for j in 0..n {
                let k = i * n + j;
                triplets.push((k, k, 4.0));
                if i + 1 < n {
                    triplets.push((k, k + n, -1.0));
                    triplets.push((k + n, k, -1.0));
                }
                if j + 1 < n {
                    triplets.push((k, k + 1, -1.0));
                    triplets.push((k + 1, k, -1.0));
                }
            }
        }
        let a = SparseMatrix::<f64>::from_triplets(dims, dims, &triplets);
        let b: Vec<f64> = (0..dims).map(|i| ((i as f64 * 7.0) % 13.0) - 6.0).collect();
        for (name, method) in [
            ("ldlt", LinearSolverMethod::Ldlt),
            ("cholmod", LinearSolverMethod::SuiteSparse),
        ] {
            let options = LinearSolverOptions {
                method,
                ..Default::default()
            };
            let mut solver = SparseLinearSolvers::<f64>::new(options);
            assert!(solver.analyze(&a, 60));
            group.bench_function(format!("{label}/{name}/warm"), |bench| {
                bench.iter(|| {
                    assert!(solver.compute(&a));
                    std::hint::black_box(solver.solve(&a, &b).unwrap())
                })
            });
        }
    }
    group.finish();
}

/// 2-D 5-point Laplacian grid (SPD), used by the linear-solver benchmarks.
fn grid_spd(n: usize) -> tinyopt::sparse::SparseMatrix<f64> {
    let mut triplets = Vec::new();
    for i in 0..n {
        for j in 0..n {
            let k = i * n + j;
            triplets.push((k, k, 4.0));
            if i + 1 < n {
                triplets.push((k, k + n, -1.0));
                triplets.push((k + n, k, -1.0));
            }
            if j + 1 < n {
                triplets.push((k, k + 1, -1.0));
                triplets.push((k + 1, k, -1.0));
            }
        }
    }
    tinyopt::sparse::SparseMatrix::from_triplets(n * n, n * n, &triplets)
}

/// Steihaug–Toint trust-region Newton-CG against Levenberg-Marquardt.
fn trust_region_cg(c: &mut Criterion) {
    use tinyopt::sparse::SparseMat;
    for dims in [100usize, 1000] {
        let acc = sparse_chain_acc(dims);
        let mut lm_options = options();
        lm_options.solver_type = Solver::LevenbergMarquardt;
        let mut lm_optimizer = lm::Optimizer::<Vec<f64>, SparseMat>::new(lm_options);
        c.bench_function(&format!("solver_chain/{dims}D/lm"), |b| {
            b.iter(|| {
                let mut x = vec![0.0; dims];
                let summary = lm_optimizer.optimize(&mut x, &acc);
                std::hint::black_box(summary.final_cost.cost)
            })
        });

        let mut trcg_options = options();
        trcg_options.solver_type = Solver::SteihaugToint;
        let mut trcg_optimizer = trcg::Optimizer::<Vec<f64>, SparseMat>::new(trcg_options);
        c.bench_function(&format!("solver_chain/{dims}D/trcg"), |b| {
            b.iter(|| {
                let mut x = vec![0.0; dims];
                let summary = trcg_optimizer.optimize(&mut x, &acc);
                std::hint::black_box(summary.final_cost.cost)
            })
        });
    }
}

/// IC(0) preconditioned CG against the direct sparse LDLᵀ.
fn sparse_solve_ic(c: &mut Criterion) {
    use tinyopt::backend::LinearSolverMethod;
    use tinyopt::options::LinearSolverOptions;
    use tinyopt::sparse::SparseLinearSolvers;

    let mut group = c.benchmark_group("sparse_solve_ic");
    group.sample_size(20);
    for (label, n) in [("grid20", 20usize), ("grid40", 40)] {
        let a = grid_spd(n);
        let dims = a.nrows();
        let b: Vec<f64> = (0..dims).map(|i| ((i as f64 * 7.0) % 13.0) - 6.0).collect();
        for (name, method) in [
            ("ldlt", LinearSolverMethod::Ldlt),
            ("icpcg", LinearSolverMethod::IncompleteCholesky),
        ] {
            let options = LinearSolverOptions {
                method,
                ..Default::default()
            };
            let mut solver = SparseLinearSolvers::<f64>::new(options);
            assert!(solver.analyze(&a, 0));
            group.bench_function(format!("{label}/{name}"), |bench| {
                bench.iter(|| {
                    assert!(solver.compute(&a));
                    std::hint::black_box(solver.solve(&a, &b).unwrap())
                })
            });
        }
    }
    group.finish();
}

#[cfg(feature = "suitesparse")]
criterion_group!(
    benches,
    sparse_chains,
    sparse_chain_orderings,
    tridiag_chain,
    block_sparse_ba,
    sparse_ordering,
    sparse_solver_cholmod,
    sparse_optimizer_cholmod,
    sparse_solve_schur,
    trust_region_cg,
    sparse_solve_ic
);

#[cfg(not(feature = "suitesparse"))]
criterion_group!(
    benches,
    sparse_chains,
    sparse_chain_orderings,
    tridiag_chain,
    block_sparse_ba,
    sparse_ordering,
    trust_region_cg,
    sparse_solve_ic
);
criterion_main!(benches);
