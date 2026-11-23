# tinyopt

A lightweight, Rust native optimization library for unconstrained optimization and
non-linear least squares, with **dense**, **sparse**, **block-tridiagonal** and
**block-sparse** Hessians. It is the Rust counterpart of
[C++ tinyopt](https://github.com/julienmichot/tinyopt) (version parity **0.8.0**) and
follows the same *accumulation pattern*: you accumulate the gradient `Jᵀr` and the
Hessian approximation `JᵀJ` directly into the linear system, without materializing
residual or Jacobian buffers.

- Solvers: Levenberg-Marquardt (default), Gauss-Newton, Dog-Leg, Steihaug–Toint
  trust-region Newton-CG, gradient descent, conjugate gradient, BFGS and L-BFGS,
  with optional line search.
- Differentiation: forward-mode autodiff (`num-dual`), numerical differentiation and
  analytical derivatives through the accumulation functions.
- Fixed parameters, box bounds (active-set projection), Schur complement (reduced
  parameters), covariance export, robust losses, custom parameter types and
  manifold retractions.
- Linear solvers: dense LDLT/LLT/LU/QR/SVD, sparse LDLT (AMD), CHOLMOD (optional)
  and IC(0)-preconditioned CG.
- Backends: [`nalgebra`](https://crates.io/crates/nalgebra) (static + dynamic) or
  [`faer`](https://crates.io/crates/faer) (dynamic).

## Benchmarks

The quick cross-library benchmark suite compares `tinyopt-rs` (nalgebra) with
`ceres-rs` on Linux x86_64 (10 samples). In the current run, tinyopt-rs was
3.2-10.3x faster across the six measured problems:

| Problem | tinyopt-rs | ceres-rs |
|---|---:|---:|
| 3D static | 1.42 µs | 14.70 µs |
| 33D dynamic | 17.59 µs | 63.22 µs |
| 100D sparse chain | 67.66 µs | 667.57 µs |
| 32x6 block tridiagonal | 129.30 µs | 471.10 µs |
| 50-observation Huber pose | 23.92 µs | 145.96 µs |
| 2-cam / 200-pts BA | 1.24 ms | 4.01 ms |
| 100-cam / 50k-pts BA, 1 Rayon thread | 6.361 s | — |
| 100-cam / 50k-pts BA, 8 Rayon threads | 4.436 s | — |
| 100-cam / 50k-pts BA, WGPU f32 | 8.973 s | — |
| 100-cam / 50k-pts BA, WGPU f64 | 10.446 s | — |

All runs converged. Iterations: 2-camera/200-point BA converged in 4 iterations;
the 100-camera/50k-point 1-thread and 8-thread Rayon runs converged in 38
iterations; WGPU f32 converged in 34 iterations and WGPU f64 in 38 iterations.
The Rayon BA timings are median total solve times from three samples after a
500 ms warmup; the 8-thread solve was 1.43x faster than the 1-thread solve.

Run `pixi run bench-quick` and `pixi run bench-report` to reproduce the suite and
generate the full report. Reproduce the Rayon BA timing with
`pixi run --environment bench cargo run --release --manifest-path benchmarks/compare/Cargo.toml --no-default-features --features rayon -- parallel-ba --samples 3 --warmup-ms 500`.
The C++ benchmark binary was not available in this run.
Results obtained in Nov. 2026, on my old laptop so may change on your fancy smartwatch (typical usecase for Tinyopt).

## Installation

```toml
[dependencies]
tinyopt = "0.8"
```

The default features are `nalgebra` and `autodiff`. Exactly one backend must be
enabled; `nalgebra` and `faer` are mutually exclusive.

| Feature | Default | Purpose |
|---|---|---|
| `nalgebra` | yes | static (`SVector`/`SMatrix`) and dynamic (`DVector`/`DMatrix`) sizes |
| `faer` | no | dynamic sizes (`Col`/`Mat`); use with `default-features = false` |
| `autodiff` | yes | forward-mode AD through `num-dual` (requires `nalgebra`) |
| `numdiff` | no | central finite differences |
| `rayon` | no | optional Rayon support for parallel accumulation and independent block-Schur work |
| `suitesparse` | no | optional CHOLMOD backend for scalar sparse solves (system SuiteSparse) |
| `solver-ldlt` | yes (via backend) | symmetric-indefinite LDLT factorization, the default dense solver |
| `solver-llt` | no | Cholesky LLT factorization |
| `solver-lu` | no | pivoted LU factorization |
| `solver-qr` | no | QR factorization |
| `solver-svd` | no | SVD factorization with pseudo-inverse fallback |

```toml
# faer backend without autodiff
tinyopt = { version = "0.8", default-features = false, features = ["faer"] }
```

## Quick start: manual accumulation

The fastest path: one closure that returns the cost and, when requested, writes the
gradient and Hessian. Here we minimize `(x0 - 1)² + 10 (x1 + 2)²`:

```rust
use tinyopt::backend::{DenseMatrix, DenseVector, Matrix, Vector};
use tinyopt::{Cost, Options, Static, optimize};

type Vec2 = Vector<f64, Static<2>>;
type Mat2 = Matrix<f64, Static<2>>;

let acc = |x: &[f64], grad: Option<&mut Vec2>, h: Option<&mut Mat2>| -> Cost {
    let (r0, r1) = (x[0] - 1.0, x[1] + 2.0);
    if let Some(g) = grad {
        g.set_at(0, 2.0 * r0);
        g.set_at(1, 20.0 * r1);
    }
    if let Some(h) = h {
        h.set_elem(0, 0, 2.0);
        h.set_elem(1, 1, 20.0);
    }
    Cost::with_num_residuals(r0 * r0 + 10.0 * r1 * r1, 2)
};

let mut x = <Vec2 as DenseVector<f64>>::zeros(2);
let summary = optimize(&mut x, acc, &Options::default());
assert!(summary.succeeded());
assert!((x.at(0) - 1.0).abs() < 1e-6);
assert!((x.at(1) + 2.0).abs() < 1e-6);
```

`grad` and `h` are `Option`s: the same closure also serves cost-only evaluations
(e.g. trial steps in the line search), so keep the `if let` guards.

## Quick start: autodiff residuals

For residual vectors, implement `ResidualsFn` once with a generic scalar; the
optimizer computes `Jᵀr` and `JᵀJ` with forward-mode autodiff:

```rust
use tinyopt::backend::{DenseVector, Vector};
use tinyopt::diff::{DualNum, ResidualsFn};
use tinyopt::{Options, Static, optimize_residuals};

struct Sqrt2;

impl ResidualsFn for Sqrt2 {
    type Scalar = f64;

    fn eval<T: DualNum<Primitive = f64>>(&self, x: &[T]) -> Vec<T> {
        let x0 = x[0].clone();
        vec![x0.clone() * x0 - 2.0]
    }
}

let mut x = <Vector<f64, Static<1>> as DenseVector<f64>>::zeros(1);
x.set_at(0, 1.0);
let summary = optimize_residuals(&mut x, &Sqrt2, &Options::default());
assert!(summary.converged());
assert!((x.at(0) - 2.0_f64.sqrt()).abs() < 1e-6);
```

`optimize_residuals` uses the default options; use `optimize_cost` for a scalar cost
function, or `optimize_robust_residuals` for per-residual losses.

## Choosing a solver

```rust
use tinyopt::{Options, Solver};

// Levenberg-Marquardt is the default.
let options = Options::new(Solver::GaussNewton);
```

| `Solver` | Module | Notes |
|---|---|---|
| `LevenbergMarquardt` (default) | `optimizers::lm` | damped NLLS, adaptive `λ` |
| `GaussNewton` | `optimizers::gn` | no damping |
| `DogLeg` | `optimizers::dl` | trust region |
| `SteihaugToint` | `optimizers::trcg` | trust-region Newton-CG (Hessian-vector products) |
| `GradientDescent` | `optimizers::gd` | first order, learning rate |
| `ConjugateGradient` | `optimizers::cg` | first order |
| `Bfgs` | `optimizers::bfgs` | quasi-Newton |
| `Lbfgs` | `optimizers::lbfgs` | limited-memory quasi-Newton |

Any parameter type can expose box bounds through `Params::bounds`; the optimizer
then projects the step and treats variables blocked by an active bound as fixed:

```rust
use tinyopt::{Bounds, Dynamic, Params};

struct Bounded {
    values: Vec<f64>,
    lower: Vec<f64>,
    upper: Vec<f64>,
}

impl Params for Bounded {
    type Scalar = f64;
    type Dim = Dynamic;

    fn dims(&self) -> usize { self.values.len() }
    fn as_slice(&self) -> &[f64] { &self.values }
    fn plus_eq(&mut self, delta: &[f64]) {
        for (value, d) in self.values.iter_mut().zip(delta) { *value += d; }
    }
    fn norm2(&self) -> f64 { self.values.iter().map(|v| v * v).sum() }

    fn bounds(&self) -> Option<Bounds<f64>> {
        Some(Bounds::from_slices(&self.lower, &self.upper))
    }
}
```

`solver_type` also selects the algorithm in the free functions
(`optimize`, `optimize_sparse`, ...). Use an optimizer object to reuse buffers,
change options between runs, or export the covariance:

```rust
use tinyopt::backend::{DenseMatrix, DenseVector, Matrix, Vector};
use tinyopt::optimizers::lm;
use tinyopt::{Cost, CovarianceOptions, Options, Static};

type Vec2 = Vector<f64, Static<2>>;
type Mat2 = Matrix<f64, Static<2>>;

let acc = |x: &[f64], g: Option<&mut Vec2>, h: Option<&mut Mat2>| -> Cost {
    let (r0, r1) = (x[0] - 1.0, x[1] + 2.0);
    if let Some(g) = g {
        g.set_at(0, 2.0 * r0);
        g.set_at(1, 2.0 * r1);
    }
    if let Some(h) = h {
        h.set_elem(0, 0, 2.0);
        h.set_elem(1, 1, 2.0);
    }
    Cost::with_num_residuals(r0 * r0 + r1 * r1, 2)
};

let mut optimizer = lm::Optimizer::<Vec2, Mat2>::new(Options::default());
let mut x = <Vec2 as DenseVector<f64>>::zeros(2);
let summary = optimizer.optimize(&mut x, acc);
assert!(summary.succeeded());

// The optimizer keeps its allocations and can be reused.
let summary = optimizer.optimize(&mut x, acc);
assert!(summary.succeeded());

// Posterior covariance from the final (undamped) Hessian.
let cov = optimizer.covariance_with(&CovarianceOptions::default());
assert!(cov.is_some());
```

## Fixed parameters (fixed dimensions)

Hold parameters fixed either on the optimizer (indices) or on the parameter type
(`Params::fixed`). Fixed rows/columns are masked out and their diagonal is set to one,
so they never move:

```rust
use tinyopt::backend::{DenseMatrix, DenseVector, Matrix, Vector};
use tinyopt::optimizers::lm;
use tinyopt::{Cost, Options, Static};

type Vec2 = Vector<f64, Static<2>>;
type Mat2 = Matrix<f64, Static<2>>;

let acc = |x: &[f64], g: Option<&mut Vec2>, h: Option<&mut Mat2>| -> Cost {
    let (r0, r1) = (x[0] - 1.0, x[1] - 2.0);
    if let Some(g) = g {
        g.set_at(0, 2.0 * r0);
        g.set_at(1, 2.0 * r1);
    }
    if let Some(h) = h {
        h.set_elem(0, 0, 2.0);
        h.set_elem(1, 1, 2.0);
    }
    Cost::with_num_residuals(r0 * r0 + r1 * r1, 2)
};

let mut optimizer = lm::Optimizer::<Vec2, Mat2>::new(Options::default());
optimizer.fix(&[1]); // keep x1 fixed
let mut x = <Vec2 as DenseVector<f64>>::zeros(2);
let summary = optimizer.optimize(&mut x, acc);
assert!(summary.succeeded());
assert!((x.at(0) - 1.0).abs() < 1e-6);
assert_eq!(x.at(1), 0.0); // unchanged
```

From the parameter type, return the indices in `Params::fixed` (do not also call
`Optimizer::fix`, the optimizer warns and uses the explicit indices):

```rust
use tinyopt::backend::{DenseMatrix, DenseVector, Matrix, Vector};
use tinyopt::{Cost, Options, Params, Static, optimize};

#[derive(Clone, Debug)]
struct Rectangle {
    size: [f64; 2],
}

impl Params for Rectangle {
    type Scalar = f64;
    type Dim = Static<2>;

    fn dims(&self) -> usize {
        2
    }
    fn as_slice(&self) -> &[f64] {
        &self.size
    }
    fn plus_eq(&mut self, delta: &[f64]) {
        self.size[0] += delta[0];
        self.size[1] += delta[1];
    }
    fn norm2(&self) -> f64 {
        self.size.iter().map(|v| v * v).sum()
    }
    // Height stays fixed at its initial value.
    fn fixed(&self) -> Option<Vec<usize>> {
        Some(vec![1])
    }
}

type Vec2 = Vector<f64, Static<2>>;
type Mat2 = Matrix<f64, Static<2>>;

let acc = |x: &[f64], g: Option<&mut Vec2>, h: Option<&mut Mat2>| -> Cost {
    let (r0, r1) = (x[0] - 3.0, x[1] - 4.0);
    if let Some(g) = g {
        g.set_at(0, 2.0 * r0);
        g.set_at(1, 2.0 * r1);
    }
    if let Some(h) = h {
        h.set_elem(0, 0, 2.0);
        h.set_elem(1, 1, 2.0);
    }
    Cost::with_num_residuals(r0 * r0 + r1 * r1, 2)
};

let mut rect = Rectangle { size: [1.0, 1.0] };
let summary = optimize(&mut rect, acc, &Options::default());
assert!(summary.succeeded());
assert!((rect.size[0] - 3.0).abs() < 1e-6);
assert_eq!(rect.size[1], 1.0); // fixed by the parameter type
```

## Custom parameter types

Any type implementing `Params` can be optimized: expose the scalar count, a flat
slice for evaluation, the tangent update `plus_eq`, and the squared norm used by the
relative step criterion.

```rust
use tinyopt::backend::{DenseMatrix, DenseVector, Matrix, Vector};
use tinyopt::{Cost, Options, Params, Static, optimize};

#[derive(Clone, Debug)]
struct Rectangle {
    size: [f64; 2],
}

impl Params for Rectangle {
    type Scalar = f64;
    type Dim = Static<2>;

    fn dims(&self) -> usize {
        2
    }
    fn as_slice(&self) -> &[f64] {
        &self.size
    }
    fn plus_eq(&mut self, delta: &[f64]) {
        self.size[0] += delta[0];
        self.size[1] += delta[1];
    }
    fn norm2(&self) -> f64 {
        self.size.iter().map(|v| v * v).sum()
    }
}

type Vec2 = Vector<f64, Static<2>>;
type Mat2 = Matrix<f64, Static<2>>;

let acc = |x: &[f64], g: Option<&mut Vec2>, h: Option<&mut Mat2>| -> Cost {
    let (r0, r1) = (x[0] - 3.0, x[1] - 4.0);
    if let Some(g) = g {
        g.set_at(0, 2.0 * r0);
        g.set_at(1, 2.0 * r1);
    }
    if let Some(h) = h {
        h.set_elem(0, 0, 2.0);
        h.set_elem(1, 1, 2.0);
    }
    Cost::with_num_residuals(r0 * r0 + r1 * r1, 2)
};

let mut rect = Rectangle { size: [0.0, 0.0] };
let summary = optimize(&mut rect, acc, &Options::default());
assert!(summary.succeeded());
assert!((rect.size[0] - 3.0).abs() < 1e-6);
assert!((rect.size[1] - 4.0).abs() < 1e-6);
```

### Manifold parameters

Put the retraction in `plus_eq` (here a 2D pose with a wrapped angle) and wrap the
value when the storage must stay at the call site:

```rust
use tinyopt::backend::{DenseMatrix, DenseVector, Matrix, Vector};
use tinyopt::{Cost, Options, Params, ParamsWrapper, Static, optimize};

#[derive(Clone, Debug)]
struct Pose2 {
    data: [f64; 3], // x, y, theta
}

impl Params for Pose2 {
    type Scalar = f64;
    type Dim = Static<3>;

    fn dims(&self) -> usize {
        3
    }
    fn as_slice(&self) -> &[f64] {
        &self.data
    }
    fn plus_eq(&mut self, delta: &[f64]) {
        self.data[0] += delta[0];
        self.data[1] += delta[1];
        self.data[2] = (self.data[2] + delta[2] + std::f64::consts::PI)
            .rem_euclid(2.0 * std::f64::consts::PI)
            - std::f64::consts::PI;
    }
    fn norm2(&self) -> f64 {
        self.data.iter().map(|v| v * v).sum()
    }
}

type Vec3 = Vector<f64, Static<3>>;
type Mat3 = Matrix<f64, Static<3>>;

let acc = |x: &[f64], mut g: Option<&mut Vec3>, mut h: Option<&mut Mat3>| -> Cost {
    let target = [1.0, -2.0, 0.5];
    let mut cost = 0.0;
    for i in 0..3 {
        let r = x[i] - target[i];
        cost += r * r;
        if let Some(g) = g.as_mut() {
            g.set_at(i, 2.0 * r);
        }
        if let Some(h) = h.as_mut() {
            h.set_elem(i, i, 2.0);
        }
    }
    Cost::with_num_residuals(cost, 3)
};

let mut wrapped = ParamsWrapper::new(Pose2 { data: [0.0, 0.0, 0.0] });
let summary = optimize(&mut wrapped, acc, &Options::default());
assert!(summary.succeeded());
assert!((wrapped.data[0] - 1.0).abs() < 1e-6);
```

## Several parameters at once

`optimize2` / `optimize3` pack several parameter values, run one optimization, and
write them back:

```rust
use tinyopt::backend::{DenseMatrix, DenseVector};
use tinyopt::{Cost, DynGradient, DynHessian, Options, optimize2};

let mut a = 0.0_f64;
let mut b = 0.0_f64;
let summary = optimize2(
    &mut a,
    &mut b,
    |av: &[f64], bv: &[f64], g: Option<&mut DynGradient>, h: Option<&mut DynHessian>| {
        let (ra, rb) = (av[0] - 1.0, bv[0] + 2.0);
        if let Some(g) = g {
            g.set_at(0, 2.0 * ra);
            g.set_at(1, 2.0 * rb);
        }
        if let Some(h) = h {
            h.set_elem(0, 0, 2.0);
            h.set_elem(1, 1, 2.0);
        }
        Cost::with_num_residuals(ra * ra + rb * rb, 2)
    },
    &Options::default(),
);
assert!(summary.succeeded());
assert!((a - 1.0).abs() < 1e-6);
assert!((b + 2.0).abs() < 1e-6);
```

## Sparse and block-structured problems

A scalar sparse Hessian (`SparseMat`) uses the same accumulation API; only the
`Option<&mut H>` type changes:

```rust
use tinyopt::backend::{DenseVector, Dynamic, Vector};
use tinyopt::sparse::SparseMat;
use tinyopt::{Cost, Options, optimize_sparse};

let mut x = vec![0.0_f64; 4];
let summary = optimize_sparse(
    &mut x,
    |xv: &[f64], mut g: Option<&mut Vector<f64, Dynamic>>, mut h: Option<&mut SparseMat>| {
        let mut cost = 0.0;
        for i in 0..4 {
            let r = xv[i] - 1.0;
            cost += r * r;
            if let Some(g) = g.as_mut() {
                g.set_at(i, g.at(i) + r);
            }
            if let Some(h) = h.as_mut() {
                *h.coeff_ref(i, i) += 1.0;
            }
            if i + 1 < 4 {
                let r = xv[i + 1] - xv[i];
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
        }
        Cost::with_num_residuals(cost, 7)
    },
    &Options::default(),
);
assert!(summary.succeeded());
```

Compile-time block-sparse Hessians (`BlockSparseMat`) and block-tridiagonal Hessians
(`BlockTridiagMat`) are declared with `block_sparse_types!` and use
`optimize_block_sparse` / `optimize_tridiag`. Any number of block sizes is
supported (`[6, 3]`, `[6, 3, 2]`, `[6, 4, 3, 2]`, ...); see
[docs/TUTORIAL.md](docs/TUTORIAL.md) for the Schur-split constraints. See
`examples/bundle_adjustment.rs` and
[docs/PERFORMANCE.md](docs/PERFORMANCE.md) for benchmarks.

```rust
use tinyopt::backend::{DenseVector, Dynamic, Vector};
use tinyopt::block_sparse::BlockSparseMat;
use tinyopt::{Cost, Options, optimize_block_sparse};

mod scene {
    tinyopt::block_sparse_types!(BaTypes, [6, 3]); // camera 6, point 3
}

let mut x = vec![0.0_f64; 9]; // 1 camera + 1 point
let summary = optimize_block_sparse::<scene::BaTypes, _, _>(
    &mut x,
    |xv: &[f64], mut g: Option<&mut Vector<f64, Dynamic>>, mut h: Option<&mut BlockSparseMat<scene::BaTypes>>| {
        let mut cost = 0.0;
        for i in 0..9 {
            let r = xv[i] - 1.0;
            cost += r * r;
            if let Some(g) = g.as_mut() {
                g.set_at(i, g.at(i) + r);
            }
        }
        if let Some(h) = h.as_mut() {
            for i in 0..6 {
                let mut e = [0.0; 6];
                e[i] = 1.0;
                h.block_mut::<6, 6>(0, 0).add_outer_product_self(1.0, &e);
            }
            for i in 0..3 {
                let mut e = [0.0; 3];
                e[i] = 1.0;
                h.block_mut::<3, 3>(0, 0).add_outer_product_self(1.0, &e);
            }
        }
        Cost::with_num_residuals(cost, 9)
    },
    &Options::default(),
);
assert!(summary.succeeded());
```

For a reduced head block (Schur complement), implement `Params::reduced()`
returning `Some((0, count))`; both scalar sparse and block-sparse solvers then
eliminate the tail.

## Robust losses

Losses are plain functions over a squared norm, with optional Jacobian scales:

```rust
use tinyopt::losses::{huber, huber_with_scale};

let squared_norm = 0.25;
let threshold2 = 0.1;
let cost = huber(squared_norm, threshold2);
let (_loss, scale) = huber_with_scale(squared_norm, threshold2);
assert!(cost > 0.0);
assert!(scale > 0.0 && scale <= 1.0);
```

Classical norms (`squared_l2`, `l2`, `l1`, `linf`), robust M-estimators (`huber`,
`tukey`, `arctan`, `cauchy`, `geman_mcclure`, `blake_zisserman`, `truncated`),
Mahalanobis whitening, activations and distances are available under `tinyopt::losses`
and `tinyopt::distances`.

## Examples and benchmarks

Runnable examples live in `examples/`:

```shell
cargo run --example astronomy
cargo run --example bundle_adjustment

```

Criterion benchmarks:

```shell
cargo bench --bench dense
cargo bench --bench sparse
```

Cross-library benchmarks (tinyopt-rs vs Ceres vs the C++ tinyopt binary) live in
`benchmarks/compare` and produce an HTML/Markdown report.

## Optional backends

### Optional Rayon parallelism

Enable `rayon` to parallelize independent factor accumulation and large
independent-block Schur computations in the block-sparse solver:

```toml
tinyopt = { version = "0.8", features = ["rayon"] }
```

`tinyopt::rayon` re-exports Rayon when the feature is enabled. Run optimization
inside a configured pool to control the number of threads:

```rust,ignore
let pool = tinyopt::rayon::ThreadPoolBuilder::new()
    .num_threads(8)
    .build()?;
pool.install(|| {
    tinyopt::optimize(&mut params, accumulation, &options)
});
```

Use `optimize_parallel` when each objective term can be evaluated independently.
The factor callback receives its index, the parameter slice, and optional
gradient/Hessian accumulators; add that factor's contribution to any provided
accumulator and return its cost:

```rust,ignore
let summary = tinyopt::optimize_parallel(&mut x, factors.len(), factor, &options);
```

For sparse Hessians, use `optimize_parallel_sparse`; for compile-time
block-sparse layouts, use `optimize_parallel_block_sparse::<BlockTypes, _, _>`.
These APIs keep the same factor callback shape while selecting the Hessian
accumulator type explicitly.

With `rayon`, sets of at least 16 factors use parallel evaluation when the
active Rayon pool has multiple threads; smaller sets, single-thread pools, and
builds without the feature use the serial path. Callbacks must be safe to invoke
concurrently, and floating-point reduction order may differ from serial
accumulation. Existing single-callback `optimize` calls remain serial.
The block-sparse solver parallelizes numeric Schur work only when eliminated
blocks are independent and the problem is large enough; coupled/fallback
systems keep the serial path. Builds without `rayon` have no Rayon dependency.

### Optional CHOLMOD backend

Enable `suitesparse` to route scalar sparse systems (`LinearSolverMethod::SuiteSparse`)
through CHOLMOD. The build script locates the headers and library via
`CHOLMOD_INCLUDE_DIR` / `CHOLMOD_LIB_DIR`, a conda `CONDA_PREFIX`, or the usual
`/opt/homebrew` and `/usr/local` prefixes. Like the C++ dispatch, CHOLMOD also
factorizes the eliminated bottom-right block of a Schur partition.

```shell
cargo test --features suitesparse
```

## License

Apache-2.0. See the `LICENSE` file in the repository.
