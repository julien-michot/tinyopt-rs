# tinyopt-rs

A lightweight, Rust native optimization library for unconstrained optimization and
non-linear least squares on **dense** systems — the Rust counterpart of
[C++ tinyopt](https://github.com/julienmichot/tinyopt), at version parity **0.8.0**.

The library follows the same *accumulation pattern* as the C++ implementation: users
accumulate the gradient `Jᵀr` and Hessian approximation `JᵀJ` directly into the linear
system instead of materializing residual and Jacobian buffers.

## Backends

Exactly one linear-algebra backend is compiled per build:

| Feature | Sizes | Types |
|---|---|---|
| `nalgebra` (default) | static + dynamic | `SVector`/`SMatrix`, `DVector`/`DMatrix` |
| `faer` | dynamic | `Col`/`Mat` |

```shell
cargo test                                              # nalgebra + autodiff
cargo test --no-default-features --features faer        # faer backend
cargo test --no-default-features --features faer,autodiff
```

## Differentiation

- `autodiff` (default): forward-mode AD through
  [`num-dual`](https://docs.rs/num-dual). `DualVec<T, D>` is the Rust equivalent of
  tinyopt's `Jet<T, N>`. Accumulation is direct (`Jᵀr` / `JᵀJ` per residual) and the
  static-dimension path uses stack-allocated dual numbers.
- `numdiff`: forward, central, and fast-central finite differences.

User functions are generic over the scalar, so a single implementation serves both
plain evaluation and autodiff:

```rust
use tinyopt_rs::diff::{DualNum, ResidualsFn};

struct Sqrt2;

impl ResidualsFn for Sqrt2 {
    type Scalar = f64;

    fn eval<T: DualNum<Primitive = f64>>(&self, x: &[T]) -> Vec<T> {
        vec![x[0] * x[0] - T::from_re(2.0)]
    }
}
```

## Quick start

```rust
use tinyopt_rs::backend::{DenseVector, Vector};
use tinyopt_rs::diff::{DualNum, ResidualsFn};
use tinyopt_rs::{Options, Static, optimize_residuals};

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
```

## Status

| Milestone | Scope | Status |
|---|---|---|
| M0 | scaffold, backend abstraction, params, num-dual autodiff | ✅ |
| M1 | Cost/Summary/Options, LM/GN, numdiff, gradient checks | ✅ |
| M2 | GD/CG/BFGS/L-BFGS/DogLeg, line search, losses, distances | ✅ |
| M3 | direct autodiff accumulation, robust blocks, AD type coverage | ✅ |
| M4 | fixed params, covariance, multi-params, manifolds | ✅ |
| M5 | examples, benchmarks, docs parity | ✅ |
| M6 | sparse, block-tridiagonal and block-sparse systems | ✅ |

## Losses and distances

```rust
use tinyopt_rs::losses::{huber, huber_loss, cauchy_loss, maha_whitened_cov};
use tinyopt_rs::distances::{euclidean, euclidean_with_jac};
```

Classical norms (`squared_l2`, `l2`, `l1`, `linf`), robust M-estimators
(`huber`, `tukey`, `arctan`, `cauchy`, `geman_mcclure`, `blake_zisserman`,
`truncated`) with optional Jacobian scales, Mahalanobis norms/whitening,
activations, softmax and vector distances are all available.

## Fixed parameters, covariance and multiple parameters

```rust
use tinyopt_rs::optimizers::lm;
use tinyopt_rs::{CovarianceOptions, Options};

let mut opt = lm::Optimizer::<Vec2>::new(Options::default());
opt.fix(&[1]);                       // hold parameter 1 fixed
let summary = opt.optimize(&mut x, acc);
let cov = opt.covariance_with(&CovarianceOptions::default());
```

`optimize2` / `optimize3` run several parameters together (packed internally and
written back once), and `ParamsWrapper` adapts manifold-style parameter types.

## Sparse and block-structured problems

Scalar sparse Hessians, block-tridiagonal Hessians and compile-time block-sparse
Hessians share the same optimizer API:

```rust
use tinyopt_rs::sparse::SparseMat;
use tinyopt_rs::{Options, optimize_sparse};

let mut x = vec![0.0_f64; 5];
let summary = optimize_sparse(&mut x, |x, grad, h| {
    // grad: Option<&mut Vector<f64, Dynamic>>
    // h:    Option<&mut SparseMat> (triplets or coeff_ref)
    tinyopt_rs::Cost::with_num_residuals(0.0, 1)
}, &Options::default());
```

- `SparseMatrix<S>` (CSC) with triplet assembly, `coeff_ref`, symmetric
  completion and a simplicial LDLᵀ (`SimplicialLdl`) with a native
  minimum-degree (AMD-style) fill-reducing ordering
  (`options.linear.ordering`, default `Ordering::Amd`; `Ordering::Natural`
  disables it), including the scalar Schur complement path.
- `BlockTridiagMat<B>` (fixed block size) and `BlockTridiagMatX<S>` (dynamic) with
  block Thomas/Takahashi factorization (`BlockTridiagLdlt`).
- `block_sparse_types!` + `BlockSparseMat<T, S>` with compile-time block-pair
  checks and an exact Schur solver (`BlockSparseLdlt`); `BlockSparseMatX<S>` for
  runtime layouts.
- `Params::reduced()` selects the Schur reduced head block
  (`has_reduced`/`num_reduced` on LM, GN and DogLeg).

## Examples and benchmarks

The seven dense C++ examples are ported one-to-one:

```shell
cargo run --example astronomy
cargo run --example biology
cargo run --example computer_vision
cargo run --example finance
cargo run --example physics
cargo run --example robotics
cargo run --example signal_processing
cargo run --example bundle_adjustment        # sparse Hessian + Schur

```

Benchmarks mirror the C++ dense suites (f64/f32, static/dynamic, prior and
1D/2D/6D profiling workloads):

```shell
cargo bench --bench dense
cargo bench --bench sparse
```

## Documentation

- [`docs/TUTORIAL.md`](docs/TUTORIAL.md) – dense tutorial (mirrors the C++ one).
- [`docs/CPP_API_MAPPING.md`](docs/CPP_API_MAPPING.md) – C++ → Rust API map.
- `cargo doc --no-deps --open` – rustdoc reference.

## Development

The Rust toolchain is managed with [pixi](https://pixi.prefix.dev/):

```shell
pixi run build
pixi run test
pixi run fmt
pixi run clippy
```

## License

Apache-2.0
