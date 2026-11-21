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
  tinyopt's `Jet<T, N>`.
- `numdiff`: central finite differences (planned, M1).

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

## Status

| Milestone | Scope | Status |
|---|---|---|
| M0 | scaffold, backend abstraction, params, num-dual autodiff | ✅ |
| M1 | core types, LM/GN, dense solvers, numdiff, gradient checks | ⏳ |
| M2 | GD/CG/BFGS/L-BFGS/DogLeg, line search, losses, distances | ⏳ |
| M3 | autodiff optimizer adapter, diffusion tests | ⏳ |
| M4 | fixed params, covariance, multi-params, manifolds | ⏳ |
| M5 | examples, benchmarks, docs parity | ⏳ |

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
