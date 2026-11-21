# tinyopt-rs tutorial

This mirrors the dense sections of the C++ tutorial (`docs/tutorials/cpp.md`).
Every snippet below is exercised by the test suite (`tests/tutorial.rs` and the
other integration tests).

## 1. Hello world

The minimal entry point takes a parameter, an accumulation function, and
options. The accumulation function computes the cost and, when requested, the
gradient and Hessian:

```rust
use tinyopt_rs::backend::{DenseMatrix, DenseVector, Matrix, Vector};
use tinyopt_rs::{Cost, Options, Static, optimize};

type V1 = Vector<f64, Static<1>>;
type M1 = Matrix<f64, Static<1>>;

let acc = |x: &[f64], grad: Option<&mut V1>, h: Option<&mut M1>| -> Cost {
    let r = x[0] - 2.0;
    if let Some(g) = grad {
        g.set_at(0, 2.0 * r);
    }
    if let Some(h) = h {
        h.set_elem(0, 0, 2.0);
    }
    Cost::new(r * r)
};

let mut x = <V1 as DenseVector<f64>>::zeros(1);
let summary = optimize(&mut x, acc, &Options::default());
assert!(summary.succeeded());
```

`grad` and `h` are `Option`s: first-order optimizers pass `h == None`, and
evaluation-only calls pass both as `None`.

## 2. Residual vectors with autodiff

Write the residual function once, generic over the scalar:

```rust
use tinyopt_rs::diff::{DualNum, ResidualsFn};

struct Circle;

impl ResidualsFn for Circle {
    type Scalar = f64;

    fn eval<T: DualNum<Primitive = f64>>(&self, x: &[T]) -> Vec<T> {
        vec![x[0].clone() - 1.0, x[1].clone() - 2.0]
    }
}
```

Then optimize with `optimize_residuals(&mut x, &Circle, &Options::default())`.
The crate seeds `num_dual::DualVec` values and accumulates `Jᵀr` and `JᵀJ`
directly; static dimensions use stack-allocated dual numbers.

## 3. Manual accumulation

When analytical derivatives are available, accumulate them yourself:

```rust
let acc = |x: &[f64], grad: Option<&mut V1>, h: Option<&mut M1>| -> Cost {
    let r = x[0] - 2.0;
    if let Some(g) = grad {
        g.set_at(0, 2.0 * r);          // Jᵀr
    }
    if let Some(h) = h {
        h.set_elem(0, 0, 2.0);         // JᵀJ
    }
    Cost::new(r * r)
};
```

## 4. Explicit optimizer and covariance

```rust
use tinyopt_rs::optimizers::lm;
use tinyopt_rs::{CovarianceOptions, Options};

let mut opt = lm::Optimizer::<V2>::new(Options::default());
let summary = opt.optimize(&mut x, acc);
let cov = opt.covariance_with(&CovarianceOptions::default());
```

Set `options.exp.final_cov = true` to receive the covariance in
`summary.final_cov` instead.

## 5. Choosing a solver

```rust
use tinyopt_rs::{Options, Solver};

let mut options = Options::new(Solver::GaussNewton);
let mut options = Options::new(Solver::GradientDescent); // options.gd.lr
let mut options = Options::new(Solver::ConjugateGradient); // options.cg.step_size
let mut options = Options::new(Solver::Bfgs);             // options.bfgs.*
let mut options = Options::new(Solver::Lbfgs);            // options.lbfgs.history_size
let mut options = Options::new(Solver::DogLeg);           // options.dl.*
```

Line searches are enabled with `options.line_search.method`:
`LineSearchMethod::{None, Backtracking, Wolfe, GoldenSection}`.

## 6. Multiple parameters

```rust
use tinyopt_rs::optimize2;

let acc = |c: &[f64], s: &[f64], grad: Option<&mut DynGradient>,
           h: Option<&mut DynHessian>| -> Cost {
    // c and s are separate slices; grad/h are dynamic backend vectors/matrices.
    let residual = c[0] - 1.0;
    let scale = s[0] - 2.0;
    if let Some(g) = grad { g.set_at(0, 2.0 * residual); g.set_at(1, 2.0 * scale); }
    if let Some(h) = h { h.set_elem(0, 0, 2.0); h.set_elem(1, 1, 2.0); }
    Cost::new(residual * residual + scale * scale)
};
let summary = optimize2(&mut center, &mut scale, acc, &Options::default());
```

(Parameters are packed internally and written back once.)

## 7. Custom parameter types

Implement `Params`; the flat scalar view is required by the accumulation API:

```rust
use tinyopt_rs::{Params, Static};

struct Rectangle { size: [f64; 2] }

impl Params for Rectangle {
    type Scalar = f64;
    type Dim = Static<2>;
    fn dims(&self) -> usize { 2 }
    fn as_slice(&self) -> &[f64] { &self.size }
    fn plus_eq(&mut self, d: &[f64]) {
        self.size[0] += d[0];
        self.size[1] += d[1];
    }
    fn norm2(&self) -> f64 { self.size.iter().map(|v| v * v).sum() }
}
```

## 8. Manifold parameters

Put the retraction in `plus_eq` and, optionally, wrap the value:

```rust
use tinyopt_rs::ParamsWrapper;

let mut wrapped = ParamsWrapper::new(pose);
let summary = opt.optimize(&mut wrapped, acc);
```

`ParamsWrapper` dereferences to the inner value. A pose-like type should keep its
scalar block contiguous (for `as_slice`) and apply the manifold update in
`plus_eq`.

## 9. Robust losses

As a scalar objective (with analytical gradient scale):

```rust
use tinyopt_rs::losses::{huber, huber_with_scale, squared_l2};

let n2 = squared_l2(&residual);
let (_, scale) = huber_with_scale(n2, delta2);
// cost = huber(n2, delta2); grad = 2 * scale * residual
```

Or as robust residual blocks for the autodiff path:

```rust
use tinyopt_rs::diff::{ResidualBlock, RobustResidualsFn};

impl RobustResidualsFn for Problem {
    type Scalar = f64;
    fn eval<T: DualNum<Primitive = f64>>(&self, x: &[T]) -> Vec<ResidualBlock<T>> {
        vec![ResidualBlock::huber(vec![x[0].clone() - 1.0], 0.5)]
    }
}
// optimize_robust_residuals(&mut x, &Problem, &options)
```

## 10. Fixed parameters

```rust
let mut opt = lm::Optimizer::<V3>::new(Options::default());
opt.fix(&[1]);                    // parameter 1 stays bit-exact
let summary = opt.optimize(&mut x, acc);
```

Alternatively return indices from `Params::fixed()`.

## 11. Recommended workflow

1. Check your derivatives with `diff::check_residuals_gradient` (NLLS) or
   `diff::check_gradient` (scalar costs) before optimizing.
2. Prefer static dimensions for small problems (allocation-free with nalgebra).
3. Use LM for NLLS by default, GN for well-conditioned problems, BFGS/L-BFGS for
   scalar costs, and DogLeg when a trust region is preferable.
4. Keep the accumulation function allocation-free: accumulate into the provided
   `grad`/`h` instead of building residual or Jacobian buffers.

## 12. Sparse Hessians

Use `SparseMatrix` when the normal equations are large and sparse. Accumulate
`Jᵀr` into the gradient and `JᵀJ` into the Hessian with triplets or `coeff_ref`:

```rust
use tinyopt_rs::sparse::SparseMat;
use tinyopt_rs::{Cost, Options, optimize_sparse};

let acc = |x: &[f64], grad: Option<&mut DynGradient>, h: Option<&mut SparseMat>| -> Cost {
    let mut cost = 0.0;
    for i in 0..x.len() {
        let r = x[i] - 1.0;
        cost += r * r;
        if let Some(g) = grad { g.set_at(i, g.at(i) + r); }
        if let Some(h) = h { *h.coeff_ref(i, i) += 1.0; }
    }
    Cost::with_num_residuals(cost, x.len() as i32)
};

let mut x = vec![0.0_f64; 100];
let summary = optimize_sparse(&mut x, acc, &Options::default());
```

Solves use a simplicial LDLᵀ with optional Schur partitioning.

## 13. Schur complement and block structures

Return the reduced head block from `Params::reduced`:

```rust
fn reduced(&self) -> Option<(usize, usize)> {
    Some((0, num_cameras * 6))   // cameras first, points eliminated
}
```

LM, GN and DogLeg then report `has_reduced()` / `num_reduced()` and solve the
Schur complement. Block-tridiagonal and block-sparse Hessians use their own
algorithms:

```rust
use tinyopt_rs::block_tridiag::BlockTridiagMat;

let acc = |x, grad, h: Option<&mut BlockTridiagMat<6, f64>>| {
    // h.diagonal(k).add_outer_product_self(1.0, &j_row);
    // h.off_diagonal(k).add_outer_product(1.0, &j_k, &j_k1);
    // ...
};
let summary = tinyopt_rs::optimize_tridiag::<6, _, _>(&mut x, acc, &options);
```

For compile-time block-sparse problems, declare the block types first:

```rust
tinyopt_rs::block_sparse_types!(BaTypes, [6, 3]);
let mut opt = tinyopt_rs::optimizers::lm::Optimizer::<
    Params, tinyopt_rs::BlockSparseMat<BaTypes, f64>>::new(options);
opt.init_with(tinyopt_rs::BlockSparseMat::with_counts(&[num_cameras, num_points]));
```

Covariance follows the Hessian storage: `FinalCovariance::{Dense, Sparse,
BlockTridiag, BlockSparse}`, selected with `CovarianceOptions::pattern`.
