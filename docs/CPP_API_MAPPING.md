# C++ tinyopt → Rust tinyopt-rs API map

This document maps the dense C++ API (tinyopt 0.8.0) to this crate. Naming keeps
the C++ identifiers, adapted to Rust conventions (snake_case functions and
modules).

## Entry points

| C++ | Rust |
|---|---|
| `Optimize(x, acc, options)` | `optimize(&mut x, acc, &options)` |
| `Optimize(x, acc)` | `optimize_default(&mut x, acc)` |
| `Optimize(x0, x1, acc)` (variadic) | `optimize2(&mut x0, &mut x1, acc, &options)` |
| `Optimize(x0, x1, x2, acc)` | `optimize3(...)` |
| residuals lambda (autodiff) | `optimize_residuals(&mut x, &residuals, &options)` |
| scalar cost lambda (autodiff) | `optimize_cost(&mut x, &cost, &options)` |
| `Residuals(Huber(res, δ), ...)` tuple | `optimize_robust_residuals(&mut x, &blocks, &options)` |

## Optimizers

| C++ | Rust |
|---|---|
| `lm::Optimizer<MatN>{options}(x, acc)` | `lm::Optimizer::<VecN>::new(options).optimize(&mut x, acc)` |
| `gn::Optimizer<...>` | `optimizers::gn::Optimizer` |
| `dl::Optimizer<...>` | `optimizers::dl::Optimizer` |
| `gd::Optimizer<...>` | `optimizers::gd::Optimizer` |
| `cg::Optimizer<...>` | `optimizers::cg::Optimizer` |
| `bfgs::Optimizer<...>` | `optimizers::bfgs::Optimizer` |
| `lbfgs::Optimizer<...>` | `optimizers::lbfgs::Optimizer` |
| `Options::Solver::LevenbergMarquardt` | `Solver::LevenbergMarquardt` (default) |
| `optimizer.fix({1, 4})` | `optimizer.fix(&[1, 4])` |
| `optimizer.Covariance()` | `optimizer.covariance()` / `covariance_with(&options)` |
| `optimizer.Gradient()`, `.H()` | use the accumulation function outputs |
| `optimizer.reset()` | called automatically by `optimize` |

First-order optimizers (GD, CG, BFGS, L-BFGS) request only the gradient, so the
accumulation function receives `h == None`.

## Options

All fields and defaults match the C++ `Options`:

| C++ group | Rust field |
|---|---|
| `options.solver_type` | `options.solver_type` (`Solver`) |
| `options.linear` | `options.linear` (`LinearSolverOptions`) |
| `options.line_search` | `options.line_search` |
| `options.exp` | `options.exp` (`Export`, incl. `cov_options`) |
| `options.opt` | `options.opt` |
| `options.hessian` | `options.hessian` |
| `options.cost` | `options.cost` |
| `options.stop` | `options.stop` |
| `options.log` | `options.log` |
| `options.lm/gd/cg/dl/bfgs/lbfgs` | same names |

Callbacks: `stop_callback`, `stop_callback2`, `step_callback` are
`Option<Arc<dyn Fn(...)>>` and take `&[f32]` slices.

## Supporting types

| C++ | Rust |
|---|---|
| `Cost(cost, num_residuals, inlier_ratio)` | `Cost { cost, num_residuals, inlier_ratio }` |
| `Summary` | `Summary`, with `succeeded()`, `converged()`, `has_final_cov()` |
| `StopReason::kMinError` | `StopReason::MinError` |
| `Summary::History` | `History` (`errs`, `deltas2`, `successes`) |
| `Summary::final_cov` (variant) | `Option<Matrix<f32, Dynamic>>` |
| `Count(cost, n)` | `Cost::with_num_residuals(cost, n)` |
| `Sum.Residuals` | `Cost::from_residuals(&[...])` |

## Parameters

| C++ | Rust |
|---|---|
| `traits::params_trait<T>` | `Params` trait (`Scalar`, `Dim`, `dims`, `as_slice`, `plus_eq`, `norm2`, `fixed`) |
| `Vec<N>` (static) | `Vector<S, Static<N>>` (nalgebra `SVector`) |
| `VecX` (dynamic) | `Vector<S, Dynamic>` (nalgebra `DVector`) |
| `std::vector<S>` | `Vec<S>` |
| `std::array<S, N>` | `[S; N]` |
| `double` / `float` | `f64` / `f32` |
| `ParamsWrapper<T&>` | `ParamsWrapper<T>` |
| `ParamsPack<Ts...>` | `ParamsPack<(A, B, ...)>` |
| `PlusEq(x, dx)` | `Params::plus_eq(&mut x, dx)` |

Custom parameter types must provide a flat scalar view via `Params::as_slice`
(the accumulation API is slice-based). Manifold retractions belong in
`Params::plus_eq`.

## Differentiation

| C++ | Rust |
|---|---|
| `diff::Jet<T, N>` | `num_dual::DualVec<T, Const<N>>` (re-exported as `tinyopt_rs::diff::DualVec`) |
| `diff::Jet<T, Dynamic>` | `num_dual::DualVec<T, Dyn>` |
| `jet.a`, `jet.v` | `dual.re`, `dual.eps` |
| `[&](const auto& x)` generic lambda | `fn eval<T: DualNum<Primitive = f64>>(&self, x: &[T])` |
| `diff::Eval(x, residuals)` | `diff::eval(&residuals, x)` |
| `diff::CalculateJac(x, f)` | `diff::calculate_jac(&f, x)` |
| `diff::NumEval(x, f)` | `diff::num_eval(x, f, method, h)` |
| `diff::EstimateNumJac(...)` | `diff::estimate_num_jac(...)` |
| `diff::CreateNumDiffFunc1/2` | `diff::create_num_diff_func(...)` |
| `diff::CheckGradient` | `diff::check_gradient` |
| `diff::CheckResidualsGradient` | `diff::check_residuals_gradient` |
| `diff::Method::kCentral` | `diff::Method::Central` |
| `TINYOPT_DISABLE_AUTODIFF` | `--no-default-features` (no `autodiff` feature) |

Scalar-cost autodiff mirrors the C++ non-NLLS path: the returned value is the
cost itself, the gradient is its derivative, and the Hessian approximation is
`g gᵀ`.

## Losses and distances

| C++ | Rust |
|---|---|
| `losses::SquaredL2(x)` | `losses::squared_l2(&x)` |
| `losses::SquaredL2(x, true)` | `losses::squared_l2_with_jac(&x)` |
| `losses::L2/L1/Linf` | `losses::l2/l1/linf` (+ `_with_jac`) |
| `losses::Huber(n2, th2)` | `losses::huber(n2, th2)` |
| `losses::Huber(n2, th2, true)` | `losses::huber_with_scale(n2, th2)` |
| `losses::HuberLoss(x, th2, true)` | `losses::huber_loss_with_scale(&x, th2)` |
| `losses::Truncated/Tukey/Arctan/Cauchy/GemanMcClure/BlakeZisserman` | same snake_case names |
| `losses::HuberWeight` | `losses::huber_weight` |
| `losses::HuberLossValue` | `losses::huber_loss_value` |
| `losses::MahaSquaredNorm` | `losses::maha_squared_norm_{scalar,vars,cov}` |
| `losses::MahaNorm` | `losses::maha_norm_{scalar,vars,cov}` |
| `losses::MahaWhitened` | `losses::maha_whitened_{scalar,vars,cov}` |
| `losses::MahaWhitenedInfoU` | `losses::maha_whitened_info_u` |
| `losses::Sigmoid/Tanh/ReLU/LeakyReLU` | same snake_case (+ `_vec`, `_with_jac`) |
| `losses::Softmax/SafeSoftmax` | `losses::softmax/safe_softmax` (+ `_with_jac`) |
| `distances::L1/L2/Linf/Cosine` | `distances::{l1_distance, euclidean, linf_distance, cosine}` (+ `_with_jac`) |
| `distances::MahaNorm` | `distances::maha_norm_{vars,cov}_distance` |

Robust residual blocks:

```rust
impl RobustResidualsFn for Problem {
    type Scalar = f64;
    fn eval<T: DualNum<Primitive = f64>>(&self, x: &[T]) -> Vec<ResidualBlock<T>> {
        vec![
            ResidualBlock::huber(vec![x[0].clone() - 1.0], 0.5),
            ResidualBlock::plain(vec![x[0].clone() - 2.0]),
        ]
    }
}
```

## Linear solvers

| C++ | Rust |
|---|---|
| `LinearSolverMethod::LDLT` (default) | `LinearSolverMethod::Ldlt` (nalgebra `LBLT`, faer `Ldlt`) |
| `LLT`, `LU`, `QR`, `SVD` | `Llt`, `Lu`, `Qr`, `Svd` |
| `tinyopt::Inverse(m)` | `tinyopt_rs::inverse(&m)` |
| `TINYOPT_ENABLE_LINEAR_SOLVER_*` | cargo features `solver-ldlt/llt/lu/qr/svd` |

## Not ported (dense scope)

- Sparse, block-sparse, block-tridiagonal, and Schur complement paths.
- SuiteSparse / CHOLMOD backends.
- The C ABI (`c_api_*`) and wasm examples.
- Sophus / Lie++ adapters (use `Params::plus_eq` or `ParamsWrapper` instead).
- Eigen malloc guards (the zero-allocation test uses a counting allocator on the
  nalgebra backend).

## Backend notes

- `nalgebra` (default): static (`Static<N>`) and dynamic (`Dynamic`) sizes.
- `faer`: dynamic sizes; `Static<N>` is runtime-checked but heap-backed, so the
  zero-allocation guarantee does not apply.
