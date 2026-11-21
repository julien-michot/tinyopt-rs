# AGENTS.md — Development Guidelines for tinyopt

`tinyopt` is the Rust native counterpart of the C++ `tinyopt` library
(version parity 0.8.0), focused on **dense** systems.

> **CRITICAL AGENT COMMIT POLICY**: AI agents must **never** create git commits
> unless explicitly commanded by the user in the active session.

## Architecture

- `src/backend/` — one backend per build: `nalgebra` (static + dynamic) or `faer`
  (dynamic). All crate code targets the `Dim`, `DenseVector`, `DenseMatrix` traits.
- `src/params.rs` — `Params` trait mirroring C++ `traits::params_trait`
  (`Scalar`, `Dim`, `dims`, `plus_eq`, `norm2`, optional `fixed`).
- `src/diff/` — autodiff via `num-dual` (`DualVec` = C++ `Jet`), numdiff (M1).
- `src/optimizers/` — shared core loop plus LM, GN, DL, GD, CG, BFGS, L-BFGS (M1+).
- `src/losses/`, `src/distances.rs` — norms, robust losses, Mahalanobis,
  activations, distances (M2).

## Cross-library benchmarks

`benchmarks/compare` (independent package) runs tinyopt-rs, Ceres (Rust binding)
and the C++ tinyopt binary on ported C++ problems, then renders an HTML/Markdown
report:

```sh
pixi run bench-quick     # short suite
pixi run bench-compare   # full suite (100c/50kp needs --full)
pixi run bench-report    # plots + report under benchmarks/compare/results/report
```

Keep `benchmarks/compare/src/options.rs` in sync with the C++ `CreateOptions()`
and the fair-comparison rules; report parameterization differences in the
generated report notes.

## Rules

1. **Zero warnings**: `cargo clippy --all-targets -- -D warnings` and
   `cargo fmt --check` must pass for every feature combination.
2. **Both backends always build and pass**: run the test matrix
   - `cargo test --no-default-features --features nalgebra,autodiff`
   - `cargo test --no-default-features --features faer`
   - `cargo test --no-default-features --features faer,autodiff`
3. **Static and dynamic parameters** must both be exercised in tests.
4. **No hidden copies**: accumulation uses slices and `add_outer_product`;
   the fixed-size paths must stay allocation-free (verified by an allocation-counter
   test, M4).
5. **Derivative verification**: every optimizer/loss test must compare analytical
   derivatives against autodiff or finite differences (mirrors C++
   `diff::CheckGradient` / `CheckResidualsGradient`).
6. **Feature parity**: every solver, loss, optimizer mode, or user-facing capability
   needs a correctness test and, where relevant, a benchmark.
7. **Naming**: keep the C++ names, adapted to Rust conventions (`Optimize` →
   `optimize`, `CheckResidualsGradient` → `check_residuals_gradient`,
   `MahaSquaredNorm` → `maha_squared_norm`). Modules mirror C++ namespaces.
8. **Commit style** (when asked): `<emoji> <type>(<scope>): <subject>` with a short
   body, same emoji table as the C++ repository.

## Commands

```shell
pixi run build        # cargo build --all-targets
pixi run test         # cargo test (default features)
pixi run test-faer    # faer backend
pixi run fmt
pixi run clippy
pixi run ci           # fmt + clippy + both backends
```
